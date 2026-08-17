"""Dual-mode survival finetuning module for OPERA."""

from __future__ import annotations

import math
from typing import List

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from bonsai.functional.checkpointing import (
    attach_checkpoint_metadata,
    attach_model_config,
)
from bonsai.functional.scheduling import optimizer_with_warmup
from opera.evaluation.metrics import (
    _km_censoring_fn,
    compute_competing_risk_metrics_at_horizon,
    compute_concordance_index,
    compute_ipcw_metrics_at_horizon,
)


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow risk sets."""
    risk_scores = risk_scores.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1)
    event_mask = events == 1
    if event_mask.sum() == 0:
        return risk_scores.sum() * 0.0

    event_times = times[event_mask]
    event_risks = risk_scores[event_mask]
    risk_set = times.unsqueeze(0) >= event_times.unsqueeze(1)
    masked_risks = risk_scores.unsqueeze(0).masked_fill(~risk_set, -torch.inf)
    log_denominator = torch.logsumexp(masked_risks, dim=1)
    return -(event_risks - log_denominator).mean()


def exact_breslow_cox_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Exact full-risk-set Cox loss in linear memory with Breslow ties.

    Patients are sorted from longest to shortest follow-up. The reverse-time
    ordering turns every risk-set denominator into a cumulative log-sum-exp.
    Competing events (``event == 2``) enter risk sets until their observed
    competing-event time but never contribute a primary-event numerator.
    """
    risk_scores = risk_scores.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1)
    if not (len(risk_scores) == len(times) == len(events)):
        raise ValueError("risk_scores, times, and events must have equal length.")
    event_mask = events == 1
    if event_mask.sum() == 0:
        return risk_scores.sum() * 0.0

    order = torch.argsort(times, descending=True, stable=True)
    sorted_times = times[order]
    sorted_risks = risk_scores[order]
    sorted_events = (events[order] == 1).to(risk_scores.dtype)
    _, group_counts = torch.unique_consecutive(sorted_times, return_counts=True)
    group_ids = torch.repeat_interleave(
        torch.arange(len(group_counts), device=times.device),
        group_counts,
    )
    event_counts = torch.zeros(
        len(group_counts), dtype=risk_scores.dtype, device=risk_scores.device
    )
    event_score_sums = torch.zeros_like(event_counts)
    event_counts.scatter_add_(0, group_ids, sorted_events)
    event_score_sums.scatter_add_(0, group_ids, sorted_events * sorted_risks)

    group_ends = torch.cumsum(group_counts, dim=0) - 1
    log_risk_denominators = torch.logcumsumexp(sorted_risks, dim=0)[group_ends]
    log_likelihood = event_score_sums - event_counts * log_risk_denominators
    return -log_likelihood.sum() / event_counts.sum()


def cox_batch_signal_counts(
    times: torch.Tensor,
    events: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return primary-event and genuinely comparable-event counts."""
    times = times.reshape(-1)
    events = events.reshape(-1)
    event_times = times[events == 1]
    n_events = torch.tensor(event_times.numel(), device=times.device)
    if event_times.numel() == 0:
        return n_events, n_events.clone()
    risk_set_sizes = (times.unsqueeze(0) >= event_times.unsqueeze(1)).sum(dim=1)
    n_comparable = (risk_set_sizes >= 2).sum()
    return n_events, n_comparable


class SurvivalFinetuneModule(L.LightningModule):
    """Train a finetune head with Cox or IPCW-weighted BCE objectives."""

    def __init__(
        self,
        model: nn.Module,
        training_mode: str,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        pos_weight: torch.Tensor = None,
        checkpoint_metadata: dict = None,
        horizon_days: float = None,
    ):
        super().__init__()
        if training_mode not in {
            "cox",
            "cox_exact_cached",
            "ipcw_bce",
            "ipcw_cif_bce",
        }:
            raise ValueError(
                "training_mode must be one of {'cox', 'cox_exact_cached', "
                "'ipcw_bce', 'ipcw_cif_bce'}."
            )
        if training_mode in {"cox", "cox_exact_cached"} and pos_weight is not None:
            raise ValueError("Cox survival training does not accept pos_weight.")
        if training_mode not in {"cox", "cox_exact_cached"} and (
            horizon_days is None or float(horizon_days) <= 0
        ):
            raise ValueError("IPCW survival training requires a positive horizon_days.")

        self.save_hyperparameters(ignore=["model"])
        attach_model_config(self, model)
        attach_checkpoint_metadata(self, checkpoint_metadata)
        self.model = model
        self.training_mode = training_mode
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.horizon_days = None if horizon_days is None else float(horizon_days)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

        self._val_risk_scores: List[torch.Tensor] = []
        self._val_times: List[torch.Tensor] = []
        self._val_events: List[torch.Tensor] = []
        self._train_risk_scores: List[torch.Tensor] = []
        self._train_times: List[torch.Tensor] = []
        self._train_events: List[torch.Tensor] = []
        self._train_subject_ids: List[torch.Tensor] = []
        if self.training_mode == "cox_exact_cached":
            self.automatic_optimization = False

    def _logits(self, batch: dict) -> torch.Tensor:
        return self.model(batch).reshape(-1)

    def _loss(self, batch: dict) -> torch.Tensor:
        logits = self._logits(batch)
        if self.training_mode == "cox":
            return cox_partial_likelihood_loss(
                logits,
                batch["time_days"].reshape(-1).float(),
                batch["event"].reshape(-1).long(),
            )

        labels = batch["target"].reshape(-1).float()
        ipcw_weights = batch["ipcw_weight"].reshape(-1).float()
        raw_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        # Weights are normalized once over the complete split. Dividing by the
        # batch size gives an unbiased mini-batch estimate of the IPCW empirical
        # risk; self-normalizing by each batch's observed weight sum does not.
        return (raw_loss * ipcw_weights).mean()

    def on_train_epoch_start(self) -> None:
        """Keep custom sampler shuffling reproducible across checkpoint resume."""
        if self.training_mode == "cox_exact_cached":
            self._train_risk_scores.clear()
            self._train_times.clear()
            self._train_events.clear()
            self._train_subject_ids.clear()
            # Exact gradient caching requires the score pass and recomputation
            # pass to describe the same deterministic network.
            self.model.eval()
            return
        dataloader = getattr(self.trainer, "train_dataloader", None)
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        if self.training_mode == "cox_exact_cached":
            self.model.eval()
            with torch.no_grad():
                logits = self._logits(batch)
            self._train_risk_scores.append(logits.detach().cpu())
            self._train_times.append(batch["time_days"].reshape(-1).detach().cpu())
            self._train_events.append(batch["event"].reshape(-1).detach().cpu())
            self._train_subject_ids.append(
                batch["subject_id"].reshape(-1).detach().cpu()
            )
            return None

        loss = self._loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        if self.training_mode == "cox":
            n_events, n_comparable = cox_batch_signal_counts(
                batch["time_days"], batch["event"]
            )
            self.log("train/events_per_batch", n_events.float(), on_step=True)
            self.log(
                "train/comparable_events_per_batch",
                n_comparable.float(),
                on_step=True,
            )
            self.log(
                "train/zero_signal_batch",
                (n_comparable == 0).float(),
                on_step=True,
                on_epoch=True,
            )
        return loss

    def on_train_epoch_end(self) -> None:
        if self.training_mode != "cox_exact_cached":
            return
        if not self._train_risk_scores:
            raise RuntimeError("Exact cached Cox received no training batches.")

        cached_scores = torch.cat(self._train_risk_scores).to(
            self.device, dtype=torch.float32
        )
        cached_scores.requires_grad_(True)
        cached_times = torch.cat(self._train_times).to(
            self.device, dtype=torch.float32
        )
        cached_events = torch.cat(self._train_events).to(
            self.device, dtype=torch.long
        )
        cached_ids = torch.cat(self._train_subject_ids).reshape(-1)
        if len(torch.unique(cached_ids)) != len(cached_ids):
            raise RuntimeError(
                "Exact cached Cox requires every training patient exactly once "
                "per full-cohort score pass."
            )

        exact_loss = exact_breslow_cox_loss(
            cached_scores,
            cached_times,
            cached_events,
        )
        score_gradients = torch.autograd.grad(exact_loss, cached_scores)[0].cpu()
        gradient_by_subject = {
            int(subject_id): float(gradient)
            for subject_id, gradient in zip(cached_ids.tolist(), score_gradients)
        }

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.model.eval()
        seen: set[int] = set()
        dataloader = self.trainer.train_dataloader
        for batch in dataloader:
            batch = self.trainer.strategy.batch_to_device(batch, self.device)
            subject_ids = batch["subject_id"].reshape(-1).detach().cpu().tolist()
            duplicate = seen.intersection(int(item) for item in subject_ids)
            if duplicate:
                raise RuntimeError(
                    "Exact cached Cox recomputation repeated subjects: "
                    f"{sorted(duplicate)[:10]}."
                )
            seen.update(int(item) for item in subject_ids)
            batch_gradients = torch.tensor(
                [gradient_by_subject[int(item)] for item in subject_ids],
                dtype=torch.float32,
                device=self.device,
            )
            with self.trainer.precision_plugin.forward_context():
                recomputed_scores = self._logits(batch)
                surrogate = torch.sum(recomputed_scores.float() * batch_gradients)
            self.manual_backward(surrogate)

        if seen != set(gradient_by_subject):
            missing = sorted(set(gradient_by_subject) - seen)
            raise RuntimeError(
                "Exact cached Cox recomputation did not cover the full cohort; "
                f"missing examples={missing[:10]}."
            )
        optimizer.step()
        # Lightning's manual-optimization step counter (trainer.global_step)
        # only advances when optimizer.step() is called inside training_step's
        # dynamic scope -- lightning/pytorch/loops/optimization/manual.py installs
        # its _on_before_step/_on_after_step hooks around that call specifically
        # and tears them down immediately after. Our real optimizer step happens
        # here instead, once per epoch, so global_step would otherwise stay 0 for
        # the entire run. ModelCheckpoint's save-best guard
        # (_last_global_step_saved == trainer.global_step) is then trivially
        # true forever, silently skipping every "best" checkpoint save (confirmed
        # by direct reproduction; save_last is unaffected since it isn't gated on
        # that check). Increment the same progress counter Lightning's own hooks
        # would have, so global_step -- and everything that depends on it --
        # reflects the step that actually happened.
        manual_opt_progress = (
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress
        )
        manual_opt_progress.increment_ready()
        manual_opt_progress.increment_completed()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        self.log(
            "train/loss",
            exact_loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        logits = self._logits(batch)
        if self.training_mode == "cox":
            loss = cox_partial_likelihood_loss(
                logits,
                batch["time_days"].reshape(-1).float(),
                batch["event"].reshape(-1).long(),
            )
        elif self.training_mode == "cox_exact_cached":
            loss = None
        else:
            labels = batch["target"].reshape(-1).float()
            ipcw_weights = batch["ipcw_weight"].reshape(-1).float()
            raw_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=self.pos_weight,
                reduction="none",
            )
            loss = (raw_loss * ipcw_weights).mean()

        # NumPy has no bfloat16 dtype. Store metric inputs as float32 so AMP
        # validation works identically for BF16 and FP16/FP32 execution.
        self._val_risk_scores.append(logits.detach().float().cpu())
        self._val_times.append(batch["time_days"].reshape(-1).detach().cpu())
        self._val_events.append(batch["event"].reshape(-1).detach().cpu())
        if loss is not None:
            self.log("val/loss", loss, prog_bar=True, batch_size=len(logits))
        return loss

    def on_validation_epoch_end(self):
        if self._val_risk_scores:
            logits = torch.cat(self._val_risk_scores).float().numpy()
            times = torch.cat(self._val_times).numpy().astype(float)
            events = torch.cat(self._val_events).numpy().astype(int)
            if self.training_mode in {"cox", "cox_exact_cached"}:
                if self.training_mode == "cox_exact_cached":
                    exact_val_loss = exact_breslow_cox_loss(
                        torch.from_numpy(logits).float(),
                        torch.from_numpy(times).float(),
                        torch.from_numpy(events).long(),
                    )
                    self.log(
                        "val/loss",
                        exact_val_loss.to(self.device),
                        prog_bar=True,
                    )
                c_index = compute_concordance_index(
                    times,
                    events,
                    logits.astype(float),
                )
                if not math.isfinite(c_index):
                    c_index = 0.0
                self.log(
                    "val/concordance_index",
                    torch.tensor(c_index, dtype=torch.float32, device=self.device),
                    prog_bar=True,
                )
            else:
                probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
                if self.training_mode == "ipcw_cif_bce":
                    metric = compute_competing_risk_metrics_at_horizon(
                        times,
                        events,
                        probabilities,
                        self.horizon_days,
                    )["cif_auc"]
                else:
                    metric = compute_ipcw_metrics_at_horizon(
                        times,
                        events,
                        probabilities,
                        self.horizon_days,
                        _km_censoring_fn(times, events),
                    )["ipcw_auc"]
                if not math.isfinite(metric):
                    metric = 0.0
                self.log(
                    "val/AUROC",
                    torch.tensor(metric, dtype=torch.float32, device=self.device),
                    prog_bar=True,
                )
        self._val_risk_scores.clear()
        self._val_times.clear()
        self._val_events.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_epsilon,
        )
        if self.training_mode == "cox_exact_cached":
            warmup_steps = max(0, round(float(self.scheduler_warmup_epochs)))
            if warmup_steps == 0:
                return optimizer
            scheduler = LinearLR(
                optimizer,
                start_factor=1e-4,
                total_iters=warmup_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer_with_warmup(
            optimizer,
            self.trainer,
            self.scheduler_warmup_epochs,
        )
