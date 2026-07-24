"""Dual-mode survival finetuning module for OPERA."""

from __future__ import annotations

import math
from typing import List

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

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
        if training_mode not in {"cox", "ipcw_bce", "ipcw_cif_bce"}:
            raise ValueError(
                "training_mode must be one of {'cox', 'ipcw_bce', 'ipcw_cif_bce'}."
            )
        if training_mode == "cox" and pos_weight is not None:
            raise ValueError("Cox survival training does not accept pos_weight.")
        if training_mode != "cox" and (
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
        dataloader = getattr(self.trainer, "train_dataloader", None)
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
        logits = self._logits(batch)
        if self.training_mode == "cox":
            loss = cox_partial_likelihood_loss(
                logits,
                batch["time_days"].reshape(-1).float(),
                batch["event"].reshape(-1).long(),
            )
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

        self._val_risk_scores.append(logits.detach().cpu())
        self._val_times.append(batch["time_days"].reshape(-1).detach().cpu())
        self._val_events.append(batch["event"].reshape(-1).detach().cpu())
        self.log("val/loss", loss, prog_bar=True, batch_size=len(logits))
        return loss

    def on_validation_epoch_end(self):
        if self._val_risk_scores:
            logits = torch.cat(self._val_risk_scores).numpy()
            times = torch.cat(self._val_times).numpy().astype(float)
            events = torch.cat(self._val_events).numpy().astype(int)
            if self.training_mode == "cox":
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
        return optimizer_with_warmup(
            optimizer,
            self.trainer,
            self.scheduler_warmup_epochs,
        )
