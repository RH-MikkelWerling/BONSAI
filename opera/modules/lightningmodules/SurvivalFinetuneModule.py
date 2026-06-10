"""Dual-mode survival finetuning module for OPERA."""

from __future__ import annotations

import math
from typing import List

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torchmetrics import AUROC
from transformers import get_linear_schedule_with_warmup

from bonsai.functional.checkpointing import (
    attach_checkpoint_metadata,
    attach_model_config,
)
from opera.evaluation.metrics import compute_concordance_index


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
    ):
        super().__init__()
        if training_mode not in {"cox", "ipcw_bce"}:
            raise ValueError("training_mode must be one of {'cox', 'ipcw_bce'}.")
        if training_mode == "cox" and pos_weight is not None:
            raise ValueError("Cox survival training does not accept pos_weight.")

        self.save_hyperparameters(ignore=["model"])
        attach_model_config(self, model)
        attach_checkpoint_metadata(self, checkpoint_metadata)
        self.model = model
        self.training_mode = training_mode
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

        self.val_auroc = AUROC(task="binary") if training_mode == "ipcw_bce" else None
        self._val_auroc_updated = False
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
        return (raw_loss * ipcw_weights).sum() / (ipcw_weights.sum() + 1e-8)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self._logits(batch)
        if self.training_mode == "cox":
            loss = cox_partial_likelihood_loss(
                logits,
                batch["time_days"].reshape(-1).float(),
                batch["event"].reshape(-1).long(),
            )
            self._val_risk_scores.append(logits.detach().cpu())
            self._val_times.append(batch["time_days"].reshape(-1).detach().cpu())
            self._val_events.append(batch["event"].reshape(-1).detach().cpu())
        else:
            labels = batch["target"].reshape(-1).float()
            ipcw_weights = batch["ipcw_weight"].reshape(-1).float()
            raw_loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=self.pos_weight,
                reduction="none",
            )
            loss = (raw_loss * ipcw_weights).sum() / (ipcw_weights.sum() + 1e-8)
            metric_mask = ipcw_weights > 0.0
            if metric_mask.sum() >= 2 and labels[metric_mask].unique().numel() > 1:
                self.val_auroc.update(
                    torch.sigmoid(logits[metric_mask]), labels[metric_mask].long()
                )
                self._val_auroc_updated = True

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.training_mode == "cox":
            if self._val_risk_scores:
                risk = torch.cat(self._val_risk_scores).numpy()
                times = torch.cat(self._val_times).numpy()
                events = torch.cat(self._val_events).numpy()
                c_index = compute_concordance_index(
                    times.astype(float),
                    events.astype(int),
                    risk.astype(float),
                )
                if not math.isfinite(c_index):
                    c_index = 0.0
                self.log(
                    "val/concordance_index",
                    torch.tensor(c_index, dtype=torch.float32, device=self.device),
                    prog_bar=True,
                )
            self._val_risk_scores.clear()
            self._val_times.clear()
            self._val_events.clear()
        else:
            if self._val_auroc_updated:
                auroc = self.val_auroc.compute()
                self.log("val/AUROC", auroc, prog_bar=True)
            else:
                self.log(
                    "val/AUROC",
                    torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    prog_bar=True,
                )
            self.val_auroc.reset()
            self._val_auroc_updated = False

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_epsilon,
        )
        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(steps_per_epoch * self.scheduler_warmup_epochs),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
