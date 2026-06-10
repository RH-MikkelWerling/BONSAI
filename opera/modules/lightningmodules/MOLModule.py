"""
Lightning module for Multi-Outcome Learning (MOL).

Structurally parallel to OperaContrastiveModule but optimises BCE
per outcome instead of SupCon.  Logs the same sigma/precision values
for direct comparison.

Additionally tracks per-outcome AUROC and AUPRC on validation, which
the contrastive module cannot do (since it has no classification heads).
"""

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchmetrics import AUROC, AveragePrecision
from typing import Dict, List
from bonsai.functional.checkpointing import (
    attach_checkpoint_metadata,
    attach_model_config,
)


class MOLModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        outcome_names: List[str],
        learning_rate: float = 1e-4,
        encoder_lr_multiplier: float = 0.1,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 1,
        checkpoint_metadata: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        attach_model_config(self, model)
        attach_checkpoint_metadata(self, checkpoint_metadata)
        self.model = model
        self.outcome_names = outcome_names
        self.learning_rate = learning_rate
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs

        # Per-outcome validation metrics
        self.val_aurocs = nn.ModuleDict(
            {name: AUROC(task="binary") for name in outcome_names}
        )
        self.val_auprcs = nn.ModuleDict(
            {name: AveragePrecision(task="binary") for name in outcome_names}
        )

    def _build_outcome_labels(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Extract outcome labels from batch dict."""
        outcome_labels = {}
        for name in self.outcome_names:
            key = f"outcome_{name}"
            if key in batch:
                outcome_labels[name] = batch[key]
            else:
                outcome_labels[name] = torch.full(
                    (batch["code"].size(0),),
                    -1,
                    dtype=torch.long,
                    device=batch["code"].device,
                )
        return outcome_labels

    def training_step(self, batch, batch_idx):
        outcome_labels = self._build_outcome_labels(batch)
        log_dict = self.model(batch, outcome_labels)
        loss = log_dict["loss"]

        self.log("train/loss", loss, prog_bar=True)
        for k, v in log_dict.items():
            if k != "loss":
                self.log(f"train/{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        outcome_labels = self._build_outcome_labels(batch)
        log_dict = self.model(batch, outcome_labels)
        loss = log_dict["loss"]

        self.log("val/loss", loss, prog_bar=True)
        for k, v in log_dict.items():
            if k != "loss":
                self.log(f"val/{k}", v)

        # Per-outcome classification metrics
        probs = self.model.predict(batch)
        for name in self.outcome_names:
            labels = outcome_labels[name]
            valid = labels >= 0
            if valid.sum() < 2:
                continue
            p = probs[name][valid]
            y = labels[valid]
            if len(y.unique()) > 1:
                self.val_aurocs[name].update(p, y)
                self.val_auprcs[name].update(p, y)

        return loss

    def on_validation_epoch_end(self):
        for name in self.outcome_names:
            try:
                auroc_val = self.val_aurocs[name].compute()
                self.log(f"val/auroc/{name}", auroc_val, prog_bar=False)
                self.val_aurocs[name].reset()
            except Exception:
                pass
            try:
                auprc_val = self.val_auprcs[name].compute()
                self.log(f"val/auprc/{name}", auprc_val, prog_bar=False)
                self.val_auprcs[name].reset()
            except Exception:
                pass

    def configure_optimizers(self):
        encoder_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": head_params, "lr": self.learning_rate},
        ]
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": self.learning_rate * self.encoder_lr_multiplier,
                }
            )

        optimizer = AdamW(param_groups, eps=self.optimizer_epsilon)

        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(steps_per_epoch * self.scheduler_warmup_epochs),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
