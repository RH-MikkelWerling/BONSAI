"""
Lightning module for OPERA joint multi-task finetuning.

Trains a single model on all cohorts and all outcomes simultaneously.
Logs per-outcome AUROC on validation so you can monitor which outcomes
are learning and which are struggling.
"""

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchmetrics import AUROC, AveragePrecision
from typing import Dict, List
from bonsai.functional.checkpointing import (
    MODEL_INIT_CONFIG_KEY,
    attach_checkpoint_metadata,
    attach_model_config,
)


class JointFinetuneModule(L.LightningModule):
    """
    Parameters
    ----------
    model         : JointFinetuneModel
    outcome_names : must match model.outcome_names
    learning_rate : head LR
    encoder_lr_multiplier : encoder LR = learning_rate * this
                            (lower than heads — encoder is already pretrained)
    """

    def __init__(
        self,
        model: nn.Module,
        outcome_names: List[str],
        learning_rate: float = 5e-4,
        encoder_lr_multiplier: float = 0.1,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 1,
        checkpoint_metadata: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        attach_model_config(self, model)
        self.hparams[MODEL_INIT_CONFIG_KEY] = {
            "hidden_size": model.encoder.config.hidden_size,
            "pooling": model.pooling,
            "freeze_encoder": model.freeze_encoder,
            "dropout": model.dropout.p,
        }
        attach_checkpoint_metadata(self, checkpoint_metadata)
        self.model = model
        self.outcome_names = outcome_names

        # Per-outcome validation metrics
        self.val_auroc = nn.ModuleDict(
            {name: AUROC(task="binary") for name in outcome_names}
        )
        self.val_auprc = nn.ModuleDict(
            {name: AveragePrecision(task="binary") for name in outcome_names}
        )

    def _build_outcome_labels(self, batch: dict) -> Dict[str, torch.Tensor]:
        return {
            name: batch[f"outcome_{name}"]
            for name in self.outcome_names
            if f"outcome_{name}" in batch
        }

    def training_step(self, batch, batch_idx):
        outcome_labels = self._build_outcome_labels(batch)
        log_dict = self.model(batch, outcome_labels)
        loss = log_dict["loss"]

        self.log("train/loss", loss, prog_bar=True)
        for k, v in log_dict.items():
            if k.startswith("loss/") or k.startswith("sigma/"):
                self.log(f"train/{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        outcome_labels = self._build_outcome_labels(batch)
        log_dict = self.model(batch, outcome_labels)
        loss = log_dict["loss"]
        self.log("val/loss", loss, prog_bar=True)

        # Update per-outcome metrics
        for name in self.outcome_names:
            labels_k = outcome_labels.get(name)
            if labels_k is None:
                continue
            valid = labels_k >= 0
            if valid.sum() < 2:
                continue
            logits_k = log_dict.get(f"logits/{name}")
            if logits_k is None:
                # recompute if not cached (shouldn't happen in normal flow)
                logits_k = self.model.predict(batch, name)[valid]
                labels_v = labels_k[valid]
            else:
                labels_v = labels_k[valid]

            probs_k = torch.sigmoid(logits_k)
            self.val_auroc[name].update(probs_k, labels_v)
            self.val_auprc[name].update(probs_k, labels_v)

        return loss

    def on_validation_epoch_end(self):
        aurocs = []
        for name in self.outcome_names:
            try:
                auroc = self.val_auroc[name].compute()
                auprc = self.val_auprc[name].compute()
                if not torch.isfinite(auroc):
                    self.val_auroc[name].reset()
                    self.val_auprc[name].reset()
                    continue
                self.log(f"val/auroc_{name}", auroc)
                if torch.isfinite(auprc):
                    self.log(f"val/auprc_{name}", auprc)
                aurocs.append(auroc)
                self.val_auroc[name].reset()
                self.val_auprc[name].reset()
            except (RuntimeError, ValueError):
                continue

        # Macro-average AUROC — used as the primary monitor metric
        if aurocs:
            self.log("val/auroc_macro", torch.stack(aurocs).mean(), prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        enc_mult = self.hparams.encoder_lr_multiplier

        encoder_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = [{"params": head_params, "lr": lr}]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": lr * enc_mult})

        optimizer = AdamW(param_groups, eps=self.hparams.optimizer_epsilon)
        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                steps_per_epoch * self.hparams.scheduler_warmup_epochs
            ),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
