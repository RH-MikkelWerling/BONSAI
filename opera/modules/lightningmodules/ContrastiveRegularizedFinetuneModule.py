"""
Contrastive-Regularized Finetuning.

Addresses the information bottleneck between OPERA and finetuning:
instead of discarding the projection head, keep it and add an
outcome-selective contrastive regularization term to the finetune loss:

    L = BCE_task + λ(t) · L_contrastive(target_outcome_only)

Key design: the regularizer is OUTCOME-SELECTIVE — it only preserves
the contrastive structure for the outcome being finetuned, not the
full multi-outcome geometry.  This avoids the problem where preserving
mortality structure hurts an adverse event prediction task.

λ decays over training (cosine schedule from λ_init to 0), so the model
starts from the contrastive geometry and gradually specializes.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
import lightning as L
from torchmetrics import MetricCollection, Accuracy, AUROC, AveragePrecision
from bonsai.functional.checkpointing import (
    attach_checkpoint_metadata,
    attach_model_config,
)
from bonsai.functional.scheduling import optimizer_with_warmup

from opera.modules.networks.opera_nets import (
    ProjectionHead,
    SupervisedContrastiveLoss,
)


class ContrastiveRegularizedFinetuneModule(L.LightningModule):
    """
    Parameters
    ----------
    model : nn.Module
        The classification model (e.g. BonsaiFinetune).
    projection_head : ProjectionHead
        The OPERA projection head (loaded from contrastive checkpoint).
    lambda_init : float
        Initial weight for the contrastive regularizer.
    lambda_schedule : str
        "cosine" — decays from lambda_init to 0 over training.
        "constant" — fixed at lambda_init throughout.
        "linear" — linear decay to 0.
    contrastive_temperature : float
        Temperature for SupCon loss (should match OPERA training).
    """

    def __init__(
        self,
        model: nn.Module,
        projection_head: Optional[ProjectionHead] = None,
        lambda_init: float = 0.1,
        lambda_schedule: str = "cosine",
        contrastive_temperature: float = 0.07,
        pos_weight: torch.Tensor = None,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        checkpoint_metadata: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "projection_head"])
        attach_model_config(self, model)
        attach_checkpoint_metadata(self, checkpoint_metadata)
        self.model = model
        self.projection_head = projection_head
        self.lambda_init = lambda_init
        self.lambda_schedule = lambda_schedule
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs

        self.task_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if projection_head is not None:
            self.contrastive_loss = SupervisedContrastiveLoss(
                temperature=contrastive_temperature
            )
        else:
            self.contrastive_loss = None

        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")

    def _configure_metrics(self, prefix):
        return MetricCollection(
            {
                f"{prefix}/Accuracy": Accuracy(task="binary", threshold=0.6),
                f"{prefix}/AUROC": AUROC(task="binary"),
                f"{prefix}/AveragePrecision": AveragePrecision(task="binary"),
            }
        )

    def _get_lambda(self) -> float:
        """Compute current λ based on training progress."""
        if self.contrastive_loss is None or self.lambda_init == 0:
            return 0.0

        if self.lambda_schedule == "constant":
            return self.lambda_init

        # Compute progress as fraction of total training
        if self.trainer.max_epochs:
            progress = self.current_epoch / self.trainer.max_epochs
        else:
            progress = self.global_step / self.trainer.estimated_stepping_batches
        progress = min(progress, 1.0)

        if self.lambda_schedule == "cosine":
            return self.lambda_init * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.lambda_schedule == "linear":
            return self.lambda_init * (1 - progress)
        else:
            return self.lambda_init

    def _get_embeddings_for_regularizer(self, batch: dict) -> torch.Tensor:
        """
        Extract pooled embeddings from the classification model and
        project through the OPERA projection head.

        This requires reaching into the model's internals to get the
        pre-classification-head representation.
        """
        if hasattr(self.model, "get_pooled_representation"):
            pooled = self.model.get_pooled_representation(batch)
            return self.projection_head(pooled)

        # Legacy fallback for non-native classification wrappers.
        outputs = self.model.encoder(batch)
        from opera.compat.bonsai import encoder_hidden_state

        hidden = encoder_hidden_state(outputs)

        # Pool using the classification head's pooler (BiGRU)
        if hasattr(self.model, "cls") and hasattr(self.model.cls, "pool"):
            pooled = self.model.cls.pool(
                hidden, batch["attention_mask"], return_embedding=True
            )
        else:
            # Fallback: CLS-last
            lengths = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

        # Project
        return self.projection_head(pooled)

    def training_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        task_loss = self.task_loss(logits, labels.float())

        # Contrastive regularization
        lam = self._get_lambda()
        if lam > 0 and self.contrastive_loss is not None:
            embeddings = self._get_embeddings_for_regularizer(batch)
            con_loss = self.contrastive_loss(embeddings, labels.squeeze())
            total_loss = task_loss + lam * con_loss
            self.log("train/contrastive_loss", con_loss, prog_bar=False)
            self.log("train/lambda", lam, prog_bar=False)
        else:
            total_loss = task_loss

        self.train_metrics(logits, labels)
        self.log("train/task_loss", task_loss, prog_bar=True)
        self.log("train/loss", total_loss, prog_bar=True)
        self.log_dict(self.train_metrics)
        return total_loss

    def validation_step(self, batch, batch_idx):
        labels = batch["target"]
        logits = self.model(batch)
        loss = self.task_loss(logits, labels.float())
        self.log("val/loss", loss, prog_bar=True)
        self.val_metrics(logits, labels)
        self.log_dict(self.val_metrics)
        return loss

    def configure_optimizers(self):
        # Include projection head params if present
        params = list(self.model.parameters())
        if self.projection_head is not None:
            params += list(self.projection_head.parameters())

        optimizer = AdamW(
            [p for p in params if p.requires_grad],
            lr=self.learning_rate,
            eps=self.optimizer_epsilon,
        )
        return optimizer_with_warmup(
            optimizer,
            self.trainer,
            self.scheduler_warmup_epochs,
        )
