"""
DAPT PretrainModule with vocabulary-aware learning rates.

Extends BONSAI's PretrainModule to support differential learning rates
when the vocabulary has been expanded with domain-specific tokens.
When no expansion is used, this behaves identically to the base module.
"""

import lightning as L
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchmetrics import MetricCollection, Precision
from bonsai.functional.checkpointing import (
    attach_checkpoint_metadata,
    attach_model_config,
)


class DAPTPretrainModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = None,
        learning_rate: float = 1e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 0,
        # Vocab expansion params
        old_vocab_size: int = 0,
        new_embed_lr_multiplier: float = 5.0,
        freeze_pretrained_embeds: bool = False,
        checkpoint_metadata: dict = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.old_vocab_size = old_vocab_size
        self.new_embed_lr_multiplier = new_embed_lr_multiplier
        self.freeze_pretrained_embeds = freeze_pretrained_embeds

        self.save_hyperparameters(model.config.to_dict(), ignore=["model"])
        attach_model_config(self, model)
        attach_checkpoint_metadata(self, checkpoint_metadata)
        self.model = model
        if compile_mode is not None:
            self.model.compile(mode=compile_mode)

        # Optionally freeze pretrained embedding rows via gradient hooks
        if freeze_pretrained_embeds and old_vocab_size > 0:
            from opera.functional.vocab_expansion import freeze_pretrained_embeddings

            freeze_pretrained_embeddings(self.model, old_vocab_size)

        self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")

    def _configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/Prec-K1": Precision(
                    task="multiclass",
                    num_classes=self.model.config.vocab_size,
                    top_k=1,
                ),
                f"{prefix}/Prec-K10": Precision(
                    task="multiclass",
                    num_classes=self.model.config.vocab_size,
                    top_k=10,
                ),
            }
        )

    def training_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.train_loss(
            logits.view(-1, self.model.config.vocab_size), labels.view(-1)
        )
        self.train_metrics(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self.model(batch)
        loss = self.val_loss(
            logits.view(-1, self.model.config.vocab_size), labels.view(-1)
        )
        self.log("val/loss", loss, prog_bar=True)
        self.val_metrics(logits, labels)
        self.log_dict(self.val_metrics)
        return loss

    def configure_optimizers(self):
        if self.old_vocab_size > 0 and not self.freeze_pretrained_embeds:
            # Differential LR: embedding/decoder layers get higher LR
            from opera.functional.vocab_expansion import get_vocab_aware_param_groups

            param_groups = get_vocab_aware_param_groups(
                self.model,
                base_lr=self.learning_rate,
                new_embed_lr_multiplier=self.new_embed_lr_multiplier,
                old_vocab_size=self.old_vocab_size,
            )
        elif self.freeze_pretrained_embeds and self.old_vocab_size > 0:
            # Pretrained rows frozen via hooks → can use higher LR on
            # the whole embedding layer since only new rows receive gradients
            embed_decoder_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "code_embedding" in name or (
                    name.startswith("decoder") and "weight" in name
                ):
                    embed_decoder_params.append(param)
                else:
                    other_params.append(param)

            param_groups = [
                {"params": other_params, "lr": self.learning_rate},
            ]
            if embed_decoder_params:
                param_groups.append(
                    {
                        "params": embed_decoder_params,
                        "lr": self.learning_rate * self.new_embed_lr_multiplier,
                    }
                )
        else:
            # No expansion — single param group, same as base PretrainModule
            param_groups = [
                {
                    "params": [p for p in self.model.parameters() if p.requires_grad],
                    "lr": self.learning_rate,
                },
            ]

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
