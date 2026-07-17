"""
DAPT PretrainModule with vocabulary-aware learning rates.

Extends BONSAI's PretrainModule to support differential learning rates
when the vocabulary has been expanded with domain-specific tokens.
When no expansion is used, this behaves identically to the base module.
"""

import lightning as L
from torch import nn
from torch.optim import AdamW
from torchmetrics import MetricCollection, Precision
from bonsai.functional.checkpointing import (
    attach_checkpoint_metadata,
    attach_model_config,
)
from bonsai.functional.scheduling import optimizer_with_warmup
from bonsai.modules.lightningmodules.PretrainModule import compute_pretrain_loss


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

        self.save_hyperparameters(dict(model.hparams), ignore=["model"])
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
        self.value_bin_loss = nn.CrossEntropyLoss()
        self.value_regression_loss = nn.MSELoss()
        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")

    def _configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/Prec-K1": Precision(
                    task="multiclass",
                    num_classes=self.model.hparams["vocab_size"],
                    top_k=1,
                ),
                f"{prefix}/Prec-K10": Precision(
                    task="multiclass",
                    num_classes=self.model.hparams["vocab_size"],
                    top_k=10,
                ),
            }
        )

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss, logits, labels, losses = compute_pretrain_loss(
            output,
            self.train_loss,
            self.value_bin_loss,
            self.value_regression_loss,
        )
        self.train_metrics(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        if "value_bin" in losses:
            self.log("train/code_loss", losses["code"], prog_bar=False)
            self.log("train/value_bin_loss", losses["value_bin"], prog_bar=False)
            self.log(
                "train/value_regression_loss",
                losses["value_regression"],
                prog_bar=False,
            )
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        loss, logits, labels, losses = compute_pretrain_loss(
            output,
            self.val_loss,
            self.value_bin_loss,
            self.value_regression_loss,
        )
        self.log("val/loss", loss, prog_bar=True)
        if "value_bin" in losses:
            self.log("val/code_loss", losses["code"], prog_bar=False)
            self.log("val/value_bin_loss", losses["value_bin"], prog_bar=False)
            self.log(
                "val/value_regression_loss",
                losses["value_regression"],
                prog_bar=False,
            )
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
                if "code_embedding" in name or name.startswith("pretrain_head"):
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

        return optimizer_with_warmup(
            optimizer,
            self.trainer,
            self.scheduler_warmup_epochs,
        )
