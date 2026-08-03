import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import MetricCollection

from bonsai.modules.metrics.metrics import SharedPrecisionAtK
from bonsai.functional.checkpointing import (
    attach_checkpoint_metadata,
    attach_model_config,
)

loss_types = {"sdpa": nn.CrossEntropyLoss, "flash": nn.CrossEntropyLoss}
try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss

    loss_types["flash"] = FACrossEntropyLoss
except Exception:
    pass


def unpack_pretrain_output(output):
    if isinstance(output, dict):
        return output["logits"], output["labels"]
    return output


def compute_pretrain_loss(
    output,
    code_loss_fn,
    value_bin_loss_fn,
    value_regression_loss_fn,
    *,
    value_bin_loss_weight: float = 1.0,
    value_regression_loss_weight: float = 1.0,
):
    logits, labels = unpack_pretrain_output(output)
    loss = code_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    losses = {"code": loss}
    if (
        isinstance(output, dict)
        and "target_value_normalized" in output
        and output["target_value_normalized"].numel() > 0
    ):
        value_regression_loss = value_regression_loss_fn(
            output["value_prediction"].view(-1),
            output["target_value_normalized"].view(-1),
        )
        losses["value_regression"] = value_regression_loss
        if output.get("value_embedding_mode") in {"combined_binning", "film"}:
            loss = loss + float(value_regression_loss_weight) * value_regression_loss
        else:
            value_bin_loss = value_bin_loss_fn(
                output["value_bin_logits"].view(
                    -1, output["value_bin_logits"].size(-1)
                ),
                output["target_value_bin"].view(-1),
            )
            losses["value_bin"] = value_bin_loss
            loss = (
                loss
                + float(value_bin_loss_weight) * value_bin_loss
                + float(value_regression_loss_weight) * value_regression_loss
            )
    losses["total"] = loss
    return loss, logits, labels, losses


class PretrainModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = None,
        learning_rate: float = 5e-4,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: float = 0,
        value_bin_loss_weight: float = 1.0,
        value_regression_loss_weight: float = 1.0,
        checkpoint_metadata: dict = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.value_bin_loss_weight = value_bin_loss_weight
        self.value_regression_loss_weight = value_regression_loss_weight

        self.model = model
        if compile_mode is not None:
            # Sparse concept/value prediction uses boolean indexing whose
            # output size varies with each batch. Dynamo can keep this in one
            # graph when dynamic output shapes are explicitly captured.
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            self.model.compile(mode=compile_mode)

        self.train_loss = loss_types[self.model.hparams["attn_type"]]()
        self.val_loss = nn.CrossEntropyLoss()
        self.value_bin_loss = nn.CrossEntropyLoss()
        self.value_regression_loss = nn.MSELoss()
        self.val_metrics = self.configure_metrics("val")

        hparams = self.model.hparams.copy()
        hparams.update(
            {
                "learning_rate": learning_rate,
                "optimizer_epsilon": optimizer_epsilon,
                "scheduler_warmup_epochs": scheduler_warmup_epochs,
                "value_bin_loss_weight": value_bin_loss_weight,
                "value_regression_loss_weight": value_regression_loss_weight,
            }
        )
        self.save_hyperparameters(hparams)
        attach_model_config(self, model)
        attach_checkpoint_metadata(self, checkpoint_metadata)

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/Precision@1": SharedPrecisionAtK(
                    k=1, max_k=100, reduce="mean"
                ),
                f"{prefix}/Precision@10": SharedPrecisionAtK(
                    k=10, max_k=100, reduce="mean"
                ),
                f"{prefix}/Precision@100": SharedPrecisionAtK(
                    k=100, max_k=100, reduce="mean"
                ),
            },
            compute_groups=[
                [
                    f"{prefix}/Precision@1",
                    f"{prefix}/Precision@10",
                    f"{prefix}/Precision@100",
                ]
            ],
        )

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss, _, _, losses = compute_pretrain_loss(
            output,
            self.train_loss,
            self.value_bin_loss,
            self.value_regression_loss,
            value_bin_loss_weight=self.value_bin_loss_weight,
            value_regression_loss_weight=self.value_regression_loss_weight,
        )
        self.log("train/loss", loss, prog_bar=True)
        if "value_regression" in losses:
            self.log("train/code_loss", losses["code"], prog_bar=False)
            if "value_bin" in losses:
                self.log("train/value_bin_loss", losses["value_bin"], prog_bar=False)
            self.log(
                "train/value_regression_loss",
                losses["value_regression"],
                prog_bar=False,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        loss, logits, labels, losses = compute_pretrain_loss(
            output,
            self.val_loss,
            self.value_bin_loss,
            self.value_regression_loss,
            value_bin_loss_weight=self.value_bin_loss_weight,
            value_regression_loss_weight=self.value_regression_loss_weight,
        )
        self.log("val/loss", loss, prog_bar=True)
        if "value_regression" in losses:
            self.log("val/code_loss", losses["code"], prog_bar=False)
            if "value_bin" in losses:
                self.log("val/value_bin_loss", losses["value_bin"], prog_bar=False)
            self.log(
                "val/value_regression_loss",
                losses["value_regression"],
                prog_bar=False,
            )
        self.val_metrics.update(logits, labels)
        self.log_dict(self.val_metrics)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_epsilon,
        )
        if self.scheduler_warmup_epochs == 0:
            return optimizer

        steps_per_epoch = max(
            1,
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
        )
        warmup_steps = max(1, round(steps_per_epoch * self.scheduler_warmup_epochs))
        scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=1e-4,
            total_iters=warmup_steps,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
