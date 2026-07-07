"""Shared optimizer-scheduler helpers."""

from __future__ import annotations

from torch.optim.lr_scheduler import LinearLR


def build_linear_warmup_scheduler(optimizer, trainer, warmup_epochs: float):
    """Build upstream's linear warmup with safe integer step accounting."""
    if float(warmup_epochs) <= 0:
        return None
    steps_per_epoch = max(
        1,
        int(trainer.estimated_stepping_batches) // max(1, int(trainer.max_epochs)),
    )
    warmup_steps = max(1, round(steps_per_epoch * float(warmup_epochs)))
    return LinearLR(
        optimizer=optimizer,
        start_factor=1e-4,
        total_iters=warmup_steps,
    )


def optimizer_with_warmup(optimizer, trainer, warmup_epochs: float):
    """Return the Lightning optimizer configuration with optional warmup."""
    scheduler = build_linear_warmup_scheduler(optimizer, trainer, warmup_epochs)
    if scheduler is None:
        return optimizer
    return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
