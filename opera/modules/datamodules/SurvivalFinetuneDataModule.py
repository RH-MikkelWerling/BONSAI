"""Survival-aware finetuning datamodule for OPERA."""

import warnings
from typing import Optional, Literal

import torch
from torch.utils.data import DataLoader

from opera.compat.bonsai import dynamic_padding, filter_subject_data
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from opera.functional.stratified_sampling import (
    build_coverage_balanced_survival_batch_sampler,
    build_event_aware_batch_sampler,
)
from opera.modules.datasets.SurvivalFinetuneDataset import SurvivalFinetuneDataset


def resolve_survival_batch_sampler_type(
    training_mode: str,
    batch_sampling: Optional[dict],
) -> str:
    """Resolve objective-aware sampling without duplicating runner logic."""
    sampler_type = str(dict(batch_sampling or {}).get("type", "auto")).lower()
    if sampler_type == "auto":
        return "coverage_balanced" if training_mode == "cox" else "none"
    if sampler_type == "random":
        return "none"
    if sampler_type == "risk_set_balanced":
        return "coverage_balanced"
    if sampler_type == "survival_event_aware":
        return "event_aware"
    return sampler_type


def survival_finetune_collate(batch: list[dict]) -> dict:
    """Collate standard BONSAI fields plus fixed-size survival tensors."""
    output = dynamic_padding(batch)
    for key in ("time_days", "event", "ipcw_weight"):
        output[key] = torch.stack([sample[key] for sample in batch])
    return output


class SurvivalFinetuneDataModule(FinetuneDataModule):
    """Use ``SurvivalFinetuneDataset`` while preserving BONSAI loaders."""

    def __init__(
        self,
        *args,
        batch_sampling: Optional[dict] = None,
        training_mode: str = "cox",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_sampling = dict(batch_sampling or {})
        self.training_mode = str(training_mode)
        if self.training_mode not in {"cox", "ipcw_bce"}:
            raise ValueError("training_mode must be 'cox' or 'ipcw_bce'.")
        self.train_batch_sampler = None

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage != "fit":
            return super().setup(stage)

        train_data = torch.load(self.path_train_data)
        val_data = torch.load(self.path_val_data)

        train_data = [
            sub for sub in train_data if sub["subject_id"] in self.train_outcomes
        ]
        val_data = [sub for sub in val_data if sub["subject_id"] in self.val_outcomes]

        train_data = filter_subject_data(train_data, self.population["subject_id"])
        val_data = filter_subject_data(val_data, self.population["subject_id"])

        if not train_data:
            raise ValueError(
                "No training subjects remain after outcome/population filtering."
            )

        if not val_data:
            raise ValueError(
                "No validation subjects remain after outcome/population filtering."
            )

        background_length = int((train_data[0]["segment"] == 0).sum())

        self.train_dataset = SurvivalFinetuneDataset(
            train_data,
            outcomes=self.train_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )
        self.val_dataset = SurvivalFinetuneDataset(
            val_data,
            outcomes=self.val_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )
        self._setup_train_sampling()

    def _setup_train_sampling(self) -> None:
        sampler_type = resolve_survival_batch_sampler_type(
            self.training_mode,
            self.batch_sampling,
        )
        if sampler_type == "none":
            self.train_batch_sampler = None
            self.train_sampler = None
            return
        if sampler_type == "coverage_balanced":
            self.train_sampler = None
            self.train_batch_sampler = build_coverage_balanced_survival_batch_sampler(
                self.train_dataset,
                outcome_name="survival",
                batch_size=self.batch_size,
                seed=int(self.batch_sampling.get("seed", 0)),
            )
            print(self.train_batch_sampler.summary())
            return
        if sampler_type != "event_aware":
            raise ValueError(f"Unknown survival batch sampler: {sampler_type!r}")

        warnings.warn(
            "Event-aware sampling duplicates and outcome-weights patients. This "
            "changes mini-batch Cox risk sets and biases IPCW-BCE unless the loss "
            "is importance-corrected. Prefer type=auto or coverage_balanced for "
            "survival finetuning; event_aware should be treated as an ablation.",
            RuntimeWarning,
            stacklevel=2,
        )

        min_valid = self.batch_sampling.get("min_valid_per_batch")
        batches_per_epoch = self.batch_sampling.get("batches_per_epoch")
        self.train_sampler = None
        self.train_batch_sampler = build_event_aware_batch_sampler(
            self.train_dataset,
            ["survival"],
            batch_size=self.batch_size,
            n_quantiles=int(self.batch_sampling.get("n_quantiles", 4)),
            min_events_per_batch=int(
                self.batch_sampling.get("min_events_per_batch", 4)
            ),
            min_valid_per_batch=None if min_valid is None else int(min_valid),
            min_unique_events_for_focus=int(
                self.batch_sampling.get("min_unique_events_for_focus", 1)
            ),
            min_unique_valid_for_focus=int(
                self.batch_sampling.get("min_unique_valid_for_focus", 4)
            ),
            batches_per_epoch=(
                None if batches_per_epoch is None else int(batches_per_epoch)
            ),
            seed=int(self.batch_sampling.get("seed", 0)),
        )
        print(self.train_batch_sampler.summary())

    def train_dataloader(self):
        if self.train_batch_sampler is not None:
            return DataLoader(
                self.train_dataset,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
                batch_sampler=self.train_batch_sampler,
                collate_fn=survival_finetune_collate,
            )
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            collate_fn=survival_finetune_collate,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            shuffle=False,
            collate_fn=survival_finetune_collate,
        )
