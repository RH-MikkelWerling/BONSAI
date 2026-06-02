"""Survival-aware finetuning datamodule for OPERA."""

from typing import Literal

import torch
from torch.utils.data import DataLoader

from opera.compat.bonsai import dynamic_padding, filter_subject_data
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from opera.modules.datasets.SurvivalFinetuneDataset import SurvivalFinetuneDataset


def survival_finetune_collate(batch: list[dict]) -> dict:
    """Collate standard BONSAI fields plus fixed-size survival tensors."""
    output = dynamic_padding(batch)
    for key in ("time_days", "event", "ipcw_weight"):
        output[key] = torch.stack([sample[key] for sample in batch])
    return output


class SurvivalFinetuneDataModule(FinetuneDataModule):
    """Use ``SurvivalFinetuneDataset`` while preserving BONSAI loaders."""

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
            raise ValueError("No training subjects remain after outcome/population filtering.")

        background_length = (train_data[0]["segment"] == 0).sum()

        self.train_dataset = SurvivalFinetuneDataset(
            train_data,
            outcomes=self.train_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
        )
        self.val_dataset = SurvivalFinetuneDataset(
            val_data,
            outcomes=self.val_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
        )

    def train_dataloader(self):
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
