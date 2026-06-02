"""Survival-aware finetuning dataset for OPERA."""

import torch

from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset


class SurvivalFinetuneDataset(FinetuneDataset):
    """BONSAI finetuning dataset augmented with survival and IPCW fields."""

    def __getitem__(self, index: int) -> dict:
        subject = super().__getitem__(index)
        subject_outcome = self.outcomes[subject["subject_id"]]
        subject["time_days"] = torch.tensor(
            [subject_outcome["time_days"]],
            dtype=torch.float32,
        )
        subject["event"] = torch.tensor(
            [subject_outcome["event"]],
            dtype=torch.long,
        )
        subject["ipcw_weight"] = torch.tensor(
            [subject_outcome.get("ipcw_weight", 1.0)],
            dtype=torch.float32,
        )
        return subject
