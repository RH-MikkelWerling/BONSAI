"""Outcome-defined finetuning views over physically sharded BONSAI data."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import torch

from bonsai.functional.subject_data import filter_subject_data
from bonsai.modules.datamodules.FinetuneDataModule import FinetuneDataModule
from bonsai.modules.datasets.FinetuneDataset import FinetuneDataset


def _unique_paths(paths: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(path) for path in paths if path))


def resolve_subject_data_paths(data_dir: str, configured=None) -> list[str]:
    """Return configured files or all conventional physical SSL partitions."""
    if configured:
        return _unique_paths(configured)
    root = Path(data_dir)
    return [
        str(path)
        for path in (
            root / "subject_data_train.pt",
            root / "subject_data_tuning.pt",
        )
        if path.exists()
    ]


def load_subject_pool(paths: Iterable[str]) -> list[dict]:
    """Load disjoint physical partitions into one outcome-addressable pool."""
    subjects = []
    seen = set()
    for path in _unique_paths(paths):
        for subject in torch.load(path, weights_only=False):
            subject_id = subject["subject_id"]
            if subject_id in seen:
                raise ValueError(
                    f"Subject {subject_id!r} occurs in multiple physical "
                    "subject-data files."
                )
            seen.add(subject_id)
            subjects.append(subject)
    return subjects


class OutcomeFinetuneDataModule(FinetuneDataModule):
    """Select temporal outcome splits after pooling physical subject shards.

    ehr2meds' train/tuning folders are the random self-supervised split used
    for preprocessing and pretraining. They are not the prospective outcome
    split. This datamodule pools those physical partitions before selecting
    subjects from the outcome manifest.
    """

    def __init__(self, *args, subject_data_paths=None, **kwargs):
        super().__init__(*args, **kwargs)
        fallback = [
            self.path_train_data,
            self.path_val_data,
            self.path_predict_data,
        ]
        self.subject_data_paths = _unique_paths(subject_data_paths or fallback)
        if not self.subject_data_paths:
            raise ValueError("At least one physical subject-data path is required.")
        self._subject_pool = None

    def _load_subject_pool(self) -> list[dict]:
        if self._subject_pool is None:
            self._subject_pool = load_subject_pool(self.subject_data_paths)
        return self._subject_pool

    def _select(self, outcomes: dict, split_name: str) -> list[dict]:
        pool = self._load_subject_pool()
        available = {subject["subject_id"] for subject in pool}
        missing = set(outcomes) - available
        if missing:
            examples = sorted(missing)[:10]
            raise ValueError(
                f"{len(missing)} {split_name} outcome subjects are absent from "
                f"the pooled subject-data files; examples: {examples}."
            )
        selected = [subject for subject in pool if subject["subject_id"] in outcomes]
        return filter_subject_data(selected, self.population["subject_id"])

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage == "fit":
            train_data = self._select(self.train_outcomes, "train")
            val_data = self._select(self.val_outcomes, "tuning")
            if not train_data or not val_data:
                raise ValueError(
                    "Outcome-defined train and tuning data must be non-empty."
                )
            background_length = int((train_data[0]["segment"] == 0).sum())
            common = {
                "predict_token_id": self.predict_token_id,
                "background_length": background_length,
                "max_len": self.max_len,
                "numeric_value_control": self.numeric_value_control,
            }
            self.train_dataset = FinetuneDataset(
                train_data, outcomes=self.train_outcomes, **common
            )
            self.val_dataset = FinetuneDataset(
                val_data, outcomes=self.val_outcomes, **common
            )
            return
        if stage == "predict":
            predict_data = self._select(self.predict_outcomes, "held_out")
            if not predict_data:
                raise ValueError("Outcome-defined held-out data must be non-empty.")
            background_length = int((predict_data[0]["segment"] == 0).sum())
            self.predict_dataset = FinetuneDataset(
                predict_data,
                outcomes=self.predict_outcomes,
                predict_token_id=self.predict_token_id,
                background_length=background_length,
                max_len=self.max_len,
                numeric_value_control=self.numeric_value_control,
            )
            return
        raise NotImplementedError("Test stage is not supported for finetuning.")
