"""
DataModule for the hybrid experiment (EHR embeddings + RKKP tabular features).
"""

from pathlib import Path
from typing import Literal, Dict, List, Optional
import pandas as pd
import lightning as L
import torch
from torch.utils.data import DataLoader
from opera.compat.bonsai import dynamic_padding, filter_subject_data
from opera.modules.datasets.HybridDataset import HybridDataset


def hybrid_collate(batch):
    """Extends BONSAI's dynamic_padding with tabular feature stacking."""
    base = dynamic_padding(batch)
    base["tabular"] = torch.stack([s["tabular"] for s in batch])
    return base


class HybridDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_data: str,
        path_val_data: str,
        path_population: str,
        path_tabular: str,
        feature_columns: List[str],
        train_outcomes: Dict[int, dict],
        val_outcomes: Dict[int, dict],
        test_outcomes: Dict[int, dict],
        predict_token_id: int,
        max_len: int,
        batch_size: int,
        num_workers: int,
        train_sampler=None,
        subject_data_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.subject_data_paths = subject_data_paths
        self.population = pd.read_csv(path_population)
        self.tabular_df = pd.read_parquet(path_tabular)
        self.feature_columns = feature_columns
        self.train_outcomes = train_outcomes
        self.val_outcomes = val_outcomes
        self.test_outcomes = test_outcomes
        self.predict_token_id = predict_token_id
        self.max_len = int(max_len)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sampler = train_sampler

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage != "fit":
            raise NotImplementedError

        from opera.modules.datamodules.OutcomeFinetuneDataModule import (
            load_subject_pool,
            resolve_subject_data_paths,
        )

        paths = resolve_subject_data_paths(
            Path(self.path_train_data).parent, self.subject_data_paths
        )
        subject_pool = load_subject_pool(paths)
        train_data = subject_pool
        val_data = subject_pool

        train_data = [s for s in train_data if s["subject_id"] in self.train_outcomes]
        val_data = [s for s in val_data if s["subject_id"] in self.val_outcomes]

        train_data = filter_subject_data(train_data, self.population["subject_id"])
        val_data = filter_subject_data(val_data, self.population["subject_id"])
        if not train_data:
            raise ValueError(
                "No hybrid training subjects remain after outcome/population filtering."
            )
        if not val_data:
            raise ValueError(
                "No hybrid validation subjects remain after outcome/population filtering."
            )

        bg_len = int((train_data[0]["segment"] == 0).sum())

        self.train_dataset = HybridDataset(
            train_data,
            self.train_outcomes,
            self.tabular_df,
            self.feature_columns,
            self.predict_token_id,
            bg_len,
            max_len=self.max_len,
        )
        self.val_dataset = HybridDataset(
            val_data,
            self.val_outcomes,
            self.tabular_df,
            self.feature_columns,
            self.predict_token_id,
            bg_len,
            max_len=self.max_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            collate_fn=hybrid_collate,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            shuffle=False,
            collate_fn=hybrid_collate,
        )
