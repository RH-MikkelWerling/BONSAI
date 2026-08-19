import polars as pl
from typing import Optional
import torch
from torch.utils.data import DataLoader
import lightning as L
from bonsai.functional.collate import dynamic_padding
from bonsai.functional.subject_data import filter_subject_data
from bonsai.modules.datasets.PretrainDataset import (
    MLMPretrainDataset,
    ARPretrainDataset,
)


class PretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_data: str,
        path_val_data: str,
        path_vocab: str,
        path_population: str,
        batch_size: int,
        num_workers: int,
        max_len: int,
        dataset_class: torch.utils.data.Dataset,
        masking_config: Optional[dict] = None,
        cutoff_date: Optional[dict] = None,
        train_truncation_strategy: str = "tail",
        val_truncation_strategy: str = "tail",
        tail_window_probability: float = 0.5,
        value_embedding_mode: str = "legacy",
        numeric_value_control: str = "observed",
        ignore_target_tokens: Optional[list[str]] = None,
        ignore_same_time_targets: bool = False,
        abspos_subject_jitter_years: float = 0.0,
    ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.vocabulary = torch.load(path_vocab)
        self.population = pl.read_csv(path_population)
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.max_len = max_len
        self.cutoff_date = cutoff_date

        self.dataset_class = dataset_class
        self.masking_config = masking_config
        self.train_truncation_strategy = train_truncation_strategy
        self.val_truncation_strategy = val_truncation_strategy
        self.tail_window_probability = tail_window_probability
        self.value_embedding_mode = value_embedding_mode
        self.numeric_value_control = numeric_value_control
        self.ignore_target_tokens = list(ignore_target_tokens or [])
        self.ignore_same_time_targets = bool(ignore_same_time_targets)
        self.abspos_subject_jitter_years = float(abspos_subject_jitter_years)

    def setup(self, stage: str):
        if stage == "fit":
            self.setup_fit()
        elif stage == "test":
            raise NotImplementedError("Test stage not supported for PretrainModule.")
        elif stage == "predict":
            raise NotImplementedError("Predict stage not supported for PretrainModule.")

    def setup_fit(self):
        train_data = torch.load(self.path_train_data)
        val_data = torch.load(self.path_val_data)

        population_subject_ids = self.population["subject_id"].to_list()
        train_data = filter_subject_data(train_data, population_subject_ids)
        val_data = filter_subject_data(val_data, population_subject_ids)
        if not train_data:
            raise ValueError("No training subjects remain after population filtering.")
        if not val_data:
            raise ValueError(
                "No validation subjects remain after population filtering."
            )

        background_length = int((train_data[0]["segment"] == 0).sum())

        if issubclass(self.dataset_class, MLMPretrainDataset):
            assert self.masking_config is not None
            self.train_dataset = self.dataset_class(
                train_data,
                max_len=self.max_len,
                cutoff_date=self.cutoff_date,
                background_length=background_length,
                vocabulary=self.vocabulary,
                masking_select_ratio=self.masking_config.masking_select_ratio,
                masking_mask_ratio=self.masking_config.masking_mask_ratio,
                masking_random_ratio=self.masking_config.masking_random_ratio,
                masking_ignore_special_tokens=self.masking_config.masking_ignore_special_tokens,
                truncation_strategy=self.train_truncation_strategy,
                tail_window_probability=self.tail_window_probability,
                numeric_value_control=self.numeric_value_control,
                abspos_subject_jitter_years=self.abspos_subject_jitter_years,
            )
            self.val_dataset = self.dataset_class(
                val_data,
                max_len=self.max_len,
                cutoff_date=self.cutoff_date,
                background_length=background_length,
                vocabulary=self.vocabulary,
                masking_select_ratio=self.masking_config.masking_select_ratio,
                masking_mask_ratio=self.masking_config.masking_mask_ratio,
                masking_random_ratio=self.masking_config.masking_random_ratio,
                masking_ignore_special_tokens=self.masking_config.masking_ignore_special_tokens,
                truncation_strategy=self.val_truncation_strategy,
                tail_window_probability=1.0,
                numeric_value_control=self.numeric_value_control,
                abspos_subject_jitter_years=0.0,
            )
        elif issubclass(self.dataset_class, ARPretrainDataset):
            self.train_dataset = self.dataset_class(
                train_data,
                self.max_len,
                background_length=background_length,
                cutoff_date=self.cutoff_date,
                truncation_strategy=self.train_truncation_strategy,
                tail_window_probability=self.tail_window_probability,
                vocabulary=self.vocabulary,
                value_embedding_mode=self.value_embedding_mode,
                numeric_value_control=self.numeric_value_control,
                ignore_target_tokens=self.ignore_target_tokens,
                ignore_same_time_targets=self.ignore_same_time_targets,
                abspos_subject_jitter_years=self.abspos_subject_jitter_years,
            )
            self.val_dataset = self.dataset_class(
                val_data,
                self.max_len,
                background_length=background_length,
                cutoff_date=self.cutoff_date,
                truncation_strategy=self.val_truncation_strategy,
                tail_window_probability=1.0,
                vocabulary=self.vocabulary,
                value_embedding_mode=self.value_embedding_mode,
                numeric_value_control=self.numeric_value_control,
                ignore_target_tokens=self.ignore_target_tokens,
                ignore_same_time_targets=self.ignore_same_time_targets,
                abspos_subject_jitter_years=0.0,
            )
        else:
            raise ValueError(f"Unexpected dataset class. Got: {self.dataset_class}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            collate_fn=dynamic_padding,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            shuffle=False,
            collate_fn=dynamic_padding,
        )
