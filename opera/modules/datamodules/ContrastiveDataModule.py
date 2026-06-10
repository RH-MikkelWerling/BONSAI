"""
DataModule for OPERA contrastive learning stage.

Reads multiple outcome parquet files and merges them into a single
multi-outcome dataset.  The collate function is extended to handle
the survival fields (``time_<n>``, ``event_<n>``) as well as the
binary ``outcome_<n>`` keys.
"""

from typing import Literal, Dict, List
import pandas as pd
import lightning as L
import torch
from torch.utils.data import DataLoader
from opera.compat.bonsai import dynamic_padding, filter_subject_data, binarize_outcomes
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_registry_eligible_outcomes,
)
from opera.modules.datasets.ContrastiveDataset import ContrastiveDataset
from opera.functional.stratified_sampling import (
    build_stratified_sampler,
    log_bucket_stats,
)


def contrastive_collate(batch):
    """
    Extends BONSAI's dynamic_padding with survival field collation.
    Handles: outcome_*, time_*, event_* keys.
    """
    base = dynamic_padding(batch)

    for prefix in ("outcome_", "time_", "event_"):
        keys = [k for k in batch[0] if k.startswith(prefix)]
        for k in keys:
            base[k] = torch.stack([sample[k] for sample in batch])

    return base


def compute_sorted_event_times(
    outcome_configs: Dict[str, dict],
    split: str = "train",
) -> Dict[str, torch.Tensor]:
    """
    Read training event times from each outcome parquet and return sorted tensors.

    Legacy helper returning unweighted event-time locations. New OPERA runs
    should prefer ``compute_event_time_probability_grids`` so the contrastive
    loss receives Kaplan-Meier event-time masses.
    """
    result: Dict[str, torch.Tensor] = {}
    for name, ocfg in outcome_configs.items():
        path = ocfg.get("path") or ocfg.get("outcome_file")
        if path is None:
            result[name] = torch.tensor([], dtype=torch.float32)
            continue
        try:
            df = pd.read_parquet(path)
            split_df = df[df["split"] == split].copy()
            if {"event", "time_days"}.issubset(split_df.columns):
                event_times = split_df.loc[split_df["event"] == 1, "time_days"].dropna()
                result[name] = torch.tensor(
                    sorted(event_times.astype(float).tolist()),
                    dtype=torch.float32,
                )
                continue
            df = attach_prediction_censor_abspos(df)
            df = filter_registry_eligible_outcomes(
                df,
                ocfg.get("registry_start_date"),
                outcome_name=name,
            )
            split_df = df[df["split"] == split].copy()
            competing_df = None
            competing_path = ocfg.get("competing_outcome_path")
            if competing_path:
                competing_df = pd.read_parquet(competing_path)
            outcomes = binarize_outcomes(
                split_df,
                n_hours_start_include=ocfg["n_hours_start_include"],
                n_hours_end_include=ocfg.get("n_hours_end_include"),
                competing_event_df=competing_df,
            )
            events = [
                record["time_days"]
                for record in outcomes.values()
                if record.get("event") == 1
            ]
            result[name] = torch.tensor(
                sorted(events),
                dtype=torch.float32,
            )
        except (KeyError, OSError, ValueError):
            result[name] = torch.tensor([], dtype=torch.float32)
    return result


def _km_event_time_probabilities(
    records: List[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate primary-event time mass with Kaplan-Meier censoring adjustment."""
    if not records:
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty
    frame = pd.DataFrame(records).dropna(subset=["time_days", "event"])
    if frame.empty or int((frame["event"] == 1).sum()) == 0:
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty
    frame = frame.sort_values("time_days")
    times = []
    probs = []
    survival = 1.0
    for t, group in frame.groupby("time_days", sort=True):
        n_at_risk = int((frame["time_days"] >= t).sum())
        n_events = int((group["event"] == 1).sum())
        if n_at_risk <= 0:
            continue
        if n_events > 0:
            event_prob = survival * (n_events / n_at_risk)
            times.append(float(t))
            probs.append(float(event_prob))
        survival *= max(0.0, 1.0 - n_events / n_at_risk)
    if not times:
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty
    prob_tensor = torch.tensor(probs, dtype=torch.float32)
    prob_sum = prob_tensor.sum()
    if prob_sum > 0:
        prob_tensor = prob_tensor / prob_sum
    return torch.tensor(times, dtype=torch.float32), prob_tensor


def compute_event_time_probability_grids(
    outcome_configs: Dict[str, dict],
    split: str = "train",
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return KM-adjusted event-time locations and primary-event masses."""
    time_grids: Dict[str, torch.Tensor] = {}
    prob_grids: Dict[str, torch.Tensor] = {}
    for name, ocfg in outcome_configs.items():
        path = ocfg.get("path") or ocfg.get("outcome_file")
        if path is None:
            empty = torch.tensor([], dtype=torch.float32)
            time_grids[name] = empty
            prob_grids[name] = empty
            continue
        try:
            df = pd.read_parquet(path)
            df = attach_prediction_censor_abspos(df)
            df = filter_registry_eligible_outcomes(
                df,
                ocfg.get("registry_start_date"),
                outcome_name=name,
            )
            split_df = df[df["split"] == split].copy()
            competing_df = None
            competing_path = ocfg.get("competing_outcome_path")
            if competing_path:
                competing_df = pd.read_parquet(competing_path)
            outcomes = binarize_outcomes(
                split_df,
                n_hours_start_include=ocfg["n_hours_start_include"],
                n_hours_end_include=ocfg.get("n_hours_end_include"),
                competing_event_df=competing_df,
            )
            times, probs = _km_event_time_probabilities(list(outcomes.values()))
            time_grids[name] = times
            prob_grids[name] = probs
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to construct event-time grid for outcome {name!r} "
                f"from {path!r}: {exc}"
            ) from exc
    return time_grids, prob_grids


class ContrastiveDataModule(L.LightningDataModule):
    """
    Parameters
    ----------
    path_train_data, path_val_data : str
        Paths to BONSAI subject_data .pt files.
    path_population : str
        Path to population CSV.
    outcome_configs : dict
        Mapping from outcome name -> dict with keys:
            - path                  : path to parquet outcome file
            - n_hours_start_include : int
            - n_hours_end_include   : int | null

        Outcome parquets contain raw dates. Labels, event times, and censoring
        status are derived at runtime with ``binarize_outcomes``.
    predict_token_id : int
    batch_size, num_workers : int
    """

    def __init__(
        self,
        path_train_data: str,
        path_val_data: str,
        path_population: str,
        outcome_configs: Dict[str, dict],
        predict_token_id: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_val_data = path_val_data
        self.population = pd.read_csv(path_population)
        self.outcome_configs = outcome_configs
        self.predict_token_id = predict_token_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.outcome_names = sorted(outcome_configs.keys())

    def _load_outcomes(self, split_key: str) -> Dict[str, Dict[int, dict]]:
        """Load and binarize all outcomes for a given split."""
        outcome_dicts = {}
        for name, ocfg in self.outcome_configs.items():
            df = pd.read_parquet(ocfg["path"])
            df = attach_prediction_censor_abspos(df)
            df = filter_registry_eligible_outcomes(
                df,
                ocfg.get("registry_start_date"),
                outcome_name=name,
            )
            split_df = df[df["split"] == split_key].copy()

            if len(split_df) == 0:
                outcome_dicts[name] = {}
                continue

            # Optional competing-event (death) table — annotates non-primary-event
            # patients who died as event=2 rather than event=0 (admin censored).
            competing_df = None
            competing_path = ocfg.get("competing_outcome_path")
            if competing_path:
                competing_df = pd.read_parquet(competing_path)

            outcome_dicts[name] = binarize_outcomes(
                split_df,
                n_hours_start_include=ocfg["n_hours_start_include"],
                n_hours_end_include=ocfg.get("n_hours_end_include"),
                competing_event_df=competing_df,
            )
        return outcome_dicts

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage != "fit":
            raise NotImplementedError(f"Stage {stage} not supported.")

        train_data = torch.load(self.path_train_data)
        val_data = torch.load(self.path_val_data)

        train_data = filter_subject_data(train_data, self.population["subject_id"])
        val_data = filter_subject_data(val_data, self.population["subject_id"])

        train_outcomes = self._load_outcomes("train")
        val_outcomes = self._load_outcomes("tuning")

        # Keep only subjects that appear in at least one outcome
        all_train_sids = set()
        for od in train_outcomes.values():
            all_train_sids.update(od.keys())
        train_data = [s for s in train_data if s["subject_id"] in all_train_sids]

        all_val_sids = set()
        for od in val_outcomes.values():
            all_val_sids.update(od.keys())
        val_data = [s for s in val_data if s["subject_id"] in all_val_sids]

        background_length = (train_data[0]["segment"] == 0).sum() if train_data else 0

        self.train_dataset = ContrastiveDataset(
            train_data,
            outcome_dicts=train_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
        )
        self.val_dataset = ContrastiveDataset(
            val_data,
            outcome_dicts=val_outcomes,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
        )

        print(log_bucket_stats(self.train_dataset, self.outcome_names))
        self.train_sampler = build_stratified_sampler(
            self.train_dataset, self.outcome_names
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            sampler=self.train_sampler,
            collate_fn=contrastive_collate,
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
            collate_fn=contrastive_collate,
        )
