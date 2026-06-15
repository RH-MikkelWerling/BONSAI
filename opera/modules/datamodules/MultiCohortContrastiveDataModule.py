"""
Multi-cohort DataModule for OPERA contrastive learning.

Combines subjects from multiple disease cohorts (DLBCL, CLL, MDS, etc.)
into a single contrastive training pool.  Outcome definitions are shared
across cohorts where they exist (e.g. mortality is defined for all),
and cohort-specific where they don't (e.g. treatment_failure may only
be defined for DLBCL).

This is the correct DataModule to use when you want cross-disease
contrastive learning — which is the point of OPERA.  The DAPT-prior
pair weighting (via frozen DAPT embeddings) handles the soft
disease-similarity modulation: patients from different cohorts can
still be contrasted, but their pair weight is proportional to how
similar their full clinical histories are.

Data layout expected on disk
─────────────────────────────
BONSAI_PROCESSED_DATA/
  dlbcl/
    subject_data_train.pt
    subject_data_tuning.pt
    outcomes/
      mortality.parquet          must have: subject_id, split, index_date,
      treatment_failure.parquet     outcome_date, censor_date
  cll/
    subject_data_train.pt
    ...
  mds/
    ...

Config
──────
See opera/configs/contrastive_multicohort.yaml for the expected structure.
"""

from typing import Dict, Literal, Optional
import os
import pandas as pd
import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset

from opera.compat.bonsai import filter_subject_data, binarize_outcomes
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
    resolve_registry_start_date,
)

from opera.modules.datasets.ContrastiveDataset import ContrastiveDataset
from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
from opera.modules.datamodules.ContrastiveDataModule import _km_event_time_probabilities
from opera.functional.stratified_sampling import (
    build_stratified_sampler,
    log_bucket_stats,
)


def _eligibility_path(data_dir: str, outcome_config: dict):
    raw = outcome_config.get("eligibility_path") or outcome_config.get(
        "eligibility_file"
    )
    if raw in (None, "", "null"):
        return None
    path = os.path.expandvars(str(raw))
    return path if os.path.isabs(path) else os.path.join(data_dir, "outcomes", path)


def compute_pooled_sorted_event_times(
    cohort_configs: Dict[str, dict],
    outcome_configs: Dict[str, dict],
    split: str = "train",
) -> Dict[str, torch.Tensor]:
    """
    Aggregate observed event times from all cohorts for each outcome.

    Legacy helper returning unweighted pooled event-time locations. New OPERA
    runs should prefer ``compute_pooled_event_time_probability_grids`` so the
    contrastive loss receives Kaplan-Meier event-time masses.
    """
    pooled: Dict[str, list] = {name: [] for name in outcome_configs}

    for cohort_name, cohort_cfg in cohort_configs.items():
        data_dir = cohort_cfg["data_dir"]
        for name, ocfg in outcome_configs.items():
            filename = ocfg.get("outcome_file") or ocfg.get("filename")
            if filename is None:
                continue
            path = os.path.join(data_dir, "outcomes", filename)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                df = filter_outcome_eligibility(
                    df,
                    _eligibility_path(data_dir, ocfg),
                    cohort=cohort_name,
                    outcome_name=name,
                )
                split_df = df[df["split"] == split].copy()
                if {"event", "time_days"}.issubset(split_df.columns):
                    pooled[name].extend(
                        split_df.loc[split_df["event"] == 1, "time_days"]
                        .dropna()
                        .astype(float)
                        .tolist()
                    )
                    continue
                df = attach_prediction_censor_abspos(df)
                df = filter_registry_eligible_outcomes(
                    df,
                    resolve_registry_start_date(cohort_cfg, ocfg),
                    cohort=cohort_cfg.get("name"),
                    outcome_name=name,
                )
                split_df = df[df["split"] == split].copy()
                competing_df = None
                competing_file = ocfg.get("competing_outcome_file")
                if competing_file:
                    competing_path = os.path.join(data_dir, "outcomes", competing_file)
                    if os.path.exists(competing_path):
                        competing_df = pd.read_parquet(competing_path)
                outcomes = binarize_outcomes(
                    split_df,
                    n_hours_start_include=ocfg["n_hours_start_include"],
                    n_hours_end_include=ocfg.get("n_hours_end_include"),
                    competing_event_df=competing_df,
                )
                pooled[name].extend(
                    record["time_days"]
                    for record in outcomes.values()
                    if record.get("event") == 1
                )
            except (KeyError, OSError, ValueError):
                continue

    return {
        name: torch.tensor(sorted(times), dtype=torch.float32)
        for name, times in pooled.items()
    }


def compute_pooled_event_time_probability_grids(
    cohort_configs: Dict[str, dict],
    outcome_configs: Dict[str, dict],
    split: str = "train",
    require_all_configured_cells: bool = False,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Aggregate KM-adjusted event-time locations and primary-event masses."""
    pooled: Dict[str, list] = {name: [] for name in outcome_configs}

    for cohort_name, cohort_cfg in cohort_configs.items():
        data_dir = cohort_cfg["data_dir"]
        for name, ocfg in outcome_configs.items():
            filename = ocfg.get("outcome_file") or ocfg.get("filename")
            if filename is None:
                if require_all_configured_cells:
                    raise ValueError(
                        f"Configured outcome {name!r} has no outcome_file for "
                        f"cohort {cohort_name!r}."
                    )
                continue
            path = os.path.join(data_dir, "outcomes", filename)
            if not os.path.exists(path):
                if require_all_configured_cells:
                    raise FileNotFoundError(
                        f"Configured outcome {name!r} is missing for cohort "
                        f"{cohort_name!r}: {path}"
                    )
                continue
            try:
                df = pd.read_parquet(path)
                df = filter_outcome_eligibility(
                    df,
                    _eligibility_path(data_dir, ocfg),
                    cohort=cohort_name,
                    outcome_name=name,
                )
                df = attach_prediction_censor_abspos(df)
                df = filter_registry_eligible_outcomes(
                    df,
                    resolve_registry_start_date(cohort_cfg, ocfg),
                    cohort=cohort_cfg.get("name"),
                    outcome_name=name,
                )
                split_df = df[df["split"] == split].copy()
                competing_df = None
                competing_file = ocfg.get("competing_outcome_file")
                if competing_file:
                    competing_path = os.path.join(data_dir, "outcomes", competing_file)
                    if os.path.exists(competing_path):
                        competing_df = pd.read_parquet(competing_path)
                    elif require_all_configured_cells:
                        raise FileNotFoundError(
                            f"Configured competing outcome for {name!r} is missing "
                            f"for cohort {cohort_name!r}: {competing_path}"
                        )
                outcomes = binarize_outcomes(
                    split_df,
                    n_hours_start_include=ocfg["n_hours_start_include"],
                    n_hours_end_include=ocfg.get("n_hours_end_include"),
                    competing_event_df=competing_df,
                )
                pooled[name].extend(outcomes.values())
            except (KeyError, OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to construct pooled event-time grid for "
                    f"outcome {name!r} in cohort {cohort_name!r}: {exc}"
                ) from exc

    time_grids: Dict[str, torch.Tensor] = {}
    prob_grids: Dict[str, torch.Tensor] = {}
    for name, records in pooled.items():
        times, probs = _km_event_time_probabilities(list(records))
        time_grids[name] = times
        prob_grids[name] = probs
    return time_grids, prob_grids


class MultiCohortContrastiveDataModule(L.LightningDataModule):
    """
    Parameters
    ----------
    cohort_configs : dict
        Mapping from cohort_name -> dict with keys:
            - data_dir : str        path to cohort's processed data folder
            - population_file : str  path to population CSV (optional,
                                    defaults to data_dir/population_full.csv)

    outcome_configs : dict
        Mapping from outcome_name -> dict with keys:
            - outcome_file : str parquet filename within each cohort's
                                 outcomes/ subdirectory (e.g. "mortality.parquet")
                                 Legacy configs may use "filename".
            - n_hours_start_include : int
            - n_hours_end_include   : int | null

        An outcome that does not exist for a given cohort simply contributes
        no pairs for that cohort's patients on that outcome dimension.

    predict_token_id : int
    batch_size, num_workers : int
    """

    def __init__(
        self,
        cohort_configs: Dict[str, dict],
        outcome_configs: Dict[str, dict],
        predict_token_id: int,
        batch_size: int,
        num_workers: int,
        require_min_followup_train: bool = False,
        require_min_followup_val: bool = False,
        require_all_configured_cells: bool = False,
        max_len: int = 8192,
    ):
        super().__init__()
        self.cohort_configs = cohort_configs
        self.outcome_configs = outcome_configs
        self.predict_token_id = predict_token_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.require_min_followup_train = require_min_followup_train
        self.require_min_followup_val = require_min_followup_val
        self.require_all_configured_cells = require_all_configured_cells
        self.max_len = max_len
        self.outcome_names = sorted(outcome_configs.keys())

    # ── Internal helpers ──────────────────────────────────────────────

    def _load_outcomes_for_cohort(
        self,
        cohort_name: str,
        cohort_cfg: dict,
        data_dir: str,
        split_key: str,
    ) -> Dict[str, Dict[int, dict]]:
        """
        Load all outcome parquets for a single cohort + split.
        Returns an outcome_dict with survival fields where available.
        """
        outcomes_dir = os.path.join(data_dir, "outcomes")
        outcome_dicts: Dict[str, Dict[int, dict]] = {}

        for name, ocfg in self.outcome_configs.items():
            filename = ocfg.get("outcome_file", ocfg.get("filename", f"{name}.parquet"))
            path = os.path.join(outcomes_dir, filename)
            if not os.path.exists(path):
                if self.require_all_configured_cells:
                    raise FileNotFoundError(
                        f"Configured outcome {name!r} is missing for cohort "
                        f"{cohort_name!r}: {path}"
                    )
                # Outcome not available for this cohort — skip silently
                outcome_dicts[name] = {}
                continue

            df = pd.read_parquet(path)
            df = filter_outcome_eligibility(
                df,
                _eligibility_path(data_dir, ocfg),
                cohort=cohort_name,
                outcome_name=name,
            )
            df = attach_prediction_censor_abspos(df)
            df = filter_registry_eligible_outcomes(
                df,
                resolve_registry_start_date(cohort_cfg, ocfg),
                cohort=cohort_name,
                outcome_name=name,
            )
            split_df = df[df["split"] == split_key].copy()

            if len(split_df) == 0:
                if self.require_all_configured_cells:
                    raise ValueError(
                        f"Configured outcome {name!r} has no eligible rows for "
                        f"cohort {cohort_name!r}, split {split_key!r}."
                    )
                outcome_dicts[name] = {}
                continue

            # Optional competing-event (death) table — annotates non-primary-event
            # patients who died as event=2 rather than event=0 (admin censored).
            competing_df = None
            competing_file = ocfg.get("competing_outcome_file")
            if competing_file:
                competing_path = os.path.join(outcomes_dir, competing_file)
                if os.path.exists(competing_path):
                    competing_df = pd.read_parquet(competing_path)
                elif self.require_all_configured_cells:
                    raise FileNotFoundError(
                        f"Configured competing outcome for {name!r} is missing "
                        f"for cohort {cohort_name!r}: {competing_path}"
                    )

            outcome_dicts[name] = binarize_outcomes(
                split_df,
                n_hours_start_include=ocfg["n_hours_start_include"],
                n_hours_end_include=ocfg.get("n_hours_end_include"),
                require_min_followup=(
                    self.require_min_followup_train
                    if split_key == "train"
                    else self.require_min_followup_val
                ),
                competing_event_df=competing_df,
            )

        return outcome_dicts

    def _build_dataset_for_cohort(
        self,
        cohort_name: str,
        cohort_cfg: dict,
        split: Literal["train", "tuning"],
    ) -> Optional[ContrastiveDataset]:
        """Load subjects + outcomes for one cohort, return dataset or None."""
        data_dir = cohort_cfg["data_dir"]
        split_file = os.path.join(
            data_dir,
            "subject_data_train.pt" if split == "train" else "subject_data_tuning.pt",
        )
        if not os.path.exists(split_file):
            if self.require_all_configured_cells:
                raise FileNotFoundError(
                    f"Configured cohort {cohort_name!r} is missing split data: "
                    f"{split_file}"
                )
            print(f"  [{cohort_name}] Missing {split_file}, skipping.")
            return None

        pop_file = cohort_cfg.get(
            "population_file",
            os.path.join(data_dir, "population_full.csv"),
        )
        population = pd.read_csv(pop_file)

        subjects = torch.load(split_file)
        subjects = filter_subject_data(subjects, population["subject_id"])

        split_key = "train" if split == "train" else "tuning"
        outcome_dicts = self._load_outcomes_for_cohort(
            cohort_name,
            cohort_cfg,
            data_dir,
            split_key,
        )

        # Keep only subjects with at least one outcome
        valid_sids: set = set()
        for od in outcome_dicts.values():
            valid_sids.update(od.keys())

        subjects = [s for s in subjects if s["subject_id"] in valid_sids]
        if not subjects:
            print(
                f"  [{cohort_name}] No subjects with outcomes for split={split}, skipping."
            )
            return None

        background_length = int((subjects[0]["segment"] == 0).sum())

        n_outcomes_present = sum(1 for od in outcome_dicts.values() if len(od) > 0)
        print(
            f"  [{cohort_name}] split={split}: "
            f"{len(subjects)} subjects, "
            f"{n_outcomes_present}/{len(self.outcome_names)} outcomes present"
        )

        return ContrastiveDataset(
            subjects,
            outcome_dicts=outcome_dicts,
            predict_token_id=self.predict_token_id,
            background_length=background_length,
            max_len=self.max_len,
        )

    # ── Lightning interface ───────────────────────────────────────────

    def setup(self, stage: Literal["fit", "test", "predict"]):
        if stage != "fit":
            raise NotImplementedError(f"Stage {stage} not supported.")

        print("Loading multi-cohort contrastive datasets...")
        train_datasets, val_datasets = [], []

        for cohort_name, cohort_cfg in self.cohort_configs.items():
            train_ds = self._build_dataset_for_cohort(cohort_name, cohort_cfg, "train")
            val_ds = self._build_dataset_for_cohort(cohort_name, cohort_cfg, "tuning")
            if train_ds:
                train_datasets.append(train_ds)
            if val_ds:
                val_datasets.append(val_ds)

        if not train_datasets:
            raise RuntimeError("No training data loaded across any cohort.")

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets) if val_datasets else None

        total_train = sum(len(d) for d in train_datasets)
        total_val = sum(len(d) for d in val_datasets) if val_datasets else 0
        print(
            f"Multi-cohort contrastive dataset ready: "
            f"{total_train} train / {total_val} val subjects "
            f"across {len(train_datasets)} cohorts"
        )

        # Build stratified sampler for training
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
        if self.val_dataset is None:
            return None
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
