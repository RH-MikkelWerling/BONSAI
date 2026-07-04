"""Canonical evaluation cohorts shared by OPERA and external baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from opera.compat.bonsai import binarize_outcomes
from opera.functional.outcomes import (
    attach_prediction_censor_abspos,
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
)

FIXED_HORIZON_REGIME = "fixed_horizon"
SURVIVAL_REGIME = "survival"
EVALUATION_REGIMES = frozenset({FIXED_HORIZON_REGIME, SURVIVAL_REGIME})


@dataclass(frozen=True)
class EvaluationCohort:
    """One immutable patient cohort and its outcome records."""

    regime: str
    records: Mapping[Any, Mapping[str, Any]]

    @property
    def subject_ids(self) -> frozenset[Any]:
        return frozenset(self.records)

    @property
    def n_events(self) -> int:
        if self.regime == FIXED_HORIZON_REGIME:
            return int(sum(int(record["label"] == 1) for record in self.records.values()))
        return int(sum(int(record.get("event", 0) == 1) for record in self.records.values()))

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame.from_dict(self.records, orient="index")
        frame.index.name = "subject_id"
        return frame.reset_index()


@dataclass(frozen=True)
class EvaluationCohorts:
    """The two prespecified evaluation regimes for one outcome split."""

    fixed_horizon: EvaluationCohort
    survival: EvaluationCohort

    def for_regime(self, regime: str) -> EvaluationCohort:
        if regime == FIXED_HORIZON_REGIME:
            return self.fixed_horizon
        if regime == SURVIVAL_REGIME:
            return self.survival
        raise ValueError(
            f"Unknown evaluation regime {regime!r}; expected {sorted(EVALUATION_REGIMES)}."
        )


def read_subject_ids(path: Optional[str | Path]) -> Optional[set[Any]]:
    """Read a subject-id restriction from CSV or parquet."""
    if path in (None, "", "null"):
        return None
    source = Path(path)
    frame = (
        pd.read_parquet(source)
        if source.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(source)
    )
    if "subject_id" not in frame.columns:
        raise ValueError(f"Subject restriction {source} must contain subject_id.")
    if frame["subject_id"].duplicated().any():
        raise ValueError(f"Subject restriction {source} contains duplicate subject_id rows.")
    return set(frame["subject_id"])


def population_subject_ids(
    population_path: Optional[str | Path],
    cohort_fine_col: Optional[str] = None,
    cohort_fine_value: Optional[str] = None,
) -> Optional[set[Any]]:
    """Return population IDs, optionally restricted to a named fine cohort."""
    if population_path in (None, "", "null"):
        if cohort_fine_col or cohort_fine_value:
            raise ValueError("Fine-cohort filtering requires a population file.")
        return None
    source = Path(population_path)
    population = (
        pd.read_parquet(source)
        if source.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(source)
    )
    if "subject_id" not in population.columns:
        raise ValueError(f"Population file {source} must contain subject_id.")
    if bool(cohort_fine_col) != bool(cohort_fine_value):
        raise ValueError("cohort_fine_col and cohort_fine_value must be provided together.")
    if cohort_fine_col:
        if cohort_fine_col not in population.columns:
            raise ValueError(
                f"cohort_fine_col={cohort_fine_col!r} not found in {source}; "
                f"columns={list(population.columns)}."
            )
        population = population[
            population[cohort_fine_col].astype(str) == str(cohort_fine_value)
        ].copy()
        if population.empty:
            raise ValueError(
                f"No patients remain after filtering {cohort_fine_col}="
                f"{cohort_fine_value!r} in {source}."
            )
    return set(population["subject_id"])


def population_subject_strata(
    population_path: str | Path,
    subject_ids: Iterable[Any],
    strata_col: str,
) -> np.ndarray:
    """Return population strata aligned to an ordered subject-id sequence."""
    source = Path(population_path)
    population = (
        pd.read_parquet(source)
        if source.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(source)
    )
    required = {"subject_id", strata_col}
    missing_columns = required - set(population.columns)
    if missing_columns:
        raise ValueError(
            f"Population file {source} is missing stratification columns "
            f"{sorted(missing_columns)}."
        )
    if population["subject_id"].duplicated().any():
        raise ValueError(
            f"Population file {source} contains duplicate subject_id rows."
        )

    ordered_ids = list(subject_ids)
    strata = population.set_index("subject_id")[strata_col].reindex(ordered_ids)
    missing_subjects = [
        ordered_ids[idx] for idx, missing in enumerate(strata.isna()) if missing
    ]
    if missing_subjects:
        raise ValueError(
            f"Cannot compute stratified concordance from {strata_col!r}: "
            f"{len(missing_subjects)} evaluated subjects have no stratum label; "
            f"examples={missing_subjects[:10]}."
        )
    return strata.to_numpy(dtype=object)


def build_evaluation_cohorts(
    outcomes: pd.DataFrame | str | Path,
    *,
    split: str,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int],
    competing_outcomes: Optional[pd.DataFrame | str | Path] = None,
    eligibility: Optional[pd.DataFrame | str | Path] = None,
    registry_start_date: Optional[Any] = None,
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
    allowed_subject_ids: Optional[Iterable[Any]] = None,
) -> EvaluationCohorts:
    """Build fixed-horizon and survival cohorts from one filtered outcome frame."""
    source_frame = (
        outcomes.copy()
        if isinstance(outcomes, pd.DataFrame)
        else pd.read_parquet(outcomes)
    )

    def prepare(scope: str) -> pd.DataFrame:
        frame = filter_outcome_eligibility(
            source_frame,
            eligibility,
            cohort=cohort,
            outcome_name=outcome_name,
            eligibility_scope=scope,
        )
        frame = attach_prediction_censor_abspos(frame)
        frame = filter_registry_eligible_outcomes(
            frame,
            registry_start_date,
            cohort=cohort,
            outcome_name=outcome_name,
        )
        frame = frame[frame["split"] == split].copy()
        if allowed_subject_ids is not None:
            frame = frame[frame["subject_id"].isin(set(allowed_subject_ids))].copy()
        if frame["subject_id"].duplicated().any():
            raise ValueError(
                f"Outcome cohort {cohort or 'unknown'}/{outcome_name or 'unknown'}/"
                f"{split} contains duplicate subject_id rows."
            )
        return frame

    fixed_frame = prepare("final")
    survival_frame = prepare("ascertainment")

    competing_frame = None
    if competing_outcomes is not None:
        competing_frame = (
            competing_outcomes.copy()
            if isinstance(competing_outcomes, pd.DataFrame)
            else pd.read_parquet(competing_outcomes)
        )

    survival_records = binarize_outcomes(
        survival_frame,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        require_min_followup=False,
        split_name=split,
        competing_event_df=competing_frame,
    )
    fixed_records = binarize_outcomes(
        fixed_frame,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        require_min_followup=True,
        split_name=split,
        competing_event_df=competing_frame,
    )
    return EvaluationCohorts(
        fixed_horizon=EvaluationCohort(FIXED_HORIZON_REGIME, fixed_records),
        survival=EvaluationCohort(SURVIVAL_REGIME, survival_records),
    )


def intersect_subject_restrictions(
    *restrictions: Optional[Iterable[Any]],
) -> Optional[set[Any]]:
    """Intersect the non-null subject restrictions."""
    present = [set(values) for values in restrictions if values is not None]
    if not present:
        return None
    result = present[0]
    for values in present[1:]:
        result &= values
    return result


def assert_cohort_parity(
    expected: EvaluationCohort,
    actual_subject_ids: Iterable[Any],
    *,
    model_name: str,
    outcome_name: str,
) -> None:
    """Fail before scoring when model rows differ from the canonical cohort."""
    actual_list = list(actual_subject_ids)
    actual = set(actual_list)
    if len(actual) != len(actual_list):
        duplicates = (
            pd.Series(actual_list)[pd.Series(actual_list).duplicated()]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Cohort parity failed for model={model_name} outcome={outcome_name} "
            f"regime={expected.regime}: duplicate subject IDs={duplicates}."
        )
    missing = expected.subject_ids - actual
    extra = actual - expected.subject_ids
    if missing or extra:
        raise ValueError(
            f"Cohort parity failed for model={model_name} outcome={outcome_name} "
            f"regime={expected.regime}: expected_n={len(expected.subject_ids)} "
            f"actual_n={len(actual)} missing_n={len(missing)} extra_n={len(extra)} "
            f"missing_examples={sorted(map(str, missing))[:10]} "
            f"extra_examples={sorted(map(str, extra))[:10]}."
        )


def cohort_summary(cohort: EvaluationCohort, outcome_name: str) -> str:
    """Return a compact auditable cohort-size line."""
    return (
        f"Cohort outcome={outcome_name} regime={cohort.regime} "
        f"n={len(cohort.subject_ids)} events={cohort.n_events}"
    )
