"""Outcome construction, binarization, and prospective split helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import pandas as pd
import polars as pl

SPLIT_CONTRACT_FIELDS = {
    "date_col",
    "train_end",
    "val_start",
    "val_end",
    "test_start",
    "test_end",
    "train_key",
    "val_key",
    "test_key",
}


def load_split_contract(contract: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    """Load a prospective split contract from YAML or an in-memory mapping."""
    if isinstance(contract, Mapping):
        payload = dict(contract)
    else:
        import yaml

        path = Path(contract)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "split" in payload and not (SPLIT_CONTRACT_FIELDS & set(payload)):
        payload = dict(payload["split"])
    return {
        key: value
        for key, value in payload.items()
        if key in SPLIT_CONTRACT_FIELDS and value is not None
    }


def resolve_split_contract(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve a split spec that may reference a canonical contract file."""
    spec = dict(spec)
    contract_ref = spec.pop("contract", None)
    contract = load_split_contract(contract_ref) if contract_ref else {}
    overrides = {
        key: value
        for key, value in spec.items()
        if key in SPLIT_CONTRACT_FIELDS and value is not None
    }
    return {**contract, **overrides}


def get_subject_first_row_for_conditions(
    df,
    conditions: List[dict],
    dependence: Literal["independent", "dependent"],
):
    """Return the first priority-matching row for each matching subject."""
    if isinstance(df, pl.DataFrame):
        return _get_subject_first_row_polars(df, conditions, dependence)
    if isinstance(df, pd.DataFrame):
        return _get_subject_first_row_pandas(df, conditions, dependence)
    raise TypeError("Expected a pandas or polars DataFrame.")


def _matched_subjects(subject_sets: List[set], dependence: str) -> set:
    if not subject_sets:
        return set()
    if dependence == "independent":
        return set.union(*subject_sets)
    if dependence == "dependent":
        return set.intersection(*subject_sets)
    raise ValueError(
        f"Dependence can only be [independent, dependent], not {dependence}"
    )


def _get_subject_first_row_polars(
    df: pl.DataFrame,
    conditions: List[dict],
    dependence: Literal["independent", "dependent"],
) -> pl.DataFrame:
    working = df.with_columns(_prio=pl.lit(None).cast(pl.Int32))
    row_mask = pl.lit(False)
    subject_sets = []

    for priority, condition in enumerate(conditions):
        condition_mask = pl.col(condition["col"]).is_in(condition["vals"])
        row_mask = row_mask | condition_mask
        working = working.with_columns(
            _prio=pl.when(condition_mask & pl.col("_prio").is_null())
            .then(pl.lit(priority))
            .otherwise(pl.col("_prio"))
        )
        subject_sets.append(set(working.filter(condition_mask)["subject_id"].to_list()))

    matched = _matched_subjects(subject_sets, dependence)
    result = working.filter(pl.col("subject_id").is_in(list(matched)) & row_mask)
    sort_columns = ["_prio", "time"] if "time" in result.columns else ["_prio"]
    return (
        result.sort(sort_columns)
        .group_by("subject_id", maintain_order=True)
        .first()
        .drop("_prio")
    )


def _get_subject_first_row_pandas(
    df: pd.DataFrame,
    conditions: List[dict],
    dependence: Literal["independent", "dependent"],
) -> pd.DataFrame:
    working = df.copy()
    working["_prio"] = pd.NA
    row_mask = pd.Series(False, index=working.index)
    subject_sets = []

    for priority, condition in enumerate(conditions):
        condition_mask = working[condition["col"]].isin(condition["vals"])
        row_mask |= condition_mask
        working.loc[condition_mask & working["_prio"].isna(), "_prio"] = priority
        subject_sets.append(set(working.loc[condition_mask, "subject_id"]))

    matched = _matched_subjects(subject_sets, dependence)
    result = working.loc[working["subject_id"].isin(matched) & row_mask].copy()
    if result.empty:
        return result.drop(columns=["_prio"])
    result["_prio"] = result["_prio"].astype(int)
    sort_columns = ["_prio", "time"] if "time" in result.columns else ["_prio"]
    return (
        result.sort_values(sort_columns)
        .drop_duplicates("subject_id", keep="first")
        .drop(columns=["_prio"])
    )


def find(
    df: pd.DataFrame,
    conditions: List[dict],
    dependence: Literal["independent", "dependent"],
) -> pd.DataFrame:
    """Pandas alias used by OPERA outcome definitions."""
    return _get_subject_first_row_pandas(df, conditions, dependence)


def get_date_from_absolute_date(absolute_date):
    if absolute_date is None:
        raise ValueError("absolute_date is required.")
    return datetime(**absolute_date)


def get_date_from_relative_date(relative_dates, relative_hour_shift):
    if relative_dates is None or relative_hour_shift is None:
        raise ValueError("relative_dates and relative_hour_shift are required.")
    return relative_dates + timedelta(hours=relative_hour_shift)


def get_date_from_exposure_date(subjects, df, dependence, conditions):
    if any(value is None for value in (subjects, df, dependence, conditions)):
        raise ValueError("subjects, df, dependence, and conditions are required.")
    if not isinstance(subjects, pl.DataFrame) or not isinstance(df, pl.DataFrame):
        raise TypeError("Exposure-date construction requires polars DataFrames.")
    result = get_subject_first_row_for_conditions(df, conditions, dependence)
    return subjects.join(
        result.select("subject_id", "time"), on="subject_id", how="left"
    )["time"]


def fill_nans_with_sampled(dates: pl.Series) -> pl.Series:
    """Replace null dates by sampling observed dates with replacement."""
    if dates.is_null().all():
        raise ValueError("No non-null indexing dates found.")
    return dates.fill_null(
        dates.drop_nulls().sample(dates.len(), with_replacement=True)
    )


def _binarize_outcomes_polars(
    outcomes: pl.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int],
) -> Dict[int, dict]:
    time_delta_hours = (pl.col("outcome_date") - pl.col("index_date")).dt.total_hours()
    in_window = pl.lit(n_hours_start_include) <= time_delta_hours
    if n_hours_end_include is not None:
        in_window &= time_delta_hours <= pl.lit(n_hours_end_include)

    labelled = outcomes.with_columns(label=in_window.fill_null(False).cast(pl.Int64))
    rows = labelled.select("subject_id", "label", "censor_abspos").to_dicts()
    return {
        row["subject_id"]: {
            "label": row["label"],
            "censor_abspos": row["censor_abspos"],
        }
        for row in rows
    }


def _first_competing_dates(
    competing_event_df: Optional[pd.DataFrame],
) -> Dict[int, pd.Timestamp]:
    if competing_event_df is None or competing_event_df.empty:
        return {}
    competing = competing_event_df.copy()
    if "outcome_date" not in competing:
        return {}
    competing["outcome_date"] = pd.to_datetime(
        competing["outcome_date"], errors="coerce"
    )
    competing = competing.dropna(subset=["subject_id", "outcome_date"])
    first_dates = competing.groupby("subject_id")["outcome_date"].min()
    return {
        int(subject_id): pd.Timestamp(value)
        for subject_id, value in first_dates.items()
    }


def _binarize_outcomes_pandas(
    outcomes: pd.DataFrame,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int],
    require_min_followup: bool,
    competing_event_df: Optional[pd.DataFrame],
) -> Dict[int, dict]:
    required = {"subject_id", "index_date", "censor_date"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Outcome frame is missing columns: {sorted(missing)}")

    working = outcomes.copy()
    working["index_date"] = pd.to_datetime(working["index_date"], errors="coerce")
    working["censor_date"] = pd.to_datetime(working["censor_date"], errors="coerce")
    if "outcome_date" not in working:
        working["outcome_date"] = pd.NaT
    working["outcome_date"] = pd.to_datetime(working["outcome_date"], errors="coerce")
    if working["subject_id"].duplicated().any():
        duplicates = sorted(
            working.loc[working["subject_id"].duplicated(), "subject_id"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Outcome frame must contain one row per subject; duplicate IDs: "
            f"{duplicates[:10]}"
        )
    if working[["index_date", "censor_date"]].isna().any().any():
        raise ValueError("index_date and censor_date must be non-null.")
    if (working["censor_date"] < working["index_date"]).any():
        raise ValueError("censor_date must be on or after index_date.")

    competing_dates = _first_competing_dates(competing_event_df)
    result: Dict[int, dict] = {}

    for row in working.itertuples(index=False):
        subject_id = int(row.subject_id)
        index_date = pd.Timestamp(row.index_date)
        censor_date = pd.Timestamp(row.censor_date)
        outcome_date = getattr(row, "outcome_date", pd.NaT)

        primary_hours = None
        if pd.notna(outcome_date):
            primary_hours = (
                pd.Timestamp(outcome_date) - index_date
            ).total_seconds() / 3600.0
        primary_in_window = (
            primary_hours is not None
            and primary_hours >= n_hours_start_include
            and (n_hours_end_include is None or primary_hours <= n_hours_end_include)
            and pd.Timestamp(outcome_date) <= censor_date
        )

        competing_date = competing_dates.get(subject_id)
        competing_hours = None
        if competing_date is not None:
            competing_hours = (competing_date - index_date).total_seconds() / 3600.0
        competing_observed = (
            competing_date is not None
            and competing_date >= index_date
            and competing_date <= censor_date
            and competing_hours >= n_hours_start_include
            and (n_hours_end_include is None or competing_hours <= n_hours_end_include)
        )

        primary_first = primary_in_window and (
            not competing_observed
            or pd.Timestamp(outcome_date) <= pd.Timestamp(competing_date)
        )

        if primary_first:
            label = 1
            event = 1
            followup_date = pd.Timestamp(outcome_date)
        elif competing_observed:
            label = 0
            event = 2
            followup_date = competing_date
        else:
            label = 0
            event = 0
            followup_date = censor_date
            if n_hours_end_include is not None:
                horizon_date = index_date + timedelta(hours=n_hours_end_include)
                followup_date = min(followup_date, horizon_date)

        followup_hours = (followup_date - index_date).total_seconds() / 3600.0
        if (
            require_min_followup
            and event == 0
            and n_hours_end_include is not None
            and followup_hours < n_hours_end_include
        ):
            continue

        censor_abspos = getattr(row, "censor_abspos", None)
        if pd.isna(censor_abspos):
            censor_abspos = None
        result[subject_id] = {
            "label": label,
            "censor_abspos": censor_abspos,
            "event": event,
            "time_days": followup_hours / 24.0,
        }

    return result


def binarize_outcomes(
    outcomes,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
    require_min_followup: bool = False,
    competing_event_df: Optional[pd.DataFrame] = None,
    split_name: Optional[str] = None,
) -> Dict[int, dict]:
    """Convert event-time rows into binary and survival outcome records."""
    del split_name
    if isinstance(outcomes, pl.DataFrame):
        if require_min_followup or competing_event_df is not None:
            outcomes = outcomes.to_pandas()
        else:
            return _binarize_outcomes_polars(
                outcomes,
                n_hours_start_include,
                n_hours_end_include,
            )
    if not isinstance(outcomes, pd.DataFrame):
        raise TypeError("Expected a pandas or polars DataFrame.")
    return _binarize_outcomes_pandas(
        outcomes,
        n_hours_start_include,
        n_hours_end_include,
        require_min_followup,
        competing_event_df,
    )


def split_and_binarize_outcomes(
    outcomes,
    train_key: str,
    val_key: str,
    test_key: str,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int] = None,
    require_min_followup_train: bool = True,
    require_min_followup_val: bool = True,
    require_min_followup_test: bool = True,
    outcome_name: Optional[str] = None,
    competing_event_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, dict]]:
    """Split an outcome frame and apply split-specific follow-up rules."""
    del outcome_name
    if isinstance(outcomes, pl.DataFrame):
        if not (
            require_min_followup_train
            or require_min_followup_val
            or require_min_followup_test
            or competing_event_df is not None
        ):
            split_frames = [
                outcomes.filter(pl.col("split") == split_key)
                for split_key in (train_key, val_key, test_key)
            ]
            return tuple(
                binarize_outcomes(frame, n_hours_start_include, n_hours_end_include)
                for frame in split_frames
            )
        outcomes = outcomes.to_pandas()

    split_requirements = (
        (train_key, require_min_followup_train),
        (val_key, require_min_followup_val),
        (test_key, require_min_followup_test),
    )
    return tuple(
        binarize_outcomes(
            outcomes.loc[outcomes["split"] == split_key].copy(),
            n_hours_start_include,
            n_hours_end_include,
            require_min_followup=require_followup,
            competing_event_df=competing_event_df,
        )
        for split_key, require_followup in split_requirements
    )


def summarize_binarized_split_outputs(
    outcomes: pd.DataFrame,
    split_outputs: Dict[str, Dict[int, dict]],
    require_min_followup_by_split: Dict[str, bool],
    n_hours_end_include: Optional[int] = None,
    outcome_name: Optional[str] = None,
) -> pd.DataFrame:
    """Summarize retained labels, events, prevalence, and exclusions."""
    rows = []
    for split_key, records in split_outputs.items():
        split_frame = outcomes.loc[outcomes["split"] == split_key]
        labels = [record["label"] for record in records.values()]
        events = [record.get("event", record["label"]) for record in records.values()]
        n_retained = len(records)
        n_total = int(len(split_frame))
        rows.append(
            {
                "split": split_key,
                "outcome": outcome_name or "",
                "n_total": n_total,
                "n_retained": n_retained,
                "n_events": int(sum(event == 1 for event in events)),
                "prevalence": (
                    float(sum(labels) / n_retained) if n_retained else float("nan")
                ),
                "n_excluded_insufficient_followup": (
                    n_total - n_retained
                    if require_min_followup_by_split.get(split_key, False)
                    else 0
                ),
                "n_hours_end_include": n_hours_end_include,
            }
        )
    return pd.DataFrame(rows)


def save_binarized_split_summary(
    outcomes: pd.DataFrame,
    split_outputs: Dict[str, Dict[int, dict]],
    require_min_followup_by_split: Dict[str, bool],
    path: str,
    n_hours_end_include: Optional[int] = None,
    outcome_name: Optional[str] = None,
) -> Path:
    """Write the split-level label audit table."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_binarized_split_outputs(
        outcomes,
        split_outputs,
        require_min_followup_by_split,
        n_hours_end_include=n_hours_end_include,
        outcome_name=outcome_name,
    )
    summary.to_csv(output_path, index=False)
    return output_path


def apply_prospective_split(
    outcomes: pd.DataFrame,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end=None,
    *,
    date_col: str = "index_date",
    train_key: str = "train",
    val_key: str = "tuning",
    test_key: str = "held_out",
) -> pd.DataFrame:
    """Assign non-overlapping prospective split labels from prediction dates."""
    if date_col not in outcomes:
        raise ValueError(f"Outcome frame is missing date column {date_col!r}.")
    boundaries = {
        "train_end": pd.Timestamp(train_end),
        "val_start": pd.Timestamp(val_start),
        "val_end": pd.Timestamp(val_end),
        "test_start": pd.Timestamp(test_start),
        "test_end": pd.Timestamp(test_end) if test_end is not None else None,
    }
    if boundaries["val_start"] > boundaries["val_end"]:
        raise ValueError("val_start must be on or before val_end.")
    if boundaries["val_end"] >= boundaries["test_start"]:
        raise ValueError("Validation and test date windows must not overlap.")

    result = outcomes.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    result["split"] = "excluded"
    result.loc[dates <= boundaries["train_end"], "split"] = train_key
    result.loc[
        dates.between(boundaries["val_start"], boundaries["val_end"]),
        "split",
    ] = val_key
    test_mask = dates >= boundaries["test_start"]
    if boundaries["test_end"] is not None:
        test_mask &= dates <= boundaries["test_end"]
    result.loc[test_mask, "split"] = test_key
    return result


def summarize_outcome_splits(
    outcomes: pd.DataFrame,
    outcome_name: str,
) -> pd.DataFrame:
    """Count subjects, observed events, and prevalence by split."""
    rows = []
    for split_key, group in outcomes.groupby("split", sort=False):
        n_subjects = int(group["subject_id"].nunique())
        if "label" in group:
            n_events = int(group["label"].fillna(0).sum())
        elif "outcome_date" in group:
            n_events = int(group["outcome_date"].notna().sum())
        else:
            n_events = 0
        rows.append(
            {
                "split": split_key,
                "outcome": outcome_name,
                "n_subjects": n_subjects,
                "n_events_observed": n_events,
                "label_prevalence": (
                    n_events / n_subjects if n_subjects else float("nan")
                ),
                "index_date_min": (
                    pd.to_datetime(group["index_date"]).min()
                    if "index_date" in group
                    else pd.NaT
                ),
                "index_date_max": (
                    pd.to_datetime(group["index_date"]).max()
                    if "index_date" in group
                    else pd.NaT
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_split_integrity(
    outcomes: pd.DataFrame,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end=None,
    *,
    date_col: str = "index_date",
    train_key: str = "train",
    val_key: str = "tuning",
    test_key: str = "held_out",
) -> dict:
    """Report subject overlap and prospective date-boundary violations."""
    required = {"subject_id", "split", date_col}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Outcome frame is missing columns: {sorted(missing)}")

    split_subjects = {
        split_key: set(group["subject_id"])
        for split_key, group in outcomes.groupby("split", sort=False)
    }
    overlaps = {}
    split_names = list(split_subjects)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            count = len(split_subjects[left] & split_subjects[right])
            if count:
                overlaps[f"{left}__{right}"] = count

    dates = pd.to_datetime(outcomes[date_col], errors="coerce")
    split_values = outcomes["split"]
    violation_masks = {
        train_key: dates > pd.Timestamp(train_end),
        val_key: ~dates.between(pd.Timestamp(val_start), pd.Timestamp(val_end)),
        test_key: dates < pd.Timestamp(test_start),
    }
    if test_end is not None:
        violation_masks[test_key] |= dates > pd.Timestamp(test_end)
    violations = {
        split_key: int(((split_values == split_key) & mask).sum())
        for split_key, mask in violation_masks.items()
    }
    violations = {key: value for key, value in violations.items() if value}
    return {
        "ok": not overlaps and not violations,
        "subject_overlap_counts": overlaps,
        "date_boundary_violations": violations,
    }
