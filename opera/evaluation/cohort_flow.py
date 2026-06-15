"""Validation and aggregation for outcome-specific eligibility audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


REQUIRED_ELIGIBILITY_COLUMNS = {
    "subject_id",
    "split",
    "eligible",
    "eligibility_reason",
}
ALLOWED_SPLITS = {"train", "tuning", "held_out"}
CRITERION_COLUMNS = (
    "source_covered",
    "baseline_adequate",
    "post_index_adequate",
    "followup_adequate",
    "outcome_observed",
)


def eligibility_file_path(
    data_dir: str,
    cohort_name: str,
    outcome_name: str,
    outcome_cfg: Mapping[str, Any],
) -> Optional[Path]:
    """Resolve an optional eligibility sidecar under a cohort data directory."""
    raw = outcome_cfg.get("eligibility_file")
    if raw in (None, "", "null"):
        return None
    rendered = str(raw).format(
        cohort=cohort_name,
        outcome=outcome_name,
        data_dir=data_dir,
    )
    path = Path(rendered)
    if path.is_absolute():
        return path
    return Path(data_dir) / "outcomes" / path


def load_eligibility_frame(path: str | Path) -> pd.DataFrame:
    """Load a CSV or parquet eligibility sidecar."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported eligibility file extension {path.suffix!r}; "
        "expected CSV or parquet."
    )


def validate_eligibility_frame(frame: pd.DataFrame) -> list[str]:
    """Return contract violations for a patient-outcome eligibility table."""
    issues: list[str] = []
    missing = REQUIRED_ELIGIBILITY_COLUMNS - set(frame.columns)
    if missing:
        return [f"missing required columns: {sorted(missing)}"]
    if frame.empty:
        issues.append("contains no rows")
        return issues

    if frame["subject_id"].isna().any():
        issues.append("subject_id contains missing values")
    duplicate_count = int(frame["subject_id"].duplicated().sum())
    if duplicate_count:
        issues.append(f"contains {duplicate_count} duplicate subject_id rows")

    invalid_splits = sorted(set(frame["split"].dropna().astype(str)) - ALLOWED_SPLITS)
    if invalid_splits:
        issues.append(f"split contains unsupported values: {invalid_splits}")
    if frame["split"].isna().any():
        issues.append("split contains missing values")

    eligible = _coerce_nullable_boolean(frame["eligible"])
    if eligible.isna().any():
        issues.append("eligible must contain only boolean-like values")
    reasons = frame["eligibility_reason"].fillna("").astype(str).str.strip()
    if ((eligible == False) & reasons.eq("")).any():  # noqa: E712
        issues.append("ineligible rows require a non-empty eligibility_reason")

    for column in CRITERION_COLUMNS:
        if column not in frame.columns:
            continue
        values = _coerce_nullable_boolean(frame[column])
        invalid = frame[column].notna() & values.isna()
        if invalid.any():
            issues.append(f"{column} must contain only boolean-like values or null")

    gate_columns = (
        "source_covered",
        "baseline_adequate",
        "post_index_adequate",
        "followup_adequate",
    )
    for column in gate_columns:
        if column not in frame.columns:
            continue
        values = _coerce_nullable_boolean(frame[column])
        if ((eligible == True) & (values == False)).any():  # noqa: E712
            issues.append(f"eligible rows must not have {column}=false")

    if {"post_index_adequate", "last_measurement_date"}.issubset(frame.columns):
        post_index = _coerce_nullable_boolean(frame["post_index_adequate"])
        missing_last = frame["last_measurement_date"].isna()
        if ((post_index == True) & missing_last).any():  # noqa: E712
            issues.append(
                "post_index_adequate=true requires a non-null last_measurement_date"
            )
    return issues


def summarize_eligibility_frame(
    frame: pd.DataFrame,
    cohort_name: str,
    outcome_name: str,
) -> pd.DataFrame:
    """Create a tidy cohort-flow table from one eligibility sidecar."""
    issues = validate_eligibility_frame(frame)
    if issues:
        raise ValueError("; ".join(issues))

    data = frame.copy()
    data["eligible"] = _coerce_nullable_boolean(data["eligible"]).astype(bool)
    rows: list[dict[str, Any]] = []
    split_order = ("train", "tuning", "held_out")

    for split in split_order:
        split_data = data[data["split"] == split]
        if split_data.empty:
            continue
        rows.append(
            _flow_row(
                cohort_name,
                outcome_name,
                split,
                "source_population",
                "total",
                None,
                len(split_data),
            )
        )
        for criterion in CRITERION_COLUMNS:
            if criterion not in split_data.columns:
                continue
            values = _coerce_nullable_boolean(split_data[criterion])
            for status, count in (
                ("pass", int((values == True).sum())),  # noqa: E712
                ("fail", int((values == False).sum())),  # noqa: E712
                ("unknown", int(values.isna().sum())),
            ):
                rows.append(
                    _flow_row(
                        cohort_name,
                        outcome_name,
                        split,
                        criterion,
                        status,
                        None,
                        count,
                    )
                )

        for status, count in (
            ("included", int(split_data["eligible"].sum())),
            ("excluded", int((~split_data["eligible"]).sum())),
        ):
            rows.append(
                _flow_row(
                    cohort_name,
                    outcome_name,
                    split,
                    "final_eligibility",
                    status,
                    None,
                    count,
                )
            )

        excluded = split_data[~split_data["eligible"]]
        reason_counts = (
            excluded["eligibility_reason"]
            .fillna("unspecified")
            .astype(str)
            .str.strip()
            .replace("", "unspecified")
            .value_counts()
        )
        for reason, count in reason_counts.items():
            rows.append(
                _flow_row(
                    cohort_name,
                    outcome_name,
                    split,
                    "exclusion_reason",
                    "excluded",
                    reason,
                    int(count),
                )
            )

    return pd.DataFrame(
        rows,
        columns=("cohort", "outcome", "split", "stage", "status", "reason", "n"),
    )


def _flow_row(
    cohort: str,
    outcome: str,
    split: str,
    stage: str,
    status: str,
    reason: Optional[str],
    count: int,
) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "outcome": outcome,
        "split": split,
        "stage": stage,
        "status": status,
        "reason": reason,
        "n": count,
    }


def _coerce_nullable_boolean(series: pd.Series) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
    }

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        key = value.strip().lower() if isinstance(value, str) else value
        return mapping.get(key, pd.NA)

    return series.map(convert).astype("boolean")
