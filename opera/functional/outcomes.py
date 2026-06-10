"""OPERA outcome-frame helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from bonsai.functional.features import compute_abspos

LOGGER = logging.getLogger(__name__)


def _pandas_abspos(timestamps: pd.Series) -> pd.Series:
    """Convert timestamps to float32 hours since Unix epoch."""
    return compute_abspos(timestamps)


def attach_prediction_censor_abspos(
    outcomes: pd.DataFrame,
    prediction_time_col: str = "index_date",
) -> pd.DataFrame:
    """Set sequence truncation position from the prediction origin.

    ``censor_date`` is retained for follow-up and time-to-event calculations.
    The model input should be truncated at ``index_date`` so downstream
    finetuning and evaluation do not see post-index information.
    """
    if prediction_time_col not in outcomes.columns:
        raise ValueError(
            f"Outcome frame is missing prediction-time column {prediction_time_col!r}."
        )
    outcomes = outcomes.copy()
    outcomes["censor_abspos"] = _pandas_abspos(outcomes[prediction_time_col])
    return outcomes


def filter_registry_eligible_outcomes(
    outcomes: pd.DataFrame,
    registry_start_date: Optional[Any] = None,
    *,
    date_col: str = "index_date",
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
) -> pd.DataFrame:
    """Restrict supervised outcome rows to registry-covered prediction times.

    Patients with prediction times before registry coverage begins can still
    contribute to MLM/DAPT pretraining, but they should not be treated as
    supervised cases, controls, or ordinary RKKP-missing rows.
    """
    if registry_start_date in (None, "", "null"):
        return outcomes.copy()
    if date_col not in outcomes.columns:
        raise ValueError(
            f"Cannot apply registry_start_date; outcome frame is missing {date_col!r}."
        )
    start = pd.Timestamp(registry_start_date)
    dates = pd.to_datetime(outcomes[date_col])
    eligible = dates >= start
    n_before = int(len(outcomes))
    filtered = outcomes.loc[eligible].copy()
    n_excluded = n_before - int(len(filtered))
    LOGGER.info(
        "registry_eligibility_filter cohort=%s outcome=%s start=%s "
        "excluded=%s/%s retained=%s",
        cohort or "unknown",
        outcome_name or "unknown",
        start.date(),
        n_excluded,
        n_before,
        len(filtered),
    )
    return filtered
