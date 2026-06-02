"""OPERA outcome-frame helpers."""

from __future__ import annotations

import pandas as pd

from bonsai.functional.features import compute_abspos


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
    outcomes["censor_abspos"] = compute_abspos(outcomes[prediction_time_col])
    return outcomes
