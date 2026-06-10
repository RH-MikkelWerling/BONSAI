"""Contextualize OPERA sigma values by outcome information content."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def build_sigma_context_table(
    sigma_values: pd.DataFrame | dict,
    outcome_metadata: Optional[pd.DataFrame] = None,
    *,
    outcome_col: str = "outcome",
    sigma_col: str = "sigma",
) -> pd.DataFrame:
    """Join learned sigma values with event counts, prevalence, and n_eff.

    The table keeps raw sigma for interpretability and adds precision plus
    count-normalized descriptors. If `n_effective_pairs` is available, it is
    used as the preferred denominator because it reflects the informative
    pair mass that actually trained the OPERA objective.
    """
    if isinstance(sigma_values, dict):
        frame = pd.DataFrame(
            {
                outcome_col: list(sigma_values.keys()),
                sigma_col: list(sigma_values.values()),
            }
        )
    else:
        frame = sigma_values.copy()
    if outcome_col not in frame.columns:
        raise ValueError(f"sigma table must contain {outcome_col!r}.")
    if sigma_col not in frame.columns:
        raise ValueError(f"sigma table must contain {sigma_col!r}.")
    if outcome_metadata is not None and not outcome_metadata.empty:
        meta = outcome_metadata.copy()
        frame = frame.merge(meta, on=outcome_col, how="left")

    sigma = np.asarray(frame[sigma_col], dtype=float)
    frame["log_sigma"] = np.log(np.maximum(sigma, 1e-12))
    frame["precision"] = 0.5 * np.exp(-2.0 * frame["log_sigma"])
    if (
        "n_events" in frame.columns
        and "n_total" in frame.columns
        and "prevalence" not in frame.columns
    ):
        frame["prevalence"] = frame["n_events"] / frame["n_total"].replace(0, np.nan)
    if "n_effective_pairs" in frame.columns:
        denom = np.maximum(
            pd.to_numeric(frame["n_effective_pairs"], errors="coerce"), 1.0
        )
        frame["sigma_per_sqrt_effective_pair"] = sigma / np.sqrt(denom)
        frame["log_n_effective_pairs"] = np.log(denom)
    if "n_events" in frame.columns:
        events = np.maximum(pd.to_numeric(frame["n_events"], errors="coerce"), 1.0)
        frame["sigma_per_sqrt_event"] = sigma / np.sqrt(events)
        frame["log_n_events"] = np.log(events)
    if {"n_censored", "n_total"}.issubset(
        frame.columns
    ) and "censoring_fraction" not in frame.columns:
        frame["censoring_fraction"] = frame["n_censored"] / frame["n_total"].replace(
            0, np.nan
        )
    return frame


def residualize_log_sigma(
    sigma_context: pd.DataFrame,
    *,
    covariates: Optional[Iterable[str]] = None,
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """Add residual log-sigma after adjusting for prevalence and pair support.

    This is an analysis/reporting layer, not a training change. It helps
    separate "the outcome had few informative pairs" from "the outcome was
    intrinsically weakly aligned with the learned representation."
    """
    frame = sigma_context.copy()
    if "log_sigma" not in frame.columns:
        frame = build_sigma_context_table(frame, outcome_col=outcome_col)
    if covariates is None:
        covariates = [
            col
            for col in (
                "log_n_effective_pairs",
                "log_n_events",
                "prevalence",
                "censoring_fraction",
            )
            if col in frame.columns
        ]
    covariates = [col for col in covariates if col in frame.columns]
    frame["log_sigma_residual"] = np.nan
    if not covariates:
        frame["log_sigma_residual"] = frame["log_sigma"]
        return frame

    model_df = (
        frame[[*covariates, "log_sigma"]].replace([np.inf, -np.inf], np.nan).dropna()
    )
    if len(model_df) <= len(covariates):
        frame["log_sigma_residual"] = frame["log_sigma"]
        return frame
    x = model_df[covariates].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = model_df["log_sigma"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    all_x = frame[covariates].replace([np.inf, -np.inf], np.nan)
    valid = all_x.notna().all(axis=1) & frame["log_sigma"].notna()
    pred_x = np.column_stack(
        [np.ones(int(valid.sum())), all_x.loc[valid].to_numpy(dtype=float)]
    )
    frame.loc[valid, "log_sigma_residual"] = (
        frame.loc[valid, "log_sigma"] - pred_x @ beta
    )
    frame["sigma_residual_ratio"] = np.exp(frame["log_sigma_residual"])
    return frame
