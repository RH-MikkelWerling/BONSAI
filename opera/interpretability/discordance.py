"""IPI-discordant patient analysis for OPERA."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from opera.evaluation.comparison import (
    compute_comparison_metrics,
    derive_horizon_labels,
    fit_linear_survival_or_logistic,
    format_table_row,
    infer_tau_days,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_IPI_RISK_MAP = {
    "low": 0.15,
    "low_intermediate": 0.30,
    "high_intermediate": 0.55,
    "high": 0.75,
}


def find_discordant_patients(
    *,
    outcomes: pd.DataFrame,
    rkkp: pd.DataFrame,
    opera_risk: pd.DataFrame,
    outcome: str = "OS_2y",
    disease: str = "DLBCL",
    threshold_percentile: float = 25,
    ipi_probability_map: Optional[Mapping[Any, float]] = None,
) -> Dict[str, Any]:
    """Identify low-IPI / high-OPERA discordant patients."""
    ipi_probability_map = dict(ipi_probability_map or DEFAULT_IPI_RISK_MAP)
    outcome_df = outcomes.loc[
        (outcomes["outcome_name"] == outcome) & (outcomes["disease_subtype"] == disease)
    ].copy()
    merged = outcome_df.merge(rkkp, on="patient_id", how="inner").merge(
        opera_risk,
        on="patient_id",
        how="inner",
    )
    merged = merged.dropna(subset=["ipi_score", "opera_risk"])
    merged["ipi_risk_probability"] = merged.apply(
        lambda row: _ipi_probability(row, ipi_probability_map),
        axis=1,
    )
    merged["ipi_risk_probability"] = _minmax_scale(merged["ipi_risk_probability"])
    merged["opera_risk"] = _minmax_scale(merged["opera_risk"])

    low_cut = np.nanpercentile(merged["ipi_risk_probability"], threshold_percentile)
    high_cut = np.nanpercentile(merged["opera_risk"], 100 - threshold_percentile)
    opera_low_cut = np.nanpercentile(merged["opera_risk"], threshold_percentile)

    discordant = merged.loc[
        (merged["ipi_risk_probability"] <= low_cut) & (merged["opera_risk"] >= high_cut)
    ].copy()
    concordant_low = merged.loc[
        (merged["ipi_risk_probability"] <= low_cut)
        & (merged["opera_risk"] <= opera_low_cut)
    ].copy()

    return {
        "outcome": outcome,
        "disease": disease,
        "threshold_percentile": threshold_percentile,
        "n_rkkp_patients": int(len(merged)),
        "discordant_ids": discordant["patient_id"].tolist(),
        "concordant_low_ids": concordant_low["patient_id"].tolist(),
        "summary_table": merged[
            [
                "patient_id",
                "disease_subtype",
                "ipi_score",
                "ipi_risk_probability",
                "opera_risk",
                "event_indicator",
                "time_to_event",
            ]
        ].copy(),
    }


def characterize_discordance(
    discordant_ids: Sequence[Any],
    concordant_low_ids: Sequence[Any],
    *,
    ehr_features: pd.DataFrame,
    feature_descriptions: Optional[Mapping[str, str]] = None,
    ig_attributions: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Find EHR and optional attribution features that distinguish discordant patients."""
    feature_descriptions = dict(feature_descriptions or {})
    compare_ids = list(discordant_ids) + list(concordant_low_ids)
    frame = ehr_features.loc[ehr_features["patient_id"].isin(compare_ids)].copy()
    group = frame["patient_id"].isin(discordant_ids).astype(int)
    frame = frame.assign(is_discordant=group.values)

    feature_rows: List[Dict[str, Any]] = []
    for column in frame.columns:
        if column in {"patient_id", "is_discordant"}:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        x = (
            frame.loc[frame["is_discordant"] == 1, column]
            .dropna()
            .to_numpy(dtype=float)
        )
        y = (
            frame.loc[frame["is_discordant"] == 0, column]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(x) == 0 or len(y) == 0:
            continue
        stat = _mann_whitney_u(x, y)
        feature_rows.append(
            {
                "feature": column,
                "effect_size": stat["rank_biserial_correlation"],
                "p_value": stat["p_value"],
                "discordant_mean": float(np.mean(x)),
                "concordant_low_mean": float(np.mean(y)),
                "feature_description": feature_descriptions.get(column, column),
            }
        )

    feature_df = pd.DataFrame(feature_rows).sort_values("p_value", na_position="last")
    if not feature_df.empty:
        feature_df["fdr_q_value"] = _benjamini_hochberg(
            feature_df["p_value"].to_numpy(dtype=float)
        )
        feature_df = feature_df.sort_values(
            ["fdr_q_value", "p_value"], na_position="last"
        ).reset_index(drop=True)

    attribution_df = pd.DataFrame()
    if ig_attributions is not None and not ig_attributions.empty:
        ig_frame = ig_attributions.loc[
            ig_attributions["patient_id"].isin(compare_ids)
        ].copy()
        ig_grouped = ig_frame.groupby(["patient_id", "event_code"], as_index=False)[
            "attribution"
        ].mean()
        ig_grouped["is_discordant"] = (
            ig_grouped["patient_id"].isin(discordant_ids).astype(int)
        )
        ig_rows = []
        for event_code, group_df in ig_grouped.groupby("event_code"):
            x = group_df.loc[group_df["is_discordant"] == 1, "attribution"].to_numpy(
                dtype=float
            )
            y = group_df.loc[group_df["is_discordant"] == 0, "attribution"].to_numpy(
                dtype=float
            )
            if len(x) == 0 or len(y) == 0:
                continue
            stat = _mann_whitney_u(x, y)
            ig_rows.append(
                {
                    "event_code": event_code,
                    "effect_size": stat["rank_biserial_correlation"],
                    "p_value": stat["p_value"],
                    "discordant_mean_attribution": float(np.mean(x)),
                    "concordant_low_mean_attribution": float(np.mean(y)),
                }
            )
        attribution_df = pd.DataFrame(ig_rows).sort_values(
            "p_value", na_position="last"
        )
        if not attribution_df.empty:
            attribution_df["fdr_q_value"] = _benjamini_hochberg(
                attribution_df["p_value"].to_numpy(dtype=float)
            )
            attribution_df = attribution_df.sort_values(
                ["fdr_q_value", "p_value"],
                na_position="last",
            ).reset_index(drop=True)

    forest_plot_df = feature_df.head(15).copy()
    if not forest_plot_df.empty:
        forest_plot_df["ci_lower"] = forest_plot_df["effect_size"] - 1.96 * np.sqrt(
            np.maximum(
                1e-8,
                (1.0 - forest_plot_df["effect_size"].abs()) / max(1, len(compare_ids)),
            )
        )
        forest_plot_df["ci_upper"] = forest_plot_df["effect_size"] + 1.96 * np.sqrt(
            np.maximum(
                1e-8,
                (1.0 - forest_plot_df["effect_size"].abs()) / max(1, len(compare_ids)),
            )
        )

    return {
        "feature_table": feature_df,
        "attribution_table": attribution_df,
        "forest_plot_table": forest_plot_df,
    }


def validate_discordant_features(
    top_features: Sequence[str],
    *,
    outcomes: pd.DataFrame,
    rkkp: pd.DataFrame,
    ehr_features: pd.DataFrame,
    outcome: str = "OS_2y",
    disease: str = "DLBCL",
    split_col: str = "split",
) -> Dict[str, Any]:
    """Test whether top discordant features improve an IPI-based baseline."""
    outcome_df = outcomes.loc[
        (outcomes["outcome_name"] == outcome) & (outcomes["disease_subtype"] == disease)
    ].copy()
    merged = outcome_df.merge(rkkp, on="patient_id", how="inner").merge(
        ehr_features[
            ["patient_id", *[f for f in top_features if f in ehr_features.columns]]
        ],
        on="patient_id",
        how="left",
    )
    merged = merged.dropna(subset=["ipi_score", "time_to_event", "event_indicator"])
    if split_col not in merged.columns:
        raise ValueError(
            "validate_discordant_features requires a split column to preserve the "
            "prospective test-set design."
        )

    tau_days = infer_tau_days(outcome)
    binary = derive_horizon_labels(
        times=merged["time_to_event"].to_numpy(dtype=float),
        events=merged["event_indicator"].to_numpy(dtype=int),
        tau_days=tau_days,
    )
    merged["binary_label"] = binary["label"]
    merged["binary_eligible"] = binary["eligible"]

    train_mask = merged[split_col].isin(["train", "tuning"])
    test_mask = merged[split_col].eq("held_out")
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Non-empty train/tuning and held_out splits are required.")

    clinical_features = merged.loc[:, ["ipi_score"]].copy()
    augmented_columns = ["ipi_score", *[f for f in top_features if f in merged.columns]]
    augmented_features = merged.loc[:, augmented_columns].copy()

    baseline_model, _, baseline_notes = fit_linear_survival_or_logistic(
        features=clinical_features.loc[train_mask],
        times=merged.loc[train_mask, "time_to_event"].to_numpy(dtype=float),
        events=merged.loc[train_mask, "event_indicator"].to_numpy(dtype=int),
        labels=merged.loc[train_mask, "binary_label"].to_numpy(dtype=float),
        eligible=merged.loc[train_mask, "binary_eligible"].to_numpy(dtype=bool),
        tau_days=tau_days,
        seed=42,
    )
    augmented_model, _, augmented_notes = fit_linear_survival_or_logistic(
        features=augmented_features.loc[train_mask],
        times=merged.loc[train_mask, "time_to_event"].to_numpy(dtype=float),
        events=merged.loc[train_mask, "event_indicator"].to_numpy(dtype=int),
        labels=merged.loc[train_mask, "binary_label"].to_numpy(dtype=float),
        eligible=merged.loc[train_mask, "binary_eligible"].to_numpy(dtype=bool),
        tau_days=tau_days,
        seed=42,
    )

    baseline_predictions = merged.loc[
        test_mask,
        [
            "patient_id",
            "time_to_event",
            "event_indicator",
            "binary_label",
            "binary_eligible",
        ],
    ].copy()
    baseline_predictions["predicted_probability"] = baseline_model.predict_proba(
        clinical_features.loc[test_mask]
    )
    baseline_predictions["predicted_risk"] = baseline_model.predict_risk(
        clinical_features.loc[test_mask]
    )

    augmented_predictions = merged.loc[
        test_mask,
        [
            "patient_id",
            "time_to_event",
            "event_indicator",
            "binary_label",
            "binary_eligible",
        ],
    ].copy()
    augmented_predictions["predicted_probability"] = augmented_model.predict_proba(
        augmented_features.loc[test_mask]
    )
    augmented_predictions["predicted_risk"] = augmented_model.predict_risk(
        augmented_features.loc[test_mask]
    )

    baseline_metrics = compute_comparison_metrics(
        baseline_predictions, tau_days=tau_days
    )
    augmented_metrics = compute_comparison_metrics(
        augmented_predictions, tau_days=tau_days
    )

    return {
        "baseline_metrics": baseline_metrics,
        "augmented_metrics": augmented_metrics,
        "metric_deltas": {
            key: augmented_metrics.get(key, float("nan"))
            - baseline_metrics.get(key, float("nan"))
            for key in ["td_auroc", "c_index", "ipcw_auroc", "ipcw_brier", "ici", "mce"]
        },
        "summary_table": pd.DataFrame(
            [
                {"model": "IPI_only", **baseline_metrics},
                {"model": "IPI_plus_features", **augmented_metrics},
            ]
        ),
        "latex_like_summary": {
            "IPI_only": format_table_row(baseline_metrics, {}),
            "IPI_plus_features": format_table_row(augmented_metrics, {}),
        },
        "notes": baseline_notes + augmented_notes,
    }


def _ipi_probability(row: pd.Series, ipi_probability_map: Mapping[Any, float]) -> float:
    if "ipi_risk_group" in row and pd.notna(row.get("ipi_risk_group")):
        key = str(row["ipi_risk_group"]).strip().lower().replace(" ", "_")
        if key in ipi_probability_map:
            return float(ipi_probability_map[key])
    return float(row["ipi_score"])


def _minmax_scale(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    lo = np.nanmin(array)
    hi = np.nanmax(array)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(array, 0.5, dtype=float)
    return (array - lo) / (hi - lo)


def _mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    rank_x = ranks[: len(x)].sum()
    u_x = rank_x - len(x) * (len(x) + 1) / 2.0
    mean_u = len(x) * len(y) / 2.0
    std_u = np.sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12.0)
    if std_u == 0.0:
        z = 0.0
        p_value = 1.0
    else:
        z = (u_x - mean_u) / std_u
        p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    rank_biserial = (2.0 * u_x / (len(x) * len(y))) - 1.0
    return {
        "u_statistic": float(u_x),
        "p_value": float(p_value),
        "rank_biserial_correlation": float(rank_biserial),
    }


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / np.sqrt(2.0))))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adjusted[i] = value
        prev = value
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


__all__ = [
    "characterize_discordance",
    "find_discordant_patients",
    "validate_discordant_features",
]
