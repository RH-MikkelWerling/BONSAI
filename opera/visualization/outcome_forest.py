"""Outcome-atlas forest plots for absolute performance and paired contrasts.

The plotting contract is deliberately model-agnostic so frequentist bootstrap
summaries and Bayesian posterior summaries can use the same artwork.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.visualization.style import (
    ANNOT_SIZE,
    CATEGORICAL,
    LABEL_SIZE,
    LEGEND_SIZE,
    NOTE_SIZE,
    PALETTE,
    TICK_SIZE,
    despine,
    model_color,
    model_label,
    save_fig,
    stable_index,
)


ESTIMATE_COLUMNS = (
    "outcome",
    "outcome_family",
    "cohort",
    "horizon_days",
    "model",
    "estimate",
    "lower",
    "upper",
    "n",
    "n_events",
)
CONTRAST_COLUMNS = (
    "outcome",
    "outcome_family",
    "cohort",
    "horizon_days",
    "contrast",
    "comparator",
    "baseline",
    "estimate",
    "lower",
    "upper",
    "probability_superior",
    "n",
    "n_events",
)


def _first_present(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _complete_schema(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result.loc[:, list(columns)]


def normalize_estimate_table(
    frame: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Normalize long absolute estimates to the forest plotting contract."""
    result = frame.copy()
    aliases = {
        "model": ("model", "model_family", "run_identity"),
        "estimate": ("estimate", "mean", "value", metric),
        "lower": ("lower", "ci_lower", f"{metric}_lower"),
        "upper": ("upper", "ci_upper", f"{metric}_upper"),
        "outcome_family": ("outcome_family", "family"),
        "cohort": ("cohort", "cohort_fine", "evaluation_cohort"),
        "n": ("n", "n_total", "n_test", "n_paired"),
        "n_events": ("n_events", "n_positive", "n_events_test", "n_cases"),
    }
    for target, candidates in aliases.items():
        if target in result:
            continue
        source = _first_present(result, candidates)
        if source is not None:
            result[target] = result[source]
    required = {"outcome", "model", "estimate"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Absolute forest estimates are missing: {sorted(missing)}")
    result["outcome"] = result["outcome"].astype(str)
    result["model"] = result["model"].astype(str)
    if "outcome_family" not in result:
        result["outcome_family"] = "Other"
    result["outcome_family"] = result["outcome_family"].fillna("Other").astype(str)
    _numeric(result, ["estimate", "lower", "upper", "horizon_days", "n", "n_events"])
    result = _complete_schema(result, ESTIMATE_COLUMNS)
    duplicate = result.duplicated(
        ["outcome", "cohort", "horizon_days", "model"], keep=False
    )
    if duplicate.any():
        keys = result.loc[
            duplicate, ["outcome", "cohort", "horizon_days", "model"]
        ].head(5)
        raise ValueError(
            "Absolute forest estimates contain duplicate cells: "
            + keys.to_dict("records").__repr__()
        )
    return result


def normalize_contrast_table(
    frame: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Normalize paired/bootstrap/Bayesian contrasts to one contract."""
    result = frame.copy()
    delta_name = f"delta_{metric}"
    aliases = {
        "estimate": ("estimate", "delta_mean", "pooled_effect", delta_name),
        "lower": ("lower", "delta_lower", "ci_lower", f"{delta_name}_lower"),
        "upper": ("upper", "delta_upper", "ci_upper", f"{delta_name}_upper"),
        "probability_superior": (
            "probability_superior",
            "posterior_probability_superior",
            "probability_gt_zero",
            "p_superior",
        ),
        "outcome_family": ("outcome_family", "family"),
        "cohort": ("cohort", "cohort_fine", "evaluation_cohort"),
        "n": ("n", "n_total", "n_test", "n_paired"),
        "n_events": ("n_events", "n_positive", "n_events_test", "n_cases"),
    }
    for target, candidates in aliases.items():
        if target in result:
            continue
        source = _first_present(result, candidates)
        if source is not None:
            result[target] = result[source]
    if "contrast" not in result:
        if {"comparator", "baseline"}.issubset(result):
            result["contrast"] = (
                result["comparator"].astype(str)
                + "_minus_"
                + result["baseline"].astype(str)
            )
        else:
            result["contrast"] = "comparison"
    required = {"outcome", "contrast", "estimate"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Forest contrasts are missing: {sorted(missing)}")
    result["outcome"] = result["outcome"].astype(str)
    result["contrast"] = result["contrast"].astype(str)
    if "outcome_family" not in result:
        result["outcome_family"] = "Other"
    result["outcome_family"] = result["outcome_family"].fillna("Other").astype(str)
    _numeric(
        result,
        [
            "estimate",
            "lower",
            "upper",
            "probability_superior",
            "horizon_days",
            "n",
            "n_events",
        ],
    )
    result = _complete_schema(result, CONTRAST_COLUMNS)
    duplicate = result.duplicated(
        ["outcome", "cohort", "horizon_days", "contrast"], keep=False
    )
    if duplicate.any():
        raise ValueError("Forest contrasts contain duplicate outcome/contrast cells.")
    return result


def paired_comparison_to_forest_tables(
    frame: pd.DataFrame,
    *,
    metric: str = "cif_auc",
    comparator: str = "opera",
    baseline: str = "tabular_ehr",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt ``compare_direct_cr_tabular`` output to the forest contract."""
    comparator_col = f"{comparator}_{metric}"
    baseline_prefix = "tabular" if baseline == "tabular_ehr" else baseline
    baseline_col = f"{baseline_prefix}_{metric}"
    delta_col = f"delta_{metric}"
    required = {"outcome", comparator_col, baseline_col, delta_col}
    missing = required - set(frame)
    if missing:
        raise ValueError(
            f"Paired comparison is missing required columns: {sorted(missing)}"
        )
    shared = [
        column
        for column in (
            "outcome",
            "outcome_family",
            "cohort",
            "cohort_fine",
            "horizon_days",
            "n_paired",
            "n_cases",
        )
        if column in frame
    ]
    estimate_parts = []
    for model, column in ((comparator, comparator_col), (baseline, baseline_col)):
        part = frame[shared].copy()
        part["model"] = model
        part["estimate"] = frame[column]
        for target, candidate in (
            ("lower", f"{column}_lower"),
            ("upper", f"{column}_upper"),
        ):
            if candidate in frame:
                part[target] = frame[candidate]
        estimate_parts.append(part)
    estimates = normalize_estimate_table(
        pd.concat(estimate_parts, ignore_index=True), metric=metric
    )
    contrasts = frame[shared].copy()
    contrasts["comparator"] = comparator
    contrasts["baseline"] = baseline
    contrasts["contrast"] = f"{comparator}_minus_{baseline}"
    contrasts["estimate"] = frame[delta_col]
    for target, candidate in (
        ("lower", f"{delta_col}_lower"),
        ("upper", f"{delta_col}_upper"),
    ):
        if candidate in frame:
            contrasts[target] = frame[candidate]
    return estimates, normalize_contrast_table(contrasts, metric=metric)


def apply_outcome_family_map(
    frame: pd.DataFrame,
    family_by_outcome: Mapping[str, str],
) -> pd.DataFrame:
    result = frame.copy()
    mapped = result["outcome"].map(family_by_outcome)
    current = result.get("outcome_family", pd.Series(index=result.index, dtype=object))
    result["outcome_family"] = mapped.fillna(current).fillna("Other")
    return result


def _display_outcome(value: str) -> str:
    text = str(value).replace("_", " ")
    text = re.sub(r"\bg([123])plus\b", r"grade \1+", text, flags=re.IGNORECASE)
    text = re.sub(r"\baki\b", "AKI", text, flags=re.IGNORECASE)
    text = re.sub(r"\biv\b", "IV", text, flags=re.IGNORECASE)
    return text if text.startswith(("AKI", "IV")) else text[:1].upper() + text[1:]


def _family_color(family: str) -> str:
    return CATEGORICAL[stable_index(family) % len(CATEGORICAL)]


def _validate_single_slice(
    estimates: pd.DataFrame,
    contrasts: pd.DataFrame | None,
) -> tuple[str | None, float | None]:
    combined = estimates
    if contrasts is not None and not contrasts.empty:
        combined = pd.concat(
            [estimates[["cohort", "horizon_days"]], contrasts[["cohort", "horizon_days"]]],
            ignore_index=True,
        )
    cohorts = combined["cohort"].dropna().astype(str).unique()
    horizons = combined["horizon_days"].dropna().astype(float).unique()
    if len(cohorts) > 1:
        raise ValueError(
            "One forest figure cannot mix evaluation cohorts; filter or facet first."
        )
    if len(horizons) > 1:
        raise ValueError(
            "One forest figure cannot mix horizons; filter to one horizon first."
        )
    return (
        None if not len(cohorts) else str(cohorts[0]),
        None if not len(horizons) else float(horizons[0]),
    )


def _ordered_outcomes(
    estimates: pd.DataFrame,
    *,
    family_order: Sequence[str] | None,
    outcome_order: Sequence[str] | None,
) -> list[str]:
    metadata = estimates[["outcome", "outcome_family"]].drop_duplicates("outcome")
    if outcome_order:
        rank = {value: index for index, value in enumerate(outcome_order)}
        metadata["_outcome_rank"] = metadata["outcome"].map(rank).fillna(len(rank))
    else:
        metadata["_outcome_rank"] = metadata["outcome"].astype(str)
    if family_order:
        family_rank = {value: index for index, value in enumerate(family_order)}
        metadata["_family_rank"] = (
            metadata["outcome_family"].map(family_rank).fillna(len(family_rank))
        )
    else:
        metadata["_family_rank"] = metadata["outcome_family"].astype(str)
    return metadata.sort_values(
        ["_family_rank", "outcome_family", "_outcome_rank", "outcome"],
        kind="stable",
    )["outcome"].tolist()


def paginate_outcomes(
    estimates: pd.DataFrame,
    *,
    max_outcomes_per_page: int = 35,
    family_order: Sequence[str] | None = None,
    outcome_order: Sequence[str] | None = None,
) -> list[list[str]]:
    """Split a long atlas into pages while keeping families together when possible."""
    if max_outcomes_per_page < 1:
        raise ValueError("max_outcomes_per_page must be positive.")
    ordered = _ordered_outcomes(
        estimates, family_order=family_order, outcome_order=outcome_order
    )
    family_by_outcome = (
        estimates[["outcome", "outcome_family"]]
        .drop_duplicates("outcome")
        .set_index("outcome")["outcome_family"]
        .to_dict()
    )
    groups: list[list[str]] = []
    for outcome in ordered:
        if not groups or family_by_outcome[outcome] != family_by_outcome[groups[-1][0]]:
            groups.append([])
        groups[-1].append(outcome)
    pages: list[list[str]] = []
    current: list[str] = []
    for group in groups:
        while len(group) > max_outcomes_per_page:
            if current:
                pages.append(current)
                current = []
            pages.append(group[:max_outcomes_per_page])
            group = group[max_outcomes_per_page:]
        if current and len(current) + len(group) > max_outcomes_per_page:
            pages.append(current)
            current = []
        current.extend(group)
    if current:
        pages.append(current)
    return pages


def plot_outcome_performance_forest(
    estimates: pd.DataFrame,
    contrasts: pd.DataFrame | None = None,
    *,
    metric_label: str = "Cumulative/dynamic AUC",
    interval_label: str = "95% interval",
    model_order: Sequence[str] | None = None,
    contrast_order: Sequence[str] | None = None,
    family_order: Sequence[str] | None = None,
    outcome_order: Sequence[str] | None = None,
    higher_is_better: bool = True,
    absolute_reference: float | None = 0.5,
    absolute_xlim: tuple[float, float] | None = None,
    delta_xlim: tuple[float, float] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Draw aligned absolute-performance and paired-difference forest panels."""
    if estimates.empty:
        raise ValueError("Forest plot requires at least one absolute estimate.")
    estimates = _complete_schema(estimates, ESTIMATE_COLUMNS)
    contrasts = (
        pd.DataFrame(columns=CONTRAST_COLUMNS)
        if contrasts is None
        else _complete_schema(contrasts, CONTRAST_COLUMNS)
    )
    cohort, horizon = _validate_single_slice(estimates, contrasts)
    outcomes = _ordered_outcomes(
        estimates, family_order=family_order, outcome_order=outcome_order
    )
    estimates = estimates[estimates["outcome"].isin(outcomes)].copy()
    contrasts = contrasts[contrasts["outcome"].isin(outcomes)].copy()
    family_by_outcome = (
        estimates[["outcome", "outcome_family"]]
        .drop_duplicates("outcome")
        .set_index("outcome")["outcome_family"]
        .fillna("Other")
        .to_dict()
    )

    rows: list[tuple[str, str]] = []
    last_family = None
    for outcome in outcomes:
        family = str(family_by_outcome.get(outcome, "Other"))
        if family != last_family:
            rows.append(("family", family))
            last_family = family
        rows.append(("outcome", outcome))
    y_by_outcome = {
        value: index for index, (kind, value) in enumerate(rows) if kind == "outcome"
    }
    n_rows = len(rows)
    has_contrasts = not contrasts.empty
    width_ratios = [1.8, 1.35, 0.95] if has_contrasts else [2.2, 0.85]
    figure_height = max(4.8, 0.30 * n_rows + 2.1)
    fig, axes = plt.subplots(
        1,
        len(width_ratios),
        figsize=(10.2 if has_contrasts else 7.4, figure_height),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.08},
    )
    axes = np.atleast_1d(axes)
    ax_absolute = axes[0]
    ax_delta = axes[1] if has_contrasts else None
    ax_info = axes[-1]

    for axis in axes:
        for index, (kind, value) in enumerate(rows):
            if kind == "family":
                axis.axhspan(index - 0.45, index + 0.45, color="#F3F5F8", zorder=0)
                axis.axhline(index + 0.45, color=PALETTE["panel_border"], lw=0.6)

    models = list(model_order or estimates["model"].dropna().astype(str).unique())
    offsets = np.linspace(-0.18, 0.18, len(models)) if len(models) > 1 else [0.0]
    markers = ("o", "s", "D", "^", "v", "P")
    for model_index, model in enumerate(models):
        subset = estimates[estimates["model"].astype(str) == model]
        for _, row in subset.iterrows():
            y = y_by_outcome.get(str(row["outcome"]))
            if y is None or not np.isfinite(row["estimate"]):
                continue
            y = y + offsets[model_index]
            lower, upper = row["lower"], row["upper"]
            if np.isfinite(lower) and np.isfinite(upper):
                ax_absolute.plot(
                    [lower, upper], [y, y], color=model_color(model), lw=1.35, zorder=2
                )
            ax_absolute.scatter(
                row["estimate"],
                y,
                color=model_color(model),
                marker=markers[model_index % len(markers)],
                s=27,
                linewidths=0,
                label=model_label(model),
                zorder=3,
            )
    if absolute_reference is not None:
        ax_absolute.axvline(
            absolute_reference, color=PALETTE["diagonal"], ls="--", lw=0.8, zorder=1
        )
    has_absolute_intervals = bool(
        (estimates["lower"].notna() & estimates["upper"].notna()).any()
    )
    absolute_axis_label = metric_label
    if has_absolute_intervals:
        absolute_axis_label += f" ({interval_label})"
    ax_absolute.set_xlabel(absolute_axis_label, fontsize=LABEL_SIZE)
    ax_absolute.set_title("Absolute held-out performance")
    if absolute_xlim is not None:
        ax_absolute.set_xlim(*absolute_xlim)
    else:
        values = pd.concat(
            [estimates["estimate"], estimates["lower"], estimates["upper"]]
        ).dropna()
        if len(values):
            low = min(float(values.min()), absolute_reference or float(values.min()))
            high = float(values.max())
            pad = max(0.015, 0.08 * max(high - low, 0.05))
            ax_absolute.set_xlim(max(0.0, low - pad), min(1.0, high + pad))
    despine(ax_absolute, "x")

    handles, labels = ax_absolute.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        fig.legend(
            unique.values(), unique.keys(), loc="upper left",
            bbox_to_anchor=(0.30, 1 - 0.43 / figure_height),
            ncol=max(1, len(unique)), fontsize=LEGEND_SIZE,
        )

    if has_contrasts and ax_delta is not None:
        contrast_names = list(
            contrast_order or contrasts["contrast"].dropna().astype(str).unique()
        )
        contrast_offsets = (
            np.linspace(-0.15, 0.15, len(contrast_names))
            if len(contrast_names) > 1
            else [0.0]
        )
        for contrast_index, contrast in enumerate(contrast_names):
            subset = contrasts[contrasts["contrast"].astype(str) == contrast]
            color = CATEGORICAL[contrast_index % len(CATEGORICAL)]
            for _, row in subset.iterrows():
                y = y_by_outcome.get(str(row["outcome"]))
                if y is None or not np.isfinite(row["estimate"]):
                    continue
                y = y + contrast_offsets[contrast_index]
                if np.isfinite(row["lower"]) and np.isfinite(row["upper"]):
                    ax_delta.plot(
                        [row["lower"], row["upper"]],
                        [y, y],
                        color=color,
                        lw=1.35,
                        zorder=2,
                    )
                ax_delta.scatter(
                    row["estimate"], y, color=color, s=27, marker="s", zorder=3
                )
        ax_delta.axvline(0, color=PALETTE["zero_line"], ls="--", lw=0.8)
        ax_delta.set_xlabel(f"Paired difference ({interval_label})", fontsize=LABEL_SIZE)
        comparator_values = contrasts["comparator"].dropna().astype(str).unique()
        baseline_values = contrasts["baseline"].dropna().astype(str).unique()
        if len(comparator_values) == 1 and len(baseline_values) == 1:
            contrast_title = (
                f"{model_label(comparator_values[0])} − "
                f"{model_label(baseline_values[0])}"
            )
        else:
            contrast_title = "Model contrast"
        ax_delta.set_title(contrast_title)
        if delta_xlim is not None:
            ax_delta.set_xlim(*delta_xlim)
        else:
            bounds = pd.concat(
                [contrasts["estimate"], contrasts["lower"], contrasts["upper"]]
            ).dropna()
            if len(bounds):
                extent = max(float(np.abs(bounds).max()) * 1.12, 0.02)
                ax_delta.set_xlim(-extent, extent)
        direction = "Higher is better" if higher_is_better else "Lower is better"
        ax_delta.text(
            0.98,
            0.985,
            direction,
            transform=ax_delta.transAxes,
            ha="right",
            va="top",
            fontsize=NOTE_SIZE,
            color=PALETTE["ink_muted"],
        )
        despine(ax_delta, "x")

    labels = [
        value if kind == "family" else _display_outcome(value)
        for kind, value in rows
    ]
    ax_absolute.set_yticks(np.arange(n_rows))
    ax_absolute.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax_absolute.invert_yaxis()
    for tick, (kind, value) in zip(ax_absolute.get_yticklabels(), rows):
        if kind == "family":
            tick.set_fontweight("semibold")
            tick.set_color(_family_color(value))

    ax_info.set_xlim(0, 1)
    shows_probability = bool(
        has_contrasts and contrasts["probability_superior"].notna().any()
    )
    ax_info.set_title(
        "n / events / P(Δ>0)" if shows_probability else "n / events",
        fontsize=9,
    )
    ax_info.axis("off")
    annotation_source = contrasts if has_contrasts else estimates
    for outcome, y in y_by_outcome.items():
        subset = annotation_source[annotation_source["outcome"] == outcome]
        if subset.empty:
            continue
        row = subset.iloc[0]
        pieces = []
        if np.isfinite(row.get("n", np.nan)):
            pieces.append(f"{int(row['n']):,}")
        if np.isfinite(row.get("n_events", np.nan)):
            pieces.append(f"{int(row['n_events']):,}")
        if shows_probability and np.isfinite(row.get("probability_superior", np.nan)):
            pieces.append(f"{float(row['probability_superior']):.2f}")
        ax_info.text(
            0.02,
            y,
            " / ".join(pieces) if pieces else "—",
            va="center",
            ha="left",
            fontsize=ANNOT_SIZE,
            color=PALETTE["ink_secondary"],
        )
    ax_info.set_ylim(ax_absolute.get_ylim())

    descriptor = []
    if cohort:
        descriptor.append(cohort)
    if horizon is not None:
        descriptor.append(f"{int(round(horizon))}-day horizon")
    figure_title = title or "Predictability across hematologic outcomes"
    fig.suptitle(
        figure_title, y=1 - 0.12 / figure_height,
        fontsize=12, fontweight="semibold",
    )
    if descriptor:
        fig.text(
            0.30, 1 - 0.35 / figure_height, ", ".join(descriptor),
            ha="left", va="top", fontsize=NOTE_SIZE,
            color=PALETTE["ink_secondary"],
        )
    fig.subplots_adjust(
        left=0.30, right=0.98,
        top=1 - 1.20 / figure_height,
        bottom=0.48 / figure_height,
    )
    save_fig(fig, None if save_path is None else str(save_path))
    return fig
