"""Focused three-panel figure for the OPERA outcome-transfer experiment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.evaluation.outcome_transfer_aggregation import (
    SEVERITY_PRIMARY_SCOPE,
    resolve_severity_evaluation_targets,
)
from opera.visualization.style import (
    CATEGORICAL,
    LEGEND_SIZE,
    NOTE_SIZE,
    PALETTE,
    add_panel_label,
    despine,
    save_fig,
)


_REPRESENTATION_LABELS = {
    "dapt": "DAPT",
    "opera_no_g3": "No-G3 OPERA",
    "opera_no_transfusion_signal": "Target-signal-held-out OPERA",
    "opera_no_hospitalisation_signal": "Target-signal-held-out OPERA",
    "opera_no_infection_family": "Family-held-out OPERA",
    "opera_no_renal_family": "Family-held-out OPERA",
    "opera_no_cardiovascular_family": "Family-held-out OPERA",
    "opera_full": "Full OPERA",
}
_REPRESENTATION_COLORS = {
    "dapt": PALETTE["dapt"],
    "opera_no_g3": "#B07A10",
    "opera_no_transfusion_signal": "#B07A10",
    "opera_no_hospitalisation_signal": "#B07A10",
    "opera_no_infection_family": "#1B7A4A",
    "opera_no_renal_family": "#1B7A4A",
    "opera_no_cardiovascular_family": "#1B7A4A",
    "opera_full": PALETTE["opera"],
}


def _auroc_rows(results: pd.DataFrame, condition: str) -> pd.DataFrame:
    required = {
        "condition",
        "comparison_condition",
        "target_outcome",
        "metric",
        "value",
        "evaluation_level",
        "evaluation_group",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Transfer results are missing columns: {sorted(missing)}")
    return results.loc[
        (results["comparison_condition"].astype(str) == condition)
        & (results["metric"].astype(str) == "auroc")
        & (results["evaluation_level"].astype(str) == "pan_hematology")
        & (results["evaluation_group"].astype(str) == "all_hematology")
    ].copy()


def _draw_representation_ladder(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    representations: list[str],
    title: str,
    note: str,
    target_order: Optional[list[str]] = None,
) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "No supported held-out results", ha="center", va="center")
        ax.set_title(title)
        return
    if target_order is None:
        target_order = sorted(frame["target_outcome"].astype(str).unique().tolist())
    target_order = [
        target
        for target in target_order
        if target in set(frame["target_outcome"].astype(str))
    ]
    if not target_order:
        ax.text(0.5, 0.5, "No supported held-out results", ha="center", va="center")
        ax.set_title(title)
        return
    y_locations = {target: index for index, target in enumerate(target_order)}
    offsets = np.linspace(-0.22, 0.22, len(representations))
    for offset, representation in zip(offsets, representations):
        subset = frame.loc[frame["condition"].astype(str) == representation].copy()
        if subset.empty:
            continue
        # Seeds are repeated measurements; show outcome-specific seed means.
        average = subset.groupby("target_outcome", as_index=False)["value"].mean()
        average = average[average["target_outcome"].astype(str).isin(y_locations)]
        y = np.asarray([y_locations[str(value)] for value in average["target_outcome"]])
        ax.scatter(
            average["value"],
            y + offset,
            s=29,
            color=_REPRESENTATION_COLORS.get(representation, CATEGORICAL[0]),
            edgecolor="white",
            linewidth=0.45,
            label=_REPRESENTATION_LABELS.get(representation, representation),
            zorder=3,
        )
        if not average.empty:
            aggregate = float(average["value"].mean())
            ax.scatter(
                aggregate,
                len(target_order) + 0.40 + offset,
                marker="D",
                s=43,
                color=_REPRESENTATION_COLORS.get(representation, CATEGORICAL[0]),
                edgecolor=PALETTE["ink"],
                linewidth=0.45,
                zorder=4,
            )
    ax.set_yticks(list(range(len(target_order) + 1)))
    ax.set_yticklabels([*target_order, "Macro target mean"])
    ax.set_xlabel("Held-out ROC-AUC")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.text(
        0.01,
        0.01,
        note,
        transform=ax.transAxes,
        fontsize=NOTE_SIZE,
        color=PALETTE["ink_muted"],
        ha="left",
        va="bottom",
    )
    despine(ax, grid_axis="x")


def _draw_severity_delta_panel(
    ax: plt.Axes,
    deltas: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
) -> None:
    """Draw only the resolver-matched Grade 2/3 primary severity contrast."""
    required = {
        "comparison_condition",
        "target_outcome",
        "contrast",
        "metric",
        "estimate",
        "evaluation_level",
        "evaluation_group",
    }
    missing = required - set(deltas.columns)
    if missing:
        raise ValueError(f"Transfer deltas are missing columns: {sorted(missing)}")
    pairs, _ = resolve_severity_evaluation_targets(plan)
    target_order = [pair["target_outcome"] for pair in pairs]
    work = deltas.loc[
        (deltas["comparison_condition"].astype(str) == "opera_no_g3")
        & deltas["target_outcome"].astype(str).isin(target_order)
        & (deltas["metric"].astype(str) == "auroc")
        & (deltas["evaluation_level"].astype(str) == "pan_hematology")
        & (deltas["evaluation_group"].astype(str) == "all_hematology")
        & deltas["contrast"]
        .astype(str)
        .isin(["transfer_vs_dapt", "full_vs_transfer", "full_vs_dapt"])
    ].copy()
    if "evaluation_role" in work.columns and not work.empty:
        if set(work["evaluation_role"].astype(str)) != {SEVERITY_PRIMARY_SCOPE}:
            raise ValueError(
                "Severity figure received rows outside the resolved matched G2/G3 primary scope."
            )
    if work.empty:
        ax.text(
            0.5,
            0.5,
            "No supported matched Grade 2/3 held-out results",
            ha="center",
            va="center",
        )
        ax.set_title("Severity transfer (matched Grade 2/3 targets)")
        return

    y_locations = {target: index for index, target in enumerate(target_order)}
    contrast_specs = (
        ("transfer_vs_dapt", "No-G3 OPERA - DAPT", "#B07A10", -0.22),
        ("full_vs_transfer", "Full OPERA - No-G3 OPERA", PALETTE["opera"], 0.0),
        ("full_vs_dapt", "Full OPERA - DAPT", CATEGORICAL[5], 0.22),
    )
    for contrast, label, color, offset in contrast_specs:
        subset = work.loc[work["contrast"].astype(str) == contrast]
        per_target = subset.groupby("target_outcome", as_index=False)["estimate"].mean()
        per_target = per_target.loc[
            per_target["target_outcome"].astype(str).isin(y_locations)
        ].copy()
        if per_target.empty:
            continue
        y = np.asarray(
            [
                y_locations[str(target)] + offset
                for target in per_target["target_outcome"]
            ]
        )
        ax.scatter(
            per_target["estimate"],
            y,
            s=25,
            alpha=0.70,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            float(per_target["estimate"].mean()),
            len(target_order) + 0.40 + offset,
            marker="D",
            s=46,
            color=color,
            edgecolor=PALETTE["ink"],
            linewidth=0.45,
            label=label,
            zorder=4,
        )
    ax.axvline(0.0, color=PALETTE["zero_line"], linewidth=0.8, zorder=1)
    ax.set_yticks(list(range(len(target_order) + 1)))
    ax.set_yticklabels([*target_order, "Macro matched-target mean"])
    ax.set_xlabel("Paired Delta ROC-AUC")
    ax.set_title("Severity transfer (matched Grade 2/3 targets)")
    ax.text(
        0.01,
        0.01,
        "Primary only: resolver-matched Grade 3+ targets with their Grade 2+ "
        "counterpart retained in No-G3 training. Points: outcome means across "
        "seeds; diamonds: macro matched-target means. All three paired "
        "representation contrasts are shown.",
        transform=ax.transAxes,
        fontsize=NOTE_SIZE,
        color=PALETTE["ink_muted"],
        ha="left",
        va="bottom",
    )
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    despine(ax, grid_axis="x")


def _draw_family_delta_panel(ax: plt.Axes, deltas: pd.DataFrame) -> None:
    required = {
        "comparison_condition",
        "target_family",
        "target_outcome",
        "contrast",
        "metric",
        "estimate",
        "evaluation_level",
        "evaluation_group",
    }
    missing = required - set(deltas.columns)
    if missing:
        raise ValueError(f"Transfer deltas are missing columns: {sorted(missing)}")
    family_conditions = (
        "opera_no_infection_family",
        "opera_no_renal_family",
        "opera_no_cardiovascular_family",
    )
    work = deltas.loc[
        deltas["comparison_condition"].astype(str).isin(family_conditions)
        & (deltas["metric"].astype(str) == "auroc")
        & (deltas["evaluation_level"].astype(str) == "pan_hematology")
        & (deltas["evaluation_group"].astype(str) == "all_hematology")
        & deltas["contrast"].astype(str).isin(["transfer_vs_dapt", "full_vs_transfer"])
    ].copy()
    if work.empty:
        ax.text(
            0.5, 0.5, "No supported held-out family results", ha="center", va="center"
        )
        ax.set_title("Family transfer")
        return
    families = [
        "Infection",
        "Renal toxicity",
        "Cardiovascular & thrombotic",
    ]
    available = set(work["target_family"].astype(str))
    families = [family for family in families if family in available]
    y_locations = {family: index for index, family in enumerate(families)}
    contrast_specs = (
        ("transfer_vs_dapt", "Family-held-out OPERA - DAPT", "#1B7A4A", -0.16),
        ("full_vs_transfer", "Full OPERA - family-held-out", PALETTE["opera"], 0.16),
    )
    for contrast, label, color, offset in contrast_specs:
        subset = work.loc[work["contrast"].astype(str) == contrast]
        for family, group in subset.groupby("target_family", sort=False):
            if str(family) not in y_locations:
                continue
            per_target = group.groupby("target_outcome", as_index=False)[
                "estimate"
            ].mean()
            y = np.full(len(per_target), y_locations[str(family)] + offset)
            ax.scatter(
                per_target["estimate"],
                y,
                s=25,
                alpha=0.70,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            if not per_target.empty:
                ax.scatter(
                    float(per_target["estimate"].mean()),
                    y_locations[str(family)] + offset,
                    marker="D",
                    s=46,
                    color=color,
                    edgecolor=PALETTE["ink"],
                    linewidth=0.45,
                    label=label if family == families[0] else None,
                    zorder=4,
                )
    ax.axvline(0.0, color=PALETTE["zero_line"], linewidth=0.8, zorder=1)
    ax.set_yticks(list(range(len(families))), labels=families)
    ax.set_xlabel("Paired Delta ROC-AUC")
    ax.set_title("Family transfer")
    ax.text(
        0.01,
        0.01,
        "Points: outcome means across seeds; diamonds: macro outcome means",
        transform=ax.transAxes,
        fontsize=NOTE_SIZE,
        color=PALETTE["ink_muted"],
        ha="left",
        va="bottom",
    )
    ax.legend(loc="best", fontsize=LEGEND_SIZE)
    despine(ax, grid_axis="x")


def plot_outcome_transfer_figure(
    results: pd.DataFrame,
    deltas: pd.DataFrame,
    *,
    plan: Mapping[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render severity, related-outcome, and family transfer panels.

    The figure uses only measured held-out output rows.  Missing/unsupported
    cells are labelled as such instead of being imputed or visually filled.
    """
    matched_pairs, _ = resolve_severity_evaluation_targets(plan)
    figure_height = max(5.4, 2.7 + 0.28 * len(matched_pairs))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(20.0, figure_height),
        layout="constrained",
    )

    _draw_severity_delta_panel(axes[0], deltas, plan=plan)

    related = pd.concat(
        [
            _auroc_rows(results, "opera_no_transfusion_signal"),
            _auroc_rows(results, "opera_no_hospitalisation_signal"),
        ],
        ignore_index=True,
    )
    _draw_representation_ladder(
        axes[1],
        related,
        representations=[
            "dapt",
            "opera_no_transfusion_signal",
            "opera_no_hospitalisation_signal",
            "opera_full",
        ],
        title="Related-outcome transfer",
        note="Separate ladders: any_transfusion and hospitalisation; no pooled proxy score.",
        target_order=["any_transfusion", "hospitalisation"],
    )
    _draw_family_delta_panel(axes[2], deltas)

    for label, axis in zip(("A", "B", "C"), axes):
        add_panel_label(axis, label, x=-0.11, y=1.06)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=LEGEND_SIZE)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="best", fontsize=LEGEND_SIZE)
    figure.suptitle(
        "OPERA outcome-guided representation transfer on locked held-out patients",
        y=1.01,
    )
    save_fig(figure, save_path)
    return figure


def write_outcome_transfer_figure(
    output_dir: str | Path,
    *,
    results: pd.DataFrame,
    deltas: pd.DataFrame,
    plan: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the required PNG and PDF figure without fabricating any results."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / "outcome_transfer.png"
    plot_outcome_transfer_figure(results, deltas, plan=plan, save_path=str(png_path))
    return {"png": png_path, "pdf": png_path.with_suffix(".pdf")}
