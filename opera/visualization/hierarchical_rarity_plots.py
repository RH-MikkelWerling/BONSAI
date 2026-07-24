"""Publication figures for the Bayesian natural-rarity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

from opera.visualization.style import (
    ANNOT_SIZE,
    CATEGORICAL,
    FIG_FULL,
    NOTE_SIZE,
    PALETTE,
    TITLE_SIZE,
    save_fig,
    setup_style,
)


METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "pr_skill": "PR skill",
    "brier_score": "Brier score",
    "brier_skill": "Brier skill",
    "log_loss": "log loss",
}

OTHER_COURSE_GROUP = "Other cohorts"


def aggregate_scatter_cells(
    deltas: pd.DataFrame,
    *,
    metric: str,
    rarity_column: str = "n_events_train",
) -> pd.DataFrame:
    """Collapse seed-level deltas to one transparent raw point per task cell."""
    data = deltas[deltas["metric"] == metric].copy()
    required = {
        "cell_id",
        "cohort",
        "outcome",
        "difference",
        "difference_se",
        rarity_column,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Scatter input is missing columns: {sorted(missing)}")
    data["outcome_family"] = data.get(
        "outcome_family", pd.Series("Other", index=data.index)
    ).fillna("Other")
    data["cohort_group"] = data.get("cohort_group", data["cohort"]).fillna(
        data["cohort"]
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        values = group["difference"].to_numpy(dtype=float)
        ses = group["difference_se"].to_numpy(dtype=float)
        n_seed = len(group)
        within = float(np.nansum(ses**2) / max(n_seed**2, 1))
        between = float(np.nanvar(values, ddof=1) / n_seed) if n_seed > 1 else 0.0
        return pd.Series(
            {
                "cohort": group["cohort"].iloc[0],
                "cohort_group": group["cohort_group"].iloc[0],
                "outcome": group["outcome"].iloc[0],
                "outcome_family": group["outcome_family"].iloc[0],
                rarity_column: group[rarity_column].iloc[0],
                "difference": float(np.nanmean(values)),
                "display_se": float(np.sqrt(within + between)),
                "n_seeds": n_seed,
                "n_test_positive": group.get(
                    "n_test_positive", pd.Series(np.nan, index=group.index)
                ).iloc[0],
                "n_test_negative": group.get(
                    "n_test_negative", pd.Series(np.nan, index=group.index)
                ).iloc[0],
                "analysis_tier": (
                    "primary"
                    if (group.get("analysis_tier", "primary") == "primary").all()
                    else "partial_pool_only"
                ),
            }
        )

    return (
        data.groupby("cell_id", as_index=False, sort=True)
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
    )


# Held-out event counts are not used to size points: under the fixed
# train/val/test split used to build each cell, they scale near-linearly with
# the training-event rarity already shown on the x-axis, so a size encoding
# would just redraw the x-position as area. All cells use one size; fill marks
# analysis tier, while shape/color are reserved for cohort and outcome family.
POINT_SIZE = 30.0

# Cohort-group marker shapes, cycled if more groups appear than symbols listed.
COHORT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "8"]

# Fixed order so disease-course legend columns are stable across runs; any
# course name not listed here (including the "Other cohorts" fallback) is
# appended in first-seen order.
COURSE_ORDER = [
    "Aggressive course group",
    "Indolent / chronic course group",
    "Plasma-cell group",
]


def _titlecase_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _ordered_values(
    values: Sequence[str], preferred: Optional[Sequence[str]]
) -> list[str]:
    """Return present values in a stable prespecified order, then alphabetically."""
    present = set(map(str, values))
    ordered = [str(value) for value in (preferred or []) if str(value) in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _selected_labels(
    cells: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    rarity_column: str,
    max_labels: int,
    highlight_cells: Optional[Sequence[Mapping[str, str]]] = None,
    highlight_outcomes: Optional[list] = None,
) -> pd.DataFrame:
    """Pick a small, curated set of cells to annotate, each tagged with why.

    Statistical extremeness (e.g. "biggest residual from the curve") tends to
    surface whichever cell happens to be noisiest, not whichever cell a
    clinical reader cares about. So the priority order is: (1) exact,
    prespecified cohort-outcome anchors; (2) prespecified outcomes spread
    across the information range; (3) rarest and most data-rich cells for
    scale context; and (4) typical cells close to the fitted curve.
    """
    if cells.empty or max_labels <= 0:
        return cells.iloc[0:0]
    expected = np.interp(
        cells[rarity_column].to_numpy(dtype=float),
        curve["training_events"].to_numpy(dtype=float),
        curve["median"].to_numpy(dtype=float),
    )
    working = cells.assign(
        residual=cells["difference"].to_numpy(dtype=float) - expected
    )
    used_ids: set = set()
    picks: list[dict] = []

    def take(column: str, *, largest: bool, category: str) -> None:
        # Re-rank the still-unused rows each time, rather than pre-ranking the
        # whole set: if the globally most-extreme row was already claimed by
        # an earlier category, this falls through to the next-best distinct
        # cell instead of silently dropping the category.
        remaining = working[~working["cell_id"].isin(used_ids)]
        if remaining.empty:
            return
        ranked = (
            remaining.nlargest(1, column) if largest else remaining.nsmallest(1, column)
        )
        row = ranked.iloc[0].to_dict()
        row["label_category"] = category
        used_ids.add(row["cell_id"])
        picks.append(row)

    # Keep two slots for the rarity-range anchors whenever the label budget
    # permits; clinical anchors still receive all remaining priority slots.
    clinical_budget = max_labels - min(2, max(0, max_labels - 1))

    # Exact cohort-outcome anchors are selected before any data-dependent
    # fallback. Optional display text keeps journal wording out of task IDs.
    for rule in highlight_cells or []:
        if len(picks) >= clinical_budget:
            break
        cohort = rule.get("cohort")
        outcome = rule.get("outcome")
        if cohort is None or outcome is None:
            continue
        pool = working[
            (working["cohort"].astype(str) == str(cohort))
            & (working["outcome"].astype(str) == str(outcome))
            & (~working["cell_id"].isin(used_ids))
        ]
        if pool.empty:
            continue
        row = pool.nlargest(1, rarity_column).iloc[0].to_dict()
        row["label_category"] = str(rule.get("category", "Clinical anchor"))
        row["annotation_label"] = str(
            rule.get(
                "label",
                f"{cohort} × {_titlecase_first(str(outcome).replace('_', ' '))}",
            )
        )
        used_ids.add(row["cell_id"])
        picks.append(row)

    # Spread key-outcome picks across the x-range by construction, rather than
    # always taking each outcome's single most data-rich cell: common outcomes
    # tend to have most of their cells clustered at the high-n end of a log
    # rarity axis, so "most data-rich instance" for every highlighted outcome
    # would pile all of their labels on top of each other in the same corner.
    present_outcomes = [
        outcome
        for outcome in dict.fromkeys(highlight_outcomes or [])
        if outcome in set(working["outcome"])
    ]
    if present_outcomes:
        log_rarity = np.log(working[rarity_column].to_numpy(dtype=float))
        log_min, log_max = float(log_rarity.min()), float(log_rarity.max())
        n_present = len(present_outcomes)
        targets = (
            np.linspace(log_min, log_max, n_present)
            if n_present > 1
            else np.array([(log_min + log_max) / 2.0])
        )
        for outcome, target_log in zip(present_outcomes, targets):
            if len(picks) >= clinical_budget:
                break
            pool = working[
                (working["outcome"] == outcome) & (~working["cell_id"].isin(used_ids))
            ]
            if pool.empty:
                continue
            pool_log = np.log(pool[rarity_column].to_numpy(dtype=float))
            row = pool.iloc[int(np.argmin(np.abs(pool_log - target_log)))].to_dict()
            row["label_category"] = "Key outcome"
            used_ids.add(row["cell_id"])
            picks.append(row)

    if len(picks) < max_labels:
        take(rarity_column, largest=False, category="Rarest evaluable")
    if len(picks) < max_labels:
        take(rarity_column, largest=True, category="Most data-rich")

    budget = max_labels - len(picks)
    remaining = working[~working["cell_id"].isin(used_ids)]
    if budget > 0 and not remaining.empty:
        log_x = np.log(remaining[rarity_column].to_numpy(dtype=float))
        edges = np.linspace(log_x.min(), log_x.max(), budget + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            bucket = remaining[
                (np.log(remaining[rarity_column]) >= lo)
                & (np.log(remaining[rarity_column]) <= hi)
                & (~remaining["cell_id"].isin(used_ids))
            ]
            if bucket.empty:
                continue
            row = bucket.loc[bucket["residual"].abs().idxmin()].to_dict()
            row["label_category"] = "Representative"
            used_ids.add(row["cell_id"])
            picks.append(row)

    if not picks:
        return cells.iloc[0:0]
    return pd.DataFrame(picks).head(max_labels)


def _label_placement(
    category: str,
    x: float,
    y: float,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    index: int,
) -> dict:
    """Choose a leader-line text anchor in relatively open plot space."""
    y_span = y_max - y_min
    if category == "Rarest evaluable":
        return dict(
            xytext=(x_min * 1.35, y_max - 0.05 * y_span),
            ha="left",
            va="top",
            textcoords="data",
        )
    if category == "Most data-rich":
        return dict(
            xytext=(x_max / 1.35, y_max - 0.14 * y_span),
            ha="right",
            va="top",
            textcoords="data",
        )
    # Local callouts point inward from the plot edges. ``index`` is the rank
    # within that side's lane, so crowded high-information cells are staggered
    # rather than all receiving the same offset.
    x_fraction = np.log(x / x_min) / np.log(x_max / x_min)
    dx = 18 if x_fraction <= 0.55 else -18
    magnitude = 16 + 10 * (index // 2)
    dy = magnitude if index % 2 == 0 else -magnitude
    if y > y_min + 0.86 * y_span and dy > 0:
        dy = -dy
    if y < y_min + 0.14 * y_span and dy < 0:
        dy = -dy
    return dict(
        xytext=(dx, dy),
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        textcoords="offset points",
    )


def _course_group(cohort_group: str, course_groups: Optional[Mapping[str, str]]) -> str:
    if not course_groups:
        return OTHER_COURSE_GROUP
    return course_groups.get(cohort_group, OTHER_COURSE_GROUP)


def _wrap_label(label: str, max_chars: int = 11) -> str:
    """Break a long legend label onto a second line at the nearest space to
    its midpoint, so a narrow legend column doesn't force the figure to
    either overflow or shrink everything else to fit the longest word."""
    if len(label) <= max_chars or " " not in label:
        return label
    mid = len(label) / 2
    best_space = min(
        (i for i, ch in enumerate(label) if ch == " "), key=lambda i: abs(i - mid)
    )
    return label[:best_space] + "\n" + label[best_space + 1 :]


_DENSE_LEGEND_SIZE = 8.0
_DENSE_TITLE_SIZE = 8.5


def _draw_unified_legend(
    fig: plt.Figure,
    *,
    family_handles: list,
    cohort_handles_by_course: "dict[str, list]",
    low_info_handle,
    interval_handles: list,
    top_y: float,
) -> None:
    """Draw one bordered legend row: family | cohort (by course) | low-info | model."""
    from matplotlib.patches import FancyBboxPatch

    panel_left, panel_right = 0.01, 0.99
    panel_top, panel_bottom = top_y, 0.02
    fig.add_artist(
        FancyBboxPatch(
            (panel_left, panel_bottom),
            panel_right - panel_left,
            panel_top - panel_bottom,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=PALETTE["panel_border"],
            linewidth=0.8,
            zorder=0,
        )
    )

    n_courses = max(1, len(cohort_handles_by_course))
    # Family stays a single column (its 8 entries are long clinical phrases;
    # a 2-up layout would need to double the column width to avoid
    # overlapping the next column, which costs more than the row count it
    # saves). Cohort and model labels wrap at _wrap_label's threshold
    # instead, so those columns can stay narrow without truncating text.
    family_x0, family_x1 = 0.02, 0.245
    cohort_x0, cohort_x1 = 0.265, 0.60
    lowinfo_x0, lowinfo_x1 = 0.62, 0.80
    model_x0, model_x1 = 0.82, 0.98
    for x in (family_x1 + 0.005, cohort_x1 + 0.005, lowinfo_x1 + 0.005):
        fig.add_artist(
            Line2D(
                [x, x],
                [panel_bottom + 0.015, panel_top - 0.015],
                transform=fig.transFigure,
                color=PALETTE["panel_border"],
                linewidth=0.8,
                zorder=1,
            )
        )

    title_y = panel_top - 0.025
    fig.legend(
        handles=family_handles,
        title="Outcome family (color)",
        loc="upper left",
        bbox_to_anchor=(family_x0, title_y),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=False,
        fontsize=_DENSE_LEGEND_SIZE,
        title_fontsize=_DENSE_TITLE_SIZE,
        handletextpad=0.45,
        labelspacing=0.62,
    )

    fig.text(
        cohort_x0,
        title_y,
        "Training cohort group (shape)",
        transform=fig.transFigure,
        fontsize=_DENSE_TITLE_SIZE,
        fontweight="semibold",
        color=PALETTE["ink"],
        ha="left",
        va="top",
    )
    course_width = (cohort_x1 - cohort_x0) / n_courses
    for index, (course, handles) in enumerate(cohort_handles_by_course.items()):
        for h in handles:
            h.set_label(_wrap_label(h.get_label()))
        wrapped_title = _wrap_label(course, max_chars=14)
        fig.legend(
            handles=handles,
            title=wrapped_title,
            loc="upper left",
            bbox_to_anchor=(cohort_x0 + index * course_width, title_y - 0.05),
            bbox_transform=fig.transFigure,
            ncol=1,
            frameon=False,
            fontsize=_DENSE_LEGEND_SIZE,
            title_fontsize=_DENSE_LEGEND_SIZE,
            handletextpad=0.45,
            labelspacing=0.6,
        )

    fig.legend(
        handles=[low_info_handle],
        title="Low-information cells",
        loc="upper left",
        bbox_to_anchor=(lowinfo_x0, title_y),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=_DENSE_LEGEND_SIZE,
        title_fontsize=_DENSE_TITLE_SIZE,
        handletextpad=0.6,
        labelspacing=1.0,
    )

    def _rewrap(handle):
        handle.set_label(_wrap_label(handle.get_label(), max_chars=13))
        return handle

    fig.legend(
        handles=[_rewrap(h) for h in interval_handles],
        title="Model components",
        loc="upper left",
        bbox_to_anchor=(model_x0, title_y),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=_DENSE_LEGEND_SIZE,
        title_fontsize=_DENSE_TITLE_SIZE,
        labelspacing=0.7,
    )
    _ = (
        family_x1,
        lowinfo_x1,
        model_x1,
    )  # bounds documented above, kept for future tuning


def plot_hierarchical_rarity_curve(
    deltas: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    metric: str = "auroc",
    rarity_column: str = "n_events_train",
    model_label: str = "OPERA",
    comparator_label: str = "XGBoost",
    title: str = "Model benefit across natural task information",
    show_predictive_interval: bool = True,
    max_labels: int = 7,
    cohort_course_groups: Optional[Mapping[str, str]] = None,
    cohort_group_labels: Optional[Mapping[str, str]] = None,
    outcome_family_order: Optional[Sequence[str]] = None,
    highlight_cells: Optional[Sequence[Mapping[str, str]]] = None,
    highlight_outcomes: Optional[list] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Draw raw task deltas, posterior curve, and distinct uncertainty bands."""
    setup_style()
    cells = aggregate_scatter_cells(
        deltas,
        metric=metric,
        rarity_column=rarity_column,
    )
    cells = cells.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[rarity_column, "difference"]
    )
    cells = cells[cells[rarity_column] > 0].copy()
    if cells.empty:
        raise ValueError("No finite cells are available for the rarity figure.")
    required_curve = {
        "training_events",
        "median",
        "lower_50",
        "upper_50",
        "lower_95",
        "upper_95",
    }
    missing = required_curve.difference(curve.columns)
    if missing:
        raise ValueError(f"Posterior curve is missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(FIG_FULL[0] * 1.25, FIG_FULL[1] * 1.28))
    x_min = min(cells[rarity_column].min(), curve["training_events"].min())
    x_max = max(cells[rarity_column].max(), curve["training_events"].max())
    y_values = np.concatenate(
        [cells["difference"].to_numpy(), curve["lower_95"], curve["upper_95"]]
    )
    y_pad = max(0.015, 0.08 * np.ptp(y_values))
    y_min, y_max = (
        float(np.nanmin(y_values) - y_pad),
        float(np.nanmax(y_values) + y_pad),
    )
    ax.axhspan(0.0, y_max, color="#EEF1FA", alpha=0.42, zorder=0)
    ax.axhspan(y_min, 0.0, color="#F4F4F4", alpha=0.58, zorder=0)

    x_curve = curve["training_events"].to_numpy(dtype=float)
    if show_predictive_interval and {
        "predictive_lower_95",
        "predictive_upper_95",
    }.issubset(curve.columns):
        ax.fill_between(
            x_curve,
            curve["predictive_lower_95"],
            curve["predictive_upper_95"],
            color="#AEB8D8",
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
    ax.fill_between(
        x_curve,
        curve["lower_95"],
        curve["upper_95"],
        color="#5264A8",
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.fill_between(
        x_curve,
        curve["lower_50"],
        curve["upper_50"],
        color="#2D3A8C",
        alpha=0.25,
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        x_curve,
        curve["median"],
        color="#1A237E",
        linewidth=2.25,
        zorder=5,
    )

    families = _ordered_values(
        cells["outcome_family"].astype(str).unique(), outcome_family_order
    )
    colors = {
        family: CATEGORICAL[index % len(CATEGORICAL)]
        for index, family in enumerate(families)
    }
    cohort_groups = sorted(cells["cohort_group"].astype(str).unique())
    markers = {
        group: COHORT_MARKERS[index % len(COHORT_MARKERS)]
        for index, group in enumerate(cohort_groups)
    }
    for (family, cohort_group), group in cells.groupby(
        ["outcome_family", "cohort_group"], sort=True
    ):
        marker = markers[str(cohort_group)]
        primary = group[group["analysis_tier"] == "primary"]
        pooled = group[group["analysis_tier"] != "primary"]
        if not primary.empty:
            ax.scatter(
                primary[rarity_column],
                primary["difference"],
                s=POINT_SIZE,
                marker=marker,
                c=colors[str(family)],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.8,
                rasterized=True,
                zorder=4,
            )
        if not pooled.empty:
            ax.scatter(
                pooled[rarity_column],
                pooled["difference"],
                s=POINT_SIZE,
                marker=marker,
                facecolors="white",
                edgecolors=colors[str(family)],
                linewidths=0.9,
                alpha=0.65,
                rasterized=True,
                zorder=4,
            )

    labels = _selected_labels(
        cells,
        curve,
        rarity_column=rarity_column,
        max_labels=max_labels,
        highlight_cells=highlight_cells,
        highlight_outcomes=highlight_outcomes,
    )
    lane_counts = {"left": 0, "right": 0}
    adjustable_texts = []
    target_x: list[float] = []
    target_y: list[float] = []
    for _, row in labels.iterrows():
        outcome_display = _titlecase_first(str(row["outcome"]).replace("_", " "))
        cell_label = row.get("annotation_label")
        if pd.isna(cell_label):
            cell_label = f"{row['cohort']} × {outcome_display}"
        text = (
            str(cell_label)
            if row["label_category"] == "Clinical anchor"
            else f"{row['label_category']}:\n{cell_label}"
        )
        x_fraction = np.log(float(row[rarity_column]) / x_min) / np.log(x_max / x_min)
        lane = "left" if x_fraction <= 0.55 else "right"
        lane_index = lane_counts[lane]
        lane_counts[lane] += 1
        point_x = float(row[rarity_column])
        point_y = float(row["difference"])
        if row["label_category"] in {"Rarest evaluable", "Most data-rich"}:
            placement = _label_placement(
                row["label_category"],
                point_x,
                point_y,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                index=lane_index,
            )
            ax.annotate(
                text,
                xy=(point_x, point_y),
                fontsize=ANNOT_SIZE,
                color=PALETTE["ink_secondary"],
                zorder=7,
                arrowprops=dict(
                    arrowstyle="-",
                    color=PALETTE["connector"],
                    linewidth=0.6,
                    shrinkA=1.5,
                    shrinkB=3.5,
                ),
                **placement,
            )
            continue

        # Alternate between separated vertical rails on each half of the plot.
        # This gives adjustText a collision-free starting layout even when
        # several clinically important cells occupy the same dense region.
        rail_fractions = (0.68, 0.34, 0.52, 0.82, 0.18)
        text_x = point_x * (1.12 if lane == "left" else 0.88)
        text_y = y_min + rail_fractions[lane_index % len(rail_fractions)] * (
            y_max - y_min
        )
        adjustable_texts.append(
            ax.text(
                text_x,
                text_y,
                text,
                fontsize=ANNOT_SIZE,
                color=PALETTE["ink_secondary"],
                ha="left" if lane == "left" else "right",
                va="center",
                zorder=7,
            )
        )
        target_x.append(point_x)
        target_y.append(point_y)

    rug_y = y_min + 0.018 * (y_max - y_min)
    ax.scatter(
        cells[rarity_column],
        np.full(len(cells), rug_y),
        marker="|",
        s=18,
        color=PALETTE["ink_muted"],
        alpha=0.3,
        linewidths=0.6,
        rasterized=True,
        zorder=2,
    )
    ax.axhline(0.0, color=PALETTE["zero_line"], linewidth=0.85, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(x_min / 1.12, x_max * 1.12)
    ax.set_ylim(y_min, y_max)
    tick_candidates = np.array(
        [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000]
    )
    ticks = tick_candidates[
        (tick_candidates >= x_min / 1.1) & (tick_candidates <= x_max * 1.1)
    ]
    if len(ticks) < 3:
        ticks = np.unique(np.round(np.geomspace(x_min, x_max, 5)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t):,}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Training events (log scale)", labelpad=8)
    metric_name = METRIC_LABELS.get(metric, metric.replace("_", " ").upper())
    ax.set_ylabel(
        f"Paired Δ{metric_name} ({model_label} − {comparator_label})", labelpad=8
    )
    ax.set_title(title, loc="left", pad=14, fontsize=TITLE_SIZE)
    ax.text(
        0.995,
        0.985,
        f"Positive values favour {model_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=NOTE_SIZE,
        fontstyle="italic",
        color=PALETTE["opera_joint"],
    )
    ax.grid(axis="x", which="minor", visible=False)
    ax.margins(x=0)

    # The curated set is small, so a deterministic final repulsion pass is
    # cheap and protects against collisions caused by the realized data. The
    # heuristic positions above remain the dependency-free fallback.
    try:
        from adjustText import adjust_text

        random_state = np.random.get_state()
        np.random.seed(2026)
        try:
            adjust_text(
                adjustable_texts,
                target_x=target_x,
                target_y=target_y,
                ax=ax,
                expand=(1.08, 1.18),
                force_text=(0.25, 0.45),
                force_static=(0.08, 0.16),
                max_move=(24, 32),
                ensure_inside_axes=True,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#999999",
                    linewidth=0.6,
                ),
            )
        finally:
            np.random.set_state(random_state)
    except ImportError:
        pass

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[family],
            markeredgecolor="white",
            markersize=6,
            label=family,
        )
        for family in families
    ]
    cohort_handles_by_course: dict = {}
    for group in cohort_groups:
        course = _course_group(group, cohort_course_groups)
        cohort_handles_by_course.setdefault(course, []).append(
            Line2D(
                [0],
                [0],
                marker=markers[group],
                linestyle="none",
                markerfacecolor=PALETTE["ink_muted"],
                markeredgecolor="white",
                markersize=6,
                label=(cohort_group_labels or {}).get(group, group),
            )
        )
    ordered_courses = [c for c in COURSE_ORDER if c in cohort_handles_by_course]
    ordered_courses += [c for c in cohort_handles_by_course if c not in ordered_courses]
    cohort_handles_by_course = {c: cohort_handles_by_course[c] for c in ordered_courses}

    low_info_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=PALETTE["ink_muted"],
        markersize=6,
        label="Hollow markers: <10 held-out\ncases or controls",
    )
    interval_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE["opera_joint"],
            linewidth=2.2,
            label="Posterior median",
        ),
        Patch(facecolor=PALETTE["opera"], alpha=0.25, label="50% credible interval"),
        Patch(facecolor="#5264A8", alpha=0.18, label="95% credible interval"),
    ]
    if show_predictive_interval:
        interval_handles.append(
            Patch(facecolor="#AEB8D8", alpha=0.14, label="95% prediction interval")
        )

    top_margin = 0.91
    bottom_margin = 0.36
    fig.subplots_adjust(left=0.075, right=0.98, top=top_margin, bottom=bottom_margin)
    _draw_unified_legend(
        fig,
        family_handles=family_handles,
        cohort_handles_by_course=cohort_handles_by_course,
        low_info_handle=low_info_handle,
        interval_handles=interval_handles,
        top_y=bottom_margin - 0.095,
    )

    if save_path:
        save_fig(fig, save_path)
        target = Path(save_path)
        fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    return fig
