"""Manuscript figures in the predictability-atlas style.

Every figure reads the long tables written by the analysis pipeline, so the
same function renders a synthetic design proof now and the real figure once
locked results exist:

``cells``
    one row per disease, outcome, horizon, contrast and metric with ``gain``,
    ``lower``, ``upper`` (paired patient-bootstrap interval), ``endpoint_group``,
    ``cohort_group`` and ``n_events_train``; ``gain`` is NaN when the comparison
    could not be estimated. This is ``contrasts.csv`` from
    ``opera.run.analyse_locked_predictions`` joined to the cell metadata.
``curve``
    the Bayesian support curve (``posterior_curve.csv``): ``contrast``,
    ``training_support``, ``mean``, ``lower``, ``upper``.
``estimates``
    absolute performance per model (``estimates.csv``).

Encodings follow the atlas: left-aligned lettered panel titles, diamonds with
bars for summaries, log-scaled training events, the indigo-to-gold horizon
ramp, a centred diverging scale for signed gains and grey for cells that could
not be estimated (never zero). Interpretation belongs in the caption.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from opera.visualization.style import PALETTE, TITLE_PAD, TITLE_SIZE, despine

ALL_HORIZONS = 0  # the panel pooled over every horizon (secondary_analysis.ALL_HORIZONS)

HORIZON_COLORS = {
    30: "#343A88",
    60: "#6650A4",
    90: "#9460A5",
    180: "#BD648E",
    365: "#D47762",
    730: "#D9A52C",
}
CONTRAST_STYLE = {
    "D-S": ("#3E8A87", "Disease-pooled vs specialist"),
    "O-S": ("#C16A58", "Outcome-pooled vs specialist"),
    "J-S": ("#2D3A8C", "Joint vs specialist"),
    "J-O": ("#7189BC", "Joint vs outcome-pooled"),
    "J-D": ("#B18B4D", "Joint vs disease-pooled"),
    "interaction": ("#8974AC", "Interaction (both pooling axes)"),
}
MODEL_STYLE = {
    "S": ("#7189BC", "Specialist"),
    "J": ("#8974AC", "Joint"),
    "R": ("#2D3A8C", "Routed"),
    # OPERA is the hero identity in deep indigo, as in opera.visualization.style;
    # the joint model takes lavender (fixed 2026-10-30: the two were swapped).
    "OPERA": ("#2D3A8C", "OPERA"),  # the configuration chosen on tuning data
    "X": ("#408579", "XGBoost"),
    "LR": ("#9E9E9E", "Logistic regression"),
}
ORIGIN_STYLE = {
    "L": ("#737D8C", "Cohort"),
    "E": ("#5C7AEA", "National"),
    "EA": ("#2D3A8C", "National, adapted"),
}
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "opera_gain", ["#B5451B", "#E7B9A5", "#F7F7F5", "#AEB8DA", "#2D3A8C"]
)
GAIN_CMAP.set_bad(PALETTE["missing"])
# Calibration bias is not better or worse by sign: a neutral diverging scale
# (ochre: predicted above observed; teal: below), unlike the gain map.
BIAS_CMAP = LinearSegmentedColormap.from_list(
    "opera_bias", ["#B18B4D", "#E6D5B5", "#F7F7F5", "#B5D3D1", "#3E8A87"])
SHORT_GROUP = {
    "Disease course and later malignancy": "Disease course",
    "Infection and antimicrobial use": "Infection",
    "Erythroid findings and transfusion": "Erythroid",
    "Neutrophil, white-cell and platelet findings": "Myeloid/platelet",
    "Lymphocyte and eosinophil counts": "Lymphocyte",
    "Liver injury and coagulation": "Liver",
    "Renal injury and support": "Renal",
    "Sodium and potassium": "Na/K",
    "Calcium and magnesium": "Ca/Mg",
    "Metabolic and cell-turnover markers": "Metabolic",
    "Cardiovascular and thrombotic events": "Cardiovascular",
    "Acute care and other events": "Acute care",
}
METRIC_LABEL = {
    "auc": "AUC gain",
    "brier": "Brier score improvement",
    "log_loss": "Log-loss improvement",
    "ipa": "IPA gain",
}


def watermark(fig, text="SYNTHETIC DESIGN PROOF"):
    if text:
        text = str(text).replace(r"\\u2014", "—").replace(r"\u2014", "—")
        fig.text(0.995, 0.002, text, ha="right", va="bottom", fontsize=6.5,
                 color=PALETTE["ink_muted"], alpha=0.8)


def _panel_title(ax, letter, title, **kwargs):
    kwargs.setdefault('fontsize', TITLE_SIZE)
    ax.set_title(f"{letter}  {title}", loc="left", **kwargs)


def _ordered(cells, key, group_key):
    frame = cells[[key, group_key]].drop_duplicates(key)
    return frame.sort_values([group_key, key])[key].tolist(), frame.set_index(key)[group_key]


def _split_word():
    """'Held-out test' for the paper; 'Tuning-set' in the tuning preview
    (OPERA_PAPER_PLOT_SPLIT=tuning), so no preview label claims test data."""
    import os

    return "Tuning-set" if os.environ.get("OPERA_PAPER_PLOT_SPLIT") == "tuning" else "Held-out test"


def _horizon_mask(frame, horizon):
    """Rows for ``horizon``. For the pooled panel (ALL_HORIZONS): a table's
    pooled cell rows where it has them, else every cell at every horizon
    (each disease-outcome-horizon cell weighing equally, as in Table 2)."""
    days = frame.horizon_days
    if horizon == ALL_HORIZONS and not (days == ALL_HORIZONS).any():
        return days != ALL_HORIZONS
    return days == horizon


def _single(horizon):
    """Plots that need one horizon (reliability curves) use one year when
    the rest of the figure is pooled."""
    return 365 if horizon == ALL_HORIZONS else horizon

def _support_totals(cells, key, horizon):
    """Training events per disease or outcome at one horizon (one row per cell)."""
    if "n_events_train" not in cells:
        raise ValueError("Ordering by support needs n_events_train in the cell table")
    unique = cells.loc[_horizon_mask(cells, _single(horizon))].drop_duplicates(
        ["cohort", "outcome", "horizon_days"])
    if key == "cohort" and "n_patients_train" in unique:
        # Diseases are ordered and labelled by training patients (head count).
        return unique.groupby(key).n_patients_train.max()
    return unique.groupby(key).n_events_train.sum()


def _support_label(cells, key):
    if key == "cohort" and "n_patients_train" in cells:
        return "Training patients"
    return "Training events"


def _axis_order(cells, key, group_key, *, order, horizon):
    """Items and their groups, by clinical group or by training events (rare first)."""
    items, groups = _ordered(cells, key, group_key)
    if order == "group":
        return items, groups
    if order != "support":
        raise ValueError("order must be 'group' or 'support'")
    totals = _support_totals(cells, key, horizon).reindex(items).fillna(0)
    return totals.sort_values(kind="stable").index.tolist(), groups


def _support_strip(ax, totals, *, axis, labels=None, fontsize=6.2, label=None):
    """Log-scale bars of training support beside a support-ordered matrix.

    The bars are placed relative to the matrix itself, so each bar lines up
    with exactly the row or column it describes. For rows (``axis="y"``) they
    sit left of the matrix, grow away from it and carry the row labels with
    the count, e.g. ``DLBCL (9,270)``; for columns they sit above it.
    """
    raw_values = np.asarray(totals, float)
    values = np.where(raw_values > 0, raw_values, np.nan)
    positions = np.arange(len(values))
    colour = "#9AA3B5"
    if axis == "y":
        bars = ax.inset_axes([-0.17, 0, 0.155, 1])
        bars.barh(positions, values, height=0.7, color=colour, linewidth=0)
        bars.set_xscale("log")
        bars.set_xlim((np.nanmax(values) if np.isfinite(values).any() else 1.) * 1.3, 0.8)  # bars grow leftward
        bars.set_ylim(ax.get_ylim())
        if labels is not None:
            ax.set_yticks(range(len(labels)), [])
            bars.set_yticks(positions, [f"{name} ({value:,.0f})" if np.isfinite(value) else f"{name} (unavailable)"
                                        for name, value in zip(labels, raw_values)])
        else:
            bars.set_yticks([])
        bars.tick_params(axis="y", which="both", length=0, labelsize=fontsize, pad=2)
        bars.tick_params(axis="x", which="both", length=0, labelbottom=False)
        if label:
            bars.set_title(label, fontsize=fontsize - 0.6, color=PALETTE["ink_secondary"],
                           loc="right", pad=2)
    else:
        bars = ax.inset_axes([0, 1.015, 1, 0.1])
        bars.bar(positions, values, width=0.75, color=colour, linewidth=0)
        bars.set_yscale("log")
        bars.set_xlim(ax.get_xlim())
        bars.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    bars.grid(False)
    for spine in bars.spines.values():
        spine.set_visible(False)
    return bars


def _group_breaks(order, groups):
    labels = [groups[item] for item in order]
    breaks, starts = [], []
    for index, label in enumerate(labels):
        if index == 0 or label != labels[index - 1]:
            starts.append((index, label))
            if index:
                breaks.append(index - 0.5)
    return breaks, starts


def _summary_rows(frame, by):
    """Saved paired domain means/CIs, or explicitly labelled median/IQR fallback."""
    rows = []
    for key, group in frame.groupby(by, sort=False):
        values = group.gain.dropna().to_numpy()
        if not len(values):
            continue
        q25, q75 = (np.quantile(values, [0.25, 0.75]) if len(values) > 1
                    else (values[0], values[0]))
        center = float(np.median(values))
        kind = "median_iqr"
        if {"group_mean", "group_lower", "group_upper"} <= set(group):
            saved = group[["group_mean", "group_lower", "group_upper"]].drop_duplicates()
            if len(saved) == 1 and np.isfinite(saved.group_mean.iloc[0]):
                center, q25, q75 = saved.iloc[0]
                kind = "paired_mean_ci"
        rows.append({**dict(zip(by if isinstance(by, list) else [by],
                                key if isinstance(key, tuple) else (key,))),
                     "median": center, "q25": q25, "q75": q75,
                     "summary_kind": kind, "n": len(values)})
    return pd.DataFrame(rows)


def _diamond_forest(ax, summary, *, row_key, row_order, series_key, series_style,
                    xlabel, low="q25", high="q75", show_counts=True, zero_line=True,
                    spread=0.27):
    if "summary_kind" in summary and set(summary.summary_kind) == {"paired_mean_ci"}:
        xlabel = xlabel.replace("median, IQR", "mean, 95% CI")
    series = [s for s in series_style if s in set(summary[series_key])]
    offsets = np.linspace(-spread, spread, len(series)) if len(series) > 1 else [0.0]
    for name, offset in zip(series, offsets):
        color = series_style[name][0]
        part = summary.loc[summary[series_key] == name].set_index(row_key)
        for y, row in enumerate(row_order):
            if row not in part.index:
                continue
            item = part.loc[row]
            ax.plot([item[low], item[high]], [y + offset] * 2, color=color, lw=1.3,
                    alpha=0.45, solid_capstyle="round", zorder=2)
            ax.scatter(item["median"], y + offset, marker="D", s=24, color=color,
                       zorder=3, linewidths=0)
    if zero_line:
        ax.axvline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
    labels = row_order
    if show_counts:
        counts = summary.groupby(row_key)["n"].max()
        labels = [f"{row} (n={counts.get(row, 0)})" for row in row_order]
    ax.set_yticks(range(len(row_order)), labels)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(-0.6, len(row_order) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    despine(ax, "x")
    return [Line2D([], [], linestyle="none", marker="D", markersize=5,
                   color=series_style[name][0], label=series_style[name][1])
            for name in series]


def _legend(ax, handles, *, loc="lower right", title=None, ncol=1, **kwargs):
    fontsize = kwargs.pop('fontsize', 7)
    title_fontsize = kwargs.pop('title_fontsize', 7.2)
    legend = ax.legend(handles=handles, loc=loc, title=title, ncol=ncol, frameon=True,
                       facecolor="white", edgecolor=PALETTE["grid"], framealpha=0.97,
                       fontsize=fontsize, title_fontsize=title_fontsize, **kwargs)
    return legend


def gain_matrix(ax, cells, *, contrast, metric="auc", horizon=ALL_HORIZONS, vmax=0.06,
                show_ylabels=True, show_xlabels=False, order="group"):
    """Disease (rows) by outcome (columns) map of one paired contrast.

    ``order="support"`` sorts rows and columns by training events, rare first,
    and replaces the group separators with event-count strips.
    """
    data = cells.loc[(cells.contrast == contrast) & (cells.metric == metric)
                     & _horizon_mask(cells, horizon)]
    diseases, disease_group = _axis_order(cells, "cohort", "cohort_group",
                                          order=order, horizon=horizon)
    outcomes, outcome_group = _axis_order(cells, "outcome", "endpoint_group",
                                          order=order, horizon=horizon)
    matrix = (data.pivot_table(index="cohort", columns="outcome", values="gain",
                               aggfunc="first")
              .reindex(index=diseases, columns=outcomes))
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), cmap=GAIN_CMAP,
                      norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto",
                      interpolation="nearest")
    if {"lower", "upper"} <= set(data):
        # Dots mark cells whose 95% interval excludes zero (white: gain, open
        # ring: loss), so the eye can tell estimated effects from noise.
        bounds = {key: (data.pivot_table(index="cohort", columns="outcome", values=key,
                                         aggfunc="first")
                        .reindex(index=diseases, columns=outcomes).to_numpy(float))
                  for key in ("lower", "upper")}
        rows, cols = np.nonzero(bounds["lower"] > 0)
        ax.scatter(cols, rows, s=3.0, color="white", edgecolors=PALETTE["ink"],
                   linewidths=0.25, zorder=3)
        rows, cols = np.nonzero(bounds["upper"] < 0)
        ax.scatter(cols, rows, s=2.2, facecolors="none", edgecolors=PALETTE["ink"],
                   linewidths=0.35, zorder=3)
    if order == "group":
        for position in _group_breaks(outcomes, outcome_group)[0]:
            ax.axvline(position, color="white", lw=1.2)
        for position in _group_breaks(diseases, disease_group)[0]:
            ax.axhline(position, color="white", lw=1.2)
    ax.set_yticks(range(len(diseases)), diseases if show_ylabels else [])
    ax.tick_params(axis="y", labelsize=5.6, length=0)
    if order == "support":
        if show_xlabels:
            from opera.visualization.outcome_names import outcome_display_name
            ax.set_xticks(range(len(outcomes)), [outcome_display_name(o) for o in outcomes],
                          rotation=90, ha='center', fontsize=6)
            ax.tick_params(axis='x', length=0, pad=3)
        else:
            ax.set_xticks([])
        ax.support_axes = (diseases, outcomes)  # callers decide which edges get bars
    elif show_xlabels:
        breaks, starts = _group_breaks(outcomes, outcome_group)
        ends = [*breaks, len(outcomes) - 0.5]
        centres = [(start - 0.5 + end) / 2 for (start, _), end in zip(starts, ends)]
        ax.set_xticks(centres, [SHORT_GROUP.get(g, g) for _, g in starts], rotation=55,
                      ha="right", rotation_mode="anchor", fontsize=5.8)
        ax.tick_params(axis="x", length=0, pad=2)
    else:
        ax.set_xticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def group_matrix(ax, cells, *, contrast, order, metric="auc", horizon=ALL_HORIZONS, vmax=0.04,
                 show_ylabels=True, show_xlabels=True, row_order="group"):
    """Diseases (rows) by endpoint groups (columns): median cell gain per tile."""
    data = cells.loc[(cells.contrast == contrast) & (cells.metric == metric)
                     & _horizon_mask(cells, horizon)]
    diseases, disease_group = _axis_order(cells, "cohort", "cohort_group",
                                          order=row_order, horizon=horizon)
    matrix = (data.groupby(["cohort", "endpoint_group"]).gain.median().unstack()
              .reindex(index=diseases, columns=order))
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), cmap=GAIN_CMAP,
                      norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto",
                      interpolation="nearest")
    ax.set_xticks(np.arange(len(order)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(diseases)) + 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    if row_order == "group":
        for position in _group_breaks(diseases, disease_group)[0]:
            ax.axhline(position, color="white", lw=2.2)
    ax.set_yticks(range(len(diseases)), diseases if show_ylabels else [])
    ax.tick_params(axis="y", labelsize=6.2, length=0, pad=2)
    if show_xlabels:
        ax.set_xticks(range(len(order)), [SHORT_GROUP.get(g, g) for g in order],
                      rotation=50, ha="right", rotation_mode="anchor", fontsize=6.4)
        ax.tick_params(axis="x", length=0, pad=2)
    else:
        ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


# Cells drawn as points in gain-by-support scatters. With a handful of training
# or test events a cell's AUC gain swings to +-0.5 or +-1 on chance alone (one
# test event ranked first or last); those cells stay in every estimate and
# curve, but are not drawn. Captions state the rule.
SCATTER_MIN_TRAIN_EVENTS = 10
SCATTER_MAX_CI_WIDTH = 0.4


def estimable_cells(cells):
    """Cells precise enough to draw as points (see SCATTER_* above)."""
    keep = cells.gain.notna() & (cells.n_events_train >= SCATTER_MIN_TRAIN_EVENTS)
    if {"lower", "upper"} <= set(cells):
        keep &= (cells.upper - cells.lower) <= SCATTER_MAX_CI_WIDTH
    return cells.loc[keep]


def shared_gain_limit(cells, *, metric="auc", horizon=ALL_HORIZONS, quantile=0.9, floor=0.02):
    """Colour limit for gain maps: the 90th percentile of |gain|, rounded up to
    0.01, so the scale follows the data instead of saturating most cells."""
    gains = cells.loc[(cells.metric == metric) & _horizon_mask(cells, horizon), "gain"]
    gains = gains.abs().dropna()
    if gains.empty:
        return 0.06
    return max(floor, float(np.ceil(gains.quantile(quantile) * 100) / 100))


def figure_pooling_cells(cells, *, metric="auc", horizon=ALL_HORIZONS, order="support",
                         include_router=True, domain=None, contrast=None, mark=None):
    """Supplement: the full disease-by-outcome maps behind the pooling figure.

    Rows and columns run from fewest to most training events by default
    (``order="support"``); ``order="group"`` keeps the clinical grouping.
    """
    if domain is not None:
        cells = cells.loc[cells.endpoint_group.eq(domain)].copy()
        if cells.empty:
            raise ValueError(f'No outcomes in pooling domain {domain}')
    panels = [("D-S", "A", "Disease-pooled vs specialist"),
              ("O-S", "B", "Outcome-pooled vs specialist"),
              ("J-S", "C", "Joint vs specialist"),
              ("interaction", "D", "Interaction (both pooling axes)")]
    if include_router:
        available = set(cells.contrast)
        router = next((c for c in ("OPERA-J", "R1") if c in available), None)
        if router:
            panels.append((router, "E", "OPERA vs joint"))
    comparable_contrasts = [p[0] for p in panels]
    if contrast is not None:
        panels = [p for p in panels if p[0] == contrast or contrast == 'OPERA-J' and p[0] == 'R1']
        if not panels:
            raise ValueError(f'Pooling comparison unavailable: {contrast}')
    width = max(8.8, .145 * cells.outcome.nunique() + 2.1)
    height = 7.0 if contrast is not None else 2.25 * len(panels) + 3.0
    fig = plt.figure(figsize=(width, height))
    top = fig.add_gridspec(len(panels), 1, hspace=0.35, left=0.18, right=0.91,
                           top=0.955, bottom=2.3 / height)
    image = None
    vmax = shared_gain_limit(cells.loc[cells.contrast.isin(comparable_contrasts)],
                             metric=metric, horizon=horizon)
    for index, (contrast, letter, title) in enumerate(panels):
        ax = fig.add_subplot(top[index, 0])
        image = gain_matrix(ax, cells, contrast=contrast, metric=metric, vmax=vmax,
                            horizon=horizon, show_ylabels=True,
                            show_xlabels=index == len(panels) - 1, order=order)
        titled = ax
        if order == "support":
            diseases, outcomes = ax.support_axes
            if index == 0:
                titled = _support_strip(ax, _support_totals(cells, "outcome", horizon)
                                        .reindex(outcomes), axis="x")
            if index == 0:
                _support_strip(ax, _support_totals(cells, "cohort", horizon).reindex(diseases),
                               axis="y", labels=diseases, fontsize=5.8,
                               label=_support_label(cells, "cohort") if index == 0 else None)
        _panel_title(titled, letter, title)
    cax = fig.add_axes([0.935, 0.48, 0.012, 0.24])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label(f"{_horizon_phrase(horizon)} {METRIC_LABEL[metric]}", fontsize=7.5)
    bar.ax.tick_params(labelsize=6.5)
    bar.outline.set_visible(False)
    watermark(fig, mark)
    return fig


def figure_pooling(cells, curve=None, *, metric="auc", horizon=ALL_HORIZONS, order="support",
                   mark=None):
    """Q1: what learning across diseases and outcomes adds (main figure).

    Tiles are the median one-year gain across the outcomes of an endpoint group
    within a fine disease; the cell-level maps are ``figure_pooling_cells``.
    """
    fig = plt.figure(figsize=(7.2, 8.6))
    top = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.06, left=0.29,
                           right=0.86, top=0.94, bottom=0.47)
    joint = cells.loc[(cells.contrast == "J-S") & (cells.metric == metric)
                      & _horizon_mask(cells, horizon)]
    if order == "support":
        columns = _support_totals(cells, "endpoint_group", horizon).sort_values().index.tolist()
    else:
        columns = joint.groupby("endpoint_group").gain.median().sort_values(
            ascending=False).index.tolist()
    # A: where the joint model gains over the specialist.
    ax = fig.add_subplot(top[0])
    image = group_matrix(ax, cells, contrast="J-S", order=columns, metric=metric,
                         horizon=horizon, row_order=order,
                         vmax=shared_gain_limit(cells.loc[cells.contrast == "J-S"],
                                                metric=metric, horizon=horizon))
    if order == "support":
        rows = _axis_order(cells, "cohort", "cohort_group", order=order, horizon=horizon)[0]
        _support_strip(ax, _support_totals(cells, "cohort", horizon).reindex(rows),
                       axis="y", labels=rows, label=_support_label(cells, "cohort"))
    _panel_title(ax, "A", "Joint vs specialist")
    # B: each disease's gain split by pooling axis, on the same rows.
    bx = fig.add_subplot(top[1], sharey=ax)
    diseases, _ = _axis_order(cells, "cohort", "cohort_group", order=order, horizon=horizon)
    year = cells.loc[(cells.metric == metric) & _horizon_mask(cells, horizon)]
    for offset, contrast in zip((-0.22, 0, 0.22), ("D-S", "O-S", "J-S")):
        color, label = CONTRAST_STYLE[contrast]
        medians = (year.loc[year.contrast == contrast].groupby("cohort").gain.mean()
                   .reindex(diseases))
        if {'disease_mean', 'disease_lower', 'disease_upper'} <= set(year):
            paired = year.loc[year.contrast == contrast,
                              ['cohort','disease_mean','disease_lower','disease_upper']].drop_duplicates()
            if paired.cohort.duplicated().any():
                raise ValueError('One saved disease summary per contrast is required')
            paired = paired.set_index('cohort').reindex(diseases)
            medians = paired.disease_mean
            bx.hlines(np.arange(len(diseases)) + offset, paired.disease_lower,
                      paired.disease_upper, color=color, lw=.9, alpha=.5)
        bx.scatter(medians.to_numpy(float), np.arange(len(diseases)) + offset,
                   marker="D", s=13 if contrast == "J-S" else 9, color=color,
                   zorder=3, label=label.replace(" vs specialist", ""))
    bx.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    bx.tick_params(axis="y", labelleft=False, length=0)
    bx.xaxis.set_major_locator(MaxNLocator(3))
    bx.set_xlabel(f"{METRIC_LABEL[metric]} over specialist\n" +
                  ('(mean, paired 95% CI)' if 'disease_mean' in year else
                   '(mean; intervals not exported)'))
    bx.grid(axis="y", visible=False)
    bx.legend(handles=[Line2D([], [], marker="D", ls="", color=CONTRAST_STYLE[c][0],
                              markersize=4.5, label=name)
                       for c, name in (("D-S", "Disease pooling"), ("O-S", "Outcome pooling"),
                                       ("J-S", "Both (joint)"))],
              loc="upper left", bbox_to_anchor=(-.08, -0.16), ncol=1, frameon=False,
              fontsize=7, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    _panel_title(bx, "B", "Gain by pooling axis")
    despine(bx, "x")
    cax = fig.add_axes([0.885, 0.62, 0.012, 0.25])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(f"{_horizon_phrase(horizon)} {METRIC_LABEL[metric]}\n(median within domain)",
                       fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.5)
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(5))

    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62, left=0.1,
                              right=0.98, top=0.25, bottom=0.065)
    ax = fig.add_subplot(bottom[0])
    data = cells.loc[(cells.metric == metric) & _horizon_mask(cells, horizon)
                     & cells.contrast.isin(["D-S", "O-S", "J-S"])]
    joint = estimable_cells(data.loc[data.contrast == "J-S"])
    gains = data.loc[data.contrast.eq('J-S'), 'gain'].dropna()
    fig.text(.29, .985, f'Joint vs specialist: {(gains > 0).sum():,} higher, '
             f'{(gains < 0).sum():,} lower, {(gains == 0).sum():,} tied / {len(gains):,} paired cells',
             ha='left', va='top', fontsize=7, color=PALETTE['ink_secondary'])
    ax.scatter(joint.n_events_train.clip(lower=1), joint.gain, s=6, alpha=0.25,
               color=PALETTE["ink_muted"], linewidths=0, zorder=2, rasterized=True)
    handles = []
    for contrast in ["D-S", "O-S", "J-S"]:
        color = CONTRAST_STYLE[contrast][0]
        if curve is not None:
            line = curve.loc[curve.contrast == contrast].sort_values("training_support")
            ax.fill_between(line.training_support, line.lower, line.upper,
                            color=color, alpha=0.16, linewidth=0, zorder=3)
            ax.plot(line.training_support, line["mean"], color=color, lw=1.7, zorder=4)
        handles.append(Line2D([], [], color=color, lw=1.7,
                              label=CONTRAST_STYLE[contrast][1]))
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Target training events (log scale)")
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.yaxis.set_major_locator(MaxNLocator(5))
    _panel_title(ax, "C", "Gain and training support")
    _legend(ax, handles, loc="upper right")
    despine(ax, "y")

    ax = fig.add_subplot(bottom[1])
    order = (data.loc[data.contrast == "J-S"].groupby("endpoint_group").gain.median()
             .sort_values(ascending=False).index.tolist())
    summary = _summary_rows(data, ["endpoint_group", "contrast"])
    summary["endpoint_group"] = summary.endpoint_group.map(lambda g: SHORT_GROUP.get(g, g))
    _diamond_forest(ax, summary, row_key="endpoint_group",
                    row_order=[SHORT_GROUP.get(g, g) for g in order],
                    series_key="contrast",
                    series_style={k: CONTRAST_STYLE[k] for k in ("D-S", "O-S", "J-S")},
                    xlabel=f"{METRIC_LABEL[metric]} (median, IQR)", show_counts=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "D", "By clinical domain")
    watermark(fig, mark)
    return fig


HEADLINE_REFERENCES = (
    # (contrast in the cell table, label, colour); joint minus reference.
    ("J-S", "Specialist (same model, one task)", MODEL_STYLE["S"][0]),
    ("J-X", "XGBoost per task", MODEL_STYLE["X"][0]),
    ("J-XR", "XGBoost + registry variables", "#2F6B5F"),
    ("J-XP", "XGBoost pooled over diseases", "#B18B4D"),
    ("J-LR", "Logistic regression per task", MODEL_STYLE["LR"][0]),
)


HEADLINE_MODELS = (
    # (label, colour, contrasts summed to give the model's gain over XGBoost)
    ("Specialist", MODEL_STYLE["S"][0], ("S-X",)),
    ("Disease-pooled", CONTRAST_STYLE["D-S"][0], ("D-S", "S-X")),
    ("Outcome-pooled", CONTRAST_STYLE["O-S"][0], ("O-S", "S-X")),
    ("Joint", MODEL_STYLE["J"][0], ("J-X",)),
    ("Routed", MODEL_STYLE["R"][0], ("R-X",)),
    ("Joint, adapted national", ORIGIN_STYLE["E"][0], ("EA-L:J", "J-X")),
)


def _cell_sum(data, contrasts):
    """Per-cell sum of contrast gains (cells present for every term)."""
    keys = ["cohort", "outcome", "horizon_days"]
    parts = [data.loc[data.contrast == c].set_index(keys).gain for c in contrasts]
    if any(part.empty for part in parts):
        return None
    total = parts[0]
    for part in parts[1:]:
        total = total + part.reindex(total.index)
    return total.dropna().reset_index()


def figure_headline(cells, models=None, subgroups=None, scores=None, *, horizon=ALL_HORIZONS,
                    focus_model="J", focus_label=None, clinical_model=None,
                    require_training_scores=False, model_overview=False,
                    overview_estimates=None, mark=None):
    """The main result at a glance.

    ``cells``: per-cell gains as in the other plates (``J-LR`` is derived from
    ``J-X`` and ``X-LR``). ``models``: one row per cell with each model's
    absolute AUC (``J``, ``X``, ``n_events_train``; ``model_cells.csv``).
    ``subgroups``: patient-subgroup gains (``covariate == "history"``).
    ``scores``: one row per clinical index (``label``, ``index_sensitivity``,
    ``model_sensitivity``).

    A: the requested model against each available reference, one point per
    cell. Saved paired means may weight cell–horizon entries separately.
    B: every cell, XGBoost against the requested model. C: events captured
    at frozen clinical operating points, whose achieved specificity may differ.
    D: every available model against XGBoost across horizons.
    E: by recorded history before treatment.
    """
    if focus_model not in {"J", "OPERA"}:
        raise ValueError("Headline focus_model must be J or OPERA")
    focus_label = focus_label or ("Joint model" if focus_model == "J" else "OPERA")
    focus_color = MODEL_STYLE["J" if focus_model == "J" else "R"][0]
    data = cells.loc[cells.metric == "auc"].copy()
    if "J-LR" not in set(data.contrast) and {"J-X", "X-LR"} <= set(data.contrast):
        derived = _cell_sum(data, ("J-X", "X-LR"))
        data = pd.concat([data, derived.assign(contrast="J-LR", metric="auc")])
    # Historical aggregate exports lack the newly registered direct contrasts.
    # Recover point differences only. Their spreads remain descriptive IQRs,
    # never sums of confidence bounds or means from different task rosters.
    if focus_model == 'OPERA':
        for name, terms in {'OPERA-D': ('OPERA-J', 'J-D'),
                            'OPERA-O': ('OPERA-J', 'J-O'),
                            'OPERA-XP': ('OPERA-J', 'J-XP'),
                            'OPERA-LR': ('OPERA-X', 'X-LR')}.items():
            if name not in set(data.contrast):
                derived = _cell_sum(data, terms)
                if derived is not None:
                    data = pd.concat([data, derived.assign(contrast=name, metric='auc')], ignore_index=True)
    reference_spec = HEADLINE_REFERENCES if focus_model == "J" else (
        ("OPERA-S", "Specialist (same model, one task)", MODEL_STYLE["S"][0]),
        ("OPERA-D", "Disease-pooled model", CONTRAST_STYLE['D-S'][0]),
        ("OPERA-O", "Outcome-pooled model", CONTRAST_STYLE['O-S'][0]),
        ("OPERA-J", "Joint model", MODEL_STYLE["J"][0]),
        ("OPERA-X", "XGBoost per task", MODEL_STYLE["X"][0]),
        ("OPERA-XP", "XGBoost pooled over diseases", "#B18B4D"),
        ("OPERA-LR", "Logistic regression", MODEL_STYLE['LR'][0]),
    )
    references = [r for r in reference_spec if r[0] in set(data.contrast)]
    if not references:
        raise ValueError(f"No direct headline contrasts for {focus_model}")
    fig = plt.figure(figsize=(8.8 if model_overview else 7.2,
                              10.4 if model_overview else 8.9 + .3 * max(0, len(references)-4)))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.6 if model_overview else 1.0, 1.35, 1.0], hspace=0.55,
                             left=0.03, right=0.985, top=0.965, bottom=0.13)
    rng = np.random.default_rng(0)

    # A: reference ladder.
    ax = fig.add_subplot(outer[0])
    ax.set_position([0.27, ax.get_position().y0, 0.5, ax.get_position().height])
    year = data.loc[_horizon_mask(data, horizon)]
    for row, (contrast, label, color) in enumerate(references):
        gains = year.loc[year.contrast == contrast, "gain"].dropna().to_numpy()
        if not len(gains):
            continue
        ax.scatter(gains, row + rng.uniform(-0.28, 0.28, len(gains)), s=3.5,
                   color=color, alpha=0.28, linewidths=0, rasterized=True)
        q1, median, q3 = np.quantile(gains, [0.25, 0.5, 0.75])
        center_label = "median (IQR)"
        if "panel_mean" in year:
            summary = year.loc[year.contrast == contrast,
                               ["panel_mean", "panel_lower", "panel_upper"]].drop_duplicates()
            if len(summary) == 1 and np.isfinite(summary.panel_mean.iloc[0]):
                median, q1, q3 = summary.iloc[0]
                center_label = "cell–horizon mean (95% CI)" if horizon == ALL_HORIZONS else "paired mean (95% CI)"
        ax.plot([q1, q3], [row, row], color=PALETTE["ink"], lw=2.2,
                solid_capstyle="round", zorder=3)
        ax.scatter([median], [row], marker="D", s=34, color="white",
                   edgecolors=PALETTE["ink"], linewidths=1.1, zorder=4)
        ax.text(1.02, row, f"{np.mean(gains > 0):.0%} of cells better\n"
                f"{center_label} {median:+.3f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=6.6, color=PALETTE["ink_secondary"])
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.8)
    ax.set_yticks(range(len(references)), [label for _, label, _ in references], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"{_horizon_phrase(horizon)} AUC gain of {focus_label} "
                  "(one point per disease–outcome cell)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    _panel_title(ax, "A", "Gain over each reference")
    despine(ax, "x")
    if model_overview:
        if models is None:
            raise ValueError('Headline overview requires absolute model-cell estimates')
        ax.clear()
        _headline_model_overview(ax, models, estimates=overview_estimates,
                                 horizon=_single(horizon))

    middle = outer[1].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.55)
    # B: every cell.
    ax = fig.add_subplot(middle[0])
    if models is not None and focus_model not in models:
        raise ValueError(f"Headline models table lacks requested {focus_model}")
    if models is not None and {focus_model, "X"} <= set(models):
        part = models.loc[_horizon_mask(models, _single(horizon))].dropna(subset=[focus_model, "X"])
        above = part[focus_model] > part.X
        size = 3 + 30 * np.sqrt(part.n_events_train.clip(lower=1)
                                / part.n_events_train.max()) if "n_events_train" in part else 8
        low = float(min(part[focus_model].min(), part.X.min())) - 0.02
        ax.plot([low, 1], [low, 1], color=PALETTE["diagonal"], lw=0.9, ls="--", zorder=1)
        ax.fill_between([low, 1], [low, 1], [1, 1], color=focus_color, alpha=0.05,
                        lw=0, zorder=0)
        ax.scatter(part.X, part[focus_model], s=size, alpha=0.45, linewidths=0, rasterized=True,
                   color=np.where(above, focus_color, "#B5451B"), zorder=2)
        ax.text(0.04, 0.96, f"{focus_label} higher AUC in\n{above.sum():,} of {len(part):,} cells",
                transform=ax.transAxes, va="top", fontsize=7.4, color=focus_color,
                fontweight="bold")
        ax.text(0.96, 0.04, "XGBoost better", transform=ax.transAxes, ha="right",
                fontsize=6.6, color="#B5451B")
        ax.set_xlim(low, 1)
        ax.set_ylim(low, 1)
        ax.set_aspect("equal", anchor="W")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("XGBoost AUC (one year)")
    ax.set_ylabel(f"{focus_label} AUC (one year)")
    _panel_title(ax, "B", f"{focus_label} vs XGBoost per cell")
    despine(ax, "both")

    # C: clinical indices.
    ax = fig.add_subplot(middle[1])
    if scores is not None and require_training_scores:
        if not {"threshold_split", "model"} <= set(scores) or clinical_model is None:
            scores = None
        else:
            scores = scores.loc[scores.threshold_split.eq("train") & scores.model.eq(clinical_model)]
    if scores is not None and len(scores):
        rows = scores.reset_index(drop=True)
        # One pooled comparison per recorded index when both pooled and older
        # disease-specific exports are present. Keep every distinct index.
        if "label" in rows:
            rows = rows.assign(_index=rows.label.str.split(" · ").str[0],
                               _pooled=rows.label.str.endswith(" · all subtypes"))
            pooled = set(rows.loc[rows._pooled, "_index"])
            rows = rows.loc[rows._pooled | ~rows._index.isin(pooled)].reset_index(drop=True)
            rows['label'] = rows.label.map(_score_label)
            if 'n_events' in rows:
                rows['label'] += rows.n_events.map(lambda n: f' ({_count_text(n)} events)')
        for y, row in rows.iterrows():
            ax.plot([row.index_sensitivity, row.model_sensitivity], [y, y],
                    color=PALETTE["grid"], lw=2, zorder=1)
            ax.scatter(row.index_sensitivity, y, s=20, color=PALETTE["ink_muted"],
                       zorder=2, label="Index" if y == 0 else None)
            ax.scatter(row.model_sensitivity, y, s=26, marker="D",
                       color=focus_color, zorder=3,
                       label="Model" if y == 0 else None)
        ax.set_yticks(range(len(rows)), rows.label, fontsize=6.3)
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=6.4, loc="lower left", handletextpad=0.2)
    else:
        ax.text(.5, .5, "Clinical refresh pending", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=PALETTE["ink_muted"])
    ax.set_xlim(-0.02, 1.04)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("Events in the high-risk group\n(frozen training-derived cutoffs)")
    _panel_title(ax, "C", "Clinical indices")
    despine(ax, "x")

    lower = outer[2].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.45)
    # D: every model against XGBoost, every horizon.
    ax = fig.add_subplot(lower[0])
    horizon_models = HEADLINE_MODELS if focus_model == "J" else (
        ("Joint", MODEL_STYLE["J"][0], ("J-X",)),
        (focus_label, MODEL_STYLE["R"][0], ("OPERA-X",)),
        ("OPERA, national", "#408579", ("E-L:R", "R-X")),
        ("OPERA, adapted national", "#B18B4D", ("EA-L:R", "R-X")),
    )
    # The selected OPERA alias may itself be a national variant. Show that
    # fitted model once, rather than repeating it under a second legend entry.
    duplicate = {'R_E': 'OPERA, national', 'R_EA': 'OPERA, adapted national'}.get(clinical_model)
    if focus_model == 'OPERA' and duplicate:
        horizon_models = tuple(item for item in horizon_models if item[0] != duplicate)
    for label, color, terms in horizon_models:
        gains = _cell_sum(data, terms)
        if gains is None or gains.empty:
            continue
        # Deriving a contrast can change its roster; do not add panel means
        # from different contrasts. Use saved paired means only for direct terms.
        if len(terms) == 1 and "panel_mean" in data:
            direct = data.loc[(data.contrast == terms[0]) & (data.horizon_days > 0)]
            medians = direct.groupby("horizon_days").panel_mean.first()
        else:
            medians = gains.loc[gains.horizon_days > 0].groupby("horizon_days").gain.mean()
        ax.plot(medians.index, medians.values, color=color, lw=1.6 if label == "Joint" else 1.1,
                marker="D", markersize=3.4 if label == "Joint" else 2.6, label=label,
                zorder=3 if label == "Joint" else 2)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.8)
    horizons = sorted(data.loc[data.horizon_days > 0, "horizon_days"].unique())
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(d) for d in horizons], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean AUC gain over XGBoost")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.2, ncol=3,
               loc="lower left", bbox_to_anchor=(0.06, 0.025),
               handlelength=1.2, columnspacing=1)
    _panel_title(ax, "D", "Gain by horizon")
    despine(ax, "y")

    # E: by recorded history.
    ax = fig.add_subplot(lower[1])
    plotted_subgroups = 0
    if subgroups is not None:
        part = subgroups.loc[subgroups.covariate.isin(["history", "history_event_count"])]
        order = list(dict.fromkeys(part.subgroup))
        subgroup_terms = (("J-S", "vs specialist", MODEL_STYLE["S"][0]),
                          ("J-X", "vs XGBoost", MODEL_STYLE["X"][0])) if focus_model == "J" else (
                          ("OPERA-S", "vs specialist", MODEL_STYLE["S"][0]),
                          ("OPERA-X", "vs XGBoost", MODEL_STYLE["X"][0]))
        for offset, (contrast, label, color) in zip((-0.12, 0.12), subgroup_terms):
            aliases = {"J-S": "P5", "J-X": "T2", "OPERA-S": "O3", "OPERA-X": "O1"}
            selected = part.loc[part.contrast.isin([contrast, aliases[contrast]])]
            if selected.empty:
                continue
            plotted_subgroups += 1
            rows = selected.set_index("subgroup").reindex(order)
            x = np.arange(len(order)) + offset
            ax.errorbar(x, rows.gain * 100, yerr=[(rows.gain - rows.lower) * 100,
                                                  (rows.upper - rows.gain) * 100],
                        fmt="D", markersize=3.6, color=color, ecolor=color,
                        elinewidth=1.1, capsize=0, label=label)
        ax.axhline(0, color=PALETTE["zero_line"], lw=0.8)
        ax.set_xticks(range(len(order)), order, fontsize=6.3, rotation=25, ha="right")
        ax.legend(frameon=False, fontsize=6.2, loc="lower left", handlelength=1)
    if not plotted_subgroups:
        ax.text(.5, .5, "Matched subgroup estimates unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=PALETTE["ink_muted"])
    ax.set_xlabel("Recorded events before treatment")
    ax.set_ylabel("Brier improvement (×100, 95% CI)")
    _panel_title(ax, "E", "Gain by recorded history", pad=20)
    despine(ax, "y")
    watermark(fig, mark)
    return fig


def _headline_model_overview(ax, models, *, estimates=None, horizon=365):
    """Every available model on a common task roster; no inferred intervals."""
    spec = [
        ('S','Specialist',MODEL_STYLE['S'][0]),
        ('D','Disease-pooled',CONTRAST_STYLE['D-S'][0]),
        ('O','Outcome-pooled',CONTRAST_STYLE['O-S'][0]),
        ('J','Joint',MODEL_STYLE['J'][0]),
        ('OPERA','OPERA (local)',MODEL_STYLE['R'][0]),
        ('J_E','Joint (national)','#5C7AEA'),
        ('J_EA','Joint (national + adaptation)','#2D3A8C'),
        ('R_E','OPERA (national)','#408579'),
        ('R_EA','OPERA (national + adaptation)','#B18B4D'),
        ('LR','Logistic regression','#697481'),
        ('X','XGBoost',MODEL_STYLE['X'][0]),
        ('XP','Pooled XGBoost','#B18B4D')]
    year=models.loc[models.horizon_days==horizon]
    spec=[row for row in spec if row[0] in year and year[row[0]].notna().any()]
    if not spec:
        raise ValueError('No model estimates at the overview horizon')
    year=year.dropna(subset=[s[0] for s in spec])
    if year.empty:
        raise ValueError('No common supported task roster for headline model overview')
    rng=np.random.default_rng(20261028)
    for y,(key,label,color) in enumerate(spec):
        values=year[key].to_numpy(float)
        ax.scatter(values,y+rng.uniform(-.22,.22,len(values)),s=3,color=color,
                   alpha=.15,linewidths=0,rasterized=True)
        mean=float(np.mean(values))
        ax.scatter(mean,y,marker='D',s=27,color=color,edgecolor='white',linewidth=.6,zorder=4)
        # Optional intervals must be computed for this exact common roster.
        if estimates is not None and {'model','n_cells','estimate','lower','upper'} <= set(estimates):
            saved=estimates.loc[estimates.model.eq(key)&estimates.n_cells.eq(len(year))]
            if len(saved)==1 and np.isclose(saved.estimate.iloc[0],mean):
                ax.plot([saved.lower.iloc[0],saved.upper.iloc[0]],[y,y],color=color,lw=1.5)
        ax.text(1.02,y,f'{mean:.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=7)
    ax.set_yticks(range(len(spec)),[s[1] for s in spec],fontsize=7.4)
    ax.set_ylim(len(spec)-.5,-.5)
    ax.set_xlim(max(0,float(year[[s[0] for s in spec]].min().min())-.02),1.01)
    ax.set_xlabel(f'One-year AUC · {len(year):,} matched tasks\nPoints: task AUC; diamonds: descriptive means')
    _panel_title(ax,'A','Performance across model variants')
    despine(ax,'x')


def figure_model_overview(models, *, horizon=365, mark=None):
    """Companion for all variants, retaining a compact reference-based headline."""
    count = sum(c in models for c in ('S','D','O','J','OPERA','J_E','J_EA','R_E','R_EA','LR','X','XP'))
    fig, ax = plt.subplots(figsize=(8.8, max(3.5, .27*count+1.1)))
    fig.subplots_adjust(left=.34, right=.92, top=.91, bottom=.18)
    _headline_model_overview(ax, models, horizon=horizon)
    watermark(fig, mark)
    return fig


def figure_tabular(cells, curve=None, *, horizon=ALL_HORIZONS, mark=None):
    """Q2: neural models against task-specific gradient-boosted trees."""
    fig = plt.figure(figsize=(8.8, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.65, wspace=0.15,
                            left=0.19, right=0.98, top=0.86, bottom=0.2)
    styles = {"S-X": (MODEL_STYLE["S"][0], "Specialist vs XGBoost"),
              "J-X": (MODEL_STYLE["J"][0], "Joint vs XGBoost"),
              "R-X": (MODEL_STYLE["R"][0], "Routed vs XGBoost"),
              "J-XP": ("#B18B4D", "Joint vs disease-pooled XGBoost"),
              "J-XR": ("#6E6E6E", "Joint vs XGBoost with registry variables")}
    data = cells.loc[_horizon_mask(cells, horizon) & cells.contrast.isin(styles)]
    order = (data.loc[(data.contrast == "J-X") & (data.metric == "auc")]
             .groupby("endpoint_group").gain.median().sort_values(ascending=False)
             .index.tolist())
    handles = None
    for column, (metric, letter) in enumerate((("auc", "A"), ("brier", "B"))):
        ax = fig.add_subplot(grid[0, column])
        summary = _summary_rows(data.loc[data.metric == metric],
                                ["endpoint_group", "contrast"])
        summary["endpoint_group"] = summary.endpoint_group.map(lambda g: SHORT_GROUP.get(g,g))
        handles = _diamond_forest(ax, summary, row_key="endpoint_group",
                                  row_order=[SHORT_GROUP.get(g,g) for g in order], series_key="contrast",
                                  series_style=styles,
                                  xlabel=f"{METRIC_LABEL[metric]} (median, IQR)",
                                  show_counts=False)
        if column:
            ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        _panel_title(ax, letter, "Discrimination" if metric == "auc" else "Accuracy")
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.56, 0.995), title='Model comparisons in A and B',
               title_fontsize=7.5)

    lower = grid[1, :].subgridspec(1, 2, wspace=0.42)
    ax = fig.add_subplot(lower[0])
    part = estimable_cells(cells.loc[(cells.contrast == "J-X") & (cells.metric == "auc")
                                     & _horizon_mask(cells, horizon)])
    ax.scatter(part.n_events_train.clip(lower=1), part.gain, s=7, alpha=0.35,
               color=MODEL_STYLE["J"][0], linewidths=0, rasterized=True)
    if curve is not None:
        line = curve.loc[curve.contrast == "J-X"].sort_values("training_support")
        ax.fill_between(line.training_support, line.lower, line.upper,
                        color=MODEL_STYLE["J"][0], alpha=0.14, linewidth=0)
        ax.plot(line.training_support, line["mean"], color=MODEL_STYLE["J"][0], lw=1.6)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Target training events (log scale)")
    ax.set_ylabel("AUC gain, joint vs XGBoost")
    _panel_title(ax, "C", "Joint vs XGBoost by training support")
    despine(ax, "y")

    ax = fig.add_subplot(lower[1])
    part = cells.loc[(cells.contrast.isin(["J-X", "X-LR"])) & (cells.metric == "auc")
                     & _horizon_mask(cells, horizon)]
    horizons_summary = cells.loc[cells.contrast.isin(["J-X", "X-LR", "XP-X", "XR-X"])
                                 & (cells.metric == "auc") & (cells.horizon_days > 0)]
    for contrast, color, label in (("J-X", MODEL_STYLE["J"][0], "Joint vs XGBoost"),
                                   ("XP-X", "#B18B4D", "Pooled vs per-disease XGBoost"),
                                   ("XR-X", "#6E6E6E", "Registry variables added (XGBoost)"),
                                   ("X-LR", MODEL_STYLE["X"][0], "XGBoost vs logistic")):
        if contrast not in set(horizons_summary.contrast):
            continue
        rows = (horizons_summary.loc[horizons_summary.contrast == contrast]
                .groupby("horizon_days").gain.median())
        ax.plot(rows.index, rows.values, color=color, lw=1.4, marker="D", markersize=4,
                label=label)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 90, 180, 365, 730], ["30", "60", "90", "180", "365", "730"])
    ax.minorticks_off()
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Median AUC gain")
    _panel_title(ax, "D", "Across horizons")
    ax.legend(loc="upper left", bbox_to_anchor=(-.08,-.32), frameon=False,
              fontsize=6.8, ncol=1, borderaxespad=0)
    despine(ax, "y")
    del part
    watermark(fig, mark)
    return fig


EFFICIENCY_FRACTIONS = (10, 25, 50, 100)


def _signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "−")


def _crossing(fractions, values, target):
    """Share of patients (log-interpolated) at which ``values`` first reach ``target``."""
    for k, (f, v) in enumerate(zip(fractions, values)):
        if v >= target:
            if k == 0:
                return f, True  # already there at the smallest share
            f0, v0 = fractions[k - 1], values[k - 1]
            t = (target - v0) / (v - v0) if v != v0 else 0.0
            return float(np.exp(np.log(f0) + t * (np.log(f) - np.log(f0)))), False
    return None, False


def figure_external(estimates, cells, panel=None, models=None, *, horizon=365, metric="auc",
                    mark=None):
    """Q3: can a national encoder become a reusable starting point?

    The national population contains the cohort, so "adapted" is domain
    adaptation (continued pretraining on the cohort), not added data.

    B (with ``models``, ``model_cells.csv``): median one-year AUC of the joint
    model from the cohort encoder and from the adapted national encoder trained
    on 10-100% of the training patients (same cells), with the national
    advantage at each share and the share at which the national model reaches
    the cohort-encoder model trained on everyone. A: panel AUC from the cohort
    encoder, the national encoder used directly and after adaptation, with the paired
    effects of each step (``panel``: one panel-level row per paired contrast;
    without it, the first matching row of ``cells``, as in the synthetic
    proofs); the routed model in outline. C-D: the adapted national versus
    cohort effect per cell (``cells``), by training events and by disease.
    """
    panel_rows = cells if panel is None else panel
    efficiency = models is not None and {"J_L_F10", "J_EA_F10"} <= set(models)
    national, local = ORIGIN_STYLE["EA"][0], ORIGIN_STYLE["L"][0]
    disease_rows = cells.loc[cells.contrast.eq('EA-L:J') & cells.metric.eq(metric)
                             & cells.horizon_days.eq(horizon)]
    n_diseases = disease_rows.cohort.nunique() if 'cohort' in disease_rows else 0
    height = max(7.2, 4.1 + .19 * n_diseases)
    fig = plt.figure(figsize=(8.8, height))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, max(1., .055*n_diseases)],
                             hspace=0.62, left=0.09, right=0.975, top=0.94, bottom=0.12)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.55)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.75)

    def contrast(name):
        row = panel_rows.loc[(panel_rows.contrast == name) & (panel_rows.metric == metric)
                             & (panel_rows.horizon_days == horizon)]
        return None if row.empty else row.iloc[0]

    # B: the same accuracy with fewer training patients.
    ax = fig.add_subplot(top[1])
    if efficiency:
        year = models.loc[models.horizon_days == horizon]
        full = {"L": "J", "EA": "J_EA"}  # full-data models in model_cells.csv
        columns = {o: [f"J_{o}_F{f}" if f < 100 else full[o] for f in EFFICIENCY_FRACTIONS]
                   for o in ("L", "EA")}
        common = year.dropna(subset=[c for cols in columns.values() for c in cols if c in year])
        curves = {}
        for origin in ("L", "EA"):
            present = [(f, c) for f, c in zip(EFFICIENCY_FRACTIONS, columns[origin])
                       if c in common]
            q = np.array([np.nanquantile(common[c], [0.25, 0.5, 0.75]) for _, c in present])
            curves[origin] = ([f for f, _ in present], q)
        (fl, ql), (fn, qn) = curves["L"], curves["EA"]
        if fl == fn:
            ax.fill_between(fn, ql[:, 1], qn[:, 1], color=national, alpha=0.10, lw=0)
        for (fractions, q), color in ((curves["L"], local), (curves["EA"], national)):
            ax.plot(fractions, q[:, 1], color=color, lw=2.0, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        target = float(ql[-1, 1])
        ax.axhline(target, color=local, lw=0.8, ls=(0, (3, 2)), zorder=1)
        share, already = _crossing(fn, qn[:, 1], target)
        if share is not None:
            ax.scatter([share], [target], s=50, color="white", edgecolor=national, lw=1.5,
                       zorder=5)
            text = (f"Reaches the cohort-encoder model\ntrained on all patients with "
                    f"{'≤' if already else '≈'}{share:.0f}% of them")
            ax.text(.02, -.31, textwrap.fill(text.replace('\n', ' '), 51),
                    transform=ax.transAxes, fontsize=7, color=national,
                    ha='left', va='top')
        part = cells.loc[(cells.metric == metric) & (cells.horizon_days == horizon)]
        codes = [f"EA-L:F{f}" for f in EFFICIENCY_FRACTIONS[:-1]] + ["EA-L:J"]
        for f, code, low, high in list(zip(fn, codes, ql[:, 1], qn[:, 1]))[:-1]:
            gains = part.loc[part.contrast == code, "gain"].dropna()
            gap = float(gains.median()) if len(gains) else high - low
            # A bar joins the two models at this share; its difference beside it.
            ax.plot([f, f], [low, high], color=national, lw=0.9, alpha=0.6, zorder=3,
                    solid_capstyle="butt")
            ax.text(f * 1.05, (low + high) / 2, _signed(gap), ha="left", va="center",
                    fontsize=6, color=national, fontweight="bold",
                    zorder=5)
        ax.text(fn[-1] * 1.08, qn[-1, 1], "National,\nadapted", color=national,
                fontsize=6.6, va="bottom", ha="left", fontweight="bold", linespacing=1.0)
        ax.text(fl[-1] * 1.08, ql[-1, 1] - 0.004, "Cohort", color=local, fontsize=6.6, va="top",
                ha="left", fontweight="bold")
        ax.text(EFFICIENCY_FRACTIONS[0] * 0.9, target, "cohort encoder, all patients", fontsize=5.8,
                color=local, va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set_xticks(EFFICIENCY_FRACTIONS, [f"{f}%" for f in EFFICIENCY_FRACTIONS])
        ax.minorticks_off()
        ax.set_xlim(EFFICIENCY_FRACTIONS[0] * 0.85, EFFICIENCY_FRACTIONS[-1] * 1.6)
        ax.set_xlabel("Share of training patients")
        ax.set_ylabel("One-year AUC (median over cells)")
        low, high = min(ql[:, 1].min(), qn[:, 1].min()), max(ql[:, 1].max(), qn[:, 1].max())
        spread = max(high - low, .01)
        ax.set_ylim(low - 0.45 * spread, high + 0.25 * spread)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        despine(ax, "y")
    else:
        ax.set_axis_off()
    _panel_title(ax, "B", "Fewer patients for the same accuracy")

    # A: from cohort to national pretraining: one row per encoder, the joint
    # model filled and the routed model in outline, effects against the cohort.
    ax = fig.add_subplot(top[0])
    level = estimates.loc[(estimates.metric == metric) & (estimates.horizon_days == horizon)]
    stages = (("L", "Cohort"), ("E", "National"), ("EA", "National,\nadapted"))
    effects = {"E": contrast("E-L:J"), "EA": contrast("EA-L:J")}
    step = contrast("EA-E:J")
    reference = level.loc[(level.origin == "L") & (level.readout == "J")]
    if len(reference):
        ax.axvline(float(reference.estimate.iloc[0]), color=ORIGIN_STYLE["L"][0], lw=0.8,
                   ls=(0, (2, 2)), zorder=1)
    for y, (origin, _) in enumerate(stages):
        for readout, offset in (("J", -0.12), ("R", 0.16)):
            row = level.loc[(level.origin == origin) & (level.readout == readout)]
            if row.empty:
                continue
            row = row.iloc[0]
            filled = readout == "J"
            color = ORIGIN_STYLE[origin][0] if filled else MODEL_STYLE["R"][0]
            ax.plot([row.lower, row.upper], [y + offset] * 2, color=color,
                    lw=1.8 if filled else 1.1, alpha=0.5 if filled else 0.4,
                    solid_capstyle="round", zorder=2)
            ax.scatter(row.estimate, y + offset, s=62 if filled else 30, zorder=4,
                       marker="D" if filled else "o", color=color if filled else "white",
                       edgecolor="white" if filled else color, lw=0.8 if filled else 1.1)
    # Each effect sits at the end of its own interval, in that row's colour.
    rights = []
    for y, (origin, _) in enumerate(stages):
        row = level.loc[(level.origin == origin) & (level.readout == "J")]
        if row.empty:
            continue
        upper = float(row.upper.iloc[0])
        rights.append(upper)
        effect = effects.get(origin)
        if origin == "L":
            text = "reference"
        elif effect is not None:
            text = (f"{_signed(effect.gain)} vs cohort\n({effect.lower:.3f} to "
                    f"{effect.upper:.3f})").replace("-", "−")
            if origin == "EA" and step is not None:
                text += f"\n{_signed(step.gain)} vs national".replace("-", "−")
        else:
            continue
        ax.annotate(text, xy=(upper, y - 0.12), xytext=(5, 0), textcoords="offset points",
                    fontsize=5.9, ha="left", va="center", linespacing=1.05,
                    color=ORIGIN_STYLE[origin][0] if origin != "L" else PALETTE["ink_muted"],
                    fontweight="bold" if origin == "EA" else "normal")
    if rights:
        low, high = ax.get_xlim()
        ax.set_xlim(low, max(rights) + 0.85 * (max(rights) - low))
    ax.set_yticks(range(len(stages)), [label for _, label in stages], fontsize=7)
    ax.set_ylim(len(stages) - 0.4, -0.95)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Panel one-year AUC (95% CI)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    despine(ax, "x")
    ax.legend(handles=[Line2D([], [], ls="none", marker="D", color=national, label="Joint"),
                       Line2D([], [], ls="none", marker="o", markerfacecolor="white",
                              color=MODEL_STYLE["R"][0], label="Routed")],
              loc="lower right", ncol=2, frameon=False,
              fontsize=6.3, handletextpad=0.3, columnspacing=1.0)
    _panel_title(ax, "A", "From cohort to national pretraining")

    # C-D: where the adapted national encoder helps (cells; shared gain axis).
    data = cells.loc[(cells.contrast == "EA-L:J") & (cells.metric == metric)
                     & (cells.horizon_days == horizon)].dropna(subset=["gain"])
    cx = fig.add_subplot(bottom[0])
    dx = fig.add_subplot(bottom[1])
    if "n_events_train" in data and data.n_events_train.gt(0).any():
        part = data.loc[data.n_events_train > 0]
        x = part.n_events_train.astype(float)
        cx.scatter(x, part.gain, s=7, color=national, alpha=0.22, lw=0, zorder=2)
        bins = pd.qcut(np.log10(x), q=min(6, max(2, len(part) // 20)), duplicates="drop")
        trend = part.groupby(bins, observed=True).agg(
            x=("n_events_train", "median"), q1=("gain", lambda v: v.quantile(0.25)),
            m=("gain", "median"), q3=("gain", lambda v: v.quantile(0.75)))
        cx.fill_between(trend.x, trend.q1, trend.q3, color=national, alpha=0.15, lw=0)
        cx.plot(trend.x, trend.m, color=national, lw=2.0, zorder=3)
        cx.set_xscale("log")
        cx.minorticks_off()
        cx.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        cx.set_xlabel("Training events in the cell")
    cx.axhline(0, color=PALETTE["zero_line"], lw=0.7)
    cx.set_ylabel(f"{METRIC_LABEL[metric]}, adapted national vs cohort")
    cx.yaxis.set_major_locator(MaxNLocator(5))
    despine(cx, "y")
    _panel_title(cx, "C", "By training events")

    disease_key = 'cohort' if 'cohort' in data and data.cohort.notna().any() else 'cohort_group'
    if disease_key == 'cohort' and 'n_patients_train' in data:
        order = data.groupby(disease_key).n_patients_train.max().sort_values().index
        ordering = 'Training cohort size, small to large'
    elif disease_key == 'cohort':
        order = _support_totals(data, 'cohort', horizon).sort_values().index
        ordering = 'Training support, low to high'
    else:
        order = data.groupby(disease_key).gain.size().sort_values().index
        ordering = 'Disease group'
    rng = np.random.default_rng(0)
    for k, group in enumerate(order):
        gains = data.loc[data[disease_key] == group, 'gain'].dropna().to_numpy()
        if not len(gains):
            continue
        dx.scatter(gains, k + rng.uniform(-.22, .22, len(gains)), s=6, color=national,
                   alpha=.22, lw=0, zorder=2)
        q1, median, q3 = np.quantile(gains, [.25, .5, .75])
        dx.plot([q1, q3], [k, k], color=national, lw=2.2, alpha=.6)
        dx.scatter(median, k, marker='D', s=24, color=national, zorder=4)
    dx.axvline(0, color=PALETTE['zero_line'], lw=.7)
    dx.set_yticks(range(len(order)), list(order), fontsize=7)
    dx.set_ylim(len(order)-.5, -.5)
    dx.set_xlabel(f'{METRIC_LABEL[metric]} (median, IQR)')
    dx.set_ylabel(ordering)
    dx.xaxis.set_major_locator(MaxNLocator(4))
    despine(dx, 'x')
    _panel_title(dx, 'D', 'By disease')
    watermark(fig, mark)
    return fig


def figure_routing(cells, gate, *, metric="auc", horizon=ALL_HORIZONS, tolerance=0.005, mark=None):
    """Supplement: routed readout and adapter-bank factorial."""
    gate = gate.copy()
    had_gate_rows = len(gate) > 0
    if "metric" in gate:
        gate = gate.loc[gate.metric == metric]
    if "horizon_days" in gate:
        # Panel summaries cannot be pooled by taking an arbitrary first row.
        gate = gate.loc[gate.horizon_days == horizon]
    if had_gate_rows and gate.empty:
        raise ValueError("Routing bank panel has no summary for the requested metric/horizon")
    if gate.contrast.duplicated().any():
        raise ValueError("Routing panel needs one summary per contrast/metric/horizon")
    fig = plt.figure(figsize=(8.8, 4.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.9, left=0.21,
                            right=0.98, top=0.83, bottom=0.22)
    ax = fig.add_subplot(grid[0])
    router_contrast = 'OPERA-J' if 'OPERA-J' in set(cells.contrast) else 'R1'
    data = cells.loc[(cells.contrast == router_contrast) & (cells.metric == metric)
                     & _horizon_mask(cells, horizon)]
    order = data.groupby("endpoint_group").gain.median().sort_values(ascending=False)
    summary = _summary_rows(data.assign(series="R1"), ["endpoint_group", "series"])
    summary["endpoint_group"] = summary.endpoint_group.map(lambda g: SHORT_GROUP.get(g,g))
    _diamond_forest(ax, summary, row_key="endpoint_group", row_order=[SHORT_GROUP.get(g,g) for g in order.index],
                    series_key="series", series_style={"R1": (MODEL_STYLE["R"][0], "")},
                    xlabel=f"{METRIC_LABEL[metric]}, routed vs joint (median, IQR)",
                    show_counts=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "A", "OPERA vs joint" if router_contrast == 'OPERA-J' else "Local routed vs joint")

    ax = fig.add_subplot(grid[1])
    labels = {"R0": "All sources masked vs joint", "R2": "Disease bank vs no banks", "R3": "Outcome bank vs no banks",
              "R4": "Add outcomes to disease bank", "R5": "Add diseases to outcome bank",
              "R6": "Both banks vs no banks", "R7": "Both minus the two single-bank gains"}
    missing_intervals = False
    for y, contrast in enumerate(labels):
        row = gate.loc[gate.contrast == contrast]
        if row.empty:
            continue
        row = row.iloc[0]
        color = PALETTE["ink_secondary"] if contrast == "R0" else MODEL_STYLE["R"][0]
        finite_ci = np.isfinite([row.lower, row.upper]).all()
        if finite_ci:
            ax.plot([row.lower, row.upper], [y, y], color=color, lw=1.3, alpha=0.55)
        else:
            missing_intervals = True
        ax.scatter(row.gain, y, marker="D", s=24, edgecolors=color,
                   facecolors=color if finite_ci else "none", zorder=3)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_yticks(range(len(labels)), [textwrap.fill(label, 28) for label in labels.values()], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Panel {METRIC_LABEL[metric]} (95% CI)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    if missing_intervals:
        ax.text(0, -0.27, "Open diamonds: CI unavailable", transform=ax.transAxes,
                fontsize=6.3, color=PALETTE["ink_secondary"])
    _panel_title(ax, "B", "Bank controls (original router)")
    despine(ax, "x")
    watermark(fig, mark)
    return fig


BORROWING_SHORT = {"TRANSFORMED_FL": "tFL", "AMYLOIDOSIS": "AL", "SolM": "SP"}
LEVEL_STYLE = {
    # Shades of the routed model's colour, darkest for the within-model level.
    "dependence": ("#4B3B7A", "Full vs other-disease corrections replaced"),
    "trained_borrowing": ("#8974AC", "Full vs trained own-task model"),
    "routing": ("#B7A9D3", "OPERA vs trained joint model"),
}


def _short_disease(code):
    return BORROWING_SHORT.get(code, code)


def _borrowing_order(pooled):
    """Target diseases ordered by lineage; group inventory covers both axes."""
    groups = sorted(set(pooled.target_group) | set(pooled.source_group))
    order = (pooled[["target", "target_group"]].drop_duplicates()
             .assign(g=lambda d: d.target_group.map({g: i for i, g in enumerate(groups)}))
             .sort_values(["g", "target"]).target.tolist())
    return order, groups


def _borrowing_sources(pooled):
    """Every exported source, including diseases without an estimable target."""
    return (pooled[['source', 'source_group']].drop_duplicates()
            .sort_values(['source_group', 'source']).source.tolist())


def _auc_levels(levels, *, all_outcomes=True):
    if levels is None:
        return None
    data = levels.loc[levels.endpoint_group == "All outcomes"] if all_outcomes else levels
    return data.loc[data.metric == "auc"] if "metric" in data else data


def _sparse_targets(pooled, min_cells):
    if "n_cells" not in pooled:
        return set()
    off = pooled.target != pooled.source
    cells = pooled.loc[off].groupby("target").n_cells.max()
    return set(cells.index[cells.fillna(0) < min_cells])


def _group_heatmap(ax, part, groups, vmax, *, labels):
    grid = (part.groupby(["target_group", "source_group"]).effect.mean().unstack()
            .reindex(index=groups, columns=groups)).to_numpy(float)
    ax.grid(False)
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=GAIN_CMAP,
                      norm=TwoSlopeNorm(0, -vmax, vmax), aspect="equal",
                      interpolation="nearest")
    ax.set_xticks(range(len(groups)), groups if labels else [""] * len(groups),
                  rotation=90, fontsize=5.2)
    ax.set_yticks(range(len(groups)), groups if labels else [""] * len(groups),
                  fontsize=5.2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _key_legend(ax, handles, *, y, ncol=1):
    """A key placed directly under its own panel, left-aligned with it."""
    return ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, y), ncol=ncol,
                     frameon=False, fontsize=6.3, handletextpad=0.3, columnspacing=0.9,
                     borderaxespad=0, labelspacing=0.35)


def _pair_labels(data):
    return [f"{_short_disease(r.source)} → {_short_disease(r.target)}, "
            f"{SHORT_GROUP.get(r.domain, r.domain).lower()}" for r in data.itertuples()]


def lineage_concentration(pooled):
    """Per target disease, from the single-source gains (``effect``, all outcomes):
    the share of its positive borrowing gain that comes from diseases of its own
    lineage (``observed``) against that lineage's share of the other sources
    (``expected``, what borrowing blind to lineage would give), and the effective
    number of sources it gains from (exp of the entropy of the gain shares)."""
    rows = []
    for target, part in pooled.loc[pooled.target != pooled.source].groupby("target"):
        gain = part.effect.clip(lower=0).to_numpy(float)
        if not np.isfinite(gain).all() or gain.sum() <= 0:
            continue
        same = (part.source_group == part.target_group).to_numpy()
        share = gain / gain.sum()
        positive = share[share > 0]
        rows.append({"target": target, "observed": float(share[same].sum()),
                     "expected": float(same.mean()), "n_sources": len(part),
                     "effective_sources": float(np.exp(-(positive * np.log(positive)).sum()))})
    return pd.DataFrame(rows, columns=["target", "observed", "expected", "n_sources",
                                       "effective_sources"])


def _borrowing_plot_table(table, estimate):
    if estimate not in {'shrunk', 'raw'}:
        raise ValueError('pair_estimate must be shrunk or raw')
    if table.duplicated(['target', 'source', 'endpoint_group']).any():
        raise ValueError('Duplicate borrowing target/source/domain rows')
    table = table.copy()
    if estimate == 'raw':
        if not {'raw_effect', 'raw_lower', 'raw_upper'} <= set(table):
            raise ValueError('Raw borrowing estimates are unavailable')
        table[['effect', 'lower', 'upper']] = table[['raw_effect', 'raw_lower', 'raw_upper']].to_numpy()
    return table


def figure_borrowing_map(table, hypotheses=None, levels=None, test=None, *, min_cells=24,
                         pair_estimate='shrunk', pair_scale_floor=1.0,
                         pair_vmax=None, lineage_panel=False, ablation_mode=None, model_identity=None, mark=None):
    """Source-alone effects and total dependence in the fixed routed model.

    A retains all source columns, including sources without estimable target
    rows. B shows total cross-disease dependence. C compares raw and shrunken
    pair estimates on an absolute scale. The optional legacy ``lineage_panel``
    replaces C with positive-effect shares, which are not causal contributions.
    D summarises dependence by outcome domain. Candidate pairs are shown in
    the supplementary detail figure rather than tested selectively here.

    ``table``: one row per target disease, source disease and outcome domain
    (``endpoint_group``; "All outcomes" pools them), with ``target_group``,
    ``source_group``, the partially pooled AUC gain of the source alone over no
    borrowing (``effect``, AUC points), its 95% interval and ``n_cells``.
    ``hypotheses``: prespecified pairs and negative controls with the unshrunk
    source-alone effect. ``levels``: rows per ``contrast``, target (with "All
    diseases") and ``endpoint_group``, with ``n_patients``; the dependence level
    is drawn. ``test``: the set test of the pairs; its ``mean_other`` (the mean
    of all other pairs of the same domains) is drawn as the typical pair. The
    variants drop source terms from the trained model without retraining, so
    they describe the fitted model, not the diseases.
    """
    from opera.visualization.paper_taxonomy import disease_name

    shrinkage_table = table.copy()
    table = _borrowing_plot_table(table, pair_estimate)
    estimate_label = 'Shrunken' if pair_estimate == 'shrunk' else 'Unshrunk'
    pooled = table.loc[table.endpoint_group == "All outcomes"]
    order, _ = _borrowing_order(pooled)
    group_of = dict(zip(pooled.target, pooled.target_group))
    sources = _borrowing_sources(pooled)
    source_group_of = dict(zip(pooled.source, pooled.source_group))
    sparse = _sparse_targets(pooled, min_cells)
    off = pooled.target != pooled.source
    scale = pooled.loc[off & ~pooled.target.isin(sparse), "effect"]
    vmax = max(float(pair_scale_floor),
               (float(np.nanpercentile(np.abs(scale.dropna()), 99))
                if scale.notna().any() else 1.0))
    if pair_vmax is not None:
        if not np.isfinite(pair_vmax) or pair_vmax <= 0:
            raise ValueError('pair_vmax must be positive and finite')
        vmax = float(pair_vmax)
    level = _auc_levels(levels)
    sizes = ({} if level is None else
             level.dropna(subset=["n_patients"]).drop_duplicates("target")
             .set_index("target").n_patients.to_dict())

    # Sizes in inches: A is square (equal cells) and B shares its rows.
    width, height = 7.2, 9.2
    side, b_width, left = 3.5, 1.55, 1.72
    a_bottom = 0.955 - side / height
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, a_bottom, side / width, side / height])
    bx = fig.add_axes([(left + side + 0.12) / width, a_bottom, b_width / width,
                       side / height], sharey=ax)

    # A: from whom each disease borrows.
    matrix = (pooled.pivot_table(index="target", columns="source", values="effect")
              .reindex(index=order, columns=sources)).to_numpy(float, copy=True)
    own = np.equal.outer(np.asarray(order), np.asarray(sources))
    matrix[own] = np.nan
    ax.grid(False)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=GAIN_CMAP,
                      norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto",
                      interpolation="nearest")
    ax.imshow(np.where(own, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list(
        "own", [PALETTE["grid"], PALETTE["grid"]]), aspect="auto", interpolation="nearest")
    # Dots use the pooled interval: per-pair unpooled intervals put about one
    # false dot in twenty null pairs, hundreds of pairs here.
    lower = (pooled.pivot_table(index="target", columns="source", values="lower")
             .reindex(index=order, columns=sources)).to_numpy(float)
    rows, cols = np.nonzero((lower > 0) & ~own) if pair_estimate == "shrunk" else ([], [])
    ax.scatter(cols, rows, s=5, color="white", edgecolors=PALETTE["ink"], linewidths=0.4,
               zorder=3)
    for b in [i for i in range(1, len(order))
              if group_of[order[i]] != group_of[order[i - 1]]]:
        ax.axhline(b - 0.5, color="white", lw=1.4)
    for b in [i for i in range(1, len(sources))
              if source_group_of[sources[i]] != source_group_of[sources[i-1]]]:
        ax.axvline(b - 0.5, color="white", lw=1.4)
    names = [f"{disease_name(c)} ({int(sizes[c]):,})" if c in sizes else disease_name(c)
             for c in order]
    ax.set_yticks(range(len(order)), names)
    ax.set_xticks(range(len(sources)), [_short_disease(c) for c in sources], rotation=90)
    ax.tick_params(length=0, labelsize=6.6)
    for label, code in zip(ax.get_yticklabels(), order):
        if code in sparse:
            label.set_color(PALETTE["ink_muted"])
    ax.set_xlabel("Source disease")
    ax.set_ylabel("Target disease (registry population n)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_title(ax, "A", "One disease source retained")
    cax = ax.inset_axes([0.0, -0.215, 1.0, 0.022])
    beyond = np.isfinite(matrix) & (np.abs(matrix) > vmax)
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal",
                           extend='both' if beyond.any() else 'neither')
    colorbar.set_label(f"{estimate_label} AUC gain, source alone (points)",
                       fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.5)
    colorbar.outline.set_visible(False)
    colorbar.ax.xaxis.set_major_locator(MaxNLocator(5))

    # B: each disease's total dependence on borrowing, as a row total.
    colour = LEVEL_STYLE["dependence"][0]
    if level is not None and (level.contrast == "dependence").any():
        dep = level.loc[level.contrast == "dependence"]
        per = dep.set_index("target").reindex(order)
        y = np.arange(len(order))
        bx.barh(y, per.effect, height=0.62, color=MODEL_STYLE["R"][0], alpha=0.55,
                linewidth=0)
        bx.hlines(y, per.lower, per.upper, color=colour, lw=0.8)
        overall = dep.loc[dep.target == "All diseases"]
        if len(overall):
            bx.axvline(overall.effect.iloc[0], color=colour, lw=0.9, ls=(0, (3, 2)))
    bx.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    bx.tick_params(axis="y", labelleft=False, length=0)
    bx.tick_params(axis="x", labelsize=6.6)
    bx.xaxis.set_major_locator(MaxNLocator(3))
    bx.set_xlabel("Full − all other diseases replaced\n(AUC points; dashed: overall)")
    bx.grid(axis="y", visible=False)
    _panel_title(bx, "B", "All disease sources together")
    despine(bx, "x")

    cx = fig.add_axes([1.85 / width, 0.1, 2.15 / width, 0.275])
    if lineage_panel:
        # C: does a disease borrow from its own lineage more than chance would give?
        conc = lineage_concentration(pooled.loc[~pooled.target.isin(sparse)])
        if len(conc):
            x_top = float(max(conc.expected.max(), 0.1)) * 1.35
            cx.plot([0, x_top], [0, x_top], color=PALETTE["ink_muted"], lw=0.8, ls=(0, (3, 2)))
            largest = max(sizes.values()) if sizes else 1
            size = 12 + 50 * np.sqrt(conc.target.map(sizes).fillna(0) / largest)
            points = cx.scatter(conc.expected, conc.observed, s=size, c=conc.effective_sources,
                                cmap=LinearSegmentedColormap.from_list(
                                    "breadth", ["#D6DBEA", MODEL_STYLE["R"][0], "#3B2A5C"]),
                                edgecolors="white", linewidths=0.5, zorder=3)
            for row in conc.loc[(conc.observed - conc.expected).abs().nlargest(3).index].itertuples():
                cx.annotate(_short_disease(row.target), (row.expected, row.observed),
                            xytext=(3, 2), textcoords="offset points", fontsize=6.2,
                            color=PALETTE["ink_secondary"])
            cx.set_xlim(0, x_top)
            cx.set_ylim(0, 1.0)
            bar = fig.colorbar(points, ax=cx, fraction=0.06, pad=0.02)
            bar.set_label("Effective number of sources", fontsize=6.4)
            bar.ax.tick_params(labelsize=6)
            bar.outline.set_visible(False)
            cx.text(0.97, 0.03, f"Effective sources: median {conc.effective_sources.median():.1f}"
                    f" of {int(conc.n_sources.max())}\nAbove the line: own lineage favoured",
                    transform=cx.transAxes, ha="right", va="bottom", fontsize=6.3,
                    color=PALETTE["ink_secondary"])
        cx.set_xlabel("Share of sources in the own lineage\n(expected if lineage did not matter)")
        cx.set_ylabel(f"Share of positive {estimate_label.lower()} estimates\nfrom the own lineage")
        cx.tick_params(labelsize=6.6)
        despine(cx, "both")
        _panel_title(cx, "C", "Related or broad?")

    else:
        diagnostic = shrinkage_table.loc[shrinkage_table.endpoint_group == 'All outcomes']
        column = 'raw_effect' if 'raw_effect' in diagnostic else 'effect'
        effects = diagnostic.loc[diagnostic.target.ne(diagnostic.source), column].dropna()
        if len(effects):
            magnitude = np.sort(np.abs(effects.to_numpy(float)))
            cx.step(np.r_[0, magnitude], np.r_[0, np.arange(1, len(magnitude)+1)/len(magnitude)],
                    where='post', color=MODEL_STYLE['R'][0], lw=1.6)
            cx.text(.97, .05, f'{len(effects):,} source–target pairs\n'
                    f'Median |gain| {np.median(magnitude):.3g}\n'
                    f'95th percentile {np.quantile(magnitude, .95):.3g}',
                    transform=cx.transAxes, ha='right', va='bottom', fontsize=7,
                    color=PALETTE['ink_secondary'])
        cx.set_ylim(0, 1.04)
        cx.set_xlim(left=0)
        cx.set_xlabel('Absolute single-source AUC gain (points)')
        cx.set_ylabel('Fraction of source–target pairs')
        cx.tick_params(labelsize=6.6)
        despine(cx, "both")
        _panel_title(cx, "C", "Size of individual-source effects")

    # D: for which outcomes borrowing matters, all diseases weighing equally.
    dx = fig.add_axes([5.3 / width, 0.1, 1.75 / width, 0.275])
    by_domain = _auc_levels(levels, all_outcomes=False)
    if by_domain is not None:
        rows_d = by_domain.loc[(by_domain.contrast == "dependence")
                               & (by_domain.target == "All diseases")
                               & (by_domain.endpoint_group != "All outcomes")]
        rows_d = rows_d.sort_values("effect", ascending=False).reset_index(drop=True)
        y = np.arange(len(rows_d))
        dx.hlines(y, rows_d.lower, rows_d.upper, color=colour, lw=1.0, alpha=0.6)
        dx.scatter(rows_d.effect, y, marker="D", s=16, color=colour, zorder=3)
        overall = by_domain.loc[(by_domain.contrast == "dependence")
                                & (by_domain.target == "All diseases")
                                & (by_domain.endpoint_group == "All outcomes")]
        if len(overall):
            dx.axvline(overall.effect.iloc[0], color=colour, lw=0.9, ls=(0, (3, 2)))
        dx.set_yticks(y, [SHORT_GROUP.get(d, d) for d in rows_d.endpoint_group])
        dx.set_ylim(len(rows_d) - 0.5, -0.5)
    dx.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    dx.tick_params(axis="y", length=0, labelsize=6.8)
    dx.tick_params(axis="x", labelsize=6.6)
    dx.xaxis.set_major_locator(MaxNLocator(3))
    dx.set_xlabel("Full − all other diseases replaced\n(AUC points, 95% CI)")
    dx.grid(axis="y", visible=False)
    despine(dx, "x")
    _panel_title(dx, "D", "Combined effect by outcome")
    if ablation_mode is not None:
        if ablation_mode not in {'mean','zero'}:
            raise ValueError('Unknown source-ablation mode')
        replacement = 'tuning-task mean' if ablation_mode == 'mean' else 'zero'
        identity = f" · {model_identity}" if model_identity else ''
        fig.text(.02,.018,f'Fixed fitted model · disease corrections replaced by {replacement} · outcome bank retained{identity}',
                 fontsize=6,color=PALETTE['ink_secondary'])
    watermark(fig, mark)
    return fig


def figure_borrowing_detail(table, hypotheses, levels=None, *, domains=None, min_cells=24,
                            pair_estimate='shrunk', pair_scale_floor=1.0, pair_vmax=None,
                            ablation_mode=None, model_identity=None, mark=None):
    """Supplement: the detail behind the borrowing figure.

    A: three levels per disease (AUC): dependence (the fitted model with other-disease terms retained versus replaced,
    keeping its outcome bank), borrowing as trained (against the router
    trained with only the own disease adapter) and routing (against the joint
    model). B: dependence and borrowing as trained against disease size.
    C: the expected pairs and negative controls, source alone and source left
    out. D: mean source-alone gain (raw or shrunken, as selected) between disease groups for
    every outcome domain (targets with few estimable cells left out).
    """
    from opera.visualization.paper_taxonomy import disease_name

    table = _borrowing_plot_table(table, pair_estimate)
    estimate_label = 'Shrunken' if pair_estimate == 'shrunk' else 'Unshrunk'
    pooled = table.loc[table.endpoint_group == "All outcomes"]
    order, groups = _borrowing_order(pooled)
    sparse = _sparse_targets(pooled, min_cells)
    domains = domains or [d for d in SHORT_GROUP if d in set(table.endpoint_group)]
    level = _auc_levels(levels)
    fig = plt.figure(figsize=(7.2, 10.0))

    # A: three levels per disease.
    ax = fig.add_axes([0.2, 0.555, 0.27, 0.405])
    present = [] if level is None else [c for c in LEVEL_STYLE if c in set(level.contrast)]
    for offset, contrast in zip(np.linspace(-0.22, 0.22, len(present)), present):
        colour, label = LEVEL_STYLE[contrast]
        part = level.loc[level.contrast == contrast].set_index("target").reindex(order)
        y = np.arange(len(order)) + offset
        ax.hlines(y, part.lower, part.upper, color=colour, lw=0.7, alpha=0.5)
        ax.scatter(part.effect, y, marker="D", s=12 if contrast == "dependence" else 9,
                   color=colour, zorder=3)
    ax.set_yticks(range(len(order)), [disease_name(c) for c in order])
    ax.tick_params(axis="y", length=0, labelsize=6.6)
    ax.tick_params(axis="x", labelsize=6.6)
    for label, code in zip(ax.get_yticklabels(), order):
        if code in sparse:
            label.set_color(PALETTE["ink_muted"])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_xlabel("AUC gain (points, 95% CI)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="y", visible=False)
    if present:
        _key_legend(ax, [Line2D([], [], marker="D", ls="", color=LEVEL_STYLE[c][0],
                                markersize=4.5, label=LEVEL_STYLE[c][1])
                         for c in present], y=-0.09)
    despine(ax, "x")
    _panel_title(ax, "A", "Three levels")

    # B: by disease size.
    ax = fig.add_axes([0.66, 0.79, 0.32, 0.17])
    handles = []
    if level is not None:
        per = level.loc[level.target != "All diseases"]
        for contrast in [c for c in ("trained_borrowing", "dependence")
                         if c in set(per.contrast)]:
            colour, label = LEVEL_STYLE[contrast]
            part = per.loc[per.contrast == contrast]
            ax.vlines(part.n_patients, part.lower, part.upper, color=colour, lw=0.7,
                      alpha=0.4, zorder=2)
            ax.scatter(part.n_patients, part.effect, marker="D", s=12, color=colour,
                       zorder=3, linewidths=0)
            handles.append(Line2D([], [], marker="D", ls="", color=colour, markersize=4.5,
                                  label=label))
        ax.set_xscale("log")
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
    ax.tick_params(labelsize=6.6)
    ax.set_xlabel("Registry population size (log scale)", fontsize=8)
    ax.set_ylabel("AUC gain (points)", fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if handles:
        ax.legend(handles=handles[::-1], loc="upper right", frameon=False, fontsize=6.3,
                  handletextpad=0.3, borderaxespad=0)
    despine(ax, "y")
    _panel_title(ax, "B", "By disease size")

    # C: pairs, source alone and source left out.
    ax = fig.add_axes([0.73, 0.555, 0.25, 0.17])
    data = hypotheses.reset_index(drop=True)
    colours = {"hypothesis": MODEL_STYLE["R"][0], "negative control": PALETTE["ink_muted"]}
    for y, row in data.iterrows():
        c = colours.get(row.kind, PALETTE["ink"])
        ax.plot([row.lower, row.upper], [y - 0.17, y - 0.17], color=c, lw=0.9, alpha=0.6)
        ax.scatter(row.effect, y - 0.17, marker="D", s=9, color=c, zorder=3)
        if np.isfinite(row.get("out_effect", np.nan)):
            ax.plot([row.out_lower, row.out_upper], [y + 0.2, y + 0.2], color=c, lw=0.6,
                    alpha=0.45)
            ax.scatter(row.out_effect, y + 0.2, marker="D", s=8, facecolor="white",
                       edgecolor=c, linewidths=0.6, zorder=3)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_yticks(range(len(data)), _pair_labels(data))
    ax.tick_params(axis="y", length=0, labelsize=5.6)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_ylim(len(data) - 0.5, -0.5)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("AUC gain (points, 95% CI)", fontsize=8)
    ax.grid(axis="y", visible=False)
    despine(ax, "x")
    _key_legend(ax, [Line2D([], [], marker="D", ls="", color=colours["hypothesis"],
                            markersize=4, label="Source alone"),
                     Line2D([], [], marker="D", ls="", markerfacecolor="white",
                            color=colours["hypothesis"], markersize=4,
                            label="Source left out")], y=-0.3, ncol=2)
    ax.set_title("C  Pairs: alone and left out", loc="left", fontsize=TITLE_SIZE,
                 pad=TITLE_PAD, x=-0.75)

    # D: every domain, between disease groups.
    columns = 4
    rows_n = int(np.ceil(len(domains) / columns))
    grid = fig.add_gridspec(rows_n, columns, left=0.13, right=0.86, top=0.43, bottom=0.10,
                            wspace=0.08, hspace=0.3)
    dense = ~table.target.isin(sparse) & (table.target != table.source)
    scale = table.loc[dense & table.endpoint_group.isin(domains), "effect"]
    vmax = max(float(pair_scale_floor),
               (float(np.nanpercentile(np.abs(scale.dropna()), 98))
                if scale.notna().any() else 1.0))
    if pair_vmax is not None:
        if not np.isfinite(pair_vmax) or pair_vmax <= 0:
            raise ValueError('pair_vmax must be positive and finite')
        vmax = float(pair_vmax)
    image = None
    for i, domain in enumerate(domains):
        ax = fig.add_subplot(grid[i // columns, i % columns])
        left_col, last = i % columns == 0, i // columns == rows_n - 1
        image = _group_heatmap(ax, table.loc[dense & (table.endpoint_group == domain)],
                               groups, vmax, labels=False)
        if left_col:
            ax.set_yticks(range(len(groups)), groups)
        if last:
            ax.set_xticks(range(len(groups)), groups, rotation=90)
        ax.tick_params(labelsize=6.2)
        ax.set_title(SHORT_GROUP.get(domain, domain), fontsize=8, pad=4)
    fig.text(0.04, 0.462, "D  Between disease groups, by outcome domain", fontsize=TITLE_SIZE,
             color=PALETTE["ink"])
    if image is not None:
        cax = fig.add_axes([0.885, 0.17, 0.012, 0.16])
        clipped = scale.notna().any() and np.abs(scale.dropna()).max() > vmax
        colorbar = fig.colorbar(image, cax=cax, extend='both' if clipped else 'neither')
        colorbar.set_label(f"{estimate_label} AUC gain from a source\ngroup alone (points;\nrows: target,"
                           " columns:\nsource)", fontsize=7.2)
        colorbar.ax.tick_params(labelsize=6.5)
        colorbar.outline.set_visible(False)
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(5))
    if ablation_mode is not None:
        if ablation_mode not in {'mean','zero'}:
            raise ValueError('Unknown source-ablation mode')
        replacement = 'tuning-task mean' if ablation_mode == 'mean' else 'zero'
        identity = f" · {model_identity}" if model_identity else ''
        fig.text(.02,.018,f'Fixed fitted model · disease corrections replaced by {replacement} · outcome bank retained{identity}',
                 fontsize=6,color=PALETTE['ink_secondary'])
    watermark(fig, mark)
    return fig


def figure_horizons(cells, *, metric="auc", contrasts=("D-S", "O-S", "J-S", "R1"), mark=None):
    """Supplement: pooling and routed-versus-joint effects over follow-up."""
    fig, axes = plt.subplots(1, len(contrasts), figsize=(max(7.2, 2.15 * len(contrasts)), 3.8), sharey=True,
                             sharex=False, squeeze=False)
    axes = axes.ravel()
    data = cells.loc[cells.metric == metric]
    order = (data.loc[(data.contrast == "J-S") & (data.horizon_days == 365)]
             .groupby("endpoint_group").gain.median().sort_values(ascending=False)
             .index.tolist())
    for index, (ax, contrast) in enumerate(zip(axes, contrasts)):
        part = data.loc[(data.contrast == contrast) & (data.horizon_days > 0)]
        paired_means = "group_mean" in part
        if paired_means:
            summary = part[["endpoint_group", "horizon_days", "group_mean", "group_lower", "group_upper"]].drop_duplicates()
            summary = summary.rename(columns={"group_mean": "median", "group_lower": "q25", "group_upper": "q75"})
            summary = summary.loc[np.isfinite(summary["median"])]
        else:
            summary = _summary_rows(part, ["endpoint_group", "horizon_days"])
        summary["endpoint_group"] = summary.endpoint_group.map(lambda g: SHORT_GROUP.get(g, g))
        handles = _diamond_forest(
            ax, summary, row_key="endpoint_group",
            row_order=[SHORT_GROUP.get(g, g) for g in order],
            series_key="horizon_days",
            series_style={h: (c, f"{h} days") for h, c in HORIZON_COLORS.items()},
            xlabel=f"{METRIC_LABEL[metric]} " + ("(mean, 95% CI)" if paired_means else "(median, IQR)"), show_counts=False)
        if index:
            ax.tick_params(axis="y", labelleft=False)
        _panel_title(ax, chr(65 + index), ("Routed vs joint" if contrast == "R1" else CONTRAST_STYLE[contrast][1]).replace(" vs ", "\nvs "))
        ax.xaxis.set_major_locator(MaxNLocator(3))
    fig.legend(handles=handles, loc="lower left", ncol=6, frameon=False, fontsize=7,
               bbox_to_anchor=(0.2, 0.0))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.22, wspace=0.14)
    watermark(fig, mark)
    return fig


def figure_atlas(atlas, models=None, *, mark=None):
    """Predictability atlas: absolute held-out performance of the joint model.

    ``models`` is accepted for a common recipe signature; the disease-by-outcome
    matrix is the supplementary ``figure_atlas_matrix``.
    """
    fig, (ax, group_ax) = plt.subplots(1, 2, figsize=(7.2, 4.4),
                                       gridspec_kw={"width_ratios": [1.0, 1.15]})
    horizons = sorted(atlas.horizon_days.unique())
    for horizon in horizons:
        part = atlas.loc[(atlas.horizon_days == horizon) & atlas.cif_auc.notna()]
        ax.scatter(part.n_events_train.clip(lower=1), part.cif_auc, s=8, alpha=0.5,
                   color=HORIZON_COLORS[horizon], linewidths=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Training events through horizon (log scale)")
    ax.set_ylabel(f"{_split_word()} AUC")
    ax.set_ylim(max(0.4, atlas.cif_auc.min() - 0.03), 1.0)
    _panel_title(ax, "A", "Outcomes")
    despine(ax, "y")
    summary = _summary_rows(atlas.rename(columns={"cif_auc": "gain"}),
                            ["endpoint_group", "horizon_days"])
    summary["endpoint_group"] = summary.endpoint_group.map(lambda g: SHORT_GROUP.get(g, g))
    order = (summary.loc[summary.horizon_days == 365].sort_values("median",
             ascending=False).endpoint_group.tolist())
    handles = _diamond_forest(
        group_ax, summary, row_key="endpoint_group", row_order=order,
        series_key="horizon_days",
        series_style={h: (c, f"{h} days") for h, c in HORIZON_COLORS.items()},
        xlabel=f"{_split_word()} AUC (median, IQR)", show_counts=True, zero_line=False)
    group_ax.set_xlim(0.5, 1.0)
    _panel_title(group_ax, "B", "Clinical domains")
    _legend(group_ax, handles, loc="lower right", title="Horizon")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.62)
    watermark(fig, mark)
    return fig


ATLAS_CMAP = LinearSegmentedColormap.from_list(
    "opera_auc", ["#F4F1EA", "#C9D3E8", "#7F96C9", "#3E5AA8", "#1F2A6B"])
ATLAS_CMAP.set_bad(PALETTE["missing"])


def figure_atlas_matrix(atlas, models=None, *, horizon=365, vmin=0.55, vmax=0.95,
                        mark=None):
    """Supplement: the joint model's held-out AUC for every outcome in every disease.

    ``models``: one row per cell (``cohort``, ``cohort_group``, ``outcome``,
    ``endpoint_group``, ``horizon_days``, ``J``), i.e. ``model_cells.csv``.
    ``atlas``: outcome-level AUC pooled over diseases at every horizon
    (``outcome``, ``endpoint_group``, ``horizon_days``, ``cif_auc``).
    A: every outcome (rows, by clinical domain) in every disease (columns, by
    disease group) at ``horizon``. B: the same outcomes at every horizon.
    Grey cells had too few events to estimate.
    """
    from pathlib import Path

    import yaml

    from opera.visualization.paper_taxonomy import DISPLAY_GROUPS, disease_group_name

    domain_order = list(yaml.safe_load(Path(DISPLAY_GROUPS).read_text()))
    if models is None:
        models = atlas.rename(columns={"cif_auc": "J"}).assign(
            cohort="All diseases", cohort_group="All")
    year = models.loc[models.horizon_days == horizon]
    lead = "J"  # Both the cell matrix and native pooled strip describe the joint fit.
    matrix = year.pivot_table(index="outcome", columns="cohort", values=lead).reindex(
        index=year.outcome.drop_duplicates(), columns=year.cohort.drop_duplicates())
    domains = (year.drop_duplicates("outcome").set_index("outcome").endpoint_group
               .reindex(matrix.index))
    rank = {d: i for i, d in enumerate(domain_order)}
    rows = sorted(matrix.index, key=lambda o: (rank.get(domains[o], 99),
                                               -np.nanmedian(matrix.loc[o])))
    groups = year.drop_duplicates("cohort").set_index("cohort").cohort_group
    columns = sorted(matrix.columns, key=lambda c: (str(groups.get(c)), str(c)))
    matrix = matrix.reindex(index=rows, columns=columns)
    strip = (atlas.pivot_table(index="outcome", columns="horizon_days", values="cif_auc")
             .reindex(rows))
    height = max(4., 0.115 * len(rows) + 2.3)
    fig = plt.figure(figsize=(7.2, height))
    grid = fig.add_gridspec(1, 3, width_ratios=[len(columns), max(len(strip.columns), 1), 0.6],
                            wspace=0.06, left=0.31, right=0.975,
                            top=1 - 1.15 / height, bottom=1.05 / height)
    norm = plt.Normalize(vmin, vmax)
    ax = fig.add_subplot(grid[0])
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap=ATLAS_CMAP, norm=norm,
                      interpolation="nearest")
    for r, c in zip(*np.nonzero(~np.isfinite(matrix.to_numpy(float)))):
        ax.add_patch(Rectangle((c-.5, r-.5), 1, 1, facecolor=PALETTE['missing'],
                              edgecolor='white', hatch='///', lw=0, zorder=2))
    ax.set_yticks(range(len(rows)), rows, fontsize=5.2)
    ax.set_xticks(range(len(columns)), columns, fontsize=5.6, rotation=90)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Domain bands: separators and labels at the left margin.
    boundaries = [i for i in range(1, len(rows)) if domains[rows[i]] != domains[rows[i - 1]]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=2.2)
    starts = [0, *boundaries]
    ends = [*boundaries, len(rows)]
    for start, stop in zip(starts, ends):
        ax.annotate(SHORT_GROUP.get(domains[rows[start]], domains[rows[start]]),
                    xy=(0, (start + stop - 1) / 2), xycoords=("figure fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", fontsize=6.2,
                    fontweight="bold", va="center", ha="left", color=PALETTE["ink"])
    group_bounds = [i for i in range(1, len(columns))
                    if groups.get(columns[i]) != groups.get(columns[i - 1])]
    for b in group_bounds:
        ax.axvline(b - 0.5, color="white", lw=2.2)
    edges = [0, *group_bounds, len(columns)]
    for start, stop in zip(edges[:-1], edges[1:]):
        name = disease_group_name(str(groups.get(columns[start])))
        ax.text((start + stop - 1) / 2, len(rows) + 0.4, name, ha="center", va="top",
                fontsize=5.4, color=PALETTE["ink_secondary"],
                rotation=0 if stop - start > 2 else 90)
    ax.set_title(f"A  Every outcome in every disease ({horizon // 365} year)" if horizon >= 365
                 else f"A  Every outcome in every disease ({horizon} days)",
                 loc="left", fontsize=8.5, pad=62, color=PALETTE["ink"])

    bx = fig.add_subplot(grid[1], sharey=ax)
    bx.imshow(strip.to_numpy(float), aspect="auto", cmap=ATLAS_CMAP, norm=norm,
              interpolation="nearest")
    bx.set_xticks(range(len(strip.columns)), [f"{d:g}" for d in strip.columns],
                  fontsize=5.6, rotation=90)
    bx.xaxis.tick_top()
    bx.tick_params(length=0, labelleft=False)
    bx.grid(False)
    for b in boundaries:
        bx.axhline(b - 0.5, color="white", lw=2.2)
    for spine in bx.spines.values():
        spine.set_visible(False)
    bx.set_title("B  All diseases,\nby horizon (days)", loc="left", fontsize=8.5, pad=30,
                 color=PALETTE["ink"])
    cax = fig.add_subplot(grid[2])
    cax.set_position([cax.get_position().x0 + 0.01, 0.35, 0.012, 0.3])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label(f"{_split_word()} AUC (" + ("OPERA" if lead == "OPERA" else "joint model")
                  + ")", fontsize=6.6)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)
    fig.text(.31, .012, 'Hatched grey: performance not estimable', fontsize=6.5,
             color=PALETTE['ink_secondary'])
    watermark(fig, mark)
    return fig


STATUS_STYLE = {
    "ineligible/early censor": ("#CDD3DC", "+", "Censored before horizon"),
    "known event-free": ("#168577", "o", "Event-free at horizon"),
    "competing death": ("#482878", "X", "Competing death"),
    "primary event": (PALETTE["positive"], "^", "Event"),
}


def _map_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _embedding_surface_mask(xx, yy, points, support, density, minimum):
    """Mask unsupported colour outside the observed patient cloud, including holes.

    Effective sample size alone is insufficient: equally tiny Gaussian weights
    can give a large effective count far from every patient.
    """
    from opera.visualization.censoring_aware_embeddings import patient_footprint_margin

    if not len(points):
        return np.ones(xx.shape, dtype=bool)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    margin, _ = patient_footprint_margin(points, grid)
    return ((support < minimum) | (density < 0.02 * np.max(density))
            | (margin.reshape(xx.shape) < 0))


def _matched_embedding_tables(coords, outcomes):
    """Canonical IDs across CSV/Parquet; never render a failed join as zero risk."""
    coords, outcomes = coords.copy(), outcomes.copy()
    for frame in (coords, outcomes):
        if frame.subject_id.isna().any():
            raise ValueError("Embedding patient identifiers must be complete")
        frame["subject_id"] = frame.subject_id.astype(str)
    if coords.subject_id.duplicated().any():
        raise ValueError("Embedding coordinates need one row per patient")
    if outcomes.duplicated(["subject_id", "outcome"]).any():
        raise ValueError("Embedding outcomes need one row per patient and outcome")
    if not outcomes.subject_id.isin(coords.subject_id).any():
        raise ValueError("No matching patient IDs between embedding coordinates and outcomes")
    return coords, outcomes


def figure_embedding_gallery(coords, outcomes, *, page=1, per_page=12, ncols=4,
                             horizon=365, grid_size=90, min_effective_support=25.0,
                             bandwidth=None, mode="predicted_points", mark=None):
    """Patient-level final predictions, or observed risk, in a shared embedding.

    ``coords``: ``subject_id``, ``x``, ``y`` (the selected checkpoint's shared projection).
    Default colours are final-model patient risk, one point per patient, as in
    the gallery the author approved ("keep original", review of 30 October).
    ``mode="observed_surface"`` restores the observed-outcome kernel diagnostic.
    ``outcomes``: long table with ``subject_id``, ``outcome`` (display name),
    ``domain``, ``domain_order``, ``time_days`` and ``event``. Each panel is the
    IPCW kernel estimate of the observed risk at the horizon among the
    patients evaluated for that outcome (as in the calibration maps), divided
    by the outcome's overall risk, so outcomes with very different event rates
    share one colour scale. Outcomes are ordered by clinical domain; a bracket
    above each run of panels names its domain. ``page`` selects ``per_page``.
    """
    from matplotlib.colors import LogNorm
    from opera.visualization.censoring_aware_embeddings import (
        _default_bandwidth, _display_grid, _kernel_surfaces, horizon_ipcw_data)

    if mode not in {"predicted_points", "observed_surface"}:
        raise ValueError("Choose predicted_points or observed_surface")
    coords, outcomes = _matched_embedding_tables(coords, outcomes)
    if mode == "predicted_points" and "risk" not in outcomes:
        raise ValueError("Predicted point atlas requires final-model risk exports")
    coords = coords.sort_values("subject_id").reset_index(drop=True)
    points = coords[["x", "y"]].to_numpy(float)
    bandwidth = bandwidth if bandwidth is not None else 1.4 * _default_bandwidth(points)
    xx, yy, _ = _display_grid(points, grid_size, min_padding=0.8)
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    order = (outcomes[["outcome", "domain", "domain_order"]].drop_duplicates()
             .sort_values(["domain_order", "outcome"]))
    selected = order.iloc[(page - 1) * per_page: page * per_page].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"Page {page} has no outcomes")
    nrows = int(np.ceil(len(selected) / ncols))
    fig = plt.figure(figsize=(7.2, 2.0 * nrows + 0.9))
    height = fig.get_figheight()
    grid = fig.add_gridspec(nrows, ncols, wspace=0.06, hspace=0.55, left=0.02,
                            right=0.98, top=1 - 0.65 / height, bottom=0.62 / height)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = plt.Normalize(0, 1) if mode == "predicted_points" else LogNorm(vmin=0.25, vmax=4)
    image = None
    position = dict(zip(coords.subject_id, range(len(coords))))
    axes = []
    for index, row in selected.iterrows():
        ax = fig.add_subplot(grid[index // ncols, index % ncols])
        axes.append(ax)
        _map_axis(ax)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect("equal")
        part = outcomes.loc[(outcomes.outcome == row.outcome)
                            & outcomes.subject_id.isin(position)]
        at = part.subject_id.map(position).to_numpy()
        data = horizon_ipcw_data(part.time_days.to_numpy(float),
                                 part.event.to_numpy(float), horizon)
        used = data.eligible
        overall = (np.sum(data.weights[used] * data.target[used])
                   / max(np.sum(data.weights[used]), 1e-12))
        if mode == "predicted_points":
            risk = part.risk.to_numpy(float)
            if not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any():
                raise ValueError("Predicted atlas risks must be finite probabilities")
            image = ax.scatter(points[at, 0], points[at, 1], c=risk, s=5,
                               cmap=cmap, norm=norm, linewidths=0, rasterized=True, zorder=4)
        elif used.sum() and overall > 0:
            _, _, surface, support, density = _kernel_surfaces(
                points[at][used], data.target[used], data.weights[used],
                bandwidth=bandwidth, grid_size=grid_size, bounds_coords=points,
                min_padding=0.8)
            relative = np.clip(surface / overall, 0.25, 4)
            relative[_embedding_surface_mask(xx, yy, points[at][used], support,
                                              density, min_effective_support)] = np.nan
            image = ax.imshow(relative, origin="lower", extent=extent, cmap=cmap,
                              norm=norm, interpolation="bilinear", zorder=2)
        ax.scatter(points[:, 0], points[:, 1], s=1.4, alpha=0.22, linewidths=0,
                   color=PALETTE["ink_secondary"], rasterized=True, zorder=3)
        if not used.any():
            ax.text(0.5, 0.5, "No eligible patients", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7)
        elif overall == 0:
            ax.text(0.5, 0.5, "No events at horizon", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7)
        ax.set_title(f"{_embedding_title(row.outcome, width=29)}\n{100 * overall:.1f}% observed at {horizon} days",
                     fontsize=6.4, pad=2, color=PALETTE["ink"], loc="center",
                     linespacing=1.25)
    # Domain brackets: one labelled rule over each run of panels in a row.
    fig.canvas.draw()
    lift = 0.38 / height
    for row_start in range(0, len(selected), ncols):
        members = selected.iloc[row_start:row_start + ncols]
        runs = (members.domain != members.domain.shift()).cumsum()
        for _, run in members.groupby(runs):
            first = axes[run.index[0]].get_position()
            last = axes[run.index[-1]].get_position()
            y = first.y1 + lift
            fig.add_artist(Line2D([first.x0, last.x1], [y, y], transform=fig.transFigure,
                                  color=PALETTE["ink_secondary"], lw=0.7))
            fig.text(first.x0, y + 0.03 / height,
                     SHORT_GROUP.get(run.domain.iloc[0], run.domain.iloc[0]), fontsize=6.6,
                     fontweight="bold", color=PALETTE["ink"], va="bottom", ha="left")
    if image is not None:
        bar = fig.add_axes([0.3, 0.22 / height, 0.4, 0.1 / height])
        cb = fig.colorbar(image, cax=bar, orientation="horizontal")
        if mode == "predicted_points":
            cb.set_ticks([0, .25, .5, .75, 1])
        else:
            cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=["¼×", "½×", "average", "2×", "4×"])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6)
        cb.set_label(f"Final-model predicted {horizon}-day risk (one point per patient)"
                     if mode == "predicted_points" else
                     f"Observed {horizon}-day risk (IPCW kernel) relative to the outcome's average",
                     fontsize=6.4)
    watermark(fig, mark)
    return fig


def _embedding_title(name, width=19):
    """Compact, wrapped map labels; endpoint definitions stay unchanged."""
    compact = {"Infection diagnosis or antimicrobial use": "Infection diagnosis / antimicrobial use",
               "Creatinine increased (G2+)": "Creatinine increase (G2+)"}
    return textwrap.fill(compact.get(str(name), str(name)), width=width,
                         break_long_words=False, break_on_hyphens=False)


def figure_patient_map(coords, outcomes, *, highlights=None, horizon=365, grid_size=110,
                       min_effective_support=25.0, disease_contours=True,
                       disease_centroids=True, calibration_style='surface',
                       calibration_outcome=None, mode="observed_surface", mark=None):
    """Where risk lies in the selected model's shared patient representation.

    ``coords``: ``subject_id``, ``x``, ``y`` and ``disease``; ``outcomes`` as for
    :func:`figure_embedding_gallery`, with predicted ``risk`` for the residual
    panel. A: disease groups on the map. B-F: patient-level final-model
    predicted risks by default; mode="observed_surface" gives observed risk
    relative to each outcome's average (IPCW kernel). G: observed minus predicted risk of the selected model for the
    selected calibration outcome (overall survival when available).
    """
    from matplotlib.colors import LogNorm
    from opera.visualization.censoring_aware_embeddings import (
        _default_bandwidth, _display_grid, _kernel_surfaces, horizon_ipcw_data,
        spatial_calibration_residual)
    from opera.visualization.style import CATEGORICAL

    if mode not in {"predicted_points", "observed_surface"}:
        raise ValueError("Choose predicted_points or observed_surface")
    if calibration_style not in {'points','surface'}:
        raise ValueError('Calibration style must be points or surface')
    coords, outcomes = _matched_embedding_tables(coords, outcomes)
    if mode == "predicted_points" and "risk" not in outcomes:
        raise ValueError("Predicted patient map requires risk exports")
    coords = coords.sort_values("subject_id").reset_index(drop=True)
    points = coords[["x", "y"]].to_numpy(float)
    # Tighter than the gallery so the maps follow the shape of the patient map.
    bandwidth = 1.4 * _default_bandwidth(points)
    xx, yy, _ = _display_grid(points, grid_size, min_padding=0.8)
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    if highlights is None:
        if "highlight_order" in outcomes and outcomes.highlight_order.notna().any():
            highlights = (outcomes.loc[outcomes.highlight_order.notna(),
                                      ["outcome", "highlight_order"]].drop_duplicates()
                          .sort_values("highlight_order").outcome.tolist())
        else:
            # Compatibility with earlier exports lacking explicit selection.
            from opera.visualization.outcome_names import outcome_display_name
            preferred = [outcome_display_name(name) for name in (
                "infection_diagnosis_or_antimicrobial_use_composite", "overall_survival",
                "neutropenia_g3plus", "hospitalisation", "creatinine_increased_g2plus")]
            present = set(outcomes.outcome)
            highlights = [name for name in preferred if name in present]
            # Fill slots if older exports use different names for preferred endpoints.
            highlights += [name for name in outcomes.outcome.drop_duplicates()
                           if name not in highlights][:5-len(highlights)]
    highlights = list(highlights)[:5]
    if not highlights:
        raise ValueError('Patient map requires at least one highlighted outcome')
    preferred_calibration = calibration_outcome or next(
        (name for name in highlights if name in {'overall_survival','Overall survival'}), highlights[0])
    if preferred_calibration not in highlights:
        raise ValueError('Calibration outcome must be one of the highlighted outcomes')
    position = dict(zip(coords.subject_id, range(len(coords))))
    n_diseases = coords.disease.nunique() if 'disease' in coords else 0
    extra_legend_height = max(0, int(np.ceil(n_diseases/2))-5) * .17
    figure_height = 6.5 + extra_legend_height
    bottom = (1.5 + extra_legend_height) / figure_height
    fig = plt.figure(figsize=(9.2, figure_height))
    grid = fig.add_gridspec(2, 5, width_ratios=[1.1, 1.1, 1, 1, 1], wspace=0.18, hspace=0.32,
                            left=0.025, right=0.99, top=0.91, bottom=bottom)

    def frame(ax):
        _map_axis(ax)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect("equal")

    ax = fig.add_subplot(grid[:, :2])
    frame(ax)
    names = sorted(coords.disease.dropna().unique(), key=str) if "disease" in coords else []
    palette = CATEGORICAL + ["#BD648E", "#697481", "#B18B4D", "#5C7AEA"]
    if len(names) > len(palette):
        from matplotlib.colors import to_hex
        palette += [to_hex(color) for color in plt.get_cmap('tab20').colors
                    if to_hex(color).lower() not in {c.lower() for c in palette}]
    if len(names) > len(palette):
        raise ValueError('Use disease facets for more categories than the map palette supports')
    colors = {name: palette[i] for i, name in enumerate(names)}
    if names:
        _category_regions(ax, coords.rename(columns={"disease": "group"}), "group", colors,
                          extent, contours=disease_contours)
        if disease_centroids:
            _label_disease_centroids(ax,coords,names,colors)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[n],
                                  label=f'{i+1} {n}' if disease_centroids else n,
                                  markersize=4) for i,n in enumerate(names)],
                  loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2, frameon=False,
                  fontsize=7.5, handletextpad=0.3, columnspacing=0.9)
    _panel_title(ax, "A", "Disease groups")

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = plt.Normalize(0, 1) if mode == "predicted_points" else LogNorm(vmin=0.25, vmax=4)
    image, residual_image = None, None
    slots = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3)]
    first = None
    for index, (outcome, (r, c)) in enumerate(zip(highlights, slots)):
        ax = fig.add_subplot(grid[r, c])
        frame(ax)
        part = outcomes.loc[(outcomes.outcome == outcome)
                            & outcomes.subject_id.isin(position)]
        at = part.subject_id.map(position).to_numpy()
        data = horizon_ipcw_data(part.time_days.to_numpy(float),
                                 part.event.to_numpy(float), horizon)
        used = data.eligible
        overall = (np.sum(data.weights[used] * data.target[used])
                   / max(np.sum(data.weights[used]), 1e-12))
        if used.sum() and overall > 0:
            _, _, surface, support, density = _kernel_surfaces(
                points[at][used], data.target[used], data.weights[used],
                bandwidth=bandwidth, grid_size=grid_size, bounds_coords=points,
                min_padding=0.8)
            relative = np.clip(surface / overall, 0.25, 4)
            relative[_embedding_surface_mask(xx, yy, points[at][used], support,
                                              density, 2 * min_effective_support)] = np.nan
            if mode == "observed_surface":
                image = ax.imshow(relative, origin="lower", extent=extent, cmap=cmap,
                                  norm=norm, interpolation="bilinear", zorder=2)
            if first is None or outcome == preferred_calibration:
                first = (outcome, part, at, data, support, density)
        if mode == "predicted_points":
            image = ax.scatter(points[at, 0], points[at, 1], c=part.risk.to_numpy(float),
                               s=5, cmap=cmap, norm=norm, linewidths=0, rasterized=True, zorder=4)
        ax.scatter(points[:, 0], points[:, 1], s=1.4, alpha=0.2, linewidths=0,
                   color=PALETTE["ink_secondary"], rasterized=True, zorder=3)
        if not used.any() or overall == 0:
            ax.text(0.5, 0.5, "No eligible patients" if not used.any()
                    else "No events at horizon", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6)
        domain = part.domain.iloc[0] if len(part) else ""
        _panel_title(ax, 'BCDEF'[index], _embedding_title(outcome, width=18), pad=5)
        ax.text(0.5, -0.04, SHORT_GROUP.get(domain, domain), transform=ax.transAxes,
                ha="center", va="top", fontsize=6, color=PALETTE["ink_secondary"])

    ax = fig.add_subplot(grid[1, 4])
    frame(ax)
    if first is not None and "risk" in outcomes:
        outcome, part, at, data, support, density = first
        residual = spatial_calibration_residual(
            points[at], part.risk.to_numpy(float), data.target, data.weights,
            bandwidth=bandwidth, grid=(xx, yy), support_mask=support >= min_effective_support,
            min_effective_support=min_effective_support, standardize=False,
            n_bootstrap=0).residual.reshape(xx.shape)
        residual[_embedding_surface_mask(xx, yy, points[at][data.eligible], support,
                                          density, min_effective_support)] = np.nan
        rcmap = plt.get_cmap("RdBu_r").copy()
        rcmap.set_bad((1, 1, 1, 0))
        finite_residual = residual[np.isfinite(residual)]
        residual_limit = max(.1, 1.02 * float(np.abs(finite_residual).max(initial=0)))
        if calibration_style == 'surface':
            residual_image = ax.imshow(residual, origin="lower", extent=extent, cmap=rcmap,
                                       vmin=-residual_limit, vmax=residual_limit, interpolation="bilinear", zorder=2)
        else:
            local = _surface_at_patients(xx, yy, residual, points)
            residual_image = ax.scatter(points[:,0], points[:,1], c=local, cmap=rcmap,
                                        vmin=-residual_limit, vmax=residual_limit, s=5, linewidths=0,
                                        rasterized=True, zorder=2)
        _panel_title(ax, 'G', 'Calibration', pad=5)
        ax.text(0.5, -0.04, textwrap.shorten(outcome, 26, placeholder="…"),
                transform=ax.transAxes, ha="center", va="top", fontsize=6,
                color=PALETTE["ink_secondary"])
    if image is not None:
        bar = fig.add_axes([0.46, 0.105, 0.27, 0.018])
        cb = fig.colorbar(image, cax=bar, orientation="horizontal")
        if mode == "predicted_points":
            cb.set_ticks([0, .25, .5, .75, 1])
        else:
            cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=["¼×", "½×", "average", "2×", "4×"])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6)
        cb.set_label("B–F: final-model predicted risk" if mode == "predicted_points"
                     else "B–F: observed risk relative to the outcome's average", fontsize=6.4)
    if residual_image is not None:
        bar = fig.add_axes([0.8, 0.1, 0.17, 0.018])
        cb = fig.colorbar(residual_image, cax=bar, orientation="horizontal")
        cb.ax.tick_params(labelsize=6)
        cb.set_label("G: observed − predicted", fontsize=6.4)
    watermark(fig, mark)
    return fig


def figure_patient_disease_groups(coords, *, ncols=4, mark=None):
    """Each observed disease on the identical map, with everyone else in grey.

    This companion avoids trying to distinguish many overlapping categories by
    colour alone. It neither refits the projection nor invents disease regions.
    """
    from opera.visualization.style import CATEGORICAL
    required = {'subject_id', 'x', 'y', 'disease'}
    if not required <= set(coords) or coords.subject_id.duplicated().any():
        raise ValueError('Disease facets need one labelled coordinate per patient')
    names = sorted(coords.disease.dropna().unique(), key=str)
    if not names:
        raise ValueError('No disease labels for patient-map facets')
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 2.25*nrows+.5),
                              squeeze=False)
    fig.subplots_adjust(left=.035, right=.985, bottom=.07, top=.92,
                         wspace=.12, hspace=.38)
    for i, ax in enumerate(axes.ravel()):
        _map_axis(ax)
        if i >= len(names):
            ax.set_visible(False)
            continue
        selected = coords.disease == names[i]
        ax.scatter(coords.x, coords.y, s=2, c=PALETTE['grid'], linewidths=0,
                   rasterized=True)
        ax.scatter(coords.loc[selected, 'x'], coords.loc[selected, 'y'], s=5,
                   c=CATEGORICAL[i % len(CATEGORICAL)], alpha=.65, linewidths=0,
                   rasterized=True)
        ax.set_xlim(coords.x.min(), coords.x.max())
        ax.set_ylim(coords.y.min(), coords.y.max())
        ax.set_aspect('equal')
        ax.set_title(textwrap.fill(str(names[i]), 24) + f'\n(n={selected.sum():,})',
                     loc='left', fontsize=TITLE_SIZE)
    watermark(fig, mark)
    return fig


def _surface_at_patients(xx, yy, surface, points):
    """Sample the same supported kernel estimate; not binary residuals."""
    from scipy.interpolate import RegularGridInterpolator
    interpolate=RegularGridInterpolator((yy[:,0],xx[0,:]),surface,
                                        bounds_error=False,fill_value=np.nan)
    return interpolate(np.asarray(points)[:,[1,0]])


def _label_disease_centroids(ax, coords, names, colors):
    """Numbered median locations with spaced callouts, matching the legend."""
    centres=[]
    for i,name in enumerate(names):
        part=coords.loc[coords.disease==name]
        centres.append((i,name,float(part.x.median()),float(part.y.median())))
    ordered=sorted(centres,key=lambda r:r[2])
    midpoint=(len(ordered)+1)//2
    for side,items in [('left',ordered[:midpoint]),('right',ordered[midpoint:])]:
        for row,y in zip(sorted(items,key=lambda r:r[3]),np.linspace(.12,.88,len(items))):
            i,name,x0,y0=row
            ax.scatter(x0,y0,s=34,marker='D',color=colors[name],edgecolor='white',
                       linewidth=.8,zorder=5)
            ax.annotate(str(i+1),xy=(x0,y0),xytext=(.03 if side=='left' else .97,y),
                        textcoords='axes fraction',ha='left' if side=='left' else 'right',
                        va='center',fontsize=8,fontweight='bold',color=colors[name],
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1),
                        arrowprops=dict(arrowstyle='-',color=colors[name],lw=.6),zorder=6)


def figure_calibration_maps(coords, outcomes, *, horizon=365, bandwidth=None,
                            grid_size=140, min_effective_support=25.0,
                            residual_limit=None, router_bins=None, router_min_count=20,
                            router_min_coherence=0.3, router_arrow_scale=None,
                            include_rows=None, calibration_style='surface', mark=None):
    """Status, router movement, observed risk, final predictions and calibration.

    ``outcomes`` maps a display name to ``(times, events, predicted_cif)`` or
    ``(times, events, predicted_cif, routed_coords)``, aligned with ``coords``.
    Every column uses the same checkpoint's single patient embedding, so columns
    differ only in the outcome. ``routed_coords`` is the same projection applied
    to the routed, target-specific representation; when any outcome supplies it,
    a row of local mean displacement arrows is added with a common, labelled
    magnification across columns (no per-outcome length normalization).
    These illustrate projected readout corrections, not causal patient changes.
    Regions with too little
    effective support are left blank rather than extrapolated.
    """
    from opera.visualization.censoring_aware_embeddings import (
        _binned_router_displacement, _default_bandwidth, _display_grid,
        _kernel_surfaces, horizon_ipcw_data, spatial_calibration_residual,
    )

    coords = np.asarray(coords, float)
    if calibration_style not in {'points','surface'}:
        raise ValueError('Calibration style must be points or surface')
    bandwidth = bandwidth if bandwidth is not None else 1.4 * _default_bandwidth(coords)
    xx, yy, _ = _display_grid(coords, grid_size, min_padding=0.8)
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    names = list(outcomes)
    has_router = any(len(value) > 3 and value[3] is not None
                     for value in outcomes.values())
    if router_arrow_scale is None:
        lengths = [np.linalg.norm(np.asarray(v[3], float) - coords, axis=1)
                   for v in outcomes.values() if len(v) > 3 and v[3] is not None]
        positive = np.concatenate(lengths) if lengths else np.array([])
        positive = positive[np.isfinite(positive) & (positive > 0)]
        router_arrow_scale = (float(np.clip(.09 * np.ptp(coords, axis=0).max() /
                                           np.median(positive), 1, 20)) if len(positive) else 1.)
    if not np.isfinite(router_arrow_scale) or router_arrow_scale <= 0:
        raise ValueError('Router arrow scale must be positive and finite')
    rows = list(include_rows or ["status", *(["router"] if has_router else []), "risk", "prediction", "residual"])
    if not rows or set(rows) - {'status', 'router', 'risk', 'prediction', 'residual'}:
        raise ValueError('Unknown embedding row')
    titles = {"status": "Status at horizon", "router": "Projected router\ncorrection",
              "risk": "Observed risk\n(IPCW kernel)", "prediction": "Predicted risk\n(final model)",
              "residual": "Calibration residual"}
    fig = plt.figure(figsize=(max(8.8, 1.5 * len(names) + 1.3), max(3.2, 1.85 * len(rows) + 0.65)))
    grid = fig.add_gridspec(len(rows), len(names) + 1,
                            width_ratios=[1] * len(names) + [0.06],
                            wspace=0.34, hspace=0.32, left=0.10, right=0.91,
                            top=0.91, bottom=0.07)
    risk_cmap = plt.get_cmap("plasma").copy()
    risk_cmap.set_bad((1, 1, 1, 0))
    residual_cmap = plt.get_cmap("RdBu_r").copy()
    residual_cmap.set_bad((1, 1, 1, 0))
    images = {}
    residual_images = []
    initial_limit = .2 if residual_limit is None else float(residual_limit)
    def backdrop(ax, alpha=0.08, zorder=1, color=PALETTE["ink_secondary"]):
        ax.scatter(coords[:, 0], coords[:, 1], s=1.5, alpha=alpha, linewidths=0,
                   color=color, rasterized=True, zorder=zorder)

    for column, name in enumerate(names):
        times, events, predicted, *rest = outcomes[name]
        routed = rest[0] if rest else None
        data = horizon_ipcw_data(times, events, horizon)
        used = data.eligible
        _, _, surface, support, density = _kernel_surfaces(
            coords[used], data.target[used], data.weights[used], bandwidth=bandwidth,
            grid_size=grid_size, bounds_coords=coords, min_padding=0.8)
        blank = _embedding_surface_mask(xx, yy, coords[used], support, density,
                                        min_effective_support)
        residual = spatial_calibration_residual(
            coords, predicted, data.target, data.weights, bandwidth=bandwidth,
            grid=(xx, yy), support_mask=~blank,
            min_effective_support=min_effective_support, standardize=False,
            n_bootstrap=0).residual.reshape(xx.shape)
        for row, kind in enumerate(rows):
            ax = fig.add_subplot(grid[row, column])
            _map_axis(ax)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_aspect("equal")
            if kind == "status":
                for status, (color, marker, _) in STATUS_STYLE.items():
                    selected = data.status == status
                    ax.scatter(coords[selected, 0], coords[selected, 1], s=4,
                               marker=marker, color=color, alpha=0.75, linewidths=0.4
                               if marker in "+X" else 0, rasterized=True)
            elif kind == "router":
                backdrop(ax, 0.1)
                if routed is not None:
                    arrows = _binned_router_displacement(
                        coords, np.asarray(routed, float), bins=router_bins,
                        min_count=router_min_count)
                    keep = arrows["coherence"] >= router_min_coherence
                    vectors = arrows["vectors"][keep]
                    if len(vectors) and np.any(np.linalg.norm(vectors, axis=1) > 0):
                        ax.quiver(arrows["centers"][keep, 0], arrows["centers"][keep, 1],
                                  vectors[:, 0] * router_arrow_scale, vectors[:, 1] * router_arrow_scale,
                                  color=MODEL_STYLE["R"][0], angles="xy", scale_units="xy",
                                  scale=1, width=0.007, headwidth=4.0, zorder=3)
                    else:
                        ax.text(.5, .5, "No coherent local shift\nat this resolution",
                                transform=ax.transAxes, ha="center", va="center", fontsize=6)
                else:
                    ax.text(0.5, 0.5, "Router vectors\nnot exported", transform=ax.transAxes,
                            ha="center", va="center", fontsize=6.5,
                            color=PALETTE["ink_muted"])
            elif kind == "prediction":
                images["prediction"] = ax.scatter(coords[:, 0], coords[:, 1], c=predicted,
                    s=5, cmap=risk_cmap, norm=plt.Normalize(0, 1), linewidths=0, rasterized=True)
            elif kind == "risk":
                images["risk"] = ax.imshow(
                    np.ma.masked_where(blank, surface), origin="lower", extent=extent,
                    cmap=risk_cmap, norm=plt.Normalize(0, 1), interpolation="bilinear",
                    zorder=2)
                backdrop(ax, 0.18, zorder=3, color="white")
            else:
                shown=np.where(blank | ~np.isfinite(residual),np.nan,residual)
                if calibration_style == 'surface':
                    images['residual']=ax.imshow(shown,origin='lower',extent=extent,
                        cmap=residual_cmap,norm=TwoSlopeNorm(0,-initial_limit,initial_limit),
                        interpolation='bilinear',zorder=2)
                else:
                    images['residual']=ax.scatter(coords[:,0],coords[:,1],
                        c=_surface_at_patients(xx,yy,shown,coords),cmap=residual_cmap,
                        norm=TwoSlopeNorm(0,-initial_limit,initial_limit),s=5,
                        linewidths=0,rasterized=True,zorder=2)
                residual_images.append(images['residual'])
            if row == 0:
                ax.set_title(_embedding_title(name, width=18), fontsize=7.2, loc="center", pad=7)
            if column == 0:
                ax.set_ylabel(titles[kind], fontsize=7.6)
                ax.text(-0.30, 1.0, "ABCDE"[row], transform=ax.transAxes, fontsize=10.5,
                        va="bottom", color=PALETTE["ink"])
    if residual_images and residual_limit is None:
        maximum = max((np.abs(im.get_array().compressed()).max(initial=0)
                       for im in residual_images), default=.1)
        shared_limit = max(.1, float(maximum) * 1.02)
        for im in residual_images:
            im.set_clim(-shared_limit, shared_limit)
    for row, kind in enumerate(rows):
        cax = fig.add_subplot(grid[row, -1])
        if kind in images:
            clipped = kind == 'residual' and any(
                np.any(np.abs(im.get_array().compressed()) > im.norm.vmax)
                for im in residual_images)
            bar = fig.colorbar(images[kind], cax=cax,
                                extend='both' if clipped else 'neither')
            bar.set_label("Cumulative\nincidence" if kind in {"risk", "prediction"}
                          else "Observed −\npredicted", fontsize=7)
            bar.ax.tick_params(labelsize=6.5)
            bar.outline.set_visible(False)
            bar.ax.yaxis.set_major_locator(MaxNLocator(5))
        else:
            cax.set_axis_off()
    handles = [Line2D([], [], ls="none", marker=marker, color=color, markersize=4.5,
                      label=label) for color, marker, label in
               reversed(list(STATUS_STYLE.values()))] if 'status' in rows else []
    if has_router and 'router' in rows:
        handles.append(Line2D([], [], color=MODEL_STYLE["R"][0], lw=1.4,
                              label=f"Router correction (×{router_arrow_scale:.1f})"))
    if handles:
        fig.legend(handles=handles, loc="upper center", ncol=min(3, len(handles)), frameon=False,
                   fontsize=6.8, bbox_to_anchor=(0.5, 0.995), handletextpad=0.3,
                   columnspacing=1.0)
    watermark(fig, mark)
    return fig


def figure_router_arrows(coords, outcomes, *, mark=None, **options):
    """Larger companion view of the same projected corrections and scale."""
    return figure_calibration_maps(coords, outcomes, include_rows=['router'],
                                   mark=mark, **options)


ROUTE_CMAP = LinearSegmentedColormap.from_list(
    "opera_route", ["#F7F7F5", "#C9CFE6", "#7189BC", "#2D3A8C", "#17212B"]
)
BANK_CMAP = LinearSegmentedColormap.from_list(
    "opera_bank", ["#C16A58", "#EBC4B8", "#F7F7F5", "#B3D3D0", "#3E8A87"]
)


SHORT_DISEASE = {"Amyloidosis": "AL", "Transformed FL": "tFL",
                 "Solitary plasmacytoma": "SP"}


def _home_masked(ax, matrix, cmap, *, label, fig, fraction):
    """Draw source shares, hatching home sources by identity on either axis."""
    values = matrix.to_numpy(float).copy()
    home = np.equal.outer(matrix.index.to_numpy(), matrix.columns.to_numpy())
    off = np.ma.masked_where(home | ~np.isfinite(values), values)
    image = ax.imshow(off, cmap=cmap, vmin=0,
                      vmax=max(float(off.max()), 1e-12) if off.count() else 1.,
                      aspect="equal", interpolation="nearest")
    for row, column in zip(*np.nonzero(home)):
        ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, facecolor="white",
                                   edgecolor=PALETTE["ink_muted"], hatch="//////",
                                   linewidth=0, zorder=3))
    bar = fig.colorbar(image, ax=ax, fraction=fraction, pad=0.02, shrink=0.7,
                       anchor=(0.0, 1.0))
    bar.set_label(label, fontsize=6.6)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)
    bar.ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_anchor("N")
    return float(np.nanmean(values[home]))


def _route_weights(routes):
    column = "mean" if "mean" in routes else "weight"
    result = routes.rename(columns={column: "w"}).copy()
    for identity in ['model', 'seed', 'split']:
        if identity in result and result[identity].nunique() > 1:
            raise ValueError(f'Router usage requires a single {identity}')
    if not np.isfinite(result.w).all() or not result.w.between(0, 1).all():
        raise ValueError('Invalid router gate weights')
    if result.duplicated(['cohort', 'outcome', 'bank', 'source']).any():
        raise ValueError('Duplicate router gate summary rows')
    if set(result.bank) == {'shared', 'disease', 'outcome'}:
        total = result.groupby(['cohort', 'outcome']).w.sum()
        if not np.allclose(total, 1., atol=1e-5):
            raise ValueError('Router gate weights do not sum to one within target context')
        counts = result.groupby(['cohort', 'outcome', 'bank']).source.nunique().unstack()
        if any(counts[bank].nunique() != 1 for bank in counts):
            raise ValueError('Router source inventories differ between target contexts')
    return result


def borrowed_source_breadth(routes):
    """Entropy of mean borrowed gates; not mean patient-level entropy."""
    borrowed = routes.loc[((routes.bank == 'disease') & (routes.source != routes.cohort))
                          | ((routes.bank == 'outcome') & (routes.source != routes.outcome))].copy()
    keys = ['cohort', 'outcome', 'target_group']
    total = borrowed.groupby(keys).w.transform('sum')
    borrowed['p'] = borrowed.w / total.where(total > 0)
    borrowed['entropy'] = -borrowed.p * np.log(borrowed.p.where(borrowed.p > 0))
    entropy = borrowed.groupby(keys).entropy.sum(min_count=1).reset_index()
    entropy['gain'] = np.exp(entropy.entropy)
    return entropy


def figure_router_usage(routes, disease_groups, outcome_groups, *, mark=None):
    """Which patients use which adapter sources, and how the two banks interact.

    ``routes`` is the gate summary written by the analysis (``route_summaries``):
    patient fine disease ``cohort``, target ``outcome``, ``bank`` (shared,
    disease or outcome), ``source`` and the mean gate weight. ``disease_groups``
    and ``outcome_groups`` map fine diseases and outcomes to their groups.
    Gate weights describe how corrections are allocated, not their predictive
    contribution. A/B show within-bank shares. C retains the actual mass
    assigned to each bank. D describes mean gates rather than patient gates.
    """
    routes = _route_weights(routes).copy()
    routes["patient_group"] = routes.cohort.map(disease_groups)
    routes["target_group"] = routes.outcome.map(outcome_groups)
    if routes[['patient_group', 'target_group']].isna().any().any():
        raise ValueError('Disease/outcome groups must cover every router target')
    fig = plt.figure(figsize=(9.2, 8.4))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.78], wspace=0.72, left=0.14,
                           right=0.93, top=0.955, bottom=0.55)
    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.82], wspace=0.75, left=0.14,
                              right=0.97, top=0.39, bottom=0.17)

    # A: disease bank, patient disease -> source disease (row-normalised).
    ax = fig.add_subplot(top[0])
    from opera.visualization.paper_taxonomy import (
        fine_cohort_groups, disease_name, disease_group_name)
    source_groups = {disease_name(k): disease_group_name(v)
                     for k, v in fine_cohort_groups().items()}
    source_groups.update(disease_groups)
    diseases = sorted(routes.cohort.unique(), key=lambda d: (disease_groups[d], d))
    disease = routes.loc[routes.bank == "disease"].copy()
    missing_sources = set(disease.source) - set(source_groups)
    if missing_sources:
        raise ValueError(f'Disease groups do not cover router sources: {sorted(missing_sources)}')
    sources = sorted(disease.source.unique(), key=lambda d: (source_groups[d], d))
    disease['share'] = disease.w / disease.groupby(['cohort', 'outcome']).w.transform('sum').replace(0, np.nan)
    matrix = (disease.groupby(["cohort", "source"]).share.mean().unstack()
              .reindex(index=diseases, columns=sources))
    _home_masked(ax, matrix, ROUTE_CMAP, fig=fig, fraction=0.045,
                 label="Share of disease-bank\nweight (home hatched)")
    breaks, _ = _group_breaks(diseases, disease_groups)
    for position in breaks:
        ax.axhline(position, color="white", lw=1.4, zorder=4)
    source_breaks, _ = _group_breaks(sources, source_groups)
    for position in source_breaks:
        ax.axvline(position, color="white", lw=1.4, zorder=4)
    ax.set_yticks(range(len(diseases)), diseases, fontsize=5.8)
    ax.set_xticks(range(len(sources)), [SHORT_DISEASE.get(d, d) for d in sources],
                  fontsize=5.8, rotation=90)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Disease source", fontsize=7.5)
    ax.set_ylabel("Patient's disease", fontsize=7.5)
    _panel_title(ax, "A", "Disease bank")

    # B: outcome bank, target group -> source group (row-normalised).
    ax = fig.add_subplot(top[1])
    groups = sorted(set(outcome_groups.values()))
    outcome = routes.loc[routes.bank == "outcome"].copy()
    outcome["source_group"] = outcome.source.map(outcome_groups)
    if outcome.source_group.isna().any():
        raise ValueError('Outcome groups must cover every router source')
    outcome['share'] = outcome.w / outcome.groupby(['cohort', 'outcome']).w.transform('sum').replace(0, np.nan)
    per_target = (outcome.groupby(['cohort', "outcome", "target_group", "source_group"]).share.sum()
                  .reset_index())
    matrix = (per_target.groupby(["target_group", "source_group"]).share.mean().unstack()
              .reindex(index=groups, columns=groups))
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    _home_masked(ax, matrix, ROUTE_CMAP, fig=fig, fraction=0.05,
                 label="Share of outcome-bank\nweight (own group hatched)")
    labels = [SHORT_GROUP.get(g, g) for g in groups]
    ax.set_yticks(range(len(groups)), labels, fontsize=6.2)
    source_counts = outcome.groupby('source_group').source.nunique()
    ax.set_xticks(range(len(groups)), [f'{label} ({source_counts.get(g, 0)})'
                  for g, label in zip(groups, labels)], fontsize=6.2, rotation=90)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Outcome source group (number of source slots)", fontsize=7.5)
    ax.set_ylabel("Target outcome group", fontsize=7.5)
    _panel_title(ax, "B", "Outcome bank")

    # C: actual allocation to borrowed disease/outcome sources and the remainder.
    # Unequal banks do not have a 50:50 reference, so do not imply one.
    ax = fig.add_subplot(bottom[0])
    borrowed = routes.loc[((routes.bank == 'disease') & routes.source.ne(routes.cohort))
                          | ((routes.bank == 'outcome') & routes.source.ne(routes.outcome))].copy()
    per_bank = (borrowed.groupby(['cohort', 'outcome', 'patient_group', 'bank']).w.sum()
                 .unstack('bank').reindex(columns=['disease', 'outcome']).fillna(0))
    per_bank['own_shared'] = (1 - per_bank.sum(axis=1)).clip(lower=0)
    patient_order = sorted(set(disease_groups.values()))
    mass = per_bank.groupby('patient_group').mean().reindex(patient_order)
    left = np.zeros(len(patient_order))
    for key, color, label in [('disease', '#3E8A87', 'Other diseases'),
                               ('outcome', '#8974AC', 'Other outcomes'),
                               ('own_shared', '#D9DDE4', 'Own / null slots')]:
        values = mass[key].to_numpy()
        ax.barh(range(len(patient_order)), values, left=left, height=.70,
                  color=color, label=label)
        left += values
    ax.set_yticks(range(len(patient_order)), patient_order, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel('Mean gate weight (all sources sum to 1)')
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.grid(False)
    _panel_title(ax, 'C', 'Where the gate weight goes')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.29), ncol=3,
                frameon=False, fontsize=7.5, handlelength=1.4, borderaxespad=0)
    despine(ax, 'none')

    # D: concentration within borrowed slots, based on exported mean gates.
    ax = fig.add_subplot(bottom[1])
    breadth = borrowed_source_breadth(routes)
    summary = _summary_rows(
        breadth.assign(series='all').groupby(
            ['cohort', 'target_group', 'series'], as_index=False).gain.mean(),
        ['target_group', 'series'])
    order = summary.sort_values('median', ascending=False).target_group.tolist()
    summary['target_group'] = summary.target_group.map(lambda g: SHORT_GROUP.get(g, g))
    _diamond_forest(ax, summary, row_key='target_group',
                    row_order=[SHORT_GROUP.get(g, g) for g in order], series_key='series',
                    series_style={'all': (MODEL_STYLE['R'][0], '')},
                    xlabel='Effective borrowed sources\n(from mean gates; median, IQR)',
                    show_counts=False, zero_line=False)
    ax.set_xlim(left=0)
    maximum_sources = borrowed.groupby(['cohort', 'outcome']).size().max()
    ax.axvline(maximum_sources, color=PALETTE['ink_muted'], lw=.8, ls='--')
    ax.set_xlim(0, maximum_sources * 1.06)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, 'D', 'Breadth of the mean gate profile')
    watermark(fig, mark)
    return fig


# ── Calibration, risk groups, subgroups, coverage and strategy figures ─────────

CAL_MODELS = {
    "OPERA": MODEL_STYLE["OPERA"],
    "J": (MODEL_STYLE["J"][0], "Joint"),
    "R": (MODEL_STYLE["R"][0], "Routed"),
    "S": (MODEL_STYLE["S"][0], "Specialist"),
    "X": (MODEL_STYLE["X"][0], "XGBoost"),
}
RISK_COLORS = ["#3E8A87", "#B18B4D", "#C16A58", "#8E2F3C"]


def _risk_colors(n):
    """Lowest to highest risk; three-group scores keep the extreme colours."""
    return [RISK_COLORS[int(round(i))] for i in np.linspace(0, len(RISK_COLORS) - 1, n)]
COVERAGE_STYLE = {
    "evaluated": ("#AEB8DA", "Joint–specialist comparison evaluable"),
    "no_training_events": ("#E7B9A5", "Too few training events for a specialist"),
    "single_class_test": ("#F2DDB0", "Only one evaluable class"),
    "insufficient_test_support": ("#D9D1E5", "Fewer than 5 cases or controls"),
    "specialist_unavailable": ("#B5451B", "Specialist unavailable (reason unverified)"),
    "training_support_unknown": ("#A7A7A7", "Training support unavailable"),
    "fit_failed": ("#80301C", "Fit failure confirmed by run log"),
    "not_applicable": ("#FFFFFF", "Not applicable"),
}


RELIABILITY_MIN_BIN = 20  # bins with fewer patients swing wildly; not drawn


def _full_slope_axis(ax, rows, label):
    """Show every slope summary and IQR, with readable compressed tails if needed."""
    values = rows[['q25', 'median', 'q75']].to_numpy(float).ravel()
    values = values[np.isfinite(values)]
    low, high = min(0., values.min(initial=0.)), max(2., values.max(initial=2.))
    if high > 6 or low < -3:
        from matplotlib.ticker import FixedLocator, FuncFormatter
        ax.set_xscale('symlog', linthresh=2, linscale=1.5)
        transform = ax.xaxis.get_transform()
        bounds = transform.transform(np.array([low, high]))
        margin = .04 * (bounds[1] - bounds[0])
        limits = transform.inverted().transform(bounds + np.array([-margin, margin]))
        ax.set_xlim(*limits)
        powers = range(1, max(1, int(np.ceil(np.log10(max(high, abs(low), 2.)))))+1)
        ticks = sorted(set([0., 1., 2., -1., -2.] +
                       [sign * 10.**power for sign in [-1, 1] for power in powers]))
        ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if limits[0] <= t <= limits[1]]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.minorticks_off()
        ax.set_xlabel(label + '\ncompressed tails; all IQRs shown')
    else:
        margin = .04 * max(2., high - low)
        ax.set_xlim(low - margin, high + margin)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(label)


def figure_calibration(summary, reliability, examples, *, horizon=ALL_HORIZONS, mark=None):
    """Calibration of the joint model against the alternatives.

    ``summary`` is ``calibration_summary`` output joined to endpoint groups;
    ``reliability`` the IPCW reliability bins; ``examples`` maps panel titles
    to (cohort, outcome) for the reliability curves.
    """
    fig = plt.figure(figsize=(8.8, 6.8))
    top = fig.add_gridspec(1, 2, wspace=0.18, left=0.19, right=0.98, top=0.88,
                           bottom=0.47)
    data = summary.loc[_horizon_mask(summary, horizon)
                       & (summary.status == "estimated")]
    # OPERA leads where analysed (and replaces its own routed duplicate).
    lead = "OPERA" if (data.model == "OPERA").any() else "J"
    if lead == "OPERA":
        data = data.loc[data.model != "R"]
    order = (data.loc[data.model == lead].groupby("endpoint_group").calibration_slope
             .median().sort_values().index.tolist())
    handles = None
    for column, (value, reference, letter, title, label) in enumerate((
            ("calibration_slope", 1.0, "A", "Calibration slope",
             "Slope (median, IQR; ideal 1)"),
            ("calibration_in_the_large", 0.0, "B", "Calibration in the large",
             "Observed − predicted risk (ideal 0)"))):
        ax = fig.add_subplot(top[column])
        rows = _summary_rows(data.rename(columns={value: "gain"}),
                             ["endpoint_group", "model"])
        rows["endpoint_group"] = rows.endpoint_group.map(lambda g: SHORT_GROUP.get(g, g))
        handles = _diamond_forest(ax, rows, row_key="endpoint_group",
                                  row_order=[SHORT_GROUP.get(g, g) for g in order],
                                  series_key="model", series_style=CAL_MODELS,
                                  xlabel=label, show_counts=False, zero_line=False)
        ax.axvline(reference, color=PALETTE["zero_line"], lw=0.7, zorder=1)
        if value == "calibration_slope":
            _full_slope_axis(ax, rows, label)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(4))
        if column:
            ax.tick_params(axis="y", labelleft=False)
        _panel_title(ax, letter, title)
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.18, 0.99))
    bottom = fig.add_gridspec(1, len(examples), wspace=0.62, left=0.08, right=0.98,
                              top=0.30, bottom=0.07)
    for index, (title, (cohort, outcome)) in enumerate(examples.items()):
        ax = fig.add_subplot(bottom[index])
        part = reliability.loc[(reliability.cohort == cohort)
                               & (reliability.outcome == outcome)
                               & (reliability.horizon_days == _single(horizon))]
        upper = max(0.05, float(np.nanmax(part[["predicted_weighted_mean",
                                                 "observed_fraction"]].to_numpy())) * 1.1)
        ax.plot([0, upper], [0, upper], color=PALETTE["diagonal"], lw=0.8, ls="--")
        for model in [m for m in ("X", "J", "OPERA") if m in set(part.model)]:
            rows = part.loc[(part.model == model) & (part.n >= RELIABILITY_MIN_BIN)
                            ].sort_values("probability_bin")
            color = CAL_MODELS[model][0]
            ax.plot(rows.predicted_weighted_mean, rows.observed_fraction, color=color,
                    lw=1.3, zorder=3)
            ax.scatter(rows.predicted_weighted_mean, rows.observed_fraction,
                       s=6 + 30 * rows.n / rows.n.max(), color=color, zorder=4,
                       linewidths=0)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=6.5)
        ax.set_title(textwrap.fill(title, 22) + ("\nat one year" if horizon == ALL_HORIZONS
                                                 else ""), fontsize=7.6, loc="left")
        if index == 0:
            ax.set_ylabel("Observed risk (IPCW)", fontsize=7.5)
            ax.text(-0.42, 1.18, "C", transform=ax.transAxes, fontsize=10.5)
        ax.set_xlabel("Predicted risk", fontsize=7.5)
        despine(ax, "both")
    watermark(fig, mark)
    return fig


def figure_benchmarks(cells, curve=None, summary=None, reliability=None, examples=None, *,
                      horizon=ALL_HORIZONS, horizon_panel=True, mark=None):
    """Against tabular models: discrimination and calibration together.

    A: AUC gain over per-task XGBoost by clinical domain (joint, specialist,
    routed; joint over pooled and registry-augmented XGBoost). B: calibration
    slope by domain (ideal 1). C: where the joint model gains over XGBoost, by
    training events. D-E: reliability of the joint model and XGBoost for two
    example outcomes. The full calibration figure is in the supplement.
    """
    fig = plt.figure(figsize=(7.2, 6.9))
    top = fig.add_gridspec(1, 2, wspace=0.08, left=0.175, right=0.985, top=0.87,
                           bottom=0.47)
    styles = {"OPERA-X": (MODEL_STYLE["OPERA"][0], "OPERA vs XGBoost"),
              "J-X": (MODEL_STYLE["J"][0], "Joint vs XGBoost"),
              "S-X": (MODEL_STYLE["S"][0], "Specialist vs XGBoost"),
              "R-X": (MODEL_STYLE["R"][0], "Routed vs XGBoost"),
              "J-XP": ("#B18B4D", "Joint vs pooled XGBoost"),
              "J-XR": ("#2F6B5F", "Joint vs XGBoost + registry")}
    data = cells.loc[_horizon_mask(cells, horizon) & (cells.metric == "auc")
                     & cells.contrast.isin(styles)]
    # OPERA (the configuration chosen on tuning data) leads where it was
    # analysed; it is usually the routed model, so R-X would repeat it.
    lead = "OPERA" if (data.contrast == "OPERA-X").any() else "J"
    if lead == "OPERA":
        styles.pop("R-X")
    styles = {k: v for k, v in styles.items() if k in set(data.contrast)}
    order = (data.loc[data.contrast == f"{lead}-X"].groupby("endpoint_group").gain.median()
             .sort_values(ascending=False).index.tolist())
    short = [SHORT_GROUP.get(g, g) for g in order]
    ax = fig.add_subplot(top[0])
    rows = _summary_rows(data, ["endpoint_group", "contrast"])
    rows["endpoint_group"] = rows.endpoint_group.map(lambda g: SHORT_GROUP.get(g, g))
    handles = _diamond_forest(ax, rows, row_key="endpoint_group", row_order=short,
                              series_key="contrast", series_style=styles,
                              xlabel="AUC gain (median, IQR)", show_counts=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "A", "Discrimination")
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.16, 0.995), handletextpad=0.2, columnspacing=0.9)
    ax = fig.add_subplot(top[1])
    if summary is not None:
        cal = summary.loc[_horizon_mask(summary, horizon)
                          & (summary.status == "estimated")]
        if lead == "OPERA":
            cal = cal.loc[cal.model != "R"]
        rows = _summary_rows(cal.rename(columns={"calibration_slope": "gain"}),
                             ["endpoint_group", "model"])
        rows["endpoint_group"] = rows.endpoint_group.map(lambda g: SHORT_GROUP.get(g, g))
        cal_handles = _diamond_forest(ax, rows, row_key="endpoint_group", row_order=short,
                                      series_key="model", series_style=CAL_MODELS,
                                      xlabel="Calibration slope (median, IQR; ideal 1)",
                                      show_counts=False, zero_line=False)
        ax.axvline(1.0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
        _full_slope_axis(ax, rows, 'Calibration slope (median, IQR; ideal 1)')
        fig.legend(handles=cal_handles, loc="upper left", ncol=2, frameon=False,
                   fontsize=6.4, bbox_to_anchor=(0.6, 0.995), handletextpad=0.2,
                   columnspacing=0.9)
    ax.tick_params(axis="y", labelleft=False)
    if summary is None:
        ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "B", "Calibration")

    shown = list((examples or {}).items())[:1]
    bottom = fig.add_gridspec(1, 1 + len(shown), width_ratios=[1.45] + [1] * len(shown),
                              wspace=0.45, left=0.09, right=0.985, top=0.34, bottom=0.08)
    ax = fig.add_subplot(bottom[0])
    lead_color = MODEL_STYLE[lead][0]
    part = estimable_cells(data.loc[data.contrast == f"{lead}-X"])
    ax.scatter(part.n_events_train.clip(lower=1), part.gain, s=6, alpha=0.3,
               color=lead_color, linewidths=0, rasterized=True)
    if curve is not None and (curve.contrast == f"{lead}-X").any():
        line = curve.loc[curve.contrast == f"{lead}-X"].sort_values("training_support")
        ax.fill_between(line.training_support, line.lower, line.upper,
                        color=lead_color, alpha=0.14, linewidth=0)
        ax.plot(line.training_support, line["mean"], color=lead_color, lw=1.6)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Target training events (log scale)")
    ax.set_ylabel(f"AUC gain, {'OPERA' if lead == 'OPERA' else 'joint'} vs XGBoost")
    _panel_title(ax, "C", "Gain by training events")
    if horizon_panel:
        ax.clear()
        for code, (color, _) in styles.items():
            part = cells.loc[cells.contrast.eq(code) & cells.metric.eq('auc')
                             & cells.horizon_days.gt(0)]
            if part.empty:
                continue
            trend = part.groupby('horizon_days').gain.median()
            ax.plot(trend.index, trend.values, marker='D', ms=3.5, lw=1.4, color=color)
        ax.axhline(0, color=PALETTE['zero_line'], lw=.7)
        ax.set_xscale('log')
        ax.set_xticks(list(HORIZON_COLORS), [str(h) for h in HORIZON_COLORS])
        ax.minorticks_off()
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Median AUC gain over comparator')
        _panel_title(ax, 'C', 'Differences across horizons')
    despine(ax, "y")
    for index, (title, (cohort, outcome)) in enumerate(shown):
        ax = fig.add_subplot(bottom[index + 1])
        rel = reliability.loc[(reliability.cohort == cohort)
                              & (reliability.outcome == outcome)
                              & (reliability.horizon_days == _single(horizon))]
        upper = max(0.05, float(np.nanmax(rel[["predicted_weighted_mean",
                                               "observed_fraction"]].to_numpy())) * 1.1)
        ax.plot([0, upper], [0, upper], color=PALETTE["diagonal"], lw=0.8, ls="--")
        for model in [m for m in ("X", "J", "OPERA") if m in set(rel.model)]:
            rows = rel.loc[(rel.model == model) & (rel.n >= RELIABILITY_MIN_BIN)].sort_values("probability_bin")
            color = CAL_MODELS[model][0]
            ax.plot(rows.predicted_weighted_mean, rows.observed_fraction, color=color,
                    lw=1.3, zorder=3)
            ax.scatter(rows.predicted_weighted_mean, rows.observed_fraction,
                       s=6 + 30 * rows.n / rows.n.max(), color=color, zorder=4,
                       linewidths=0)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Predicted risk", fontsize=7.2)
        if index == 0:
            ax.set_ylabel("Observed risk (IPCW)", fontsize=7.2)
        _panel_title(ax, "DE"[index], textwrap.fill(title, 24))
        if horizon == ALL_HORIZONS:
            ax.text(0.04, 0.96, "at one year", transform=ax.transAxes, fontsize=6.4,
                    color=PALETTE["ink_muted"], va="top")
        despine(ax, "both")
    watermark(fig, mark)
    return fig


def cumulative_incidence(time, event, horizon_grid, cause=1):
    """Aalen–Johansen cumulative incidence of ``cause`` (equals 1 − KM without
    competing events), evaluated on ``horizon_grid``."""
    order = np.argsort(time, kind="mergesort")
    time, event = np.asarray(time, float)[order], np.asarray(event, int)[order]
    unique = np.unique(time)
    at_risk = len(time) - np.searchsorted(time, unique, side="left")
    d_cause = np.array([np.sum((time == t) & (event == cause)) for t in unique])
    d_any = np.array([np.sum((time == t) & (event > 0)) for t in unique])
    survival = np.cumprod(1 - d_any / at_risk)
    before = np.r_[1.0, survival[:-1]]
    cif = np.cumsum(before * d_cause / at_risk)
    index = np.searchsorted(unique, horizon_grid, side="right") - 1
    return np.where(index >= 0, cif[np.clip(index, 0, None)], 0.0)


def _estimate_with_ci(part, horizon_days, event_free, *, draws=200, seed=0):
    """Point estimate and 95% bootstrap interval at the horizon (CIF, or 1 - CIF)."""
    time = part.time_days.to_numpy(float)
    event = part.event.to_numpy(int)
    point, low, high = _cif_band(time, event, [horizon_days], np.random.default_rng(seed),
                                 draws=draws)
    if event_free:
        return 1 - point[0], 1 - high[0], 1 - low[0]
    return point[0], low[0], high[0]


def _mean_with_ci(values, *, draws=1000, seed=0):
    """Mean and 95% bootstrap interval of the mean."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _survival_label(outcome_label):
    """Axis label for a death-inclusive endpoint shown as survival."""
    label = str(outcome_label)
    if "survival" in label.lower():
        return label[0].upper() + label[1:]
    return f"{label}-free survival".capitalize()


def _count_text(count, minimum=5):
    """Patient or event count for display; counts below ``minimum`` are not shown."""
    count = int(count)
    return f"<{minimum}" if 0 < count < minimum else f"{count:,}"


def _cell_borders(ax, shape, colour="white", width=1.6):
    """White separators between the cells of a grid drawn with imshow."""
    rows, columns = shape
    ax.set_xticks(np.arange(1, columns) - 0.5, minor=True)  # between cells only
    ax.set_yticks(np.arange(1, rows) - 0.5, minor=True)
    ax.grid(which="minor", color=colour, linewidth=width)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.set_axisbelow(False)  # separators drawn over the cells, edges included
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)


def _count_grid(ax, counts, *, xlabels, xlabel=None, show_y=True, height=0.3,
                gap=0.1, fontsize=5.6, label_size=5.4):
    """A compact grid of patient counts below a survival grid, HemaSphere style.

    Columns line up with the grid above and the rows are much shorter, so the
    survival pattern and the support behind it read at a glance. The counts
    are shaded lightly in one tone, so they never compete with the survival
    colours. Counts below five are shown as <5. The x labels move to this grid.
    """
    from matplotlib.colors import LinearSegmentedColormap as _Map

    counts = np.asarray(counts, float)
    grid = ax.inset_axes([0, -height - gap, 1, height])
    tone = _Map.from_list("counts", ["#F6F7F9", "#C9D0DD"])
    grid.imshow(counts, cmap=tone, vmin=0, vmax=max(1.0, counts.max()), aspect="auto",
                interpolation="nearest")
    for (i, j), value in np.ndenumerate(counts):
        grid.text(j, i, _count_text(value) if value else "0", ha="center", va="center",
                  fontsize=fontsize, color=PALETTE["ink_secondary"])
    grid.set_xticks(range(counts.shape[1]), _risk_tick_labels(xlabels), fontsize=max(6.6, label_size))
    labels = [t.get_text() for t in ax.get_yticklabels()]
    grid.set_yticks(range(counts.shape[0]), labels if show_y else [""] * counts.shape[0])
    grid.tick_params(axis="y", labelsize=label_size, length=0, pad=2,
                     labelcolor=PALETTE["ink_secondary"])
    grid.tick_params(axis="x", length=0, pad=3)
    if xlabel:
        grid.set_xlabel(xlabel)
    grid.text(0, 1.04, "Number of patients", transform=grid.transAxes, ha="left",
              va="bottom", fontsize=max(5.8, label_size), color=PALETTE["ink_secondary"])
    _cell_borders(grid, counts.shape, width=1.2)
    for spine in grid.spines.values():
        spine.set_visible(False)
    _cell_borders(ax, counts.shape)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.set_xlabel("")
    return grid


def _estimate_text(value, low, high, count=None):
    """'78%' over '(71–84)', optionally over the patient count."""
    if not np.isfinite(value):
        return "–" if count is None else f"–\nn={_count_text(count)}"
    text = f"{value:.0%}\n({100 * low:.0f}–{100 * high:.0f})"
    return text if count is None else f"{text}\nn={_count_text(count)}"


def _cif_band(time, event, grid, rng, draws=150):
    point = cumulative_incidence(time, event, grid)
    samples = []
    for _ in range(draws):
        pick = rng.integers(0, len(time), len(time))
        samples.append(cumulative_incidence(time[pick], event[pick], grid))
    low, high = np.quantile(np.vstack(samples), [0.025, 0.975], axis=0)
    return point, low, high


def figure_risk_groups(patients, *, outcome_label, horizon_days=730,
                       thresholds=(0.1, 0.4), ipi_label="IPI",
                       ipi_order=("Low", "Intermediate", "High"),
                       mark=None, seed=0, event_free=False, grouping_note=None):
    """Cumulative incidence by model risk group versus the clinical score.

    ``patients`` has ``time_days``, ``event`` (0 censored, 1 outcome, 2
    competing death), ``model_risk`` (predicted risk at ``horizon_days``) and
    ``ipi_group``. Only patients with a clinical score are shown, so both
    groupings describe the same people. Thresholds are prespecified.
    """
    if event_free and (patients.event > 1).any():
        raise ValueError("Event-free KM requires death included in the composite, not a competing cause")
    rng = np.random.default_rng(seed)
    data = patients.loc[patients.ipi_group.notna()].copy()
    names = list(ipi_order)
    if len(names) != len(thresholds) + 1 or len(names) > len(RISK_COLORS):
        raise ValueError("Register 2–4 groups with matching cutoffs")
    data['model_group'] = np.asarray(names)[np.digitize(data.model_risk, thresholds)]
    grid = np.unique(np.r_[np.linspace(0, horizon_days * 1.25, 120), horizon_days])
    fig = plt.figure(figsize=(9.2, 7.8))
    top = fig.add_gridspec(1, 2, wspace=0.12, left=0.08, right=0.98, top=0.94,
                           bottom=0.63)
    estimates = []
    for column, (key, order, title, letter) in enumerate((
            ("model_group", names, "Model risk group", "A"),
            ("ipi_group", list(ipi_order), ipi_label, "B"))):
        ax = fig.add_subplot(top[column])
        for group_index, (color, group) in enumerate(zip(_risk_colors(len(order)), order)):
            part = data.loc[data[key] == group]
            if len(part) < 10:
                ax.plot([], [], color=color, ls=['-', '--', '-.', ':'][group_index],
                        label=f'{group} (n={_count_text(len(part))}; insufficient)')
                continue
            point, low, high = _cif_band(part.time_days.to_numpy(float),
                                         part.event.to_numpy(int), grid, rng)
            if event_free:
                point, low, high = 1 - point, 1 - high, 1 - low
            ax.fill_between(grid / 365.25, low, high, color=color, alpha=0.14,
                            linewidth=0, step="post")
            events = int(((part.event == 1) & (part.time_days <= horizon_days)).sum())
            ax.step(grid / 365.25, point, where="post", color=color, lw=1.6,
                    ls=['-', '--', '-.', ':'][group_index],
                    label=f"{group} (n={_count_text(len(part))})")
            at = np.searchsorted(grid, horizon_days)
            estimates.append({"system": title, "group": group, "n": len(part), "events": events,
                              "estimate": point[at], "lower": low[at], "upper": high[at],
                              "color": color})
        ax.axvline(horizon_days / 365.25, color=PALETTE["ink_muted"], lw=0.7, ls="--")
        ax.set_xlabel("Years since first-line treatment")
        if column == 0:
            ax.set_ylabel(_survival_label(outcome_label) if event_free
                          else f"Cumulative incidence of\n{outcome_label.lower()}")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        _panel_title(ax, letter, title)
        _legend(ax, ax.get_legend_handles_labels()[0],
                loc="lower left" if event_free else "upper left",
                title='Model-derived groups' if column == 0 else f'{ipi_label} groups',
                fontsize=8, title_fontsize=8, handlelength=2.6, labelspacing=.4)
        risk_times = np.array([0, horizon_days / 2, horizon_days])
        risk_counts = [[_count_text((data.loc[data[key] == group, "time_days"] >= t).sum())
                        for t in risk_times] for group in order]
        table = ax.table(cellText=risk_counts, rowLabels=order,
                         colLabels=["Start", f"{horizon_days / 2 / 365.25:.1f} y",
                                    f"{horizon_days / 365.25:.1f} y"],
                         bbox=[0.18, -0.52, 0.82, 0.27], cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        colors = dict(zip(order, _risk_colors(len(order))))
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            if col == -1 and row > 0:  # group names in their curve colours
                cell.get_text().set_color(colors[order[row - 1]])
                cell.get_text().set_fontweight("bold")
            if row == 0:
                cell.get_text().set_color(PALETTE["ink_secondary"])
        ax.text(0.0, -0.25, "Number at risk", transform=ax.transAxes, fontsize=7,
                color=PALETTE["ink_secondary"], va="bottom")
        despine(ax, "y")
    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.6, left=0.22,
                              right=0.97, top=0.37, bottom=0.17)
    ax = fig.add_subplot(bottom[0])
    estimates = pd.DataFrame(estimates)
    labels = []
    for y, row in enumerate(estimates.itertuples()):
        ax.plot([row.lower, row.upper], [y, y], color=row.color, lw=1.4, alpha=0.55)
        ax.scatter(row.estimate, y, marker="D", s=26, color=row.color, zorder=3)
        labels.append(f"{row.group} ({_count_text(row.events)}/{_count_text(row.n)})")
    ax.set_yticks(range(len(labels)), labels, fontsize=7.4)
    split = int((estimates.system == "Model risk group").sum())
    ax.axhline(split - 0.5, color=PALETTE["grid"], lw=1)
    for label, centre in (("Model", (split - 1) / 2),
                          (ipi_label, split + (len(estimates) - split - 1) / 2)):
        ax.text(-0.78, centre, label, transform=ax.get_yaxis_transform(), ha="center",
                va="center", fontsize=7.6, color=PALETTE["ink_secondary"], rotation=90,
                fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-.04, 1.04)
    ax.set_xlabel(f"{horizon_days // 365}-year "
                  + (_survival_label(outcome_label).lower() if event_free
                     else "cumulative incidence")
                  + " (95% CI)")
    _panel_title(ax, "C", "Survival at the horizon" if event_free else "Risk at the horizon")
    despine(ax, "x")

    ax = fig.add_subplot(bottom[1])
    table = []
    for ipi in ipi_order:
        row = []
        for group in names:
            part = data.loc[(data.ipi_group == ipi) & (data.model_group == group)]
            estimate = (_estimate_with_ci(part, horizon_days, event_free)
                        if len(part) >= 10 else (np.nan, np.nan, np.nan))
            row.append((*estimate, len(part)))
        table.append(row)
    values = np.array([[cell[0] for cell in row] for row in table], float)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=LinearSegmentedColormap.from_list(
        "risk", (["#8E2F3C", "#C16A58", "#E7B9A5", "#F7F7F5"] if event_free
                 else ["#F7F7F5", "#E7B9A5", "#C16A58", "#8E2F3C"])), vmin=0, vmax=1,
        aspect="auto")
    for i, row in enumerate(table):
        for j, (value, low, high, count) in enumerate(row):
            if count:
                text = _estimate_text(value, low, high)
                ax.text(j, i, text, ha="center", va="center", fontsize=7.2, linespacing=1.1,
                        color="white" if np.isfinite(value) and
                        (value < 0.45 if event_free else value > 0.55)
                        else PALETTE["ink"])
    ax.set_xticks(range(len(names)), names,
                  fontsize=7.4)
    ax.set_yticks(range(len(ipi_order)), list(ipi_order), fontsize=7.4)
    ax.set_xlabel("Model risk group")
    ax.set_ylabel(ipi_label)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_title(ax, "D", "Reclassification")
    ax.set_ylabel(f"{ipi_label} category")
    # The diagonal is agreement; left of it the model places patients lower than
    # the index, right of it higher. Outlined cells and the side labels say so.
    from matplotlib.patches import Rectangle
    for k in range(min(len(names), len(ipi_order))):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, lw=1.3,
                               edgecolor=PALETTE["ink"], zorder=4))
    _count_grid(ax, [[cell[3] for cell in row] for row in table], xlabels=names,
                xlabel="Model risk group", fontsize=7, label_size=6.8)
    del image
    if grouping_note:
        fig.text(.5, .075, grouping_note, ha='center', fontsize=7, color=PALETTE['ink_secondary'])
    watermark(fig, mark)
    return fig


def _known_status(data, horizon_days):
    """Patients whose status at the horizon is known: event before it, or
    followed beyond it. The binary label is an event before the horizon."""
    known = (data.time_days >= horizon_days) | (data.event > 0)
    label = (data.event == 1) & (data.time_days <= horizon_days)
    return data.loc[known], label[known].astype(int)


def _risk_tick_labels(names):
    """Keep native category names legible in narrow four-column heatmaps."""
    return [str(name).replace('-intermediate', '-\nintermediate') for name in names]


def _risk_cutoff_labels(thresholds):
    """Preserve visibly distinct small cutoffs rather than printing 0.00."""
    values = np.asarray(thresholds, float)
    for digits in range(2, 7):
        rounded = np.round(values, digits)
        if np.array_equal(rounded > 0, values > 0) and len(np.unique(rounded)) == len(np.unique(values)):
            return [f'{v:.{digits}f}' for v in values]
    return [f'{v:.6g}' for v in values]


def _score_label(label):
    import re
    return re.sub(r'\s*·\s*all (?:eligible )?(?:lymphoma )?subtypes\b', '',
                  str(label), flags=re.IGNORECASE)


def _clinical_precision_recall_panel(ax, known, label, *, ipi_label, ipi_high,
                                     model_threshold):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    if len(known) == 0 or label.nunique() < 2:
        ax.text(.5, .5, 'Precision–recall unavailable\nRequires events and non-events',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return ['Operating-point comparison unavailable on this evaluation sample.']
    precision, recall, _ = precision_recall_curve(label, known.model_risk)
    model_ap = average_precision_score(label, known.model_risk)
    score_ap = average_precision_score(label, known.ipi_score)
    ax.plot(recall, precision, color=MODEL_STYLE['J'][0], lw=1.5,
            label=f'Model (AP {model_ap:.2f})')
    points = []
    for cut in sorted(known.ipi_score.unique())[1:]:
        flagged = known.ipi_score >= cut
        if flagged.any():
            points.append((float(flagged[label == 1].mean()), float(label[flagged].mean())))
    if points:
        x, y = np.array(points).T
        ax.plot(x, y, color=PALETTE['ink_muted'], ls='none', marker='o', ms=3.5,
                label=f'{ipi_label} thresholds (AP {score_ap:.2f})')
    high = known.ipi_score >= ipi_high
    specificity = float((~high[label == 0]).mean())
    cut = float(model_threshold) if model_threshold is not None else float(
        np.quantile(known.loc[label == 0, 'model_risk'], specificity))
    if not np.isfinite(cut):
        return ['Independent development classifier unavailable; continuous comparison shown.']
    # The exported binary metrics and pd.cut risk groups include threshold ties.
    flagged = known.model_risk >= cut
    sensitivity = float(flagged[label == 1].mean())
    ppv = float(label[flagged].mean()) if flagged.any() else np.nan
    achieved_specificity = float((~flagged[label == 0]).mean())
    if np.isfinite(ppv):
        ax.scatter(sensitivity, ppv, s=40, color=MODEL_STYLE['J'][0], zorder=4, marker='D')
    if not high.any():
        return [f'No patients meet the index high-risk cutoff (≥ {ipi_high:g}).',
                f'Frozen model cutoff: recall {sensitivity:.2f}, specificity {achieved_specificity:.2f}.']
    reference = (float(label[high].mean()), float(high[label == 1].mean()))
    ax.scatter(reference[1], reference[0], s=36, color=PALETTE['ink_muted'], zorder=4, marker='s')
    at_recall = float(precision[recall >= reference[1]].max())
    at_precision = float(recall[precision >= reference[0]].max())
    projection = dict(color=MODEL_STYLE['J'][0], lw=1, ls=(0, (2, 1.5)), zorder=3)
    ax.plot([reference[1], reference[1]], [reference[0], at_recall], **projection)
    ax.plot([reference[1], at_precision], [reference[0], reference[0]], **projection)
    ax.scatter([reference[1], at_precision], [at_recall, reference[0]], s=10,
               color=MODEL_STYLE['J'][0], zorder=4)
    ppv_text = f'{ppv:.2f}' if np.isfinite(ppv) else 'unavailable'
    return [f'At index recall: precision {reference[0]:.2f} → {at_recall:.2f}',
            f'At index precision: recall {reference[1]:.2f} → {at_precision:.2f}',
            f'Frozen cutoff: recall {sensitivity:.2f}, precision {ppv_text}, specificity {achieved_specificity:.2f}']


def figure_clinical_score(patients, *, outcome_label, horizon_days=730,
                          thresholds=(0.1, 0.4), ipi_label="IPI",
                          ipi_order=("Low", "Intermediate", "High"),
                          ipi_high=6, age_bins=(0, 60, 75, np.inf), mark=None,
                          event_free=False, model_threshold=None, grouping_note=None):
    """The model against the clinical score on the patients who have one.

    Supply ``model_threshold`` locked on training data for held-out evaluation.
    Without it, the matched-specificity point is descriptive on this sample.
    ``event_free`` shows KM complements for a death-inclusive composite;
    it rejects competing-cause records.

    ``patients`` as for :func:`figure_risk_groups`, plus ``ipi_score``
    (integer) and ``age``. A: precision–recall of the model and of every
    integer score threshold, with the model at the score's high-risk
    specificity. B: model risk within each score group. C–D: cumulative
    incidence at the horizon by age and risk group.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    if event_free and (patients.event > 1).any():
        raise ValueError("Event-free KM requires a death-inclusive composite")
    data = patients.loc[patients.ipi_group.notna() & patients.ipi_score.notna()].copy()
    names = list(ipi_order)
    data['model_group'] = np.asarray(names)[np.digitize(data.model_risk, thresholds)]
    known, label = _known_status(data, horizon_days)
    fig = plt.figure(figsize=(9.2, 8.1))
    top = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.46, left=0.09,
                           right=0.97, top=0.87, bottom=0.61)
    ax = fig.add_subplot(top[0])
    summary = _clinical_precision_recall_panel(
        ax, known, label, ipi_label=ipi_label, ipi_high=ipi_high,
        model_threshold=model_threshold)
    for y, text in zip((.475, .445, .415), summary):
        fig.text(.09, y, textwrap.fill(text, 75), ha="left", va="top", fontsize=7,
                 color=PALETTE["ink_secondary"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper left", bbox_to_anchor=(.09, .55), bbox_transform=fig.transFigure,
              borderaxespad=0, frameon=False, fontsize=7.5, ncol=1, handlelength=2.5,
              title='Continuous risk and index thresholds', title_fontsize=7.5)
    horizon_label = "1 year" if horizon_days == 365 else (f"{horizon_days / 365:g} years" if horizon_days % 365 == 0 else f"{horizon_days:g} days")
    _panel_title(ax, "A", f"Events by {horizon_label}")
    ax.set_title(f"A  Events by {horizon_label}", loc="left", fontsize=TITLE_SIZE, pad=16)
    despine(ax, "both")

    ax = fig.add_subplot(top[1])
    rng = np.random.default_rng(0)
    for y, group in enumerate(names):
        lower = [-np.inf, *thresholds][y]
        upper = [*thresholds, np.inf][y]
        ax.axvspan(max(lower, 0), min(upper, 1), color=_risk_colors(len(names))[y], alpha=0.18,
                   linewidth=0)
    for boundary in thresholds:
        ax.axvline(boundary, color=PALETTE["ink_secondary"], lw=0.8,
                   ls="--", alpha=0.65, zorder=1)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=color, alpha=.35, label=name)
                        for name, color in zip(names, _risk_colors(len(names)))],
               title="Model-derived risk groups", loc="upper left",
               bbox_to_anchor=(.60, .475), ncol=2, frameon=False,
               fontsize=8, title_fontsize=8, handlelength=1.3,
               columnspacing=.7, borderaxespad=0)
    cutoff_labels = _risk_cutoff_labels(thresholds)
    ax.text(0, -0.30, (grouping_note + '\n' if grouping_note else "Model cutoffs (fixed on training):\n") + " / ".join(
                [f"< {cutoff_labels[0]}",
                 *[f"{lo}–< {hi}" for lo, hi in zip(cutoff_labels[:-1], cutoff_labels[1:])],
                 f"≥ {cutoff_labels[-1]}"]),
            transform=ax.transAxes, ha="left", fontsize=7,
            color=PALETTE["ink_secondary"])
    for y, group in enumerate(ipi_order):
        part = known.loc[known.ipi_group == group]
        events = label.loc[part.index] == 1
        jitter = rng.uniform(-0.28, 0.28, len(part))
        for mask, color, size in ((~events, PALETTE["ink_muted"], 3),
                                  (events, "#B5451B", 4)):
            ax.scatter(part.model_risk[mask], y + jitter[mask.to_numpy()], s=size,
                       color=color, alpha=0.4, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(ipi_order)), list(ipi_order), fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Model risk at {horizon_label}")
    ax.grid(False)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.5, color="#B5451B",
                              label=f"Event by {horizon_label}"),
                       Line2D([], [], ls="none", marker="o", ms=3.5,
                              color=PALETTE["ink_muted"], label=f"Event-free at {horizon_label}")],
              loc="upper right", frameon=True, facecolor="white", framealpha=0.95,
              edgecolor="none", fontsize=8, ncol=1, handletextpad=0.3)
    _panel_title(ax, "B", f"Model risk by {ipi_label} group")
    ax.set_title(f"B  Model risk by {ipi_label} group", loc="left", fontsize=TITLE_SIZE, pad=20)
    despine(ax, "none")

    bottom = fig.add_gridspec(1, 2, wspace=0.12, left=0.13, right=0.97, top=0.35,
                              bottom=0.16)
    ages = pd.cut(data.age, list(age_bins), right=False)
    age_labels = [f"< {int(b.right)}" if not np.isfinite(b.left) or b.left <= 0
                  else (f"≥ {int(b.left)}" if not np.isfinite(b.right)
                        else f"{int(b.left)}–{int(b.right) - 1}")
                  for b in ages.cat.categories]
    cmap = LinearSegmentedColormap.from_list(
        "risk", (["#8E2F3C", "#C16A58", "#E7B9A5", "#F7F7F5"] if event_free
                 else ["#F7F7F5", "#E7B9A5", "#C16A58", "#8E2F3C"]))
    for column, (key, letter, title) in enumerate((
            ("model_group", "C", "Model risk group by age"),
            ("ipi_group", "D", f"{ipi_label} by age"))):
        ax = fig.add_subplot(bottom[column])
        values = np.full((len(age_labels), len(names)), np.nan)
        lows, highs = values.copy(), values.copy()
        counts = np.zeros_like(values, dtype=int)
        for i, interval in enumerate(ages.cat.categories):
            for j, group in enumerate(names):
                part = data.loc[(ages == interval) & (data[key] == group)]
                counts[i, j] = len(part)
                if len(part) >= 10:
                    values[i, j], lows[i, j], highs[i, j] = _estimate_with_ci(
                        part, horizon_days, event_free)
        ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(age_labels)):
            for j in range(len(names)):
                if counts[i, j]:
                    value = values[i, j]
                    text = _estimate_text(value, lows[i, j], highs[i, j])
                    ax.text(j, i, text, ha="center", va="center", fontsize=5.8,
                            linespacing=1.1,
                            color="white" if np.isfinite(value) and
                            (value < 0.45 if event_free else value > 0.55)
                            else PALETTE["ink"])
        ax.set_xticks(range(len(names)), _risk_tick_labels(names),
                      fontsize=6.8)
        ax.set_yticks(range(len(age_labels)), age_labels, fontsize=6.8)
        if column:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("Age (years)")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _panel_title(ax, letter, title)
        _count_grid(ax, counts, xlabels=names, show_y=not column)
    metric_label = (_survival_label(outcome_label) if event_free
                    else f"Cumulative incidence of {outcome_label.lower()}")
    fig.text(0.55, 0.015, f"{metric_label} by "
             f"{horizon_label} (95% CI), with patients per cell below", ha="center",
             fontsize=7, color=PALETTE["ink_secondary"])
    watermark(fig, mark)
    return fig


def figure_clinical_operating_points(low_groups, detection, *, estimand='cif_ipcw', mark=None):
    """Clinical counts at separate frozen Low and high-specificity thresholds."""
    detection = detection.loc[detection.estimand.eq(estimand)].copy()
    order = list(dict.fromkeys(detection.score_id))
    labels = detection.set_index('score_id').label.to_dict()
    height = max(4., .32 * len(order) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, height), sharey=True)
    fig.subplots_adjust(left=.20, right=.97, top=.82, bottom=.30, wspace=.5)
    for ax, letter, title in zip(axes, 'AB', ['Additional low-risk patients', 'Additional events identified']):
        ax.axvline(0, color=PALETTE['zero_line'], lw=.8)
        ax.set_yticks(range(len(order)), [labels[s] for s in order], fontsize=8)
        ax.set_ylim(len(order)-.5, -.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        _panel_title(ax, letter, title)
        despine(ax, 'x')
    for y, key in enumerate(order):
        if len(low_groups) and 'additional_low_per_1000' in low_groups:
            low = low_groups.loc[low_groups.score_id.eq(key)]
            if len(low) == 1 and np.isfinite(low.additional_low_per_1000.iloc[0]):
                row = low.iloc[0]
                axes[0].plot([row.additional_low_per_1000_lower, row.additional_low_per_1000_upper],
                             [y, y], color=MODEL_STYLE['OPERA'][0], lw=1.4, alpha=.55)
                axes[0].scatter(row.additional_low_per_1000, y, marker='D', s=25,
                                color=MODEL_STYLE['OPERA'][0])
            else:
                axes[0].text(.98, y, 'Not estimable', transform=axes[0].get_yaxis_transform(),
                             ha='right', va='center', fontsize=7, color=PALETTE['ink_muted'])
        else:
            axes[0].text(.98, y, 'Not estimable', transform=axes[0].get_yaxis_transform(),
                         ha='right', va='center', fontsize=7, color=PALETTE['ink_muted'])
        row = detection.loc[detection.score_id.eq(key)].iloc[0]
        axes[1].plot([row.lower, row.upper], [y, y], color=MODEL_STYLE['OPERA'][0], lw=1.4, alpha=.55)
        axes[1].scatter(row.additional_events_per_1000, y, marker='D', s=25,
                        color=MODEL_STYLE['OPERA'][0])
    axes[0].set_xlabel('Additional patients per 1,000 (95% CI)\nDevelopment KM-matched Low cutoff')
    axes[1].set_xlabel('Additional events per 1,000 (95% CI)\nDevelopment specificity-matched cutoff')
    axes[1].tick_params(labelleft=False)
    fig.text(.20, .04, 'Cutoffs fixed before evaluation. Achieved survival and specificity are reported separately.\n'
             + ('Event counts use the full risk set with IPCW.' if estimand == 'cif_ipcw'
                else 'Event counts use patients with known horizon status.'),
             fontsize=7, color=PALETTE['ink_secondary'])
    watermark(fig, mark)
    return fig


def figure_score_overview(scores, *, mark=None, seed=0):
    """Model risk groups against every recorded prognostic index.

    ``scores`` is a list of dicts, one per index and population, each with
    ``title`` (e.g. "CLL-IPI · CLL"), ``endpoint`` (e.g. "Overall survival"),
    ``patients`` (as for :func:`figure_risk_groups`), ``thresholds``,
    ``ipi_order``, ``horizon_days`` and ``event_free``. Each row shows survival
    by model group and by index category on the same patients, and both
    systems' estimates at the horizon.
    """
    rng = np.random.default_rng(seed)
    height = sum(2.1 if len(s['ipi_order']) > 3 else 1.8 for s in scores) + .95
    fig = plt.figure(figsize=(8.8, height))
    # Column 3 is an empty gutter that holds the estimate labels of column 4.
    grid = fig.add_gridspec(len(scores), 4, width_ratios=[1, 1, 0.7, 0.9],
                            wspace=0.16, hspace=0.5, left=0.09, right=0.985,
                            top=1 - 0.72 / fig.get_figheight(),
                            bottom=0.48 / fig.get_figheight())
    for row, spec in enumerate(scores):
        data = spec["patients"].loc[spec["patients"].ipi_group.notna()].copy()
        order = list(spec["ipi_order"])
        names = order
        data['model_group'] = np.asarray(names)[np.digitize(data.model_risk, spec['thresholds'])]
        horizon = spec["horizon_days"]
        grid_days = np.linspace(0, horizon * 1.25, 100)
        estimates = []
        for column, (key, groups, label) in enumerate((
                ("model_group", names, "Model"), ("ipi_group", order, "Index"))):
            ax = fig.add_subplot(grid[row, column])
            for color, group in zip(_risk_colors(len(groups)), groups):
                part = data.loc[data[key] == group]
                if len(part) < 10:
                    continue
                point, low, high = _cif_band(part.time_days.to_numpy(float),
                                             part.event.to_numpy(int), grid_days, rng,
                                             draws=80)
                if spec.get("event_free", True):
                    point, low, high = 1 - point, 1 - high, 1 - low
                ax.fill_between(grid_days / 365.25, low, high, color=color, alpha=0.13,
                                linewidth=0, step="post")
                ax.step(grid_days / 365.25, point, where="post", color=color, lw=1.2)
                at = np.searchsorted(grid_days, horizon)
                estimates.append((label, group, len(part), point[at], low[at], high[at],
                                  color))
            ax.axvline(horizon / 365.25, color=PALETTE["ink_muted"], lw=0.6, ls="--")
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=6.5)
            if column:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_ylabel(textwrap.fill(spec["endpoint"], 18), fontsize=7)
            if row == len(scores) - 1:
                ax.set_xlabel("Years since treatment start", fontsize=7)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row == 0:
                fig.text(ax.get_position().x0, 1 - 0.22 / fig.get_figheight(),
                         "A  Model risk groups" if column == 0 else "B  Native index groups",
                         fontsize=10.5, color=PALETTE["ink"], va="top")
            despine(ax, "y")
        ax = fig.add_subplot(grid[row, 3])
        for y, (label, group, n, value, low, high, color) in enumerate(estimates):
            ax.plot([low, high], [y, y], color=color, lw=1.2, alpha=0.55)
            ax.scatter(value, y, marker="D" if label == "Model" else "o", s=18,
                       color=color, zorder=3)
        ax.set_yticks(range(len(estimates)),
                      [f"{label[0]}: {group} ({_count_text(n)})" for label, group, n, *_ in estimates],
                      fontsize=6.4)
        ax.invert_yaxis()
        ax.set_xlim(-.04, 1.04)
        ax.set_xticks([0, .5, 1])
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=6.5)
        if row == 0:
            fig.text(ax.get_position().x0, 1 - 0.22 / fig.get_figheight(),
                     "C  At the horizon (95% CI)", fontsize=10.5,
                     color=PALETTE["ink"], va="top")
        if row != len(scores) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        despine(ax, "x")
        fig.text(0.005, ax.get_position().y1 + 0.006, _score_label(spec["title"]), fontsize=8.5,
                 fontweight="bold", color=PALETTE["ink"], va="bottom")
    watermark(fig, mark)
    return fig


# Schematic positions of the five Danish regions (x, y), used when no boundary
# file is supplied: Nordjylland top, Midtjylland and Syddanmark down Jutland,
# Hovedstaden and Sjælland on Zealand.
REGION_TILES = {"Nordjylland": (1.0, 3.0), "Midtjylland": (0.6, 2.0),
                "Syddanmark": (0.8, 1.0), "Hovedstaden": (2.6, 1.9),
                "Sjælland": (2.3, 0.95)}


REGION_CODES = {"1081": "Nordjylland", "1082": "Midtjylland", "1083": "Syddanmark",
                "1084": "Hovedstaden", "1085": "Sjælland"}


def _region_name(name):
    text = str(name).replace("Region ", "").strip()
    text = REGION_CODES.get(text.removesuffix(".0"), text)
    for region in REGION_TILES:
        if text.lower().startswith(region.lower()[:5]):
            return region
    return text


def _draw_denmark(ax, values, geometry, norm, cmap, *, fmt="{:+.3f}", intervals=None,
                  reference=None):
    """Regions coloured by ``values`` (region -> number); Bornholm as an inset.

    ``intervals`` (region -> (lower, upper)) prints each 95% CI under the value;
    with a ``reference`` (e.g. 0 for gains), regions whose interval includes it
    are hatched, so the colour alone never overstates a difference."""
    import matplotlib.patheffects as effects
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

    halo = [effects.withStroke(linewidth=2.2, foreground="white")]
    _map_axis(ax)
    if not geometry:
        ax.set_aspect("equal")
        for name, (x, y) in REGION_TILES.items():
            value = values.get(name, np.nan)
            color = cmap(norm(value)) if np.isfinite(value) else PALETTE["missing"]
            ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.4), 0.84, 0.8,
                                        boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor=color, edgecolor="white", lw=1.2,
                                        hatch="//////" if _includes(intervals, name, reference)
                                        else None))
            text = f"{name}\n{fmt.format(value)}" if np.isfinite(value) else name
            ax.text(x, y, text, ha="center", va="center", fontsize=6,
                    color=PALETTE["ink"], path_effects=halo)
        ax.set_xlim(-0.1, 3.3)
        ax.set_ylim(0.3, 3.7)
        return
    ax.set_aspect(1 / np.cos(np.radians(56)))
    shift = np.array([-2.45, 1.5])  # Bornholm into the north-east corner
    anchors = {"Nordjylland": (9.85, 57.15), "Midtjylland": (9.25, 56.2),
               "Syddanmark": (8.95, 55.35), "Hovedstaden": (12.3, 56.0),
               "Sjælland": (11.65, 55.42)}
    for feature in geometry["features"]:
        props = feature.get("properties", {})
        name = _region_name(next((v for k, v in props.items()
                                  if "nav" in k.lower() or "name" in k.lower()), ""))
        value = values.get(name, np.nan)
        color = cmap(norm(value)) if np.isfinite(value) else PALETTE["missing"]
        geom = feature["geometry"]
        polygons = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                    else [geom["coordinates"]])
        for polygon in polygons:
            ring = np.asarray(polygon[0], float)
            if ring[:, 0].mean() > 14:
                ring = ring + shift
            ax.add_patch(Polygon(ring, closed=True, facecolor=color,
                                 edgecolor=PALETTE["ink_muted"], lw=0.25, zorder=2))
            if _includes(intervals, name, reference):
                ax.add_patch(Polygon(ring, closed=True, facecolor="none", hatch="//////",
                                     edgecolor=(1, 1, 1, 0.75), lw=0, zorder=3))
        if name in anchors:
            ax.text(*anchors[name], name, ha="center", va="bottom", fontsize=5.6,
                    color=PALETTE["ink_secondary"], path_effects=halo, zorder=4)
            if np.isfinite(value):
                # Round first so a tiny negative never prints as "-0.000".
                shown = round(float(value), 3) + 0.0
                ax.text(anchors[name][0], anchors[name][1] - 0.02, fmt.format(shown),
                        ha="center", va="top", fontsize=6.6, fontweight="bold",
                        color=PALETTE["ink"], path_effects=halo, zorder=4)
                if intervals and name in intervals:
                    low, high = (round(float(v), 3) + 0.0 for v in intervals[name])
                    ax.text(anchors[name][0], anchors[name][1] - 0.17,
                            f"{fmt.format(low)} to {fmt.format(high)}", ha="center",
                            va="top", fontsize=4.9, color=PALETTE["ink_secondary"],
                            path_effects=halo, zorder=4)
    ax.add_patch(Rectangle((12.12, 56.42), 0.78, 0.52, fill=False,
                           edgecolor=PALETTE["grid"], lw=0.6))
    ax.text(12.16, 56.9, "Bornholm", fontsize=5, va="top", color=PALETTE["ink_muted"])
    ax.set_xlim(8.0, 13.0)
    ax.set_ylim(54.55, 57.8)


def _includes(intervals, name, reference):
    if not intervals or reference is None or name not in intervals:
        return False
    low, high = intervals[name]
    return bool(np.isfinite(low) and np.isfinite(high) and low <= reference <= high)


def _region_forest(ax, table, series, value="gain", reference=0.0):
    """Rows: regions; one marker per series with its 95% interval."""
    regions = [r for r in REGION_TILES if r in set(table.region)]
    offsets = np.linspace(-0.14, 0.14, len(series)) if len(series) > 1 else [0]
    for offset, (key, column, (color, label)) in zip(offsets, series):
        rows = table.loc[table[column] == key].set_index("region").reindex(regions)
        y = np.arange(len(regions)) + offset
        ax.errorbar(rows[value], y, xerr=[rows[value] - rows.lower, rows.upper - rows[value]],
                    fmt="o" if key.endswith("-S") else "D", markersize=3.6, color=color, ecolor=color, elinewidth=1.2,
                    capsize=0, label=label)
    if reference is not None:
        ax.axvline(reference, color=PALETTE["zero_line"], lw=0.7)
    ax.set_yticks(range(len(regions)), regions, fontsize=6.8)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(4))
    despine(ax, "x")


def _horizon_phrase(horizon):
    """'All-horizon' for the pooled panel (horizon_days == 0), else e.g. 'One-year'."""
    if horizon == 0:
        return "All-horizon"
    return {365: "One-year", 730: "Two-year"}.get(horizon, f"{horizon}-day")


def figure_regions(effects, models=None, coverage=None, geometry=None, *, horizon=0,
                   mark=None):
    """One national model across Denmark's regions.

    Test patients are grouped by the region of most of their pre-treatment
    contacts and every comparison is repeated within each region (same fits).
    ``effects``: per-region panel effects (``region`` or ``stratum_value``,
    ``contrast`` such as ``J-X``/``J-S``, ``metric``, ``gain``, ``lower``,
    ``upper``). ``models``: per-region absolute metrics (``model``, ``metric``,
    ``estimate``, ``lower``, ``upper``). ``coverage``: per-region medians of
    recorded events and laboratory results before treatment. ``geometry``:
    region GeoJSON; by default the packaged geoBoundaries regions, ``False``
    for schematic tiles.

    ``horizon``: 0 (default) is the panel pooled over every horizon, each
    disease-outcome-horizon cell weighing equally, as in Table 2.
    Maps: A, AUC gain over XGBoost of OPERA (the configuration chosen on tuning
    data, when analysed) or else the joint model; B, its AUC; C, its calibration
    in the large; each with its 95% CI, hatched where the interval includes 0.
    Below, with 95% CI: D, gains over the specialist and XGBoost; E, AUC of the
    joint model, OPERA and XGBoost; F, calibration in the large; G, records
    before treatment.
    """
    import json
    from pathlib import Path

    if geometry is None:
        packaged = Path(__file__).resolve().parents[1] / "resources" / "denmark_regions.geojson"
        geometry = json.loads(packaged.read_text()) if packaged.is_file() else False

    def regional(frame):
        frame = frame.loc[frame.horizon_days == horizon].copy() if "horizon_days" in frame \
            else frame.copy()
        source = "region" if "region" in frame else "stratum_value"
        frame["region"] = frame[source].map(_region_name)
        return frame

    effects = regional(effects)
    auc = effects.loc[effects.metric == "auc"]
    absolute = (regional(models).rename(columns={"estimate": "gain"})
                if models is not None and len(models) else None)
    fig = plt.figure(figsize=(8.8, 10.2))
    maps = fig.add_gridspec(1, 3, wspace=0.08, left=0.01, right=0.99, top=0.95, bottom=0.70)

    def intervals_of(frame):
        if not {"lower", "upper"} <= set(frame):
            return None
        rows = frame.set_index("region")
        return {r: (rows.lower[r], rows.upper[r]) for r in rows.index}

    def map_panel(index, letter, title, values, cmap, norm, label, fmt, intervals=None,
                  reference=None):
        ax = fig.add_subplot(maps[index])
        _draw_denmark(ax, values, geometry, norm, cmap, fmt=fmt, intervals=intervals,
                      reference=reference)
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=TITLE_SIZE, color=PALETTE["ink"],
                     pad=6)
        box = ax.get_position()
        bar = fig.add_axes([box.x0 + 0.03, box.y0 - 0.035, box.width - 0.06, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar,
                          orientation="horizontal")
        cb.ax.tick_params(labelsize=5.8)
        cb.ax.xaxis.set_major_locator(MaxNLocator(3))
        cb.outline.set_visible(False)
        cb.set_label(label, fontsize=6.2)

    # Default: the panel pooled over every horizon (the headline estimand).
    when = _horizon_phrase(horizon)
    # The best model (OPERA) where it was analysed, else the joint model.
    lead = "OPERA" if (auc.contrast == "OPERA-X").any() else "J"
    lead_name = "OPERA" if lead == "OPERA" else "Joint model"
    vs_x = auc.loc[auc.contrast == f"{lead}-X"]
    gain = vs_x.set_index("region").gain
    span = (float(np.nanmax(np.abs(gain))) or 0.01) if np.isfinite(gain).any() else 0.01
    map_panel(0, "A", "Gain over XGBoost", gain.to_dict(), GAIN_CMAP,
              TwoSlopeNorm(0, -span, span), f"{when} AUC gain", "{:+.3f}",
              intervals_of(vs_x), reference=0.0)
    if absolute is not None:
        joint = absolute.loc[(absolute.metric == "auc") & (absolute.model == lead)]
        values = joint.set_index("region").gain
        low, high = (float(values.min()), float(values.max())) if np.isfinite(values).any() else (0., 1.)
        pad = max((high - low) * 0.15, 0.005)
        map_panel(1, "B", f"{lead_name} AUC", values.to_dict(), ATLAS_CMAP,
                  plt.Normalize(low - pad, high + pad), f"{when} AUC", "{:.3f}",
                  intervals_of(joint))
        bias_rows = absolute.loc[(absolute.metric == "calibration_bias")
                                 & (absolute.model == lead)]
        bias = bias_rows.set_index("region").gain
        span = (float(np.nanmax(np.abs(bias))) or 0.01) if np.isfinite(bias).any() else 0.01
        map_panel(2, "C", "Calibration in the large", bias.to_dict(), BIAS_CMAP,
                  TwoSlopeNorm(0, -span, span), "Observed − predicted risk", "{:+.3f}",
                  intervals_of(bias_rows), reference=0.0)
    fig.text(0.01, 0.004, ("Panel pooled over every horizon. " if horizon == 0 else "")
             + "Maps: 95% CI under each value; hatched where the interval includes 0.",
             fontsize=6.2, color=PALETTE["ink_muted"], ha="left", va="bottom")

    lower = fig.add_gridspec(2, 2, wspace=.40, hspace=1.0, left=.17, right=.97, top=.54,
                             bottom=.075)
    styles = {"J-S": (MODEL_STYLE["J"][0], "Joint vs specialist"),
              "J-X": (MODEL_STYLE["J"][0], "Joint vs XGBoost")}
    model_styles = {"J": (MODEL_STYLE["J"][0], "Joint"), "X": (MODEL_STYLE["X"][0], "XGBoost")}
    if lead == "OPERA":
        styles["OPERA-X"] = (MODEL_STYLE["OPERA"][0], "OPERA vs XGBoost")
        model_styles = {"J": model_styles["J"], "OPERA": MODEL_STYLE["OPERA"],
                        "X": model_styles["X"]}

    def header(ax, letter, title):
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=TITLE_SIZE, pad=44,
                     color=PALETTE["ink"])

    ax = fig.add_subplot(lower[0])
    _region_forest(ax, auc, [(k, "contrast", v) for k, v in styles.items()])
    ax.set_xlabel("AUC gain (95% CI)", fontsize=6.8)
    ax.legend(frameon=False, fontsize=5.9, loc="lower left", bbox_to_anchor=(-0.05, 1.0),
              ncol=1, handletextpad=0.1)
    header(ax, "D", "Gain")
    if absolute is not None:
        ax = fig.add_subplot(lower[1])
        _region_forest(ax, absolute.loc[absolute.metric == "auc"],
                       [(k, "model", v) for k, v in model_styles.items()], reference=None)
        ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("AUC (95% CI)", fontsize=6.8)
        ax.legend(frameon=False, fontsize=5.9, loc="lower left", bbox_to_anchor=(-0.05, 1.0),
                  ncol=1, handletextpad=0.1)
        header(ax, "E", "Performance")
        ax = fig.add_subplot(lower[2])
        _region_forest(ax, absolute.loc[absolute.metric == "calibration_bias"],
                       [(k, "model", v) for k, v in model_styles.items()])
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Observed − predicted\n(95% CI; 0 ideal)", fontsize=6.8)
        header(ax, "F", "Calibration")
    if coverage is not None and len(coverage):
        ax = fig.add_subplot(lower[3])
        cov = coverage.assign(region=coverage.region.map(_region_name)).set_index("region")
        regions = [r for r in REGION_TILES if r in cov.index]
        y = np.arange(len(regions))
        for offset, column, color, label in (
                (-0.15, "history_event_count", "#60748B", "All records"),
                (0.15, "laboratory_event_count", "#B87C52", "Laboratory")):
            if f"{column}_median" not in cov:
                continue
            rows = cov.reindex(regions)
            ax.errorbar(rows[f"{column}_median"], y + offset,
                        xerr=[rows[f"{column}_median"] - rows[f"{column}_q1"],
                              rows[f"{column}_q3"] - rows[f"{column}_median"]],
                        fmt="o", markersize=3.2, color=color, ecolor=color, elinewidth=1.1,
                        capsize=0, label=label)
        ax.set_xscale("log")
        from matplotlib.ticker import LogLocator
        ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4))
        ax.margins(x=.2)
        ax.set_yticks(y, regions, fontsize=6.4)
        ax.invert_yaxis()
        ax.set_xlabel("Records before\ntreatment (median, IQR)", fontsize=6.8)
        ax.legend(frameon=False, fontsize=5.9, loc="lower left", bbox_to_anchor=(-0.05, 1.0),
                  ncol=1, handletextpad=0.1)
        header(ax, "G", "Data")
        despine(ax, "x")
    watermark(fig, mark)
    return fig


def figure_subgroup_effects(table, *, contrast_styles, covariate_order, xlabel,
                            title_letters="A", panel_title="", scale=100, mark=None,
                            ax=None, row_order=None, show_legend=True):
    """Patient subgroup effects (diamond, 95% CI), grouped by covariate."""
    rows, labels, headers = [], [], []
    for covariate, title in covariate_order.items():
        source = table if row_order is None else row_order
        part = source.loc[source.covariate == covariate]
        if part.empty:
            continue
        headers.append((len(labels), title))
        for subgroup in part.subgroup.drop_duplicates():
            rows.append((covariate, subgroup))
            labels.append(f"  {subgroup}")
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4.6, max(3.0, 0.24 * len(labels) + 0.9 * len(headers) * 0.3
                                        + 1.0)))
    else:
        fig = ax.figure
    positions, y = {}, 0.0
    header_y = []
    for index, row in enumerate(rows):
        if any(start == index for start, _ in headers):
            header_y.append(y)
            y += 1.0
        positions[row] = y
        y += 1.0
    offsets = np.linspace(-0.2, 0.2, len(contrast_styles)) if len(contrast_styles) > 1 else [0]
    for (contrast, (color, label)), offset in zip(contrast_styles.items(), offsets):
        part = table.loc[table.contrast == contrast]
        for row in part.itertuples():
            key = (row.covariate, row.subgroup)
            if key not in positions:
                continue
            yy = positions[key] + offset
            ax.plot([row.lower * scale, row.upper * scale], [yy, yy], color=color,
                    lw=1.3, alpha=0.5)
            ax.scatter(row.gain * scale, yy, marker="D", s=22, color=color, zorder=3,
                       label=label)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_yticks(list(positions.values()), [s for _, s in rows], fontsize=7)
    for (start, title), yy in zip(headers, header_y):
        ax.text(-0.02, yy, title, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=7.4, fontweight="bold", color=PALETTE["ink"])
    ax.set_ylim(y - 0.4, -0.6)
    if not rows:
        ax.text(0.5, 0.5, "No matching subgroup estimates",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
    if table.empty and rows:
        ax.text(0.5, 0.5, "No supported estimates available", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=PALETTE["ink_secondary"])
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    handles = [Line2D([], [], ls="none", marker="D", color=color, label=label)
               for color, label in contrast_styles.values()]
    if show_legend:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
                  frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0.3)
    if panel_title:
        _panel_title(ax, title_letters, panel_title)
    despine(ax, "x")
    if standalone:
        fig.subplots_adjust(left=0.42, right=0.97, top=1 - 0.75 / fig.get_figheight(),
                            bottom=0.75 / fig.get_figheight())
        watermark(fig, mark)
    return fig


def figure_who_benefits(table, auc, *, contrast_styles, covariate_order, mark=None):
    """Paired subgroup AUC and Brier gains, with a common row inventory."""
    rows = pd.concat([table, auc], ignore_index=True).drop_duplicates(
        ["covariate", "subgroup"])
    height = max(3.2, 0.24 * len(rows) + 0.27 * len(covariate_order) + 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, height))
    for ax, data, letter, title, label in zip(
        axes, [auc, table], "AB", ["Discrimination", "Probability accuracy"],
        ["AUC improvement (×100, 95% CI)", "Brier score improvement (×100, 95% CI)"]
    ):
        figure_subgroup_effects(
            data, contrast_styles=contrast_styles, covariate_order=covariate_order,
            xlabel=label, title_letters=letter, panel_title=title,
            ax=ax, row_order=rows, show_legend=False)
    handles = [Line2D([], [], ls="none", marker="D", color=color, label=label)
               for color, label in contrast_styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.19, right=0.98, wspace=0.85,
                        top=1 - 0.9 / height, bottom=0.75 / height)
    watermark(fig, mark)
    return fig


def figure_coverage(coverage, *, mark=None):
    """Which disease–outcome cells every comparison could be evaluated in."""
    diseases, disease_group = _ordered(coverage, "cohort", "cohort_group")
    outcomes, outcome_group = _ordered(coverage, "outcome", "endpoint_group")
    coverage = coverage.copy()
    # Historical exports used fit_failed for any missing specialist, without a log.
    if "failure_verified" not in coverage:
        coverage.loc[coverage.status.eq("fit_failed"), "status"] = "specialist_unavailable"
    else:
        unverified = coverage.status.eq("fit_failed") & ~coverage.failure_verified.fillna(False).astype(bool)
        coverage.loc[unverified,"status"] = "specialist_unavailable"
    keys = [k for k in COVERAGE_STYLE if k in set(coverage.status)]
    codes = (coverage.assign(code=coverage.status.map({k: i for i, k in enumerate(keys)}))
             .pivot_table(index="cohort", columns="outcome", values="code", aggfunc="first")
             .reindex(index=diseases, columns=outcomes))
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([COVERAGE_STYLE[k][0] for k in keys])
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.imshow(codes.to_numpy(float), cmap=cmap,
              norm=BoundaryNorm(np.arange(len(keys) + 1) - 0.5, len(keys)),
              aspect="auto", interpolation="nearest")
    breaks, starts = _group_breaks(outcomes, outcome_group)
    for position in breaks:
        ax.axvline(position, color="white", lw=1.4)
    for position in _group_breaks(diseases, disease_group)[0]:
        ax.axhline(position, color="white", lw=1.4)
    ends = [*breaks, len(outcomes) - 0.5]
    centres = [(start - 0.5 + end) / 2 for (start, _), end in zip(starts, ends)]
    ax.set_xticks(centres, [SHORT_GROUP.get(g, g) for _, g in starts], rotation=45,
                  ha="right", rotation_mode="anchor", fontsize=6.4)
    ax.set_yticks(range(len(diseases)), diseases, fontsize=5.8)
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    counts = coverage.status.value_counts()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COVERAGE_STYLE[k][0],
                             edgecolor=PALETTE["panel_border"], lw=0.5,
                             label=f"{COVERAGE_STYLE[k][1]} ({counts.get(k, 0):,})")
               for k in keys if counts.get(k, 0)]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.17, right=0.99, top=0.86, bottom=0.17)
    watermark(fig, mark)
    return fig


def figure_support(cells, training=None, coverage=None, *, horizon=ALL_HORIZONS, edges=(20, 100, 500), mark=None):
    """Training events available to each disease–outcome specialist."""
    source = cells if training is None else training
    data = (source.loc[_horizon_mask(source, _single(horizon))]
            .drop_duplicates(["cohort", "outcome"]))
    if coverage is not None:
        from opera.visualization.paper_taxonomy import fine_cohort_groups, disease_group_name, disease_name
        roster=coverage[['cohort','outcome','endpoint_group','status']].drop_duplicates(['cohort','outcome'])
        traincols=['cohort','outcome','n_events_train']
        data=roster.merge(data[traincols],on=['cohort','outcome'],how='left',validate='one_to_one')
        group_of={disease_name(c):disease_group_name(g) for c,g in fine_cohort_groups().items()}
        data['cohort_group']=data.cohort.map(group_of).fillna('Other')
    diseases, disease_group = _ordered(data, "cohort", "cohort_group")
    outcomes, outcome_group = _ordered(data, "outcome", "endpoint_group")
    matrix = (data.pivot_table(index="cohort", columns="outcome", values="n_events_train",
                               aggfunc="first").reindex(index=diseases, columns=outcomes))
    not_applicable=np.zeros(matrix.shape,dtype=bool)
    if 'status' in data:
        not_applicable=(data.pivot_table(index='cohort',columns='outcome',values='status',aggfunc='first')
                         .reindex(index=diseases,columns=outcomes).eq('not_applicable').to_numpy())
    fig = plt.figure(figsize=(8.8, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.34], wspace=0.2, left=0.15,
                            right=0.98, top=0.9, bottom=0.2)
    ax = fig.add_subplot(grid[0])
    from matplotlib.colors import LogNorm
    support_cmap = plt.get_cmap("viridis").copy()
    support_cmap.set_bad(PALETTE["missing"])
    values = matrix.to_numpy(float)
    image = ax.imshow(np.ma.masked_where(~np.isfinite(values) | (values <= 0), values),
                      cmap=support_cmap, norm=LogNorm(1, max(10, np.nanmax(values))),
                      aspect="auto", interpolation="nearest")
    if not_applicable.any():
        from matplotlib.colors import ListedColormap
        ax.imshow(np.ma.masked_where(~not_applicable,np.ones(matrix.shape)),
                  cmap=ListedColormap(["white"]),aspect="auto",interpolation="nearest")
    for i,j in zip(*np.where(~np.isfinite(values) & ~not_applicable)):
        ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='///',
                               edgecolor='#999999',linewidth=0))
    # Evaluation support is distinct from training support. Mark exclusions,
    # retaining every training cell rather than filtering the heatmap.
    eval_year = cells.loc[(cells.horizon_days == (365 if horizon == ALL_HORIZONS else horizon))]
    if {'n_cases','n_controls'} <= set(eval_year):
        eval_year = eval_year.drop_duplicates(['cohort','outcome'])
        for row in eval_year.itertuples():
            if row.cohort in diseases and row.outcome in outcomes and (
                    row.n_cases < 5 or row.n_controls < 5):
                ax.scatter(outcomes.index(row.outcome),diseases.index(row.cohort),
                           s=3,marker='x',linewidths=.35,color='white')
        fig.text(.15,.015,'× Evaluation excluded: fewer than 5 cases or controls. '
                 'This is an evaluation rule, not a training-event cutoff.',fontsize=6.4)
    breaks, starts = _group_breaks(outcomes, outcome_group)
    for position in breaks:
        ax.axvline(position, color="white", lw=1.4)
    for position in _group_breaks(diseases, disease_group)[0]:
        ax.axhline(position, color="white", lw=1.4)
    ends = [*breaks, len(outcomes) - 0.5]
    centres = [(start - 0.5 + end) / 2 for (start, _), end in zip(starts, ends)]
    ax.set_xticks(centres, [SHORT_GROUP.get(g, g) for _, g in starts], rotation=45,
                  ha="right", rotation_mode="anchor", fontsize=6.4)
    ax.set_yticks(range(len(diseases)), diseases, fontsize=5.8)
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_title(ax, "A", "Training events per disease and outcome")
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.24,
                       aspect=45, shrink=0.55)
    support_label = ("One-year training events (grey: zero; hatched: unavailable; white: not applicable)"
                     if horizon == ALL_HORIZONS else
                     f"Training events by {horizon} days (grey: zero; hatched: unavailable; white: not applicable)")
    bar.set_label(support_label, fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    bar.outline.set_visible(False)
    ax = fig.add_subplot(grid[1])
    counts = np.sort(values.ravel()[(np.isfinite(values) & ~not_applicable).ravel()])
    if not len(counts):
        raise ValueError('No known training support for the cumulative distribution')
    share = np.arange(1, len(counts) + 1) / len(counts)
    ax.step(np.clip(counts, 0.8, None), share, where="post", color=MODEL_STYLE["J"][0],
            lw=1.5)
    for edge in edges:
        fraction = float(np.mean(counts < edge))
        ax.axvline(edge, color=PALETTE["grid"], lw=1)
        ax.scatter(edge, fraction, s=10, color=MODEL_STYLE["J"][0], zorder=3)
        ax.text(edge / 1.25, fraction + 0.015, f"{fraction:.0%} < {edge}", fontsize=6.4,
                ha="right", va="bottom", color=PALETTE["ink_secondary"])
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_xlabel("Training events (log)")
    ax.set_ylabel("Share of cells")
    ax.set_ylim(0, 1.02)
    unknown=(~np.isfinite(values) & ~not_applicable).sum()
    notes=[f'{len(counts):,} known counts']
    if (counts == 0).any():
        notes.append(f'{(counts == 0).sum():,} zero-event cells (left edge)')
    if unknown:
        notes.append(f'{unknown:,} counts unavailable (excluded)')
    fig.text(.755,.105,'\n'.join(notes),fontsize=6.2,color=PALETTE['ink_secondary'],va='top')
    _panel_title(ax, "B", "Cumulative")
    despine(ax, "y")
    watermark(fig, mark)
    return fig


# Router decomposition: own corrections, borrowing, both (lock addenda 2026-10-26).
HIERARCHY_STYLE = {"own": ("#B18B4D", "Own-task vs global-only model"),
                   "borrow": (MODEL_STYLE["R"][0], "Full OPERA vs own-task model"),
                   "both": (PALETTE["opera"], "Full OPERA vs global-only model")}
SUPPORT_BANDS = [(0, 20, "<20"), (20, 100, "20–100"), (100, 500, "100–500"),
                 (500, np.inf, "≥500")]
BAND_COLORS = ["#C3CAE6", "#8C9AD0", "#5367B0", "#1F2A72"]  # blue arm of GAIN_CMAP


def _log_support_fit(x, y, grid, *, n_boot=200, seed=0):
    """Quadratic in log10 training events, with a 95% bootstrap band over cells
    (descriptive; the inferential support curves are the hierarchical ones)."""
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    if not len(x):
        return (np.full(len(grid), np.nan),) * 3
    degree = min(2, len(np.unique(x)) - 1)
    rng = np.random.default_rng(seed)
    lx, lg = np.log10(x), np.log10(grid)
    centre = np.polyval(np.polyfit(lx, y, degree), lg)
    fits = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        fitted_degree = min(degree, len(np.unique(lx[i])) - 1)
        fits.append(np.polyval(np.polyfit(lx[i], y[i], fitted_degree), lg))
    return centre, *np.quantile(np.array(fits), [0.025, 0.975], axis=0)


def figure_support_sensitivity(cells, *, horizon=365, cutoffs=(5, 10, 20, 50),
                               contrasts=None, mark=None):
    """Descriptive support sensitivity on one common finite comparison roster."""
    if contrasts is None:
        contrasts = ('J-S', 'OPERA-J' if 'OPERA-J' in set(cells.contrast) else 'R1')
    data = cells.loc[cells.metric.eq('auc') & _horizon_mask(cells, horizon)
                     & cells.contrast.isin(contrasts)].copy()
    if not {'n_cases', 'n_controls'} <= set(data):
        raise ValueError('Support sensitivity requires actual evaluation cases and controls')
    keys = ['cohort', 'outcome', 'horizon_days']
    valid = data.loc[data.gain.notna()].groupby(keys).contrast.nunique()
    shared = valid[valid.eq(len(contrasts))].index
    data = data.loc[pd.MultiIndex.from_frame(data[keys]).isin(shared)]
    if data.empty:
        raise ValueError('No common supported comparison roster')
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
    fig.subplots_adjust(left=.10, right=.97, bottom=.24, top=.88, wspace=.5)
    for code in contrasts:
        selected = data.loc[data.contrast.eq(code)]
        means, counts = [], []
        for cutoff in cutoffs:
            kept = selected.loc[selected.n_cases.ge(cutoff) & selected.n_controls.ge(cutoff)]
            means.append(kept.gain.mean())
            counts.append(len(kept))
        color, label = CONTRAST_STYLE.get(code, (MODEL_STYLE['OPERA'][0], 'OPERA vs joint'))
        axes[0].plot(cutoffs, means, marker='D', ms=4, lw=1.4, color=color, label=label)
        axes[1].plot(cutoffs, counts, marker='D', ms=4, lw=1.4, color=color)
    for ax, letter, title in zip(axes, 'AB', ['Performance on retained cells', 'How many cells remain']):
        ax.set_xscale('log')
        ax.set_xticks(cutoffs, [str(c) for c in cutoffs])
        ax.minorticks_off()
        ax.set_xlabel('Minimum cases and minimum controls')
        _panel_title(ax, letter, title)
        despine(ax, 'y')
    axes[0].axhline(0, color=PALETTE['zero_line'], lw=.7)
    axes[0].set_ylabel('AUC gain (descriptive mean)')
    axes[1].set_ylabel('Disease–outcome cells')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower left',
               bbox_to_anchor=(.09, .025), ncol=2, frameon=False, fontsize=8)
    watermark(fig, mark)
    return fig


def figure_whole_cohort(estimates, coverage, *, horizon=365, page=1, per_page=12, mark=None):
    """Common-population pooled AUC alongside each model's prediction coverage."""
    from opera.visualization.outcome_names import outcome_display_name
    data = estimates.loc[estimates.metric.eq('auc') & estimates.horizon_days.eq(horizon)].copy()
    cover = coverage.loc[coverage.horizon_days.eq(horizon)].copy()
    outcomes = sorted(set(data.outcome) | set(cover.outcome))[(page-1)*per_page:page*per_page]
    if not outcomes:
        raise ValueError('Whole-cohort page has no outcomes')
    models = list(dict.fromkeys(cover.model))
    names = {'J_L': 'Joint', 'R_L': 'OPERA (local)', 'R_AB_L': 'OPERA (local)',
             'OPERA': 'OPERA', 'X': 'Specialist XGBoost portfolio', 'XP': 'Disease-pooled XGBoost'}
    colors = {'J_L': MODEL_STYLE['J'][0], 'X': MODEL_STYLE['X'][0], 'XP': '#B18B4D'}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, max(4.8, .38*len(outcomes)+1.6)), sharey=True)
    fig.subplots_adjust(left=.29, right=.97, top=.86, bottom=.22, wspace=.35)
    for offset, model in zip(np.linspace(-.23, .23, len(models)), models):
        color = colors.get(model, MODEL_STYLE['OPERA'][0])
        part = data.loc[data.model.eq(model)].set_index('outcome').reindex(outcomes)
        if len(part):
            axes[0].hlines(np.arange(len(outcomes))+offset, part.lower, part.upper, color=color, alpha=.55, lw=1)
            axes[0].scatter(part.estimate, np.arange(len(outcomes))+offset, color=color, marker='D', s=16)
        part = cover.loc[cover.model.eq(model)].set_index('outcome').reindex(outcomes)
        if 'union_n' in part:
            share = part.available_n / part.union_n
        else:
            share = part.available_n / cover.groupby('outcome').available_n.max().reindex(outcomes)
        axes[1].scatter(share, np.arange(len(outcomes))+offset, color=color, marker='D', s=16,
                        label=names.get(model, model))
    for ax, letter, title in zip(axes, 'AB', ['AUC on the common population', 'Prediction coverage']):
        ax.set_yticks(range(len(outcomes)), [outcome_display_name(o) for o in outcomes], fontsize=7)
        ax.set_ylim(len(outcomes)-.5, -.5)
        _panel_title(ax, letter, title)
        despine(ax, 'x')
    axes[0].set_xlabel('Pooled AUC (paired 95% CI)')
    axes[0].xaxis.set_major_locator(MaxNLocator(4))
    axes[1].set_xlim(-.03, 1.03)
    axes[1].set_xticks([0, .5, 1], ['0%', '50%', '100%'])
    axes[1].set_xlabel('Patients with predictions / union roster')
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower left',
               bbox_to_anchor=(.28, .02), ncol=2, frameon=False, fontsize=7)
    watermark(fig, mark)
    return fig


def figure_hierarchy(cells, *, metric="auc", horizon=ALL_HORIZONS, mark=None):
    """Where pooling hurts, and how routing's own corrections and borrowing
    depend on training support.

    ``cells`` is ``hierarchy_cells.csv``: per cell, ``tuning_joint_vs_specialist``
    and the evaluation-split gains ``own`` (home-only vs global-only),
    ``borrow`` (routed vs home-only) and ``both`` (routed vs global-only).
    A: cells by tuning harm, coloured by support band. B: own corrections and
    borrowing against training events. C: median and IQR by support band.
    """
    data = cells.loc[(cells.metric == metric) & _horizon_mask(cells, horizon)
                     & cells.n_events_train.notna()].copy()
    data["band"] = np.digitize(data.n_events_train, [b[1] for b in SUPPORT_BANDS[:-1]])
    fig = plt.figure(figsize=(7.2, 6.6))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.34, left=0.095,
                           right=0.975, top=0.93, bottom=0.6)
    bottom = fig.add_gridspec(1, 1, left=0.3, right=0.975, top=0.41, bottom=0.075)
    unit = {"auc": "AUC", "brier": "Brier", "log_loss": "Log loss"}.get(metric, metric)

    # A: cells by tuning harm, coloured by support band (the rows of C).
    ax = fig.add_subplot(top[0])
    shown = estimable_cells(data.rename(columns={"both": "gain", "both_lower": "lower",
                                                 "both_upper": "upper"}))
    for i, color in enumerate(BAND_COLORS):
        part = shown.loc[shown.band == i]
        ax.scatter(part.tuning_joint_vs_specialist, part.gain, s=6, color=color,
                   alpha=0.85, linewidths=0, rasterized=True, zorder=2 + i)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    harmed = data.loc[data.tuning_joint_vs_specialist < 0]
    if len(harmed):
        ax.text(0.03, 0.03, f"Left of the line (hurt by pooling):\n"
                f"{(harmed.both > 0).mean():.0%} higher vs global-only,\n"
                f"median {harmed.both.median():+.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.5, color=PALETTE["ink_secondary"],
                linespacing=1.3, zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    key = [Line2D([], [], ls="", marker="s", markersize=5, color=c, label=b[2])
           for c, b in zip(BAND_COLORS, SUPPORT_BANDS)]
    ax.legend(handles=key, title="Training events", loc="upper right", ncol=2,
              frameon=False, fontsize=6.5, title_fontsize=6.5, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0.2)
    ax.set_xlabel(f"Joint vs specialist, tuning ({unit})")
    ax.set_ylabel(f"Routed vs global-only ({unit})")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    _panel_title(ax, "A", "Cells hurt by pooling")
    despine(ax, "y")

    # B: own corrections and borrowing against support; grey cells, coloured curves.
    ax = fig.add_subplot(top[1])
    x = data.n_events_train.clip(lower=1).to_numpy(float)
    grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 120)
    curves = {}
    for column in ("own", "borrow"):
        points = estimable_cells(data.rename(columns={
            column: "gain", f"{column}_lower": "lower", f"{column}_upper": "upper"}))
        ax.scatter(points.n_events_train.clip(lower=1), points.gain, s=3, color="#9AA3B5",
                   alpha=0.18, linewidths=0, rasterized=True, zorder=2)
        color = HIERARCHY_STYLE[column][0]
        centre, low, high = _log_support_fit(x, data[column].to_numpy(float), grid)
        ax.fill_between(grid, low, high, color=color, alpha=0.18, linewidth=0, zorder=3)
        ax.plot(grid, centre, color=color, lw=2.0, zorder=4, solid_capstyle="round")
        curves[column] = centre
    for column, offset, va in (("borrow", 8, "bottom"), ("own", -8, "top")):
        ax.annotate({'own': 'Own-task corrections', 'borrow': 'Borrowed corrections'}[column],
                    (grid[2], curves[column][2]), xytext=(0, offset),
                    textcoords="offset points", color=HIERARCHY_STYLE[column][0],
                    fontsize=7.5, fontweight="bold", va=va)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
    ax.set_xscale("log")
    edges = [x.min()] + [b[1] for b in SUPPORT_BANDS[:-1]] + [x.max()]
    for i, color in enumerate(BAND_COLORS):
        if edges[i + 1] > edges[i]:
            ax.axvspan(edges[i], edges[i + 1], ymin=0, ymax=0.022, color=color, lw=0,
                       zorder=5)
    ax.set_xlabel("Target training events (log scale)")
    ax.set_ylabel(f"{unit} gain")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    _panel_title(ax, "B", "Own correction and borrowing")
    despine(ax, "y")

    # C: median and IQR by support band, with the band's cell count as a bar.
    ax = fig.add_subplot(bottom[0])
    labels = [f"{b[2]} events" for b in SUPPORT_BANDS]
    data["band_label"] = [labels[i] for i in data.band]
    long = data.melt(id_vars=["band_label"], value_vars=["own", "borrow", "both"],
                     var_name="series", value_name="gain")
    summary = _summary_rows(long, ["band_label", "series"])
    present = [label for label in labels if label in set(data.band_label)]
    handles = _diamond_forest(ax, summary, row_key="band_label", row_order=present,
                              series_key="series", series_style=HIERARCHY_STYLE,
                              xlabel=f"{unit} gain (median, IQR)", show_counts=False,
                              spread=0.24)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    counts = data.band_label.value_counts().reindex(present)
    bars = ax.inset_axes([-0.16, 0, 0.14, 1])
    bars.barh(range(len(present)), counts, height=0.42, linewidth=0,
              color=[BAND_COLORS[labels.index(label)] for label in present])
    bars.set_xlim(counts.max() * 1.25, 0)
    bars.set_ylim(ax.get_ylim())
    ax.set_yticks(range(len(present)), [])
    bars.set_yticks(range(len(present)),
                    [f"{label} ({n:,})" for label, n in zip(present, counts)])
    bars.tick_params(axis="y", length=0, labelsize=7, pad=2)
    bars.tick_params(axis="x", length=0, labelbottom=False)
    for side in bars.spines.values():
        side.set_visible(False)
    bars.set_title("Cells", fontsize=6.4, color=PALETTE["ink_secondary"], loc="right", pad=2)
    ax.tick_params(axis="y", length=0)
    fig.text(0.095, 0.49, "C  By training support", fontsize=10.5, color=PALETTE["ink"],
             ha="left", va="bottom")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=1,
              frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.1)
    watermark(fig, mark)
    return fig


def figure_rescue(cells, *, metric="auc", horizon=ALL_HORIZONS, tolerance=0.0, mark=None):
    """Cell scatter and summaries within frozen tuning-defined harm strata."""
    data = cells.loc[(cells.metric == metric) & _horizon_mask(cells, horizon)].copy()
    data['stratum'] = np.where(data.tuning_joint_vs_specialist < -tolerance,
                                'Joint worse on tuning', 'Joint not worse on tuning')
    data['routed_vs_joint'] = data.routed_vs_specialist - data.joint_vs_specialist
    strata = ['Joint worse on tuning', 'Joint not worse on tuning']
    fig = plt.figure(figsize=(9.2, 4.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=.10, right=.98,
                            top=.82, bottom=.24, wspace=.70)
    ax = fig.add_subplot(grid[0])
    harmed = data.stratum == strata[0]
    ax.scatter(data.joint_vs_specialist, data.routed_vs_joint, s=12, alpha=.48,
                c=np.where(harmed, '#C16A58', '#8995A4'), linewidths=0, rasterized=True)
    ax.axhline(0, color=PALETTE['zero_line'], lw=.8)
    ax.axvline(0, color=PALETTE['grid'], lw=1)
    ax.set_xlabel(f'Joint vs specialist ({_split_word().lower()} AUC)')
    ax.set_ylabel('OPERA vs joint (AUC)')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    _panel_title(ax, 'A', 'Cells by tuning harm')
    despine(ax, 'both')
    ax = fig.add_subplot(grid[1])
    styles = {'joint_vs_specialist': (MODEL_STYLE['J'][0], 'Joint vs specialist'),
              'routed_vs_specialist': (MODEL_STYLE['R'][0], 'OPERA vs specialist')}
    long = data.drop(columns=['gain'], errors='ignore').melt(
        id_vars=['stratum'], value_vars=list(styles), var_name='series', value_name='gain')
    summary = _summary_rows(long, ['stratum', 'series'])
    _diamond_forest(ax, summary, row_key='stratum', row_order=strata,
        series_key='series', series_style=styles,
        xlabel=f'{_split_word()} AUC gain (median, IQR)', show_counts=True)
    ax.set_yticklabels([t.get_text().replace(' on tuning', '\non tuning').replace(' (n=', '\n(n=')
                         for t in ax.get_yticklabels()])
    ax.set_ylim(1.5, -.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, 'B', 'Gains within the tuning strata')
    ax.legend(handles=[Line2D([], [], marker='D', ls='', color=c, label=label)
                         for c, label in styles.values()], loc='lower left',
               bbox_to_anchor=(-.35, 1.12), ncol=1, frameon=False, fontsize=8)
    fig.text(.10, .07, 'Orange: joint worse on tuning. Strata fixed on tuning; tuning-on-tuning results are descriptive.',
              fontsize=7.2, color=PALETTE['ink_secondary'])
    watermark(fig, mark)
    return fig


def figure_strategy(cells, selection, *, horizon=ALL_HORIZONS, mark=None):
    """One joint model versus a specialist, and versus the best model per cell.

    ``cells`` has held-out AUC per cell for ``specialist``, ``joint`` and
    ``best_tuning`` (the specialist, disease-pooled or outcome-pooled model
    chosen per cell on tuning data) and ``n_events_train``; ``selection`` the
    chosen strategy per cell with its ``endpoint_group``.
    """
    fig = plt.figure(figsize=(8.8, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25,1.25], hspace=.65, wspace=.35, left=.22,
                            right=.97, top=.94, bottom=.16)
    data = cells.loc[_horizon_mask(cells, horizon)]
    data = data.copy()
    if 'best_tuning' not in data:
        data['best_tuning'] = np.nan
    if data[['specialist', 'joint']].dropna().empty:
        raise ValueError('Strategy figure has no paired joint/specialist estimates')
    low = float(np.nanmin(data[["specialist", "joint", "best_tuning"]].to_numpy())) - 0.02
    sizes = 4 + 40 * np.sqrt(data.n_events_train / data.n_events_train.max())
    for column, (x, letter, title, label) in enumerate((
            ("specialist", "A", "Joint vs specialist", "Specialist AUC"),
            ("best_tuning", "B", "Joint vs tuning-selected S/D/O", "Tuning-selected S/D/O AUC"))):
        ax = fig.add_subplot(grid[0,column])
        paired = data.loc[data[['joint', x]].notna().all(axis=1)]
        if paired.empty:
            ax.text(.5, .5, 'Comparison unavailable\nRequires held-out predictions',
                    transform=ax.transAxes, ha='center', va='center', fontsize=7,
                    color=PALETTE['ink_muted'])
            _panel_title(ax, letter, title)
            ax.set_axis_off()
            continue
        above = paired.joint > paired[x]
        ax.plot([low, 1], [low, 1], color=PALETTE["diagonal"], lw=0.8, ls="--")
        ax.scatter(paired[x], paired.joint, s=sizes.loc[paired.index], alpha=0.4, linewidths=0,
                   color=np.where(above, MODEL_STYLE["J"][0], "#B5451B"), rasterized=True)
        ax.text(0.04, 0.96, f"joint better: {above.mean():.0%}", transform=ax.transAxes,
                fontsize=6.8, va="top", color=PALETTE["ink_secondary"])
        ax.set_xlim(low, 1)
        ax.set_ylim(low, 1)
        ax.set_aspect("equal", anchor="N")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(label)
        if column == 0:
            ax.set_ylabel("Joint model AUC")
        _panel_title(ax, letter, title)
        despine(ax, "both")
    ax = fig.add_subplot(grid[1,:])
    shares = (selection.groupby("endpoint_group").selected_model.value_counts(normalize=True)
              .unstack().fillna(0))
    order = shares.sort_values("S", ascending=True).index if "S" in shares else shares.index
    left = np.zeros(len(order))
    styles = {"S": (MODEL_STYLE["S"][0], "Specialist"),
              "D": (CONTRAST_STYLE["D-S"][0], "Disease-pooled"),
              "O": (CONTRAST_STYLE["O-S"][0], "Outcome-pooled"),
              "J": (MODEL_STYLE['J'][0], 'Joint'),
              "OPERA": (MODEL_STYLE['OPERA'][0], 'OPERA')}
    unknown = set(selection.selected_model.dropna()) - set(styles)
    if unknown:
        raise ValueError(f'Unknown selected models would be omitted from strategy bars: {sorted(unknown)}')
    candidates = set(selection.selected_model.dropna())
    if 'candidate_models' in selection:
        for names in selection.candidate_models.dropna().unique():
            candidates.update(str(names).replace('_L', '').split('|'))
    overall = selection.selected_model.value_counts(normalize=True)
    for key, (color, label) in styles.items():
        if key not in candidates:
            continue
        values = shares.reindex(order).get(key, pd.Series(0, index=order)).to_numpy()
        ax.barh(range(len(order)), values, left=left, color=color, height=0.72,
                label=f'{label} ({overall.get(key, 0):.0%} overall)')
        left += values
    ax.set_yticks(range(len(order)), [SHORT_GROUP.get(g, g) for g in order], fontsize=6.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("Share of cells")
    ax.grid(False)
    _panel_title(ax, "C", "Model selected on tuning data")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", ncol=3,
               frameon=False, fontsize=8, handlelength=1, columnspacing=0.8,
               bbox_to_anchor=(0.21, 0.045))
    fig.text(.22, .015, 'Overall shares count tuning-selected cells. Panel B retains the original S/D/O policy.',
              fontsize=7, color=PALETTE['ink_secondary'])
    despine(ax, "none")
    watermark(fig, mark)
    return fig


def figure_embedding_comparison(coords, *, model_order=None, color_rows=None,
                                metadata_style="points", mark=None):
    """Model columns with fixed coordinates and shared metadata colour rows.

    Metadata must agree for every patient across models. Missing scores remain
    visible in grey; they are never assigned a clinical risk group.
    """
    from opera.visualization.style import CATEGORICAL, sequential_cmap, setup_style
    setup_style()
    if metadata_style not in {"points", "smooth"}:
        raise ValueError("metadata_style must be points or smooth")
    rows = color_rows or [
        {"column": "disease", "label": "Disease group", "kind": "categorical"},
        {"column": "age", "label": "Age at treatment (years)", "kind": "continuous"},
        {"column": "sequence_length", "label": "Recorded events before treatment",
         "kind": "continuous", "log": True},
        {"column": "ipi_group", "label": "IPI category", "kind": "categorical",
         "order": ["Low", "Low-intermediate", "Intermediate", "High-intermediate",
                   "High", "Very high"]},
    ]
    required = {"model", "patient_id", "x", "y"} | {r["column"] for r in rows}
    if not required.issubset(coords.columns):
        raise ValueError(f"Missing coordinate columns: {sorted(required - set(coords.columns))}")
    if coords[["model", "patient_id", "x", "y"]].isna().any().any():
        raise ValueError("Coordinates and identifiers must be complete")
    if coords.duplicated(["model", "patient_id"]).any():
        raise ValueError("One coordinate per patient and model is required")
    if not np.isfinite(coords[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("Coordinates must be finite")
    models = list(model_order or coords.model.drop_duplicates())
    if len(models) != len(set(models)) or set(models) != set(coords.model):
        raise ValueError("Model order must include each model exactly once")
    patients = set(coords.loc[coords.model == models[0], "patient_id"])
    for model in models:
        if set(coords.loc[coords.model == model, "patient_id"]) != patients:
            raise ValueError("Model panels must contain identical patients")
    for row in rows:
        if (coords.groupby("patient_id")[row["column"]].nunique(dropna=False) != 1).any():
            raise ValueError("Patient metadata must agree across models")
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable
    size = 1.95
    fig = plt.figure(figsize=(size * len(models) + 1.55, size * len(rows) + 0.75))
    grid = fig.add_gridspec(len(rows), len(models) + 1,
                            width_ratios=[1] * len(models) + [0.62],
                            left=0.035, right=0.985, bottom=0.025,
                            top=1 - 0.55 / fig.get_figheight(), hspace=0.16, wspace=0.06)
    base = coords.loc[coords.model == models[0]]
    for ri, row in enumerate(rows):
        column = row["column"]
        categorical = row["kind"] == "categorical"
        ordinal = row["kind"] == "ordinal"
        legend_ax = fig.add_subplot(grid[ri, -1])
        legend_ax.axis("off")
        if ordinal:
            present = set(base[column].dropna())
            levels = [c for c in row["order"] if c in present]
            if present - set(levels):
                raise ValueError(f"Unmapped categories in {column}")
            rank = {c: i for i, c in enumerate(levels)}
            cmap = LinearSegmentedColormap.from_list("risk", _risk_colors(len(levels)))
            cmap.set_bad((1, 1, 1, 0))
            norm = Normalize(0, len(levels) - 1)
            cax = legend_ax.inset_axes([0.08, 0.22, 0.1, 0.56])
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                               ticks=range(len(levels)))
            bar.ax.set_yticklabels(levels, fontsize=6.5)
            bar.outline.set_visible(False)
            cax.set_title(textwrap.fill(row["label"] + (" (mean)" if metadata_style == "smooth" else ""), 14), fontsize=7.5,
                          loc="left", pad=6)
        elif categorical:
            present = set(base[column].dropna())
            categories = ([c for c in row["order"] if c in present]
                          + sorted(present - set(row["order"]), key=str)
                          if "order" in row else sorted(present, key=str))
            palette = (CATEGORICAL + ["#BD648E", "#697481"]) if column == "disease" else (
                _risk_colors(len(categories)) if "order" in row else CATEGORICAL)
            if column == 'ipi_group':
                clinical_colors = {'Low': '#3E8A87', 'Low-intermediate': '#B18B4D',
                    'Intermediate': '#B18B4D', 'High-intermediate': '#C16A58',
                    'High': '#8E2F3C', 'Very high': '#58162B'}
                palette = [clinical_colors.get(c, palette[i]) for i, c in enumerate(categories)]
            if len(categories) > len(palette):
                raise ValueError("Too many categories for the fixed figure palette")
            colors = {c: palette[i] for i, c in enumerate(categories)}
            handles = [Line2D([], [], marker="o", linestyle="", color=colors[c],
                              label=str(c), markersize=4.5) for c in categories]
            if base[column].isna().any():
                handles.append(Line2D([], [], marker="o", linestyle="",
                                      color=PALETTE["missing"], label="Not recorded",
                                      markersize=4.5))
            legend_ax.legend(handles=handles, loc="center left", frameon=False,
                             fontsize=7, handletextpad=0.3, labelspacing=0.35,
                             title=row["label"], title_fontsize=7.5,
                             alignment="left")
        else:
            values = pd.to_numeric(base[column], errors="raise").dropna()
            if values.empty or not np.isfinite(values).all():
                raise ValueError(f"No finite metadata for {column}")
            lo, hi = float(np.quantile(values, 0.02)), float(np.quantile(values, 0.98))
            if row.get("log") and lo <= 0:
                raise ValueError("Log colour scales require positive values")
            if lo == hi:
                hi = lo + 1
            norm = LogNorm(lo, hi) if row.get("log") else Normalize(lo, hi)
            cmap = sequential_cmap().copy()
            cmap.set_bad((1, 1, 1, 0))
            cax = legend_ax.inset_axes([0.08, 0.22, 0.1, 0.56])
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            if row.get("log"):
                ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
                         if lo <= t <= hi]
                bar.set_ticks(ticks, labels=[f"{t:,}" for t in ticks])
                bar.ax.minorticks_off()
            bar.ax.tick_params(labelsize=6.5)
            bar.outline.set_visible(False)
            cax.set_title(textwrap.fill(row["label"], 14), fontsize=7.5, loc="left", pad=6)
        for mi, model in enumerate(models):
            ax = fig.add_subplot(grid[ri, mi])
            data = coords.loc[coords.model == model].sort_values("patient_id")
            pad_x = 0.04 * (data.x.max() - data.x.min())
            pad_y = 0.04 * (data.y.max() - data.y.min())
            extent = (data.x.min() - pad_x, data.x.max() + pad_x,
                      data.y.min() - pad_y, data.y.max() + pad_y)
            ax.scatter(data.x, data.y, s=1.5, color=PALETTE["grid"], alpha=0.6,
                       linewidths=0, rasterized=True, zorder=1)
            if categorical:
                _category_regions(ax, data, column, colors, extent, contours=column == "disease")
            elif metadata_style == "points":
                values = (data[column].map(rank) if ordinal
                          else pd.to_numeric(data[column], errors="raise"))
                # Every colour is one observed patient's metadata, not a mean
                # interpolated into empty regions of a two-dimensional map.
                ax.scatter(data.x, data.y, c=values.to_numpy(float), cmap=cmap,
                           norm=norm, s=5, alpha=0.8, linewidths=0,
                           rasterized=True, zorder=2)
            else:
                # Smooth local mean (Gaussian kernel), as in the risk maps; blank
                # where too few patients carry the value.
                from opera.visualization.censoring_aware_embeddings import (
                    _default_bandwidth, _kernel_surfaces)
                points = data[["x", "y"]].to_numpy(float)
                valid = data.loc[data[column].notna()]
                values = (valid[column].map(rank).to_numpy(dtype=float) if ordinal
                          else valid[column].to_numpy(dtype=float))
                if row.get("log"):
                    values = np.log(values)
                bandwidth = 1.4 * _default_bandwidth(points)
                xx, yy, surface, support, density = _kernel_surfaces(
                    valid[["x", "y"]].to_numpy(float), values, np.ones(len(values)),
                    bandwidth=bandwidth, grid_size=180, bounds_coords=points,
                    min_padding=0.0)
                if row.get("log"):
                    surface = np.exp(surface)
                # Blank where the map has too few patients nearby.
                surface[_embedding_surface_mask(
                    xx, yy, valid[["x", "y"]].to_numpy(float), support, density, 25)] = np.nan
                ax.imshow(surface, origin="lower", cmap=cmap, norm=norm,
                          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                          interpolation="bilinear", aspect="auto", alpha=0.92,
                          zorder=2, rasterized=True)
                ax.scatter(data.x, data.y, s=0.8, color="white", alpha=0.25,
                           linewidths=0, rasterized=True, zorder=3)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_aspect("auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ri == 0:
                title = str(model).replace(" (tuning-selected)", "")
                ax.set_title(f"{chr(65 + mi)}  " + textwrap.fill(title, 20), loc="left",
                             fontsize=9, pad=5)
    watermark(fig, mark)
    return fig


def _category_regions(ax, data, column, colors, extent, *, label=False, mass=0.5,
                      contours=True):
    """Patient points, optionally with contours around each category's dense half."""
    from scipy.stats import gaussian_kde

    xx, yy = np.meshgrid(np.linspace(*extent[:2], 90), np.linspace(*extent[2:], 90))
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    for category, color in colors.items():
        part = data.loc[data[column] == category]
        ax.scatter(part.x, part.y, s=4.0, color=color, alpha=0.55, linewidths=0,
                   rasterized=True, zorder=2)
        if not contours or len(part) < 15:
            continue
        points = np.vstack([part.x, part.y])
        kde = gaussian_kde(points)
        level = np.quantile(kde(points), 1 - mass)
        density = kde(grid_points).reshape(xx.shape)
        ax.contour(xx, yy, density, levels=[level], colors=[color], linewidths=1.1,
                   zorder=3)
        if label:
            peak = grid_points[:, density.argmax()]
            ax.text(peak[0], peak[1], textwrap.fill(str(category), 12), fontsize=5.8,
                    color=color, ha="center", va="center", fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.75))



def figure_clinical_score_age(patients, *, outcome_label, horizon_days=730,
                             thresholds=(0.1, 0.4), ipi_label="IPI",
                             ipi_order=("Low", "Intermediate", "High"),
                             event_free=False, mark=None, seed=0, grouping_note=None,
                             age_bins=(0, 50, 60, 70, 80, np.inf)):
    """Separate counts, mean predictions and observed risk on identical cells."""
    from opera.visualization.style import sequential_cmap
    if event_free and (patients.event > 1).any():
        raise ValueError("Event-free survival requires death included in endpoint")
    names = list(ipi_order)
    if len(names) != len(thresholds) + 1:
        raise ValueError("Group/cutoff mismatch")
    data = patients.loc[patients.ipi_group.notna()].copy()
    data['model_group'] = np.asarray(names)[np.digitize(data.model_risk, thresholds)]
    ages = pd.cut(data.age, age_bins, right=False)
    labels = [f"<{int(b.right)}" if b.left == 0 else
              (f"≥{int(b.left)}" if not np.isfinite(b.right) else
               f"{int(b.left)}–{int(b.right)-1}") for b in ages.cat.categories]
    values = np.full((3, 2, len(labels), len(names)), np.nan)
    intervals = np.full((3, 2, len(labels), len(names), 2), np.nan)
    for col, key in enumerate(("model_group", "ipi_group")):
        for i, band in enumerate(ages.cat.categories):
            for j, group in enumerate(names):
                part = data.loc[(ages == band) & (data[key] == group)]
                values[0, col, i, j] = len(part)
                if len(part) >= 10:
                    prediction, p_low, p_high = _mean_with_ci(part.model_risk, seed=seed)
                    if event_free:
                        prediction, p_low, p_high = 1 - prediction, 1 - p_high, 1 - p_low
                    observed, low, high = _estimate_with_ci(part, horizon_days, event_free)
                    values[1, col, i, j] = prediction
                    values[2, col, i, j] = observed
                    intervals[1, col, i, j] = (p_low, p_high)
                    intervals[2, col, i, j] = (low, high)
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4),
                             gridspec_kw={"height_ratios": [0.55, 1, 1]})
    fig.subplots_adjust(left=.13, right=.86, bottom=.1, top=.92, hspace=.58, wspace=.16)
    row_titles = ["Patients", "Model-predicted event-free survival" if event_free else "Model-predicted risk",
                  "Observed event-free survival" if event_free else "Observed cumulative incidence"]
    for row in range(3):
        for col in range(2):
            ax = axes[row, col]
            limit = max(1, float(np.nanmax(values[0]))) if row == 0 else 1
            cmap = sequential_cmap("pretrain") if row == 0 else sequential_cmap()
            im = ax.imshow(np.ma.masked_invalid(values[row,col]), cmap=cmap,
                           vmin=0, vmax=limit, aspect="auto")
            for i in range(len(labels)):
                for j in range(len(names)):
                    v=values[row,col,i,j]
                    if row:
                        text = _estimate_text(v, *intervals[row, col, i, j])
                    else:
                        text = "–" if not np.isfinite(v) else _count_text(v)
                    ax.text(j,i,text,ha="center",va="center",fontsize=6.4 if row else 7,
                            linespacing=1.1,
                            color=PALETTE["ink"] if row==0 or not np.isfinite(v) or v>.6 else "white")
            ax.set_xticks(range(len(names)), _risk_tick_labels(names), fontsize=6.5)
            ax.set_yticks(range(len(labels)),labels,fontsize=7)
            if col:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("Age (years)")
            ax.set_title(chr(65+row*2+col)+"  "+textwrap.fill(row_titles[row], 29)+
                         "\n"+("Model groups" if col==0 else ipi_label),
                         loc="left", fontsize=8, pad=6)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        bar = fig.colorbar(im,ax=list(axes[row]),fraction=.025,pad=.04)
        bar.ax.tick_params(labelsize=7)
    fig.text(.13,.025,f"{outcome_label} · {horizon_days} days · 95% bootstrap intervals; "
             "estimates suppressed for n < 10",
             fontsize=7,color=PALETTE["ink_muted"])
    watermark(fig,mark)
    return fig


# Integrated-gradients attribution figures ----------------------------------
# Sources share the register colours of Figure S3 (model inputs).
SOURCE_COLORS = {
    # Identity of a source of records. Kept clear of the sienna-indigo pair that
    # the paper uses for signed effects (raises/lowers, gains/losses), so a
    # colour never means both "laboratory" and "raises risk". The patient
    # register is one purple family (diagnoses, procedures, contacts). Chosen so
    # every pair differs by CIE76 dE >= 18 for normal vision and simulated
    # deuteranopia, protanopia and tritanopia (Machado et al. 2009).
    "Demographics": "#2E3440", "Diagnoses": "#6A4C93", "Procedures": "#9886C2",
    "Hospital contacts": "#D4B9DA", "Pathology": "#8C6D4F", "Medication": "#3E9C8A",
    "Laboratory": "#C08A1E", "Other": "#DDE1E6",
}
RISK_CMAP = LinearSegmentedColormap.from_list(
    "opera_risk", ["#2D3A8C", "#AEB8DA", "#F7F7F5", "#E7B9A5", "#B5451B"])
RISK_CMAP.set_bad(PALETTE["missing"])
# The attribution figures live in attribution_overview; exposed here so figure
# recipes can name them like every other paper figure.
def figure_attribution_atlas(variance_sources, variance_windows, concepts, *, domains=None,
                             mark=None):
    from opera.visualization.attribution_overview import figure_attribution_atlas as render
    return render(variance_sources, variance_windows, concepts, domains=domains, mark=mark)


def figure_attribution_structure(similarity, restrictions=None, *, domains=None, mark=None):
    from opera.visualization.attribution_overview import figure_attribution_structure as render
    return render(similarity, restrictions, domains=domains, mark=mark)


def figure_attribution_main(variance_sources, concepts, restrictions, episodes, *,
                            domains=None, mark=None):
    from opera.visualization.attribution_overview import figure_attribution_main as render
    return render(variance_sources, concepts, restrictions, episodes, domains=domains,
                  mark=mark)


def figure_patient_tracks(episodes, *, outcomes=None, scale="logit", mark=None):
    from opera.visualization.attribution_overview import figure_patient_tracks as render
    return render(episodes, outcomes=outcomes, scale=scale, mark=mark)


def figure_recent_record_gaps(restrictions, *, horizons=None, mark=None):
    """Paired AUC changes by clinical family, at at most two horizons per page."""
    labels = [f'Last {days} days hidden' for days in (7, 14, 21, 28)]
    data = restrictions.loc[restrictions.restriction.isin(labels)].copy()
    if data.empty:
        raise ValueError('No recent-record gap results available')
    if 'horizon_days' not in data:
        data['horizon_days'] = 365
    finite_limits = 100 * data[['lower', 'upper', 'mean_delta']].to_numpy(float).ravel()
    finite_limits = finite_limits[np.isfinite(finite_limits)]
    limits = (min(0., finite_limits.min()), max(0., finite_limits.max())) if len(finite_limits) else (-1., 1.)
    margin = max((limits[1] - limits[0]) * .06, .1)
    horizons = list(horizons) if horizons is not None else ([365] if 365 in set(data.horizon_days) else sorted(data.horizon_days.unique())[:2])
    if not 1 <= len(horizons) <= 2 or set(horizons) - set(data.horizon_days):
        raise ValueError('Choose one or two available horizons per family page')
    data = data.loc[data.horizon_days.isin(horizons)]
    if data.duplicated(['restriction', 'group', 'horizon_days']).any():
        raise ValueError('Duplicate recent-record family/horizon estimates')
    groups = list(dict.fromkeys(data.group))
    order = ['All outcomes', *SHORT_GROUP]
    groups = [g for g in order if g in groups] + [g for g in groups if g not in order]
    fig, axes = plt.subplots(1, len(horizons), squeeze=False, sharey=True,
                             figsize=(9.2 if len(horizons) == 2 else 8.8, max(4.5, len(groups) * .46 + 1.6)))
    fig.subplots_adjust(left=.25, right=.97, top=.84, bottom=.14, wspace=.15)
    for panel, (ax, horizon) in enumerate(zip(axes[0], horizons)):
        for i, label in enumerate(labels):
            part = data.loc[data.restriction.eq(label) & data.horizon_days.eq(horizon)].set_index('group').reindex(groups)
            y = np.arange(len(groups)) + (i - 1.5) * .16
            color = list(HORIZON_COLORS.values())[i]
            ax.hlines(y, 100 * part.lower, 100 * part.upper, color=color, alpha=.65, lw=1.3)
            ax.scatter(100 * part.mean_delta, y, s=22, marker='D', color=color, label=label)
        ax.axvline(0, color=PALETTE['zero_line'], linewidth=.8)
        ax.set(yticks=np.arange(len(groups)), yticklabels=[SHORT_GROUP.get(g, g) for g in groups],
               xlabel='AUC change (×100; paired 95% CI)')
        ax.tick_params(labelsize=8.5)
        ax.set_xlim(limits[0] - margin, limits[1] + margin)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        label = f'{horizon:g} days' if horizon < 365 else f'{horizon / 365:g} year' + ('s' if horizon > 365 else '')
        _panel_title(ax, chr(65 + panel), label)
        despine(ax, 'x')
    axes[0, 0].invert_yaxis()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.6,.99),
               ncol=2, fontsize=8, frameon=False)
    fig.text(.60, .04, 'Equal outcome means within clinical families; diseases pooled within each outcome',
             ha='center', fontsize=7, color=PALETTE['ink_secondary'])
    watermark(fig, mark)
    return fig


def figure_recent_record_gap_horizons(restrictions, *, mark=None):
    """Overall recent-history reliance across prediction horizons."""
    data = restrictions.loc[restrictions.group.eq('All outcomes')].copy()
    if 'horizon_days' not in data:
        data['horizon_days'] = 365
    if data.empty or data.duplicated(['restriction', 'horizon_days']).any():
        raise ValueError('Need unique overall horizon/restriction results')
    horizons = sorted(data.horizon_days.unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=.13, right=.97, bottom=.20, top=.73)
    for i, days in enumerate((7, 14, 21, 28)):
        label = f'Last {days} days hidden'
        part = data.loc[data.restriction.eq(label)].set_index('horizon_days').reindex(horizons)
        x = np.arange(len(horizons)) + (i - 1.5) * .045
        color = list(HORIZON_COLORS.values())[i]
        ax.plot(x, 100 * part.mean_delta, color=color, lw=1.2, marker='D', ms=4, label=label)
        ax.vlines(x, 100 * part.lower, 100 * part.upper, color=color, lw=1, alpha=.6)
    ax.axhline(0, color=PALETTE['zero_line'], lw=.8)
    ax.set_xticks(range(len(horizons)), [f'{int(h)} d' if h < 365 else f'{h / 365:g} y' for h in horizons])
    ax.set_xlabel('Prediction horizon')
    ax.set_ylabel('AUC change (×100; paired 95% CI)')
    _panel_title(ax, 'A', 'Reliance on recent clinical history')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.55,.98),
               ncol=2, frameon=False, fontsize=8)
    despine(ax, 'both')
    watermark(fig, mark)
    return fig
