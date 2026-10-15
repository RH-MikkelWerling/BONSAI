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
from matplotlib.ticker import MaxNLocator

from opera.visualization.style import PALETTE, despine

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
    "J": ("#2D3A8C", "Joint"),
    "R": ("#8974AC", "Routed"),
    "X": ("#408579", "XGBoost"),
    "LR": ("#9E9E9E", "Logistic regression"),
}
ORIGIN_STYLE = {
    "L": ("#737D8C", "Local"),
    "E": ("#5C7AEA", "National"),
    "EA": ("#2D3A8C", "National + local"),
}
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "opera_gain", ["#B5451B", "#E7B9A5", "#F7F7F5", "#AEB8DA", "#2D3A8C"]
)
GAIN_CMAP.set_bad(PALETTE["missing"])
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
}


def watermark(fig, text="SYNTHETIC DESIGN PROOF"):
    if text:
        fig.text(0.995, 0.002, text, ha="right", va="bottom", fontsize=6.5,
                 color=PALETTE["ink_muted"], alpha=0.8)


def _panel_title(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=10.5)


def _ordered(cells, key, group_key):
    frame = cells[[key, group_key]].drop_duplicates(key)
    return frame.sort_values([group_key, key])[key].tolist(), frame.set_index(key)[group_key]


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
    """Median and IQR of cell gains per group (atlas-style spread, not CI)."""
    rows = []
    for key, group in frame.groupby(by, sort=False):
        values = group.gain.dropna().to_numpy()
        if not len(values):
            continue
        q25, q75 = (np.quantile(values, [0.25, 0.75]) if len(values) > 1
                    else (values[0], values[0]))
        rows.append({**dict(zip(by if isinstance(by, list) else [by],
                                key if isinstance(key, tuple) else (key,))),
                     "median": float(np.median(values)), "q25": q25, "q75": q75,
                     "n": len(values)})
    return pd.DataFrame(rows)


def _diamond_forest(ax, summary, *, row_key, row_order, series_key, series_style,
                    xlabel, low="q25", high="q75", show_counts=True, zero_line=True,
                    spread=0.27):
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
    legend = ax.legend(handles=handles, loc=loc, title=title, ncol=ncol, frameon=True,
                       facecolor="white", edgecolor=PALETTE["grid"], framealpha=0.97,
                       fontsize=7, title_fontsize=7.2, **kwargs)
    return legend


def gain_matrix(ax, cells, *, contrast, metric="auc", horizon=365, vmax=0.06,
                show_ylabels=True, show_xlabels=False):
    """Disease (rows) by outcome (columns) map of one paired contrast."""
    data = cells.loc[(cells.contrast == contrast) & (cells.metric == metric)
                     & (cells.horizon_days == horizon)]
    diseases, disease_group = _ordered(cells, "cohort", "cohort_group")
    outcomes, outcome_group = _ordered(cells, "outcome", "endpoint_group")
    matrix = (data.pivot_table(index="cohort", columns="outcome", values="gain",
                               aggfunc="first")
              .reindex(index=diseases, columns=outcomes))
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), cmap=GAIN_CMAP,
                      norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto",
                      interpolation="nearest")
    for position in _group_breaks(outcomes, outcome_group)[0]:
        ax.axvline(position, color="white", lw=1.2)
    for position in _group_breaks(diseases, disease_group)[0]:
        ax.axhline(position, color="white", lw=1.2)
    ax.set_yticks(range(len(diseases)), diseases if show_ylabels else [])
    ax.tick_params(axis="y", labelsize=5.6, length=0)
    if show_xlabels:
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


def group_matrix(ax, cells, *, contrast, order, metric="auc", horizon=365, vmax=0.04,
                 show_ylabels=True, show_xlabels=True):
    """Diseases (rows) by endpoint groups (columns): median cell gain per tile."""
    data = cells.loc[(cells.contrast == contrast) & (cells.metric == metric)
                     & (cells.horizon_days == horizon)]
    diseases, disease_group = _ordered(cells, "cohort", "cohort_group")
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


def figure_pooling_cells(cells, *, metric="auc", horizon=365, mark=None):
    """Supplement: the full disease-by-outcome maps behind the pooling figure."""
    fig = plt.figure(figsize=(7.2, 6.4))
    top = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.05, left=0.13, right=0.9,
                           top=0.95, bottom=0.13)
    panels = [("D-S", "A", "Disease-pooled vs specialist"),
              ("O-S", "B", "Outcome-pooled vs specialist"),
              ("J-S", "C", "Joint vs specialist"),
              ("interaction", "D", "Interaction (both pooling axes)")]
    image = None
    for index, (contrast, letter, title) in enumerate(panels):
        ax = fig.add_subplot(top[index // 2, index % 2])
        image = gain_matrix(ax, cells, contrast=contrast, metric=metric,
                            horizon=horizon, show_ylabels=index % 2 == 0,
                            show_xlabels=index >= 2)
        _panel_title(ax, letter, title)
    cax = fig.add_axes([0.915, 0.58, 0.012, 0.3])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label(f"One-year {METRIC_LABEL[metric]}", fontsize=7.5)
    bar.ax.tick_params(labelsize=6.5)
    bar.outline.set_visible(False)
    watermark(fig, mark)
    return fig


def figure_pooling(cells, curve=None, *, metric="auc", horizon=365, mark=None):
    """Q1: what learning across diseases and outcomes adds (main figure).

    Tiles are the median one-year gain across the outcomes of an endpoint group
    within a fine disease; the cell-level maps are ``figure_pooling_cells``.
    """
    fig = plt.figure(figsize=(7.2, 8.3))
    top = fig.add_gridspec(2, 2, hspace=0.17, wspace=0.04, left=0.135, right=0.9,
                           top=0.965, bottom=0.43)
    panels = [("D-S", "A", "Disease-pooled vs specialist"),
              ("O-S", "B", "Outcome-pooled vs specialist"),
              ("J-S", "C", "Joint vs specialist"),
              ("interaction", "D", "Interaction (both pooling axes)")]
    joint = cells.loc[(cells.contrast == "J-S") & (cells.metric == metric)
                      & (cells.horizon_days == horizon)]
    order = joint.groupby("endpoint_group").gain.median().sort_values(
        ascending=False).index.tolist()
    image = None
    for index, (contrast, letter, title) in enumerate(panels):
        ax = fig.add_subplot(top[index // 2, index % 2])
        image = group_matrix(ax, cells, contrast=contrast, order=order, metric=metric,
                             horizon=horizon, show_ylabels=index % 2 == 0,
                             show_xlabels=index >= 2)
        _panel_title(ax, letter, title)
    cax = fig.add_axes([0.915, 0.66, 0.012, 0.27])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(f"One-year {METRIC_LABEL[metric]}\n(median within group)",
                       fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.5)
    colorbar.outline.set_visible(False)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(5))

    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62, left=0.1,
                              right=0.98, top=0.295, bottom=0.075)
    ax = fig.add_subplot(bottom[0])
    data = cells.loc[(cells.metric == metric) & (cells.horizon_days == horizon)
                     & cells.contrast.isin(["D-S", "O-S", "J-S"])]
    joint = data.loc[(data.contrast == "J-S") & data.gain.notna()]
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
    _panel_title(ax, "E", "Gain and training support")
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
    _panel_title(ax, "F", "By endpoint group")
    watermark(fig, mark)
    return fig


def figure_tabular(cells, curve=None, *, horizon=365, mark=None):
    """Q2: neural models against task-specific gradient-boosted trees."""
    fig = plt.figure(figsize=(7.2, 5.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.12,
                            left=0.19, right=0.98, top=0.88, bottom=0.1)
    styles = {"S-X": (MODEL_STYLE["S"][0], "Specialist vs XGBoost"),
              "J-X": (MODEL_STYLE["J"][0], "Joint vs XGBoost"),
              "R-X": (MODEL_STYLE["R"][0], "Routed vs XGBoost"),
              "J-XP": ("#B18B4D", "Joint vs disease-pooled XGBoost")}
    data = cells.loc[(cells.horizon_days == horizon) & cells.contrast.isin(styles)]
    order = (data.loc[(data.contrast == "J-X") & (data.metric == "auc")]
             .groupby("endpoint_group").gain.median().sort_values(ascending=False)
             .index.tolist())
    handles = None
    for column, (metric, letter) in enumerate((("auc", "A"), ("brier", "B"))):
        ax = fig.add_subplot(grid[0, column])
        summary = _summary_rows(data.loc[data.metric == metric],
                                ["endpoint_group", "contrast"])
        handles = _diamond_forest(ax, summary, row_key="endpoint_group",
                                  row_order=order, series_key="contrast",
                                  series_style=styles,
                                  xlabel=f"{METRIC_LABEL[metric]} (median, IQR)",
                                  show_counts=False)
        if column:
            ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        _panel_title(ax, letter, "Discrimination" if metric == "auc" else "Accuracy")
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.18, 0.995))

    lower = grid[1, :].subgridspec(1, 2, wspace=0.42)
    ax = fig.add_subplot(lower[0])
    part = cells.loc[(cells.contrast == "J-X") & (cells.metric == "auc")
                     & (cells.horizon_days == horizon) & cells.gain.notna()]
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
    _panel_title(ax, "C", "Where the joint model gains")
    despine(ax, "y")

    ax = fig.add_subplot(lower[1])
    part = cells.loc[(cells.contrast.isin(["J-X", "X-LR"])) & (cells.metric == "auc")
                     & (cells.horizon_days == horizon)]
    horizons_summary = cells.loc[cells.contrast.isin(["J-X", "X-LR", "XP-X"])
                                 & (cells.metric == "auc")]
    for contrast, color, label in (("J-X", MODEL_STYLE["J"][0], "Joint vs XGBoost"),
                                   ("XP-X", "#B18B4D", "Pooled vs per-disease XGBoost"),
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
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]), top=ax.get_ylim()[1] * 1.9)
    _legend(ax, ax.get_legend_handles_labels()[0], loc="upper left", ncol=1)
    despine(ax, "y")
    del part
    watermark(fig, mark)
    return fig


def figure_external(estimates, cells, panel=None, *, horizon=365, metric="auc", mark=None):
    """Q3: nationwide pretraining and local adaptation under fixed readouts.

    ``panel`` holds one panel-level row per paired contrast (B); ``cells`` the
    cell-level adapted-national effects (C). Without ``panel``, B reads the
    first matching row of ``cells``, as in the synthetic proofs.
    """
    panel_rows = cells if panel is None else panel
    fig = plt.figure(figsize=(7.2, 3.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.0, 1.15], wspace=0.85,
                            left=0.09, right=0.98, top=0.87, bottom=0.16)
    ax = fig.add_subplot(grid[0])
    panel = estimates.loc[(estimates.metric == metric)
                          & (estimates.horizon_days == horizon)]
    for x, origin in enumerate(("L", "E", "EA")):
        for y, readout in enumerate(("J", "R")):
            row = panel.loc[(panel.origin == origin) & (panel.readout == readout)]
            if row.empty:
                continue
            row = row.iloc[0]
            color = MODEL_STYLE[readout][0]
            offset = -0.12 if readout == "J" else 0.12
            ax.plot([x + offset] * 2, [row.lower, row.upper], color=color, lw=1.3,
                    alpha=0.5)
            ax.scatter(x + offset, row.estimate, marker="D" if readout == "J" else "o",
                       s=26, color=color, zorder=3)
    ax.set_xticks(range(3), ["Local", "National", "National\n+ local"], fontsize=7.2)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Panel one-year AUC")
    _panel_title(ax, "A", "Encoder origin")
    despine(ax, "y")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _legend(ax, [Line2D([], [], ls="none", marker="D", color=MODEL_STYLE["J"][0],
                        label="Joint"),
                 Line2D([], [], ls="none", marker="o", color=MODEL_STYLE["R"][0],
                        label="Routed")], loc="lower right")

    ax = fig.add_subplot(grid[1])
    contrasts = [("E-L", "National\nvs local"), ("EA-E", "Local\nadaptation"),
                 ("EA-L", "Adapted national\nvs local")]
    for y, (contrast, label) in enumerate(contrasts):
        for readout, offset, marker in (("J", -0.14, "D"), ("R", 0.14, "o")):
            row = panel_rows.loc[(panel_rows.contrast == f"{contrast}:{readout}")
                                 & (panel_rows.metric == metric)
                                 & (panel_rows.horizon_days == horizon)]
            if row.empty:
                continue
            row = row.iloc[0]
            color = MODEL_STYLE[readout][0]
            ax.plot([row.lower, row.upper], [y + offset] * 2, color=color, lw=1.3,
                    alpha=0.5)
            ax.scatter(row.gain, y + offset, marker=marker, s=26, color=color, zorder=3)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_yticks(range(3), [label for _, label in contrasts], fontsize=7)
    ax.set_ylim(-0.6, 2.6)
    ax.invert_yaxis()
    ax.set_xlabel(f"Panel {METRIC_LABEL[metric]} (95% CI)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "B", "Paired contrasts")
    despine(ax, "x")

    ax = fig.add_subplot(grid[2])
    data = cells.loc[(cells.contrast == "EA-L:J") & (cells.metric == metric)
                     & (cells.horizon_days == horizon)]
    order = data.groupby("cohort_group").gain.median().sort_values(ascending=False).index
    summary = _summary_rows(data.assign(series="EA-L"), ["cohort_group", "series"])
    _diamond_forest(ax, summary, row_key="cohort_group", row_order=list(order),
                    series_key="series",
                    series_style={"EA-L": (ORIGIN_STYLE["EA"][0], "")},
                    xlabel=f"{METRIC_LABEL[metric]} (median, IQR)", show_counts=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "C", "By disease group")
    watermark(fig, mark)
    return fig


def figure_routing(cells, gate, *, metric="auc", horizon=365, tolerance=0.005, mark=None):
    """Supplement: routed readout and adapter-bank factorial."""
    fig = plt.figure(figsize=(7.2, 3.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.9, left=0.21,
                            right=0.98, top=0.86, bottom=0.17)
    ax = fig.add_subplot(grid[0])
    data = cells.loc[(cells.contrast == "R1") & (cells.metric == metric)
                     & (cells.horizon_days == horizon)]
    order = data.groupby("endpoint_group").gain.median().sort_values(ascending=False)
    summary = _summary_rows(data.assign(series="R1"), ["endpoint_group", "series"])
    _diamond_forest(ax, summary, row_key="endpoint_group", row_order=list(order.index),
                    series_key="series", series_style={"R1": (MODEL_STYLE["R"][0], "")},
                    xlabel=f"{METRIC_LABEL[metric]}, routed vs joint (median, IQR)",
                    show_counts=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "A", "Routed vs joint")

    ax = fig.add_subplot(grid[1])
    labels = {"R0": "None vs joint (gate)", "R2": "Disease bank", "R3": "Outcome bank",
              "R4": "Outcome given disease", "R5": "Disease given outcome",
              "R6": "Both banks", "R7": "Bank interaction"}
    ax.axvspan(-tolerance, tolerance, color=PALETTE["grid"], zorder=0)
    for y, contrast in enumerate(labels):
        row = gate.loc[gate.contrast == contrast]
        if row.empty:
            continue
        row = row.iloc[0]
        color = PALETTE["ink_secondary"] if contrast == "R0" else MODEL_STYLE["R"][0]
        ax.plot([row.lower, row.upper], [y, y], color=color, lw=1.3, alpha=0.55)
        ax.scatter(row.gain, y, marker="D", s=24, color=color, zorder=3)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.set_yticks(range(len(labels)), list(labels.values()), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Panel {METRIC_LABEL[metric]} (95% CI)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.text(tolerance, len(labels) - 0.45, "equivalence\nmargin", fontsize=6.5,
            color=PALETTE["ink_muted"], ha="right", va="bottom")
    _panel_title(ax, "B", "Adapter banks")
    despine(ax, "x")
    watermark(fig, mark)
    return fig


def figure_horizons(cells, *, metric="auc", contrasts=("D-S", "O-S", "J-S"), mark=None):
    """Supplement: stability of pooling effects over follow-up."""
    fig, axes = plt.subplots(1, len(contrasts), figsize=(7.2, 3.8), sharey=True,
                             sharex=True)
    data = cells.loc[cells.metric == metric]
    order = (data.loc[(data.contrast == "J-S") & (data.horizon_days == 365)]
             .groupby("endpoint_group").gain.median().sort_values(ascending=False)
             .index.tolist())
    for index, (ax, contrast) in enumerate(zip(axes, contrasts)):
        summary = _summary_rows(data.loc[data.contrast == contrast],
                                ["endpoint_group", "horizon_days"])
        handles = _diamond_forest(
            ax, summary, row_key="endpoint_group", row_order=order,
            series_key="horizon_days",
            series_style={h: (c, f"{h} days") for h, c in HORIZON_COLORS.items()},
            xlabel=METRIC_LABEL[metric], show_counts=False)
        if index:
            ax.tick_params(axis="y", labelleft=False)
        _panel_title(ax, "ABC"[index], CONTRAST_STYLE[contrast][1].replace(" vs ", "\nvs "))
        ax.xaxis.set_major_locator(MaxNLocator(3))
    fig.legend(handles=handles, loc="lower left", ncol=6, frameon=False, fontsize=7,
               bbox_to_anchor=(0.2, 0.0))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.86, bottom=0.22, wspace=0.14)
    watermark(fig, mark)
    return fig


def figure_atlas(atlas, *, mark=None):
    """Predictability atlas: absolute held-out performance of the joint model."""
    fig, (ax, group_ax) = plt.subplots(1, 2, figsize=(7.2, 4.4),
                                       gridspec_kw={"width_ratios": [1.0, 1.15]})
    horizons = sorted(atlas.horizon_days.unique())
    for horizon in horizons:
        part = atlas.loc[(atlas.horizon_days == horizon) & atlas.cif_auc.notna()]
        ax.scatter(part.n_events_train.clip(lower=1), part.cif_auc, s=8, alpha=0.5,
                   color=HORIZON_COLORS[horizon], linewidths=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Training events through horizon (log scale)")
    ax.set_ylabel("Held-out AUC")
    ax.set_ylim(max(0.4, atlas.cif_auc.min() - 0.03), 1.0)
    _panel_title(ax, "A", "Outcomes")
    despine(ax, "y")
    summary = _summary_rows(atlas.rename(columns={"cif_auc": "gain"}),
                            ["endpoint_group", "horizon_days"])
    order = (summary.loc[summary.horizon_days == 365].sort_values("median",
             ascending=False).endpoint_group.tolist())
    handles = _diamond_forest(
        group_ax, summary, row_key="endpoint_group", row_order=order,
        series_key="horizon_days",
        series_style={h: (c, f"{h} days") for h, c in HORIZON_COLORS.items()},
        xlabel="Held-out AUC (median, IQR)", show_counts=True, zero_line=False)
    group_ax.set_xlim(0.5, 1.0)
    _panel_title(group_ax, "B", "Endpoint groups")
    _legend(group_ax, handles, loc="lower right", title="Horizon")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=1.0)
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


def figure_calibration_maps(coords, outcomes, *, horizon=365, bandwidth=None,
                            grid_size=140, min_effective_support=15.0,
                            residual_limit=0.2, router_bins=9, router_min_count=20,
                            router_min_coherence=0.3, mark=None):
    """Status, router movement, IPCW kernel risk and local calibration per outcome.

    ``outcomes`` maps a display name to ``(times, events, predicted_cif)`` or
    ``(times, events, predicted_cif, routed_coords)``, aligned with ``coords``.
    Every column uses the joint model's single patient embedding, so columns
    differ only in the outcome. ``routed_coords`` is the same projection applied
    to the routed, target-specific representation; when any outcome supplies it,
    a row of local mean displacement arrows is added. Regions with too little
    effective support are left blank rather than extrapolated.
    """
    from opera.visualization.censoring_aware_embeddings import (
        _binned_router_displacement, _default_bandwidth, _display_grid,
        _kernel_surfaces, horizon_ipcw_data, spatial_calibration_residual,
    )

    coords = np.asarray(coords, float)
    bandwidth = bandwidth or _default_bandwidth(coords)
    xx, yy, _ = _display_grid(coords, grid_size, min_padding=0.8)
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    names = list(outcomes)
    has_router = any(len(value) > 3 and value[3] is not None
                     for value in outcomes.values())
    rows = ["status", *(["router"] if has_router else []), "risk", "residual"]
    titles = {"status": "Status at horizon", "router": "Router movement",
              "risk": "IPCW kernel risk", "residual": "Calibration residual"}
    fig = plt.figure(figsize=(7.2, 1.62 * len(rows) + 0.55))
    grid = fig.add_gridspec(len(rows), len(names) + 1,
                            width_ratios=[1] * len(names) + [0.06],
                            wspace=0.04, hspace=0.08, left=0.09, right=0.91,
                            top=0.93, bottom=0.03)
    risk_cmap = plt.get_cmap("plasma").copy()
    risk_cmap.set_bad((1, 1, 1, 0))
    residual_cmap = plt.get_cmap("RdBu_r").copy()
    residual_cmap.set_bad((1, 1, 1, 0))
    images = {}
    span = float(np.ptp(coords, axis=0).max())

    def backdrop(ax, alpha=0.08, zorder=1, color=PALETTE["ink_secondary"]):
        ax.scatter(coords[:, 0], coords[:, 1], s=1.5, alpha=alpha, linewidths=0,
                   color=color, rasterized=True, zorder=zorder)

    for column, name in enumerate(names):
        times, events, predicted, *rest = outcomes[name]
        routed = rest[0] if rest else None
        data = horizon_ipcw_data(times, events, horizon)
        used = data.eligible
        _, _, surface, support, _ = _kernel_surfaces(
            coords[used], data.target[used], data.weights[used], bandwidth=bandwidth,
            grid_size=grid_size, bounds_coords=coords, min_padding=0.8)
        blank = support < min_effective_support
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
                    if len(vectors):
                        length = np.linalg.norm(vectors, axis=1)
                        scale = 0.06 * span / max(float(np.median(length[length > 0])),
                                                  1e-12)
                        ax.quiver(arrows["centers"][keep, 0], arrows["centers"][keep, 1],
                                  vectors[:, 0] * scale, vectors[:, 1] * scale,
                                  color="#D55E00", angles="xy", scale_units="xy",
                                  scale=1, width=0.006, headwidth=3.6, zorder=3)
                else:
                    ax.text(0.5, 0.5, "shared readout", transform=ax.transAxes,
                            ha="center", va="center", fontsize=6.5,
                            color=PALETTE["ink_muted"])
            elif kind == "risk":
                images["risk"] = ax.imshow(
                    np.ma.masked_where(blank, surface), origin="lower", extent=extent,
                    cmap=risk_cmap, norm=plt.Normalize(0, 1), interpolation="bilinear",
                    zorder=2)
                backdrop(ax, 0.18, zorder=3, color="white")
            else:
                images["residual"] = ax.imshow(
                    np.ma.masked_where(blank | ~np.isfinite(residual), residual),
                    origin="lower", extent=extent, cmap=residual_cmap,
                    norm=TwoSlopeNorm(0, -residual_limit, residual_limit),
                    interpolation="bilinear", zorder=2)
            if row == 0:
                ax.set_title(name, fontsize=8.2, loc="center", pad=4)
            if column == 0:
                ax.set_ylabel(titles[kind], fontsize=7.6)
                ax.text(-0.14, 1.0, "ABCD"[row], transform=ax.transAxes, fontsize=10.5,
                        va="bottom", color=PALETTE["ink"])
    for row, kind in enumerate(rows):
        cax = fig.add_subplot(grid[row, -1])
        if kind in images:
            bar = fig.colorbar(images[kind], cax=cax)
            bar.set_label("Local cumulative\nincidence" if kind == "risk"
                          else "Observed −\npredicted", fontsize=7)
            bar.ax.tick_params(labelsize=6.5)
            bar.outline.set_visible(False)
            bar.ax.yaxis.set_major_locator(MaxNLocator(5))
        else:
            cax.set_axis_off()
    handles = [Line2D([], [], ls="none", marker=marker, color=color, markersize=4.5,
                      label=label) for color, marker, label in
               reversed(list(STATUS_STYLE.values()))]
    if has_router:
        handles.append(Line2D([], [], color="#D55E00", lw=1.4,
                              label="Router displacement (local mean)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, 0.995), handletextpad=0.3,
               columnspacing=1.0)
    watermark(fig, mark)
    return fig


ROUTE_CMAP = LinearSegmentedColormap.from_list(
    "opera_route", ["#F7F7F5", "#C9CFE6", "#7189BC", "#2D3A8C", "#17212B"]
)
BANK_CMAP = LinearSegmentedColormap.from_list(
    "opera_bank", ["#C16A58", "#EBC4B8", "#F7F7F5", "#B3D3D0", "#3E8A87"]
)


SHORT_DISEASE = {"Amyloidosis": "AL", "Transformed FL": "tFL",
                 "Solitary plasmacytoma": "SP"}


def _home_masked(ax, matrix, cmap, *, label, fig, fraction):
    """Draw a row-normalised source matrix with the home diagonal hatched.

    The home source dominates every row; masking it lets the colour scale
    show where patients borrow from other sources.
    """
    values = matrix.to_numpy(float).copy()
    home = np.eye(len(values), dtype=bool)
    off = np.ma.masked_where(home | ~np.isfinite(values), values)
    image = ax.imshow(off, cmap=cmap, vmin=0,
                      vmax=float(np.quantile(off.compressed(), 0.98)),
                      aspect="equal", interpolation="nearest")
    for index in range(len(values)):
        ax.add_patch(plt.Rectangle((index - 0.5, index - 0.5), 1, 1, facecolor="white",
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
    return routes.rename(columns={column: "w"})


def figure_router_usage(routes, disease_groups, outcome_groups, *, mark=None):
    """Which patients use which adapter sources, and how the two banks interact.

    ``routes`` is the gate summary written by the analysis (``route_summaries``):
    patient fine disease ``cohort``, target ``outcome``, ``bank`` (shared,
    disease or outcome), ``source`` and the mean gate weight. ``disease_groups``
    and ``outcome_groups`` map fine diseases and outcomes to their groups.
    Gate weights describe how corrections are allocated, not their predictive
    contribution; within-bank shares are shown so bank size does not dominate.
    """
    routes = _route_weights(routes).copy()
    routes["patient_group"] = routes.cohort.map(disease_groups)
    routes["target_group"] = routes.outcome.map(outcome_groups)
    fig = plt.figure(figsize=(7.2, 6.5))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.78], wspace=0.72, left=0.14,
                           right=0.93, top=0.955, bottom=0.5)
    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.82], wspace=0.75, left=0.14,
                              right=0.97, top=0.445, bottom=0.12)

    # A: disease bank, patient disease -> source disease (row-normalised).
    ax = fig.add_subplot(top[0])
    diseases = sorted(disease_groups, key=lambda d: (disease_groups[d], d))
    disease = routes.loc[routes.bank == "disease"]
    matrix = (disease.groupby(["cohort", "source"]).w.mean().unstack()
              .reindex(index=diseases, columns=diseases))
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    _home_masked(ax, matrix, ROUTE_CMAP, fig=fig, fraction=0.045,
                 label="Share of disease-bank\nweight (home hatched)")
    breaks, _ = _group_breaks(diseases, disease_groups)
    for position in breaks:
        ax.axhline(position, color="white", lw=1.4, zorder=4)
        ax.axvline(position, color="white", lw=1.4, zorder=4)
    ax.set_yticks(range(len(diseases)), diseases, fontsize=5.8)
    ax.set_xticks(range(len(diseases)), [SHORT_DISEASE.get(d, d) for d in diseases],
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
    per_target = (outcome.groupby(["outcome", "target_group", "source_group"]).w.sum()
                  .reset_index())
    matrix = (per_target.groupby(["target_group", "source_group"]).w.mean().unstack()
              .reindex(index=groups, columns=groups))
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    _home_masked(ax, matrix, ROUTE_CMAP, fig=fig, fraction=0.05,
                 label="Share of outcome-bank\nweight (own group hatched)")
    labels = [SHORT_GROUP.get(g, g) for g in groups]
    ax.set_yticks(range(len(groups)), labels, fontsize=6.2)
    ax.set_xticks(range(len(groups)), labels, fontsize=6.2, rotation=90)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Outcome source group", fontsize=7.5)
    ax.set_ylabel("Target outcome group", fontsize=7.5)
    _panel_title(ax, "B", "Outcome bank")

    # C: interaction - which bank carries the correction, by patient x target.
    ax = fig.add_subplot(bottom[0])
    banks = routes.loc[routes.bank != "shared"].assign(
        bank=lambda frame: frame.bank + "_mass")
    mass = (banks.groupby(["cohort", "outcome", "patient_group", "target_group", "bank"])
            .w.sum().unstack("bank").fillna(0).reset_index())
    mass["disease_share"] = mass.disease_mass / (mass.disease_mass + mass.outcome_mass)
    patient_order = sorted(set(disease_groups.values()))
    table = (mass.groupby(["patient_group", "target_group"]).disease_share.mean().unstack()
             .reindex(index=patient_order, columns=groups))
    spread = float(np.nanmax(np.abs(table.to_numpy(float) - 0.5)))
    spread = min(0.5, max(spread, 0.05))
    image = ax.imshow(table.to_numpy(float), cmap=BANK_CMAP,
                      norm=TwoSlopeNorm(0.5, 0.5 - spread, 0.5 + spread), aspect="auto",
                      interpolation="nearest")
    ax.set_xticks(np.arange(len(groups)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(patient_order)) + 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_yticks(range(len(patient_order)), patient_order, fontsize=6.4)
    ax.set_xticks(range(len(groups)), labels, fontsize=6.2, rotation=50, ha="right",
                  rotation_mode="anchor")
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Target outcome group", fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_title(ax, "C", "Which bank carries the correction")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_ticks([0.5 - spread, 0.5, 0.5 + spread],
                  labels=[f"Outcome\n{1 - (0.5 - spread):.0%}", "Equal",
                          f"Disease\n{0.5 + spread:.0%}"])
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)

    # D: how much of the gate leaves the shared path, by target group.
    ax = fig.add_subplot(bottom[1])
    shared = routes.loc[routes.bank == "shared"].copy()
    shared["off_shared"] = 1 - shared.w
    summary = _summary_rows(
        shared.assign(gain=shared.off_shared, series="all")
        .groupby(["cohort", "target_group", "series"], as_index=False).gain.mean(),
        ["target_group", "series"])
    order = summary.sort_values("median", ascending=False).target_group.tolist()
    summary["target_group"] = summary.target_group.map(lambda g: SHORT_GROUP.get(g, g))
    _diamond_forest(ax, summary, row_key="target_group",
                    row_order=[SHORT_GROUP.get(g, g) for g in order], series_key="series",
                    series_style={"all": (MODEL_STYLE["R"][0], "")},
                    xlabel="Gate weight off the shared path\n(median, IQR across diseases)",
                    show_counts=False, zero_line=False)
    ax.set_xlim(0, min(1.0, ax.get_xlim()[1] * 1.08))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "D", "Routing intensity")
    watermark(fig, mark)
    return fig


# ── Calibration, risk groups, subgroups, coverage and strategy figures ─────────

CAL_MODELS = {
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
    "evaluated": ("#AEB8DA", "All models evaluated"),
    "no_training_events": ("#E7B9A5", "Too few training events for a specialist"),
    "single_class_test": ("#F2DDB0", "No events in test period"),
    "fit_failed": ("#B5451B", "Fit failed"),
    "not_applicable": ("#FFFFFF", "Not applicable"),
}


def figure_calibration(summary, reliability, examples, *, horizon=365, mark=None):
    """Calibration of the joint model against the alternatives.

    ``summary`` is ``calibration_summary`` output joined to endpoint groups;
    ``reliability`` the IPCW reliability bins; ``examples`` maps panel titles
    to (cohort, outcome) for the reliability curves.
    """
    fig = plt.figure(figsize=(7.2, 6.4))
    top = fig.add_gridspec(1, 2, wspace=0.1, left=0.19, right=0.98, top=0.9,
                           bottom=0.47)
    data = summary.loc[(summary.horizon_days == horizon)
                       & (summary.status == "estimated")]
    order = (data.loc[data.model == "J"].groupby("endpoint_group").calibration_slope
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
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if column:
            ax.tick_params(axis="y", labelleft=False)
        _panel_title(ax, letter, title)
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.18, 0.99))
    bottom = fig.add_gridspec(1, len(examples), wspace=0.28, left=0.08, right=0.98,
                              top=0.34, bottom=0.08)
    for index, (title, (cohort, outcome)) in enumerate(examples.items()):
        ax = fig.add_subplot(bottom[index])
        part = reliability.loc[(reliability.cohort == cohort)
                               & (reliability.outcome == outcome)
                               & (reliability.horizon_days == horizon)]
        upper = max(0.05, float(np.nanmax(part[["predicted_weighted_mean",
                                                 "observed_fraction"]].to_numpy())) * 1.1)
        ax.plot([0, upper], [0, upper], color=PALETTE["diagonal"], lw=0.8, ls="--")
        for model in ("X", "J"):
            rows = part.loc[part.model == model].sort_values("probability_bin")
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
        ax.set_title(textwrap.fill(title, 22), fontsize=7.6, loc="left")
        if index == 0:
            ax.set_ylabel("Observed risk (IPCW)", fontsize=7.5)
            ax.text(-0.42, 1.18, "C", transform=ax.transAxes, fontsize=10.5)
        ax.set_xlabel("Predicted risk", fontsize=7.5)
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
                       mark=None, seed=0, event_free=False):
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
    data["model_group"] = pd.cut(data.model_risk, [-np.inf, *thresholds, np.inf],
                                 labels=names, right=False).astype(str)
    grid = np.unique(np.r_[np.linspace(0, horizon_days * 1.25, 120), horizon_days])
    fig = plt.figure(figsize=(7.2, 6.2))
    top = fig.add_gridspec(1, 2, wspace=0.12, left=0.08, right=0.98, top=0.93,
                           bottom=0.60)
    estimates = []
    for column, (key, order, title, letter) in enumerate((
            ("model_group", names, "Model risk group", "A"),
            ("ipi_group", list(ipi_order), ipi_label, "B"))):
        ax = fig.add_subplot(top[column])
        for color, group in zip(_risk_colors(len(order)), order):
            part = data.loc[data[key] == group]
            if len(part) < 10:
                continue
            point, low, high = _cif_band(part.time_days.to_numpy(float),
                                         part.event.to_numpy(int), grid, rng)
            if event_free:
                point, low, high = 1 - point, 1 - high, 1 - low
            ax.fill_between(grid / 365.25, low, high, color=color, alpha=0.14,
                            linewidth=0, step="post")
            ax.step(grid / 365.25, point, where="post", color=color, lw=1.5,
                    label=f"{group} (n={len(part):,})")
            at = np.searchsorted(grid, horizon_days)
            estimates.append({"system": title, "group": group, "n": len(part),
                              "estimate": point[at], "lower": low[at], "upper": high[at],
                              "color": color})
        ax.axvline(horizon_days / 365.25, color=PALETTE["ink_muted"], lw=0.7, ls="--")
        ax.set_xlabel("Years since first-line treatment")
        if column == 0:
            ax.set_ylabel("Treatment-failure-free survival" if event_free
                          else f"Cumulative incidence of\n{outcome_label.lower()}")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        _panel_title(ax, letter, title)
        _legend(ax, ax.get_legend_handles_labels()[0],
                loc="lower left" if event_free else "upper left")
        risk_times = np.array([0, horizon_days / 2, horizon_days])
        risk_counts = [[str(int((data.loc[data[key] == group, "time_days"] >= t).sum()))
                        for t in risk_times] for group in order]
        table = ax.table(cellText=risk_counts, rowLabels=order,
                         colLabels=["At risk: baseline", f"{horizon_days / 2 / 365.25:.1f} y",
                                    f"{horizon_days / 365.25:.1f} y"],
                         bbox=[0.05, -0.43, 0.95, 0.20], cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(5)
        for cell in table.get_celld().values():
            cell.set_edgecolor("white")
        despine(ax, "y")
    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.55, left=0.2,
                              right=0.97, top=0.40, bottom=0.09)
    ax = fig.add_subplot(bottom[0])
    estimates = pd.DataFrame(estimates)
    labels = []
    for y, row in enumerate(estimates.itertuples()):
        ax.plot([row.lower, row.upper], [y, y], color=row.color, lw=1.4, alpha=0.55)
        ax.scatter(row.estimate, y, marker="D", s=26, color=row.color, zorder=3)
        labels.append(f"{row.group} ({row.n:,})")
    ax.set_yticks(range(len(labels)), labels, fontsize=6.8)
    split = int((estimates.system == "Model risk group").sum())
    ax.axhline(split - 0.5, color=PALETTE["grid"], lw=1)
    for label, centre in (("Model", (split - 1) / 2),
                          (ipi_label, split + (len(estimates) - split - 1) / 2)):
        ax.text(-0.72, centre, label, transform=ax.get_yaxis_transform(), ha="center",
                va="center", fontsize=7, color=PALETTE["ink_muted"], rotation=90)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{horizon_days // 365}-year "
                  + ("event-free survival" if event_free else "cumulative incidence")
                  + " (95% CI)")
    _panel_title(ax, "C", "Survival at the horizon" if event_free else "Risk at the horizon")
    despine(ax, "x")

    ax = fig.add_subplot(bottom[1])
    table = []
    for ipi in ipi_order:
        row = []
        for group in names:
            part = data.loc[(data.ipi_group == ipi) & (data.model_group == group)]
            value = (cumulative_incidence(part.time_days.to_numpy(float),
                                          part.event.to_numpy(int), [horizon_days])[0]
                     if len(part) >= 10 else np.nan)
            if event_free:
                value = 1 - value
            row.append((value, len(part)))
        table.append(row)
    values = np.array([[cell[0] for cell in row] for row in table], float)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=LinearSegmentedColormap.from_list(
        "risk", (["#8E2F3C", "#C16A58", "#E7B9A5", "#F7F7F5"] if event_free
                 else ["#F7F7F5", "#E7B9A5", "#C16A58", "#8E2F3C"])), vmin=0, vmax=1,
        aspect="auto")
    for i, row in enumerate(table):
        for j, (value, count) in enumerate(row):
            if count:
                text = f"{value:.0%}\n{count:,}" if np.isfinite(value) else f"–\n{count:,}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.2,
                        color="white" if np.isfinite(value) and
                        (value < 0.45 if event_free else value > 0.55)
                        else PALETTE["ink"])
    ax.set_xticks(range(len(names)), names,
                  fontsize=6.8)
    ax.set_yticks(range(len(ipi_order)), list(ipi_order), fontsize=6.8)
    ax.set_xlabel("Model risk group")
    ax.set_ylabel(ipi_label)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_title(ax, "D", "Reclassification")
    del image
    watermark(fig, mark)
    return fig


def _known_status(data, horizon_days):
    """Patients whose status at the horizon is known: event before it, or
    followed beyond it. The binary label is an event before the horizon."""
    known = (data.time_days >= horizon_days) | (data.event > 0)
    label = (data.event == 1) & (data.time_days <= horizon_days)
    return data.loc[known], label[known].astype(int)


def figure_clinical_score(patients, *, outcome_label, horizon_days=730,
                          thresholds=(0.1, 0.4), ipi_label="IPI",
                          ipi_order=("Low", "Intermediate", "High"),
                          ipi_high=6, age_bins=(0, 60, 75, np.inf), mark=None,
                          event_free=False, model_threshold=None):
    """The model against the clinical score on the patients who have one.

    Supply ``model_threshold`` locked on tuning data for held-out evaluation.
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
    data["model_group"] = pd.cut(data.model_risk, [-np.inf, *thresholds, np.inf],
                                 labels=names, right=False).astype(str)
    known, label = _known_status(data, horizon_days)
    fig = plt.figure(figsize=(7.2, 6.0))
    top = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.42, left=0.09,
                           right=0.97, top=0.93, bottom=0.57)
    ax = fig.add_subplot(top[0])
    precision, recall, _ = precision_recall_curve(label, known.model_risk)
    model_ap = average_precision_score(label, known.model_risk)
    score_ap = average_precision_score(label, known.ipi_score)
    ax.plot(recall, precision, color=MODEL_STYLE["J"][0], lw=1.5,
            label=f"Model (AP {model_ap:.2f})")
    points = []
    for cut in sorted(known.ipi_score.unique())[1:]:
        flagged = known.ipi_score >= cut
        if flagged.sum():
            points.append((cut, float(label[flagged].mean()),
                           float(flagged[label == 1].mean())))
    cuts, score_precision, score_recall = map(np.array, zip(*points))
    ax.plot(score_recall, score_precision, color=PALETTE["ink_muted"], lw=1, ls="--",
            marker="o", ms=3, label=f"{ipi_label} (AP {score_ap:.2f})")
    high = known.ipi_score >= ipi_high
    specificity = float((~high[label == 0]).mean())
    cut = (float(model_threshold) if model_threshold is not None else
           float(np.quantile(known.model_risk[label == 0], specificity)))
    flagged = known.model_risk > cut
    matched = (float(label[flagged].mean()), float(flagged[label == 1].mean()))
    reference = (float(label[high].mean()), float(high[label == 1].mean()))
    ax.scatter(reference[1], reference[0], s=36, color=PALETTE["ink_muted"], zorder=4,
               marker="s")
    ax.scatter(matched[1], matched[0], s=40, color=MODEL_STYLE["J"][0], zorder=4,
               marker="D")
    ax.annotate("", xy=(matched[1], matched[0]), xytext=(reference[1], reference[0]),
                arrowprops=dict(arrowstyle="->", color=PALETTE["ink_secondary"], lw=0.8))
    ax.text(0.97, 0.97, f"Index specificity {specificity:.2f}\n"
            f"recall {reference[1]:.2f} → {matched[1]:.2f}\n"
            f"precision {reference[0]:.2f} → {matched[0]:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
            color=PALETTE["ink_secondary"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=False, fontsize=6.8)
    _panel_title(ax, "A", f"Events by {horizon_days // 365} years")
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
    edges = [0, *thresholds, 1]
    band_labels = names
    for index, label_text in enumerate(band_labels):
        centre = (edges[index] + edges[index + 1]) / 2
        ax.text(centre, 1.025, label_text, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.8,
                color=_risk_colors(len(band_labels))[index],
                fontweight="bold", clip_on=False)
    ax.text(0, -0.25, "Candidate model groups: " + " / ".join(
                [f"< {thresholds[0]:.2f}",
                 *[f"{lo:.2f}–< {hi:.2f}" for lo, hi in zip(thresholds[:-1], thresholds[1:])],
                 f"≥ {thresholds[-1]:.2f}"]),
            transform=ax.transAxes, ha="left", fontsize=5.7,
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
    ax.set_xlabel(f"Model risk at {horizon_days // 365} years")
    ax.grid(False)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.5, color="#B5451B",
                              label="Event"),
                       Line2D([], [], ls="none", marker="o", ms=3.5,
                              color=PALETTE["ink_muted"], label="No event")],
              loc="upper right", frameon=True, facecolor="white", framealpha=0.95,
              edgecolor="none", fontsize=6.6, ncol=1, handletextpad=0.1)
    _panel_title(ax, "B", f"Model risk by {ipi_label} group")
    ax.title.set_y(1.07)  # clear of the risk-group band labels
    despine(ax, "none")

    bottom = fig.add_gridspec(1, 2, wspace=0.12, left=0.13, right=0.97, top=0.41,
                              bottom=0.1)
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
        counts = np.zeros_like(values, dtype=int)
        for i, interval in enumerate(ages.cat.categories):
            for j, group in enumerate(names):
                part = data.loc[(ages == interval) & (data[key] == group)]
                counts[i, j] = len(part)
                if len(part) >= 10:
                    values[i, j] = cumulative_incidence(
                        part.time_days.to_numpy(float), part.event.to_numpy(int),
                        [horizon_days])[0]
                    if event_free:
                        values[i, j] = 1 - values[i, j]
        ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(age_labels)):
            for j in range(len(names)):
                if counts[i, j]:
                    value = values[i, j]
                    text = (f"{value:.0%}" if np.isfinite(value)
                            else "–")
                    ax.text(j, i, text, ha="center", va="center", fontsize=6.2,
                            color="white" if np.isfinite(value) and
                            (value < 0.45 if event_free else value > 0.55)
                            else PALETTE["ink"])
        ax.set_xticks(range(len(names)), names,
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
    metric_label = ("Treatment-failure-free survival" if event_free
                    else f"Cumulative incidence of {outcome_label.lower()}")
    fig.text(0.55, 0.015, f"{metric_label} by "
             f"{horizon_days // 365} years; patient counts in the separate age plate", ha="center",
             fontsize=7, color=PALETTE["ink_secondary"])
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
    fig = plt.figure(figsize=(7.2, 1.45 * len(scores) + 0.95))
    grid = fig.add_gridspec(len(scores), 3, width_ratios=[1, 1, 0.95], wspace=0.28,
                            hspace=0.42, left=0.09, right=0.985,
                            top=1 - 0.72 / fig.get_figheight(),
                            bottom=0.48 / fig.get_figheight())
    for row, spec in enumerate(scores):
        data = spec["patients"].loc[spec["patients"].ipi_group.notna()].copy()
        order = list(spec["ipi_order"])
        names = ["Low", "Intermediate", "High"][:len(spec["thresholds"]) + 1] \
            if len(spec["thresholds"]) == 2 else order
        data["model_group"] = pd.cut(data.model_risk,
                                     [-np.inf, *spec["thresholds"], np.inf],
                                     labels=names, right=False).astype(str)
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
                         "By model risk group" if column == 0 else "By index category",
                         fontsize=8.5, color=PALETTE["ink"], va="top")
            despine(ax, "y")
        ax = fig.add_subplot(grid[row, 2])
        for y, (label, group, n, value, low, high, color) in enumerate(estimates):
            ax.plot([low, high], [y, y], color=color, lw=1.2, alpha=0.55)
            ax.scatter(value, y, marker="D" if label == "Model" else "o", s=18,
                       color=color, zorder=3)
        ax.set_yticks(range(len(estimates)),
                      [f"{label[0]}: {group} ({n:,})" for label, group, n, *_ in estimates],
                      fontsize=5.8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=6.5)
        if row == 0:
            fig.text(ax.get_position().x0, 1 - 0.22 / fig.get_figheight(),
                     f"At {horizon // 365} years (95% CI)", fontsize=8.5,
                     color=PALETTE["ink"], va="top")
        if row != len(scores) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        despine(ax, "x")
        fig.text(0.005, ax.get_position().y1 + 0.006, spec["title"], fontsize=8.5,
                 fontweight="bold", color=PALETTE["ink"], va="bottom")
    watermark(fig, mark)
    return fig


def figure_subgroup_effects(table, *, contrast_styles, covariate_order, xlabel,
                            title_letters="A", panel_title="", scale=100, mark=None):
    """Patient subgroup effects (diamond, 95% CI), grouped by covariate."""
    rows, labels, headers = [], [], []
    for covariate, title in covariate_order.items():
        part = table.loc[table.covariate == covariate]
        headers.append((len(labels), title))
        for subgroup in part.subgroup.drop_duplicates():
            rows.append((covariate, subgroup))
            labels.append(f"  {subgroup}")
    fig, ax = plt.subplots(figsize=(4.6, 0.24 * len(labels) + 0.9 * len(headers) * 0.3
                                    + 1.0))
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
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    handles = [Line2D([], [], ls="none", marker="D", color=color, label=label)
               for color, label in contrast_styles.values()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0.3)
    if panel_title:
        _panel_title(ax, title_letters, panel_title)
    despine(ax, "x")
    fig.subplots_adjust(left=0.42, right=0.97, top=1 - 0.75 / fig.get_figheight(),
                        bottom=0.75 / fig.get_figheight())
    watermark(fig, mark)
    return fig


def figure_coverage(coverage, *, mark=None):
    """Which disease–outcome cells every comparison could be evaluated in."""
    diseases, disease_group = _ordered(coverage, "cohort", "cohort_group")
    outcomes, outcome_group = _ordered(coverage, "outcome", "endpoint_group")
    keys = list(COVERAGE_STYLE)
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


def figure_support(cells, *, horizon=365, edges=(20, 100, 500), mark=None):
    """Training events available to each disease–outcome specialist."""
    data = (cells.loc[cells.horizon_days == horizon]
            .drop_duplicates(["cohort", "outcome"]))
    diseases, disease_group = _ordered(data, "cohort", "cohort_group")
    outcomes, outcome_group = _ordered(data, "outcome", "endpoint_group")
    matrix = (data.pivot_table(index="cohort", columns="outcome", values="n_events_train",
                               aggfunc="first").reindex(index=diseases, columns=outcomes))
    fig = plt.figure(figsize=(7.2, 3.9))
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
    bar.set_label(f"Training events by {horizon} days (log scale; grey: none)", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    bar.outline.set_visible(False)
    ax = fig.add_subplot(grid[1])
    counts = np.sort(np.nan_to_num(values.ravel(), nan=0))
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
    _panel_title(ax, "B", "Cumulative")
    despine(ax, "y")
    watermark(fig, mark)
    return fig


def figure_rescue(cells, *, metric="auc", horizon=365, tolerance=0.0, mark=None):
    """Does routing recover cells where joint learning was worse than specialists?

    ``cells`` holds, per cell, ``tuning_joint_vs_specialist`` (defines the
    strata on tuning data) and held-out ``joint_vs_specialist`` and
    ``routed_vs_specialist`` gains.
    """
    data = cells.loc[(cells.metric == metric) & (cells.horizon_days == horizon)].copy()
    data["stratum"] = np.where(data.tuning_joint_vs_specialist < -tolerance,
                               "Joint worse on tuning", "Joint not worse")
    fig = plt.figure(figsize=(7.2, 3.3))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.55, left=0.09,
                            right=0.98, top=0.8, bottom=0.17)
    ax = fig.add_subplot(grid[0])
    colors = {"Joint worse on tuning": "#B5451B",
              "Joint not worse": PALETTE["ink_muted"]}
    for stratum, color in colors.items():
        part = data.loc[data.stratum == stratum]
        ax.scatter(part.tuning_joint_vs_specialist,
                   part.routed_vs_specialist - part.joint_vs_specialist, s=8,
                   alpha=0.45, color=color, linewidths=0, rasterized=True)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7)
    ax.axvline(-tolerance, color=PALETTE["grid"], lw=1)
    ax.set_xlabel("Joint vs specialist on tuning (AUC)")
    ax.set_ylabel("Routed vs joint on test (AUC)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "A", "Cells by tuning harm")
    despine(ax, "y")
    ax = fig.add_subplot(grid[1])
    long = data.drop(columns=["gain"], errors="ignore").melt(
        id_vars=["stratum"], value_vars=["joint_vs_specialist",
                                                      "routed_vs_specialist"],
                     var_name="series", value_name="gain")
    summary = _summary_rows(long, ["stratum", "series"])
    handles = _diamond_forest(
        ax, summary, row_key="stratum", row_order=list(colors), series_key="series",
        series_style={"joint_vs_specialist": (MODEL_STYLE["J"][0], "Joint vs specialist"),
                      "routed_vs_specialist": (MODEL_STYLE["R"][0], "Routed vs specialist")},
        xlabel="Test AUC gain (median, IQR)", show_counts=True, spread=0.16)
    ax.set_yticklabels([t.get_text().replace(" (n=", "\n(n=")
                        for t in ax.get_yticklabels()])
    ax.set_ylim(1.55, -0.55)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "B", "Test-period gains")
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.985, 0.995), handletextpad=0.3)
    watermark(fig, mark)
    return fig


def figure_strategy(cells, selection, *, horizon=365, mark=None):
    """One joint model versus a specialist, and versus the best model per cell.

    ``cells`` has held-out AUC per cell for ``specialist``, ``joint`` and
    ``best_tuning`` (the specialist, disease-pooled or outcome-pooled model
    chosen per cell on tuning data) and ``n_events_train``; ``selection`` the
    chosen strategy per cell with its ``endpoint_group``.
    """
    fig = plt.figure(figsize=(7.2, 3.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.5, left=0.08,
                            right=0.98, top=0.8, bottom=0.15)
    data = cells.loc[cells.horizon_days == horizon]
    low = float(np.nanmin(data[["specialist", "joint", "best_tuning"]].to_numpy())) - 0.02
    sizes = 4 + 40 * np.sqrt(data.n_events_train / data.n_events_train.max())
    for column, (x, letter, title, label) in enumerate((
            ("specialist", "A", "Joint vs specialist", "Specialist AUC"),
            ("best_tuning", "B", "Joint vs best per cell", "Best-per-cell AUC"))):
        ax = fig.add_subplot(grid[column])
        above = data.joint >= data[x]
        ax.plot([low, 1], [low, 1], color=PALETTE["diagonal"], lw=0.8, ls="--")
        ax.scatter(data[x], data.joint, s=sizes, alpha=0.4, linewidths=0,
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
    ax = fig.add_subplot(grid[2])
    shares = (selection.groupby("endpoint_group").selected_model.value_counts(normalize=True)
              .unstack().fillna(0))
    order = shares.sort_values("S", ascending=True).index if "S" in shares else shares.index
    left = np.zeros(len(order))
    styles = {"S": (MODEL_STYLE["S"][0], "Specialist"),
              "D": (CONTRAST_STYLE["D-S"][0], "Disease-pooled"),
              "O": (CONTRAST_STYLE["O-S"][0], "Outcome-pooled")}
    for key, (color, label) in styles.items():
        values = shares.reindex(order).get(key, pd.Series(0, index=order)).to_numpy()
        ax.barh(range(len(order)), values, left=left, color=color, height=0.72,
                label=label)
        left += values
    ax.set_yticks(range(len(order)), [SHORT_GROUP.get(g, g) for g in order], fontsize=6.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("Share of cells")
    ax.grid(False)
    _panel_title(ax, "C", "Chosen on tuning")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=3,
               frameon=False, fontsize=6.6, handlelength=1, columnspacing=0.8,
               bbox_to_anchor=(0.985, 0.995))
    despine(ax, "none")
    watermark(fig, mark)
    return fig


def figure_embedding_comparison(coords, *, model_order=None, color_rows=None, mark=None):
    """Model columns with fixed coordinates and shared metadata colour rows.

    Metadata must agree for every patient across models. Missing scores remain
    visible in grey; they are never assigned a clinical risk group.
    """
    from opera.visualization.style import CATEGORICAL, sequential_cmap, setup_style
    setup_style()
    rows = color_rows or [
        {"column": "disease", "label": "Disease group", "kind": "categorical"},
        {"column": "ipi_group", "label": "IPI category", "kind": "ordinal",
         "order": ["Low", "Low-intermediate", "Intermediate", "High-intermediate",
                   "High"]},
        {"column": "age", "label": "Age (years)", "kind": "continuous"},
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
            norm = Normalize(0, len(levels) - 1)
            cax = legend_ax.inset_axes([0.08, 0.22, 0.1, 0.56])
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                               ticks=range(len(levels)))
            bar.ax.set_yticklabels(levels, fontsize=6.5)
            bar.outline.set_visible(False)
            cax.set_title(textwrap.fill(row["label"] + " (mean)", 14), fontsize=7.5,
                          loc="left", pad=6)
        elif categorical:
            present = set(base[column].dropna())
            categories = ([c for c in row["order"] if c in present]
                          + sorted(present - set(row["order"]), key=str)
                          if "order" in row else sorted(present, key=str))
            palette = (CATEGORICAL + ["#BD648E", "#697481"]) if column == "disease" else (
                _risk_colors(len(categories)) if "order" in row else CATEGORICAL)
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
            cmap = sequential_cmap()
            cax = legend_ax.inset_axes([0.08, 0.22, 0.1, 0.56])
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
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
                _category_regions(ax, data, column, colors, extent)
            else:
                valid = data.loc[data[column].notna()]
                values = (valid[column].map(rank).to_numpy(dtype=float) if ordinal
                          else valid[column].to_numpy(dtype=float))
                ax.hexbin(valid.x, valid.y, C=values,
                          reduce_C_function=np.mean if ordinal else np.median,
                          gridsize=20, extent=extent, mincnt=4, cmap=cmap, norm=norm,
                          linewidths=0, zorder=2, rasterized=True)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_aspect("auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(PALETTE["grid"])
                spine.set_linewidth(0.6)
            if ri == 0:
                title = str(model).replace(" (tuning-selected)", "")
                ax.set_title(f"{chr(65 + mi)}  " + textwrap.fill(title, 20), loc="left",
                             fontsize=9, pad=5)
    watermark(fig, mark)
    return fig


def _category_regions(ax, data, column, colors, extent, *, label=False, mass=0.5):
    """Faint points plus one contour around the densest half of each category."""
    from scipy.stats import gaussian_kde

    xx, yy = np.meshgrid(np.linspace(*extent[:2], 90), np.linspace(*extent[2:], 90))
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    for category, color in colors.items():
        part = data.loc[data[column] == category]
        ax.scatter(part.x, part.y, s=2.2, color=color, alpha=0.32, linewidths=0,
                   rasterized=True, zorder=2)
        if len(part) < 15:
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
                             event_free=False, mark=None, seed=0,
                             age_bins=(0, 50, 60, 70, 80, np.inf)):
    """Separate counts, mean predictions and observed risk on identical cells."""
    from opera.visualization.style import sequential_cmap
    if event_free and (patients.event > 1).any():
        raise ValueError("Event-free survival requires death included in endpoint")
    names = list(ipi_order)
    if len(names) != len(thresholds) + 1:
        raise ValueError("Group/cutoff mismatch")
    data = patients.loc[patients.ipi_group.notna()].copy()
    data["model_group"] = pd.cut(data.model_risk, [-np.inf, *thresholds, np.inf],
                                 labels=names, right=False)
    ages = pd.cut(data.age, age_bins, right=False)
    labels = [f"<{int(b.right)}" if b.left == 0 else
              (f"≥{int(b.left)}" if not np.isfinite(b.right) else
               f"{int(b.left)}–{int(b.right)-1}") for b in ages.cat.categories]
    values = np.full((3, 2, len(labels), len(names)), np.nan)
    for col, key in enumerate(("model_group", "ipi_group")):
        for i, band in enumerate(ages.cat.categories):
            for j, group in enumerate(names):
                part = data.loc[(ages == band) & (data[key] == group)]
                values[0, col, i, j] = len(part)
                if len(part) >= 10:
                    prediction = float(part.model_risk.mean())
                    observed = cumulative_incidence(part.time_days.to_numpy(),
                                                     part.event.to_numpy(), [horizon_days])[0]
                    values[1, col, i, j] = 1-prediction if event_free else prediction
                    values[2, col, i, j] = 1-observed if event_free else observed
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.2))
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
                    text = "–" if not np.isfinite(v) else (f"{v:,.0f}" if row==0 else f"{v:.0%}")
                    ax.text(j,i,text,ha="center",va="center",fontsize=7,
                            color=PALETTE["ink"] if row==0 or not np.isfinite(v) or v>.6 else "white")
            ax.set_xticks(range(len(names)),names,fontsize=6.5,rotation=15)
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
    fig.text(.13,.025,f"{outcome_label} · {horizon_days} days; risk estimates suppressed for n < 10",
             fontsize=7,color=PALETTE["ink_muted"])
    watermark(fig,mark)
    return fig
