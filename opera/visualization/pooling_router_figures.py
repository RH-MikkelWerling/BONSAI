"""Figures for the partial-pooling router (R_PP; lock amendment 2026-10-30).

Inputs are the frozen-checkpoint mechanism audit's outputs, which reconstruct
the fitted router exactly: ``borrowed_weights_by_task.csv`` (mean gate of every
borrowed source for each disease x outcome task), ``report.json``
(``partial_pooling``: tau and per-task training events; ``route_concentration``)
and the paper cell metadata (lineage, endpoint family and endpoint group).

Routes are shown relative to an even mix (log2 enrichment), so a group's
number of sources does not masquerade as routing, and an even router reads as
neutral rather than as pale structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from opera.visualization.paper_figures import (
    GAIN_CMAP, MODEL_STYLE, SHORT_DISEASE, SHORT_GROUP, _group_breaks, _panel_title, watermark,
)
from opera.visualization.style import PALETTE, despine

LEVELS = (("lineage", "Lineage"), ("endpoint_family", "Endpoint family"),
          ("endpoint_group", "Endpoint group"))
ENRICHMENT_LIMIT = 2.0  # log2 scale: a quarter to four times an even mix


def _disease_label(name):
    try:
        from opera.visualization.paper_taxonomy import disease_name
        return disease_name(name)
    except Exception:  # noqa: BLE001 - display fallback only
        return name


def _outcome_label(name):
    from opera.visualization.outcome_names import outcome_display_name
    return outcome_display_name(name)


def bank_enrichment(tasks, bank):
    """Per task and borrowed source in one bank: share of that bank's borrowed
    weight and its log2 ratio to an even mix over the bank's available sources."""
    part = tasks.loc[(tasks.bank == bank) & (tasks.mean_gate > 1e-12)].copy()
    part["name"] = part.source.str.split(":", n=1).str[1]
    keys = ["disease", "outcome"]
    part["share"] = part.mean_gate / part.groupby(keys).mean_gate.transform("sum")
    part["even"] = 1.0 / part.groupby(keys).name.transform("size")
    part["log2_enrichment"] = np.log2(part.share / part.even)
    return part


def _enrichment_image(ax, matrix, fig, *, label, home=None):
    values = np.clip(matrix.to_numpy(float), -ENRICHMENT_LIMIT, ENRICHMENT_LIMIT)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=GAIN_CMAP,
                      norm=TwoSlopeNorm(0, -ENRICHMENT_LIMIT, ENRICHMENT_LIMIT),
                      aspect="equal", interpolation="nearest")
    if home is not None:
        for row, column in zip(*np.nonzero(home)):
            ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, facecolor="white",
                                       edgecolor=PALETTE["ink_muted"], hatch="//////",
                                       linewidth=0, zorder=3))
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02, shrink=0.7, anchor=(0.0, 1.0))
    ticks = np.arange(-ENRICHMENT_LIMIT, ENRICHMENT_LIMIT + 1)
    bar.set_ticks(ticks, labels=[f"{2 ** t:g}×" if t >= 0 else f"1/{2 ** -t:g}×" for t in ticks])
    bar.set_label(label, fontsize=6.6)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=1.5)
    ax.set_anchor("N")


def _own_weight_panel(ax, pooling):
    tau = float(pooling["tau"])
    events = np.asarray(pooling["cell_train_events"], float).ravel()
    events = events[events > 0]
    grid = np.logspace(0, np.log10(max(events.max() if len(events) else 10, tau) * 2), 200)
    color = MODEL_STYLE["OPERA"][0]
    ax.plot(grid, grid / (grid + tau), color=color, lw=1.8, zorder=3)
    ax.axvline(tau, color=PALETTE["ink_muted"], lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.axhline(0.5, color=PALETTE["grid"], lw=0.7, zorder=1)
    ax.text(tau * 1.08, 0.04, f"τ = {tau:.0f} events", fontsize=7, color=PALETTE["ink_muted"])
    ax.plot(events, np.full(len(events), -0.045), "|", color=PALETTE["ink_muted"],
            markersize=5, alpha=0.35, clip_on=False)
    ax.set_xscale("log")
    ax.set_ylim(-0.08, 1.02)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel("Training events of the task (log scale)")
    ax.set_ylabel("Weight on own corrections")
    despine(ax, "y")
    handles = [Line2D([], [], color=color, lw=1.8, label="Own weight n / (n + τ)"),
               Line2D([], [], color=PALETTE["ink_muted"], marker="|", linestyle="none",
                      alpha=0.6, label=f"Tasks ({len(events):,})")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    _panel_title(ax, "A", "How much to borrow")


def _concentration_panel(ax, concentration, reference=None):
    rows = [(key, label) for key, label in LEVELS
            if (concentration.get(key) or {}).get("status") == "complete"]
    for y, (key, _) in enumerate(rows):
        item = concentration[key]
        expected = item["expected_in_group_share"]
        ax.plot([expected, item["observed_in_group_share"]], [y, y],
                color=MODEL_STYLE["OPERA"][0], lw=1.0, alpha=0.4, zorder=1)
        ax.scatter(expected, y, marker="o", s=26, facecolors="white",
                   edgecolors=PALETTE["ink_muted"], linewidths=1.0, zorder=3)
        ax.scatter(item["observed_in_group_share"], y, marker="D", s=30,
                   color=MODEL_STYLE["OPERA"][0], linewidths=0, zorder=4)
        if reference and (reference.get(key) or {}).get("status") == "complete":
            ax.scatter(reference[key]["observed_in_group_share"], y + 0.22, marker="D", s=20,
                       color=PALETTE["ink_muted"], alpha=0.7, linewidths=0, zorder=3)
        ax.annotate(f"×{item['enrichment']:.1f}", (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                    color=PALETTE["ink"])
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.set_xlabel("Borrowed weight within the target's own group")
    despine(ax, "x")
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                      markeredgecolor=PALETTE["ink_muted"], label="Even mix"),
               Line2D([], [], marker="D", linestyle="none", color=MODEL_STYLE["OPERA"][0],
                      label="Partial-pooling router")]
    if reference:
        handles.append(Line2D([], [], marker="D", linestyle="none", color=PALETTE["ink_muted"],
                              alpha=0.7, label="Earlier router (R_AB)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3,
              frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.0)
    _panel_title(ax, "B", "Within prespecified groups")


def _disease_routes(ax, fig, tasks, lineage):
    part = bank_enrichment(tasks, "disease")
    mean = part.groupby(["disease", "name"]).log2_enrichment.mean().unstack()
    order = sorted(set(mean.index) | set(mean.columns), key=lambda d: (lineage.get(d, "~"), d))
    mean = mean.reindex(index=order, columns=order)
    home = np.equal.outer(np.array(order), np.array(order))
    _enrichment_image(ax, mean, fig, label="Borrowing relative\nto an even mix", home=home)
    groups = {d: lineage.get(d, "~") for d in order}
    for position in _group_breaks(order, groups)[0]:
        ax.axhline(position, color="white", lw=1.4, zorder=4)
        ax.axvline(position, color="white", lw=1.4, zorder=4)
    labels = [_disease_label(d) for d in order]
    ax.set_yticks(range(len(order)), labels, fontsize=5.8)
    ax.set_xticks(range(len(order)), [SHORT_DISEASE.get(l, l) for l in labels],
                  fontsize=5.8, rotation=90)
    ax.set_xlabel("Source disease", fontsize=7.5)
    ax.set_ylabel("Target disease", fontsize=7.5)
    _panel_title(ax, "C", "Which diseases are borrowed from")


def _outcome_group_routes(ax, fig, tasks, endpoint_group):
    part = bank_enrichment(tasks, "outcome")
    part["target_group"] = part.outcome.map(endpoint_group)
    part["source_group"] = part.name.map(endpoint_group)
    keys = ["disease", "outcome", "target_group", "source_group"]
    grouped = part.groupby(keys).agg(share=("share", "sum"), even=("even", "sum")).reset_index()
    grouped["log2_enrichment"] = np.log2(grouped.share / grouped.even)
    groups = sorted(set(endpoint_group.values()))
    matrix = (grouped.groupby(["target_group", "source_group"]).log2_enrichment.mean()
              .unstack().reindex(index=groups, columns=groups))
    _enrichment_image(ax, matrix, fig, label="Borrowing relative\nto an even mix")
    for index in range(len(groups)):
        ax.add_patch(plt.Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False,
                                   edgecolor=PALETTE["ink"], linewidth=0.8, zorder=4))
    labels = [SHORT_GROUP.get(g, g) for g in groups]
    ax.set_yticks(range(len(groups)), labels, fontsize=6.2)
    ax.set_xticks(range(len(groups)), labels, fontsize=6.2, rotation=90)
    ax.set_xlabel("Source outcome group", fontsize=7.5)
    ax.set_ylabel("Target outcome group", fontsize=7.5)
    _panel_title(ax, "D", "Which outcome groups are borrowed from")


def _groupings(cell_metadata):
    meta = cell_metadata.drop_duplicates(["cohort", "outcome"])
    lineage = meta.drop_duplicates("cohort").set_index("cohort").cohort_group.astype(str).to_dict()
    outcomes = meta.drop_duplicates("outcome").set_index("outcome")
    return (lineage, outcomes.endpoint_group.astype(str).to_dict(),
            outcomes.loss_endpoint.astype(str).to_dict() if "loss_endpoint" in outcomes else {})


def figure_how_to_borrow(tasks, report, cell_metadata, *, reference=None, mark=None):
    """How much each task borrows (A), whether borrowing concentrates within
    prespecified groups (B), and which diseases (C) and outcome groups (D) are
    borrowed from, relative to an even mix. ``reference`` is the earlier
    router's ``route_concentration`` for comparison in B."""
    lineage, endpoint_group, _ = _groupings(cell_metadata)
    fig = plt.figure(figsize=(9.2, 9.0))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.85], wspace=0.55, left=0.09,
                           right=0.9, top=0.955, bottom=0.66)
    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.55, left=0.12,
                              right=0.95, top=0.5, bottom=0.1)
    _own_weight_panel(fig.add_subplot(top[0]), report["partial_pooling"])
    _concentration_panel(fig.add_subplot(top[1]), report.get("route_concentration", {}), reference)
    _disease_routes(fig.add_subplot(bottom[0]), fig, tasks, lineage)
    _outcome_group_routes(fig.add_subplot(bottom[1]), fig, tasks, endpoint_group)
    watermark(fig, mark)
    return fig


def figure_outcome_routes(tasks, cell_metadata, *, mark=None):
    """Every target outcome against every source outcome (supplement), ordered
    by endpoint group and family; own outcome hatched, families outlined."""
    _, endpoint_group, family = _groupings(cell_metadata)
    part = bank_enrichment(tasks, "outcome")
    mean = part.groupby(["outcome", "name"]).log2_enrichment.mean().unstack()
    order = sorted(set(mean.index) | set(mean.columns),
                   key=lambda o: (endpoint_group.get(o, "~"), family.get(o, o), o))
    mean = mean.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(9.2, 9.0))
    fig.subplots_adjust(left=0.22, right=0.9, top=0.97, bottom=0.2)
    _enrichment_image(ax, mean, fig, label="Borrowing relative\nto an even mix",
                      home=np.equal.outer(np.array(order), np.array(order)))
    groups = {o: endpoint_group.get(o, "~") for o in order}
    for position in _group_breaks(order, groups)[0]:
        ax.axhline(position, color="white", lw=1.4, zorder=4)
        ax.axvline(position, color="white", lw=1.4, zorder=4)
    families = {o: family.get(o, o) for o in order}
    for start, label in _group_breaks(order, families)[1]:
        size = sum(1 for o in order if families[o] == label)
        if size > 1:
            ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), size, size, fill=False,
                                       edgecolor=PALETTE["ink"], linewidth=0.7, zorder=5))
    labels = [_outcome_label(o) for o in order]
    ax.set_yticks(range(len(order)), labels, fontsize=5)
    ax.set_xticks(range(len(order)), labels, fontsize=5, rotation=90)
    ax.set_xlabel("Source outcome", fontsize=7.5)
    ax.set_ylabel("Target outcome", fontsize=7.5)
    watermark(fig, mark)
    return fig
