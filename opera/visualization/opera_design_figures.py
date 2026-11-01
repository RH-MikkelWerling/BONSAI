"""Figures for the new OPERA design (outcome-specific reading + own corrections).

They replace the retired router figures (1 November 2026) and read only
aggregate tables:

``figure_gain_sources`` (main text; replaces ``figure_hierarchy``)
    ``gain_sources_cells.csv`` from ``build_paper_plot_tables``: per cell and
    ladder step (``corrections``, ``reading``, ``routing``, ``no_disease``,
    ``no_outcome``) the paired gain with saved panel and domain summaries.
``figure_design_search`` (replaces ``figure_routing``)
    ``design_search.csv``: every tuned design against the selected design and
    against the primary router, paired panel intervals on tuning data.
``figure_task_organisation`` (replaces ``figure_borrowing_map``) and
``figure_specialisation`` (replaces ``figure_router_usage``)
    the task-structure export of ``export_opera_task_structure``.
``figure_correction_arrows`` (replaces ``figure_router_arrows``)
    binned map displacements of ``export_opera_correction_arrows``.
``figure_xgboost_gap``
    ``paired_cells.csv`` (OPERA-X, J-X, OPERA-J by clinical domain).

Heatmaps are ordered by the clinical taxonomy (endpoint family for outcomes,
lineage for diagnoses), clustered within each group, with the group boundaries
drawn: if the model organises its tasks the way clinicians do, blocks on the
diagonal stand out. Clustering across groups would hide that comparison.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from opera.visualization.paper_figures import (
    ALL_HORIZONS, GAIN_CMAP, MODEL_STYLE, SHORT_GROUP, _diamond_forest, _embedding_title,
    _horizon_mask, _log_support_fit, _panel_title, _summary_rows, _split_word, estimable_cells,
    watermark,
)
from opera.visualization.style import CATEGORICAL, PALETTE, despine

# Signed cosine similarity is neither good nor bad: a neutral diverging scale
# (ochre: opposite, indigo: alike), centred at zero, grey for unavailable.
SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    "opera_similarity", ["#B18B4D", "#E6D5B5", "#F7F7F5", "#AEB8DA", "#2D3A8C"])
SIMILARITY_CMAP.set_bad(PALETTE["missing"])

# Twelve clinical domains: the categorical palette plus five further tones,
# with a redundant marker shape, assigned by the paper's domain order.
DOMAIN_TONES = [*CATEGORICAL[:7], "#BD648E", "#D9A52C", "#5C7AEA", "#6E1A2E", "#8E9AAB"]
DOMAIN_MARKERS = ("o", "s", "^", "D")


def _domain_order():
    try:
        from opera.visualization.paper_taxonomy import outcome_endpoint_groups
        return list(dict.fromkeys(outcome_endpoint_groups().values()))
    except Exception:  # pragma: no cover - taxonomy files are part of the repo
        return []


def _lineage_order():
    try:
        from opera.visualization.paper_taxonomy import fine_cohort_groups
        return list(dict.fromkeys(fine_cohort_groups().values()))
    except Exception:  # pragma: no cover
        return []


def _group_styles(groups, reference_order=()):
    """Colour and marker per group, fixed by the reference order when known."""
    order = [g for g in reference_order if g in set(groups)]
    order += sorted(set(groups) - set(order), key=str)
    return {g: (DOMAIN_TONES[i % len(DOMAIN_TONES)], DOMAIN_MARKERS[(i // 7) % len(DOMAIN_MARKERS)])
            for i, g in enumerate(order)}, order


def _short_domain(name):
    return SHORT_GROUP.get(name, name)


def _lineage_label(name):
    from opera.visualization.paper_taxonomy import disease_group_name
    return disease_group_name(name) if isinstance(name, str) else name


def _outcome_label(name):
    from opera.visualization.outcome_names import outcome_display_name
    return outcome_display_name(name)


def _disease_label(name):
    from opera.visualization.paper_taxonomy import disease_name
    return disease_name(name)


def _as_square(frame) -> pd.DataFrame:
    """A similarity matrix as written (index column) or as read back from CSV."""
    frame = pd.DataFrame(frame)
    first = frame.columns[0]
    if str(first).startswith("Unnamed") or (
            not pd.api.types.is_numeric_dtype(frame[first])
            and frame[first].astype(str).tolist() == [str(c) for c in frame.columns[1:]]):
        frame = frame.set_index(first)
        frame.index.name = None
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def _mapping(source, key, value) -> dict:
    """Name -> group from a dict, a Series or a table with ``key`` and ``value``."""
    if source is None:
        return {}
    if isinstance(source, pd.DataFrame):
        data = source.dropna(subset=[value]) if value in source else source.iloc[0:0]
        return dict(zip(data[key].astype(str), data[value])) if key in data else {}
    return {str(k): v for k, v in dict(source).items() if v is not None and v == v}


def _p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ── Where the gain comes from (main text) ─────────────────────────────────────

STEP_STYLE = {
    "corrections": ("#B18B4D", "Correct for disease and outcome"),
    "reading": ("#2D3A8C", "Each outcome reads the record its own way"),
    "routing": ("#3E8A87", "Learn whom to borrow from"),
    "no_disease": ("#C16A58", "Remove disease correction"),
    "no_outcome": ("#BD648E", "Remove outcome correction"),
}
LADDER = (  # step, starts from (None: the joint model), connector style
    ("corrections", None),
    ("reading", "corrections"),
    ("routing", "corrections"),
    ("no_disease", "reading"),
    ("no_outcome", "reading"),
)


def _panel_estimate(part, *, draws=1000, seed=0):
    """Saved paired panel mean and 95% CI, or a labelled descriptive fallback
    (mean of cells with a bootstrap over cells)."""
    if {"panel_mean", "panel_lower", "panel_upper"} <= set(part):
        saved = part[["panel_mean", "panel_lower", "panel_upper"]].drop_duplicates().dropna()
        if len(saved) == 1:
            return (*saved.iloc[0].to_numpy(float), "paired")
    values = part.gain.dropna().to_numpy(float)
    if not len(values):
        return np.nan, np.nan, np.nan, "missing"
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]), "cells"


def figure_gain_sources(cells, *, metric="auc", horizon=ALL_HORIZONS, mark=None):
    """Where OPERA's gain over the joint model comes from.

    A: the ladder from the joint model through own corrections to own reading
    (OPERA), learned routing as a side branch from own corrections, and the
    stage-12 ablations from OPERA; each step is a paired difference with its
    95% interval. B: each step against cell training events, with a descriptive
    smoother. C: each step by clinical domain. Points follow the conventions of
    the gain-by-support figures (>= 10 training events, CI width < 0.4).
    """
    data = cells.loc[(cells.metric == metric) & _horizon_mask(cells, horizon)].copy()
    present = [s for s, _ in LADDER if s in set(data.step) and data.loc[data.step == s].gain.notna().any()]
    if "corrections" not in present and "reading" not in present:
        raise ValueError("Gain-source figure needs the corrections or reading step")
    unit = {"auc": "AUC", "brier": "Brier", "log_loss": "Log loss"}.get(metric, metric)
    fig = plt.figure(figsize=(7.2, 7.6))
    top = fig.add_gridspec(1, 1, left=0.37, right=0.74, top=0.93, bottom=0.66)
    bottom = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.62, left=0.095,
                              right=0.975, top=0.52, bottom=0.075)

    # A: the ladder (waterfall of paired step differences).
    ax = fig.add_subplot(top[0])
    estimates = {s: _panel_estimate(data.loc[data.step == s]) for s in present}
    levels = {None: 0.0}
    rows = [("joint", None)] + [(s, src) for s, src in LADDER if s in present]
    for step, source in LADDER:
        if step in present:
            start = levels.get(source, levels.get("corrections", 0.0)) if source else 0.0
            levels[step] = start + estimates[step][0]
    labels, fallback = [], False
    for y, (step, source) in enumerate(rows):
        if step == "joint":
            ax.scatter(0, y, marker="o", s=34, color=MODEL_STYLE["J"][0], zorder=4, linewidths=0)
            labels.append("Joint model (one shared summary)")
            continue
        color, label = STEP_STYLE[step]
        start = levels.get(source, 0.0) if source else 0.0
        estimate, low, high, kind = estimates[step]
        fallback |= kind == "cells"
        end = start + estimate
        ax.barh(y, end - start, left=start, height=0.46, color=color, alpha=0.22, linewidth=0,
                zorder=2)
        ax.plot([start + low, start + high], [y, y], color=color, lw=1.4, zorder=3,
                solid_capstyle="butt")
        for edge in (start + low, start + high):
            ax.plot([edge, edge], [y - 0.13, y + 0.13], color=color, lw=1.0, zorder=3)
        ax.scatter(end, y, marker="D", s=30, color=color, zorder=4, linewidths=0)
        # Connector from the level this step starts at.
        origin_row = 0 if source is None else [r[0] for r in rows].index(source)
        ax.plot([start, start], [origin_row, y], color=PALETTE["connector"], lw=0.7,
                ls=(0, (2, 2)), zorder=1)
        prefix = "− " if step.startswith("no_") else "+ "
        labels.append(prefix + label)
        ax.text(1.03, y, f"{estimate:+.3f}  ({low:+.3f} to {high:+.3f})".replace("-", "−"),
                transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7, color=PALETTE["ink_secondary"])
    if "reading" in present:
        y = [r[0] for r in rows].index("reading")
        ax.annotate("OPERA", (levels["reading"], y), xytext=(0, 7.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["ink"])
        ax.axvline(levels["reading"], color=STEP_STYLE["reading"][0], lw=0.6, alpha=0.35,
                   ls=(0, (1, 2)), zorder=0)
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
    ax.set_yticks(range(len(rows)), labels, fontsize=7.6)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylim(len(rows) - 0.45, -0.75)
    ax.set_xlabel(f"{unit} gain over the joint model"
                  + (" (mean of cells, 95% CI)" if fallback else " (paired 95% CI)"))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.text(1.03, -0.75, "Step (95% CI)", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=6.8, color=PALETTE["ink_muted"])
    _panel_title(ax, "A", "Where the gain comes from")
    despine(ax, "x")

    # B: each step against training support (grey cells, coloured smoothers).
    ax = fig.add_subplot(bottom[0])
    support_steps = [s for s in ("corrections", "reading", "routing") if s in present]
    known = data.loc[data.n_events_train.notna() & (data.n_events_train > 0)]
    x = known.n_events_train.to_numpy(float)
    grid = np.logspace(np.log10(max(x.min(), 1)), np.log10(x.max()), 120) if len(x) else np.array([1.0])
    shown_values, curves = [], {}
    for step in support_steps:
        part = known.loc[known.step == step]
        points = estimable_cells(part)
        ax.scatter(points.n_events_train, points.gain, s=3, color="#9AA3B5", alpha=0.2,
                   linewidths=0, rasterized=True, zorder=2)
        shown_values.extend(points.gain.tolist())
        color = STEP_STYLE[step][0]
        centre, low, high = _log_support_fit(part.n_events_train.to_numpy(float),
                                             part.gain.to_numpy(float), grid)
        ax.fill_between(grid, low, high, color=color, alpha=0.16, linewidth=0, zorder=3)
        ax.plot(grid, centre, color=color, lw=1.9, zorder=4, solid_capstyle="round")
        curves[step] = centre
    names = {"corrections": "Own corrections", "reading": "Own reading", "routing": "Routing"}
    ax.legend(handles=[Line2D([], [], color=STEP_STYLE[s][0], lw=1.9, label=names[s]) for s in curves],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=6.6,
              handlelength=1.4, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    ax.axhline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
    ax.set_xscale("log")
    if shown_values:
        low, high = np.nanquantile(shown_values, [0.01, 0.99])
        span = max(high - low, 0.02)
        ax.set_ylim(low - 0.08 * span, high + 0.08 * span)
    ax.set_xlabel("Cell training events (log scale)")
    ax.set_ylabel(f"{unit} gain")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    _panel_title(ax, "B", "By training support", pad=22)
    despine(ax, "y")

    # C: each step by clinical domain (saved paired domain summaries).
    ax = fig.add_subplot(bottom[1])
    domain = data.loc[data.step.isin(support_steps)].rename(columns={"step": "series"})
    summary = _summary_rows(domain, ["endpoint_group", "series"])
    sort_step = "reading" if "reading" in support_steps else support_steps[0]
    order = (summary.loc[summary.series == sort_step].sort_values("median", ascending=False)
             .endpoint_group.tolist())
    order += [g for g in summary.endpoint_group.unique() if g not in order]
    summary["endpoint_group"] = summary.endpoint_group.map(_short_domain)
    style = {s: (STEP_STYLE[s][0], names[s]) for s in support_steps}
    handles = _diamond_forest(ax, summary, row_key="endpoint_group",
                              row_order=[_short_domain(g) for g in order], series_key="series",
                              series_style=style, xlabel=f"{unit} gain (median, IQR)",
                              show_counts=False, spread=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
              frameon=False, fontsize=6.6, handletextpad=0.1, columnspacing=0.7,
              borderaxespad=0.2)
    _panel_title(ax, "C", "By clinical domain", pad=22)
    watermark(fig, mark)
    return fig


# ── The design search ─────────────────────────────────────────────────────────

DESIGN_REFERENCE_TITLES = {"J_OA_OWN": "Against the selected design",
                           "R_AB": "Against the router"}


def _seed_band(designs, reference):
    """Range over the selected design's seeds (s42 and the seed repeats)."""
    part = designs.loc[designs.reference == reference]
    seeds = (part.loc[part.seed_of_selected.astype(bool)].gain.dropna().tolist()
             if "seed_of_selected" in part else [])
    base = [0.0] if reference == "J_OA_OWN" else part.loc[part.selected].gain.dropna().tolist()
    values = seeds + base
    return (min(values), max(values)) if len(values) >= 2 else None


def figure_design_search(designs, *, mark=None):
    """Every tuned design against the selected design (A) and the primary router
    (B): paired panel AUC differences with 95% intervals on tuning data, grouped
    by what the design changes. The shaded band is the spread of the selected
    design over seeds 42-45; the selected design is in bold."""
    data = designs.copy()
    data["selected"] = data["selected"].astype(bool) if "selected" in data else False
    if "order" not in data:
        data["order"] = range(len(data))
    rows = (data.drop_duplicates("model").sort_values("order")
            [["model", "code", "label", "group", "order", "selected"]])
    # Groups in catalogue order (the order of their first design).
    groups = list(dict.fromkeys(rows.group))
    layout = []  # (kind, payload)
    for group in groups:
        layout.append(("group", group))
        layout.extend(("design", row) for row in rows.loc[rows.group == group].itertuples())
    references = [r for r in DESIGN_REFERENCE_TITLES if r in set(data.reference)]
    height = 0.6 + 0.128 * len(layout) + 1.15
    fig = plt.figure(figsize=(7.2, height))
    grid = fig.add_gridspec(1, len(references), left=0.43, right=0.86, wspace=0.12,
                            top=1 - 0.55 / height, bottom=0.9 / height)
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(references))]
    colour, selected_colour = PALETTE["ink_secondary"], PALETTE["opera"]
    for index, (ax, reference) in enumerate(zip(axes, references)):
        part = data.loc[data.reference == reference].set_index("model")
        band = _seed_band(data, reference)
        if band is not None:
            ax.axvspan(*band, color=selected_colour, alpha=0.11, lw=0, zorder=0)
        for y, (kind, payload) in enumerate(layout):
            if kind == "group":
                ax.axhline(y - 0.5, color=PALETTE["panel_border"], lw=0.6, zorder=1)
                continue
            if payload.model in part.index:
                row = part.loc[payload.model]
                c = selected_colour if payload.selected else colour
                if np.isfinite([row.lower, row.upper]).all():
                    ax.plot([row.lower, row.upper], [y, y], color=c, lw=1.1, alpha=0.6, zorder=2)
                ax.scatter(row.gain, y, marker="D", s=22 if payload.selected else 13, color=c,
                           zorder=3, linewidths=0)
            elif payload.selected and reference == "J_OA_OWN":
                ax.scatter(0, y, marker="D", s=22, color=selected_colour, zorder=3, linewidths=0)
        ax.axvline(0, color=PALETTE["zero_line"], lw=0.7, zorder=1)
        ax.set_ylim(len(layout) - 0.5, -0.6)
        ax.set_yticks(range(len(layout)))
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("AUC difference (paired 95% CI)", fontsize=8)
        despine(ax, "x")
        _panel_title(ax, "AB"[index], DESIGN_REFERENCE_TITLES[reference], fontsize=9.5)
        if index:
            ax.set_yticklabels([])
        else:
            # Catalogue labels indented by two spaces are variants of the row above.
            labels = [payload if kind == "group" else
                      ("– " + payload.label.strip() if payload.label.startswith("  ") else payload.label)
                      for kind, payload in layout]
            ax.set_yticklabels(labels, fontsize=6.9)
            for tick, (kind, payload) in zip(ax.get_yticklabels(), layout):
                if kind == "group":
                    tick.set_fontweight("bold")
                    tick.set_color(PALETTE["ink"])
                elif payload.selected:
                    tick.set_fontweight("bold")
                    tick.set_color(selected_colour)
                else:
                    tick.set_color(PALETTE["ink_secondary"])
    last = axes[-1]
    for y, (kind, payload) in enumerate(layout):
        if kind == "design":
            last.text(1.04, y, payload.code, transform=last.get_yaxis_transform(), ha="left",
                      va="center", fontsize=5.6, color=PALETTE["ink_muted"], family="DejaVu Sans Mono")
    handles = [Line2D([], [], ls="", marker="s", markersize=7, color=selected_colour, alpha=0.25,
                      label="Seed spread of the selected design (seeds 42–45)"),
               Line2D([], [], ls="", marker="D", markersize=5, color=selected_colour,
                      label="Selected design")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.30, 0.08 / height), ncol=2,
               frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
    watermark(fig, mark)
    return fig


# ── How the model organises its tasks ─────────────────────────────────────────

def _within_group_order(similarity: pd.DataFrame, groups: pd.Series, group_order=()) -> list:
    """Taxonomy order (the reference order, then groups by size and name);
    within a group, average-linkage clustering on 1 - similarity."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    order = []
    sizes = groups.reindex(similarity.index).value_counts()
    known = [g for g in group_order if g in sizes.index]
    rest = sorted(set(sizes.index) - set(known), key=lambda g: (-sizes[g], str(g)))
    for group in [*known, *rest]:
        members = [n for n in similarity.index if groups.get(n) == group]
        if len(members) > 2:
            block = 1 - similarity.loc[members, members].to_numpy(float)
            np.fill_diagonal(block, 0)
            block = np.nan_to_num((block + block.T) / 2, nan=1.0)
            members = [members[i] for i in leaves_list(linkage(squareform(np.clip(block, 0, None), checks=False),
                                                               "average"))]
        order += members
    order += [n for n in similarity.index if groups.get(n) is None]
    return order


def _taxonomy_heatmap(ax, similarity, groups, *, order, label_fn=None, group_label=str,
                      limit=1.0, group_names="right", fontsize=6.0):
    """Similarity matrix in a fixed order with group separators, a group strip
    and group names; row labels only when ``label_fn`` is given."""
    values = similarity.loc[order, order].to_numpy(float).copy()
    np.fill_diagonal(values, np.nan)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=SIMILARITY_CMAP,
                      norm=TwoSlopeNorm(0, -limit, limit), interpolation="nearest", aspect="equal")
    labels = [groups.get(n) for n in order]
    edges = [i for i in range(1, len(order)) if labels[i] != labels[i - 1]]
    for edge in edges:
        ax.axhline(edge - 0.5, color="white", lw=1.0)
        ax.axvline(edge - 0.5, color="white", lw=1.0)
    starts, ends = [0, *edges], [*edges, len(order)]
    strip = ax.get_yaxis_transform()
    for k, (start, end) in enumerate(zip(starts, ends)):
        tone = PALETTE["ink_secondary"] if k % 2 == 0 else "#B9C0CA"
        ax.add_patch(plt.Rectangle((-0.028, start - 0.5), 0.018, end - start, transform=strip,
                                   color=tone, clip_on=False, lw=0))
        name = labels[start]
        if name is None or group_names is None:
            continue
        centre = (start + end - 1) / 2
        side = 1.012 if group_names == "right" else -0.04
        ax.text(side, centre, group_label(name), transform=strip, fontsize=6.2,
                ha="left" if group_names == "right" else "right", va="center",
                color=PALETTE["ink_secondary"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    if label_fn is not None:
        ax.set_yticks(range(len(order)), [label_fn(n) for n in order], fontsize=fontsize)
        ax.tick_params(axis="y", pad=11, length=0)
    else:
        ax.set_yticks([])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    return image


def _stat_line(ax, stat, unit):
    if not stat:
        return
    signed = lambda v: f"{v:+.2f}".replace("-0.00", "+0.00")  # noqa: E731
    ax.text(0.0, -0.02, f"Within {unit} {signed(stat['within'])} vs between {signed(stat['between'])}; "
            f"label permutation {_p_text(stat['p_permutation'])}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.8, color=PALETTE["ink_secondary"])


def figure_task_organisation(outcome_similarity, disease_similarity, outcome_family=None,
                             disease_lineage=None, summary=None, *, outcome_correction_similarity=None,
                             outcome_label=None, disease_label=None, mark=None):
    """A: outcomes by similarity of their attention queries (how they read the
    record), by clinical domain. B: outcomes by similarity of their own
    corrections, in A's order. C: diagnoses by similarity of their own
    corrections, by lineage. Each panel reports the mean similarity within
    versus between groups with a label-permutation p (from the export).

    ``outcome_family`` / ``disease_lineage``: mappings, or the export's
    specialisation tables (``outcome``/``endpoint_family``, ``disease``/``lineage``).
    """
    summary = summary or {}
    outcome_label = outcome_label or _outcome_label
    disease_label = disease_label or _disease_label
    outcomes = _as_square(outcome_similarity)
    diseases = _as_square(disease_similarity)
    corrections = None if outcome_correction_similarity is None else _as_square(outcome_correction_similarity)
    family = pd.Series(_mapping(outcome_family, "outcome", "endpoint_family"), dtype=object)
    lineage = pd.Series(_mapping(disease_lineage, "disease", "lineage"), dtype=object)

    def _limit(*matrices):
        values = [np.abs(m.to_numpy(float)[~np.eye(len(m), dtype=bool)]) for m in matrices]
        values = np.concatenate([v[np.isfinite(v)] for v in values])
        return float(values.max()) if values.size and values.max() > 0 else 1.0

    outcome_order = _within_group_order(outcomes, family, _domain_order())
    disease_order = _within_group_order(diseases, lineage, _lineage_order())
    bars = []
    if corrections is None:
        # Two panels, one colourbar: a colour means the same cosine in both.
        limit = _limit(outcomes, diseases)
        correction_limit = limit
        fig = plt.figure(figsize=(7.2, 4.6))
        a = fig.add_axes([0.25, 0.14, 0.40, 0.40 * 7.2 / 4.6])
        c = fig.add_axes([0.81, 0.40, 0.17, 0.17 * 7.2 / 4.6])
        cax = fig.add_axes([0.81, 0.2, 0.15, 0.018])
        image = _taxonomy_heatmap(a, outcomes, family, order=outcome_order, label_fn=outcome_label,
                                  group_label=_short_domain, limit=limit, group_names=None,
                                  fontsize=5.0 if len(outcome_order) > 30 else 6.5)
        bars.append((image, cax, "Cosine similarity"))
        c_letter = "B"
    else:
        # Queries (A) and correction maps (B, C) are different objects: each
        # kind has its own symmetric scale, shared by the panels of that kind.
        limit = _limit(outcomes)
        correction_limit = _limit(corrections, diseases)
        width, size, small = 7.2, 4.25, 2.3
        height = 0.42 + size + 1.05 + 0.45 + small + 0.95
        fig = plt.figure(figsize=(width, height))
        top = lambda inches: 1 - inches / height  # noqa: E731
        a = fig.add_axes([1.6 / width, top(0.42 + size), size / width, size / height])
        cax = fig.add_axes([1.6 / width, top(0.42 + size + 0.62), 1.9 / width, 0.07 / height])
        lower = 0.42 + size + 1.05 + 0.45 + small
        b = fig.add_axes([0.95 / width, top(lower), small / width, small / height])
        c = fig.add_axes([4.75 / width, top(lower), small / width, small / height])
        cbx = fig.add_axes([2.6 / width, top(lower + 0.68), 1.9 / width, 0.07 / height])
        image = _taxonomy_heatmap(a, outcomes, family, order=outcome_order, label_fn=outcome_label,
                                  group_label=_short_domain, limit=limit, group_names="right",
                                  fontsize=4.6 if len(outcome_order) > 45 else 6.0)
        corrected = _taxonomy_heatmap(b, corrections.reindex(index=outcome_order, columns=outcome_order),
                                      family, order=outcome_order, group_label=_short_domain,
                                      limit=correction_limit, group_names="left")
        _panel_title(b, "B", "How outcomes are corrected", fontsize=9.5)
        _stat_line(b, summary.get("outcome_correction_similarity"), "domain")
        bars += [(image, cax, "Cosine similarity of outcome queries (A)"),
                 (corrected, cbx, "Cosine similarity of correction maps (B, C)")]
        c_letter = "C"
    _taxonomy_heatmap(c, diseases, lineage, order=disease_order, label_fn=disease_label,
                      group_label=_lineage_label, limit=correction_limit, group_names="right",
                      fontsize=5.6 if len(disease_order) > 20 else 6.5)
    _panel_title(a, "A", "How outcomes read the record", fontsize=9.5)
    _stat_line(a, summary.get("outcome_query_similarity"), "domain")
    _panel_title(c, c_letter, "How diagnoses are corrected", fontsize=9.5)
    _stat_line(c, summary.get("disease_correction_similarity"), "lineage")
    for mappable, axis, label in bars:
        bar = fig.colorbar(mappable, cax=axis, orientation="horizontal")
        bar.set_label(label, fontsize=7.5)
        bar.ax.tick_params(labelsize=6.5)
        bar.outline.set_visible(False)
        bar.ax.xaxis.set_major_locator(MaxNLocator(5))
    if mark:
        watermark(fig, mark)
    return fig


# ── Specialisation per task ──────────────────────────────────────────────────

def _spearman(x, y):
    from scipy.stats import spearmanr
    keep = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[keep], y[keep]).statistic if keep.sum() > 2 else np.nan


def _label_extremes(ax, data, x, y, labels, n=3):
    """Name the ``n`` largest values, staggering offsets to keep labels apart."""
    top = data.nlargest(n, y)
    for rank, (_, row) in enumerate(top.iterrows()):
        name = textwrap.shorten(str(labels(row)), 24, placeholder="…")
        ax.annotate(name, (row[x], row[y]), xytext=(-7, -6 - 10 * rank), textcoords="offset points",
                    fontsize=6, color=PALETTE["ink_secondary"], ha="right", va="center", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6),
                    arrowprops=dict(arrowstyle="-", color=PALETTE["connector"], lw=0.5,
                                    shrinkA=0, shrinkB=2))


def figure_specialisation(outcomes: pd.DataFrame, diseases: pd.DataFrame | None = None, *,
                          outcome_label=None, disease_label=None, mark=None):
    """How far each task departs from what is shared, against its data.

    A: each outcome's attention gate |tanh g| (own reading). B: the size of its
    own correction. C: the size of each diagnosis's own correction against the
    diagnosis's training events. Outcomes are coloured by clinical domain,
    diagnoses by lineage; Spearman correlations with training events.
    """
    outcome_label = outcome_label or _outcome_label
    disease_label = disease_label or _disease_label
    data = outcomes.copy()
    support = ("training_events" if "training_events" in data and data.training_events.notna().any()
               else "validation_cases")
    data = data.loc[pd.to_numeric(data[support], errors="coerce") > 0].copy()
    data[support] = data[support].astype(float)
    data["endpoint_family"] = data.endpoint_family.fillna("Other")
    styles, order = _group_styles(data.endpoint_family, _domain_order())
    has_diseases = (diseases is not None and "size" in diseases
                    and pd.to_numeric(diseases["size"], errors="coerce").gt(0).any())
    fig = plt.figure(figsize=(7.2, 4.1))
    grid = fig.add_gridspec(1, 3 if has_diseases else 2, wspace=0.55, left=0.085, right=0.985,
                            top=0.88, bottom=0.42)
    xlabel = {"training_events": "Training events (log scale)",
              "validation_cases": "Validation cases at 1 year (log scale)"}[support]
    panels = [("gate", "A", "Own reading", "Attention gate |tanh g|"),
              ("correction_size", "B", "Own outcome correction", "Correction size (norm)")]
    for index, (column, letter, title, ylabel) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        for family in order:
            part = data.loc[data.endpoint_family == family]
            color, marker = styles[family]
            ax.scatter(part[support], part[column], s=17, color=color, marker=marker, alpha=0.9,
                       linewidths=0, zorder=3)
        rho = _spearman(data[support].to_numpy(float), data[column].to_numpy(float))
        ax.text(0.97, 0.03, f"Spearman ρ {rho:+.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.8, color=PALETTE["ink_secondary"])
        _label_extremes(ax, data, support, column, lambda r: outcome_label(r.outcome))
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _panel_title(ax, letter, title, fontsize=9.5)
        despine(ax, "both")
    handles = [Line2D([], [], ls="", marker=styles[f][1], color=styles[f][0], markersize=4.5,
                      label=f"{_short_domain(f)} ({int((data.endpoint_family == f).sum())})")
               for f in order]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.29), ncol=4 if has_diseases else 3,
               frameon=False, fontsize=6.6, handletextpad=0.2, columnspacing=0.9,
               title="Clinical domain (outcomes)", title_fontsize=6.8, alignment="left")
    if has_diseases:
        part = diseases.copy()
        part["size"] = pd.to_numeric(part["size"], errors="coerce")
        part = part.loc[part["size"] > 0]
        part["lineage"] = part.lineage.fillna("Other")
        lineage_styles, lineage_order = _group_styles(part.lineage, _lineage_order())
        ax = fig.add_subplot(grid[0, 2])
        for group in lineage_order:
            members = part.loc[part.lineage == group]
            color, marker = lineage_styles[group]
            ax.scatter(members["size"], members.correction_size, s=20, color=color, marker=marker,
                       linewidths=0, zorder=3)
        rho = _spearman(part["size"].to_numpy(float), part.correction_size.to_numpy(float))
        ax.text(0.97, 0.03, f"Spearman ρ {rho:+.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.8, color=PALETTE["ink_secondary"])
        _label_extremes(ax, part, "size", "correction_size", lambda r: disease_label(r.disease))
        ax.set_xscale("log")
        ax.set_xlabel("Diagnosis training events (log scale)", fontsize=8)
        ax.set_ylabel("Correction size (norm)", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _panel_title(ax, "C", "Own diagnosis correction", fontsize=9.5)
        despine(ax, "both")
        lineage_handles = [Line2D([], [], ls="", marker=lineage_styles[g][1], color=lineage_styles[g][0],
                                  markersize=4.5, label=_lineage_label(g)) for g in lineage_order]
        fig.legend(handles=lineage_handles, loc="upper left", bbox_to_anchor=(0.69, 0.29), ncol=2,
                  frameon=False, fontsize=6.6, handletextpad=0.2, columnspacing=0.8,
                  title="Lineage (diagnoses)", title_fontsize=6.8, alignment="left")
    if mark:
        watermark(fig, mark)
    return fig


# ── Projected corrections on the shared map ──────────────────────────────────

COMPONENT_STYLE = {
    "total": (PALETTE["ink"], "Reading and corrections together"),
    "reading": (STEP_STYLE["reading"][0], "Own reading"),
    "corrections": (STEP_STYLE["corrections"][0], "Own corrections"),
}


def _density_image(density):
    """Grid image of the binned patient counts (cells below the export's
    minimum are absent and stay blank)."""
    xs, ys = np.sort(density.x.unique()), np.sort(density.y.unique())
    width, height = float(density.width.iloc[0]), float(density.height.iloc[0])
    x0, y0 = xs.min(), ys.min()
    nx = int(round((xs.max() - x0) / width)) + 1
    ny = int(round((ys.max() - y0) / height)) + 1
    image = np.full((ny, nx), np.nan)
    i = np.rint((density.x - x0) / width).astype(int)
    j = np.rint((density.y - y0) / height).astype(int)
    image[j, i] = np.log10(density["count"].to_numpy(float))
    extent = (x0 - width / 2, x0 + (nx - 0.5) * width, y0 - height / 2, y0 + (ny - 0.5) * height)
    return image, extent


def figure_correction_arrows(arrows, density=None, meta=None, *, outcomes=None, components=None,
                             min_coherence=0.3, arrow_scale=None, mark=None):
    """Where each outcome moves patients away from the shared representation.

    On the shared-representation map (the patient map), arrows show the binned
    mean displacement from a patient's shared summary to the outcome's final
    representation (reading and corrections), and optionally its two parts.
    Only bins with at least 20 patients are exported and only bins whose shifts
    agree in direction (coherence >= ``min_coherence``) are drawn. One arrow
    magnification is shared by every panel and stated in the key.
    """
    from opera.visualization.paper_figures import _map_axis

    data = arrows.copy()
    names = list(dict.fromkeys(outcomes or (meta or {}).get("outcomes") or data.outcome))
    names = [n for n in names if n in set(data.outcome)][:6]
    rows = [c for c in (components or COMPONENT_STYLE) if c in set(data.component)]
    if not names or not rows:
        raise ValueError("No correction arrows to draw")
    kept = data.loc[data.outcome.isin(names) & data.component.isin(rows)
                    & (data.coherence >= min_coherence)]
    if density is not None and len(density):
        image, extent = _density_image(density)
    else:
        image = None
        pad = 0.05
        x, y = data.center_x, data.center_y
        extent = (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)
    # The view covers the backdrop and every drawn arrow bin, with a margin.
    xs = [extent[0], extent[1], *kept.center_x]
    ys = [extent[2], extent[3], *kept.center_y]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    view = (min(xs) - 0.08 * span, max(xs) + 0.08 * span, min(ys) - 0.08 * span, max(ys) + 0.08 * span)
    if arrow_scale is None:
        lengths = np.hypot(kept.dx, kept.dy).to_numpy(float)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
        # As the router arrows (0.09 of the map per median arrow), but no
        # longer than 0.15 of the map for the 90th-percentile arrow.
        arrow_scale = (float(np.clip(min(0.09 * span / np.median(lengths),
                                         0.15 * span / np.quantile(lengths, 0.9)), 1, 50))
                       if len(lengths) else 1.0)
    if not np.isfinite(arrow_scale) or arrow_scale <= 0:
        raise ValueError("Arrow scale must be positive and finite")
    columns = len(names)
    panel = min(1.45, 6.4 / columns)
    width = 0.75 + panel * columns
    ratio = float(np.clip((view[3] - view[2]) / (view[1] - view[0]), 0.55, 1.0))
    height = 0.75 + panel * ratio * len(rows) + 0.35
    fig = plt.figure(figsize=(max(width, 5.0), height))
    left = 0.62 / max(width, 5.0)
    grid = fig.add_gridspec(len(rows), columns, wspace=0.14, hspace=0.1, left=left,
                            right=left + panel * columns / max(width, 5.0),
                            top=1 - 0.72 / height, bottom=0.12 / height)
    backdrop = LinearSegmentedColormap.from_list("opera_density", ["#EEF1F5", "#C3CAD6"])
    for row, component in enumerate(rows):
        color = COMPONENT_STYLE[component][0]
        for column, name in enumerate(names):
            ax = fig.add_subplot(grid[row, column])
            _map_axis(ax)
            if image is not None:
                ax.imshow(np.ma.masked_invalid(image), origin="lower", extent=extent, cmap=backdrop,
                          interpolation="bilinear", zorder=1, alpha=0.9)
            part = kept.loc[(kept.outcome == name) & (kept.component == component)]
            if len(part):
                ax.quiver(part.center_x, part.center_y, part.dx * arrow_scale, part.dy * arrow_scale,
                          color=color, angles="xy", scale_units="xy", scale=1, width=0.0075,
                          headwidth=4.0, headlength=4.0, headaxislength=3.6, zorder=3)
            else:
                ax.text(0.5, 0.5, "No coherent local shift", transform=ax.transAxes, ha="center",
                        va="center", fontsize=6, color=PALETTE["ink_muted"])
            ax.set_xlim(view[:2])
            ax.set_ylim(view[2:])
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(_embedding_title(name, width=18), fontsize=7.2, loc="center", pad=4)
            if column == 0:
                ax.set_ylabel(textwrap.fill(COMPONENT_STYLE[component][1], 16), fontsize=7.4)
                ax.text(-0.36, 1.0, "ABC"[row], transform=ax.transAxes, fontsize=11,
                        fontweight="bold", va="top", color=PALETTE["ink"])
    handles = [Line2D([], [], color=COMPONENT_STYLE[c][0], lw=1.6, label=COMPONENT_STYLE[c][1])
               for c in rows]
    handles.append(Line2D([], [], ls="", label=f"Arrows magnified ×{arrow_scale:.1f}"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 1 - 0.04 / height), handletextpad=0.4, columnspacing=1.1)
    fig.opera_arrow_scale = arrow_scale
    watermark(fig, mark)
    return fig


# ── Where XGBoost still leads ────────────────────────────────────────────────

GAP_STYLE = {
    "J-X": (MODEL_STYLE["J"][0], "Joint vs XGBoost"),
    "OPERA-X": (MODEL_STYLE["OPERA"][0], "OPERA vs XGBoost"),
}


def figure_xgboost_gap(cells, *, metric="auc", horizon=ALL_HORIZONS, mark=None):
    """The gap to per-task XGBoost by clinical domain (A), and OPERA's gain over
    the joint model in the same domains (B). Rows run from the largest gap;
    dashed lines mark the all-cell means."""
    data = cells.loc[(cells.metric == metric) & _horizon_mask(cells, horizon)].copy()
    opera_x = "OPERA-X" if "OPERA-X" in set(data.contrast) else "R-X"
    opera_j = "OPERA-J" if "OPERA-J" in set(data.contrast) else "R1"
    gap_style = {"J-X": GAP_STYLE["J-X"], opera_x: GAP_STYLE["OPERA-X"]}
    data = data.loc[data.contrast.isin([*gap_style, opera_j])]
    if data.loc[data.contrast == opera_x].gain.notna().sum() == 0:
        raise ValueError("XGBoost-gap figure needs OPERA vs XGBoost cells")
    unit = {"auc": "AUC", "brier": "Brier", "log_loss": "Log loss"}.get(metric, metric)
    summary = _summary_rows(data.rename(columns={"contrast": "series"}), ["endpoint_group", "series"])
    order = (summary.loc[summary.series == opera_x].sort_values("median").endpoint_group.tolist())
    order += [g for g in summary.endpoint_group.unique() if g not in order]
    counts = (data.loc[data.contrast == opera_x].dropna(subset=["gain"])
              .groupby("endpoint_group").size())
    labels = [f"{_short_domain(g)} ({counts.get(g, 0)})" for g in order]
    summary["row"] = summary.endpoint_group.map(dict(zip(order, labels)))
    fig = plt.figure(figsize=(7.2, 4.1))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12, left=0.2, right=0.98,
                            top=0.8, bottom=0.14)
    ax = fig.add_subplot(grid[0])
    handles = _diamond_forest(ax, summary.loc[summary.series.isin(gap_style)], row_key="row",
                              row_order=labels, series_key="series", series_style=gap_style,
                              xlabel=f"{unit} difference (median, IQR)", show_counts=False, spread=0.18)
    ax.tick_params(axis="y", length=0, labelsize=7.2)
    _overall_means(ax, data, gap_style)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
              frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0)
    _panel_title(ax, "A", "Gap to per-task XGBoost", pad=20)
    ax = fig.add_subplot(grid[1], sharey=ax)
    gain_style = {opera_j: (MODEL_STYLE["OPERA"][0], "OPERA vs joint")}
    _diamond_forest(ax, summary.loc[summary.series == opera_j], row_key="row", row_order=labels,
                    series_key="series", series_style=gain_style,
                    xlabel=f"{unit} difference (median, IQR)", show_counts=False)
    _overall_means(ax, data, gain_style)
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _panel_title(ax, "B", "OPERA vs the joint model", pad=20)
    watermark(fig, mark)
    return fig


def _overall_means(ax, data, styles):
    for series, (color, _) in styles.items():
        part = data.loc[data.contrast == series]
        estimate = _panel_estimate(part)[0]
        if np.isfinite(estimate):
            ax.axvline(estimate, color=color, lw=0.8, ls=(0, (3, 2)), alpha=0.8, zorder=1)
