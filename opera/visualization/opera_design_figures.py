"""Figures for the new OPERA design (outcome-specific reading + own corrections).

Inputs are the aggregate tables of ``opera.run.export_opera_task_structure``.
Heatmaps are ordered by the clinical taxonomy (endpoint family for outcomes,
lineage for diagnoses), clustered within each group, with the group boundaries
drawn: if the model organises its tasks the way clinicians do, blocks on the
diagonal stand out. Clustering across groups would hide that comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from opera.visualization.paper_figures import GAIN_CMAP, watermark
from opera.visualization.style import CATEGORICAL, despine


def _within_group_order(similarity: pd.DataFrame, groups: pd.Series) -> list:
    """Taxonomy order (groups by size, then name); within a group, average-linkage
    clustering on 1 - similarity."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    order = []
    sizes = groups.value_counts()
    for group in sorted(sizes.index, key=lambda g: (-sizes[g], str(g))):
        members = [n for n in similarity.index if groups.get(n) == group]
        if len(members) > 2:
            block = 1 - similarity.loc[members, members].to_numpy()
            np.fill_diagonal(block, 0)
            members = [members[i] for i in leaves_list(linkage(squareform(np.clip(block, 0, None), checks=False),
                                                               "average"))]
        order += members
    order += [n for n in similarity.index if groups.get(n) is None]
    return order


def _taxonomy_heatmap(ax, similarity, groups, *, title, label_fn, stat=None):
    order = _within_group_order(similarity, groups)
    values = similarity.loc[order, order].to_numpy(float).copy()
    np.fill_diagonal(values, np.nan)
    limit = float(np.nanmax(np.abs(values))) or 1.0
    image = ax.imshow(np.ma.masked_invalid(values), cmap=GAIN_CMAP, norm=TwoSlopeNorm(0, -limit, limit),
                      interpolation="nearest", aspect="equal")
    labels = [groups.get(n) for n in order]
    edges = [i for i in range(1, len(order)) if labels[i] != labels[i - 1]]
    for edge in edges:
        ax.axhline(edge - 0.5, color="white", lw=1.2)
        ax.axvline(edge - 0.5, color="white", lw=1.2)
    starts = [0, *edges]
    ends = [*edges, len(order)]
    colours = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(dict.fromkeys(labels))}
    # Group strip left of the matrix: fixed width in axes units, rows in data units.
    strip = ax.get_yaxis_transform()
    for start, end in zip(starts, ends):
        ax.add_patch(plt.Rectangle((-0.035, start - 0.5), 0.025, end - start, transform=strip,
                                   color=colours[labels[start]], clip_on=False, lw=0))
    ax.grid(False)
    ax.tick_params(axis="y", pad=12, length=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label_fn(n) for n in order], fontsize=5 if len(order) > 30 else 7)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(order) - 0.5)
    subtitle = ""
    if stat:
        p = stat["p_permutation"]
        subtitle = (f"\nwithin {stat['within']:+.2f} vs between {stat['between']:+.2f} "
                    f"(permutation p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})")
    ax.set_title(title + subtitle, fontsize=9, loc="left")
    return image


def figure_task_organisation(outcome_similarity, disease_similarity, outcome_family, disease_lineage,
                             summary=None, *, outcome_label=str, disease_label=str, mark=None):
    """A: outcomes by similarity of their queries (how they read the record),
    ordered by endpoint family. B: diagnoses by similarity of their own
    corrections (how they are adjusted), ordered by lineage."""
    summary = summary or {}
    fig = plt.figure(figsize=(11, 6.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 0.04], wspace=0.45)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    a = _taxonomy_heatmap(axes[0], outcome_similarity, pd.Series(outcome_family),
                          title="A  How outcomes read the record", label_fn=outcome_label,
                          stat=summary.get("outcome_query_similarity"))
    _taxonomy_heatmap(axes[1], disease_similarity, pd.Series(disease_lineage),
                      title="B  How diagnoses are adjusted", label_fn=disease_label,
                      stat=summary.get("disease_correction_similarity"))
    bar = fig.colorbar(a, cax=axes[2])
    bar.set_label("Cosine similarity", fontsize=8)
    if mark:
        watermark(fig, mark)
    return fig


def figure_specialisation(outcomes: pd.DataFrame, *, outcome_label=str, mark=None):
    """How far each outcome departs from what is shared, against its data. A:
    attention gate |tanh g| (own reading). B: size of its own correction."""
    from scipy.stats import spearmanr

    data = outcomes.dropna(subset=["validation_cases"]).copy()
    families = list(dict.fromkeys(data.endpoint_family.fillna("Other")))
    colours = {f: CATEGORICAL[i % len(CATEGORICAL)] for i, f in enumerate(families)}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    for ax, column, title in ((axes[0], "gate", "A  Own reading (attention gate)"),
                              (axes[1], "correction_size", "B  Own correction (size)")):
        for family in families:
            part = data[data.endpoint_family.fillna("Other") == family]
            ax.scatter(part.validation_cases, part[column], s=18, color=colours[family], label=family,
                       alpha=0.85, lw=0)
        rho = spearmanr(data.validation_cases, data[column]).statistic
        ax.set_xscale("log")
        ax.set_xlabel("Validation cases (1 year)")
        ax.set_title(f"{title}\nSpearman with cases {rho:+.2f}", fontsize=9, loc="left")
        despine(ax, grid_axis="both")
        for _, row in data.nlargest(3, column).iterrows():
            ax.annotate(outcome_label(row.outcome), (row.validation_cases, row[column]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
    axes[0].set_ylabel("|tanh(gate)|")
    axes[1].set_ylabel("‖up · down‖")
    axes[1].legend(fontsize=6, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if mark:
        watermark(fig, mark)
    return fig
