"""Render the supplementary figures that the recover archive (19 Nov) makes possible:
fig_atlas, supp_strategy, supp_rescue, supp_specialisation and supp_task_organisation.

Aggregates only: per-cell AUCs (model_cells.csv), tuning-split cell contrasts, the
strategy selections made on tuning data, the pooled per-outcome atlas metrics and the
task-structure export of the adapted OPERA checkpoint.

Usage:
  python -m scripts.render_recovered_supplement \
      scratch/download_recover_20261119/ready/recover paper/figures/results_thesis_overlay
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.visualization import opera_design_figures as od
from opera.visualization import paper_figures as pf
from opera.visualization.style import setup_style
from opera.run.paper_story_figures import thesis_style


def _cells(root: Path):
    cells = pd.read_csv(root / "focused_core/model_cells.csv")
    return cells


def _label(frame):
    return frame.assign(outcome=frame.outcome.map(od._outcome_label))


def atlas(root: Path, out: Path):
    """Joint model (cohort encoder), AUC per outcome pooled over diseases, by horizon."""
    data = _label(pd.read_csv(root / "atlas/direct_cr_pooled_metrics.csv"))
    cells = _cells(root)
    events = (cells.loc[cells.horizon_days.ne(0)]
              .groupby(["outcome", "horizon_days"]).n_events_train.sum().rename("n_events_train"))
    groups = cells.drop_duplicates("outcome").set_index("outcome").endpoint_group
    data = data.join(events, on=["outcome", "horizon_days"]).assign(
        endpoint_group=lambda d: d.outcome.map(groups),
        horizon_days=lambda d: d.horizon_days.astype(int))
    if data.endpoint_group.isna().any():
        raise ValueError(f"Atlas outcomes without a group: {data.loc[data.endpoint_group.isna(), 'outcome'].unique()}")
    # Training events come from the evaluable disease cells; five 30-day outcome rows have
    # no evaluable cell, so no event count, and are left out (stated in the caption).
    dropped = data.loc[data.n_events_train.isna(), ["outcome", "horizon_days"]]
    print("atlas rows without training-event counts:", dropped.values.tolist())
    data = data.loc[data.n_events_train.notna()]
    _save(pf.figure_atlas(data), out / "fig_atlas")


def strategy(root: Path, out: Path):
    cells = _cells(root)
    cells = cells.loc[cells.horizon_days.eq(0)].copy()
    chosen = {}
    for name in ("strategy_selection", "strategy_selection_inclusive"):
        s = _label(pd.read_csv(root / f"focused_core/{name}.csv"))
        s = s.loc[s.status.eq("selected_on_tuning")].copy()
        s["selected_model"] = s.selected_model.str.removesuffix("_L").replace({"J": "J"})
        chosen[name] = s
    pick = chosen["strategy_selection"].set_index(["cohort", "outcome"]).selected_model
    cells["choice"] = [pick.get((c, o)) for c, o in zip(cells.cohort, cells.outcome)]
    cells["best_tuning"] = [row[row.choice] if isinstance(row.choice, str) else np.nan
                            for _, row in cells.iterrows()]
    cells = cells.rename(columns={"S": "specialist", "J": "joint"})
    groups = cells.drop_duplicates("outcome").set_index("outcome").endpoint_group
    inclusive = chosen["strategy_selection_inclusive"].assign(
        endpoint_group=lambda d: d.outcome.map(groups))
    fig = pf.figure_strategy(cells, inclusive)
    for text in list(fig.texts):  # the footnote repeats the caption
        text.remove()
    fig.legends[0].set_bbox_to_anchor((0.22, 0.0), transform=fig.transFigure)
    _save(fig, out / "supp_strategy")
    return cells, inclusive


def rescue(root: Path, out: Path):
    """Strata from the tuning split (joint vs specialist, cohort encoder); held-out gains of
    the joint model and of OPERA (cohort encoder) over the specialist in the same cells."""
    tuning = pd.read_csv(root / "tuning/contrasts_all_horizons.csv", low_memory=False)
    tuning = _label(tuning.loc[tuning.contrast.eq("P5") & tuning.metric.eq("auc")
                               & tuning.horizon_days.eq(0)])
    harm = tuning.set_index(["cohort", "outcome"]).gain.rename("tuning_joint_vs_specialist")
    cells = _cells(root)
    cells = cells.loc[cells.horizon_days.eq(0)].join(harm, on=["cohort", "outcome"])
    cells = cells.assign(metric="auc", joint_vs_specialist=cells.J - cells.S,
                         routed_vs_specialist=cells.OPERA - cells.S)
    cells = cells.dropna(subset=["tuning_joint_vs_specialist", "joint_vs_specialist",
                                 "routed_vs_specialist"])
    fig = pf.figure_rescue(cells)
    for text in list(fig.texts):  # its footnote called held-out results tuning-on-tuning
        text.remove()
    _save(fig, out / "supp_rescue")
    return cells


def task_structure(root: Path, out: Path):
    ts = root / "task_structure_R_EA"
    outcomes = pd.read_csv(ts / "outcome_specialisation.csv")
    diseases = pd.read_csv(ts / "disease_specialisation.csv")
    _save(od.figure_specialisation(outcomes, diseases), out / "supp_specialisation")
    read = lambda name: pd.read_csv(ts / name, index_col=0)
    _save(od.figure_task_organisation(
        read("outcome_query_similarity.csv"), read("disease_correction_similarity.csv"),
        outcomes, diseases, json.loads((ts / "task_structure_summary.json").read_text()),
        outcome_correction_similarity=read("outcome_correction_similarity.csv")),
        out / "supp_task_organisation")



def _read_square(ts, name):
    return od._as_square(pd.read_csv(ts / name, index_col=0))


def _limit(*matrices):
    values = np.concatenate([np.abs(m.to_numpy(float)[~np.eye(len(m), dtype=bool)]) for m in matrices])
    return float(np.nanmax(values))


def task_organisation_v2(root: Path, out: Path):
    """Corrections first (A outcomes, B diagnoses, one scale); the outcome queries (C) last,
    small and on their own scale: they are nearly two-dimensional (effective rank 2.3), so
    their cosines are close to +-1 by construction and say little on their own."""
    ts = root / "task_structure_R_EA"
    outcomes = pd.read_csv(ts / "outcome_specialisation.csv")
    diseases = pd.read_csv(ts / "disease_specialisation.csv")
    summary = json.loads((ts / "task_structure_summary.json").read_text())
    corrections = _read_square(ts, "outcome_correction_similarity.csv")
    disease_sim = _read_square(ts, "disease_correction_similarity.csv")
    queries = _read_square(ts, "outcome_query_similarity.csv")
    family = pd.Series(od._mapping(outcomes, "outcome", "endpoint_family"), dtype=object)
    lineage = pd.Series(od._mapping(diseases, "disease", "lineage"), dtype=object)
    order = od._within_group_order(corrections, family, od._domain_order())
    disease_order = od._within_group_order(disease_sim, lineage, od._lineage_order())
    limit = _limit(corrections, disease_sim)
    width, size, small = 7.2, 4.25, 2.3
    height = 0.42 + size + 1.05 + 0.45 + small + 0.95
    fig = plt.figure(figsize=(width, height))
    top = lambda inches: 1 - inches / height  # noqa: E731
    a = fig.add_axes([1.6 / width, top(0.42 + size), size / width, size / height])
    lower = 0.42 + size + 1.05 + 0.45 + small
    b = fig.add_axes([1.6 / width, top(lower), small / width, small / height])
    c = fig.add_axes([4.75 / width, top(lower), small / width, small / height])
    cax = fig.add_axes([(1.6 + (size - 1.9) / 2) / width, top(0.42 + size + 0.62), 1.9 / width, 0.07 / height])
    ccx = fig.add_axes([(4.75 + (small - 1.6) / 2) / width, top(lower + 0.68), 1.6 / width, 0.07 / height])
    image = od._taxonomy_heatmap(a, corrections, family, order=order, label_fn=od._outcome_label,
                                 group_label=od._short_domain, limit=limit, group_names="right",
                                 fontsize=4.6, cmap=od.CORRECTION_SIMILARITY_CMAP)
    od._panel_title(a, "A", "How outcomes are corrected", fontsize=9.5)
    od._stat_line(a, summary.get("outcome_correction_similarity"), "domain")
    od._taxonomy_heatmap(b, disease_sim, lineage, order=disease_order, label_fn=od._disease_label,
                         group_label=od._lineage_label, limit=limit, group_names=None, fontsize=5.4,
                         cmap=od.CORRECTION_SIMILARITY_CMAP)
    od._panel_title(b, "B", "How diagnoses are corrected", fontsize=9.5)
    od._stat_line(b, summary.get("disease_correction_similarity"), "group")
    query = od._taxonomy_heatmap(c, queries.reindex(index=order, columns=order), family, order=order,
                                 group_label=od._short_domain, limit=1.0, group_names=None,
                                 cmap=od.SIMILARITY_CMAP)
    od._panel_title(c, "C", "How outcomes read the record", fontsize=9.5)
    od._stat_line(c, summary.get("outcome_query_similarity"), "domain")
    for mappable, axis, label in ((image, cax, "Cosine similarity of correction maps (A, B)"),
                                  (query, ccx, "Cosine similarity of queries (C)")):
        bar = fig.colorbar(mappable, cax=axis, orientation="horizontal")
        bar.set_label(label, fontsize=7.2)
        bar.ax.tick_params(labelsize=6.5)
        bar.outline.set_visible(False)
        bar.set_ticks([-mappable.norm.vmax, 0, mappable.norm.vmax])
        bar.ax.set_xticklabels([f"{-mappable.norm.vmax:.2f}".replace("-", "−"), "0",
                                f"{mappable.norm.vmax:.2f}"])
    for text in b.texts[-1:]:  # the stat line under B is long: two lines
        text.set_text(text.get_text().replace("; ", ";\n"))
    for text in c.texts[-1:]:
        text.set_text(text.get_text().replace("; ", ";\n"))
    _save(fig, out / "supp_task_organisation")


def disease_map(root: Path, out: Path):
    """Average-linkage tree of the diagnoses by similarity of their learned corrections."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from matplotlib.lines import Line2D
    ts = root / "task_structure_R_EA"
    diseases = pd.read_csv(ts / "disease_specialisation.csv")
    sim = _read_square(ts, "disease_correction_similarity.csv")
    distance = 1 - sim.to_numpy(float)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform((distance + distance.T) / 2, checks=False), "average")
    lineage = diseases.set_index("disease").lineage
    styles, order = od._group_styles(lineage, od._lineage_order())
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    result = dendrogram(tree, orientation="right", labels=list(sim.index), ax=ax,
                        color_threshold=0, above_threshold_color="#9AA3AF", leaf_font_size=7.5)
    ax.set_yticklabels([od._disease_label(t.get_text()) for t in ax.get_yticklabels()])
    for tick, name in zip(ax.get_yticklabels(), result["ivl"]):
        tick.set_color(styles[lineage[name]][0])
    for tick, name in zip(ax.get_yticks(), result["ivl"]):
        color, marker = styles[lineage[name]]
        ax.scatter([0], [tick], marker=marker, s=22, color=color, zorder=3, clip_on=False)
    ax.set_xlabel("1 − cosine similarity of the disease corrections (average linkage)", fontsize=8)
    ax.invert_xaxis()
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0, pad=6)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    handles = [Line2D([], [], ls="", marker=styles[g][1], color=styles[g][0], markersize=5,
                      label=od._lineage_label(g)) for g in order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               fontsize=7, title="Disease group", title_fontsize=7.2, handletextpad=0.2,
               columnspacing=1.0)
    fig.subplots_adjust(left=0.03, right=0.80, top=0.97, bottom=0.24)
    _save(fig, out / "supp_disease_map")


def correction_value(root: Path, out: Path, *, cell_effects=None):
    """Correction size against what OPERA's task-specific parts add over the joint model
    (both on the adapted encoder), per outcome and per diagnosis: mean over cells of the
    AUC averaged over horizons. Descriptive; no intervals."""
    from scipy.stats import spearmanr
    from matplotlib.lines import Line2D
    ts = root / "task_structure_R_EA"
    if cell_effects is None:
        cells = _cells(root)
        cells = cells.loc[cells.horizon_days.eq(0)].assign(added=lambda d: d.R_EA - d.J_EA)
    else:
        cells = cell_effects.copy()
        if cells.duplicated(['cohort', 'outcome']).any():
            raise ValueError('Need one paired all-horizon effect per disease and outcome')
    outcomes = pd.read_csv(ts / "outcome_specialisation.csv")
    outcomes["label"] = outcomes.outcome.map(od._outcome_label)
    outcomes = outcomes.join(cells.groupby("outcome").added.mean(), on="label")
    diseases = pd.read_csv(ts / "disease_specialisation.csv")
    diseases = diseases.join(cells.groupby("cohort").added.mean(), on="disease").dropna(subset=["added"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), sharey=True)
    panels = [(axes[0], outcomes, "endpoint_family", od._domain_order(), "A", "Outcomes",
               "Outcome correction size"),
              (axes[1], diseases, "lineage", od._lineage_order(), "B", "Diagnoses",
               "Disease correction size")]
    for ax, data, key, reference, letter, title, xlabel in panels:
        styles, order = od._group_styles(data[key], reference)
        for group in order:
            part = data.loc[data[key] == group]
            ax.scatter(part.correction_size, part.added, s=18, color=styles[group][0],
                       marker=styles[group][1], linewidths=0, zorder=3)
        rho = spearmanr(data.correction_size, data.added)[0]
        ax.text(0.03, 0.97, f"Spearman ρ {rho:+.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=pf.PALETTE["ink_secondary"])
        ax.axhline(0, color=pf.PALETTE["zero_line"], lw=0.7)
        ax.set_xlabel(xlabel + r" ($\|UD\|_{\mathrm{F}}$)", fontsize=8)
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=9, fontweight="bold")
        pf.despine(ax, "both")
    axes[0].set_ylabel("AUC added by OPERA's parts\n(adapted OPERA − adapted joint)", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.17, wspace=0.12)
    _save(fig, out / "supp_correction_value")
    return outcomes, diseases


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {stem}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recover", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    setup_style()
    thesis_style()
    jobs = {"atlas": atlas, "strategy": strategy, "rescue": rescue, "task_structure": task_structure,
            "task_organisation_v2": task_organisation_v2, "disease_map": disease_map,
            "correction_value": correction_value}
    for name, job in jobs.items():
        if not a.only or name in a.only:
            job(a.recover, a.out)


if __name__ == "__main__":
    main()
