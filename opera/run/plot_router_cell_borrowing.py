"""Tuning-only, fine-disease-by-target-outcome router allocation figure.

Estimand: mean route weight among eligible tuning patients with one fine
disease for one target outcome. These are descriptive gates, not effects on
predicted risk or evidence that a source caused model performance.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import yaml

from opera.visualization.outcome_names import outcome_display_name
from opera.visualization.style import PALETTE, despine, setup_style


setup_style()

PARTS = (
    ("Shared", "shared", "#254B78"),
    ("Own disease", "own_disease", "#3B8C88"),
    ("Other diseases", "other_diseases", "#92C9B4"),
    ("Own outcome", "own_outcome", "#C36B55"),
    ("Other outcomes", "other_outcomes", "#E8A76E"),
)


def parse_cell(value: str, config: dict) -> tuple[str, str, int]:
    if ":" not in value:
        raise ValueError("Cell must be FINE_DISEASE:outcome")
    disease, outcome = value.split(":", 1)
    disease_names = sorted(config["competing_risk"]["fine_names"])
    if disease not in disease_names or outcome not in config["outcomes"]:
        raise ValueError(f"Cell {value!r} is absent from the router config")
    return disease, outcome, disease_names.index(disease)


def read_cell_routes(path: Path, *, fine_index: int, outcome: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        filters=[("split", "=", "tuning"), ("fine_index", "=", fine_index),
                 ("outcome", "=", outcome)],
    )
    if not frame.empty and not frame["split"].eq("tuning").all():
        raise ValueError("Route export contains non-tuning rows")
    return frame


def verify_route_provenance(routes: Path, config: Path, checkpoint: Path) -> dict:
    summary_path = routes.parent / "direct_cr_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Route export summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("split") != "tuning":
        raise ValueError("Route export summary is not tuning-only")
    if Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError("Route export and requested checkpoint differ")
    if config.resolve().parent.parent != checkpoint.resolve().parent:
        raise ValueError("Saved router config and checkpoint must come from the same run")
    if summary.get("route_removal_diagnostic"):
        raise ValueError("Use an unmasked route export for gate-preference figures")
    return summary


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(values[rng.integers(n, size=n)].mean())
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def summarize_cell(
    frame: pd.DataFrame, config: dict, *, disease: str, outcome: str,
    min_patients: int = 20, n_boot: int = 400, seed: int = 42,
) -> dict:
    """Condition on fine disease and target; patients have equal weight."""
    if min_patients < 1 or n_boot < 1:
        raise ValueError("min_patients and n_boot must be positive")
    disease_names = sorted(config["competing_risk"]["fine_names"])
    outcome_names = sorted(config["outcomes"])
    fine_index = disease_names.index(disease)
    if outcome not in outcome_names:
        raise ValueError(f"Unknown outcome: {outcome}")
    if frame.empty or len(frame) < min_patients:
        raise ValueError(f"Cell {disease}:{outcome} has fewer than {min_patients} tuning patients")
    if not frame["split"].eq("tuning").all():
        raise ValueError("Borrowing figure requires tuning-only routes")
    if not frame["fine_index"].eq(fine_index).all() or not frame["outcome"].eq(outcome).all():
        raise ValueError("Route rows do not match the requested fine-disease/outcome cell")
    if frame["subject_id"].duplicated().any():
        raise ValueError("Duplicate patient-target route rows")
    disease_keys = ["disease:" + name for name in disease_names]
    outcome_keys = ["outcome:" + name for name in outcome_names]
    d = np.empty((len(frame), len(disease_keys)), dtype=float)
    o = np.empty((len(frame), len(outcome_keys)), dtype=float)
    for i, (raw_d, raw_o) in enumerate(zip(frame["disease_weights"], frame["outcome_weights"])):
        dw, ow = json.loads(raw_d), json.loads(raw_o)
        if set(dw) != set(disease_keys) or set(ow) != set(outcome_keys):
            raise ValueError("Route source inventory differs from resolved config")
        d[i] = [dw[key] for key in disease_keys]
        o[i] = [ow[key] for key in outcome_keys]
    shared = pd.to_numeric(frame["shared_weight"], errors="raise").to_numpy(dtype=float)
    all_weights = np.column_stack((shared, d, o))
    if (not np.isfinite(all_weights).all() or (all_weights < -1e-6).any()
            or not np.allclose(all_weights.sum(axis=1), 1, atol=1e-3)):
        raise ValueError("Invalid or non-normalized router weights")
    own_outcome_index = outcome_names.index(outcome)
    budget_values = {
        "shared": shared,
        "own_disease": d[:, fine_index],
        "other_diseases": d.sum(axis=1) - d[:, fine_index],
        "own_outcome": o[:, own_outcome_index],
        "other_outcomes": o.sum(axis=1) - o[:, own_outcome_index],
    }
    budget = {key: float(values.mean()) for key, values in budget_values.items()}
    sources = []
    selectivity = []
    for kind, names, weights, home in (
        ("disease", disease_names, d, disease),
        ("outcome", outcome_names, o, outcome),
    ):
        off_home_mass = budget["other_diseases" if kind == "disease" else "other_outcomes"]
        n_off_home = len(names) - 1
        off_home_indices = [index for index, name in enumerate(names) if name != home]
        off_home = weights[:, off_home_indices]
        patient_mass = off_home.sum(axis=1)
        eligible = patient_mass > 1e-12
        if n_off_home and eligible.any():
            conditional = off_home[eligible] / patient_mass[eligible, None]
            peak_enrichment = conditional.max(axis=1) * n_off_home
            effective_fraction = 1 / (n_off_home * np.square(conditional).sum(axis=1))
            selectivity.append({"kind": kind, "n_off_home_sources": n_off_home,
                                "n_patients_with_off_home_mass": int(eligible.sum()),
                                "mean_off_home_bank_weight": off_home_mass,
                                "median_patient_peak_enrichment": float(np.median(peak_enrichment)),
                                "p90_patient_peak_enrichment": float(np.quantile(peak_enrichment, .9)),
                                "median_effective_source_fraction": float(np.median(effective_fraction)),
                                "mean_route_peak_enrichment": float(
                                    off_home.mean(axis=0).max() * n_off_home / off_home_mass)})
        for index, name in enumerate(names):
            if name == home:
                continue
            values = weights[:, index]
            low, high = _bootstrap_mean_ci(
                values, seed=seed + index + (0 if kind == "disease" else 1000),
                n_boot=n_boot,
            )
            sources.append({"kind": kind, "source": name,
                            "display_name": outcome_display_name(name) if kind == "outcome" else name,
                            "mean_weight": float(values.mean()),
                            "ci_low": low, "ci_high": high,
                            "patient_p10_weight": float(np.quantile(values, .1)),
                            "patient_p90_weight": float(np.quantile(values, .9)),
                            "off_home_bank_share": (float(values.mean()) / off_home_mass
                                                    if off_home_mass > 0 else float("nan")),
                            "uniform_bank_enrichment": (float(values.mean()) * n_off_home / off_home_mass
                                                        if off_home_mass > 0 else float("nan"))})
    return {"disease": disease, "outcome": outcome, "n_patients": len(frame),
            "budget": budget, "sources": pd.DataFrame(sources),
            "selectivity": pd.DataFrame(selectivity),
            "route_mean": all_weights.mean(axis=0),
            "route_patient_l1": float(np.abs(all_weights - all_weights.mean(axis=0)).sum(axis=1).mean()),
            "source_names": ["shared", *disease_keys, *outcome_keys]}


def compare_cells(summaries: list[dict]) -> pd.DataFrame:
    """Off-home source excess over the equal-cell mean of other displayed cells."""
    rows = []
    for item in summaries:
        for source in item["sources"].itertuples():
            references = []
            for other in summaries:
                if other is item:
                    continue
                match = other["sources"].loc[
                    (other["sources"]["kind"] == source.kind)
                    & (other["sources"]["source"] == source.source), "mean_weight"]
                if not match.empty:
                    references.append(float(match.iloc[0]))
            if references:
                reference = float(np.mean(references))
                rows.append({"disease": item["disease"], "outcome": item["outcome"],
                             "kind": source.kind, "source": source.source,
                             "mean_weight": source.mean_weight,
                             "reference_mean_weight": reference,
                             "excess_weight": source.mean_weight - reference,
                             "n_reference_cells": len(references)})
    return pd.DataFrame(rows)


def route_similarity(summaries: list[dict]) -> pd.DataFrame:
    """Compare bank-normalized gate vectors on common off-home sources."""
    rows = []
    if len(summaries) < 2:
        return pd.DataFrame(rows)
    names = summaries[0]["source_names"]
    for i, left in enumerate(summaries):
        if left["source_names"] != names:
            raise ValueError("Source inventories differ across cells")
        for right in summaries[i + 1:]:
            for kind in ("disease", "outcome"):
                homes = ({left["disease"], right["disease"]} if kind == "disease"
                         else {left["outcome"], right["outcome"]})
                indices = [j for j, name in enumerate(names)
                           if name.startswith(kind + ":") and name.split(":", 1)[1] not in homes]
                a, b = left["route_mean"][indices], right["route_mean"][indices]
                if not indices or a.sum() <= 0 or b.sum() <= 0:
                    continue
                rows.append({"cell_a": left["disease"] + ":" + left["outcome"],
                             "cell_b": right["disease"] + ":" + right["outcome"],
                             "bank": kind, "common_sources": len(indices),
                             "off_home_total_variation": float(.5 * np.abs(a / a.sum() - b / b.sum()).sum())})
    return pd.DataFrame(rows)


def plot_cells(summaries: list[dict], output: Path, *, top_k: int = 4,
               contrasts: pd.DataFrame | None = None) -> None:
    if not summaries or top_k < 1:
        raise ValueError("At least one cell and one source per bank are required")
    fig, axes = plt.subplots(len(summaries), 3,
                             figsize=(14.2, 2.75 * len(summaries) + 1.5),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1.05, 1.2, 1.55]})
    # The main figure shows absolute allocation, with each task's even-split
    # reference. Cross-cell contrasts and patient selectivity stay in CSV.
    source_limits = {}
    for kind in ("disease", "outcome"):
        source_limits[kind] = max((float(item["sources"].loc[
            item["sources"]["kind"].eq(kind), "mean_weight"].max()) * 1.22
            for item in summaries if item["sources"]["kind"].eq(kind).any()), default=.01)
    for row, summary in enumerate(summaries):
        ax = axes[row, 0]
        left = 0.0
        for label, key, color in PARTS:
            value = summary["budget"][key] * 100
            ax.barh([0], [value], left=left, height=.36, color=color,
                    label=label if row == 0 else None)
            left += value
        ax.set_xlim(0, 100)
        ax.set_ylim(-.55, .55)
        ax.set_yticks([])
        ax.set_xticks([0, 50, 100], ["0", "50", "100"])
        ax.set_xlabel("Total route weight (%)" if row == len(summaries) - 1 else "")
        ax.set_title(f"{summary['disease']}  →  {outcome_display_name(summary['outcome'])}\n"
                     f"{summary['n_patients']:,} patients", loc="left", fontsize=9.5,
                     fontweight="normal", color=PALETTE["ink"])
        despine(ax, "none")
        for col, kind, color in ((1, "disease", "#3B8C88"), (2, "outcome", "#C36B55")):
            ax = axes[row, col]
            top = summary["sources"].loc[lambda x: x["kind"].eq(kind)].nlargest(top_k, "mean_weight")
            top = top.iloc[::-1]
            y = np.arange(len(top))
            bank_mass = summary["budget"]["other_diseases" if kind == "disease" else "other_outcomes"]
            even_share = 100 * bank_mass / max(len(summary["sources"].loc[
                summary["sources"]["kind"].eq(kind)]), 1)
            ax.axvline(even_share, color=PALETTE["panel_border"], lw=1,
                       linestyle=(0, (2, 2)), zorder=1)
            if len(top):
                values = top["mean_weight"].to_numpy() * 100
                ax.hlines(y, np.minimum(even_share, values),
                          np.maximum(even_share, values), color=color, lw=1.5, alpha=.65,
                          zorder=2)
                ax.scatter(values, y, s=32, color=color, edgecolors="white",
                           linewidths=.7, zorder=3)
                for position, value in enumerate(values):
                    ax.text(1.01, position, f"{value:.2f}%", ha="left", va="center",
                            transform=ax.get_yaxis_transform(), fontsize=8,
                            color=PALETTE["ink_secondary"], clip_on=False)
            labels = ([textwrap.fill(outcome_display_name(name), 23,
                                     break_long_words=False) for name in top["source"]]
                      if kind == "outcome" else list(top["source"]))
            ax.set_yticks(y, labels, fontsize=8)
            ax.set_xlim(0, source_limits[kind] * 100)
            ax.set_ylim(-.55, max(len(top) - .45, .55))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.set_xlabel("Mean route weight (%)" if row == len(summaries) - 1 else "")
            if row == 0:
                ax.set_title("Other disease sources" if kind == "disease"
                             else "Other outcome sources", loc="left", fontsize=10)
            despine(ax, "x")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.51, .945), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("How does each task allocate source weight?", x=.055,
                 y=.995, ha="left", fontsize=13)
    fig.text(.055, .018,
             "Tuning patients. Dots: mean source gate weight; dashed line: even split among other sources in that bank. "
             "Gate weight is not predictive contribution.", fontsize=8,
             color=PALETTE["ink_secondary"])
    fig.subplots_adjust(left=.055, right=.925, top=.87, bottom=.095,
                        hspace=.77, wspace=.72)
    fig.savefig(output / "router_cell_borrowing.png", dpi=220)
    fig.savefig(output / "router_cell_borrowing.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, action="append", required=True,
                        help="Repeat once per --cell, or provide one full-panel export.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cell", action="append", required=True,
                        help="FINE_DISEASE:outcome; repeat for each plotted row")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-patients", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if len(args.routes) not in {1, len(args.cell)}:
        raise ValueError("Provide one full-panel route export or one route export per cell")
    route_paths = args.routes * len(args.cell) if len(args.routes) == 1 else args.routes
    for route_path in set(route_paths):
        verify_route_provenance(route_path, args.config, args.checkpoint)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["competing_risk"].get("readout_mode") not in {"source_anchored_adapters", "unanchored_adapters"}:
        raise ValueError("Figure requires a source router")
    scale = float(config["competing_risk"].get("adapter_residual_scale", 0))
    if not 0 < scale <= 1:
        raise ValueError("Figure requires a valid router residual scale")
    if config["competing_risk"].get("adapter_query_mode") != "interacting":
        raise ValueError("Figure requires the interacting router")
    summaries = []
    for label, route_path in zip(args.cell, route_paths):
        disease, outcome, fine_index = parse_cell(label, config)
        frame = read_cell_routes(route_path, fine_index=fine_index, outcome=outcome)
        summaries.append(summarize_cell(frame, config, disease=disease, outcome=outcome,
                                        min_patients=args.min_patients,
                                        n_boot=args.bootstrap))
    args.output_dir.mkdir(parents=True)
    budget_rows = []
    source_rows = []
    selectivity_rows = []
    for item in summaries:
        prefix = {"disease": item["disease"], "outcome": item["outcome"],
                  "n_patients": item["n_patients"]}
        budget_rows.append({**prefix, **item["budget"]})
        for source in item["sources"].to_dict("records"):
            source_rows.append({**prefix, **source})
        for bank in item["selectivity"].to_dict("records"):
            selectivity_rows.append({**prefix, **bank})
    pd.DataFrame(budget_rows).to_csv(args.output_dir / "route_budget_by_cell.csv", index=False)
    pd.DataFrame(source_rows).to_csv(args.output_dir / "all_source_weights_by_cell.csv", index=False)
    pd.DataFrame(selectivity_rows).to_csv(args.output_dir / "route_selectivity_by_cell.csv", index=False)
    contrasts = compare_cells(summaries)
    contrasts.to_csv(args.output_dir / "source_preference_vs_displayed_cells.csv", index=False)
    similarity = route_similarity(summaries)
    similarity.to_csv(args.output_dir / "off_home_route_similarity.csv", index=False)
    pd.DataFrame([{"disease": item["disease"], "outcome": item["outcome"],
                   "n_patients": item["n_patients"],
                   "mean_patient_l1_from_cell_mean": item["route_patient_l1"]}
                  for item in summaries]).to_csv(args.output_dir / "patient_route_variation.csv", index=False)
    (args.output_dir / "estimand_INTERNAL.json").write_text(json.dumps({
        "split": "tuning", "checkpoint_config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "routes": [str(path.resolve()) for path in route_paths], "cells": args.cell,
        "estimand": "Unweighted mean router gate weight over eligible tuning patients "
                    "within one fine disease and one target outcome; home sources "
                    "are separated, not called borrowing.",
        "interval": "Percentile patient bootstrap conditional on one fitted checkpoint; "
                    "not uncertainty over training seeds or predictive benefit.",
        "display": "Top off-home sources by absolute mean gate weight. Source labels also "
                   "show enrichment relative to an equal split of the off-home bank: "
                   "source mean / (off-home bank mean / number of off-home sources). "
                   "Whiskers show patient 10th–90th percentiles, not uncertainty "
                   "in the mean. Patient-level peak and effective-number diagnostics "
                   "distinguish flat individual routes from flat averages. "
                   "Disease and outcome source axes differ; all source means retained in CSV. "
                   "Cross-cell contrasts remain a separate descriptive CSV.",
        "diagnostic": "Pairwise total variation compares bank-normalized routes on common "
                      "off-home sources. Zero means identical; one means disjoint. "
                      "Patient L1 measures mean distance from the cell mean over all gate weights.",
    }, indent=2) + "\n", encoding="utf-8")
    plot_cells(summaries, args.output_dir, contrasts=contrasts)
    print(json.dumps({"status": "tuning_cell_router_preferences",
                      "output_dir": str(args.output_dir.resolve()),
                      "n_cells": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
