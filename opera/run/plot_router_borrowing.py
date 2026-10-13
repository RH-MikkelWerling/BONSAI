"""Aggregate tuning-only router weights into directed source-borrowing heatmaps.

Input is the INTERNAL source_route_weights.parquet from
evaluate_direct_competing_risk --split tuning --export-routes. Output contains
aggregate route weights, not patient rows or performance effects.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from opera.visualization.style import setup_style


# Keep this standalone report on the same typography as the neural atlas.
setup_style()


def _inventory(config: dict) -> tuple[list[str], list[str], dict[str, list[str]]]:
    competing = config["competing_risk"]
    if competing.get("readout_mode") not in {"source_anchored_adapters", "unanchored_adapters"}:
        raise ValueError("Config does not describe a source-adapter router")
    disease = sorted(competing["fine_names"])
    outcome = sorted(config["outcomes"])
    families = dict(config["cross_outcome"]["outcome_families"])
    members = [name for group in families.values() for name in group]
    if len(members) != len(set(members)) or set(members) != set(outcome):
        raise ValueError("Outcome families must partition the configured outcomes")
    return disease, outcome, families


def summarize_routes(frame: pd.DataFrame, config: dict, *, min_rows: int = 20) -> dict:
    """Mean directed route mass; excludes each row's own home adapter."""
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    required = {"subject_id", "split", "outcome", "fine_index", "shared_weight",
                "disease_weights", "outcome_weights"}
    if required - set(frame):
        raise ValueError(f"Route export lacks columns: {sorted(required - set(frame))}")
    if frame.empty or not frame["split"].eq("tuning").all():
        raise ValueError("Borrowing plot requires nonempty tuning-only routes")
    identity = ["subject_id", "outcome"]
    if "cohort" in frame:
        identity.append("cohort")
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate patient-target route rows")
    disease, outcome, families = _inventory(config)
    if not frame["outcome"].isin(outcome).all():
        raise ValueError("Route targets differ from configured outcomes")
    fine = pd.to_numeric(frame["fine_index"], errors="raise").to_numpy(dtype=int)
    known = (fine >= 0) & (fine < len(disease))
    if not known.any():
        raise ValueError("No known fine-disease identities in tuning routes")
    target_index = frame["outcome"].map({name: i for i, name in enumerate(outcome)}).to_numpy(dtype=int)
    shared = pd.to_numeric(frame["shared_weight"], errors="raise").to_numpy(dtype=float)
    dweights = np.empty((len(frame), len(disease)), dtype=np.float32)
    oweights = np.empty((len(frame), len(outcome)), dtype=np.float32)
    disease_keys = ["disease:" + name for name in disease]
    outcome_keys = ["outcome:" + name for name in outcome]
    for i, (d_raw, o_raw) in enumerate(zip(frame["disease_weights"], frame["outcome_weights"])):
        d = json.loads(d_raw)
        o = json.loads(o_raw)
        if set(d) != set(disease_keys) or set(o) != set(outcome_keys):
            raise ValueError(f"Source names differ from resolved config at row {i}")
        dweights[i] = [d[name] for name in disease_keys]
        oweights[i] = [o[name] for name in outcome_keys]
    totals = shared + dweights.sum(axis=1) + oweights.sum(axis=1)
    if (not np.isfinite(totals).all() or not np.isfinite(dweights).all()
            or not np.isfinite(oweights).all()
            or (shared < -1e-6).any() or (dweights < -1e-6).any()
            or (oweights < -1e-6).any() or not np.allclose(totals, 1.0, atol=1e-3)):
        raise ValueError("Invalid or non-normalized router weights")

    disease_mean = np.full((len(disease), len(disease)), np.nan)
    disease_n = np.zeros(len(disease), dtype=int)
    for i in range(len(disease)):
        selected = known & (fine == i)
        disease_n[i] = int(selected.sum())
        if disease_n[i] >= min_rows:
            disease_mean[i] = dweights[selected].mean(axis=0)
    outcome_mean = np.full((len(outcome), len(outcome)), np.nan)
    outcome_disease_mean = np.full((len(outcome), len(disease)), np.nan)
    outcome_n = np.zeros(len(outcome), dtype=int)
    for i in range(len(outcome)):
        selected = known & (target_index == i)
        outcome_n[i] = int(selected.sum())
        if outcome_n[i] >= min_rows:
            outcome_mean[i] = oweights[selected].mean(axis=0)
            outcome_disease_mean[i] = dweights[selected].mean(axis=0)

    # Remove the guaranteed home-source component before labelling mass as
    # borrowing. Within-family off-diagonal cells remain informative.
    np.fill_diagonal(disease_mean, np.nan)
    np.fill_diagonal(outcome_mean, np.nan)
    family_names = list(families)
    family_mean = np.full((len(family_names), len(family_names)), np.nan)
    family_per_adapter = np.full_like(family_mean, np.nan)
    by_outcome = {name: i for i, name in enumerate(outcome)}
    for i, target_family in enumerate(family_names):
        target_rows = [by_outcome[name] for name in families[target_family]]
        for j, source_family in enumerate(family_names):
            source_cols = [by_outcome[name] for name in families[source_family]]
            values = outcome_mean[np.ix_(target_rows, source_cols)]
            # Each target outcome has equal standing, rather than each eligible
            # patient-target record. Source-family mass is summed over members.
            per_target = np.nansum(values, axis=1)
            valid = np.isfinite(values).any(axis=1)
            if valid.any():
                family_mean[i, j] = float(per_target[valid].mean())
                # A summed family can look bright simply because it contains
                # more source outcomes. Use the mean individual-adapter weight
                # for the overview so it is comparable to the disease panel.
                family_per_adapter[i, j] = float(np.nanmean(values))

    mass_rows = []
    for i, target in enumerate(outcome):
        selected = known & (target_index == i)
        if int(selected.sum()) < min_rows:
            continue
        own_disease = dweights[np.flatnonzero(selected), fine[selected]]
        own_outcome = oweights[selected, i]
        mass_rows.append({
            "target_outcome": target, "n_patient_target_rows": int(selected.sum()),
            "mean_shared": float(shared[selected].mean()),
            "mean_disease_bank": float(dweights[selected].sum(axis=1).mean()),
            "mean_outcome_bank": float(oweights[selected].sum(axis=1).mean()),
            "mean_own_disease": float(own_disease.mean()),
            "mean_own_outcome": float(own_outcome.mean()),
            "mean_borrowed_disease": float(
                (dweights[selected].sum(axis=1) - own_disease).mean()
            ),
            "mean_borrowed_outcome": float(
                (oweights[selected].sum(axis=1) - own_outcome).mean()
            ),
        })
    return {
        "disease_names": disease, "outcome_names": outcome,
        "family_names": family_names,
        "disease_matrix": disease_mean, "outcome_matrix": outcome_mean,
        "outcome_disease_matrix": outcome_disease_mean,
        "family_matrix": family_mean,
        "family_per_adapter_matrix": family_per_adapter,
        "disease_counts": disease_n, "outcome_counts": outcome_n,
        "mass_by_target": pd.DataFrame(mass_rows),
        "n_rows": len(frame), "n_unknown_fine": int((~known).sum()),
        "min_rows": min_rows,
    }


def _heatmap(ax, matrix: np.ndarray, rows: list[str], columns: list[str], title: str,
             vmax: float):
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="magma", vmin=0, vmax=vmax,
                      aspect="auto", interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Source adapter (borrowed from)")
    ax.set_ylabel("Target (borrows)")
    ax.set_xticks(range(len(columns)), columns, rotation=70, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    return image


def _plot_route_budget(ax, mass_by_target: pd.DataFrame) -> None:
    """Show the mutually exclusive gate-mass components, outcome-balanced."""
    parts = [
        ("Shared path", "mean_shared", "#254B78"),
        ("Own disease", "mean_own_disease", "#3B8C88"),
        ("Other diseases", "mean_borrowed_disease", "#92C9B4"),
        ("Own outcome", "mean_own_outcome", "#C36B55"),
        ("Other outcomes", "mean_borrowed_outcome", "#E8A76E"),
    ]
    values = mass_by_target[[column for _, column, _ in parts]].mean()
    for y, (label, column, color) in enumerate(parts):
        value = float(values[column])
        ax.barh(y, value, color=color, height=.66)
        ax.text(value + .008, y, f"{value:.1%}", va="center", fontsize=9)
    ax.set_yticks(range(len(parts)), [label for label, _, _ in parts])
    ax.invert_yaxis()
    ax.set_xlim(0, max(.2, float(values.max()) * 1.25))
    ax.set_xlabel("Mean fraction of route weight")
    ax.set_title("C  Where route mass goes", loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.2)


def _source_centered(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deviation from each source's mean use across eligible targets."""
    matrix = np.asarray(matrix, dtype=float)
    baseline = np.full(matrix.shape[1], np.nan)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        valid = np.isfinite(values)
        if valid.any():
            baseline[column] = float(values[valid].mean())
    return matrix - baseline[None, :], baseline


def _specificity_heatmap(ax, matrix: np.ndarray, names: list[str], title: str,
                         limit: float) -> None:
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r",
                      vmin=-limit, vmax=limit, aspect="equal",
                      interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names)),
                  [textwrap.fill(name, 17) for name in names],
                  rotation=65, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(names)),
                  [textwrap.fill(name, 24) for name in names], fontsize=6.5)
    ax.set_xlabel("Source adapter")
    ax.set_ylabel("Target")
    return image


def _anchor_target_figure(summary: dict, output: Path) -> None:
    """Readable, prespecified outcome-level view alongside full matrices."""
    supported = set(summary["mass_by_target"]["target_outcome"])
    anchors = [name for name in (
        "overall_survival", "treatment_failure", "sepsis",
        "neutropenia_g3plus", "aki_first_documented",
    ) if name in summary["outcome_names"] and name in supported]
    if not anchors:
        return
    indices = [summary["outcome_names"].index(name) for name in anchors]
    outcome_specificity, _ = _source_centered(summary["outcome_matrix"])
    disease_specificity, _ = _source_centered(summary["outcome_disease_matrix"])
    # Source selection affects display only, not the stored complete matrix.
    candidate = outcome_specificity[indices]
    finite = np.isfinite(candidate)
    relevance = np.divide(np.nansum(np.abs(candidate), axis=0), finite.sum(axis=0),
                          out=np.zeros(candidate.shape[1]), where=finite.sum(axis=0) > 0)
    top = np.argsort(-relevance, kind="stable")[:15]
    outcome_view = candidate[:, top]
    disease_view = disease_specificity[indices]
    finite_values = np.concatenate([
        matrix[np.isfinite(matrix)] for matrix in (outcome_view, disease_view)
    ])
    limit = max(float(np.max(np.abs(finite_values))) if finite_values.size else 0, .001)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5),
                             gridspec_kw={"width_ratios": [.85, 1.15, 1.5]})
    budget = summary["mass_by_target"].set_index("target_outcome").loc[anchors]
    parts = [
        ("Shared", "mean_shared", "#254B78"),
        ("Own disease", "mean_own_disease", "#3B8C88"),
        ("Other diseases", "mean_borrowed_disease", "#92C9B4"),
        ("Own outcome", "mean_own_outcome", "#C36B55"),
        ("Other outcomes", "mean_borrowed_outcome", "#E8A76E"),
    ]
    left = np.zeros(len(anchors))
    for label, column, color in parts:
        values = budget[column].to_numpy(dtype=float)
        axes[0].barh(range(len(anchors)), values, left=left, color=color,
                     label=label, height=.65)
        left += values
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks(range(len(anchors)), anchors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Fraction of route weight")
    axes[0].set_title("A  Route budget", loc="left", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2,
                   fontsize=7, frameon=False)
    for ax, matrix, names, title in (
        (axes[1], outcome_view, [summary["outcome_names"][i] for i in top],
         "B  Selective outcome sources"),
        (axes[2], disease_view, summary["disease_names"],
         "C  Selective disease sources"),
    ):
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r",
                          vmin=-limit, vmax=limit, aspect="auto",
                          interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(names)),
                      [textwrap.fill(name, 15) for name in names],
                      rotation=70, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(anchors)), anchors, fontsize=7)
        ax.set_xlabel("Source adapter")
    fig.subplots_adjust(left=.09, right=.97, top=.85, bottom=.33, wspace=.42)
    fig.colorbar(image, ax=axes[1:], location="bottom", shrink=.65, pad=.27,
                 label="Weight minus source's mean use across all target outcomes")
    fig.suptitle("How five clinical targets use the router", x=.09, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(.09, .02,
             "Outcome-source columns: top 15 by mean absolute specificity among these five targets; "
             "all sources remain in CSV. Disease weights average over patients' home diagnoses. "
             "Gate preference is not predictive contribution.", fontsize=8)
    fig.savefig(output / "router_borrowing_anchor_targets.png", dpi=220)
    fig.savefig(output / "router_borrowing_anchor_targets.pdf")
    plt.close(fig)


def write_report(summary: dict, config: dict, routes: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    for stem, matrix, rows, columns in (
        ("disease_to_disease", summary["disease_matrix"],
         summary["disease_names"], summary["disease_names"]),
        ("outcome_to_outcome", summary["outcome_matrix"],
         summary["outcome_names"], summary["outcome_names"]),
        ("outcome_to_disease", summary["outcome_disease_matrix"],
         summary["outcome_names"], summary["disease_names"]),
        ("family_to_family", summary["family_matrix"],
         summary["family_names"], summary["family_names"]),
        ("family_to_family_per_adapter", summary["family_per_adapter_matrix"],
         summary["family_names"], summary["family_names"]),
    ):
        pd.DataFrame(matrix, index=rows, columns=columns).to_csv(output / f"{stem}_mean.csv")
    summary["mass_by_target"].to_csv(output / "route_mass_by_target.csv", index=False)
    disease_specificity, disease_baseline = _source_centered(summary["disease_matrix"])
    family_specificity, family_baseline = _source_centered(summary["family_per_adapter_matrix"])
    pd.DataFrame(disease_specificity, index=summary["disease_names"],
                 columns=summary["disease_names"]).to_csv(
                     output / "disease_source_specificity.csv")
    pd.DataFrame(family_specificity, index=summary["family_names"],
                 columns=summary["family_names"]).to_csv(
                     output / "family_source_specificity_per_adapter.csv")
    outcome_specificity, _ = _source_centered(summary["outcome_matrix"])
    pd.DataFrame(outcome_specificity, index=summary["outcome_names"],
                 columns=summary["outcome_names"]).to_csv(
                     output / "outcome_source_specificity.csv")
    pd.DataFrame({"source_disease": summary["disease_names"],
                  "mean_off_home_weight": disease_baseline}).to_csv(
                      output / "disease_source_baseline.csv", index=False)
    pd.DataFrame({"source_family": summary["family_names"],
                  "mean_off_home_per_adapter_weight": family_baseline}).to_csv(
                      output / "family_source_baseline.csv", index=False)
    report = {
        "status": "internal_tuning_route_diagnostic_not_performance_effect",
        "routes": str(routes.resolve()),
        "n_patient_target_rows": summary["n_rows"],
        "n_unknown_fine_rows_excluded": summary["n_unknown_fine"],
        "min_rows_per_target": summary["min_rows"],
        "n_disease_sources": len(summary["disease_names"]),
        "n_outcome_sources": len(summary["outcome_names"]),
        "n_outcome_families": len(summary["family_names"]),
        "source_mode": config["competing_risk"].get("adapter_source_mode", "both"),
        "orientation": "rows borrow from columns",
        "aggregation": "mean route weight over eligible tuning patient-target rows; "
                       "family rows average target outcomes equally; home-source diagonal excluded",
        "overview_family_scale": "mean weight per individual source-outcome adapter; "
                                 "summed source-family mass remains in family_to_family_mean.csv",
        "caveat": "Weights show routing preference, not causal contribution or performance gain.",
    }
    (output / "route_report_INTERNAL.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    matrices = [summary["disease_matrix"], summary["family_per_adapter_matrix"]]
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    vmax = max(float(finite.max()) if finite.size else 0, .01)
    fig = plt.figure(figsize=(20, 9.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.15, .7],
                            left=.075, right=.96, bottom=.25, top=.88, wspace=.52)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    image = _heatmap(axes[0], matrices[0], summary["disease_names"], summary["disease_names"],
                     "A  Disease adapter borrowing", vmax)
    _heatmap(axes[1], matrices[1], summary["family_names"], summary["family_names"],
             "B  Outcome adapter borrowing, grouped by family", vmax)
    _plot_route_budget(axes[2], summary["mass_by_target"])
    colorbar = fig.colorbar(image, ax=axes[:2], location="bottom", shrink=.72,
                            pad=.24, aspect=40)
    colorbar.set_label("Mean weight per source adapter; common color scale")
    fig.suptitle("Directed router borrowing on tuning patients", x=.075, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.075, .025,
             "Rows borrow from columns; home adapter omitted. White = own adapter or unsupported. "
             "Family cells average individual source adapters (not family totals). "
             "Gate weights are not performance effects.", fontsize=8)
    fig.savefig(output / "router_borrowing_overview.png", dpi=220)
    fig.savefig(output / "router_borrowing_overview.pdf")
    plt.close(fig)

    # Primary reading figure: column-centering removes globally popular
    # sources, revealing which targets *selectively* use them. The raw-mass
    # overview above remains necessary for judging total route allocation.
    values = np.concatenate([
        matrix[np.isfinite(matrix)]
        for matrix in (disease_specificity, family_specificity)
    ])
    limit = max(float(np.max(np.abs(values))) if values.size else 0, .001)
    fig, axes = plt.subplots(1, 2, figsize=(17, 9.5),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    image = _specificity_heatmap(
        axes[0], disease_specificity, summary["disease_names"],
        "A  Disease-source specificity", limit,
    )
    _specificity_heatmap(
        axes[1], family_specificity, summary["family_names"],
        "B  Outcome-source specificity (family summary)", limit,
    )
    fig.subplots_adjust(left=.13, right=.95, bottom=.30, top=.89, wspace=.40)
    fig.colorbar(image, ax=axes, location="bottom", shrink=.56, pad=.25,
                 label="Route weight minus this source's mean across targets")
    fig.suptitle("Which targets preferentially use which sources?", x=.13,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(.13, .02,
             "Red = above this source's average; blue = below. Home diagonal omitted. "
             "Family cells are per source adapter. This is preference, not contribution to prediction.",
             fontsize=8)
    fig.savefig(output / "router_borrowing_specificity.png", dpi=220)
    fig.savefig(output / "router_borrowing_specificity.pdf")
    plt.close(fig)

    _anchor_target_figure(summary, output)

    fig, ax = plt.subplots(figsize=(17, 15))
    outcome_matrix = summary["outcome_matrix"]
    finite = outcome_matrix[np.isfinite(outcome_matrix)]
    image = _heatmap(ax, outcome_matrix, summary["outcome_names"], summary["outcome_names"],
                     "Outcome-to-outcome borrowing (home diagonal excluded)",
                     max(float(finite.max()) if finite.size else 0, .01))
    fig.colorbar(image, ax=ax, shrink=.66, label="Mean route weight")
    fig.subplots_adjust(left=.22, right=.95, bottom=.22, top=.95)
    fig.savefig(output / "router_borrowing_outcome_full.png", dpi=220)
    plt.close(fig)

    # Unlike the disease-to-disease overview, this retains target-outcome
    # identity when examining disease-source weights.
    matrix = summary["outcome_disease_matrix"]
    fig, ax = plt.subplots(figsize=(11, 19))
    finite = matrix[np.isfinite(matrix)]
    image = _heatmap(ax, matrix, summary["outcome_names"], summary["disease_names"],
                     "Target outcome to disease-source route weights",
                     max(float(finite.max()) if finite.size else 0, .01))
    fig.colorbar(image, ax=ax, shrink=.55, label="Mean route weight")
    fig.text(.21, .025, "Averaged across eligible tuning patients and their home diagnoses; "
             "not a disease-by-outcome interaction estimate.", fontsize=8)
    fig.subplots_adjust(left=.22, right=.95, bottom=.11, top=.97)
    fig.savefig(output / "router_borrowing_outcome_to_disease.png", dpi=220)
    fig.savefig(output / "router_borrowing_outcome_to_disease.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="The same router run's resolved hydra/config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=20)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.routes)
    summary = summarize_routes(frame, config, min_rows=args.min_rows)
    write_report(summary, config, args.routes, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "n_patient_target_rows": summary["n_rows"],
                      "n_disease_sources": len(summary["disease_names"]),
                      "n_outcome_sources": len(summary["outcome_names"]),
                      "n_outcome_families": len(summary["family_names"])}, indent=2))


if __name__ == "__main__":
    main()
