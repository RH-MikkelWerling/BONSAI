"""One-page, count-fair summary of all tuning disease/outcome router tasks.

Consumes the aggregate cube from export_router_borrowing_atlas; no patient rows
or held-out labels are needed. Route preferences are not predictive benefit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml

from opera.visualization.style import PALETTE, setup_style
from opera.visualization.outcome_names import outcome_display_name


setup_style()

FAMILY_SHORT_NAMES = {
    "Disease course and later malignancy": "Disease course",
    "Infection and antimicrobial use": "Infection",
    "Erythroid abnormalities and transfusion": "Erythroid",
    "White-cell abnormalities": "White cells",
    "Platelet and severe cytopenia": "Platelets",
    "Kidney injury and renal support": "Kidney",
    "Hepatic and coagulation toxicity": "Liver/coag.",
    "Sodium and potassium": "Sodium/K",
    "Calcium and magnesium": "Calcium/Mg",
    "Metabolic and cell-turnover markers": "Metabolic",
    "Cardiovascular and thrombotic events": "Cardiovascular",
    "Acute events and care use": "Acute care",
}


def short_family_name(name: str) -> str:
    return FAMILY_SHORT_NAMES.get(name, name.replace(" and ", "/")[:17])


def summarize(cube: np.ndarray, counts: np.ndarray, diseases: list[str],
              outcomes: list[str], families: dict[str, list[str]],
              min_patients: int) -> dict[str, np.ndarray]:
    """Equal weight per eligible disease/outcome cell, not per record."""
    d, o = len(diseases), len(outcomes)
    if cube.shape != (d, o, 1 + d + o) or counts.shape != (d, o):
        raise ValueError("Route cube dimensions do not match source inventory")
    members = [name for group in families.values() for name in group]
    if len(members) != len(set(members)) or set(members) != set(outcomes):
        raise ValueError("Outcome families must partition target outcomes")
    valid = counts >= min_patients
    routes = np.where(valid[:, :, None], cube, np.nan)
    own_d = np.stack([routes[i, :, 1 + i] for i in range(d)])
    own_o = np.stack([routes[:, j, 1 + d + j] for j in range(o)], axis=1)
    other_d = np.nansum(routes[:, :, 1:1 + d], axis=2) - own_d
    other_o = np.nansum(routes[:, :, 1 + d:], axis=2) - own_o
    other_d[~valid] = np.nan
    other_o[~valid] = np.nan
    index = {name: j for j, name in enumerate(outcomes)}
    family_indices = [[index[name] for name in group] for group in families.values()]
    def family_mean(values: np.ndarray) -> np.ndarray:
        result = []
        for cols in family_indices:
            selected = values[:, cols]
            n = np.isfinite(selected).sum(axis=1)
            result.append(np.divide(np.nansum(selected, axis=1), n,
                                    out=np.full(d, np.nan), where=n > 0))
        return np.stack(result, axis=1)
    disease_mass = family_mean(other_d)
    outcome_mass = family_mean(other_o)
    # Conditional lift: each off-home source's share divided by an even split.
    # Normalizing *within each task* prevents high-bank-mass tasks dominating.
    dlift = np.full((d, d), np.nan)
    for target in range(d):
        for source in range(d):
            if source != target:
                numerator = routes[target, :, 1 + source]
                ratio = np.divide((d - 1) * numerator, other_d[target],
                                  out=np.full(o, np.nan), where=other_d[target] > 0)
                if np.isfinite(ratio).any():
                    dlift[target, source] = np.nanmean(ratio)
    olift = np.full((o, o), np.nan)
    for target_o in range(o):
        for source_o in range(o):
            if source_o != target_o:
                ratio = np.divide((o - 1) * routes[:, target_o, 1 + d + source_o],
                                  other_o[:, target_o], out=np.full(d, np.nan),
                                  where=other_o[:, target_o] > 0)
                if np.isfinite(ratio).any():
                    olift[target_o, source_o] = np.nanmean(ratio)
    flift = np.full((len(families), len(families)), np.nan)
    for target_f, target_cols in enumerate(family_indices):
        for source_f, source_cols in enumerate(family_indices):
            values = []
            for target_o in target_cols:
                for source_o in source_cols:
                    if source_o == target_o:
                        continue
                    ratio = np.divide((o - 1) * routes[:, target_o, 1 + d + source_o],
                                      other_o[:, target_o], out=np.full(d, np.nan),
                                      where=other_o[:, target_o] > 0)
                    values.extend(ratio[np.isfinite(ratio)].tolist())
            if values:
                flift[target_f, source_f] = float(np.mean(values))
    return {"other_disease_mass": disease_mass, "other_outcome_mass": outcome_mass,
            "disease_source_lift": dlift, "outcome_source_lift": olift,
            "outcome_family_source_lift": flift,
            "eligible_cells": valid}


def draw(summary: dict[str, np.ndarray], diseases: list[str],
         families: list[str], output: Path) -> None:
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    teal = LinearSegmentedColormap.from_list("opera_teal", ["#F8FAFB", "#B9D8D1", "#287C7B"])
    warm = LinearSegmentedColormap.from_list("opera_warm", ["#F8FAFB", "#EBC6B8", "#B65F4D"])
    diverge = colormaps["PuOr_r"]
    fig = plt.figure(figsize=(16.8, 11.8), facecolor="white")
    # Fixed axes prevent long labels from pushing colorbars into neighbouring panels.
    boxes = [(.075, .635, .36, .245), (.59, .635, .36, .245),
             (.075, .235, .36, .245), (.59, .235, .36, .245)]
    axes = [fig.add_axes(box) for box in boxes]
    bar_boxes = [(.105, .535, .29, .012), (.62, .535, .29, .012),
                 (.105, .105, .29, .012), (.62, .105, .29, .012)]
    mass_max = max(np.nanquantile(summary[key], .98) for key in
                   ("other_disease_mass", "other_outcome_mass"))
    disease_lift = summary["disease_source_lift"]
    outcome_lift = summary["outcome_family_source_lift"]
    disease_max = max(1.1, float(np.nanquantile(disease_lift, .98)))
    outcome_max = max(1.1, float(np.nanquantile(outcome_lift, .98)))
    panels = [
        ("other_disease_mass", "A  Other-disease borrowing", diseases, families,
         teal, None, mass_max, "Mean off-home gate mass"),
        ("other_outcome_mass", "B  Other-outcome borrowing", diseases, families,
         warm, None, mass_max, "Mean off-home gate mass"),
        ("disease_source_lift", "C  Which diseases are preferred?", diseases, diseases,
         diverge, TwoSlopeNorm(vmin=0, vcenter=1, vmax=disease_max), None,
         "Preference vs even split (×)"),
        ("outcome_family_source_lift", "D  Which outcome families are preferred?",
         families, families, diverge, TwoSlopeNorm(vmin=0, vcenter=1, vmax=outcome_max),
         None, "Preference vs even split (×)"),
    ]
    for panel_index, (ax, (key, title, rows, cols, cmap, norm, vmax, label)) in enumerate(zip(axes, panels)):
        values = np.ma.masked_invalid(summary[key])
        shown_rows = [short_family_name(x) for x in rows] if key == "outcome_family_source_lift" else rows
        shown_cols = [short_family_name(x) for x in cols] if key != "disease_source_lift" else cols
        image = ax.imshow(values, cmap=cmap, norm=norm, vmin=None if norm else 0,
                          vmax=vmax, aspect="auto", interpolation="nearest")
        image.cmap.set_bad("#EDF0F3")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=12)
        ax.set_yticks(range(len(rows)), shown_rows, fontsize=8)
        ax.set_xticks(range(len(cols)), shown_cols, rotation=55, ha="right", fontsize=7.5)
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_ylabel("Patient disease" if key.endswith("mass") else "Target", labelpad=7)
        bar = fig.colorbar(image, cax=fig.add_axes(bar_boxes[panel_index]),
                           orientation="horizontal")
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    fig.text(.075, .97, "A map of borrowing across diseases and outcomes",
             fontweight="bold", fontsize=16, color=PALETTE["ink"])
    fig.text(.075, .943, "All eligible tuning tasks  ·  patient-balanced within tasks  ·  outcomes equally weighted within families",
             fontsize=9, color=PALETTE["ink_secondary"])
    fig.text(.075, .018,
             "A–B: total off-home gate mass; a larger outcome bank can have more total mass without selective routing. "
             "C–D: 1× means an even split within each bank; preference panels have separate scales.\n"
             "Grey = insufficient tuning patients or a home source. Gate weight is not predictive benefit; "
             "outcome-family preference is per source adapter, not summed family mass.",
             fontsize=8, color=PALETTE["ink_secondary"], va="bottom")
    fig.savefig(output / "router_borrowing_publication_atlas.pdf")
    fig.savefig(output / "router_borrowing_publication_atlas.png", dpi=250)
    plt.close(fig)


def draw_individual_outcomes(summary: dict[str, np.ndarray], outcomes: list[str],
                             families: dict[str, list[str]], output: Path) -> None:
    """Full-resolution individual-outcome source map for the supplement."""
    from matplotlib import colormaps
    order = [outcomes.index(name) for members in families.values() for name in members]
    values = summary["outcome_source_lift"][np.ix_(order, order)]
    display = [outcome_display_name(outcomes[index]) for index in order]
    high = max(1.1, float(np.nanquantile(values, .99)))
    fig, ax = plt.subplots(figsize=(23, 22), facecolor="white")
    fig.subplots_adjust(left=.23, right=.93, top=.94, bottom=.27)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=colormaps["PuOr_r"],
                      norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=high),
                      interpolation="nearest", aspect="equal")
    image.cmap.set_bad("#EDF0F3")
    ax.set_xticks(range(len(order)), display, rotation=90, fontsize=5.9)
    ax.set_yticks(range(len(order)), display, fontsize=5.9)
    ax.tick_params(length=0)
    ax.grid(False)
    boundaries = np.cumsum([len(members) for members in families.values()])[:-1] - .5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=1.3)
        ax.axvline(boundary, color="white", linewidth=1.3)
    ax.set_xlabel("Other outcome source (borrowed from)", labelpad=12)
    ax.set_ylabel("Target outcome (borrows)", labelpad=12)
    fig.text(.23, .975, "Which individual outcomes does the router prefer?",
             fontsize=17, fontweight="bold", color=PALETTE["ink"])
    fig.text(.23, .957, "Each cell is an individual target–source outcome pair; ordered by clinical endpoint group.",
             fontsize=10, color=PALETTE["ink_secondary"])
    bar_ax = fig.add_axes([.23, .085, .65, .012])
    bar = fig.colorbar(image, cax=bar_ax, orientation="horizontal")
    bar.set_label("Source preference relative to an even split among other outcomes (1×)")
    fig.text(.23, .025,
             "Mean of patient-balanced eligible disease × target cells. Grey diagonal = home source; "
             "grey off-diagonal = insufficient support. Color limits are specific to this outcome panel. "
             "Preference is not a predictive effect.",
             fontsize=9, color=PALETTE["ink_secondary"])
    fig.savefig(output / "individual_outcome_source_preference.pdf")
    fig.savefig(output / "individual_outcome_source_preference.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-patients", type=int, default=20)
    args = parser.parse_args()
    if args.min_patients < 1:
        raise ValueError("min-patients must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    with np.load(args.cube, allow_pickle=False) as cube:
        means, counts = cube["mean"], cube["count"]
        diseases = cube["disease_names"].tolist()
        outcomes = cube["outcome_names"].tolist()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    families = config["cross_outcome"]["outcome_families"]
    result = summarize(means, counts, diseases, outcomes, families, args.min_patients)
    args.output_dir.mkdir(parents=True)
    draw(result, diseases, list(families), args.output_dir)
    draw_individual_outcomes(result, outcomes, families, args.output_dir)
    for key in ("other_disease_mass", "other_outcome_mass"):
        pd.DataFrame(result[key], index=diseases, columns=list(families)).to_csv(
            args.output_dir / f"{key}.csv")
    pd.DataFrame(result["disease_source_lift"], index=diseases, columns=diseases).to_csv(
        args.output_dir / "disease_source_lift.csv")
    pd.DataFrame(result["outcome_family_source_lift"], index=list(families),
                 columns=list(families)).to_csv(args.output_dir / "outcome_family_source_lift.csv")
    pd.DataFrame(result["outcome_source_lift"], index=outcomes,
                 columns=outcomes).to_csv(args.output_dir / "outcome_source_lift.csv")
    (args.output_dir / "figure_metadata.json").write_text(json.dumps({
        "split": "tuning", "source_cube": str(args.cube),
        "eligible_task_cells": int(result["eligible_cells"].sum()),
        "estimand": "Equal-outcome-within-family task means of off-home route mass; conditional source preference relative to uniform within each off-home bank.",
        "warning": "Descriptive route allocation, not predictive or causal benefit."}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
