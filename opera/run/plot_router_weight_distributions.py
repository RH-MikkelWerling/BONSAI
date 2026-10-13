"""Patient-level tuning route-weight distributions for every named source.

One page per disease/outcome task and source bank. The small transparent dots
are individual patients, not bootstrap draws or uncertainty in a mean.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from opera.run.compare_router_cell_routes import route_matrix
from opera.run.plot_router_cell_borrowing import (
    parse_cell, read_cell_routes, verify_route_provenance,
)
from opera.visualization.outcome_names import outcome_display_name
from opera.visualization.style import PALETTE, despine, save_fig, setup_style


setup_style()


def source_distribution_summary(
    weights: np.ndarray, names: list[str], *, home: str, bank: str,
) -> pd.DataFrame:
    """Describe raw gate weight for every source, including the home source."""
    if weights.ndim != 2 or weights.shape[1] != len(names) or not len(weights):
        raise ValueError("Weights must have one row per patient and one column per source")
    if bank not in {"disease", "outcome"} or home not in names or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid source inventory or route weights")
    rows = []
    for index, name in enumerate(names):
        values = weights[:, index]
        rows.append({
            "source": name,
            "display_name": outcome_display_name(name) if bank == "outcome" else name,
            "is_home": name == home,
            "n_patients": len(values),
            "mean_weight": float(values.mean()),
            "median_weight": float(np.median(values)),
            "p10_weight": float(np.quantile(values, .1)),
            "p90_weight": float(np.quantile(values, .9)),
            "min_weight": float(values.min()),
            "max_weight": float(values.max()),
            "sd_weight": float(values.std(ddof=0)),
        })
    return pd.DataFrame(rows)


def plot_bank(
    weights: np.ndarray, names: list[str], *, bank: str,
    disease: str, outcome: str, output: Path,
) -> pd.DataFrame:
    if bank not in {"disease", "outcome"}:
        raise ValueError("Bank must be disease or outcome")
    home = disease if bank == "disease" else outcome
    summary = source_distribution_summary(weights, names, home=home, bank=bank)
    order = sorted(range(len(names)),
                   key=lambda i: (names[i] != home, -summary.iloc[i]["mean_weight"], names[i]))
    height = max(4.2, .34 * len(names) + 1.75)
    fig, ax = plt.subplots(figsize=(9.6, height))
    color = "#3B8C88" if bank == "disease" else "#C36B55"
    own_color = PALETTE["opera"]
    rng = np.random.default_rng(42)
    for position, index in enumerate(order):
        values = weights[:, index] * 100
        is_home = names[index] == home
        point_color = own_color if is_home else color
        jitter = rng.uniform(-.22, .22, size=len(values))
        ax.scatter(values, position + jitter, s=5, alpha=.095,
                   color=point_color, edgecolors="none", rasterized=True, zorder=2)
        low, median, high = np.quantile(values, [.1, .5, .9])
        ax.hlines(position, low, high, color=point_color, linewidth=1.7, zorder=3)
        ax.scatter([median], [position], marker="D", s=23, color=point_color,
                   edgecolors="white", linewidths=.6, zorder=4)
    even_share = weights.sum(axis=1).mean() * 100 / len(names)
    ax.axvline(even_share, color=PALETTE["ink_muted"], linewidth=.9,
               linestyle=(0, (2, 2)), zorder=1)
    labels = [textwrap.fill(outcome_display_name(names[i]), 31,
                            break_long_words=False) if bank == "outcome" else names[i]
              for i in order]
    ax.set_yticks(range(len(order)), labels, fontsize=7.8 if bank == "outcome" else 8.5)
    ax.set_ylim(len(names) - .5, -.5)
    ax.set_xlim(0, max(float(weights.max() * 100) * 1.07, even_share * 1.1, .1))
    ax.set_xlabel("Route weight for this source (% of total)")
    ax.set_title(f"{bank.capitalize()} sources  ·  {disease} → {outcome_display_name(outcome)}",
                 loc="left", fontsize=12, color=PALETTE["ink"])
    despine(ax, "x")
    fig.subplots_adjust(left=.38 if bank == "outcome" else .25, right=.97,
                        top=.94, bottom=.05 if len(names) > 12 else .13)
    fig.text(.02, .012,
             f"{len(weights):,} tuning patients. Faint dots: patients; diamond: median; line: 10th–90th percentile. "
             "Dotted rule: even split within this bank. Home source shown first. Gate weight is not predictive benefit.",
             fontsize=7.5, color=PALETTE["ink_secondary"])
    save_fig(fig, str(output), dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, action="append", required=True)
    parser.add_argument("--cell", action="append", required=True,
                        help="FINE_DISEASE:outcome; repeat once per route export")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-patients", type=int, default=20)
    args = parser.parse_args()
    if len(args.cell) != len(args.routes):
        parser.error("Supply one --routes path per --cell")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["competing_risk"].get("readout_mode") not in {"source_anchored_adapters", "unanchored_adapters"}:
        raise ValueError("Distribution plots require a source router")
    disease_names = sorted(config["competing_risk"]["fine_names"])
    outcome_names = sorted(config["outcomes"])
    names = ["shared", *("disease:" + name for name in disease_names),
             *("outcome:" + name for name in outcome_names)]
    validated = []
    for label, path in zip(args.cell, args.routes):
        disease, outcome, fine_index = parse_cell(label, config)
        if not re.fullmatch(r"[A-Za-z0-9_]+", disease) or not re.fullmatch(r"[A-Za-z0-9_]+", outcome):
            raise ValueError("Cell identifiers must be safe path components")
        verify_route_provenance(path, args.config, args.checkpoint)
        frame = read_cell_routes(path, fine_index=fine_index, outcome=outcome)
        if len(frame) < args.min_patients or frame["subject_id"].duplicated().any():
            raise ValueError(f"Cell {label} has too few or duplicate tuning patients")
        matrix = route_matrix(frame, names)
        validated.append((disease, outcome, path, matrix))
    args.output_dir.mkdir(parents=True)
    records = []
    for disease, outcome, path, matrix in validated:
        cell_dir = args.output_dir / f"{disease}__{outcome}"
        cell_dir.mkdir()
        for bank, bank_names, start in (
            ("disease", disease_names, 1),
            ("outcome", outcome_names, 1 + len(disease_names)),
        ):
            values = matrix[:, start:start + len(bank_names)]
            summary = plot_bank(values, bank_names, bank=bank, disease=disease,
                                outcome=outcome, output=cell_dir / f"{bank}_sources.png")
            summary.insert(0, "bank", bank)
            summary.insert(0, "target_outcome", outcome)
            summary.insert(0, "disease", disease)
            records.extend(summary.to_dict("records"))
    pd.DataFrame(records).to_csv(args.output_dir / "source_weight_distributions.csv", index=False)
    (args.output_dir / "provenance_INTERNAL.json").write_text(json.dumps({
        "split": "tuning", "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "routes": [str(path.resolve()) for path in args.routes],
        "cells": args.cell,
        "estimand": "Raw source gate-weight distribution among eligible tuning patients "
                    "in each fine-disease and target-outcome cell. Every patient contributes one dot "
                    "per source; no patient identifiers are included in the artwork.",
        "caveat": "Gate weight is not predictive contribution or a causal borrowing effect.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "tuning_patient_route_distributions",
                      "cells": len(validated), "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
