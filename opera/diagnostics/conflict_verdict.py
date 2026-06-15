"""Classify representation-gradient conflicts using batch-level uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.visualization.style import CATEGORICAL, FIG_FULL, save_fig, setup_style


PAIR_COLUMNS = {
    "batch_index",
    "outcome_a",
    "outcome_b",
    "cosine",
    "joint_support",
}


def _validate_pair_batches(pair_batches: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(PAIR_COLUMNS.difference(pair_batches.columns))
    if missing:
        raise ValueError(
            "Batch pair artifact is missing required columns: " + ", ".join(missing)
        )
    frame = pair_batches.copy()
    frame["cosine"] = pd.to_numeric(frame["cosine"], errors="coerce")
    frame["joint_support"] = pd.to_numeric(
        frame["joint_support"],
        errors="coerce",
    )
    if frame["joint_support"].isna().any():
        raise ValueError("joint_support must be numeric for every batch-pair row.")
    if (frame["joint_support"] < 0).any():
        raise ValueError("joint_support cannot be negative.")
    ordered = np.sort(
        frame[["outcome_a", "outcome_b"]].astype(str).to_numpy(),
        axis=1,
    )
    frame["outcome_a"] = ordered[:, 0]
    frame["outcome_b"] = ordered[:, 1]
    frame = frame[frame["outcome_a"] != frame["outcome_b"]].copy()
    return frame.reset_index(drop=True)


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a percentile interval for a one-dimensional mean."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan"), float("nan")
    samples = rng.choice(
        finite,
        size=(n_bootstrap, finite.size),
        replace=True,
    )
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def classify_gradient_pairs(
    pair_batches: pd.DataFrame,
    *,
    min_mean_support: float = 8.0,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> pd.DataFrame:
    """Summarize each outcome pair and classify its cosine interval."""
    if min_mean_support < 0:
        raise ValueError("min_mean_support cannot be negative.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")

    frame = _validate_pair_batches(pair_batches)
    rng = np.random.default_rng(seed)
    rows = []
    for (outcome_a, outcome_b), group in frame.groupby(
        ["outcome_a", "outcome_b"],
        sort=True,
    ):
        cosines = group["cosine"].dropna().to_numpy(dtype=float)
        mean_support = float(group["joint_support"].mean())
        adequate_support = bool(mean_support >= min_mean_support)
        lower, upper = bootstrap_mean_interval(
            cosines,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            rng=rng,
        )
        classification = "indeterminate"
        if adequate_support and np.isfinite(lower) and np.isfinite(upper):
            if upper < 0.0:
                classification = "genuine_conflict"
            elif lower > 0.0:
                classification = "genuine_alignment"
        rows.append(
            {
                "outcome_a": outcome_a,
                "outcome_b": outcome_b,
                "mean_cosine": (
                    float(np.mean(cosines)) if cosines.size else float("nan")
                ),
                "ci_lower": lower,
                "ci_upper": upper,
                "mean_joint_support": mean_support,
                "n_batches": int(group["batch_index"].nunique()),
                "n_cosine_batches": int(cosines.size),
                "adequate_support": adequate_support,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows)


def _connected_components(edges: list[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for first, second in edges:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    components = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, set()).difference(component))
        components.append(component)
        unseen.difference_update(component)
    return sorted(components, key=lambda item: (-len(item), sorted(item)))


def summarize_conflict_structure(pair_summary: pd.DataFrame) -> dict[str, Any]:
    """Describe whether significant conflict edges form a dense component."""
    conflicts = pair_summary[
        pair_summary["classification"] == "genuine_conflict"
    ].copy()
    edges = list(
        conflicts[["outcome_a", "outcome_b"]].itertuples(index=False, name=None)
    )
    components = _connected_components(edges)
    records = []
    for component in components:
        component_edges = [
            edge for edge in edges if edge[0] in component and edge[1] in component
        ]
        possible_edges = len(component) * (len(component) - 1) / 2
        records.append(
            {
                "outcomes": sorted(component),
                "n_outcomes": len(component),
                "n_conflict_edges": len(component_edges),
                "edge_density": (
                    float(len(component_edges) / possible_edges)
                    if possible_edges
                    else 0.0
                ),
            }
        )

    largest = records[0] if records else None
    captures = (
        float(largest["n_conflict_edges"] / len(edges))
        if largest is not None and edges
        else 0.0
    )
    coherent = bool(
        largest is not None
        and largest["n_outcomes"] >= 3
        and largest["edge_density"] >= 0.5
        and captures >= 0.5
    )
    return {
        "n_conflict_edges": len(edges),
        "n_components": len(records),
        "components": records,
        "largest_component_edge_fraction": captures,
        "coherent_conflict_group": coherent,
    }


def build_surgery_verdict(
    pair_summary: pd.DataFrame,
    conflict_structure: dict[str, Any],
    *,
    surgery_conflict_fraction: float = 0.1,
) -> tuple[str, dict[str, Any]]:
    """Produce a one-sentence decision and machine-readable class fractions."""
    adequate = pair_summary[pair_summary["adequate_support"]].copy()
    counts = adequate["classification"].value_counts()
    denominator = len(adequate)
    fractions = {
        name: float(counts.get(name, 0) / denominator) if denominator else None
        for name in (
            "genuine_conflict",
            "genuine_alignment",
            "indeterminate",
        )
    }
    if denominator == 0:
        verdict = (
            "Gradient-surgery need is indeterminate because no outcome pair "
            "has adequate shared support."
        )
    else:
        conflict_fraction = fractions["genuine_conflict"] or 0.0
        coherent = bool(conflict_structure["coherent_conflict_group"])
        if conflict_fraction >= surgery_conflict_fraction or coherent:
            verdict = (
                "Gradient surgery is warranted for follow-up because "
                f"{conflict_fraction:.1%} of adequately supported pairs show "
                "genuine conflict"
                + (
                    " in a coherent outcome group."
                    if coherent
                    else " across scattered outcome pairs."
                )
            )
        else:
            verdict = (
                "Gradient surgery is not warranted by this diagnostic because "
                f"only {conflict_fraction:.1%} of adequately supported pairs "
                "show genuine conflict and no coherent conflict group is present."
            )
    summary = {
        "n_adequately_supported_pairs": denominator,
        "class_counts": {
            name: int(counts.get(name, 0))
            for name in (
                "genuine_conflict",
                "genuine_alignment",
                "indeterminate",
            )
        },
        "class_fractions": fractions,
        "surgery_conflict_fraction": surgery_conflict_fraction,
    }
    return verdict, summary


def plot_pair_intervals(
    pair_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot sorted batch-bootstrap cosine intervals for all outcome pairs."""
    setup_style()
    plot_data = pair_summary.sort_values("mean_cosine").reset_index(drop=True)
    fig_height = max(3.5, min(12.0, 0.22 * max(len(plot_data), 1)))
    fig, ax = plt.subplots(figsize=(FIG_FULL[0], fig_height))
    if not plot_data.empty:
        color_map = {
            "genuine_conflict": CATEGORICAL[1],
            "genuine_alignment": CATEGORICAL[2],
            "indeterminate": CATEGORICAL[-1],
        }
        y = np.arange(len(plot_data))
        for index, row in plot_data.iterrows():
            color = color_map[row["classification"]]
            if np.isfinite(row["ci_lower"]) and np.isfinite(row["ci_upper"]):
                ax.plot(
                    [row["ci_lower"], row["ci_upper"]],
                    [index, index],
                    color=color,
                    linewidth=1.2,
                )
            ax.scatter(row["mean_cosine"], index, color=color, s=24, zorder=3)
        labels = (
            plot_data["outcome_a"].astype(str)
            + " vs "
            + plot_data["outcome_b"].astype(str)
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Representation-gradient cosine with bootstrap CI")
    ax.set_ylabel("Outcome pair")
    ax.set_title("Cross-outcome gradient conflict")
    fig.tight_layout()
    save_fig(fig, str(output_path))
    plt.close(fig)


def run_conflict_verdict(
    pair_batches: pd.DataFrame,
    output_dir: Path,
    *,
    min_mean_support: float = 8.0,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
    surgery_conflict_fraction: float = 0.1,
) -> dict[str, Any]:
    """Run the complete verdict analysis and write paper-ready artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = classify_gradient_pairs(
        pair_batches,
        min_mean_support=min_mean_support,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    structure = summarize_conflict_structure(pairs)
    verdict, class_summary = build_surgery_verdict(
        pairs,
        structure,
        surgery_conflict_fraction=surgery_conflict_fraction,
    )
    summary = {
        "verdict": verdict,
        "min_mean_support": min_mean_support,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        **class_summary,
        "conflict_structure": structure,
    }
    pairs.to_csv(output_dir / "conflict_pair_verdicts.csv", index=False)
    (output_dir / "conflict_verdict.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "conflict_verdict.txt").write_text(
        verdict + "\n",
        encoding="utf-8",
    )
    plot_pair_intervals(
        pairs,
        output_dir / "conflict_pair_intervals.png",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify batch-level representation-gradient conflicts.",
    )
    parser.add_argument(
        "--pair-batches",
        type=Path,
        required=True,
        help="gradient_pair_batches.csv from the gradient diagnostic",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-mean-support", type=float, default=8.0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--surgery-conflict-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pair_batches = pd.read_csv(args.pair_batches)
    summary = run_conflict_verdict(
        pair_batches,
        args.output_dir,
        min_mean_support=args.min_mean_support,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
        surgery_conflict_fraction=args.surgery_conflict_fraction,
    )
    print(summary["verdict"])


if __name__ == "__main__":
    main()
