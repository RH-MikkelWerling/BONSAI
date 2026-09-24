"""Discover outcome trunks from training-only representation-gradient affinity.

This module consumes the batch artifact written by
``representation_gradient_conflict``.  Known label-construction dependencies
(nested grades, composites, and aliases) are kept together, while all other
assignments are driven by repeated-batch gradient agreement.  Dependency pairs
are excluded when scoring a partition so deterministic overlap cannot make a
candidate look better than it is.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from opera.diagnostics.endpoint_dependencies import (
    classify_endpoint_pairs,
    dependency_config,
)


REQUIRED_COLUMNS = {
    "batch_index",
    "outcome_a",
    "outcome_b",
    "cosine",
    "joint_support",
}
MUST_LINK_RELATIONSHIPS = {
    "nested_threshold_control",
    "component_composite",
    "alternate_definition",
}


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _validate_batches(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError("Gradient artifact is missing columns: " + ", ".join(missing))
    data = frame.copy()
    if "gradient_component" in data:
        data = data[data["gradient_component"].astype(str) == component].copy()
        if data.empty:
            raise ValueError(f"No rows found for gradient component {component!r}.")
    for column in ("cosine", "joint_support"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "excess_alignment" in data:
        data["excess_alignment"] = pd.to_numeric(
            data["excess_alignment"], errors="coerce"
        )
    data = data.dropna(subset=["outcome_a", "outcome_b", "batch_index"])
    ordered = np.sort(data[["outcome_a", "outcome_b"]].astype(str).to_numpy(), axis=1)
    data["outcome_a"], data["outcome_b"] = ordered[:, 0], ordered[:, 1]
    return data[data["outcome_a"] != data["outcome_b"]].reset_index(drop=True)


def _relationship_frame(
    outcomes: Sequence[str], dependency_cfg: Mapping[str, Any] | None
) -> pd.DataFrame:
    composites, aliases = dependency_config(dependency_cfg)
    return pd.DataFrame(
        classify_endpoint_pairs(outcomes, composites=composites, aliases=aliases)
    )


def _must_link_components(
    outcomes: Sequence[str], relationships: pd.DataFrame
) -> list[tuple[str, ...]]:
    union = _UnionFind(outcomes)
    constrained = relationships[
        relationships["relationship"].isin(MUST_LINK_RELATIONSHIPS)
    ]
    for row in constrained.itertuples(index=False):
        union.union(str(row.outcome_a), str(row.outcome_b))
    groups: dict[str, list[str]] = {}
    for outcome in outcomes:
        groups.setdefault(union.find(outcome), []).append(outcome)
    return sorted((tuple(sorted(group)) for group in groups.values()), key=lambda x: x)


def summarize_affinity(
    pair_batches: pd.DataFrame,
    *,
    component: str = "full_likelihood",
    min_support: float = 8.0,
    prefer_null_corrected: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return an outcome affinity matrix and auditable pair summary."""
    data = _validate_batches(pair_batches, component)
    value_column = "cosine"
    if (
        prefer_null_corrected
        and "excess_alignment" in data
        and data["excess_alignment"].notna().any()
    ):
        value_column = "excess_alignment"
    outcomes = sorted(set(data["outcome_a"]) | set(data["outcome_b"]))
    data = data[(data["joint_support"] >= min_support) & data[value_column].notna()].copy()
    if data.empty:
        raise ValueError(
            f"No usable gradient pairs remain at min_support={min_support}."
        )
    grouped = (
        data.groupby(["outcome_a", "outcome_b"], as_index=False)
        .agg(
            affinity=(value_column, "mean"),
            affinity_std=(value_column, "std"),
            mean_joint_support=("joint_support", "mean"),
            n_batches=("batch_index", "nunique"),
        )
        .sort_values(["outcome_a", "outcome_b"])
    )
    matrix = pd.DataFrame(np.nan, index=outcomes, columns=outcomes, dtype=float)
    for outcome in outcomes:
        matrix.loc[outcome, outcome] = 1.0
    for row in grouped.itertuples(index=False):
        matrix.loc[row.outcome_a, row.outcome_b] = row.affinity
        matrix.loc[row.outcome_b, row.outcome_a] = row.affinity
    return matrix, grouped, value_column


def _component_affinity(
    first: Sequence[str], second: Sequence[str], affinity: pd.DataFrame
) -> float:
    values = affinity.loc[list(first), list(second)].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def _cluster_affinity(
    component: Sequence[str], cluster: Sequence[Sequence[str]], affinity: pd.DataFrame
) -> float:
    if not cluster:
        return 0.0
    return _component_affinity(component, [x for group in cluster for x in group], affinity)


def _balanced_partition_once(
    components: Sequence[tuple[str, ...]],
    affinity: pd.DataFrame,
    *,
    n_clusters: int,
    min_size: int,
    max_size: int,
    rng: np.random.Generator,
) -> list[list[tuple[str, ...]]] | None:
    """Greedy, capacity-constrained partition of indivisible must-link groups."""
    if any(len(component) > max_size for component in components):
        return None
    shuffled = list(components)
    rng.shuffle(shuffled)
    # Large dependency components and poorly connected outcomes are placed first.
    global_mean = {
        component: np.nanmean(
            affinity.loc[list(component)].to_numpy(dtype=float)
        )
        for component in shuffled
    }
    ordered = sorted(
        shuffled,
        key=lambda item: (-len(item), global_mean[item] if np.isfinite(global_mean[item]) else 0),
    )

    seeds = [ordered.pop(0)]
    while len(seeds) < n_clusters:
        candidate = min(
            ordered,
            key=lambda item: max(
                _component_affinity(item, seed, affinity) for seed in seeds
            ),
        )
        seeds.append(candidate)
        ordered.remove(candidate)
    clusters: list[list[tuple[str, ...]]] = [[seed] for seed in seeds]

    for component in ordered:
        feasible = [
            index
            for index, cluster in enumerate(clusters)
            if sum(map(len, cluster)) + len(component) <= max_size
        ]
        if not feasible:
            return None
        # The small balance term only breaks near-ties in affinity.
        chosen = max(
            feasible,
            key=lambda index: (
                _cluster_affinity(component, clusters[index], affinity)
                - 0.015 * sum(map(len, clusters[index])),
                -sum(map(len, clusters[index])),
                -index,
            ),
        )
        clusters[chosen].append(component)

    # Repair undersized clusters by moving the least damaging dependency component.
    while True:
        sizes = [sum(map(len, cluster)) for cluster in clusters]
        undersized = [i for i, size in enumerate(sizes) if size < min_size]
        if not undersized:
            return clusters
        target = min(undersized, key=lambda i: sizes[i])
        moves = []
        for donor, cluster in enumerate(clusters):
            if donor == target:
                continue
            for component in cluster:
                weight = len(component)
                if sizes[donor] - weight < min_size or sizes[target] + weight > max_size:
                    continue
                gain = _cluster_affinity(component, clusters[target], affinity)
                loss = _cluster_affinity(
                    component, [x for x in cluster if x != component], affinity
                )
                moves.append((gain - loss, donor, component))
        if not moves:
            return None
        _, donor, component = max(moves, key=lambda item: item[0])
        clusters[donor].remove(component)
        clusters[target].append(component)


def _pair_score(
    clusters: Sequence[Sequence[tuple[str, ...]]],
    pair_summary: pd.DataFrame,
    relationships: pd.DataFrame,
) -> dict[str, float]:
    membership = {}
    for index, cluster in enumerate(clusters):
        for component in cluster:
            for outcome in component:
                membership[outcome] = index
    evidence = pair_summary.merge(
        relationships[["outcome_a", "outcome_b", "scaffold_eligible"]],
        on=["outcome_a", "outcome_b"],
        how="left",
    )
    evidence = evidence[evidence["scaffold_eligible"].fillna(True)].copy()
    evidence["within"] = [
        membership.get(a) == membership.get(b)
        for a, b in evidence[["outcome_a", "outcome_b"]].itertuples(index=False)
    ]
    within = evidence.loc[evidence["within"], "affinity"].to_numpy(dtype=float)
    between = evidence.loc[~evidence["within"], "affinity"].to_numpy(dtype=float)
    within_mean = float(np.nanmean(within)) if within.size else float("nan")
    between_mean = float(np.nanmean(between)) if between.size else float("nan")
    contrast = within_mean - between_mean
    return {
        "within_atomic_affinity": within_mean,
        "between_atomic_affinity": between_mean,
        "affinity_contrast": contrast,
        "n_within_atomic_pairs": int(within.size),
        "n_between_atomic_pairs": int(between.size),
    }


def discover_partition(
    affinity: pd.DataFrame,
    pair_summary: pd.DataFrame,
    relationships: pd.DataFrame,
    *,
    min_size: int = 5,
    max_size: int = 10,
    n_clusters: int | None = None,
    attempts: int = 64,
    seed: int = 17,
) -> tuple[list[list[tuple[str, ...]]], pd.DataFrame]:
    outcomes = list(affinity.index)
    components = _must_link_components(outcomes, relationships)
    minimum_k = math.ceil(len(outcomes) / max_size)
    maximum_k = len(outcomes) // min_size
    if minimum_k > maximum_k:
        raise ValueError("No cluster count can satisfy the requested size bounds.")
    candidate_ks = [n_clusters] if n_clusters is not None else list(range(minimum_k, maximum_k + 1))
    if any(k is None or not minimum_k <= k <= maximum_k for k in candidate_ks):
        raise ValueError(
            f"n_clusters must be between {minimum_k} and {maximum_k} for "
            f"{len(outcomes)} outcomes."
        )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    partitions: dict[int, list[list[tuple[str, ...]]]] = {}
    for k in candidate_ks:
        best = None
        best_score = -np.inf
        successes = 0
        for _ in range(attempts):
            partition = _balanced_partition_once(
                components,
                affinity,
                n_clusters=int(k),
                min_size=min_size,
                max_size=max_size,
                rng=rng,
            )
            if partition is None:
                continue
            successes += 1
            score = _pair_score(partition, pair_summary, relationships)
            if score["affinity_contrast"] > best_score:
                best, best_score = partition, score["affinity_contrast"]
        if best is None:
            continue
        partitions[int(k)] = best
        sizes = [sum(map(len, cluster)) for cluster in best]
        rows.append(
            {
                "n_clusters": int(k),
                "min_cluster_size": min(sizes),
                "max_cluster_size": max(sizes),
                "successful_attempts": successes,
                **_pair_score(best, pair_summary, relationships),
            }
        )
    if not rows:
        raise RuntimeError("Could not construct a valid capacity-constrained partition.")
    candidates = pd.DataFrame(rows).sort_values(
        ["affinity_contrast", "n_clusters"], ascending=[False, True]
    )
    # K is a structural choice, not something to maximize on the same noisy
    # affinities used for assignment.  Unless prespecified, choose the feasible
    # count closest to the midpoint of the requested 5--10 style size range;
    # retain the affinity ranking for sensitivity analysis.
    target_k = int(round(len(outcomes) / ((min_size + max_size) / 2.0)))
    target_k = min(max(target_k, minimum_k), maximum_k)
    selected_k = (
        int(n_clusters)
        if n_clusters is not None
        else int(
            min(
                partitions,
                key=lambda k: (abs(k - target_k), -float(
                    candidates.loc[
                        candidates["n_clusters"] == k, "affinity_contrast"
                    ].iloc[0]
                )),
            )
        )
    )
    candidates["affinity_rank"] = candidates["affinity_contrast"].rank(
        method="min", ascending=False
    ).astype(int)
    candidates["selected"] = candidates["n_clusters"] == selected_k
    return partitions[selected_k], candidates


def _resampled_affinity(
    data: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    value_column: str,
    min_support: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    batch_ids = data["batch_index"].drop_duplicates().to_numpy()
    sampled = rng.choice(batch_ids, size=len(batch_ids), replace=True)
    pieces = []
    for replicate, batch_id in enumerate(sampled):
        block = data[data["batch_index"] == batch_id].copy()
        block["_replicate"] = replicate
        pieces.append(block)
    boot = pd.concat(pieces, ignore_index=True)
    boot = boot[(boot["joint_support"] >= min_support) & boot[value_column].notna()]
    means = boot.groupby(["outcome_a", "outcome_b"])[value_column].mean()
    matrix = pd.DataFrame(np.nan, index=outcomes, columns=outcomes, dtype=float)
    for outcome in outcomes:
        matrix.loc[outcome, outcome] = 1.0
    for (first, second), value in means.items():
        matrix.loc[first, second] = value
        matrix.loc[second, first] = value
    return matrix


def bootstrap_coassignment(
    pair_batches: pd.DataFrame,
    relationships: pd.DataFrame,
    reference_partition: Sequence[Sequence[tuple[str, ...]]],
    *,
    component: str,
    value_column: str,
    min_support: float,
    min_size: int,
    max_size: int,
    n_bootstrap: int,
    attempts: int,
    seed: int,
) -> tuple[pd.DataFrame, float, int]:
    data = _validate_batches(pair_batches, component)
    outcomes = sorted(
        {x for cluster in reference_partition for group in cluster for x in group}
    )
    components = _must_link_components(outcomes, relationships)
    n_clusters = len(reference_partition)
    counts = pd.DataFrame(0.0, index=outcomes, columns=outcomes)
    rng = np.random.default_rng(seed)
    completed = 0
    for _ in range(n_bootstrap):
        matrix = _resampled_affinity(
            data,
            outcomes,
            value_column=value_column,
            min_support=min_support,
            rng=rng,
        )
        partition = None
        best_score = -np.inf
        # Pair summary is unnecessary here; maximize mean observed within affinity.
        for _ in range(attempts):
            trial = _balanced_partition_once(
                components,
                matrix,
                n_clusters=n_clusters,
                min_size=min_size,
                max_size=max_size,
                rng=rng,
            )
            if trial is None:
                continue
            values = []
            for cluster in trial:
                names = [x for group in cluster for x in group]
                for i, first in enumerate(names):
                    for second in names[i + 1 :]:
                        value = matrix.loc[first, second]
                        if np.isfinite(value):
                            values.append(value)
            score = float(np.mean(values)) if values else -np.inf
            if score > best_score:
                partition, best_score = trial, score
        if partition is None:
            continue
        completed += 1
        for cluster in partition:
            names = [x for group in cluster for x in group]
            counts.loc[names, names] += 1.0
    if completed == 0:
        raise RuntimeError("No bootstrap replicate produced a valid partition.")
    coassignment = counts / completed
    within = []
    for cluster in reference_partition:
        names = [x for group in cluster for x in group]
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                within.append(coassignment.loc[first, second])
    return coassignment, float(np.mean(within)) if within else float("nan"), completed


def _membership_frame(
    partition: Sequence[Sequence[tuple[str, ...]]],
    coassignment: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cluster_index, cluster in enumerate(partition, start=1):
        names = sorted(x for group in cluster for x in group)
        for outcome in names:
            peers = [name for name in names if name != outcome]
            rows.append(
                {
                    "trunk": f"learned_trunk_{cluster_index:02d}",
                    "outcome": outcome,
                    "trunk_size": len(names),
                    "must_link_component": "|".join(
                        next(group for group in cluster if outcome in group)
                    ),
                    "mean_bootstrap_coassignment_with_trunk": (
                        float(coassignment.loc[outcome, peers].mean()) if peers else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_ordered_matrix(matrix: pd.DataFrame, membership: pd.DataFrame, path: Path) -> None:
    order = membership.sort_values(["trunk", "outcome"])["outcome"].tolist()
    ordered = matrix.loc[order, order]
    size = max(9.0, min(24.0, 0.28 * len(order)))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(ordered, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(order)), labels=order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), labels=order, fontsize=6)
    boundaries = membership.groupby("trunk").size().cumsum().to_numpy()[:-1] - 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=0.7)
        ax.axvline(boundary, color="black", linewidth=0.7)
    ax.set_title("Training gradient affinity ordered by learned trunk")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_trunk_discovery(
    pair_batches: pd.DataFrame,
    output_dir: Path,
    *,
    dependency_cfg: Mapping[str, Any] | None = None,
    component: str = "full_likelihood",
    min_support: float = 8.0,
    min_size: int = 5,
    max_size: int = 10,
    n_clusters: int | None = None,
    attempts: int = 64,
    n_bootstrap: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    """Run trunk discovery and write a directly reusable family config."""
    if min_size < 2 or max_size < min_size:
        raise ValueError("Require 2 <= min_size <= max_size.")
    if attempts < 1 or n_bootstrap < 1:
        raise ValueError("attempts and n_bootstrap must be positive.")
    affinity, pair_summary, value_column = summarize_affinity(
        pair_batches,
        component=component,
        min_support=min_support,
    )
    relationships = _relationship_frame(list(affinity.index), dependency_cfg)
    partition, candidates = discover_partition(
        affinity,
        pair_summary,
        relationships,
        min_size=min_size,
        max_size=max_size,
        n_clusters=n_clusters,
        attempts=attempts,
        seed=seed,
    )
    coassignment, stability, completed = bootstrap_coassignment(
        pair_batches,
        relationships,
        partition,
        component=component,
        value_column=value_column,
        min_support=min_support,
        min_size=min_size,
        max_size=max_size,
        n_bootstrap=n_bootstrap,
        attempts=max(8, attempts // 4),
        seed=seed + 1,
    )
    membership = _membership_frame(partition, coassignment)
    output_dir.mkdir(parents=True, exist_ok=True)
    affinity.to_csv(output_dir / "outcome_gradient_affinity.csv")
    pair_summary.merge(
        relationships, on=["outcome_a", "outcome_b"], how="left"
    ).to_csv(output_dir / "outcome_pair_evidence.csv", index=False)
    candidates.to_csv(output_dir / "candidate_cluster_counts.csv", index=False)
    coassignment.to_csv(output_dir / "bootstrap_coassignment.csv")
    membership.to_csv(output_dir / "learned_trunk_membership.csv", index=False)
    families = {
        trunk: sorted(group["outcome"].tolist())
        for trunk, group in membership.groupby("trunk", sort=True)
    }
    (output_dir / "learned_outcome_families.yaml").write_text(
        yaml.safe_dump({"outcome_families": families}, sort_keys=False),
        encoding="utf-8",
    )
    _plot_ordered_matrix(
        affinity, membership, output_dir / "learned_trunk_affinity_heatmap.png"
    )
    summary = {
        "warning": (
            "Discovery evidence must come only from training patients. Freeze the "
            "selected grouping before tuning/held-out evaluation."
        ),
        "gradient_component": component,
        "affinity_value": value_column,
        "null_corrected": value_column == "excess_alignment",
        "n_outcomes": len(affinity),
        "n_trunks": len(partition),
        "min_size": min_size,
        "max_size": max_size,
        "must_link_relationships": sorted(MUST_LINK_RELATIONSHIPS),
        "bootstrap_requested": n_bootstrap,
        "bootstrap_completed": completed,
        "mean_within_trunk_coassignment": stability,
        "selected_candidate": candidates[candidates["selected"]].iloc[0].to_dict(),
        "outcome_families": families,
    }
    (output_dir / "trunk_discovery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover size-constrained outcome trunks from gradient affinity."
    )
    parser.add_argument("--pair-batches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dependency-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "endpoint_dependencies.yaml",
    )
    parser.add_argument("--component", default="full_likelihood")
    parser.add_argument("--min-support", type=float, default=8.0)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--attempts", type=int, default=64)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dependency_cfg = yaml.safe_load(args.dependency_config.read_text(encoding="utf-8"))
    summary = run_trunk_discovery(
        pd.read_csv(args.pair_batches),
        args.output_dir,
        dependency_cfg=dependency_cfg,
        component=args.component,
        min_support=args.min_support,
        min_size=args.min_size,
        max_size=args.max_size,
        n_clusters=args.n_clusters,
        attempts=args.attempts,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(
        f"Selected {summary['n_trunks']} trunks across {summary['n_outcomes']} outcomes; "
        f"bootstrap within-trunk stability={summary['mean_within_trunk_coassignment']:.3f}."
    )
    print(f"Reusable config: {args.output_dir / 'learned_outcome_families.yaml'}")


if __name__ == "__main__":
    main()
