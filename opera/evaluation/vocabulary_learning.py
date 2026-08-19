"""Diagnostics for whether BONSAI learns useful vocabulary representations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from opera.evaluation.treatment_embeddings import embedding_columns


def count_token_exposure(
    subjects: Iterable[Mapping], id_to_token: Mapping[int, str]
) -> pd.DataFrame:
    """Count token occurrences and subjects containing each token."""
    occurrences: dict[int, int] = {}
    subject_counts: dict[int, int] = {}
    n_subjects = 0
    for subject in subjects:
        n_subjects += 1
        codes = subject["code"].detach().cpu().numpy().astype(int, copy=False)
        ids, counts = np.unique(codes, return_counts=True)
        for token_id, count in zip(ids.tolist(), counts.tolist()):
            occurrences[token_id] = occurrences.get(token_id, 0) + int(count)
            subject_counts[token_id] = subject_counts.get(token_id, 0) + 1
    rows = []
    for token_id, token in id_to_token.items():
        rows.append(
            {
                "token_id": int(token_id),
                "token": str(token),
                "n_occurrences": int(occurrences.get(token_id, 0)),
                "n_subjects": int(subject_counts.get(token_id, 0)),
                "subject_fraction": (
                    float(subject_counts.get(token_id, 0) / n_subjects)
                    if n_subjects
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def frequency_strata(counts: Sequence[int]) -> pd.Categorical:
    """Stable, interpretable frequency bins including unseen vocabulary rows."""
    values = pd.Series(counts).fillna(0).astype(int)
    edges = [-1, 0, 1, 4, 9, 49, 99, 499, 999, np.inf]
    labels = [
        "unseen",
        "1",
        "2-4",
        "5-9",
        "10-49",
        "50-99",
        "100-499",
        "500-999",
        "1000+",
    ]
    return pd.cut(values, bins=edges, labels=labels, ordered=True)


def vocabulary_coverage(exposure: pd.DataFrame, thresholds: Sequence[int]) -> pd.DataFrame:
    """Quantify token and event coverage retained at candidate frequency cutoffs."""
    total_events = int(exposure["n_occurrences"].sum())
    rows = []
    for threshold in thresholds:
        retained = exposure["n_occurrences"] >= int(threshold)
        retained_events = int(exposure.loc[retained, "n_occurrences"].sum())
        rows.append(
            {
                "minimum_occurrences": int(threshold),
                "retained_tokens": int(retained.sum()),
                "removed_tokens": int((~retained).sum()),
                "retained_token_fraction": float(retained.mean()),
                "retained_events": retained_events,
                "retained_event_fraction": (
                    float(retained_events / total_events) if total_events else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def stratified_token_sample(
    geometry: pd.DataFrame, *, max_tokens: int, seed: int = 42
) -> pd.DataFrame:
    """Bound quadratic neighbour work while retaining every exposure stratum."""
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two.")
    if len(geometry) <= max_tokens:
        return geometry.copy()
    strata = geometry["frequency_stratum"].dropna().unique().tolist()
    per_stratum = max(1, max_tokens // max(1, len(strata)))
    parts = []
    for offset, stratum in enumerate(strata):
        group = geometry[geometry["frequency_stratum"] == stratum]
        parts.append(
            group.sample(
                n=min(per_stratum, len(group)), random_state=int(seed + offset)
            )
        )
    sampled = pd.concat(parts).drop_duplicates("token_id")
    remainder = max_tokens - len(sampled)
    if remainder > 0:
        remaining = geometry[~geometry["token_id"].isin(sampled["token_id"])]
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(
                    n=min(remainder, len(remaining)), random_state=int(seed + 1000)
                ),
            ]
        )
    return sampled.sort_values("token_id").reset_index(drop=True)


def token_geometry(
    trained: pd.DataFrame,
    exposure: pd.DataFrame,
    initial: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join exposure, norms, and optional exact checkpoint-to-checkpoint movement."""
    cols = embedding_columns(trained)
    out = trained.drop(columns=cols).merge(
        exposure, on=["token_id", "token"], how="left", validate="one_to_one"
    )
    out["frequency_stratum"] = frequency_strata(out["n_occurrences"])
    if initial is not None:
        init_cols = embedding_columns(initial)
        if len(cols) != len(init_cols):
            raise ValueError("Initial and trained embedding dimensions differ.")
        left = trained[["token_id", "token", *cols]]
        right = initial[["token_id", "token", *init_cols]]
        merged = left.merge(
            right,
            on=["token_id", "token"],
            suffixes=("_trained", "_initial"),
            validate="one_to_one",
        )
        trained_values = merged[[f"{c}_trained" for c in cols]].to_numpy(float)
        initial_values = merged[[f"{c}_initial" for c in init_cols]].to_numpy(float)
        delta = trained_values - initial_values
        denom = np.linalg.norm(trained_values, axis=1) * np.linalg.norm(
            initial_values, axis=1
        )
        cosine = np.divide(
            np.sum(trained_values * initial_values, axis=1),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        movement = pd.DataFrame(
            {
                "token_id": merged["token_id"],
                "l2_from_initial": np.linalg.norm(delta, axis=1),
                "cosine_from_initial": cosine,
            }
        )
        out = out.merge(movement, on="token_id", validate="one_to_one")
    return out


def all_token_neighbours(frame: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    """Compute neighbours in chunks without materializing an NxN matrix."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    cols = embedding_columns(frame)
    values = frame[cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    rows = []
    chunk_size = 256
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        similarities = values[start:stop] @ values.T
        for local, source_idx in enumerate(range(start, stop)):
            similarities[local, source_idx] = -np.inf
            k = min(top_k, len(frame) - 1)
            candidates = np.argpartition(similarities[local], -k)[-k:]
            candidates = candidates[np.argsort(similarities[local, candidates])[::-1]]
            source = frame.iloc[source_idx]
            for rank, target_idx in enumerate(candidates.tolist(), start=1):
                target = frame.iloc[target_idx]
                rows.append(
                    {
                        "token_id": int(source.token_id),
                        "token": source.token,
                        "token_family": source.token_family,
                        "neighbor_rank": rank,
                        "neighbor_token_id": int(target.token_id),
                        "neighbor_token": target.token,
                        "neighbor_family": target.token_family,
                        "cosine_similarity": float(similarities[local, target_idx]),
                        "same_family": bool(source.token_family == target.token_family),
                    }
                )
    return pd.DataFrame(rows)


def neighbour_coherence(
    neighbours: pd.DataFrame, geometry: pd.DataFrame
) -> pd.DataFrame:
    """Summarize broad-family precision@k within exposure strata."""
    annotated = neighbours.merge(
        geometry[["token_id", "n_occurrences", "frequency_stratum"]],
        on="token_id",
        how="left",
        validate="many_to_one",
    )
    return (
        annotated.groupby("frequency_stratum", observed=False)
        .agg(
            n_tokens=("token_id", "nunique"),
            mean_occurrences=("n_occurrences", "mean"),
            same_family_precision_at_k=("same_family", "mean"),
            mean_neighbor_cosine=("cosine_similarity", "mean"),
        )
        .reset_index()
    )


def neighbour_permutation_null(
    neighbours: pd.DataFrame,
    geometry: pd.DataFrame,
    *,
    n_permutations: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Frequency-stratified null for broad-family neighbour coherence."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    token_ids = pd.Index(
        pd.unique(pd.concat([neighbours["token_id"], neighbours["neighbor_token_id"]]))
    )
    labels = geometry.set_index("token_id").reindex(token_ids)[
        ["token_family", "frequency_stratum"]
    ]
    observed = float(neighbours["same_family"].mean())
    rng = np.random.default_rng(seed)
    null = []
    source = neighbours["token_id"].map({token: i for i, token in enumerate(token_ids)}).to_numpy()
    target = neighbours["neighbor_token_id"].map({token: i for i, token in enumerate(token_ids)}).to_numpy()
    original = labels["token_family"].astype(str).to_numpy()
    strata = labels["frequency_stratum"].astype(str).to_numpy()
    for _ in range(n_permutations):
        permuted = original.copy()
        for stratum in np.unique(strata):
            idx = np.flatnonzero(strata == stratum)
            permuted[idx] = rng.permutation(permuted[idx])
        null.append(float(np.mean(permuted[source] == permuted[target])))
    null_values = np.asarray(null)
    std = float(null_values.std(ddof=1)) if len(null_values) > 1 else np.nan
    return pd.DataFrame([{
        "n_edges": len(neighbours),
        "n_permutations": n_permutations,
        "observed_same_family_precision": observed,
        "null_mean": float(null_values.mean()),
        "null_std": std,
        "excess_over_null": observed - float(null_values.mean()),
        "z_score": (observed - float(null_values.mean())) / std if std > 0 else np.nan,
        "permutation_p_upper": float((1 + np.sum(null_values >= observed)) / (n_permutations + 1)),
    }])
