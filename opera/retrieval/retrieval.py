"""Patient retrieval and validation utilities for OPERA embeddings."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
SEED = 42

RKKP_SIMILARITY_COLUMNS = [
    "ipi_score",
    "ecog",
    "ldh_ratio",
    "ann_arbor_stage",
    "extranodal_sites",
]


@dataclass
class RetrievalResult:
    """Ranked retrieval output for one query patient."""

    query_patient_id: Any
    neighbors: List[Dict[str, Any]]


class PatientRetriever:
    """Nearest-neighbor retrieval over OPERA embeddings with validation helpers."""

    def __init__(
        self,
        *,
        embeddings_base: Optional[Mapping[Any, np.ndarray]] = None,
        embeddings_dapt: Optional[Mapping[Any, np.ndarray]] = None,
        embeddings_opera: Mapping[Any, np.ndarray],
        outcomes: pd.DataFrame,
        rkkp: Optional[pd.DataFrame],
        disease_cohorts: Mapping[str, Sequence[Any]],
        seed: int = SEED,
        use_faiss: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.embeddings_base = embeddings_base
        self.embeddings_dapt = embeddings_dapt
        self.embeddings_opera = embeddings_opera
        self.outcomes = outcomes.copy()
        self.rkkp = rkkp.copy() if rkkp is not None else None
        self.disease_cohorts = disease_cohorts

        self._embedding_frames = {
            "opera": self._build_embedding_frame(embeddings_opera),
        }
        if embeddings_base is not None:
            self._embedding_frames["base"] = self._build_embedding_frame(embeddings_base)
        if embeddings_dapt is not None:
            self._embedding_frames["dapt"] = self._build_embedding_frame(embeddings_dapt)

        self._embedding_df = self._embedding_frames["opera"]
        self._metadata = self._build_metadata_frame()
        self._normalized_matrix = self._embedding_df.drop(columns=["patient_id"]).to_numpy(dtype=np.float32)
        self._normalized_matrix = _normalize_rows(self._normalized_matrix)
        self._patient_ids = self._embedding_df["patient_id"].tolist()
        self._id_to_index = {patient_id: idx for idx, patient_id in enumerate(self._patient_ids)}
        self._normalized_embeddings = {
            name: _normalize_rows(frame.drop(columns=["patient_id"]).to_numpy(dtype=np.float32))
            for name, frame in self._embedding_frames.items()
        }
        self._embedding_patient_ids = {
            name: frame["patient_id"].tolist()
            for name, frame in self._embedding_frames.items()
        }
        self._embedding_id_to_index = {
            name: {patient_id: idx for idx, patient_id in enumerate(patient_ids)}
            for name, patient_ids in self._embedding_patient_ids.items()
        }
        self._rkkp_standardized = self._build_standardized_rkkp_table()

        self.index_backend = "numpy"
        self.faiss_index = None
        if use_faiss:
            self._try_build_faiss_index()

    def _build_embedding_frame(self, embeddings: Mapping[Any, np.ndarray]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for patient_id, vector in embeddings.items():
            arr = np.asarray(vector, dtype=np.float32).reshape(-1)
            row = {"patient_id": patient_id}
            row.update({f"emb_{i}": float(value) for i, value in enumerate(arr)})
            rows.append(row)
        if not rows:
            raise ValueError("embeddings_opera is empty.")
        return pd.DataFrame(rows)

    def _build_metadata_frame(self) -> pd.DataFrame:
        disease = (
            self.outcomes[["patient_id", "disease_subtype"]]
            .drop_duplicates(subset=["patient_id"])
            .copy()
        )
        outcome_summary = (
            self.outcomes.groupby("patient_id")
            .apply(self._summarize_patient_outcomes)
            .rename("outcomes_summary")
            .reset_index()
        )
        metadata = disease.merge(outcome_summary, on="patient_id", how="left")
        if self.rkkp is not None:
            metadata = metadata.merge(self.rkkp, on="patient_id", how="left")
        return metadata

    def _build_standardized_rkkp_table(self) -> Optional[pd.DataFrame]:
        if self.rkkp is None:
            return None
        missing = [column for column in RKKP_SIMILARITY_COLUMNS if column not in self.rkkp.columns]
        if missing:
            LOGGER.warning(
                "RKKP table is missing required similarity columns: %s",
                ", ".join(missing),
            )
            return None
        frame = self.rkkp[["patient_id", *RKKP_SIMILARITY_COLUMNS]].copy()
        complete = frame.dropna(subset=RKKP_SIMILARITY_COLUMNS).copy()
        if complete.empty:
            return None
        values = complete[RKKP_SIMILARITY_COLUMNS].to_numpy(dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        complete.loc[:, RKKP_SIMILARITY_COLUMNS] = (values - mean) / std
        return complete.reset_index(drop=True)

    def _try_build_faiss_index(self) -> None:
        try:
            import faiss

            matrix = self._normalized_matrix.astype(np.float32, copy=False)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self.faiss_index = index
            self.index_backend = "faiss"
        except Exception as exc:
            LOGGER.info("FAISS unavailable; falling back to numpy retrieval: %s", type(exc).__name__)

    def retrieve_similar(
        self,
        patient_id: Any,
        k: int = 5,
        restrict_to_same_disease: bool = False,
    ) -> List[Tuple[Any, float, Any, Dict[str, Any]]]:
        """Retrieve the top-k nearest neighbors by cosine similarity."""
        result = self._retrieve_neighbors(
            patient_id=patient_id,
            k=k,
            restrict_to_same_disease=restrict_to_same_disease,
        )
        return [
            (
                row["patient_id"],
                row["cosine_similarity"],
                row["disease_subtype"],
                row["outcomes_summary"],
            )
            for row in result.neighbors
        ]

    def _retrieve_neighbors(
        self,
        *,
        patient_id: Any,
        k: int,
        restrict_to_same_disease: bool,
    ) -> RetrievalResult:
        if patient_id not in self._id_to_index:
            raise KeyError(f"Unknown patient_id={patient_id!r}.")

        query_idx = self._id_to_index[patient_id]
        query_vector = self._normalized_matrix[query_idx]
        same_disease = None
        if restrict_to_same_disease:
            same_disease = self._disease_for_patient(patient_id)

        if self.faiss_index is not None:
            import faiss

            n_search = min(len(self._patient_ids), max(k + 25, k + 1))
            scores, indices = self.faiss_index.search(query_vector.reshape(1, -1), n_search)
            candidate_pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            similarities = self._normalized_matrix @ query_vector
            order = np.argsort(-similarities)
            candidate_pairs = [(int(idx), float(similarities[idx])) for idx in order[: max(k + 25, k + 1)]]

        neighbors: List[Dict[str, Any]] = []
        for idx, score in candidate_pairs:
            if idx == query_idx:
                continue
            candidate_id = self._patient_ids[idx]
            candidate_disease = self._disease_for_patient(candidate_id)
            if same_disease is not None and candidate_disease != same_disease:
                continue
            neighbors.append(
                {
                    "patient_id": candidate_id,
                    "cosine_similarity": float(score),
                    "disease_subtype": candidate_disease,
                    "outcomes_summary": self._outcome_summary_for_patient(candidate_id),
                }
            )
            if len(neighbors) >= k:
                break

        return RetrievalResult(query_patient_id=patient_id, neighbors=neighbors)

    def validate_outcome_concordance(
        self,
        k: int = 5,
        n_bootstrap: int = 1000,
        outcome_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compare retrieval outcome concordance against random and IPI baselines."""
        selected_outcome = outcome_name or self._default_outcome_name()
        rows: List[Dict[str, Any]] = []
        grouped = self._metadata.groupby("disease_subtype", dropna=False)
        for disease, group in grouped:
            disease_label = "unknown" if pd.isna(disease) else str(disease)
            disease_ids = [pid for pid in group["patient_id"].tolist() if pid in self._id_to_index]
            opera_scores = []
            random_scores = []
            ipi_scores = []
            ipi_queries = 0
            for patient_id in disease_ids:
                query_value = self._patient_outcome_value(patient_id, selected_outcome)
                if query_value is None:
                    continue

                opera_neighbors = self._retrieve_neighbors(
                    patient_id=patient_id,
                    k=k,
                    restrict_to_same_disease=True,
                ).neighbors
                opera_scores.append(
                    _binary_concordance(
                        query_value,
                        [self._patient_outcome_value(n["patient_id"], selected_outcome) for n in opera_neighbors],
                    )
                )

                random_neighbors = self._random_same_disease_neighbors(
                    patient_id=patient_id,
                    disease_ids=disease_ids,
                    k=k,
                )
                random_scores.append(
                    _binary_concordance(
                        query_value,
                        [self._patient_outcome_value(pid, selected_outcome) for pid in random_neighbors],
                    )
                )

                closest_ipi = self._closest_ipi_neighbors(patient_id, disease_ids=disease_ids, k=k)
                if closest_ipi:
                    ipi_queries += 1
                    ipi_scores.append(
                        _binary_concordance(
                            query_value,
                            [self._patient_outcome_value(pid, selected_outcome) for pid in closest_ipi],
                        )
                    )

            rows.append(
                {
                    "disease_subtype": disease_label,
                    "outcome_name": selected_outcome,
                    "n_queries": int(len([v for v in opera_scores if np.isfinite(v)])),
                    "n_ipi_queries": int(ipi_queries),
                    "opera_mean_concordance": float(np.nanmean(opera_scores)) if opera_scores else float("nan"),
                    "opera_ci_lower": _bootstrap_mean_ci(opera_scores, n_bootstrap, self.seed)[0],
                    "opera_ci_upper": _bootstrap_mean_ci(opera_scores, n_bootstrap, self.seed)[1],
                    "random_mean_concordance": float(np.nanmean(random_scores)) if random_scores else float("nan"),
                    "random_ci_lower": _bootstrap_mean_ci(random_scores, n_bootstrap, self.seed + 1)[0],
                    "random_ci_upper": _bootstrap_mean_ci(random_scores, n_bootstrap, self.seed + 1)[1],
                    "ipi_mean_concordance": float(np.nanmean(ipi_scores)) if ipi_scores else float("nan"),
                    "ipi_ci_lower": _bootstrap_mean_ci(ipi_scores, n_bootstrap, self.seed + 2)[0],
                    "ipi_ci_upper": _bootstrap_mean_ci(ipi_scores, n_bootstrap, self.seed + 2)[1],
                }
            )
        return pd.DataFrame(rows)

    def validate_rkkp_gradient(
        self,
        max_rank: int = 50,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """Measure how registry similarity decays with embedding-rank distance."""
        if self.rkkp is None:
            raise ValueError("RKKP data is required for validate_rkkp_gradient().")

        per_rank: Dict[int, List[float]] = {rank: [] for rank in range(1, max_rank + 1)}
        correlations: List[float] = []

        eligible_ids = [
            patient_id
            for patient_id in self._patient_ids
            if self._has_rkkp_profile(patient_id)
        ]
        for patient_id in eligible_ids:
            neighbors = self._retrieve_neighbors(
                patient_id=patient_id,
                k=max_rank,
                restrict_to_same_disease=False,
            ).neighbors
            rank_scores = []
            rank_positions = []
            for rank, neighbor in enumerate(neighbors, start=1):
                similarity = self._rkkp_cosine_similarity(patient_id, neighbor["patient_id"])
                if similarity is None:
                    continue
                per_rank[rank].append(similarity)
                rank_positions.append(rank)
                rank_scores.append(similarity)
            if len(rank_positions) >= 2:
                correlations.append(_spearman_rank_correlation(rank_positions, rank_scores))

        rows = []
        for rank in range(1, max_rank + 1):
            lower, upper = _bootstrap_mean_ci(per_rank[rank], n_bootstrap, self.seed + rank)
            rows.append(
                {
                    "rank": rank,
                    "mean_rkkp_similarity": float(np.nanmean(per_rank[rank])) if per_rank[rank] else float("nan"),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_pairs": int(len(per_rank[rank])),
                }
            )

        corr_lower, corr_upper = _bootstrap_mean_ci(correlations, n_bootstrap, self.seed + 999)
        return {
            "per_rank": pd.DataFrame(rows),
            "spearman_summary": {
                "mean_spearman": float(np.nanmean(correlations)) if correlations else float("nan"),
                "ci_lower": corr_lower,
                "ci_upper": corr_upper,
                "n_queries": int(len(correlations)),
            },
        }

    def validate_rkkp_embedding_correlation(
        self,
        n_pairs: int = 50000,
        seed: int = 42,
        n_bootstrap: int = 1000,
        cache_path: Optional[str | Path] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Compare pairwise RKKP similarity to embedding similarity across model stages."""
        if self._rkkp_standardized is None:
            raise ValueError(
                "RKKP data with complete similarity columns is required for "
                "validate_rkkp_embedding_correlation()."
            )
        required_versions = {"base", "dapt", "opera"}
        available_versions = set(self._normalized_embeddings.keys())
        missing_versions = required_versions - available_versions
        if missing_versions:
            raise ValueError(
                "validate_rkkp_embedding_correlation requires embeddings for "
                f"{sorted(required_versions)}; missing {sorted(missing_versions)}."
            )

        cache_file = (
            Path(cache_path)
            if cache_path is not None
            else Path("opera") / "retrieval" / "pairwise_rkkp_embedding_correlation.json"
        )
        if use_cache and cache_file.exists():
            LOGGER.info("Loading cached pairwise correlation results from %s", cache_file)
            with cache_file.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            return _deserialize_pairwise_results(cached)

        eligible_ids = sorted(
            set(self._rkkp_standardized["patient_id"].tolist())
            & set(self._embedding_id_to_index["base"].keys())
            & set(self._embedding_id_to_index["dapt"].keys())
            & set(self._embedding_id_to_index["opera"].keys())
        )
        if len(eligible_ids) < 2:
            raise ValueError("At least two patients with RKKP and all embedding versions are required.")

        sampled_pairs, total_pairs = self._sample_patient_pairs(
            patient_ids=eligible_ids,
            n_pairs=n_pairs,
            seed=seed,
        )
        if total_pairs < n_pairs:
            LOGGER.info(
                "Requested n_pairs=%d but only %d unique patient pairs were available.",
                n_pairs,
                total_pairs,
            )

        pair_df = self._build_pairwise_similarity_frame(sampled_pairs)
        version_results: Dict[str, Any] = {}
        for version in ("base", "dapt", "opera"):
            version_results[version] = self._summarize_pairwise_correlations(
                pair_df=pair_df,
                embedding_column=f"{version}_embedding_similarity",
                disease_min_pairs=100,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

        opera_summary = version_results["opera"]["overall"]
        base_summary = version_results["base"]["overall"]
        dapt_summary = version_results["dapt"]["overall"]
        summary_sentence = (
            "OPERA embedding similarity correlates with RKKP-defined clinical "
            f"similarity (Spearman rho = {opera_summary['spearman_rho']:.3f} "
            f"[95% CI {opera_summary['spearman_ci_lower']:.3f}-{opera_summary['spearman_ci_upper']:.3f}], "
            f"p {opera_summary['spearman_p_text']}, N = {opera_summary['n_pairs']} pairs), "
            f"compared to rho = {base_summary['spearman_rho']:.3f} for the base pretrained model "
            f"and rho = {dapt_summary['spearman_rho']:.3f} after domain-adaptive pretraining alone."
        )

        result = {
            "n_pairs_requested": int(n_pairs),
            "n_pairs_used": int(len(pair_df)),
            "n_total_possible_pairs": int(total_pairs),
            "embedding_versions": version_results,
            "scatter_plot_data": version_results["opera"]["scatter_plot_data"].copy(),
            "comparison_scatter_data": pair_df[
                [
                    "rkkp_similarity",
                    "base_embedding_similarity",
                    "dapt_embedding_similarity",
                    "opera_embedding_similarity",
                    "disease_subtype",
                ]
            ].copy(),
            "summary_sentence": summary_sentence,
            "cache_path": str(cache_file),
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(_serialize_pairwise_results(result), handle, indent=2, default=str)
        LOGGER.info("Saved pairwise correlation results to %s", cache_file)
        return result

    def get_case_study(
        self,
        patient_id: Any,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Build a structured case-study table for one patient and their neighbors."""
        query = self._patient_profile(patient_id)
        retrieval = self._retrieve_neighbors(
            patient_id=patient_id,
            k=k,
            restrict_to_same_disease=False,
        )
        rows = []
        for neighbor in retrieval.neighbors:
            profile = self._patient_profile(neighbor["patient_id"])
            rows.append(
                {
                    **profile,
                    "embedding_similarity": neighbor["cosine_similarity"],
                    "rkkp_similarity": self._rkkp_cosine_similarity(patient_id, neighbor["patient_id"]),
                }
            )
        return {
            "query_patient": query,
            "retrieved_patients": rows,
        }

    def _default_outcome_name(self) -> str:
        names = self.outcomes["outcome_name"].dropna().unique().tolist()
        if not names:
            raise ValueError("No outcome_name values available.")
        return str(sorted(names)[0])

    def _summarize_patient_outcomes(self, group: pd.DataFrame) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for _, row in group.iterrows():
            summary[str(row["outcome_name"])] = {
                "event_indicator": int(row["event_indicator"]),
                "time_to_event": float(row["time_to_event"]),
            }
        return summary

    def _patient_outcome_value(self, patient_id: Any, outcome_name: str) -> Optional[float]:
        summary = self._outcome_summary_for_patient(patient_id)
        payload = summary.get(outcome_name)
        if payload is None:
            return None
        return float(payload.get("event_indicator", 0))

    def _outcome_summary_for_patient(self, patient_id: Any) -> Dict[str, Any]:
        rows = self._metadata.loc[self._metadata["patient_id"] == patient_id, "outcomes_summary"]
        return rows.iloc[0] if len(rows) else {}

    def _disease_for_patient(self, patient_id: Any) -> Any:
        rows = self._metadata.loc[self._metadata["patient_id"] == patient_id, "disease_subtype"]
        return rows.iloc[0] if len(rows) else None

    def _patient_profile(self, patient_id: Any) -> Dict[str, Any]:
        row = self._metadata.loc[self._metadata["patient_id"] == patient_id]
        if row.empty:
            raise KeyError(f"Unknown patient_id={patient_id!r}.")
        record = row.iloc[0].to_dict()
        return {
            "patient_id": patient_id,
            "disease_subtype": record.get("disease_subtype"),
            "outcomes_summary": record.get("outcomes_summary", {}),
            "rkkp_features": {
                key: record.get(key)
                for key in row.columns
                if key not in {"patient_id", "disease_subtype", "outcomes_summary"}
            },
        }

    def _random_same_disease_neighbors(
        self,
        patient_id: Any,
        disease_ids: Sequence[Any],
        k: int,
    ) -> List[Any]:
        candidates = [pid for pid in disease_ids if pid != patient_id]
        if len(candidates) <= k:
            return candidates
        return self.rng.choice(candidates, size=k, replace=False).tolist()

    def _closest_ipi_neighbors(
        self,
        patient_id: Any,
        disease_ids: Sequence[Any],
        k: int,
    ) -> List[Any]:
        if self.rkkp is None:
            return []
        query_row = self.rkkp.loc[self.rkkp["patient_id"] == patient_id]
        if query_row.empty or pd.isna(query_row.iloc[0].get("ipi_score")):
            return []
        query_ipi = float(query_row.iloc[0]["ipi_score"])
        rows = self.rkkp.loc[
            self.rkkp["patient_id"].isin(disease_ids) & (self.rkkp["patient_id"] != patient_id)
        ].copy()
        rows = rows.dropna(subset=["ipi_score"])
        if rows.empty:
            return []
        rows["ipi_distance"] = np.abs(rows["ipi_score"].astype(float) - query_ipi)
        return rows.sort_values(["ipi_distance", "patient_id"]).head(k)["patient_id"].tolist()

    def _has_rkkp_profile(self, patient_id: Any) -> bool:
        return self._rkkp_vector(patient_id) is not None

    def _rkkp_vector(self, patient_id: Any) -> Optional[np.ndarray]:
        if self._rkkp_standardized is None:
            return None
        row = self._rkkp_standardized.loc[self._rkkp_standardized["patient_id"] == patient_id]
        if row.empty:
            return None
        return row.loc[:, RKKP_SIMILARITY_COLUMNS].iloc[0].to_numpy(dtype=float)

    def _rkkp_cosine_similarity(self, left_id: Any, right_id: Any) -> Optional[float]:
        left = self._rkkp_vector(left_id)
        right = self._rkkp_vector(right_id)
        if left is None or right is None:
            return None
        matrix = np.vstack([left, right])
        matrix = _normalize_rows(matrix)
        return float(np.clip(np.dot(matrix[0], matrix[1]), -1.0, 1.0))

    def _embedding_cosine_similarity(
        self,
        left_id: Any,
        right_id: Any,
        version: str,
    ) -> Optional[float]:
        if version not in self._embedding_id_to_index:
            return None
        id_to_index = self._embedding_id_to_index[version]
        if left_id not in id_to_index or right_id not in id_to_index:
            return None
        matrix = self._normalized_embeddings[version]
        left = matrix[id_to_index[left_id]]
        right = matrix[id_to_index[right_id]]
        return float(np.clip(np.dot(left, right), -1.0, 1.0))

    def _sample_patient_pairs(
        self,
        *,
        patient_ids: Sequence[Any],
        n_pairs: int,
        seed: int,
    ) -> Tuple[List[Tuple[Any, Any]], int]:
        n_patients = len(patient_ids)
        total_pairs = n_patients * (n_patients - 1) // 2
        if total_pairs <= n_pairs:
            pairs: List[Tuple[Any, Any]] = []
            for left_idx in range(n_patients - 1):
                for right_idx in range(left_idx + 1, n_patients):
                    pairs.append((patient_ids[left_idx], patient_ids[right_idx]))
            return pairs, total_pairs

        rng = np.random.RandomState(seed)
        pair_ranks = rng.choice(total_pairs, size=n_pairs, replace=False)
        pairs = [
            (
                patient_ids[left_idx],
                patient_ids[right_idx],
            )
            for left_idx, right_idx in (_pair_from_rank(int(rank), n_patients) for rank in pair_ranks)
        ]
        return pairs, total_pairs

    def _build_pairwise_similarity_frame(
        self,
        patient_pairs: Sequence[Tuple[Any, Any]],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for left_id, right_id in patient_pairs:
            rkkp_similarity = self._rkkp_cosine_similarity(left_id, right_id)
            if rkkp_similarity is None:
                continue
            disease_left = self._disease_for_patient(left_id)
            disease_right = self._disease_for_patient(right_id)
            rows.append(
                {
                    "left_patient_id": left_id,
                    "right_patient_id": right_id,
                    "rkkp_similarity": rkkp_similarity,
                    "base_embedding_similarity": self._embedding_cosine_similarity(left_id, right_id, "base"),
                    "dapt_embedding_similarity": self._embedding_cosine_similarity(left_id, right_id, "dapt"),
                    "opera_embedding_similarity": self._embedding_cosine_similarity(left_id, right_id, "opera"),
                    "disease_subtype": (
                        disease_left if disease_left == disease_right else "cross_disease"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _summarize_pairwise_correlations(
        self,
        *,
        pair_df: pd.DataFrame,
        embedding_column: str,
        disease_min_pairs: int,
        n_bootstrap: int,
        seed: int,
    ) -> Dict[str, Any]:
        scatter_df = pair_df[
            ["rkkp_similarity", embedding_column, "disease_subtype"]
        ].dropna().rename(columns={embedding_column: "embedding_similarity"}).reset_index(drop=True)
        overall = _correlation_summary(
            x=scatter_df["rkkp_similarity"].to_numpy(dtype=float),
            y=scatter_df["embedding_similarity"].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        stratified_rows = []
        for disease_subtype, group in scatter_df.groupby("disease_subtype", dropna=False):
            if disease_subtype == "cross_disease" or len(group) < disease_min_pairs:
                continue
            summary = _correlation_summary(
                x=group["rkkp_similarity"].to_numpy(dtype=float),
                y=group["embedding_similarity"].to_numpy(dtype=float),
                n_bootstrap=n_bootstrap,
                seed=seed + len(stratified_rows) + 1,
            )
            stratified_rows.append(
                {
                    "disease_subtype": disease_subtype,
                    **summary,
                }
            )
        return {
            "overall": overall,
            "scatter_plot_data": scatter_df,
            "stratified_by_disease": pd.DataFrame(stratified_rows),
        }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _binary_concordance(query_value: float, neighbor_values: Sequence[Optional[float]]) -> float:
    valid = [float(value) for value in neighbor_values if value is not None and np.isfinite(value)]
    if not valid:
        return float("nan")
    neighbor_mean = float(np.mean(valid))
    return 1.0 - abs(float(query_value) - neighbor_mean)


def _bootstrap_mean_ci(values: Sequence[float], n_bootstrap: int, seed: int) -> Tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample = array[rng.randint(0, len(array), size=len(array))]
        boot.append(float(sample.mean()))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    corr_fn,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    boot = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        value = corr_fn(x[idx], y[idx])
        if np.isfinite(value):
            boot.append(float(value))
    if not boot:
        return float("nan"), float("nan")
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def _spearman_rank_correlation(rank_positions: Sequence[int], similarities: Sequence[float]) -> float:
    x = np.asarray(rank_positions, dtype=float)
    y = np.asarray(similarities, dtype=float)
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def _correlation_summary(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    spearman = _spearman_rank_correlation(x, y)
    pearson = _pearson_correlation(x, y)
    spearman_ci = _bootstrap_correlation_ci(
        x,
        y,
        corr_fn=lambda x_b, y_b: _spearman_rank_correlation(x_b, y_b),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    pearson_ci = _bootstrap_correlation_ci(
        x,
        y,
        corr_fn=_pearson_correlation,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    spearman_p = _large_sample_corr_pvalue(spearman, len(x))
    pearson_p = _large_sample_corr_pvalue(pearson, len(x))
    return {
        "n_pairs": int(len(x)),
        "spearman_rho": float(spearman),
        "spearman_ci_lower": spearman_ci[0],
        "spearman_ci_upper": spearman_ci[1],
        "spearman_p_value": float(spearman_p),
        "spearman_p_text": _format_p_value(spearman_p),
        "pearson_r": float(pearson),
        "pearson_ci_lower": pearson_ci[0],
        "pearson_ci_upper": pearson_ci[1],
        "pearson_p_value": float(pearson_p),
        "pearson_p_text": _format_p_value(pearson_p),
    }


def _large_sample_corr_pvalue(correlation: float, n: int) -> float:
    if not np.isfinite(correlation) or n < 4:
        return float("nan")
    z = abs(correlation) * np.sqrt(max(1.0, n - 3.0))
    return float(2.0 * (1.0 - _normal_cdf(z)))


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / np.sqrt(2.0))))


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "= NA"
    if value < 1e-4:
        return "< 1e-4"
    return f"= {value:.4f}"


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.arange(1, len(values) + 1, dtype=float)
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = sorted_ranks[start:end].mean()
        start = end
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def _serialize_pairwise_results(result: Dict[str, Any]) -> Dict[str, Any]:
    serialized = dict(result)
    serialized["scatter_plot_data"] = result["scatter_plot_data"].to_dict(orient="records")
    serialized["comparison_scatter_data"] = result["comparison_scatter_data"].to_dict(orient="records")
    serialized["embedding_versions"] = {}
    for version, payload in result["embedding_versions"].items():
        serialized["embedding_versions"][version] = {
            "overall": payload["overall"],
            "scatter_plot_data": payload["scatter_plot_data"].to_dict(orient="records"),
            "stratified_by_disease": payload["stratified_by_disease"].to_dict(orient="records"),
        }
    return serialized


def _deserialize_pairwise_results(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result["scatter_plot_data"] = pd.DataFrame(result.get("scatter_plot_data", []))
    result["comparison_scatter_data"] = pd.DataFrame(result.get("comparison_scatter_data", []))
    restored_versions = {}
    for version, version_payload in result.get("embedding_versions", {}).items():
        restored_versions[version] = {
            "overall": version_payload.get("overall", {}),
            "scatter_plot_data": pd.DataFrame(version_payload.get("scatter_plot_data", [])),
            "stratified_by_disease": pd.DataFrame(version_payload.get("stratified_by_disease", [])),
        }
    result["embedding_versions"] = restored_versions
    return result


def _pair_from_rank(rank: int, n_patients: int) -> Tuple[int, int]:
    low = 0
    high = n_patients - 1
    while low < high:
        mid = (low + high) // 2
        if _pair_rank_prefix(mid + 1, n_patients) <= rank:
            low = mid + 1
        else:
            high = mid
    left_idx = low
    seen = _pair_rank_prefix(left_idx, n_patients)
    offset = rank - seen
    right_idx = left_idx + 1 + offset
    if right_idx < n_patients:
        return left_idx, right_idx
    if left_idx + 1 < n_patients:
        return left_idx, n_patients - 1
    raise ValueError(f"Pair rank {rank} is out of range for n_patients={n_patients}.")


def _pair_rank_prefix(left_idx: int, n_patients: int) -> int:
    return int(left_idx * (2 * n_patients - left_idx - 1) // 2)


__all__ = ["PatientRetriever", "RetrievalResult"]
