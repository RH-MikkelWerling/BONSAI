"""Utilities for inspecting learned vocabulary/code embeddings.

Patient embeddings answer "where are patients in representation space?"
Vocabulary embeddings answer "where are medical concepts and codes in the
model's lookup table?"  Keeping these analyses separate avoids mixing the
statistical unit of analysis while still letting us compare the geometry of
codes across pretraining, DAPT, and OPERA stages.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from opera.evaluation.treatment_embeddings import embedding_columns


CODE_EMBEDDING_SUFFIX = "embeddings.code_embedding.weight"
PREFERRED_CODE_EMBEDDING_KEYS = (
    "model.embeddings.code_embedding.weight",
    "model.encoder.embeddings.code_embedding.weight",
    "embeddings.code_embedding.weight",
    "encoder.embeddings.code_embedding.weight",
)
SPECIAL_TOKENS = {"[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"}


def load_vocabulary(path: str | Path) -> dict[str, int]:
    """Load a BONSAI token-to-id vocabulary from ``.pt``, JSON, CSV, or parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    elif suffix == ".json":
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    elif suffix in {".parquet", ".pq"}:
        loaded = pd.read_parquet(path)
    else:
        loaded = pd.read_csv(path)
    return _coerce_vocabulary(loaded)


def _coerce_vocabulary(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        for key in ("vocabulary", "vocab", "token_to_id"):
            if key in value and isinstance(value[key], Mapping):
                value = value[key]
                break
        if all(isinstance(token, str) and _is_int_like(idx) for token, idx in value.items()):
            vocabulary = {str(token): int(idx) for token, idx in value.items()}
        elif all(_is_int_like(idx) and isinstance(token, str) for idx, token in value.items()):
            vocabulary = {str(token): int(idx) for idx, token in value.items()}
        else:
            raise ValueError("Vocabulary mapping must be token->id or id->token.")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vocabulary = {str(token): idx for idx, token in enumerate(value)}
    elif isinstance(value, pd.DataFrame):
        if {"token", "token_id"}.issubset(value.columns):
            token_col, id_col = "token", "token_id"
        elif {"code", "code_id"}.issubset(value.columns):
            token_col, id_col = "code", "code_id"
        elif value.shape[1] >= 2:
            token_col, id_col = value.columns[:2]
        else:
            raise ValueError("Vocabulary table must contain token and token_id columns.")
        vocabulary = {
            str(row[token_col]): int(row[id_col])
            for _, row in value[[token_col, id_col]].dropna().iterrows()
        }
    else:
        raise ValueError("Unsupported vocabulary artifact.")
    _validate_vocabulary(vocabulary)
    return vocabulary


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _validate_vocabulary(vocabulary: Mapping[str, int]) -> None:
    if not vocabulary:
        raise ValueError("Vocabulary is empty.")
    ids = list(vocabulary.values())
    if len(set(ids)) != len(ids):
        raise ValueError("Vocabulary token ids must be unique.")
    if min(ids) < 0:
        raise ValueError("Vocabulary token ids must be non-negative.")


def invert_vocabulary(vocabulary: Mapping[str, int]) -> dict[int, str]:
    """Invert a token-to-id vocabulary."""
    return {int(token_id): str(token) for token, token_id in vocabulary.items()}


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint or raw state dict and return the tensor state mapping."""
    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(loaded, Mapping) and isinstance(loaded.get("state_dict"), Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path} is not a checkpoint or state-dict mapping.")
    return {str(key): value for key, value in loaded.items() if torch.is_tensor(value)}


def find_code_embedding_key(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Return the state-dict key for the token/code embedding matrix."""
    for key in PREFERRED_CODE_EMBEDDING_KEYS:
        if key in state_dict:
            return key
    candidates = sorted(
        (key for key in state_dict if key.endswith(CODE_EMBEDDING_SUFFIX)),
        key=lambda item: (item.count("."), len(item), item),
    )
    if not candidates:
        raise ValueError(
            "Could not find a code embedding matrix ending in "
            f"{CODE_EMBEDDING_SUFFIX!r}."
        )
    return candidates[0]


def extract_vocabulary_embedding_frame(
    checkpoint_path: str | Path,
    vocabulary_path: str | Path,
    *,
    stage: Optional[str] = None,
    token_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract ``code_embedding.weight`` into a token-level embedding table."""
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    embedding_key = find_code_embedding_key(state_dict)
    matrix = state_dict[embedding_key].detach().cpu().float().numpy()
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{embedding_key} must be a non-empty 2D matrix.")

    vocabulary = load_vocabulary(vocabulary_path)
    id_to_token = invert_vocabulary(vocabulary)
    if len(vocabulary) != matrix.shape[0]:
        raise ValueError(
            f"Vocabulary size ({len(vocabulary)}) does not match checkpoint "
            f"embedding rows ({matrix.shape[0]}) for key {embedding_key!r}."
        )
    token_ids = np.array(sorted(id_to_token), dtype=int)
    if token_ids[-1] >= matrix.shape[0]:
        raise ValueError("Vocabulary contains token ids outside the embedding matrix.")

    frame = pd.DataFrame(
        matrix[token_ids],
        columns=[f"embedding_{index}" for index in range(matrix.shape[1])],
    )
    frame.insert(0, "token", [id_to_token[int(token_id)] for token_id in token_ids])
    frame.insert(0, "token_id", token_ids)
    frame["token_source"] = [infer_token_source(token) for token in frame["token"]]
    frame["token_family"] = [infer_token_family(token) for token in frame["token"]]
    frame["is_special"] = frame["token"].map(is_special_token)
    frame["embedding_norm"] = np.linalg.norm(matrix[token_ids], axis=1)
    frame["embedding_key"] = embedding_key
    if stage is not None:
        frame.insert(0, "embedding_stage", stage)
    if token_metadata is not None:
        frame = merge_token_metadata(frame, token_metadata)
    return frame


def infer_token_source(token: str) -> str:
    """Infer the namespace/source prefix of a token using BONSAI conventions."""
    token = str(token)
    if is_special_token(token):
        return "special"
    if "//" in token:
        return token.split("//", 1)[0] or "unprefixed"
    for separator in ("::", ":", "/", "|"):
        if separator in token:
            prefix = token.split(separator, 1)[0].strip()
            if prefix:
                return prefix
    return "unprefixed"


def infer_token_family(token: str) -> str:
    """Infer a broad display family for code-token visualizations."""
    token = str(token)
    source = infer_token_source(token)
    if source != "unprefixed":
        return source
    if re.match(r"^D_CODE_\d+$", token):
        return "diagnosis_code"
    if re.match(r"^[A-Z]\d{2}[A-Z]{2}\d{2}", token):
        return "medication_code"
    if re.match(r"^[A-Z]\d{2}", token):
        return "diagnosis_code"
    return "unprefixed"


def is_special_token(token: str) -> bool:
    """Return whether a token is a model special token."""
    token = str(token)
    return token in SPECIAL_TOKENS or (token.startswith("[") and token.endswith("]"))


def merge_token_metadata(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    token_col: str = "token",
    token_id_col: str = "token_id",
) -> pd.DataFrame:
    """Merge optional token metadata by ``token_id`` or ``token``."""
    if token_id_col in metadata.columns:
        key = token_id_col
    elif token_col in metadata.columns:
        key = token_col
    else:
        raise ValueError("Token metadata must contain token_id or token.")
    if metadata[key].duplicated().any():
        raise ValueError(f"Token metadata contains duplicate {key!r} values.")
    renamed = metadata.copy()
    conflicts = (
        set(renamed.columns)
        & set(embeddings.columns)
        - {token_col, token_id_col}
    )
    renamed = renamed.rename(columns={column: f"{column}_metadata" for column in conflicts})
    return embeddings.merge(renamed, on=key, how="left", validate="one_to_one")


def compute_token_movement(
    stage_frames: Mapping[str, pd.DataFrame],
    *,
    reference_stage: str,
) -> pd.DataFrame:
    """Compare code-token movement from one reference stage to other stages."""
    if reference_stage not in stage_frames:
        raise ValueError(f"reference_stage={reference_stage!r} is not available.")
    reference = stage_frames[reference_stage]
    ref_cols = embedding_columns(reference)
    ref = reference[["token_id", "token", "token_family", *ref_cols]].copy()

    outputs = []
    for stage, frame in stage_frames.items():
        if stage == reference_stage:
            continue
        cols = embedding_columns(frame)
        current = frame[["token_id", "token", *cols]].copy()
        merged = ref.merge(
            current,
            on=["token_id", "token"],
            suffixes=("_reference", "_current"),
            validate="one_to_one",
        )
        if merged.empty:
            continue
        ref_values = merged[[f"{col}_reference" for col in ref_cols]].to_numpy(
            dtype=float
        )
        cur_values = merged[[f"{col}_current" for col in cols]].to_numpy(dtype=float)
        if ref_values.shape[1] != cur_values.shape[1]:
            raise ValueError(
                f"Embedding dimensions differ for {reference_stage!r} and {stage!r}."
            )
        cosine = np.sum(_l2_normalize(ref_values) * _l2_normalize(cur_values), axis=1)
        delta = cur_values - ref_values
        out = merged[["token_id", "token", "token_family"]].copy()
        out["reference_stage"] = reference_stage
        out["embedding_stage"] = stage
        out["cosine_similarity"] = cosine
        out["cosine_distance"] = 1.0 - cosine
        out["l2_distance"] = np.linalg.norm(delta, axis=1)
        out["reference_norm"] = np.linalg.norm(ref_values, axis=1)
        out["embedding_norm"] = np.linalg.norm(cur_values, axis=1)
        outputs.append(out)
    if not outputs:
        return pd.DataFrame(
            columns=[
                "token_id",
                "token",
                "token_family",
                "reference_stage",
                "embedding_stage",
                "cosine_similarity",
                "cosine_distance",
                "l2_distance",
                "reference_norm",
                "embedding_norm",
            ]
        )
    movement = pd.concat(outputs, ignore_index=True)
    return movement.sort_values(
        ["embedding_stage", "cosine_distance", "l2_distance"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def nearest_token_neighbors(
    frame: pd.DataFrame,
    query_tokens: Sequence[str],
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Find nearest code-token neighbors by cosine similarity."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    cols = embedding_columns(frame)
    vectors = _l2_normalize(frame[cols].to_numpy(dtype=float))
    token_to_index = {str(token): idx for idx, token in enumerate(frame["token"])}
    rows: list[dict[str, Any]] = []
    for query in query_tokens:
        if query not in token_to_index:
            raise ValueError(f"Query token {query!r} is not in the embedding frame.")
        index = token_to_index[query]
        similarities = vectors @ vectors[index]
        order = np.argsort(-similarities)
        rank = 0
        for neighbor_index in order:
            if neighbor_index == index:
                continue
            rank += 1
            rows.append(
                {
                    "query_token": query,
                    "neighbor_rank": rank,
                    "neighbor_token": frame["token"].iloc[neighbor_index],
                    "neighbor_token_id": int(frame["token_id"].iloc[neighbor_index]),
                    "neighbor_family": frame["token_family"].iloc[neighbor_index],
                    "cosine_similarity": float(similarities[neighbor_index]),
                }
            )
            if rank >= top_k:
                break
    return pd.DataFrame(rows)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)
