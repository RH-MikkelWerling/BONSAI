"""Cross-stage validation for the OPERA temporal split contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from bonsai.functional.outcomes import (
    load_split_contract,
    validate_split_integrity,
)

DEFAULT_SPLIT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "manifests"
    / "temporal_split.yaml"
)


def load_temporal_split_contract(path: str | Path | None = None) -> dict[str, Any]:
    """Load the canonical OPERA split contract unless another path is supplied."""
    return load_split_contract(path or DEFAULT_SPLIT_CONTRACT_PATH)


def _read_outcome(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_subject_ids(path: str | Path) -> set[int]:
    subjects = torch.load(path, map_location="cpu")
    if isinstance(subjects, Mapping):
        iterable = subjects.values()
    else:
        iterable = subjects
    return {
        int(subject["subject_id"])
        for subject in iterable
        if isinstance(subject, Mapping) and "subject_id" in subject
    }


def _load_embedding_store_ids(path: str | Path) -> set[int]:
    store = torch.load(path, map_location="cpu")
    if not isinstance(store, Mapping):
        raise TypeError(f"Embedding store must be a mapping, got {type(store)!r}.")
    return {int(subject_id) for subject_id in store}


def _split_subjects(
    outcome_paths: list[str | Path],
    contract: Mapping[str, Any],
) -> tuple[dict[str, set[int]], list[str], dict[str, Any]]:
    split_keys = {
        "train": contract.get("train_key", "train"),
        "tuning": contract.get("val_key", "tuning"),
        "held_out": contract.get("test_key", "held_out"),
    }
    subjects_by_split = {key: set() for key in split_keys.values()}
    subject_to_splits: dict[int, set[str]] = {}
    issues: list[str] = []
    details: dict[str, Any] = {"outcomes": {}}
    for path in outcome_paths:
        frame = _read_outcome(path)
        report = validate_split_integrity(frame, **contract)
        details["outcomes"][str(path)] = report
        if not report["ok"]:
            issues.append(f"Outcome split contract failed for {path}: {report}")
        for split_key, group in frame.groupby("split", sort=False):
            if split_key in subjects_by_split:
                subject_ids = {int(value) for value in group["subject_id"].dropna()}
                subjects_by_split[split_key].update(subject_ids)
                for subject_id in subject_ids:
                    subject_to_splits.setdefault(subject_id, set()).add(split_key)
    multi_split = {
        subject_id: sorted(values)
        for subject_id, values in subject_to_splits.items()
        if len(values) > 1
    }
    if multi_split:
        issues.append(
            "Split assignment is not patient-level; subjects appear in multiple "
            f"splits: {dict(list(multi_split.items())[:10])}"
        )
    details["n_subjects_by_split"] = {
        split: len(subjects) for split, subjects in subjects_by_split.items()
    }
    return subjects_by_split, issues, details


def validate_cross_stage_split_contract(
    *,
    outcome_paths: list[str | Path],
    subject_data_paths: Mapping[str, str | Path] | None = None,
    dapt_subject_data_paths: Mapping[str, str | Path] | None = None,
    embedding_store_paths: list[str | Path] | None = None,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate labels, split subject data, DAPT inputs, and embedding stores."""
    contract = load_temporal_split_contract(contract_path)
    subjects_by_split, issues, details = _split_subjects(outcome_paths, contract)
    train_key = contract.get("train_key", "train")
    val_key = contract.get("val_key", "tuning")
    test_key = contract.get("test_key", "held_out")
    held_out_ids = subjects_by_split.get(test_key, set())
    train_val_ids = subjects_by_split.get(train_key, set()) | subjects_by_split.get(
        val_key,
        set(),
    )

    def check_subject_data(paths: Mapping[str, str | Path], label: str) -> None:
        details.setdefault(label, {})
        for split_name, path in paths.items():
            ids = _load_subject_ids(path)
            details[label][str(path)] = {
                "split": split_name,
                "n_subjects": len(ids),
                "n_unlabelled_or_unseen": len(
                    ids - held_out_ids - train_val_ids
                ),
            }
            if split_name in {train_key, "train", val_key, "tuning"}:
                leaked = ids & held_out_ids
                if leaked:
                    issues.append(
                        f"{label} {path} contains held-out subjects: "
                        f"{sorted(leaked)[:10]}"
                    )
            if split_name in {test_key, "held_out"}:
                leaked = ids & train_val_ids
                if leaked:
                    issues.append(
                        f"{label} {path} contains train/tuning subjects: "
                        f"{sorted(leaked)[:10]}"
                    )

    if subject_data_paths:
        check_subject_data(subject_data_paths, "subject_data")
    if dapt_subject_data_paths:
        check_subject_data(dapt_subject_data_paths, "dapt_subject_data")

    details["embedding_stores"] = {}
    for path in embedding_store_paths or []:
        ids = _load_embedding_store_ids(path)
        leaked = ids & held_out_ids
        details["embedding_stores"][str(path)] = {
            "n_subjects": len(ids),
            "n_held_out_subjects": len(leaked),
            "n_unlabelled_or_unseen": len(ids - held_out_ids - train_val_ids),
        }
        if leaked:
            issues.append(
                f"Embedding store {path} contains held-out subjects: "
                f"{sorted(leaked)[:10]}"
            )

    return {
        "ok": not issues,
        "issues": issues,
        "contract": contract,
        "details": details,
    }
