"""Content-addressed, split-scoped cache for outcome survival labels.

Cache files contain patient-level labels and must remain on the secure server.
Only train/tuning splits are accepted while model selection remains open.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import torch


CACHE_VERSION = 1


def _file_stamp(path: str | None) -> dict | None:
    if not path:
        return None
    resolved = Path(os.path.expandvars(path)).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved), "exists": True,
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def cache_contract(
    *, cohort_name: str, outcome_name: str, split: str,
    cohort_config: dict, outcome_config: dict, eligibility_scope: str,
    require_min_followup: bool, source_paths: dict[str, str | None],
) -> dict:
    if split not in {"train", "tuning"}:
        raise ValueError("Outcome label cache may only hold train/tuning labels")
    return {
        "cache_version": CACHE_VERSION,
        "cohort": cohort_name,
        "outcome": outcome_name,
        "split": split,
        "cohort_config": cohort_config,
        "outcome_config": outcome_config,
        "eligibility_scope": eligibility_scope,
        "require_min_followup": bool(require_min_followup),
        "sources": {name: _file_stamp(path) for name, path in sorted(source_paths.items())},
    }


def load_or_build_labels(
    cache_dir: str | Path, contract: dict, build: Callable[[], dict],
) -> dict:
    """Reuse exact post-eligibility labels or publish them atomically.

    Path, size and nanosecond mtime invalidate the cache when input files are
    replaced. Input releases must still be frozen/immutable for research use.
    ``torch.load`` is limited to this trusted, locally generated cache.
    """
    root = Path(cache_dir)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = json.dumps(contract, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path = root / f"labels_{digest}.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("contract") != contract or not isinstance(payload.get("records"), dict):
            raise ValueError(f"Outcome label cache contract mismatch: {path}")
        return payload["records"]
    records = build()
    if not isinstance(records, dict):
        raise TypeError("Outcome label builder must return a dictionary")
    # Unique temporary files permit concurrent builders; only complete files
    # become visible at the final path. Equivalent builders have one contract.
    fd, temporary = tempfile.mkstemp(prefix=".labels_", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save({"contract": contract, "records": records}, handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return records
