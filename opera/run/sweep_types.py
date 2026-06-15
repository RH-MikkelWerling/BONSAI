"""
Typed orchestration primitives for the OPERA evaluation sweep.

This module defines the data structures used to record per-cell status during a
sweep run. Decomposing the monolithic ``run_sweep`` into typed, testable units
makes the reproducibility audit trail (``sweep_cell_status.csv`` /
``sweep_cell_status.jsonl``) explicit and unit-testable rather than an ad-hoc
list of dicts.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import pandas as pd

VALID_STAGES: frozenset[str] = frozenset(
    {
        "ipi",
        "finetune",
        "evaluate",
        "evaluate_predictions",
        "results_file",
        "cached",
        "dry_run",
        "configuration",
    }
)

VALID_STATUSES: frozenset[str] = frozenset(
    {
        "success",
        "failed",
        "skipped",
        "dry_run",
    }
)


@dataclass(frozen=True)
class SweepCellRecord:
    """Immutable record of the outcome of one sweep cell stage.

    A "cell" is a single ``(cohort, outcome, variant, seed)`` combination. Each
    stage of processing that cell (finetune, evaluate, ...) emits one record so
    the full audit trail can be reconstructed after an offline run.
    """

    cohort: str
    outcome: str
    variant: str
    seed: int
    stage: str
    status: str
    output_dir: str
    timestamp: float = 0.0
    artifact: Optional[str] = None
    reason: Optional[str] = None
    evaluation_subset: Optional[str] = None
    training_fraction: Optional[float] = None

    def to_dict(self) -> dict:
        """Return a plain dict of all fields (including ``None`` values).

        The full key set is always present so that the resulting DataFrame and
        JSONL have a stable schema regardless of which optional fields were set.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "SweepCellRecord":
        """Build a record from arbitrary kwargs, ignoring unknown fields.

        Callers historically passed extra keys to ``append_cell_status``; this
        keeps backward compatibility by silently dropping any kwargs that are
        not declared fields of :class:`SweepCellRecord`.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in known}
        return cls(**filtered)


@dataclass
class StatusTracker:
    """Mutable collector of :class:`SweepCellRecord` entries for a sweep run."""

    _records: list[SweepCellRecord] = field(default_factory=list)

    def __init__(self) -> None:
        self._records = []

    def append(self, **kwargs) -> SweepCellRecord:
        """Build, timestamp, store, and return a new status record."""
        kwargs["timestamp"] = time.time()
        record = SweepCellRecord.from_kwargs(**kwargs)
        self._records.append(record)
        return record

    @property
    def records(self) -> list[SweepCellRecord]:
        """Return the list of recorded cell statuses."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a DataFrame (one row per record)."""
        return pd.DataFrame([r.to_dict() for r in self._records])

    def n_failed(self) -> int:
        """Number of recorded cells whose status is ``failed``."""
        return sum(1 for r in self._records if r.status == "failed")

    def n_skipped(self) -> int:
        """Number of recorded cells whose status is ``skipped``."""
        return sum(1 for r in self._records if r.status == "skipped")

    def n_success(self) -> int:
        """Number of recorded cells whose status is ``success``."""
        return sum(1 for r in self._records if r.status == "success")

    def write(self, output_dir: Path) -> None:
        """Persist the audit trail as ``sweep_cell_status.{csv,jsonl}``.

        Does nothing when no records have been collected, matching the prior
        behaviour where empty sweeps wrote no status files.
        """
        if not self._records:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = self.to_dataframe()
        frame.to_csv(output_dir / "sweep_cell_status.csv", index=False)
        with open(output_dir / "sweep_cell_status.jsonl", "w") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")
