"""Export only explicitly validation-labelled scalar evaluation results.

Reads ``evaluation/<outcome>/result.jsonl`` and writes one CSV. It never
exports predictions, patient identifiers, plots, or held-out evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

VALIDATION_SPLITS = {"tuning", "validation", "val"}
IDENTITY = (
    "model_family",
    "outcome",
    "cohort",
    "analysis_level",
    "evaluation_regime",
    "split",
    "seed",
    "training_fraction",
    "config_hash",
)
METRICS = (
    "n_total", "n_positive", "prevalence", "auroc", "auroc_lower",
    "auroc_upper", "auprc", "brier_score", "ece", "mce",
    "calibration_slope", "calibration_intercept", "hl_pvalue",
    "competing_risk_n_total", "n_primary_events", "n_competing_events",
    "concordance_index", "c_index_within_fine", "c_index_macro_within_fine",
)


def _scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def collect(evaluation_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return approved aggregate rows and a no-data/rejected-file inventory."""
    rows, inventory = [], []
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(evaluation_dir)
    for outcome_dir in sorted(p for p in evaluation_dir.iterdir() if p.is_dir()):
        path = outcome_dir / "result.jsonl"
        if not path.is_file():
            inventory.append({"outcome_dir": outcome_dir.name, "status": "no_result_jsonl"})
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(records) != 1 or not isinstance(records[0], dict):
            inventory.append({"outcome_dir": outcome_dir.name, "status": "invalid_result_count"})
            continue
        record = records[0]
        split = str(record.get("split", "")).lower()
        if split not in VALIDATION_SPLITS:
            inventory.append({"outcome_dir": outcome_dir.name, "status": "excluded_nonvalidation"})
            continue
        if record.get("outcome") != outcome_dir.name:
            inventory.append({"outcome_dir": outcome_dir.name, "status": "outcome_mismatch"})
            continue
        row = {key: _scalar(record.get(key)) for key in (*IDENTITY, *METRICS)}
        for key, value in record.items():
            if key.startswith(("cif_auc_", "cif_brier_", "ipcw_auc_", "ipcw_brier_")):
                row[key] = _scalar(value)
        rows.append(row)
        inventory.append({"outcome_dir": outcome_dir.name, "status": "included_validation"})
    return rows, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, inventory = collect(args.evaluation_dir)
    if not rows:
        raise SystemExit("No explicitly validation-labelled results; nothing exported.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(item["status"] == status for item in inventory)
              for status in sorted({item["status"] for item in inventory})}
    print(f"Wrote {len(rows)} validation aggregate rows to {args.output}")
    print(f"Inventory: {counts}")


if __name__ == "__main__":
    main()
