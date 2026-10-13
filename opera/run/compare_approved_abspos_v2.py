"""Compare the two completed 69-outcome runs on tuning data only.

Reads aggregate Lightning logs and selection inventories, never held-out data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


MONITOR = "val/cr_cif_auc_endpoint_group_macro"
FAMILY_PREFIX = "val/cr_cif_auc_family/"


def _metrics_paths(run: Path) -> list[Path]:
    found = list((run / "contrastive_multicohort_runs").glob("version_*/metrics.csv"))
    if not found:
        raise ValueError(f"Expected Lightning metrics.csv under {run}")
    if any(not path.parent.name.removeprefix("version_").isdigit() for path in found):
        raise ValueError(f"Unexpected Lightning logger version under {run}: {found}")
    return sorted(found, key=lambda path: int(path.parent.name.removeprefix("version_")))


def _epochs(paths: Path | list[Path]) -> dict[int, dict[str, float]]:
    epochs: dict[int, dict[str, float]] = {}
    for path in [paths] if isinstance(paths, Path) else paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or MONITOR not in reader.fieldnames or "epoch" not in reader.fieldnames:
                raise ValueError(f"Missing selection metric or epoch column in {path}")
            for row in reader:
                if not row.get("epoch"):
                    continue
                epoch = int(float(row["epoch"])) + 1
                values = epochs.setdefault(epoch, {})
                for name, raw in row.items():
                    if raw and name != "epoch":
                        try:
                            value = float(raw)
                        except ValueError:
                            continue
                        if math.isfinite(value):
                            values[name] = value
    result = {epoch: values for epoch, values in epochs.items() if MONITOR in values}
    if not result:
        raise ValueError(f"No completed tuning selection epochs in {path}")
    return result


def _run(run: Path) -> dict:
    for name in ("best.ckpt", "checkpoint_metadata.json", "training_complete.json",
                 "endpoint_selection_inventory.json"):
        if not (run / name).is_file():
            raise ValueError(f"Run is incomplete or lacks selection audit: {run / name}")
    completion = json.loads((run / "training_complete.json").read_text(encoding="utf-8"))
    if completion.get("status") != "completed":
        raise ValueError(f"Run is not marked completed: {run}")
    inventory = json.loads((run / "endpoint_selection_inventory.json").read_text(encoding="utf-8"))
    metrics_paths = _metrics_paths(run)
    epochs = _epochs(metrics_paths)
    best = max(epochs, key=lambda epoch: epochs[epoch][MONITOR])
    last = max(epochs)
    selected = epochs[best]
    families = {name.removeprefix(FAMILY_PREFIX): value for name, value in selected.items()
                if name.startswith(FAMILY_PREFIX)}
    return {"run": str(run.resolve()),
            "metrics_files": [str(path.resolve()) for path in metrics_paths],
            "validated_epochs": len(epochs),
            "best_epoch": best, "best_endpoint_group_auc": selected[MONITOR],
            "last_epoch": last, "last_endpoint_group_auc": epochs[last][MONITOR],
            "selection_groups": selected.get("val/cr_cif_auc_selection_groups"),
            "selection_cells": selected.get("val/cr_cif_selection_cells"),
            "family_auc_at_best_epoch": families,
            "per_epoch_endpoint_group_auc": {str(k): v[MONITOR] for k, v in epochs.items()},
            "inventory": inventory}


def compare(fourier: Path, time2vec: Path) -> dict:
    f, t = _run(fourier), _run(time2vec)
    if f["inventory"] != t["inventory"]:
        raise ValueError("The two runs used different tuning selection inventories")
    f.pop("inventory")
    inventory = t.pop("inventory")
    if (f["selection_groups"] != t["selection_groups"]
            or f["selection_cells"] != t["selection_cells"]):
        raise ValueError("Selected tuning support differs between runs")
    delta = t["best_endpoint_group_auc"] - f["best_endpoint_group_auc"]
    return {"comparison": "scaled_time2vec_minus_fourier",
            "best_auc_difference": delta,
            "provisional_tuning_winner": "scaled_time2vec" if delta > 0 else "fourier" if delta < 0 else "tie",
            "selection_inventory_cells": len(inventory),
            "fourier": f, "scaled_time2vec": t,
            "note": "Single-seed tuning comparison; no held-out data used. Inspect per-epoch stability and family tradeoffs before choosing an encoder."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourier-run", type=Path, required=True)
    parser.add_argument("--time2vec-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.fourier_run, args.time2vec_run), indent=2))


if __name__ == "__main__":
    main()
