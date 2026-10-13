"""Paired tuning CIF-AUC for one independent CLL specialist versus pooled routers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from opera.run.compare_approved_neural_tuning_predictions import (
    _read_run, compare_predictions, fast_cif_auc,
)

HORIZONS = (30, 60, 90, 180, 365, 730)


def compare_cell(paths: dict[str, Path], outcome: str, *,
                 n_bootstrap: int = 200, min_cases: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(paths) != {"specialist", "clean", "loose"}:
        raise ValueError("Expected specialist, clean and loose prediction paths")
    estimates, contrasts = [], []
    for horizon in HORIZONS:
        frames = {name: _read_run(path, [outcome], horizon) for name, path in paths.items()}
        for name, frame in frames.items():
            if not frame["cohort"].eq("CLL").all():
                raise ValueError(f"{name} prediction export is not CLL-only")
        identity = ["subject_id", "cohort", "time_days", "event"]
        for name in ("clean", "loose"):
            if not frames[name][identity].equals(frames["specialist"][identity]):
                raise ValueError(f"Patient set or labels differ for {name} versus specialist")
        # Compare_predictions verifies exact patient/cohort/time/event identity.
        reference = frames["specialist"]
        _, n_cases, n_controls = fast_cif_auc(
            reference["time_days"].to_numpy(float),
            reference["event"].to_numpy(int),
            reference[f"cif_{horizon}d"].to_numpy(float), horizon,
        )
        if min(n_cases, n_controls) < min_cases:
            estimates.append({"outcome": outcome, "horizon_days": horizon,
                              "status": "unsupported", "n": len(reference),
                              "n_cases": n_cases, "n_controls": n_controls})
            continue
        try:
            scores, deltas, _ = compare_predictions(
                frames, baseline="specialist", outcomes=[outcome], horizon=horizon,
                n_bootstrap=n_bootstrap, seed=42, family_by_outcome={outcome: "cell"},
                min_cases=min_cases, skip_under_supported=True,
            )
        except ValueError as error:
            if str(error) != "No outcomes had sufficient support for the forest":
                raise
            estimates.append({"outcome": outcome, "horizon_days": horizon,
                              "status": "insufficient_valid_bootstrap", "n": len(reference),
                              "n_cases": n_cases, "n_controls": n_controls})
            continue
        scores.insert(0, "status", "estimable")
        estimates.extend(scores.to_dict("records"))
        contrasts.extend(deltas.to_dict("records"))
    return pd.DataFrame(estimates), pd.DataFrame(contrasts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--loose", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    paths = {name: getattr(args, name) for name in ("specialist", "clean", "loose")}
    estimates, contrasts = compare_cell(paths, args.outcome, n_bootstrap=args.bootstrap)
    args.output_dir.mkdir(parents=True)
    estimates.to_csv(args.output_dir / "cll_specialist_estimates.csv", index=False)
    contrasts.to_csv(args.output_dir / "cll_specialist_contrasts.csv", index=False)
    (args.output_dir / "provenance_INTERNAL.json").write_text(json.dumps({
        "outcome": args.outcome, "split": "tuning", "paths": {k: str(v.resolve()) for k,v in paths.items()},
        "bootstrap": args.bootstrap, "estimand": "CIF-AUC in the exact same CLL patient/outcome risk set",
        "selection_caveat": "Each model uses its own prespecified tuning-selected checkpoint; "
                            "intervals are exploratory and do not correct checkpoint selection.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": args.outcome,
                      "estimable_horizons": int(estimates.loc[estimates.status.eq("estimable"),
                                                         "horizon_days"].nunique()),
                      "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
