"""Paired, patient-bootstrap CIF comparisons of fixed approved neural checkpoints.

Inputs are INTERNAL tuning prediction Parquets from evaluate_direct_competing_risk.
Intervals condition on the selected checkpoints; they do not correct tuning
checkpoint/model selection and are not held-out generalization intervals.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score


REQUIRED = {"subject_id", "cohort", "split", "outcome", "time_days", "event"}


def fast_cif_auc(times: np.ndarray, events: np.ndarray, risk: np.ndarray,
                 horizon: float) -> tuple[float, int, int]:
    """Same IPCW case/control AUC as compute_competing_risk_metrics_at_horizon.

    Uses a vectorized KM of administrative censoring and weighted ROC AUC,
    avoiding the quadratic case-control comparison inside bootstrap loops.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if times.ndim != 1 or events.shape != times.shape or risk.shape != times.shape:
        raise ValueError("times, events and risk must be matching vectors")
    if not len(times) or not np.isfinite(times).all() or not np.isfinite(risk).all():
        raise ValueError("Empty or nonfinite evaluation vectors")
    if (times < 0).any() or not np.isin(events, [0, 1, 2]).all():
        raise ValueError("Invalid time or competing-event codes")
    order = np.argsort(times, kind="mergesort")
    unique, starts, _ = np.unique(times[order], return_index=True, return_counts=True)
    at_risk = len(times) - starts
    admin = np.add.reduceat((events[order] == 0).astype(int), starts)
    g_after = np.cumprod(1. - admin / at_risk)

    def g_before(query: np.ndarray) -> np.ndarray:
        index = np.searchsorted(unique, query, side="left") - 1
        result = np.ones(len(query), dtype=float)
        valid = index >= 0
        result[valid] = g_after[index[valid]]
        return np.maximum(result, 1e-6)

    case = (times <= horizon) & (events == 1)
    competing = (times <= horizon) & (events == 2)
    control = competing | (times > horizon) | ((times >= horizon) & (events == 0))
    n_case, n_control = int(case.sum()), int(control.sum())
    if not n_case or not n_control:
        return float("nan"), n_case, n_control
    weights = np.concatenate((1. / g_before(times[case]),
                              1. / g_before(np.where(competing[control],
                                                   times[control], horizon))))
    labels = np.concatenate((np.ones(n_case), np.zeros(n_control)))
    score = np.concatenate((risk[case], risk[control]))
    return float(roc_auc_score(labels, score, sample_weight=weights)), n_case, n_control


def _read_run(path: Path, outcomes: list[str], horizon: int,
              *, allow_missing: bool = False) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = sorted(REQUIRED | {f"cif_{horizon}d"})
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty or not frame["split"].eq("tuning").all():
        raise ValueError(f"Predictions are not tuning-only: {path}")
    frame = frame.loc[frame["outcome"].isin(outcomes)].copy()
    if not allow_missing and frame["outcome"].nunique() != len(outcomes):
        raise ValueError(f"Missing requested outcome in {path}")
    if frame.duplicated(["subject_id", "outcome"]).any():
        raise ValueError(f"Duplicate patient/outcome rows in {path}")
    return frame.sort_values(["outcome", "subject_id"]).reset_index(drop=True)


def _families(path: Path, outcomes: list[str]) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = payload.get("cross_outcome", {}).get("outcome_families") if "cross_outcome" in payload else payload
    if not isinstance(groups, dict):
        raise ValueError("Family file must contain outcome-family mapping")
    mapping = {outcome: family for family, members in groups.items() for outcome in members}
    if not set(outcomes).issubset(mapping):
        raise ValueError("Family file does not cover the requested outcomes")
    return mapping


def _registry_outcomes(path: Path) -> list[str]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Registry must be a YAML mapping")
    approved = registry.get("outcomes")
    if not isinstance(approved, (list, dict)) or not approved:
        raise ValueError("Registry must contain a nonempty outcomes list or mapping")
    names = list(approved)
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Registry outcomes must have unique nonempty names")
    return names


def compare_predictions(runs: dict[str, pd.DataFrame], *, baseline: str,
                        outcomes: list[str], horizon: int, n_bootstrap: int,
                        seed: int, family_by_outcome: dict[str, str],
                        min_cases: int = 20,
                        skip_under_supported: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if baseline not in runs or len(runs) < 2 or n_bootstrap < 1:
        raise ValueError("Need a baseline, at least two runs, and positive bootstrap count")
    rng = np.random.default_rng(seed)
    reference = runs[baseline]
    keys = ["subject_id", "outcome"]
    reference_keys = reference[keys + ["cohort", "time_days", "event"]]
    for name, frame in runs.items():
        check = frame[keys + ["cohort", "time_days", "event"]]
        if not check.equals(reference_keys):
            raise ValueError(f"Patient sets, order, cohort or labels differ: {name}")
    estimates, contrasts, inventory = [], [], {}
    risk_col = f"cif_{horizon}d"
    for outcome in outcomes:
        selection = reference["outcome"].eq(outcome).to_numpy()
        if not selection.any():
            if skip_under_supported:
                inventory[outcome] = {"status": "excluded_no_predictions",
                                      "n_patients": 0, "n_cases": 0,
                                      "n_controls": 0, "n_valid_bootstrap": 0}
                continue
            raise ValueError(f"No predictions for {outcome}")
        times = reference.loc[selection, "time_days"].to_numpy(float)
        events = reference.loc[selection, "event"].to_numpy(int)
        risks = {name: frame.loc[selection, risk_col].to_numpy(float)
                 for name, frame in runs.items()}
        _, n_cases, n_controls = fast_cif_auc(times, events, risks[baseline], horizon)
        if n_cases < min_cases or n_controls < min_cases:
            if skip_under_supported:
                inventory[outcome] = {"status": "excluded_low_support",
                                      "n_patients": len(times), "n_cases": n_cases,
                                      "n_controls": n_controls,
                                      "n_valid_bootstrap": 0}
                continue
            raise ValueError(f"Too little primary-event/control support for {outcome}: "
                             f"{n_cases}/{n_controls}")
        for left, right in combinations(risks, 2):
            if np.array_equal(risks[left], risks[right]):
                raise ValueError(
                    f"Identical patient-level predictions for {left} and {right} "
                    f"on {outcome}; audit run/checkpoint paths before comparison"
                )
        scores = {name: fast_cif_auc(times, events, values, horizon)[0]
                  for name, values in risks.items()}
        draws = {name: [] for name in runs}
        delta_draws = {name: [] for name in runs if name != baseline}
        for _ in range(n_bootstrap):
            index = rng.integers(0, len(times), size=len(times))
            t, e = times[index], events[index]
            sampled = {name: fast_cif_auc(t, e, risk[index], horizon)[0]
                       for name, risk in risks.items()}
            if not all(np.isfinite(value) for value in sampled.values()):
                continue
            for name, value in sampled.items():
                draws[name].append(value)
                if name != baseline:
                    delta_draws[name].append(value - sampled[baseline])
        if min(map(len, draws.values())) < max(50, int(.8 * n_bootstrap)):
            if skip_under_supported:
                inventory[outcome] = {"status": "excluded_insufficient_valid_bootstrap",
                                      "n_patients": len(times), "n_cases": n_cases,
                                      "n_controls": n_controls,
                                      "n_valid_bootstrap": len(draws[baseline])}
                continue
            raise ValueError(f"Too few valid bootstrap replicates for {outcome}")
        common = {"outcome": outcome, "outcome_family": family_by_outcome[outcome],
                  "cohort": "pooled", "horizon_days": horizon,
                  "n": len(times), "n_events": n_cases}
        for name, value in scores.items():
            lo, hi = np.quantile(draws[name], [.025, .975])
            estimates.append({**common, "model": name, "estimate": value,
                              "lower": float(lo), "upper": float(hi)})
            if name != baseline:
                lo, hi = np.quantile(delta_draws[name], [.025, .975])
                contrasts.append({**common, "contrast": f"{name}_minus_{baseline}",
                                  "comparator": name, "baseline": baseline,
                                  "estimate": value - scores[baseline],
                                  "lower": float(lo), "upper": float(hi),
                                  "probability_superior": float(np.mean(np.asarray(delta_draws[name]) > 0))})
        inventory[outcome] = {"status": "included", "n_patients": len(times), "n_cases": n_cases,
                              "n_controls": n_controls,
                              "n_valid_bootstrap": len(draws[baseline])}
    if not estimates:
        raise ValueError("No outcomes had sufficient support for the forest")
    report = {"status": "tuning_only_fixed_selected_checkpoints_selection_bias_not_corrected",
              "baseline": baseline, "horizon_days": horizon,
              "n_bootstrap_requested": n_bootstrap, "seed": seed,
              "n_outcomes_requested": len(outcomes),
              "n_outcomes_included": sum(item["status"] == "included" for item in inventory.values()),
              "n_outcomes_excluded": sum(item["status"] != "included" for item in inventory.values()),
              "patient_matching": "exact patient/outcome/cohort/time/event identity across runs",
              "bootstrap_unit": "patient within each outcome; same resample across models",
              "metric": "IPCW cumulative-incidence AUC; censoring KM re-estimated per draw",
              "inventory": inventory}
    return pd.DataFrame(estimates), pd.DataFrame(contrasts), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PREDICTIONS.parquet")
    parser.add_argument("--baseline", required=True)
    outcome_source = parser.add_mutually_exclusive_group(required=True)
    outcome_source.add_argument("--outcome", action="append")
    outcome_source.add_argument("--registry", type=Path,
                                help="Use every outcome in the approved registry, in registry order")
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--n-bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-under-supported", action="store_true",
                        help="Record and omit outcomes below support/valid-bootstrap thresholds")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.registry:
        outcomes = _registry_outcomes(args.registry)
    else:
        outcomes = list(dict.fromkeys(args.outcome))
    paths = {}
    for item in args.run:
        name, sep, value = item.partition("=")
        if not sep or not name or not value or name in paths:
            parser.error(f"Expected unique NAME=PATH, got {item!r}")
        paths[name] = Path(value)
    families = _families(args.family_map, outcomes)
    frames = {name: _read_run(path, outcomes, args.horizon,
                              allow_missing=args.skip_under_supported)
              for name, path in paths.items()}
    estimates, contrasts, report = compare_predictions(
        frames, baseline=args.baseline, outcomes=outcomes,
        horizon=args.horizon, n_bootstrap=args.n_bootstrap,
        seed=args.seed, family_by_outcome=families,
        skip_under_supported=args.skip_under_supported,
    )
    report["sources"] = {name: str(path.resolve()) for name, path in paths.items()}
    args.output_dir.mkdir(parents=True)
    estimates.to_csv(args.output_dir / "tuning_forest_estimates.csv", index=False)
    contrasts.to_csv(args.output_dir / "tuning_forest_contrasts.csv", index=False)
    (args.output_dir / "tuning_forest_manifest_INTERNAL.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "outcomes": outcomes, "runs": list(paths),
                      "n_bootstrap": args.n_bootstrap}, indent=2))


if __name__ == "__main__":
    main()
