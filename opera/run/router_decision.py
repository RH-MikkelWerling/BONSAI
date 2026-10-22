"""Apply the prespecified router rules to tuning predictions and record the decision.

Rules (docs/ANALYSIS_LOCK_DRAFT_20261015.md, "Router decisions", 2026-10-21):

* Tuning patients only, scored on the full-risk-set IPCW estimand, with one
  paired patient bootstrap (shared draws for every fit).
* Panel gain = AUC difference averaged over every estimable cell (>= 5 cases and
  >= 5 controls), each disease weighing equally, each outcome equally within a
  disease and each horizon equally within an outcome, so all six horizons count.
* Noise floor: the router with every source switched off (the joint model from
  a different initialisation) against the joint model; |panel AUC gain| is the
  noise floor. The machinery check (|gain| <= gate_tolerance) is reported, not
  used to choose.
* Each step chooses the best option: the highest panel AUC. Options within the
  noise floor of the best count as tied, and the simplest of them is chosen.
  An option is not eligible if its panel IPA is worse than the simplest
  option's by more than the IPA noise floor (calibration guard).
* Options, simplest first: sources disease_only, outcome_only, both; rank 8,
  16, 32; query additive 128, interacting 128, interacting 512.

Decisions are written only with --write; the report is always saved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from opera.evaluation.secondary_analysis import (
    _interval,
    normalize_predictions,
    paired_analysis,
)

PHASES = ("source_screen", "rank_screen", "query_screen")
DEFAULT_RANK, DEFAULT_QUERY = 8, ("interacting", 128)  # used by the source and rank steps
HORIZONS = (30, 60, 90, 180, 365, 730)


def panel(contrasts, draws, contrast, metric, *, minimum=5):
    """Equal-weight panel over estimable cells: disease, then outcome, then horizon."""
    rows = contrasts.loc[(contrasts.contrast == contrast) & (contrasts.metric == metric)]
    rows = rows.loc[np.isfinite(rows.gain) & (rows.n_cases >= minimum)
                    & (rows.n_controls >= minimum)].copy()
    if rows.empty:
        return {"gain": np.nan, "lower": np.nan, "upper": np.nan, "n_cells": 0,
                "n_diseases": 0}
    diseases = rows.cohort.nunique()
    outcomes = rows.groupby("cohort").outcome.transform("nunique")
    horizons = rows.groupby(["cohort", "outcome"]).horizon_days.transform("nunique")
    rows["weight"] = 1.0 / (diseases * outcomes * horizons)
    keys = ["split", "seed", "cohort", "outcome", "horizon_days"]
    sampled = draws.loc[(draws.contrast == contrast) & (draws.metric == metric)].merge(
        rows[[*keys, "weight"]], on=keys, validate="one_to_one")
    arrays = np.stack(sampled.gain_draws.to_numpy())
    # Per replicate, average over the cells that are finite in it (a small
    # tuning cell can lack cases in a resample), renormalising the weights.
    # The point estimate, and so every choice, uses all estimable cells.
    weights = np.where(np.isfinite(arrays), sampled.weight.to_numpy()[:, None], 0.0)
    total = weights.sum(axis=0)
    weighted = np.where(total > 0, np.nansum(arrays * weights, axis=0) / np.where(total > 0, total, 1),
                        np.nan)
    interval = _interval(weighted)
    return {"gain": float((rows.gain * rows.weight).sum()), "lower": interval["lower"],
            "upper": interval["upper"], "n_cells": int(len(rows)),
            "n_diseases": int(diseases)}


def decide(predictions, *, joint, null, options, gate_tolerance=0.005,
           replicates=1000, seed=2026):
    """Best option on tuning, ties within the noise floor to the simplest.

    ``options`` is ordered simplest first; gains are against the simplest.
    ``predictions`` holds normalized tuning predictions whose ``model`` column
    uses the aliases passed here.
    """
    simplest, others = options[0], list(options[1:])
    contrasts = {"noise": f"gain({null},{joint})"}
    # Every option against the joint model: the routing gain the paper asks about.
    contrasts.update({f"joint:{name}": f"gain({name},{joint})" for name in options})
    contrasts.update({f"vs:{name}": f"gain({name},{simplest})" for name in others})
    tables = paired_analysis(predictions, contrasts, replicates=replicates, seed=seed,
                             patient_gain_contrasts=())
    summary = {name: {metric: panel(tables["contrasts"], tables["bootstrap_draws"],
                                    name, metric) for metric in ("auc", "ipa")}
               for name in contrasts}
    noise = {metric: abs(summary["noise"][metric]["gain"]) for metric in ("auc", "ipa")}
    zero = {"gain": 0.0, "lower": np.nan, "upper": np.nan}
    rows = [{"model": simplest, "rank": 0, "auc": zero, "ipa": zero}] + [
        {"model": name, "rank": i + 1, "auc": summary[f"vs:{name}"]["auc"],
         "ipa": summary[f"vs:{name}"]["ipa"]} for i, name in enumerate(others)]
    for row in rows:
        row["eligible"] = bool(np.isfinite(row["auc"]["gain"]) and (
            row["rank"] == 0 or row["ipa"]["gain"] >= -noise["ipa"]))
    eligible = [row for row in rows if row["eligible"]]
    best = max(row["auc"]["gain"] for row in eligible)
    tied = [row for row in eligible if row["auc"]["gain"] >= best - noise["auc"]]
    chosen = min(tied, key=lambda row: row["rank"])
    for row in rows:
        row["best"] = row["auc"]["gain"] == best
        row["tied_with_best"] = row in tied
        row["vs_joint"] = {metric: summary[f"joint:{row['model']}"][metric]
                           for metric in ("auc", "ipa")}
    return {
        "noise": {"auc": summary["noise"]["auc"], "ipa": summary["noise"]["ipa"],
                  "floor_auc": noise["auc"], "floor_ipa": noise["ipa"],
                  "machinery_check": {"tolerance": gate_tolerance,
                                      "passed": bool(noise["auc"] <= gate_tolerance)}},
        "options": rows,
        "chosen": chosen["model"],
        "status": ("best" if chosen["auc"]["gain"] == best
                   else "simplest_within_noise_of_best"),
    }


def roles(phase, decisions):
    """The options of one step, as router keys, simplest first."""
    if phase == "source_screen":
        return [(source, DEFAULT_RANK, *DEFAULT_QUERY)
                for source in ("disease_only", "outcome_only", "both")]
    source = decisions.get("source_mode")
    if source is None:
        raise ValueError("The rank and query steps need the recorded source_mode")
    if phase == "rank_screen":
        return [(source, rank, *DEFAULT_QUERY) for rank in (8, 16, 32)]
    rank = decisions.get("rank")
    if rank is None:
        raise ValueError("The query step needs the recorded rank")
    return [(source, rank, "additive", 128), (source, rank, "interacting", 128),
            (source, rank, "interacting", 512)]


def router_key(config):
    risk = config["competing_risk"]
    return (risk.get("adapter_source_mode"), int(risk.get("adapter_bottleneck", 0)),
            risk.get("adapter_query_mode"), int(risk.get("adapter_query_hidden_dim", 0)))


def alias(key):
    source, rank, mode, width = key
    return f"R_{source}_r{rank}_{mode}{width}"


def load_tuning(row, prediction_dir, metadata, model, horizons):
    from opera.run.analyse_locked_predictions import read_artifact

    directory = Path(prediction_dir) / row["run_directory"] / "tuning"
    path = directory / "direct_cr_predictions.parquet"
    if not (path.is_file() and (directory / "evaluation_complete.json").is_file()):
        raise FileNotFoundError(
            f"No completed tuning predictions for {row['name']}: run "
            "`run_paper_experiments.sh evaluate --split tuning` for this fit first")
    data, _ = read_artifact(
        {"path": str(path), "format": "direct_cif", "model": model, "seed": row["seed"],
         "split": "tuning", "data_contract": "router_decision",
         "cohort_from_metadata": True, "include_horizons_days": list(horizons)},
        metadata, "cif_ipcw")
    return data


def main():
    from opera.run.paper_prediction_queue import bundle_runs

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True,
                        help="Population file with subject_id and cohort_fine")
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router-lock", type=Path,
                        help="query step: also write the final router lock here")
    parser.add_argument("--gate-tolerance", type=float, default=0.005)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--write", action="store_true",
                        help="Record the decision (default: report only)")
    args = parser.parse_args()

    decisions = json.loads(args.decisions.read_text()) if args.decisions.is_file() else {}
    options = roles(args.phase, decisions)
    _, rows = bundle_runs(args.bundle_dir)
    joint = next(row for row, _, config in rows if row["phase"] == "supervision"
                 and row.get("regime") == "J" and row["origin"] == "L")
    routers = {router_key(config): row for row, _, config in rows
               if row["phase"] in PHASES and row["origin"] == "L"}
    null = ("none", DEFAULT_RANK, *DEFAULT_QUERY)
    wanted = {"J": joint, alias(null): routers[null],
              **{alias(key): routers[key] for key in options}}

    metadata = (pd.read_parquet(args.membership) if args.membership.suffix in {".parquet", ".pq"}
                else pd.read_csv(args.membership))
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata = metadata.drop_duplicates("subject_id")
    frames = [load_tuning(row, args.prediction_dir, metadata, name, HORIZONS)
              for name, row in wanted.items()]
    predictions, _ = normalize_predictions(pd.concat(frames, ignore_index=True), "cif_ipcw")
    result = decide(predictions, joint="J", null=alias(null),
                    options=[alias(key) for key in options],
                    gate_tolerance=args.gate_tolerance, replicates=args.replicates)

    key = next(k for k in options if alias(k) == result["chosen"])
    record = {"phase": args.phase, "decided_at": datetime.now(timezone.utc).isoformat(),
              "fits": {name: row["name"] for name, row in wanted.items()},
              "rules": {"choice": "best panel AUC; ties within the noise floor to the simplest",
                        "gate_tolerance": args.gate_tolerance,
                        "replicates": args.replicates, "horizons": list(HORIZONS),
                        "estimand": "cif_ipcw"},
              "chosen_key": list(key), **result}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / f"router_decision_{args.phase}.json"
    report.write_text(json.dumps(record, indent=2, default=float) + "\n")

    update = {"source_screen": {"source_mode": key[0]}, "rank_screen": {"rank": key[1]},
              "query_screen": {"query_mode": key[2], "query_width": key[3]}}[args.phase]
    print(json.dumps({"chosen": result["chosen"], "status": result["status"],
                      "noise_floor_auc": result["noise"]["floor_auc"],
                      "machinery_check": result["noise"]["machinery_check"],
                      "options": [{"model": o["model"], "auc_gain_vs_simplest": o["auc"]["gain"],
                                   "auc_interval": [o["auc"]["lower"], o["auc"]["upper"]],
                                   "ipa_gain_vs_simplest": o["ipa"]["gain"],
                                   "auc_gain_vs_joint": o["vs_joint"]["auc"]["gain"],
                                   "auc_vs_joint_interval": [o["vs_joint"]["auc"]["lower"],
                                                             o["vs_joint"]["auc"]["upper"]],
                                   "eligible": o["eligible"], "tied_with_best": o["tied_with_best"]}
                                  for o in result["options"]],
                      "decision": update, "report": str(report)}, indent=2, default=float))
    if args.write:
        decisions.update(update)
        args.decisions.write_text(json.dumps(decisions) + "\n")
        if args.phase == "query_screen" and args.router_lock:
            lock = {k: decisions[k] for k in ("source_mode", "rank", "query_mode", "query_width")}
            args.router_lock.write_text(json.dumps(lock) + "\n")
        print(f"Recorded {update} in {args.decisions}")


if __name__ == "__main__":
    main()
