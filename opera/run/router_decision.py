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

Initialisation rule (``--phase init_rule``, lock addendum of 2026-10-25; stage-2
bundle): R_L, the locked router, stays primary unless RSI_L (the same router
with the shared logit started at 3.0) beats it by more than the noise floor in
panel AUC and passes the IPA guard; then one more seed of each is required
before RSI_L can be adopted. The noise floor here is N_L (the locked router with
every source switched off) against J_L. Both routers' routing is summarised
from their exported route weights: entropy, effective number of sources,
shared weight, and the weight on the patient's own disease and own outcome
versus borrowed ones. The router lock itself is never changed by this step.

Router-fix rule (``--phase router_fix_rule``, lock addendum 2026-10-26;
stage-3 bundle). Routing health, per routed fit, on contributions (route
weight x the source's support shrinkage) over the named disease and outcome
sources: (1) no single source is the top contributor in more than half of the
rows; (2) at least 3 different sources are the most common top borrowed
source across diseases; (3) the median effective number of contributing
sources exceeds 1.5. The seed noise floor is |panel AUC of R_AB (seed 43)
minus R_AB (seed 42)|. Candidates, simplest first: R_A, R_AB, then R_ABT only
if R_AB is unhealthy. Among healthy candidates the best panel AUC wins, ties
within the seed noise floor to the simplest; it replaces R_L unless its panel
AUC is below R_L's by more than the noise floor. The borrowing contrasts
(chosen vs RBH_AB and N_A; RBH_AB vs N_A; N_A vs J_L) are reported, never
used to choose.

Stage 4 (same phase, run on the stage-4 bundle; lock addendum "own sources
outside the mix"): R_O replaces R_L if its routing is healthy by the same three
criteria, with contributions measured as route weight x the source's
correction size rebuilt from the checkpoint, and its panel AUC is not below
R_L's by more than |R_O_s43 - R_O|; otherwise the stage-3 outcome stands.
Borrowing is reported as R_O vs RBH_O, own corrections as RBH_O vs N_A.

Decisions are written only with --write; the report is always saved.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from opera.evaluation.secondary_analysis import (
    _interval,
    normalize_predictions,
    paired_analysis,
)

PHASES = ("source_screen", "rank_screen", "query_screen", "init_rule", "router_fix_rule")
DEFAULT_RANK, DEFAULT_QUERY = 8, ("interacting", 128)  # used by the source and rank steps
HORIZONS = (30, 60, 90, 180, 365, 730)


def panel(contrasts, draws, contrast, metric, *, minimum=5, return_draws=False):
    """Equal-weight panel over estimable cells: disease, then outcome, then horizon.

    ``return_draws`` adds the per-replicate panel gains (``draws``).
    """
    rows = contrasts.loc[(contrasts.contrast == contrast) & (contrasts.metric == metric)]
    rows = rows.loc[np.isfinite(rows.gain) & (rows.n_cases >= minimum)
                    & (rows.n_controls >= minimum)].copy()
    if rows.empty:
        return {"gain": np.nan, "lower": np.nan, "upper": np.nan, "n_cells": 0,
                "n_diseases": 0, **({"draws": np.array([])} if return_draws else {})}
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
            "n_diseases": int(diseases), **({"draws": weighted} if return_draws else {})}


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


def read_membership(path):
    metadata = (pd.read_parquet(path) if path.suffix in {".parquet", ".pq"}
                else pd.read_csv(path))
    metadata["subject_id"] = metadata.subject_id.astype(str)
    return metadata.drop_duplicates("subject_id")


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


def routing_diagnostics(path):
    """Routing summary from one fit's exported route weights (one row per
    patient and outcome): overall and per disease, averaged over rows."""
    from opera.functional.cohort_groups import fine_to_grouped

    routes = pd.read_parquet(path)
    rows, top_sources, mismatched = [], {}, 0
    for record in routes.itertuples(index=False):
        disease = json.loads(record.disease_weights)
        outcome = json.loads(record.outcome_weights)
        weights = np.array([record.shared_weight, *disease.values(), *outcome.values()])
        positive = weights[weights > 0]
        entropy = float(-(positive * np.log(positive)).sum())
        # The export's cohort is the disease group (CLL_SLL); the patient's own
        # disease source is the one at its fine index (sources follow fine_names).
        own_name = list(disease)[record.fine_index]
        fine = own_name.split(":")[-1]
        mismatched += fine_to_grouped(fine) not in (record.cohort, fine)
        own_disease = disease[own_name]
        own_outcome = sum(w for k, w in outcome.items() if _same(k, record.outcome))
        names = ["shared", *disease, *outcome]
        top = names[int(weights.argmax())]
        top_sources[top] = top_sources.get(top, 0) + 1
        rows.append({
            "cohort": fine,
            "entropy": entropy,
            "effective_sources": math.exp(entropy),
            "shared": record.shared_weight,
            "own_disease": own_disease,
            "own_outcome": own_outcome,
            "borrowed_disease": sum(disease.values()) - own_disease,
            "borrowed_outcome": sum(outcome.values()) - own_outcome,
            "top_weight": float(weights.max()),
        })
    table = pd.DataFrame(rows)
    columns = [c for c in table if c != "cohort"]
    if mismatched:
        raise ValueError(f"{mismatched} route rows: fine index does not match the disease group")
    common = sorted(top_sources.items(), key=lambda item: -item[1])[:5]
    return {
        "rows": len(table),
        "overall": table[columns].mean().round(4).to_dict(),
        "top_sources": [{"source": name, "share_of_rows": round(n / len(table), 4)}
                        for name, n in common],
        "by_disease": table.groupby("cohort")[columns].mean().round(4)
        .reset_index().to_dict("records"),
    }


def routing_health(routes_path, support_path, sizes=None):
    """The prespecified routing-health criteria for one routed fit.

    Contributions are route weight x support shrinkage (stage 3) or, given
    ``sizes`` (source -> mean correction size after shrinkage, rebuilt from the
    checkpoint), route weight x correction size (stage 4)."""
    routes = pd.read_parquet(routes_path)
    support = json.loads(Path(support_path).read_text())
    shrinkage = dict(zip(support["sources"], support["shrinkage"]))
    if sizes is not None:
        shrinkage = {name: sizes[name] for name in shrinkage}
    tops, effective, top_borrowed = {}, [], {}
    for record in routes.itertuples(index=False):
        weights = {**json.loads(record.disease_weights), **json.loads(record.outcome_weights)}
        names = list(weights)
        contribution = np.array([weights[n] * shrinkage[n] for n in names])
        total = contribution.sum()
        if total <= 0:
            continue
        share = contribution / total
        positive = share[share > 0]
        effective.append(float(np.exp(-(positive * np.log(positive)).sum())))
        top = names[int(share.argmax())]
        tops[top] = tops.get(top, 0) + 1
        own = {list(json.loads(record.disease_weights))[record.fine_index],
               *(n for n in names if _same(n, record.outcome))}
        borrowed = [(share[i], n) for i, n in enumerate(names) if n not in own]
        if borrowed:
            disease = list(json.loads(record.disease_weights))[record.fine_index]
            counts = top_borrowed.setdefault(disease, {})
            name = max(borrowed)[1]
            counts[name] = counts.get(name, 0) + 1
    rows = sum(tops.values())
    top_source, top_count = max(tops.items(), key=lambda item: item[1])
    per_disease = {d: max(c.items(), key=lambda item: item[1])[0] for d, c in top_borrowed.items()}
    criteria = {
        "top_source_share": {"source": top_source, "share": round(top_count / rows, 4),
                             "passed": top_count / rows <= 0.5},
        "distinct_top_borrowed_across_diseases": {
            "count": len(set(per_disease.values())), "passed": len(set(per_disease.values())) >= 3},
        "median_effective_sources": {"value": round(float(np.median(effective)), 3),
                                     "passed": float(np.median(effective)) > 1.5},
    }
    return {"healthy": all(c["passed"] for c in criteria.values()), "criteria": criteria,
            "top_borrowed_by_disease": per_disease}


def router_fix_rule(args, metadata):
    """Health, seed noise, the choice of router, and the borrowing contrasts."""
    from opera.run.paper_prediction_queue import bundle_runs, canonical_models

    bundle, rows = bundle_runs(args.bundle_dir)
    by_alias = {}
    for row, _, config in rows:
        for name in canonical_models(row, config, bundle):
            by_alias.setdefault(name, []).append(row)
    names = ("J_L", "R_L", "N_A_L", "RBH_AB_L", "R_A_L", "R_AB_L", "R_ABT_L", "R_AB_s43_L")
    stage4 = bool(by_alias.get("R_O_L"))  # own sources outside the mix (addendum)
    if stage4:
        names += STAGE4_FITS
    wanted = {}
    for name in names:
        fits = by_alias.get(name, [])
        if len(fits) != 1:
            raise ValueError(f"The router-fix rule needs exactly one {name} fit in the "
                             f"bundle; found {len(fits)}")
        wanted[name] = fits[0]
    frames = []
    for name, row in wanted.items():
        frame = load_tuning(row, args.prediction_dir, metadata, name, HORIZONS)
        frame["seed"] = 42  # pair the seed-43 replicate on the same patients
        frames.append(frame)
    predictions, _ = normalize_predictions(pd.concat(frames, ignore_index=True), "cif_ipcw")
    contrasts = {
        "seed_noise": "gain(R_AB_s43_L,R_AB_L)",
        **{f"{n}_vs_R_L": f"gain({n},R_L)" for n in ("R_A_L", "R_AB_L", "R_ABT_L")},
        **{f"{a}_vs_{b}": f"gain({a},{b})" for a, b in (
            ("R_AB_L", "R_A_L"), ("R_ABT_L", "R_A_L"), ("R_ABT_L", "R_AB_L"))},
        **{f"{n}_vs_RBH_AB_L": f"gain({n},RBH_AB_L)" for n in ("R_A_L", "R_AB_L", "R_ABT_L")},
        **{f"{n}_vs_N_A_L": f"gain({n},N_A_L)" for n in ("R_A_L", "R_AB_L", "R_ABT_L", "RBH_AB_L")},
        "N_A_L_vs_J_L": "gain(N_A_L,J_L)", "R_L_vs_J_L": "gain(R_L,J_L)",
    }
    if stage4:
        contrasts |= {
            "seed_noise_O": "gain(R_O_s43_L,R_O_L)",
            # claim c: borrowing, with own sources applied outside the mix
            "R_O_L_vs_RBH_O_L": "gain(R_O_L,RBH_O_L)",
            # own corrections at full weight, without borrowing
            "RBH_O_L_vs_N_A_L": "gain(RBH_O_L,N_A_L)",
            "R_O_L_vs_R_L": "gain(R_O_L,R_L)", "R_O_L_vs_R_AB_L": "gain(R_O_L,R_AB_L)",
            "R_O_L_vs_J_L": "gain(R_O_L,J_L)",
        }
    tables = paired_analysis(predictions, contrasts, replicates=args.replicates, seed=2026,
                             patient_gain_contrasts=())
    summary = {name: {metric: panel(tables["contrasts"], tables["bootstrap_draws"], name, metric)
                      for metric in ("auc", "ipa")} for name in contrasts}
    noise = abs(summary["seed_noise"]["auc"]["gain"])
    by_disease = {}
    for name in ("R_AB_L_vs_RBH_AB_L", "R_AB_L_vs_N_A_L", "RBH_AB_L_vs_N_A_L",
                 "R_AB_L_vs_R_L", "N_A_L_vs_J_L",
                 *(("R_O_L_vs_RBH_O_L", "RBH_O_L_vs_N_A_L", "R_O_L_vs_R_L") if stage4 else ())):
        cells, draws = tables["contrasts"], tables["bootstrap_draws"]
        by_disease[name] = {
            cohort: {k: panel(cells.loc[cells.cohort == cohort], draws.loc[draws.cohort == cohort],
                              name, "auc")[k] for k in ("gain", "lower", "upper", "n_cells")}
            for cohort in sorted(cells.cohort.unique())}

    health = {}
    for name in ("R_L", "R_A_L", "R_AB_L", "R_ABT_L", "R_AB_s43_L"):
        directory = Path(args.prediction_dir) / wanted[name]["run_directory"] / "tuning"
        support = directory / "source_route_support.json"
        health[name] = (routing_health(directory / "source_route_weights.parquet", support)
                        if support.is_file() else {"missing": f"{support} (re-evaluate with routes)"})

    def healthy(name):
        return bool(health[name].get("healthy"))

    candidates = ["R_A_L", "R_AB_L"] + ([] if healthy("R_AB_L") else ["R_ABT_L"])
    eligible = [n for n in candidates if healthy(n)]
    vs_r = {n: summary[f"{n}_vs_R_L"]["auc"]["gain"] for n in candidates}
    if eligible:
        best = max(vs_r[n] for n in eligible)
        tied = [n for n in eligible if vs_r[n] >= best - noise]
        choice = tied[0]  # simplest first
        primary = choice if vs_r[choice] >= -noise else "R_L"
    else:
        choice, primary = None, "R_L"
    result = {"seed_noise_auc": noise, "health": health, "candidates": candidates,
              "eligible": eligible, "chosen_fix": choice, "primary": primary,
              "contrasts": summary, "by_disease_auc": by_disease}
    if stage4:
        result["stage3_primary"] = primary
        result["stage4"] = stage4_rule(args, wanted, summary)
        result["primary"] = result["stage4"]["primary"] or primary
    return wanted, result


STAGE4_FITS = ("R_O_L", "RBH_O_L", "R_O_s43_L")


def stage4_rule(args, wanted, summary):
    """Lock addendum, own sources outside the mix: R_O replaces R_L if its
    routing is healthy (contributions = weight x correction size, over borrowed
    sources) and its panel AUC is not below R_L's by more than |R_O_s43 - R_O|;
    otherwise the stage-3 rule decides (primary None here)."""
    from opera.run.audit_router_hub import correction_sizes

    noise = abs(summary["seed_noise_O"]["auc"]["gain"])
    runs = Path(args.runs_dir or Path(args.prediction_dir).parent / "neural_runs")
    health = {}
    for name in ("R_O_L", "R_O_s43_L"):
        run = wanted[name]["run_directory"]
        directory = Path(args.prediction_dir) / run / "tuning"
        needed = [directory / f for f in ("source_route_weights.parquet",
                                          "source_route_support.json", "patient_embeddings.npz")]
        absent = [str(p) for p in needed if not p.is_file()]
        if absent:
            health[name] = {"missing": absent}
            continue
        sizes = correction_sizes(runs / run, directory)
        health[name] = routing_health(needed[0], needed[1], sizes=sizes)
    healthy = bool(health["R_O_L"].get("healthy"))
    vs_r = summary["R_O_L_vs_R_L"]["auc"]["gain"]
    replaces = healthy and vs_r >= -noise
    return {"seed_noise_auc": noise, "health": health, "R_O_vs_R_L_auc": vs_r,
            "borrowing_auc": summary["R_O_L_vs_RBH_O_L"]["auc"],
            "replaces_R_L": replaces, "primary": "R_O_L" if replaces else None}


def _same(source_name, target):
    """Route-export source names may carry a prefix (e.g. "disease:CLL")."""
    return str(source_name) == str(target) or str(source_name).split(":")[-1] == str(target)


def init_rule(args, metadata):
    """R_L vs RSI_L on the tuning panel, with N_L vs J_L as the noise floor."""
    from opera.run.paper_prediction_queue import bundle_runs, canonical_models

    bundle, rows = bundle_runs(args.bundle_dir)
    by_alias = {}
    for row, _, config in rows:
        for name in canonical_models(row, config, bundle):
            by_alias.setdefault(name, []).append(row)
    wanted = {}
    for name in ("J_L", "N_L", "R_L", "RSI_L"):
        fits = by_alias.get(name, [])
        if len(fits) != 1:
            raise ValueError(f"The initialisation rule needs exactly one {name} fit in "
                             f"the stage-2 bundle; found {len(fits)}")
        wanted[name] = fits[0]
    frames = [load_tuning(row, args.prediction_dir, metadata, name, HORIZONS)
              for name, row in wanted.items()]
    predictions, _ = normalize_predictions(pd.concat(frames, ignore_index=True), "cif_ipcw")
    result = decide(predictions, joint="J_L", null="N_L", options=["R_L", "RSI_L"],
                    gate_tolerance=args.gate_tolerance, replicates=args.replicates)
    outcome = {"R_L": "keep_R", "RSI_L": "second_seed_required"}[result["chosen"]]
    diagnostics = {}
    for name in ("R_L", "RSI_L"):
        path = (Path(args.prediction_dir) / wanted[name]["run_directory"] / "tuning"
                / "source_route_weights.parquet")
        diagnostics[name] = (routing_diagnostics(path) if path.is_file() else
                             {"missing": f"{path} (evaluate with --export-routes)"})
    return wanted, result, outcome, diagnostics


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
    parser.add_argument("--runs-dir", type=Path,
                        help="neural run directories (default: <prediction-dir>/../neural_runs)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router-lock", type=Path,
                        help="query step: also write the final router lock here")
    parser.add_argument("--gate-tolerance", type=float, default=0.005)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--write", action="store_true",
                        help="Record the decision (default: report only)")
    args = parser.parse_args()

    decisions = json.loads(args.decisions.read_text()) if args.decisions.is_file() else {}
    if args.phase == "router_fix_rule":
        wanted, result = router_fix_rule(args, read_membership(args.membership))
        record = {"phase": "router_fix_rule",
                  "decided_at": datetime.now(timezone.utc).isoformat(),
                  "fits": {name: row["name"] for name, row in wanted.items()},
                  "rules": "lock addendum 2026-10-26 (see the module docstring)",
                  "replicates": args.replicates, "estimand": "cif_ipcw", **result}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = args.output_dir / "router_decision_router_fix_rule.json"
        report.write_text(json.dumps(record, indent=2, default=float) + "\n")
        auc = {name: [round(v["auc"][k], 4) for k in ("gain", "lower", "upper")]
               for name, v in result["contrasts"].items()}
        stage4 = result.get("stage4")
        print(json.dumps({"primary": result["primary"], "chosen_fix": result["chosen_fix"],
                          "seed_noise_auc": result["seed_noise_auc"],
                          **({"stage4": {k: stage4[k] for k in (
                              "seed_noise_auc", "R_O_vs_R_L_auc", "borrowing_auc",
                              "replaces_R_L")} | {"healthy": {
                                  n: h.get("healthy", h) for n, h in stage4["health"].items()}}}
                             if stage4 else {}),
                          "healthy": {n: h.get("healthy", h) for n, h in result["health"].items()},
                          "auc_gain_with_interval": auc, "report": str(report)},
                         indent=2, default=float))
        if args.write:
            decisions["router_fix"] = {"primary": result["primary"],
                                       "decided_at": record["decided_at"], "report": str(report)}
            args.decisions.write_text(json.dumps(decisions) + "\n")
            print(f"Recorded router-fix outcome {result['primary']!r} in {args.decisions}")
        return
    if args.phase == "init_rule":
        metadata = read_membership(args.membership)
        wanted, result, outcome, diagnostics = init_rule(args, metadata)
        record = {"phase": "init_rule", "decided_at": datetime.now(timezone.utc).isoformat(),
                  "fits": {name: row["name"] for name, row in wanted.items()},
                  "rules": {"choice": "R_L primary; RSI_L only if better than R_L by more "
                            "than the noise floor (N_L vs J_L) in panel AUC and within "
                            "the IPA guard, and then only after a second seed of each",
                            "replicates": args.replicates, "horizons": list(HORIZONS),
                            "estimand": "cif_ipcw"},
                  "outcome": outcome, **result, "routing": diagnostics}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = args.output_dir / "router_decision_init_rule.json"
        report.write_text(json.dumps(record, indent=2, default=float) + "\n")
        rsi = next(o for o in result["options"] if o["model"] == "RSI_L")
        print(json.dumps({
            "outcome": outcome, "noise_floor_auc": result["noise"]["floor_auc"],
            "noise_floor_ipa": result["noise"]["floor_ipa"],
            "rsi_minus_r_auc": [rsi["auc"]["gain"], rsi["auc"]["lower"], rsi["auc"]["upper"]],
            "rsi_minus_r_ipa": rsi["ipa"]["gain"], "rsi_eligible": rsi["eligible"],
            "vs_joint_auc": {o["model"]: o["vs_joint"]["auc"]["gain"] for o in result["options"]},
            "routing_overall": {name: d.get("overall", d) for name, d in diagnostics.items()},
            "report": str(report)}, indent=2, default=float))
        if args.write:
            decisions["initialisation"] = {"outcome": outcome, "decided_at": record["decided_at"],
                                           "report": str(report)}
            args.decisions.write_text(json.dumps(decisions) + "\n")
            print(f"Recorded initialisation outcome {outcome!r} in {args.decisions}")
        return
    options = roles(args.phase, decisions)
    _, rows = bundle_runs(args.bundle_dir)
    joint = next(row for row, _, config in rows if row["phase"] == "supervision"
                 and row.get("regime") == "J" and row["origin"] == "L")
    routers = {router_key(config): row for row, _, config in rows
               if row["phase"] in PHASES and row["origin"] == "L"}
    null = ("none", DEFAULT_RANK, *DEFAULT_QUERY)
    wanted = {"J": joint, alias(null): routers[null],
              **{alias(key): routers[key] for key in options}}

    metadata = read_membership(args.membership)
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
