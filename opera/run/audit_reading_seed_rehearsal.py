"""Paired tuning comparisons across explicitly named training-seed replicas.

Uses existing prediction exports. Training seeds remain in the fit manifest;
seed=0 in the paired tables is an analysis context, never a training seed.
No model selection, training or held-out evaluation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.evaluation.secondary_analysis import effect_summaries, normalize_predictions, paired_analysis
from opera.run.analyse_locked_predictions import read_artifact, read_table
from opera.run.run_paper_queue import file_hash

PAIRS = {
    "OAOWN_J42": ("J_OA_OWN_L", "J_L"),
    "OAOWN_J_s43": ("J_OA_OWN_s43_L", "J_s43_L"),
    "OAOWN_NF": ("J_OA_OWN_L", "J_OA_OWN_s43_L"),
    "OAOWN_NF44": ("J_OA_OWN_L", "J_OA_OWN_s44_L"),
    "OAOWN_NF45": ("J_OA_OWN_L", "J_OA_OWN_s45_L"),
    "OAOWN_NF4345": ("J_OA_OWN_s43_L", "J_OA_OWN_s45_L"),
    "J_NF": ("J_L", "J_s43_L"),
    "AB_NF": ("R_AB_L", "R_AB_s43_L"),
    "OAOWN_AB": ("J_OA_OWN_L", "R_AB_L"),
}


def pairing_context(frame):
    """Pair replicas as distinct models, without altering the primary frame."""
    if set(frame.split) != {"tuning"}:
        raise ValueError("Seed rehearsal accepts tuning predictions only")
    seeds = frame.groupby("model").seed.unique()
    if any(len(values) != 1 for values in seeds):
        raise ValueError("Each replica alias must identify exactly one training seed")
    identities = {model: int(values[0]) for model, values in seeds.items()}
    keys = ["subject_id", "split", "cohort", "outcome", "horizon_days"]
    if frame.duplicated(["model", *keys]).any():
        raise ValueError("Duplicate patient/task rows within a replica")
    if not frame.groupby(keys).model.nunique().eq(len(identities)).all():
        raise ValueError("Replica patient/task rosters differ; no silent intersection")
    context = frame.copy()
    context["training_seed"] = context.seed
    context["seed"] = 0
    return context, identities


def analyse(frame, cell_metadata, *, replicates=1000, min_cases=5, min_controls=5):
    context, identities = pairing_context(frame)
    context, _ = normalize_predictions(context, "cif_ipcw")
    pairs = {name: pair for name, pair in PAIRS.items() if set(pair) <= set(identities)}
    if not pairs:
        raise ValueError("No registered replica comparison is available")
    tables = paired_analysis(context, {name: f"gain({a},{b})" for name, (a, b) in pairs.items()},
                             replicates=replicates, seed=2026, patient_gain_contrasts=[],
                             min_cases=min_cases, min_controls=min_controls)
    fit_pairs = pd.DataFrame([dict(contrast=name, left_model=a, right_model=b,
                                   left_training_seed=identities[a], right_training_seed=identities[b])
                              for name, (a, b) in pairs.items()])
    summaries = []
    for weighting in ("equal_cell", "disease_balanced"):
        _, summary = effect_summaries(tables["contrasts"], tables["bootstrap_draws"], cell_metadata,
                                      strata=["cohort"], aggregation=weighting)
        summaries.append(summary.assign(weighting=weighting))
    tables["effect_summaries"] = pd.concat(summaries, ignore_index=True).merge(
        fit_pairs, on="contrast", validate="many_to_one")
    for name in ("contrasts", "contrast_coverage"):
        tables[name] = tables[name].merge(fit_pairs, on="contrast", validate="many_to_one")
    # Absolute AUCs on the same finite cell roster; seed ranges are descriptive.
    estimates = tables["estimates"].loc[tables["estimates"].metric.eq("auc")]
    panel = estimates.pivot(index=["cohort", "outcome", "horizon_days"], columns="model", values="estimate")
    model_rows = []
    for horizon in [0, *sorted(frame.horizon_days.unique())]:
        block = panel if horizon == 0 else panel.loc[panel.index.get_level_values("horizon_days") == horizon]
        block = block.dropna(subset=list(identities))
        if block.empty:
            continue
        keys = block.reset_index()[["cohort", "outcome", "horizon_days"]]
        counts = keys.groupby("cohort").outcome.transform("nunique")
        horizons = keys.groupby(["cohort", "outcome"]).horizon_days.transform("nunique")
        balanced = 1 / (keys.cohort.nunique() * counts * horizons)
        for weighting, weights in (("equal_cell", np.full(len(block), 1 / len(block))),
                                    ("disease_balanced", balanced.to_numpy())):
            for model in identities:
                model_rows.append(dict(model=model, training_seed=identities[model],
                                       horizon_days=int(horizon), weighting=weighting,
                                       n_cells=len(block), auc=float(np.dot(block[model], weights))))
    tables["model_auc"] = pd.DataFrame(model_rows)
    tables["fit_pairs"] = fit_pairs
    return tables, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[30, 60, 90, 180, 365, 730])
    args = parser.parse_args()
    if args.output_dir.exists() or args.replicates < 20:
        parser.error("Use a fresh output directory and at least 20 draws")
    config = yaml.safe_load(args.analysis_config.read_text())
    if config.get("estimand") != "cif_ipcw":
        parser.error("Use the primary cif_ipcw analysis inventory")
    metadata = read_table(config["patient_metadata"]).rename(columns=config.get("patient_metadata_column_map", {}))
    metadata.subject_id = metadata.subject_id.astype(str)
    wanted = {name for pair in PAIRS.values() for name in pair}
    specifications = [a for a in config["artifacts"] if a.get("model") in wanted and a.get("split") == "tuning"]
    if any(a.get("format") != "direct_cif" for a in specifications):
        parser.error("Replica audit requires native direct_cif exports")
    if not specifications:
        parser.error("No declared tuning replica artifacts")
    frames, inputs = [], {}
    for specification in specifications:
        path = Path(specification["path"])
        data, _ = read_artifact(specification | {"include_horizons_days": args.horizons}, metadata, "cif_ipcw")
        frames.append(data)
        inputs[str(path)] = file_hash(path)
    tables, identities = analyse(pd.concat(frames, ignore_index=True), read_table(config["cell_metadata"]),
                                 replicates=args.replicates,
                                 min_cases=config.get("performance_min_cases", 5),
                                 min_controls=config.get("performance_min_controls", 5))
    args.output_dir.mkdir(parents=True)
    for name in ("model_auc", "fit_pairs", "effect_summaries", "contrasts", "contrast_coverage"):
        tables[name].to_csv(args.output_dir / f"{name}.csv", index=False)
    manifest = dict(split="tuning", training_seeds=identities, pairing_context_seed=0,
                    missing_model_aliases=sorted(wanted - set(identities)),
                    interpretation="Matched-patient contrasts between fixed replicas. Intervals condition on these fits; seed range is not a confidence interval.",
                    input_sha256=inputs, config_sha256=file_hash(args.analysis_config),
                    cell_metadata_sha256=file_hash(config["cell_metadata"]),
                    metadata_sha256=file_hash(config["patient_metadata"]),
                    replicates=args.replicates, output_dir=str(args.output_dir.resolve()))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest | {"input_sha256": f"{len(inputs)} prediction artifacts"}, indent=2))


if __name__ == "__main__":
    main()
