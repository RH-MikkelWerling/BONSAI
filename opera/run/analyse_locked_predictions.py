"""Run secondary OPERA analyses from an explicit locked prediction inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.evaluation.secondary_analysis import (
    CELL,
    ESTIMANDS,
    ROUTER_GATED_CONTRASTS,
    annotate_router_gate,
    audit_router_identity,
    calibration_summary,
    coverage_inventory,
    effect_summaries,
    negative_transfer_strata,
    normalize_predictions,
    paired_analysis,
    panel_estimates,
    panel_subgroup_gains,
    patient_bootstrap,
    reliability_tables,
    route_summaries,
    router_null_equivalence,
    seed_averaged_effects,
    subgroup_patient_gains,
    tuning_selected_strategies,
)


DEFAULT_PANEL_ESTIMATE_GROUPS = {
    "encoder_origin": ["J_L", "R_L", "J_E", "R_E", "J_EA", "R_EA"],
}


def read_table(path):
    path = Path(path)
    return (
        pd.read_parquet(path)
        if path.suffix in {".parquet", ".pq"}
        else pd.read_csv(path, dtype={"subject_id": str})
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_artifact(specification, metadata=None, estimand="binary_known_status_unweighted"):
    """Adapt native wide CIFs or explicitly keyed long prediction tables."""
    frame = read_table(specification["path"])
    frame["subject_id"] = frame.subject_id.astype(str)
    if specification.get("cohort_from_metadata", False):
        if metadata is None or "cohort_fine" not in metadata:
            raise ValueError("Pooled native exports need metadata.cohort_fine")
        frame = frame.merge(
            metadata[["subject_id", "cohort_fine"]].rename(
                columns={"cohort_fine": "analysis_cohort"}
            ),
            on="subject_id",
            how="left",
            validate="many_to_one",
        )
        if frame.analysis_cohort.isna().any():
            raise ValueError("Missing fine disease membership")
        frame["cohort"] = frame.pop("analysis_cohort")
    if specification.get("format") == "direct_cif":
        prefix = re.escape(specification.get("probability_prefix", "cif_"))
        columns = [column for column in frame if re.fullmatch(prefix + r"\d+d", column)]
        if specification.get("include_horizons_days"):
            wanted = {int(day) for day in specification["include_horizons_days"]}
            columns = [
                column
                for column in columns
                if int(re.search(r"(\d+)d$", column).group(1)) in wanted
            ]
            present = {
                int(re.search(r"(\d+)d$", column).group(1)) for column in columns
            }
            if present != wanted:
                raise ValueError(
                    "Native CIF artifact is missing a requested analysis horizon"
                )
        if not columns or not {"time_days", "event"} <= set(frame):
            raise ValueError("Native CIF export needs cif_*d, time_days and event")
        frame = frame.melt(
            id_vars=[column for column in frame if column not in columns],
            value_vars=columns,
            var_name="horizon",
            value_name="probability",
        )
        frame["horizon_days"] = (
            frame.pop("horizon").str.extract(r"(\d+)")[0].astype(int)
        )
        positive = (frame.event == 1) & (frame.time_days <= frame.horizon_days)
        frame["label"] = positive.astype(int)
        frame["known"] = (
            positive
            | (frame.time_days >= frame.horizon_days)
            | ((frame.event == 2) & (frame.time_days <= frame.horizon_days))
        )
    for key in (
        "model",
        "seed",
        "split",
        "cohort",
        "outcome",
        "horizon_days",
        "data_contract",
    ):
        if key in specification:
            if key in frame and not frame[key].eq(specification[key]).all():
                raise ValueError(f"Artifact declaration conflicts with rows: {key}")
            frame[key] = specification[key]
    return normalize_predictions(frame, estimand)


def json_safe(value):
    """Preserve YAML infinite bin edges in a strict JSON provenance manifest."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def read_route_artifact(config, metadata):
    """Accept long gates or the native JSON-bank route export."""
    frame = read_table(config["route_weights"])
    if "shared_weight" not in frame:
        return frame
    if metadata is None or "cohort_fine" not in metadata:
        raise ValueError("Native route exports need subject-to-fine metadata")
    frame["subject_id"] = frame.subject_id.astype(str)
    frame = frame.merge(
        metadata[["subject_id", "cohort_fine"]].rename(
            columns={"cohort_fine": "analysis_cohort"}
        ),
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    if frame.analysis_cohort.isna().any():
        raise ValueError("Native route export has unknown fine membership")
    rows = []
    for record in frame.to_dict("records"):
        identity = {key: record[key] for key in ("subject_id", "split", "outcome")}
        identity.update(
            model=config["route_model"],
            seed=config["route_seed"],
            cohort=record["analysis_cohort"],
        )
        rows.append(
            {
                **identity,
                "source": "shared",
                "bank": "shared",
                "weight": record["shared_weight"],
            }
        )
        for bank in ("disease", "outcome"):
            for source, weight in json.loads(record[bank + "_weights"]).items():
                rows.append(
                    {**identity, "source": source, "bank": bank, "weight": weight}
                )
    return pd.DataFrame(rows)


def run(config, output_dir):
    """Produce reviewable tables with explicit missing-input/coverage statuses."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            "Use a new analysis directory; existing results are immutable"
        )
    if not config.get("artifacts"):
        raise ValueError("Provide explicit artifact paths and run identities")
    estimand = config.get("estimand")
    if estimand not in ESTIMANDS:
        raise ValueError(
            f"Declare the analysis estimand explicitly: one of {ESTIMANDS}. "
            "Known-status and full-risk-set IPCW results are separate analyses."
        )
    metadata = (
        read_table(config["patient_metadata"])
        if config.get("patient_metadata")
        else None
    )
    if metadata is not None:
        mapping = config.get("patient_metadata_column_map", {})
        if any(source not in metadata for source in mapping):
            raise ValueError(
                "Patient metadata column mapping names absent source columns"
            )
        if any(
            target in metadata and source != target
            for source, target in mapping.items()
        ):
            raise ValueError(
                "Patient metadata mapping would overwrite existing columns"
            )
        metadata = metadata.rename(columns=mapping)
        metadata["subject_id"] = metadata.subject_id.astype(str)
        if "index_date" in metadata and "index_year" not in metadata:
            metadata["index_year"] = pd.to_datetime(
                metadata.index_date, errors="raise"
            ).dt.year
        if metadata.subject_id.duplicated().any():
            raise ValueError("Patient metadata has duplicate subject IDs")
    inputs, audit, hashes = [], [], {}
    for specification in config["artifacts"]:
        data, summary = read_artifact(specification, metadata, estimand)
        inputs.append(data)
        audit.append({"path": specification["path"], **summary})
        hashes[specification["path"]] = file_hash(specification["path"])
    predictions, combined_audit = normalize_predictions(
        pd.concat(inputs, ignore_index=True), estimand
    )
    if config.get("analysis_horizons_days"):
        predictions = predictions.loc[
            predictions.horizon_days.isin(config["analysis_horizons_days"])
        ].copy()
        if predictions.empty:
            raise ValueError("No predictions at the requested analysis horizons")
    contrast_manifest = yaml.safe_load(Path(config["contrast_manifest"]).read_text())
    contrasts = {
        name: value
        for family in contrast_manifest["essential_contrasts"].values()
        for name, value in family.items()
    }
    contrasts.update(config.get("extra_contrasts", {}))
    candidates = config.get("strategy_candidates", ["S_L", "D_L", "O_L"])
    selected, selection = tuning_selected_strategies(predictions, candidates)
    if not selected.empty:
        predictions = pd.concat([predictions, selected], ignore_index=True)
        contrasts.update(
            {
                "J_vs_best_tuning": "gain(J_L,BEST_TUNING)",
                "R_vs_best_tuning": "gain(R_L,BEST_TUNING)",
            }
        )
    replicates = int(config.get("bootstrap_replicates", 1000))
    if replicates < 1:
        raise ValueError("At least one paired bootstrap replicate is required")
    draws = patient_bootstrap(
        predictions, replicates, int(config.get("bootstrap_seed", 2026))
    )
    tables = paired_analysis(
        predictions,
        contrasts,
        replicates=replicates,
        draws=draws,
        patient_gain_contrasts=config.get("patient_gain_contrasts"),
    )
    # Absolute metrics on a common panel, e.g. joint/routed under each encoder origin.
    tables["panel_estimates"] = panel_estimates(
        tables["estimates"],
        tables.pop("model_bootstrap_draws"),
        config.get("panel_estimate_groups", DEFAULT_PANEL_ESTIMATE_GROUPS),
    )
    cell_metadata = read_table(config["cell_metadata"])
    for column, definition in config.get("cell_groups", {}).items():
        if definition.get("provenance") != "train_only":
            raise ValueError("Cell heterogeneity groups must be train-only")
        cell_metadata[column] = pd.cut(
            pd.to_numeric(cell_metadata[definition["column"]]),
            definition["bins"],
            include_lowest=True,
        ).astype(str)
    annotated, summaries = effect_summaries(
        tables["contrasts"],
        tables["bootstrap_draws"],
        cell_metadata,
        strata=config.get("cell_summary_strata"),
    )
    tables.update(
        contrasts=annotated,
        effect_summaries=summaries,
        calibration=reliability_tables(predictions),
        calibration_summary=calibration_summary(predictions),
        strategy_selection=selection,
        negative_transfer=negative_transfer_strata(
            tables["contrasts"], **config.get("rescue", {})
        ),
    )
    tables["seed_averaged_effects"] = seed_averaged_effects(
        tables["contrasts"], tables["bootstrap_draws"]
    )
    gate_settings = {
        **contrast_manifest.get("evaluation", {}).get("router_null_equivalence", {}),
        **config.get("router_null_equivalence", {}),
    }
    gate_contrast = gate_settings.get("contrast", "R0")
    gated = tuple(gate_settings.get("gates", ROUTER_GATED_CONTRASTS))
    tables["router_null_equivalence"] = router_null_equivalence(
        tables["effect_summaries"],
        gate_settings.get("tolerance"),
        contrast=gate_contrast,
        gated=gated,
    )
    tables["contrasts"] = annotate_router_gate(
        tables["contrasts"], tables["router_null_equivalence"], gated=gated
    )
    tables["effect_summaries"] = annotate_router_gate(
        tables["effect_summaries"], tables["router_null_equivalence"], gated=gated
    )
    if "n_events_train" in annotated:
        rarity = annotated.loc[
            annotated.metric.isin(["auc", "brier", "log_loss"])
        ].copy()
        rarity["metric"] = rarity.metric.replace(
            {"auc": "auroc", "brier": "brier_score"}
        )
        rarity["difference"] = rarity.gain
        rarity["minority_class"] = rarity[["n_cases", "n_controls"]].min(axis=1)
        rarity["cell_id"] = (
            rarity[["cohort", "outcome", "horizon_days", "contrast"]]
            .astype(str)
            .agg("::".join, axis=1)
        )
        tables["rarity_inputs"] = rarity
    if not tables["negative_transfer"].empty:
        _, tables["negative_transfer_summary"] = effect_summaries(
            tables["negative_transfer"],
            tables["bootstrap_draws"],
            cell_metadata[["cohort", "outcome", "horizon_days"]],
            strata=["stratum"],
        )
    statuses = []
    if metadata is not None:
        tables["router_identity_audit"] = audit_router_identity(metadata)
        needs_review = (
            tables["router_identity_audit"]
            .status.isin(
                [
                    "missing_identity_time_provenance",
                    "missing_identity_evidence",
                    "same_or_postindex_evidence_needs_review",
                ]
            )
            .any()
        )
        statuses.append(
            {
                "analysis": "router_identity_provenance",
                "status": "needs_review"
                if needs_review
                else "evidence_recorded_not_independently_certified",
            }
        )
    if (
        metadata is not None
        and config.get("identity_date_sensitivity", True)
        and (
            "cohort_identity_post_index" in metadata
            or {"index_date", "cohort_identity_evidence_date"} <= set(metadata)
        )
    ):
        if "cohort_identity_post_index" in metadata:
            # Flag from how the subtype was assigned (first-line pipeline pass).
            late = metadata.cohort_identity_post_index.astype(bool)
            retained_ids = set(metadata.loc[~late, "subject_id"])
        else:
            index = pd.to_datetime(metadata.index_date, errors="raise")
            diagnosis = pd.to_datetime(
                metadata.cohort_identity_evidence_date, errors="raise"
            )
            retained_ids = set(
                metadata.loc[
                    diagnosis.notna() & index.notna() & (diagnosis <= index),
                    "subject_id",
                ]
            )
        subset = predictions.loc[predictions.subject_id.isin(retained_ids)]
        before = (
            predictions.drop_duplicates([*CELL, "subject_id"])
            .groupby(CELL)
            .size()
            .rename("n_known_original")
        )
        after = (
            subset.drop_duplicates([*CELL, "subject_id"])
            .groupby(CELL)
            .size()
            .rename("n_known_retained")
        )
        tables["identity_sensitivity_counts"] = (
            before.to_frame().join(after).fillna({"n_known_retained": 0}).reset_index()
        )
        if subset.empty:
            statuses.append(
                {
                    "analysis": "late_diagnosis_exclusion_sensitivity",
                    "status": "no_dated_patients_retained",
                }
            )
        else:
            sensitivity = paired_analysis(
                subset,
                contrasts,
                replicates=replicates,
                draws=draws,
                patient_gain_contrasts=[],
            )
            tables["late_diagnosis_exclusion_sensitivity"] = sensitivity["contrasts"]
            statuses.append(
                {
                    "analysis": "late_diagnosis_exclusion_sensitivity",
                    "status": "computed_same_fits_restricted_evaluation_population",
                }
            )
    if (
        metadata is not None
        and config.get("patient_groups")
        and not tables["patient_gains"].empty
    ):
        tables["patient_subgroups"] = subgroup_patient_gains(
            tables["patient_gains"], metadata, config["patient_groups"], draws=draws
        )
        tables["patient_subgroup_panels"] = panel_subgroup_gains(
            tables["patient_gains"], metadata, config["patient_groups"], draws=draws
        )
        statuses.append({"analysis": "patient_characteristics", "status": "computed"})
    else:
        statuses.append(
            {
                "analysis": "patient_characteristics",
                "status": "missing_metadata_or_group_definitions",
            }
        )
    if config.get("coverage_roster"):
        tables["model_coverage"] = coverage_inventory(
            predictions, read_table(config["coverage_roster"])
        )
        statuses.append({"analysis": "specialist_coverage", "status": "computed"})
    else:
        statuses.append(
            {
                "analysis": "specialist_coverage",
                "status": "missing_planned_coverage_roster",
            }
        )
    if config.get("route_weights"):
        tables["route_weights"] = route_summaries(read_route_artifact(config, metadata))
        statuses.append({"analysis": "route_descriptions", "status": "computed"})
    else:
        statuses.append(
            {"analysis": "route_descriptions", "status": "missing_route_export"}
        )
    if config.get("compute_inventory"):
        tables["compute_inventory"] = read_table(config["compute_inventory"])
        compute = tables["compute_inventory"]
        keys = [key for key in ("model", "seed", "cohort", "outcome") if key in compute]
        if not {"model", "seed"} <= set(keys) or compute.duplicated(keys).any():
            raise ValueError(
                "Compute inventory needs unique model/seed and optional cohort/outcome fit keys"
            )
        tables["cost_performance"] = tables["estimates"].merge(
            compute, on=keys, how="left", validate="many_to_one"
        )
    gate = tables["router_null_equivalence"]
    if not gate.empty:
        statuses.append(
            {
                "analysis": "router_null_equivalence",
                "status": "passed"
                if gate.status.eq("equivalent_within_tolerance").all()
                else "bank_effects_not_interpretable",
                "detail": ",".join(sorted(set(gate.status))),
            }
        )
    available = set(tables["contrasts"].contrast)
    for analysis, required in {
        "pooling_axes_and_scarcity": {"P1", "P2", "P3", "P4"},
        "dapt_discrimination_and_calibration": {"A1", "A2"},
        "dapt_by_router_interaction": {"I2"},
    }.items():
        missing = required - available
        statuses.append(
            {
                "analysis": analysis,
                "status": "computed" if not missing else "missing_contrasts",
                "detail": ",".join(sorted(missing)),
            }
        )
    statuses.append(
        {
            "analysis": "negative_transfer_rescue",
            "status": "computed"
            if not tables["negative_transfer"].empty
            and tables["negative_transfer"].stratum.ne("missing_tuning_evidence").any()
            else "missing_tuning_or_held_out_pairs",
        }
    )
    statuses.append(
        {
            "analysis": "tuning_selected_cell_strategies",
            "status": "computed" if not selected.empty else "missing_strategy_inputs",
        }
    )
    statuses.append(
        {
            "analysis": "horizon_heterogeneity",
            "status": "computed"
            if predictions.horizon_days.nunique() > 1
            else "single_horizon_only",
        }
    )
    if "patient_subgroups" in tables:
        calendar_columns = set(config.get("calendar_covariates", ["index_year"]))
        tables["calendar_heterogeneity"] = tables["patient_subgroups"].loc[
            tables["patient_subgroups"].covariate.isin(calendar_columns)
        ]
        mismatch_columns = set(config.get("domain_mismatch_covariates", []))
        tables["domain_mismatch"] = tables["patient_subgroups"].loc[
            tables["patient_subgroups"].covariate.isin(mismatch_columns)
            & tables["patient_subgroups"].contrast.isin(
                ["A1", "A2", "E1", "E2", "E3", "E4"]
            )
        ]
    for name in ("calendar_heterogeneity", "domain_mismatch"):
        statuses.append(
            {
                "analysis": name,
                "status": "computed"
                if name in tables and not tables[name].empty
                else "missing_covariates_or_contrasts",
            }
        )
    removal_names = set(config.get("source_removal_contrasts", []))
    tables["source_removal_sensitivity"] = tables["contrasts"].loc[
        tables["contrasts"].contrast.isin(removal_names)
    ]
    statuses.append(
        {
            "analysis": "source_removal_sensitivity",
            "status": "computed"
            if not tables["source_removal_sensitivity"].empty
            else "missing_source_removal_inference_exports",
        }
    )
    statuses.append(
        {
            "analysis": "compute_cost",
            "status": "inventory_saved"
            if "compute_inventory" in tables
            else "missing_benchmark_inventory",
        }
    )
    tables["analysis_status"] = pd.DataFrame(statuses)
    tables["input_audit"] = pd.DataFrame(audit)
    for key in (
        "patient_metadata",
        "cell_metadata",
        "coverage_roster",
        "route_weights",
        "compute_inventory",
        "contrast_manifest",
    ):
        if config.get(key):
            hashes[config[key]] = file_hash(config[key])
    # All validation/analysis precedes the first output mutation.
    output_dir.mkdir(parents=True)
    for name, table in tables.items():
        if name == "bootstrap_draws":
            np.savez_compressed(
                output_dir / "paired_bootstrap_draws_INTERNAL.npz",
                gains=np.stack(table.gain_draws),
            )
            table.drop(columns="gain_draws").to_csv(
                output_dir / "paired_bootstrap_index.csv", index=False
            )
            continue
        suffix = "_INTERNAL" if name in {"patient_gains", "bootstrap_draws"} else ""
        table.to_csv(output_dir / f"{name}{suffix}.csv", index=False)
    summary = {
        "estimand": config["estimand"],
        "bootstrap_replicates": replicates,
        "contrast_expressions": contrasts,
        "shared_bootstrap_axes": ["models", "outcomes", "horizons", "training_seeds"],
        "bootstrap_unit": ["split", "cohort", "subject_id"],
        "input_sha256": hashes,
        "config": json_safe(config),
        "combined_input_audit": combined_audit,
        "intervals": "patient_uncertainty_conditional_on_selected_fits_per_seed",
        "aggregation": "equal_cell_descriptive; endpoint/disease weighting must be locked separately",
        "tables": {name: len(table) for name, table in tables.items()},
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    summary = run(config, args.output_dir)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "tables": summary["tables"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
