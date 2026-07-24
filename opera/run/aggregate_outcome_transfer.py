"""Aggregate frozen-probe OPERA transfer predictions and render the figure.

This command is intentionally specific to ``outcome_transfer``.  It does not
enter the general sweep or rarity aggregation paths.  It performs patient-level
paired bootstrap comparisons only after verifying every compared held-out
denominator is identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from opera.evaluation.outcome_transfer_aggregation import (
    aggregate_transfer_predictions,
    summarize_grouped_cohort_deltas,
    summarize_severity_deltas,
    write_transfer_aggregation_outputs,
)
from opera.functional.outcome_transfer import (
    DEFAULT_MANIFEST,
    resolve_transfer_manifest,
)
from opera.visualization.outcome_transfer import write_outcome_transfer_figure


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Required transfer output does not exist: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate paired frozen-probe OPERA outcome-transfer results."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--base-config", default=None)
    parser.add_argument(
        "--predictions",
        required=True,
        help="transfer_predictions.parquet produced by outcome_transfer_evaluate.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="transfer_results.csv from the evaluator; defaults next to --predictions.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--allow-small-bootstrap",
        action="store_true",
        help="Allow fewer than 2,000 replicates for synthetic tests only; production inference must use >=2,000.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write CSV aggregation outputs without rendering outcome_transfer.png/.pdf.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least one.")
    if args.n_bootstrap < 2000 and not args.allow_small_bootstrap:
        raise ValueError(
            "Production outcome-transfer inference requires --n-bootstrap >= 2000. "
            "Use --allow-small-bootstrap only for synthetic tests."
        )
    plan = resolve_transfer_manifest(
        args.manifest,
        registry_path=args.registry,
        base_config_path=args.base_config,
    )
    prediction_path = Path(args.predictions)
    predictions = _read_table(prediction_path)
    deltas, cohort_results, family_summary, aggregation_failures = (
        aggregate_transfer_predictions(
            plan,
            predictions,
            n_bootstrap=args.n_bootstrap,
        )
    )
    cohort_summary = summarize_grouped_cohort_deltas(cohort_results)
    severity_primary_summary = summarize_severity_deltas(plan, deltas, scope="primary")
    severity_secondary_summary = summarize_severity_deltas(
        plan, deltas, scope="secondary"
    )
    existing_failure_path = prediction_path.parent / "transfer_failures.csv"
    existing_failures = (
        _read_table(existing_failure_path)
        if existing_failure_path.exists()
        else pd.DataFrame()
    )
    failures = (
        pd.concat(
            [
                frame
                for frame in (existing_failures, aggregation_failures)
                if not frame.empty
            ],
            ignore_index=True,
            sort=False,
        )
        if (not existing_failures.empty or not aggregation_failures.empty)
        else pd.DataFrame()
    )
    written = write_transfer_aggregation_outputs(
        args.output_dir,
        deltas=deltas,
        cohort_results=cohort_results,
        family_summary=family_summary,
        failures=failures,
        cohort_summary=cohort_summary,
        severity_primary_summary=severity_primary_summary,
        severity_secondary_summary=severity_secondary_summary,
    )
    figure_paths: dict[str, Path] = {}
    if not args.skip_figure:
        result_path = (
            Path(args.results)
            if args.results
            else prediction_path.parent / "transfer_results.csv"
        )
        figure_paths = write_outcome_transfer_figure(
            args.output_dir,
            results=_read_table(result_path),
            deltas=deltas,
            plan=plan,
        )
    metadata_path = Path(args.output_dir) / "transfer_aggregation_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "name": "outcome_transfer_paired_aggregation",
                "manifest": plan["manifest"],
                "manifest_hash": plan["manifest_hash"],
                "registry": plan["registry"],
                "registry_hash": plan["registry_hash"],
                "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
                "split_contract": plan["split_contract"],
                "split_contract_hash": plan["split_contract_hash"],
                "n_bootstrap": args.n_bootstrap,
                "bootstrap_unit": "patient_within_fixed_target_seed_denominator",
                "grouped_cohort_method": "re-stratification_of_pan_hematology_held_out_predictions",
                "family_method": "macro_outcome_after_seed_mean",
                "severity_primary_method": (
                    "paired_patient_delta_macro_over_programmatically_resolved_"
                    "matched_g2_g3_targets_after_seed_mean"
                ),
                "severity_secondary_method": (
                    "paired_patient_delta_macro_over_explicit_unmatched_g3_targets_"
                    "after_seed_mean"
                ),
                "output_files": {
                    **{name: str(path) for name, path in written.items()},
                    **{
                        f"figure_{name}": str(path)
                        for name, path in figure_paths.items()
                    },
                },
                "n_pan_hematology_delta_rows": int(len(deltas)),
                "n_grouped_delta_rows": int(len(cohort_results)),
                "n_grouped_macro_summary_rows": int(len(cohort_summary)),
                "n_severity_primary_summary_rows": int(len(severity_primary_summary)),
                "n_severity_secondary_summary_rows": int(
                    len(severity_secondary_summary)
                ),
                "n_failures": int(len(failures)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(deltas):,} pan-hematology paired-delta rows and "
        f"{len(cohort_results):,} grouped re-stratified rows to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
