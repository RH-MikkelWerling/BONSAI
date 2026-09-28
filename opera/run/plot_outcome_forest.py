"""Render publication forest plots from exported result summaries.

The command is intended to run locally after copying compact CSV/Parquet
artifacts from the server. It accepts either canonical long estimate/contrast
tables or the paired comparison table produced by
``opera.run.compare_direct_cr_tabular``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

from opera.visualization.outcome_forest import (
    apply_outcome_family_map,
    normalize_contrast_table,
    normalize_estimate_table,
    paginate_outcomes,
    paired_comparison_to_forest_tables,
    plot_outcome_performance_forest,
)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "all")).strip("_")
    return text.lower() or "all"


def _family_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(payload, dict):
            raise ValueError("Family YAML must contain a mapping.")
        cross_outcome = payload.get("cross_outcome", payload)
        families = cross_outcome.get("outcome_families", cross_outcome)
        if not isinstance(families, dict):
            raise ValueError("Family YAML has no outcome_families mapping.")
        return {
            str(outcome): str(family)
            for family, outcomes in families.items()
            for outcome in outcomes
        }
    frame = _read_table(path)
    required = {"outcome", "outcome_family"}
    if not required.issubset(frame):
        raise ValueError("Family table requires outcome and outcome_family columns.")
    return (
        frame[["outcome", "outcome_family"]]
        .drop_duplicates("outcome")
        .set_index("outcome")["outcome_family"]
        .astype(str)
        .to_dict()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--estimates", type=Path)
    source.add_argument("--paired-comparison", type=Path)
    parser.add_argument("--contrasts", type=Path, default=None)
    parser.add_argument("--family-map", type=Path, default=None)
    parser.add_argument("--metric", default="cif_auc")
    parser.add_argument("--metric-label", default="Cumulative/dynamic AUC")
    parser.add_argument("--interval-label", default="95% interval")
    parser.add_argument("--comparator", default="opera")
    parser.add_argument("--baseline", default="tabular_ehr")
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--cohort", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--contrast", action="append", default=[])
    parser.add_argument("--family-order", default=None)
    parser.add_argument("--max-outcomes-per-page", type=int, default=18)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="outcome_forest")
    parser.add_argument("--title", default=None)
    return parser


def _filter_slice(
    frame: pd.DataFrame,
    *,
    horizon: float | None,
    cohort: str | None,
) -> pd.DataFrame:
    result = frame.copy()
    if horizon is not None and "horizon_days" in result:
        result = result[
            pd.to_numeric(result["horizon_days"], errors="coerce").eq(float(horizon))
        ]
    if cohort is not None and "cohort" in result:
        result = result[result["cohort"].astype(str) == str(cohort)]
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.max_outcomes_per_page < 1:
        raise ValueError("--max-outcomes-per-page must be positive.")
    if args.paired_comparison:
        estimates, contrasts = paired_comparison_to_forest_tables(
            _read_table(args.paired_comparison),
            metric=args.metric,
            comparator=args.comparator,
            baseline=args.baseline,
        )
    else:
        estimates = normalize_estimate_table(
            _read_table(args.estimates), metric=args.metric
        )
        contrasts = (
            pd.DataFrame()
            if args.contrasts is None
            else normalize_contrast_table(
                _read_table(args.contrasts), metric=args.metric
            )
        )
    family_map = _family_map(args.family_map)
    if family_map:
        estimates = apply_outcome_family_map(estimates, family_map)
        if not contrasts.empty:
            contrasts = apply_outcome_family_map(contrasts, family_map)
    if args.model:
        estimates = estimates[estimates["model"].isin(args.model)].copy()
    if args.contrast and not contrasts.empty:
        contrasts = contrasts[contrasts["contrast"].isin(args.contrast)].copy()
    if estimates.empty:
        raise ValueError("No absolute estimates remain after filtering.")

    available_horizons = sorted(estimates["horizon_days"].dropna().unique())
    if args.horizon is None and len(available_horizons) > 1:
        raise ValueError(
            "Input contains multiple horizons; select one with --horizon."
        )
    horizon = args.horizon or (
        None if not available_horizons else float(available_horizons[0])
    )
    available_cohorts = estimates["cohort"].dropna().astype(str).unique().tolist()
    cohorts = args.cohort or (available_cohorts if available_cohorts else [None])
    family_order = (
        None
        if args.family_order is None
        else [value.strip() for value in args.family_order.split(",") if value.strip()]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotted_rows = []
    output_files = []
    for cohort in cohorts:
        cohort_estimates = _filter_slice(estimates, horizon=horizon, cohort=cohort)
        cohort_contrasts = _filter_slice(contrasts, horizon=horizon, cohort=cohort)
        if cohort_estimates.empty:
            continue
        pages = paginate_outcomes(
            cohort_estimates,
            max_outcomes_per_page=args.max_outcomes_per_page,
            family_order=family_order,
        )
        for page_index, outcomes in enumerate(pages, start=1):
            page_estimates = cohort_estimates[
                cohort_estimates["outcome"].isin(outcomes)
            ].copy()
            page_contrasts = cohort_contrasts[
                cohort_contrasts["outcome"].isin(outcomes)
            ].copy()
            suffix = f"{_slug(cohort)}"
            if horizon is not None:
                suffix += f"_{int(round(horizon))}d"
            if len(pages) > 1:
                suffix += f"_p{page_index:02d}"
            output = args.output_dir / f"{args.output_prefix}_{suffix}.png"
            fig = plot_outcome_performance_forest(
                page_estimates,
                page_contrasts,
                metric_label=args.metric_label,
                interval_label=args.interval_label,
                model_order=args.model or None,
                contrast_order=args.contrast or None,
                family_order=family_order,
                higher_is_better=not args.lower_is_better,
                title=args.title,
                save_path=output,
            )
            plt.close(fig)
            output_files.extend([str(output), str(output.with_suffix(".pdf"))])
            plotted_rows.extend(
                page_estimates.assign(record_type="absolute", page=page_index).to_dict(
                    "records"
                )
            )
            plotted_rows.extend(
                page_contrasts.assign(record_type="contrast", page=page_index).to_dict(
                    "records"
                )
            )
    if not output_files:
        raise ValueError("No cohort/horizon slice produced a forest plot.")
    pd.DataFrame(plotted_rows).to_csv(
        args.output_dir / "forest_plot_data.csv", index=False
    )
    (args.output_dir / "forest_plot_manifest.json").write_text(
        json.dumps(
            {
                "metric": args.metric,
                "horizon_days": horizon,
                "cohorts": cohorts,
                "comparator": args.comparator,
                "baseline": args.baseline,
                "interval_label": args.interval_label,
                "files": output_files,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
