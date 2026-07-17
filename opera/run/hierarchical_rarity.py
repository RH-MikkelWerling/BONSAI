"""Run the publication-grade hierarchical natural-rarity analysis.

Examples
--------
Assemble and validate prediction cells without PyMC::

    python -m opera.run.hierarchical_rarity --mode assemble

Fit and draw the final figure after installing ``.[bayesian]``::

    python -m opera.run.hierarchical_rarity --mode all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from opera.analysis.bayesian_rarity import (
    fit_hierarchical_rarity_model,
    prepare_rarity_data,
    write_model_input_artifacts,
)
from opera.evaluation.hierarchical_rarity import (
    apply_outcome_families,
    attach_task_metadata,
    build_paired_delta_tables,
    build_task_size_metadata,
    discover_prediction_artifacts,
    summarize_patient_overlap,
)
from opera.visualization.hierarchical_rarity_plots import (
    aggregate_scatter_cells,
    plot_hierarchical_rarity_curve,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Hierarchical rarity config must be a mapping.")

    def expand(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    config = expand(config)
    family_file = config.pop("outcome_families_file", None)
    if family_file:
        source = Path(str(family_file))
        if not source.is_absolute():
            source = config_path.parent / source
        payload = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("outcome_families"), dict
        ):
            raise ValueError(
                f"Outcome-family file {source} must contain an outcome_families mapping."
            )
        if config.get("outcome_families"):
            raise ValueError(
                "Specify outcome_families_file or outcome_families, not both."
            )
        inverse: dict[str, str] = {}
        for family, outcomes in payload["outcome_families"].items():
            if not isinstance(outcomes, list):
                raise ValueError(
                    f"Outcome family {family!r} in {source} must list outcomes."
                )
            for outcome in outcomes:
                if outcome in inverse:
                    raise ValueError(
                        f"Outcome {outcome!r} appears in multiple families in {source}."
                    )
                inverse[str(outcome)] = str(family)
        config["outcome_families"] = inverse
    return config


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def _read_frame(path: Path) -> pd.DataFrame:
    parquet = path.with_suffix(".parquet")
    csv = path.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {parquet} nor {csv} exists.")


def assemble(config: dict[str, Any]) -> dict[str, Path]:
    """Discover, validate, pair, bootstrap, and audit all prediction cells."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    analysis = config["analysis"]
    artifacts = discover_prediction_artifacts(
        config["results_root"],
        evaluation_subset=analysis.get("evaluation_subset", "full"),
    )
    analysis_level = analysis.get("analysis_level")
    if analysis_level is not None:
        if analysis_level != "fine":
            raise ValueError(
                "The primary hierarchical rarity workflow only accepts "
                "analysis_level='fine'."
            )
        if "analysis_level" not in artifacts.columns:
            raise ValueError(
                "Discovered result artifacts lack analysis_level metadata; "
                "cannot prove that grouped/global cells are excluded."
            )
        observed_levels = artifacts["analysis_level"].fillna("").astype(str)
        wrong_levels = sorted(set(observed_levels) - {"fine"})
        if wrong_levels:
            raise ValueError(
                "Primary rarity analysis refuses non-fine artifacts: "
                f"{wrong_levels}."
            )

    task_metadata_path = config.get("task_metadata")
    if task_metadata_path:
        task_metadata = _read_frame(Path(task_metadata_path).with_suffix(""))
    else:
        task_metadata = build_task_size_metadata(config["sweep_config"])
    task_metadata = apply_outcome_families(
        task_metadata,
        config.get("outcome_families"),
        require_complete=bool(analysis.get("require_mapped_outcomes", True)),
    )
    artifacts = attach_task_metadata(artifacts, task_metadata)
    _write_frame(task_metadata, output / "task_metadata")
    _write_frame(artifacts, output / "artifact_inventory")

    deltas, bootstrap_draws, memberships = build_paired_delta_tables(
        artifacts,
        model_family=analysis["model_family"],
        comparator_family=analysis["comparator_family"],
        n_bootstrap=int(analysis.get("bootstrap_replicates", 1000)),
        seed=int(analysis.get("bootstrap_seed", 2026)),
        metrics=tuple(
            analysis.get("metrics", [analysis.get("primary_metric", "auroc")])
        ),
        min_test_positive=int(analysis.get("minimum_test_positive", 1)),
        min_test_negative=int(analysis.get("minimum_test_negative", 1)),
        primary_test_positive=int(analysis.get("primary_test_positive", 25)),
        primary_test_negative=int(analysis.get("primary_test_negative", 25)),
        small_sample_minority_threshold=int(
            analysis.get("small_sample_minority_threshold", 10)
        ),
    )
    _write_frame(deltas, output / "paired_deltas")
    _write_frame(bootstrap_draws, output / "paired_bootstrap_draws")
    overlap_summary, overlap_pairs = summarize_patient_overlap(memberships)
    overlap_summary_path = output / "patient_overlap_summary.json"
    overlap_summary_path.write_text(json.dumps(overlap_summary, indent=2))
    _write_frame(overlap_pairs, output / "patient_overlap_pairs")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "results_root": str(Path(config["results_root"]).resolve()),
        "sweep_config": str(Path(config["sweep_config"]).resolve()),
        "model_family": analysis["model_family"],
        "comparator_family": analysis["comparator_family"],
        "metrics": analysis.get("metrics"),
        "n_artifacts": int(len(artifacts)),
        "n_paired_rows": int(len(deltas)),
        "n_cells": int(deltas["cell_id"].nunique()),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16],
    }
    manifest_path = output / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "deltas": output / "paired_deltas.parquet",
        "bootstrap_draws": output / "paired_bootstrap_draws.parquet",
        "overlap": overlap_summary_path,
        "manifest": manifest_path,
    }


def fit(config: dict[str, Any]) -> dict[str, Path]:
    """Fit the PyMC hierarchy using already assembled paired deltas."""
    output = Path(config["output_dir"])
    deltas = _read_frame(output / "paired_deltas")
    analysis = config["analysis"]
    model_cfg = config.get("model", {})
    prepared = prepare_rarity_data(
        deltas,
        metric=analysis.get("primary_metric", "auroc"),
        rarity_column=analysis.get("rarity_column", "n_events_train"),
        spline_knots=int(model_cfg.get("spline_knots", 6)),
        standard_error_floor=float(model_cfg.get("standard_error_floor", 1e-3)),
        fit_tiers=tuple(model_cfg.get("fit_tiers", ["primary", "partial_pool_only"])),
        grid_size=int(model_cfg.get("grid_size", 200)),
    )
    write_model_input_artifacts(prepared, output / "model")
    sampling = config.get("sampling", {})
    return fit_hierarchical_rarity_model(
        prepared,
        output_dir=output / "model",
        draws=int(sampling.get("draws", 2000)),
        tune=int(sampling.get("tune", 2000)),
        chains=int(sampling.get("chains", 4)),
        cores=sampling.get("cores"),
        target_accept=float(sampling.get("target_accept", 0.95)),
        random_seed=int(sampling.get("seed", 2026)),
        prior_scale=float(model_cfg.get("prior_scale", 0.10)),
        random_effect_scale=float(model_cfg.get("random_effect_scale", 0.05)),
        require_convergence=bool(model_cfg.get("require_convergence", True)),
        nest_outcomes=bool(model_cfg.get("nest_outcomes", True)),
        min_report_minority=int(model_cfg.get("min_report_minority", 25)),
    )


def plot(config: dict[str, Any]) -> Path:
    """Regenerate the primary figure entirely from cached analysis artifacts."""
    output = Path(config["output_dir"])
    deltas = _read_frame(output / "paired_deltas")
    curve = pd.read_csv(output / "model" / "posterior_curve.csv")
    analysis = config["analysis"]
    figure = config.get("figure", {})
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = (
        figure_dir
        / f"hierarchical_rarity_{analysis.get('primary_metric', 'auroc')}.png"
    )
    scatter = aggregate_scatter_cells(
        deltas,
        metric=analysis.get("primary_metric", "auroc"),
        rarity_column=analysis.get("rarity_column", "n_events_train"),
    )
    _write_frame(scatter, figure_dir / "hierarchical_rarity_scatter_data")
    plot_hierarchical_rarity_curve(
        deltas,
        curve,
        metric=analysis.get("primary_metric", "auroc"),
        rarity_column=analysis.get("rarity_column", "n_events_train"),
        model_label=figure.get("model_label", "OPERA"),
        comparator_label=figure.get("comparator_label", "XGBoost"),
        title=figure.get("title", "Model benefit across natural task information"),
        show_predictive_interval=bool(figure.get("show_predictive_interval", True)),
        max_labels=int(figure.get("max_labels", 7)),
        cohort_course_groups=figure.get("cohort_course_groups"),
        cohort_group_labels=figure.get("cohort_group_labels"),
        outcome_family_order=figure.get("outcome_family_order"),
        highlight_cells=figure.get("highlight_cells"),
        highlight_outcomes=figure.get("highlight_outcomes"),
        save_path=str(figure_path),
    )
    return figure_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical Bayesian natural-rarity analysis"
    )
    parser.add_argument(
        "--config",
        default="opera/configs/hierarchical_rarity.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("assemble", "fit", "plot", "all"),
        default="all",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.mode in {"assemble", "all"}:
        assemble(config)
    if args.mode in {"fit", "all"}:
        fit(config)
    if args.mode in {"plot", "all"}:
        path = plot(config)
        print(f"Wrote hierarchical rarity figure: {path}")


if __name__ == "__main__":
    main()
