"""Prepare primary and bounded sensitivity recipes without fitting any model.

Recipes are derived from the locked prediction analysis. For every question
family and metric, a prespecified estimability rule writes an explicit cell
roster and a reasoned exclusion table (single-class evaluation, missing model
arms, degenerate bootstrap draws). Each recipe's data preparation is validated
without sampling, so families whose model arms were not run (e.g. external
pretraining) or sensitivities a sparse panel cannot support are reported as
not estimable instead of failing at fit time.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.analysis.bayesian_contrasts import (
    FAMILIES,
    estimable_cell_roster,
    prepare_contrast_data,
)

METRICS = ("auc", "brier", "log_loss")
# Deliberately small. "exchangeable" answers: how large is the typical effect,
# and how much does it vary between diseases and outcomes? "support_spline"
# answers: does the effect change with the target's training data, in any
# smooth shape (few knots, shrinkage prior: flat unless the data say otherwise)?
# "support_bins" is a transparent piecewise-constant check of that curve.
# Linear, clinical nesting and covariance variants remain available manually.
VARIANTS = (
    ("exchangeable", {"trend": "none"}),
    ("support_spline", {"trend": "spline"}),
    ("support_bins", {"trend": "bins"}),
)
PRIMARY_VARIANT = "support_spline"
NESTING_COLUMNS = {"nest_diseases": "cohort_group", "nest_outcomes": "outcome_family"}
METRIC_LOCKED_KEYS = (
    "prior_scale",
    "random_effect_scale",
    "slope_scale",
    "practical_tolerance",
)


def load_analysis(analysis_dir):
    root = Path(analysis_dir)
    effects = pd.read_csv(root / "contrasts.csv")
    index = pd.read_csv(root / "paired_bootstrap_index.csv")
    with np.load(root / "paired_bootstrap_draws_INTERNAL.npz", allow_pickle=False) as archive:
        gains = archive["gains"]
    return effects, index, gains


def recipe_config(template, metric, *, family, analysis_dir, seeds, output_dir):
    """One metric's recipe; metric-scale priors come from ``metric_settings``."""
    config = copy.deepcopy(template)
    settings = config.pop("metric_settings", None) or {}
    config.update({key: None for key in METRIC_LOCKED_KEYS if key not in config})
    if metric != template.get("metric"):
        # Flat priors/tolerances were written for the template's metric scale.
        config.update({key: None for key in METRIC_LOCKED_KEYS})
    config.update(
        {k: v for k, v in (settings.get(metric) or {}).items() if v is not None}
    )
    config.update(
        analysis_dir=str(Path(analysis_dir).resolve()),
        family=family,
        metric=metric,
        seeds=list(seeds),
        trend="none",
        nest_diseases=False,
        nest_outcomes=False,
        output_dir=str(Path(output_dir).resolve()),
    )
    return config


def prepare(
    template,
    output,
    analysis_dir,
    results_root,
    seeds,
    *,
    metrics=METRICS,
    families=None,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh Bayesian recipe directory")
    effects, index, gains = load_analysis(analysis_dir)
    manifest_path = Path(analysis_dir) / "analysis_manifest.json"
    if manifest_path.is_file():
        # Recipes must target the estimand the analysis actually computed.
        template = {**template, "estimand": json.loads(manifest_path.read_text())["estimand"]}
    rule = template.get("estimability", {}) or {}
    horizon = template.get("horizon_days", 365)
    split = template.get("split", "held_out")
    rarity = template.get("rarity_column", "n_events_train")
    recipes, primary, status = [], [], []
    rosters = {}
    for family in families or FAMILIES:
        for metric in metrics:
            roster, exclusions = estimable_cell_roster(
                effects,
                index,
                gains,
                family=family,
                metric=metric,
                horizon_days=horizon,
                split=split,
                seeds=seeds,
                rarity_column=rarity,
                min_cases=int(rule.get("min_cases", 5)),
                min_controls=int(rule.get("min_controls", 5)),
                minimum_finite_fraction=float(rule.get("minimum_finite_fraction", 0.8)),
            )
            rosters[(family, metric)] = (roster, exclusions)
            for name, overrides in VARIANTS:
                config = recipe_config(
                    template,
                    metric,
                    family=family,
                    analysis_dir=analysis_dir,
                    seeds=seeds,
                    output_dir=Path(results_root) / family / metric / name,
                )
                config.update(overrides)
                config["cell_roster"] = str(
                    (output / "rosters" / f"{family}_{metric}.csv").resolve()
                )
                record = {
                    "family": family,
                    "metric": metric,
                    "variant": name,
                    "n_cells": len(roster),
                    "n_excluded": len(exclusions),
                }
                missing_nesting = [
                    column
                    for flag, column in NESTING_COLUMNS.items()
                    if config.get(flag) and column not in effects
                ]
                try:
                    if missing_nesting:
                        raise ValueError(
                            "nesting metadata absent: " + ",".join(missing_nesting)
                        )
                    if roster.empty:
                        raise ValueError("no estimable cells under the roster rule")
                    data = prepare_contrast_data(
                        effects,
                        index,
                        gains,
                        family=family,
                        metric=metric,
                        horizon_days=horizon,
                        split=split,
                        seeds=tuple(seeds),
                        trend=config["trend"],
                        support_bins=config.get("support_bins"),
                        weighting=config.get(
                            "weighting", "equal_disease_endpoint_group"
                        ),
                        rarity_column=rarity,
                        covariance_shrinkage=config.get("covariance_shrinkage", 0.02),
                        se_floor=config.get("se_floor", 1e-6),
                        spline_knots=config.get("spline_knots", 4),
                        cell_roster=roster,
                    )
                except ValueError as error:
                    record.update(status="not_estimable", reason=str(error))
                else:
                    record.update(status="validated_not_fitted", n_cells=len(data.cells))
                    path = output / f"{family}_{metric}_{name}.yaml"
                    recipes.append((path, config))
                    if name == PRIMARY_VARIANT:
                        primary.append(str(path.resolve()))
                status.append(record)
    output.mkdir(parents=True)
    (output / "rosters").mkdir()
    for (family, metric), (roster, exclusions) in rosters.items():
        roster.to_csv(output / "rosters" / f"{family}_{metric}.csv", index=False)
        exclusions.to_csv(
            output / "rosters" / f"{family}_{metric}_exclusions.csv", index=False
        )
    for path, config in recipes:
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    pd.DataFrame(status).to_csv(output / "recipe_status.csv", index=False)
    (output / "recipe_manifest.json").write_text(
        json.dumps(
            {
                "primary_recipes": primary,
                "n_recipes": len(recipes),
                "not_estimable": [
                    row for row in status if row["status"] == "not_estimable"
                ],
                "estimability_rule": {
                    "min_cases": int(rule.get("min_cases", 5)),
                    "min_controls": int(rule.get("min_controls", 5)),
                    "minimum_finite_fraction": float(
                        rule.get("minimum_finite_fraction", 0.8)
                    ),
                    "depends_on_effect_size": False,
                },
                "status": "prepared_not_fitted_priors_and_tolerances_require_lock",
            },
            indent=2,
        )
        + "\n"
    )
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--metrics", default=",".join(METRICS))
    args = parser.parse_args()
    primary = prepare(
        yaml.safe_load(args.template.read_text()),
        args.output_dir,
        args.analysis_dir,
        args.results_root,
        tuple(int(x) for x in args.seeds.split(",")),
        metrics=tuple(m for m in args.metrics.split(",") if m),
    )
    print(
        json.dumps(
            {"primary_recipes": primary, "status": "prepared_no_sampling"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
