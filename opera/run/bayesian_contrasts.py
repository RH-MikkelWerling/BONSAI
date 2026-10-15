"""Fit question-aligned Bayesian contrasts from locked prediction-analysis outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.analysis.bayesian_contrasts import (
    FAMILIES,
    fit_contrast_model,
    prepare_contrast_data,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config):
    root = Path(config["analysis_dir"])
    files = [
        root / name
        for name in (
            "contrasts.csv",
            "paired_bootstrap_index.csv",
            "paired_bootstrap_draws_INTERNAL.npz",
            "analysis_manifest.json",
        )
    ]
    manifest = json.loads(files[-1].read_text())
    if "training_seeds" not in manifest.get("shared_bootstrap_axes", []):
        raise ValueError(
            "Regenerate prediction analysis with shared patient draws across training seeds"
        )
    # Prevent accidentally presenting binary known-status effects as IPCW effects.
    if manifest.get("estimand") != config["estimand"]:
        raise ValueError("Requested estimand differs from the locked analysis manifest")
    if any(
        config.get(k) is None
        for k in (
            "practical_tolerance",
            "prior_scale",
            "random_effect_scale",
            "slope_scale",
        )
    ):
        raise ValueError(
            "Lock metric-scale priors and practical_tolerance before fitting"
        )
    effects, index = pd.read_csv(files[0]), pd.read_csv(files[1])
    with np.load(files[2], allow_pickle=False) as archive:
        gains = archive["gains"]
    from opera.evaluation.secondary_analysis import contrast_coefficients

    canonical_path = (
        Path(__file__).resolve().parents[1]
        / "configs/manifests/final_contrasts_20261014.yaml"
    )
    canonical = yaml.safe_load(canonical_path.read_text())
    expressions = {
        name: expr
        for family in canonical["essential_contrasts"].values()
        for name, expr in family.items()
    }
    actual = manifest.get("contrast_expressions", {})
    definitions = FAMILIES[config["family"]]
    for name, coefficients in definitions.items():
        if name not in actual or contrast_coefficients(
            actual[name]
        ) != contrast_coefficients(expressions[name]):
            raise ValueError(
                f"Regenerate/verify canonical contrast definitions: {name}"
            )
        derived = {}
        for primitive, multiplier in coefficients.items():
            for model, coefficient in contrast_coefficients(
                expressions[primitive]
            ).items():
                derived[model] = derived.get(model, 0) + multiplier * coefficient
        derived = {key: value for key, value in derived.items() if value}
        if derived != contrast_coefficients(expressions[name]):
            raise ValueError(
                f"Canonical manifest does not match the model's algebra: {name}"
            )
    files.append(canonical_path)
    data = prepare_contrast_data(
        effects,
        index,
        gains,
        family=config["family"],
        metric=config["metric"],
        horizon_days=config.get("horizon_days", 365),
        split=config.get("split", "held_out"),
        seeds=tuple(config["seeds"]),
        trend=config.get("trend", "none"),
        support_bins=config.get("support_bins"),
        weighting=config.get("weighting", "equal_disease_loss_endpoint"),
        rarity_column=config.get("rarity_column", "n_events_train"),
        covariance_shrinkage=config.get("covariance_shrinkage", 0.02),
        se_floor=config.get("se_floor", 1e-6),
        spline_knots=config.get("spline_knots", 4),
        cell_roster=pd.read_csv(config["cell_roster"])
        if config.get("cell_roster")
        else None,
    )
    if config.get("cell_roster"):
        files.append(Path(config["cell_roster"]))
    data.specification.update(
        estimand=manifest["estimand"],
        input_sha256={str(p): file_hash(p) for p in files},
        analysis_input_sha256=manifest.get("input_sha256"),
        covariate_interpretation="association_not_causal_scarcity",
    )
    output = Path(config["output_dir"])
    if config.get("validate_only", False):
        print(
            json.dumps(
                {
                    "status": "validated_no_sampling",
                    "family": config["family"],
                    "cells": len(data.cells),
                    "basis": data.basis,
                    "covariance_blocks": len(data.blocks),
                },
                indent=2,
            )
        )
        return data
    fit_contrast_model(
        data,
        output_dir=output,
        prior_scale=config["prior_scale"],
        random_effect_scale=config["random_effect_scale"],
        slope_scale=config["slope_scale"],
        practical_tolerance=config["practical_tolerance"],
        prior_only=config.get("prior_only", False),
        nest_diseases=config.get("nest_diseases", False),
        nest_outcomes=config.get("nest_outcomes", False),
        **config.get("sampling", {}),
    )
    (output / "recipe.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n"
    )
    if config.get("plot", True) and not config.get("prior_only", False):
        from opera.visualization.bayesian_contrast_plots import plot_contrast_results

        plot_contrast_results(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family", choices=list(FAMILIES))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prior-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.family:
        config["family"] = args.family
    if args.validate_only:
        config["validate_only"] = True
    if args.prior_only:
        config["prior_only"] = True
        target = Path(config["output_dir"])
        config["output_dir"] = str(target.with_name(target.name + "__prior_check"))
    run(config)


if __name__ == "__main__":
    main()
