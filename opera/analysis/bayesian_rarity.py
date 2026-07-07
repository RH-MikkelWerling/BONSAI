"""Bayesian hierarchical spline model for natural task rarity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer


@dataclass
class PreparedRarityData:
    """Numerical design and audited tables used by the PyMC model."""

    observations: pd.DataFrame
    cells: pd.DataFrame
    spline_observations: np.ndarray
    spline_cells: np.ndarray
    spline_grid: np.ndarray
    grid_event_counts: np.ndarray
    cohort_index: np.ndarray
    outcome_index: np.ndarray
    family_index: np.ndarray
    cell_index: np.ndarray
    cohorts: list[str]
    outcomes: list[str]
    families: list[str]
    basis_center: np.ndarray
    basis_scale: np.ndarray


def prepare_rarity_data(
    deltas: pd.DataFrame,
    *,
    metric: str = "auroc",
    rarity_column: str = "n_events_train",
    spline_knots: int = 6,
    standard_error_floor: float = 1e-3,
    fit_tiers: tuple[str, ...] = ("primary", "partial_pool_only"),
    grid_size: int = 200,
) -> PreparedRarityData:
    """Validate and transform paired delta rows for hierarchical fitting."""
    required = {
        "cell_id",
        "cohort",
        "outcome",
        "metric",
        "difference",
        "difference_se",
        rarity_column,
    }
    missing = required.difference(deltas.columns)
    if missing:
        raise ValueError(f"Paired delta table is missing columns: {sorted(missing)}")
    observations = deltas[deltas["metric"] == metric].copy()
    if "analysis_tier" in observations:
        observations = observations[observations["analysis_tier"].isin(fit_tiers)]
    for column in ("difference", "difference_se", rarity_column):
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
    observations = observations.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["difference", "difference_se", rarity_column]
    )
    observations = observations[observations[rarity_column] > 0].copy()
    if observations.empty:
        raise ValueError("No finite positive-rarity rows are available for modelling.")
    if observations["cell_id"].nunique() < 8:
        raise ValueError("At least eight cohort-outcome cells are required.")
    observations["difference_se"] = observations["difference_se"].clip(
        lower=float(standard_error_floor)
    )
    observations["outcome_family"] = observations.get(
        "outcome_family", pd.Series("Other", index=observations.index)
    ).fillna("Other")
    observations["cohort_group"] = observations.get(
        "cohort_group", observations["cohort"]
    ).fillna(observations["cohort"])
    observations["log2_rarity"] = np.log2(observations[rarity_column].astype(float))

    cell_columns = [
        "cell_id",
        "cohort",
        "cohort_group",
        "outcome",
        "outcome_family",
        rarity_column,
        "log2_rarity",
    ]
    cells = observations[cell_columns].drop_duplicates("cell_id").copy()
    consistency = observations.groupby("cell_id")[rarity_column].nunique()
    if (consistency > 1).any():
        raise ValueError("Rarity counts changed across seeds within a task cell.")
    cells = cells.sort_values("cell_id").reset_index(drop=True)
    cell_lookup = {name: index for index, name in enumerate(cells["cell_id"])}
    observations["cell_index"] = observations["cell_id"].map(cell_lookup).astype(int)

    cohorts = sorted(cells["cohort"].astype(str).unique())
    outcomes = sorted(cells["outcome"].astype(str).unique())
    families = sorted(cells["outcome_family"].astype(str).unique())
    cohort_lookup = {value: index for index, value in enumerate(cohorts)}
    outcome_lookup = {value: index for index, value in enumerate(outcomes)}
    family_lookup = {value: index for index, value in enumerate(families)}

    x_cells = cells[["log2_rarity"]].to_numpy(dtype=float)
    transformer = SplineTransformer(
        n_knots=max(4, int(spline_knots)),
        degree=3,
        knots="quantile",
        include_bias=False,
        extrapolation="constant",
    )
    basis_cells_raw = transformer.fit_transform(x_cells)
    basis_center = basis_cells_raw.mean(axis=0)
    # Centering makes alpha the mean-information intercept. Do not scale each
    # basis column separately: the RW2 prior below acts on adjacent B-spline
    # coefficients, and unequal column scaling would destroy that geometry.
    basis_scale = np.ones(basis_cells_raw.shape[1], dtype=float)

    def transform(values: np.ndarray) -> np.ndarray:
        raw = transformer.transform(values.reshape(-1, 1))
        return (raw - basis_center) / basis_scale

    grid_log = np.linspace(
        float(cells["log2_rarity"].min()),
        float(cells["log2_rarity"].max()),
        int(grid_size),
    )
    spline_cells = transform(cells["log2_rarity"].to_numpy(dtype=float))
    spline_observations = spline_cells[observations["cell_index"].to_numpy()]
    spline_grid = transform(grid_log)

    return PreparedRarityData(
        observations=observations.reset_index(drop=True),
        cells=cells,
        spline_observations=spline_observations,
        spline_cells=spline_cells,
        spline_grid=spline_grid,
        grid_event_counts=np.power(2.0, grid_log),
        cohort_index=cells["cohort"].astype(str).map(cohort_lookup).to_numpy(),
        outcome_index=cells["outcome"].astype(str).map(outcome_lookup).to_numpy(),
        family_index=(
            cells["outcome_family"].astype(str).map(family_lookup).to_numpy()
        ),
        cell_index=observations["cell_index"].to_numpy(dtype=int),
        cohorts=cohorts,
        outcomes=outcomes,
        families=families,
        basis_center=basis_center,
        basis_scale=basis_scale,
    )


def _require_bayesian_dependencies():
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:
        raise RuntimeError(
            "Hierarchical fitting requires the optional Bayesian dependencies. "
            "Install with `pip install -e '.[bayesian]'`."
        ) from exc
    return pm, az


def fit_hierarchical_rarity_model(
    prepared: PreparedRarityData,
    *,
    output_dir: str | Path,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    cores: Optional[int] = None,
    target_accept: float = 0.95,
    random_seed: int = 2026,
    prior_scale: float = 0.10,
    random_effect_scale: float = 0.05,
    require_convergence: bool = True,
) -> dict[str, Path]:
    """Fit the robust crossed-effects spline model and persist all artifacts."""
    pm, az = _require_bayesian_dependencies()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observations = prepared.observations
    cells = prepared.cells

    coords = {
        "observation": np.arange(len(observations)),
        "cell": cells["cell_id"].tolist(),
        "cohort": prepared.cohorts,
        "outcome": prepared.outcomes,
        "family": prepared.families,
        "spline": np.arange(prepared.spline_cells.shape[1]),
        "grid": np.arange(len(prepared.grid_event_counts)),
    }
    with pm.Model(coords=coords):
        spline_cells = pm.Data(
            "spline_cells",
            prepared.spline_cells,
            dims=("cell", "spline"),
        )
        spline_grid = pm.Data(
            "spline_grid",
            prepared.spline_grid,
            dims=("grid", "spline"),
        )
        cell_index = pm.Data(
            "cell_index",
            prepared.cell_index,
            dims="observation",
        )
        observed_se = pm.Data(
            "observed_se",
            observations["difference_se"].to_numpy(dtype=float),
            dims="observation",
        )

        alpha = pm.Normal("alpha", mu=0.0, sigma=prior_scale)
        smooth_scale = pm.HalfNormal("smooth_scale", sigma=prior_scale / 2.0)
        n_splines = prepared.spline_cells.shape[1]
        if n_splines >= 3:
            spline_initial = pm.Normal(
                "spline_initial",
                mu=0.0,
                sigma=prior_scale,
                shape=2,
            )
            spline_innovation = pm.Normal(
                "spline_innovation",
                mu=0.0,
                sigma=smooth_scale,
                shape=n_splines - 2,
            )
            coefficient_values = [spline_initial[0], spline_initial[1]]
            for index in range(n_splines - 2):
                coefficient_values.append(
                    2.0 * coefficient_values[-1]
                    - coefficient_values[-2]
                    + spline_innovation[index]
                )
            spline_coef = pm.Deterministic(
                "spline_coef",
                pm.math.stack(coefficient_values),
                dims="spline",
            )
        else:
            spline_coef = pm.Normal(
                "spline_coef",
                mu=0.0,
                sigma=prior_scale,
                dims="spline",
            )

        variance_components = []
        if len(prepared.cohorts) > 1:
            sigma_cohort = pm.HalfNormal("sigma_cohort", sigma=random_effect_scale)
            cohort_z = pm.Normal("cohort_z", 0.0, 1.0, dims="cohort")
            cohort_effect = sigma_cohort * cohort_z[prepared.cohort_index]
            variance_components.append(sigma_cohort**2)
        else:
            cohort_effect = np.zeros(len(cells), dtype=float)

        if len(prepared.outcomes) > 1:
            sigma_outcome = pm.HalfNormal("sigma_outcome", sigma=random_effect_scale)
            outcome_z = pm.Normal("outcome_z", 0.0, 1.0, dims="outcome")
            outcome_effect = sigma_outcome * outcome_z[prepared.outcome_index]
            variance_components.append(sigma_outcome**2)
        else:
            outcome_effect = np.zeros(len(cells), dtype=float)

        if len(prepared.families) > 1:
            sigma_family = pm.HalfNormal("sigma_family", sigma=random_effect_scale)
            family_z = pm.Normal("family_z", 0.0, 1.0, dims="family")
            family_effect = sigma_family * family_z[prepared.family_index]
            variance_components.append(sigma_family**2)
        else:
            family_effect = np.zeros(len(cells), dtype=float)

        sigma_cell = pm.HalfNormal("sigma_cell", sigma=random_effect_scale)
        variance_components.append(sigma_cell**2)
        sigma_training = pm.HalfNormal(
            "sigma_training", sigma=random_effect_scale / 2.0
        )

        cell_z = pm.Normal("cell_z", 0.0, 1.0, dims="cell")

        smooth_cell = pm.math.dot(spline_cells, spline_coef)
        cell_theta = pm.Deterministic(
            "cell_theta",
            alpha
            + smooth_cell
            + cohort_effect
            + outcome_effect
            + family_effect
            + sigma_cell * cell_z,
            dims="cell",
        )
        pm.Deterministic(
            "population_curve",
            alpha + pm.math.dot(spline_grid, spline_coef),
            dims="grid",
        )
        pm.Deterministic(
            "population_predictive_sd",
            pm.math.sqrt(sum(variance_components)),
        )
        nu = pm.Deterministic("nu", 2.0 + pm.Exponential("nu_minus_two", 0.1))
        likelihood_scale = pm.math.sqrt(observed_se**2 + sigma_training**2)
        pm.StudentT(
            "observed_delta",
            nu=nu,
            mu=cell_theta[cell_index],
            sigma=likelihood_scale,
            observed=observations["difference"].to_numpy(dtype=float),
            dims="observation",
        )

        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=cores,
            target_accept=float(target_accept),
            random_seed=int(random_seed),
            return_inferencedata=True,
        )
        pm.sample_posterior_predictive(
            idata,
            var_names=["observed_delta"],
            random_seed=int(random_seed),
            extend_inferencedata=True,
        )

    posterior_path = output / "posterior.nc"
    idata.to_netcdf(posterior_path)

    curve = (
        idata.posterior["population_curve"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "grid")
        .values
    )
    predictive_sd = (
        idata.posterior["population_predictive_sd"]
        .stack(sample=("chain", "draw"))
        .values.reshape(-1)
    )
    rng = np.random.default_rng(random_seed)
    predictive = curve + rng.normal(size=curve.shape) * predictive_sd[:, None]

    def quantiles(values: np.ndarray) -> dict[str, np.ndarray]:
        q = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
        return {
            "lower_95": q[0],
            "lower_50": q[1],
            "median": q[2],
            "upper_50": q[3],
            "upper_95": q[4],
        }

    curve_summary = pd.DataFrame(
        {
            "training_events": prepared.grid_event_counts,
            **quantiles(curve),
            "probability_benefit": (curve > 0).mean(axis=0),
            "predictive_lower_95": np.quantile(predictive, 0.025, axis=0),
            "predictive_upper_95": np.quantile(predictive, 0.975, axis=0),
        }
    )
    curve_path = output / "posterior_curve.csv"
    curve_summary.to_csv(curve_path, index=False)

    cell_draws = (
        idata.posterior["cell_theta"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "cell")
        .values
    )
    cell_quantiles = quantiles(cell_draws)
    cell_summary = cells.copy()
    for name, values in cell_quantiles.items():
        cell_summary[f"posterior_{name}"] = values
    cell_summary["posterior_probability_benefit"] = (cell_draws > 0).mean(axis=0)
    cell_path = output / "posterior_cells.csv"
    cell_summary.to_csv(cell_path, index=False)

    diagnostics = az.summary(idata, round_to=None).reset_index(names="parameter")
    diagnostics_path = output / "diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    divergences = int(idata.sample_stats["diverging"].sum().values)
    finite_rhat = diagnostics["r_hat"].dropna() if "r_hat" in diagnostics else []
    max_rhat = float(max(finite_rhat)) if len(finite_rhat) else float("nan")
    diagnostic_summary = {
        "divergences": divergences,
        "max_rhat": max_rhat,
        "n_observations": len(observations),
        "n_cells": len(cells),
        "n_cohorts": len(prepared.cohorts),
        "n_outcomes": len(prepared.outcomes),
    }
    diagnostic_json = output / "diagnostics.json"
    diagnostic_json.write_text(json.dumps(diagnostic_summary, indent=2))
    if require_convergence and (
        divergences > 0 or (np.isfinite(max_rhat) and max_rhat > 1.01)
    ):
        raise RuntimeError(
            "Bayesian rarity model failed the convergence gate: "
            f"divergences={divergences}, max_rhat={max_rhat:.4f}. "
            f"Inspect {diagnostics_path}."
        )
    return {
        "posterior": posterior_path,
        "curve": curve_path,
        "cells": cell_path,
        "diagnostics": diagnostics_path,
        "diagnostics_json": diagnostic_json,
    }


def write_model_input_artifacts(
    prepared: PreparedRarityData,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist the exact observations and spline transform used for fitting."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observations_path = output / "model_observations.csv"
    cells_path = output / "model_cells.csv"
    specification_path = output / "model_design.json"
    prepared.observations.to_csv(observations_path, index=False)
    prepared.cells.to_csv(cells_path, index=False)
    specification_path.write_text(
        json.dumps(
            {
                "cohorts": prepared.cohorts,
                "outcomes": prepared.outcomes,
                "families": prepared.families,
                "basis_center": prepared.basis_center.tolist(),
                "basis_scale": prepared.basis_scale.tolist(),
                "grid_event_counts": prepared.grid_event_counts.tolist(),
            },
            indent=2,
        )
    )
    return {
        "observations": observations_path,
        "cells": cells_path,
        "design": specification_path,
    }
