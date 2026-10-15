"""Question-aligned partial pooling of locked, positive-oriented paired gains.

The likelihood uses shared patient-bootstrap covariance within fine disease.
Seeds are averaged before fitting: inference conditions on the fitted roster.
Derived factorial contrasts are deterministic, never additional observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

# Rows are output contrasts; columns are independent fitted contrasts.
FAMILIES = {
    "pooling": {
        "P1": {"P1": 1},
        "P2": {"P2": 1},
        "P6": {"P6": 1},
        "P3": {"P1": 1, "P6": 1},
        "P4": {"P2": 1, "P6": 1},
        "P5": {"P1": 1, "P2": 1, "P6": 1},
    },
    "routing": {"R1": {"R1": 1}},
    "source_banks": {
        "R2": {"R2": 1},
        "R3": {"R3": 1},
        "R7": {"R7": 1},
        "R4": {"R3": 1, "R7": 1},
        "R5": {"R2": 1, "R7": 1},
        "R6": {"R2": 1, "R3": 1, "R7": 1},
    },
    "tabular": {
        "T2": {"T2": 1},
        "T3": {"T3": 1},
        "R1": {"R1": 1},
        "T4": {"T4": 1},
        "T1": {"T2": 1, "R1": 1},
    },
    "external_tabular": {
        "T2": {"T2": 1},
        "R1": {"R1": 1},
        "E1": {"E1": 1},
        "A1": {"A1": 1},
        "I1": {"I1": 1},
        "I2": {"I2": 1},
        "T5_shared": {"T2": 1, "E1": 1, "A1": 1},
        "T5_router": {"T2": 1, "R1": 1, "E1": 1, "A1": 1, "I1": 1, "I2": 1},
    },
    "external": {
        "E1": {"E1": 1},
        "A1": {"A1": 1},
        "I1": {"I1": 1},
        "I2": {"I2": 1},
        "E2": {"E1": 1, "I1": 1},
        "A2": {"A1": 1, "I2": 1},
        "E3": {"E1": 1, "A1": 1},
        "E4": {"E1": 1, "A1": 1, "I1": 1, "I2": 1},
        "I3": {"I1": 1, "I2": 1},
    },
}
KEY = ["split", "seed", "cohort", "outcome", "horizon_days", "contrast", "metric"]


@dataclass
class ContrastData:
    cells: pd.DataFrame
    observed: np.ndarray
    bootstrap: np.ndarray
    basis: list[str]
    contrasts: list[str]
    transform: np.ndarray
    blocks: list[dict]
    weights: np.ndarray
    design: np.ndarray
    grid_design: np.ndarray
    grid_support: np.ndarray
    cohort_index: np.ndarray
    outcome_index: np.ndarray
    seeds: tuple[int, ...]
    specification: dict


def contrast_design(family):
    definitions = FAMILIES[family]
    basis = list(dict.fromkeys(name for row in definitions.values() for name in row))
    contrasts = list(definitions)
    matrix = np.array([[definitions[c].get(b, 0) for b in basis] for c in contrasts])
    return basis, contrasts, matrix


# Panel weights: every disease counts equally; within a disease, every loss
# endpoint (grade variants of one finding together) counts equally
# (primary), or every clinical display domain (sensitivity).
WEIGHTING_COLUMNS = {"equal_disease_loss_endpoint": "loss_endpoint",
                     "equal_disease_endpoint_group": "endpoint_group"}


def panel_weights(cells, weighting):
    if weighting == "equal_cell":
        return np.full(len(cells), 1 / len(cells))
    if weighting not in WEIGHTING_COLUMNS:
        raise ValueError("Unknown panel weighting")
    column = WEIGHTING_COLUMNS[weighting]
    if column == "loss_endpoint" and column not in cells:
        # Older cell metadata: grade variants (_g2plus/_g3plus) form one endpoint.
        cells = cells.assign(loss_endpoint=cells.outcome.astype(str).str.replace(
            r"_g[23]plus$", "", regex=True))
    if column not in cells or cells[column].isna().any():
        raise ValueError(f"{weighting} weighting requires {column} in the cell metadata")
    groups = cells.groupby("cohort")[column].transform("nunique")
    members = cells.groupby(["cohort", column]).outcome.transform("count")
    return (1 / (cells.cohort.nunique() * groups * members)).to_numpy(float)


def estimable_cell_roster(
    effects,
    index,
    gains,
    *,
    family,
    metric,
    horizon_days=365,
    split="held_out",
    seeds=(42, 43, 44),
    rarity_column="n_events_train",
    min_cases=5,
    min_controls=5,
    minimum_draws=100,
    minimum_finite_fraction=0.8,
):
    """Apply a prespecified estimability rule and return (roster, exclusions).

    The rule depends only on support, missingness and bootstrap degeneracy,
    never on the size or sign of an effect. A cell enters the roster when every
    basis contrast of ``family`` exists for every seed with finite gain and
    training support, AUC cells have at least ``min_cases``/``min_controls`` in
    every evaluated row, and enough seed-averaged bootstrap draws are finite.
    Within each disease, cells are then removed (fewest finite draws first)
    until the jointly finite draws required for the covariance block exist.
    Every excluded cell is reported with its reason; nothing is intersected
    silently, and the roster is written for review before fitting.
    """
    basis, _, _ = contrast_design(family)
    seeds = tuple(int(seed) for seed in seeds)
    gains = np.asarray(gains, dtype=float)
    n_draws = gains.shape[1]
    required = max(minimum_draws, int(np.ceil(minimum_finite_fraction * n_draws)))
    if n_draws < minimum_draws:
        raise ValueError("Too few bootstrap draws for any Bayesian contrast fit")
    target = (
        effects.metric.eq(metric)
        & effects.horizon_days.eq(horizon_days)
        & effects.split.eq(split)
        & effects.seed.isin(seeds)
    )
    candidates = (
        effects.loc[target, ["cohort", "outcome"]]
        .drop_duplicates()
        .sort_values(["cohort", "outcome"])
    )
    selected = effects.loc[target & effects.contrast.isin(basis)].set_index(KEY)
    rows_by_key = (
        index.reset_index(drop=True).reset_index(names="draw_row").set_index(KEY)
    )
    kept, excluded, draws = [], [], {}
    for cell in candidates.itertuples(index=False):
        keys = [
            (split, seed, cell.cohort, cell.outcome, horizon_days, contrast, metric)
            for contrast in basis
            for seed in seeds
        ]
        reason = None
        if not all(key in selected.index for key in keys) or not all(
            key in rows_by_key.index for key in keys
        ):
            reason = "missing_contrast_model_or_seed"
        else:
            values = selected.loc[keys]
            if rarity_column not in values or not np.isfinite(
                pd.to_numeric(values[rarity_column], errors="coerce")
            ).all():
                reason = "missing_train_support"
            elif metric == "auc" and (
                {"n_cases", "n_controls"} - set(values)
                or (values.n_cases < min_cases).any()
                or (values.n_controls < min_controls).any()
            ):
                reason = "auc_evaluation_support_below_minimum_or_single_class"
            elif not np.isfinite(pd.to_numeric(values.gain, errors="coerce")).all():
                reason = "nonfinite_point_estimate"
            else:
                sampled = gains[rows_by_key.loc[keys].draw_row.to_numpy(int)].reshape(
                    len(basis), len(seeds), n_draws
                ).mean(axis=1)
                if (np.isfinite(sampled).sum(axis=1) < required).any():
                    reason = "insufficient_finite_bootstrap_draws"
                else:
                    draws[(cell.cohort, cell.outcome)] = sampled
        if reason is None:
            kept.append((cell.cohort, cell.outcome))
        else:
            excluded.append(
                {"cohort": cell.cohort, "outcome": cell.outcome, "reason": reason}
            )
    roster = []
    for cohort in sorted({cohort for cohort, _ in kept}):
        block = [pair for pair in kept if pair[0] == cohort]
        while block:
            stacked = np.concatenate([draws[pair] for pair in block])
            if np.isfinite(stacked).all(axis=0).sum() >= required:
                break
            worst = min(
                block, key=lambda pair: (np.isfinite(draws[pair]).sum(), pair)
            )
            block.remove(worst)
            excluded.append(
                {
                    "cohort": worst[0],
                    "outcome": worst[1],
                    "reason": "joint_bootstrap_degeneracy_in_disease_block",
                }
            )
        roster.extend(block)
    return (
        pd.DataFrame(roster, columns=["cohort", "outcome"]),
        pd.DataFrame(excluded, columns=["cohort", "outcome", "reason"]),
    )


def prepare_contrast_data(
    effects,
    index,
    gains,
    *,
    family,
    metric,
    horizon_days=365,
    split="held_out",
    seeds=(42, 43, 44),
    trend="linear",
    spline_knots=4,
    rarity_column="n_events_train",
    weighting="equal_disease_loss_endpoint",
    covariance_shrinkage=0.02,
    se_floor=1e-6,
    minimum_draws=100,
    minimum_cells=8,
    grid_size=100,
    cell_roster=None,
    support_bins=None,
    minimum_cells_per_bin=3,
):
    """Exact row alignment; no implicit seed, horizon or cell intersections.

    ``trend`` describes how gains may depend on target-cell training support:
    ``none`` (exchangeable cells), ``bins`` (one mean per prespecified support
    bin; no shape assumption), or the sensitivity models ``linear`` (a fixed
    change per doubling of support) and ``spline``.
    """
    if family not in FAMILIES or metric not in {"auc", "ipa", "brier", "log_loss"}:
        raise ValueError(
            "Choose a declared question family and positive-oriented metric"
        )
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify a unique nonempty fitted seed roster")
    if trend not in {"none", "bins", "linear", "spline"}:
        raise ValueError("trend must be none, bins, linear or spline")
    if trend == "bins":
        edges = [float(edge) for edge in (support_bins or [])]
        if len(edges) < 2 or edges[0] != 0 or edges != sorted(set(edges)):
            raise ValueError(
                "Binned support needs prespecified increasing edges starting at 0"
            )
    if not 0 <= covariance_shrinkage <= 1 or se_floor <= 0:
        raise ValueError("Invalid covariance regularization")
    basis, contrasts, transform = contrast_design(family)
    for table in (effects, index):
        if set(KEY) - set(table) or table[KEY].isna().any().any():
            raise ValueError("Missing contrast identity columns")
        if table.duplicated(KEY).any():
            raise ValueError("Duplicate contrast/seed/cell rows")
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 2 or len(gains) != len(index) or gains.shape[1] < minimum_draws:
        raise ValueError(
            "Bootstrap array must align exactly with its index and have enough draws"
        )

    def mask(frame):
        return (
            frame.metric.eq(metric)
            & frame.horizon_days.eq(horizon_days)
            & frame.split.eq(split)
            & frame.contrast.isin(basis)
            & frame.seed.isin(seeds)
        )

    target_mask = (
        effects.metric.eq(metric)
        & effects.horizon_days.eq(horizon_days)
        & effects.split.eq(split)
        & effects.seed.isin(seeds)
    )
    available_cells = set(
        map(
            tuple,
            effects.loc[target_mask, ["cohort", "outcome"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if cell_roster is None:
        declared_cells = available_cells
    else:
        if (
            {"cohort", "outcome"} - set(cell_roster)
            or cell_roster[["cohort", "outcome"]].isna().any().any()
            or cell_roster.duplicated(["cohort", "outcome"]).any()
        ):
            raise ValueError(
                "Cell roster must uniquely declare nonmissing disease/outcome pairs"
            )
        declared_cells = set(map(tuple, cell_roster[["cohort", "outcome"]].to_numpy()))
        if not declared_cells or not declared_cells <= available_cells:
            raise ValueError(
                "Declared cell roster contains missing prediction-analysis cells"
            )
    rows = effects.loc[mask(effects)].copy()
    rows = rows.loc[
        [
            tuple(pair) in declared_cells
            for pair in rows[["cohort", "outcome"]].to_numpy()
        ]
    ]
    if rows.empty or {"gain", rarity_column} - set(rows):
        raise ValueError("Missing gain or train-only support for requested contrasts")
    for column in ("gain", rarity_column):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if not np.isfinite(rows[column]).all():
            raise ValueError(
                "Nonestimable effect/support: declare coverage before Bayesian fitting"
            )
    if (rows[rarity_column] < 0).any():
        raise ValueError("Training support cannot be negative")
    # A cell must mean one disease/outcome, with stable train-only metadata.
    identity = ["cohort", "outcome"]
    metadata = [rarity_column] + [
        c
        for c in (
            "endpoint_group",
            "loss_endpoint",
            "cohort_group",
            "outcome_family",
            "minority_class",
            "n_cases",
            "n_controls",
        )
        if c in rows
    ]
    for column in metadata:
        if (
            rows[column].isna().any()
            or (rows.groupby(identity)[column].nunique() > 1).any()
        ):
            raise ValueError(f"Cell metadata changes across contrasts/seeds: {column}")
    for child, parent in (
        ("outcome", "endpoint_group"),
        ("outcome", "loss_endpoint"),
        ("cohort", "cohort_group"),
        ("outcome", "outcome_family"),
    ):
        if parent in rows and (rows.groupby(child)[parent].nunique() > 1).any():
            raise ValueError(f"Each {child} must have one {parent}")
    cells = (
        rows[identity + metadata]
        .drop_duplicates(identity)
        .sort_values(identity)
        .reset_index(drop=True)
    )
    if set(map(tuple, cells[identity].to_numpy())) != declared_cells:
        raise ValueError(
            "Some declared cells have no required family contrasts; lock coverage explicitly"
        )
    if len(cells) < minimum_cells:
        raise ValueError(f"Need at least {minimum_cells} complete cells")
    expected = pd.MultiIndex.from_tuples(
        [
            (split, seed, cell.cohort, cell.outcome, horizon_days, contrast, metric)
            for cell in cells.itertuples()
            for contrast in basis
            for seed in seeds
        ],
        names=KEY,
    )
    selected = rows.set_index(KEY)
    indexed = index.reset_index(drop=True).reset_index(names="draw_row").set_index(KEY)
    if (
        not expected.isin(selected.index).all()
        or not expected.isin(indexed.index).all()
    ):
        raise ValueError(
            "Incomplete contrast/cell/seed roster; no automatic intersection is allowed"
        )
    shape = (len(cells), len(basis), len(seeds))
    observed_by_seed = selected.reindex(expected).gain.to_numpy(float).reshape(shape)
    bootstrap_by_seed = gains[indexed.reindex(expected).draw_row.to_numpy(int)].reshape(
        *shape, gains.shape[1]
    )
    observed = observed_by_seed.mean(axis=2)
    # Require every seed finite for a draw; never shrink SE by treating seeds as patients.
    bootstrap = bootstrap_by_seed.mean(axis=2)
    weights = panel_weights(cells, weighting)
    blocks = []
    for cohort, group in cells.groupby("cohort", sort=True):
        positions = group.index.to_numpy()
        vectors = bootstrap[positions].reshape(-1, gains.shape[1])
        finite = np.isfinite(vectors).all(axis=0)
        if finite.sum() < max(minimum_draws, int(np.ceil(0.8 * gains.shape[1]))):
            raise ValueError(
                f"Insufficient jointly finite bootstrap draws for {cohort}"
            )
        raw_cov = np.atleast_2d(np.cov(vectors[:, finite], ddof=1))
        covariance = (
            (1 - covariance_shrinkage) * raw_cov
            + covariance_shrinkage * np.diag(np.diag(raw_cov))
            + se_floor**2 * np.eye(len(vectors))
        )
        blocks.append(
            {
                "cohort": str(cohort),
                "positions": positions,
                "covariance": covariance,
                "cholesky": np.linalg.cholesky(covariance),
                "finite_draws": int(finite.sum()),
                "condition_number": float(np.linalg.cond(covariance)),
            }
        )
    x = np.log2(1 + cells[rarity_column].to_numpy(float))
    center = float(x @ weights)
    grid = np.linspace(x.min(), x.max(), grid_size)
    if trend == "spline":
        if len(np.unique(x)) < spline_knots:
            raise ValueError(
                "Spline sensitivity needs enough distinct training-support values"
            )
        transformer = SplineTransformer(
            n_knots=spline_knots, degree=3, include_bias=False, knots="quantile"
        )
        raw = transformer.fit_transform(x[:, None])
        basis_center = weights @ raw
        design = raw - basis_center
        grid_design = transformer.transform(grid[:, None]) - basis_center
        transform_spec = {
            "knots": transformer.bsplines_[0].t.tolist(),
            "center": basis_center.tolist(),
            "degree": 3,
        }
    elif trend == "bins":
        support = cells[rarity_column].to_numpy(float)
        upper = [*edges[1:], np.inf]
        labels = [
            f"{edges[k]:g}-{upper[k]:g}" if np.isfinite(upper[k]) else f">={edges[k]:g}"
            for k in range(len(edges))
        ]
        membership = np.searchsorted(edges, support, side="right") - 1
        counts = np.bincount(membership, minlength=len(edges))
        if (counts < minimum_cells_per_bin).any():
            raise ValueError(
                "Each prespecified support bin needs at least "
                f"{minimum_cells_per_bin} cells; counts {dict(zip(labels, counts.tolist()))}"
            )
        one_hot = np.eye(len(edges))[membership]
        bin_center = weights @ one_hot
        # Weighted-centred indicators keep alpha as the panel-weighted mean.
        design = one_hot - bin_center
        grid_design = np.eye(len(edges)) - bin_center
        grid = np.log2(1 + np.asarray(edges))
        transform_spec = {
            "support_bins": edges,
            "bin_labels": labels,
            "cells_per_bin": counts.tolist(),
            "interpretation": "piecewise-constant mean gain per prespecified support bin",
        }
    elif trend == "linear":
        design, grid_design = (x - center)[:, None], (grid - center)[:, None]
        transform_spec = {
            "log_support_center": center,
            "slope_unit": "one doubling of (1 + training support)",
        }
    else:
        design, grid_design = np.zeros((len(cells), 1)), np.zeros((grid_size, 1))
        transform_spec = {}
    specification = {
        "family": family,
        "metric": metric,
        "split": split,
        "horizon_days": horizon_days,
        "trend": trend,
        "rarity_column": rarity_column,
        "weighting": weighting,
        "covariance_shrinkage": covariance_shrinkage,
        "se_floor": se_floor,
        "cell_effect_inference": "analytically_marginalized_gaussian_residuals_then_exact_conditional_simulation",
        "covariance": "shared_patient_bootstrap_within_fine_disease_across_endpoints_contrasts_and_seeds",
        "seed_uncertainty": "conditional_on_fitted_roster_not_all_training_randomness",
        "n_draws": gains.shape[1],
        "cell_roster_explicit": cell_roster is not None,
        "available_cells": len(available_cells),
        "declared_cells": len(declared_cells),
        "transform": transform_spec,
    }
    return ContrastData(
        cells,
        observed,
        bootstrap,
        basis,
        contrasts,
        transform,
        blocks,
        weights,
        design,
        grid_design,
        np.exp2(grid) - 1,
        pd.Categorical(cells.cohort, categories=sorted(cells.cohort.unique())).codes,
        pd.Categorical(cells.outcome, categories=sorted(cells.outcome.unique())).codes,
        seeds,
        specification,
    )


def build_contrast_model(
    data,
    *,
    prior_scale,
    random_effect_scale,
    slope_scale,
    nest_diseases=False,
    nest_outcomes=False,
):
    """Crossed intercepts per primitive effect, with a known correlated likelihood."""
    from opera.analysis.bayesian_rarity import _require_bayesian_dependencies

    pm, _ = _require_bayesian_dependencies()
    if (
        not np.isfinite([prior_scale, random_effect_scale, slope_scale]).all()
        or min(prior_scale, random_effect_scale, slope_scale) <= 0
    ):
        raise ValueError(
            "Prior scales must be positive and specified on the selected metric scale"
        )
    coords = {
        "cell": np.arange(len(data.cells)),
        "basis": data.basis,
        "contrast": data.contrasts,
        "cohort": sorted(data.cells.cohort.unique()),
        "outcome": sorted(data.cells.outcome.unique()),
        "feature": np.arange(data.design.shape[1]),
        "grid": np.arange(len(data.grid_support)),
    }
    for nested, column, coordinate in (
        (nest_diseases, "cohort_group", "disease_group"),
        (nest_outcomes, "outcome_family", "family"),
    ):
        if nested:
            if column not in data.cells:
                raise ValueError(f"Nesting requires explicit {column} metadata")
            coords[coordinate] = sorted(data.cells[column].unique())
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0, prior_scale, dims="basis")
        if data.specification["trend"] == "none":
            coefficient = np.zeros((data.design.shape[1], len(data.basis)))
        else:
            coefficient = pm.Normal(
                "trend_coefficient", 0, slope_scale, dims=("feature", "basis")
            )
        cell_mean = alpha + pm.math.dot(data.design, coefficient)
        curve = alpha + pm.math.dot(data.grid_design, coefficient)
        for label, coordinate, index in (
            ("disease", "cohort", data.cohort_index),
            ("outcome", "outcome", data.outcome_index),
        ):
            if len(coords[coordinate]) > 1:
                scale = pm.HalfNormal(
                    f"sigma_{label}", random_effect_scale, dims="basis"
                )
                z = pm.Normal(f"z_{label}", 0, 1, dims=(coordinate, "basis"))
                cell_mean = cell_mean + scale * (z - pm.math.mean(z, axis=0))[index]
        for enabled, label, column, coordinate in (
            (nest_diseases, "disease_group", "cohort_group", "disease_group"),
            (nest_outcomes, "family", "outcome_family", "family"),
        ):
            if enabled and len(coords[coordinate]) > 1:
                index = pd.Categorical(
                    data.cells[column], categories=coords[coordinate]
                ).codes
                scale = pm.HalfNormal(
                    f"sigma_{label}", random_effect_scale, dims="basis"
                )
                z = pm.Normal(f"z_{label}", 0, 1, dims=(coordinate, "basis"))
                cell_mean = cell_mean + scale * (z - pm.math.mean(z, axis=0))[index]
        sigma_cell = pm.HalfNormal("sigma_cell", random_effect_scale, dims="basis")
        mean = pm.Deterministic("basis_mean", cell_mean, dims=("cell", "basis"))
        pm.Deterministic(
            "population_curve",
            pm.math.dot(curve, data.transform.T),
            dims=("grid", "contrast"),
        )
        import pytensor.tensor as pt

        for number, block in enumerate(data.blocks):
            variance = pt.tile(sigma_cell**2, len(block["positions"]))
            # Marginalize Gaussian cell residuals; do not sample a high-dimensional
            # cell funnel when paired effects are precise or nearly identical.
            pm.MvNormal(
                f"observed_{number}",
                mu=mean[block["positions"]].reshape((-1,)),
                cov=block["covariance"] + pt.diag(variance),
                observed=data.observed[block["positions"]].reshape(-1),
            )
    return model


def add_conditional_cell_draws(idata, data, *, prior=False, random_seed=2026):
    """Exact Gaussian conditioning after marginal sampling, including cell variance.

    Conditional simulation uses a prior latent/noise pair and its residual
    correction, avoiding a second conditional-covariance eigendecomposition.
    """
    from scipy.linalg import cho_factor, cho_solve
    import xarray as xr

    group = idata.prior if prior else idata.posterior
    mean_array = group["basis_mean"].transpose("chain", "draw", "cell", "basis")
    mean = mean_array.values
    sigma = group["sigma_cell"].transpose("chain", "draw", "basis").values
    rng = np.random.default_rng(random_seed)
    latent = mean + rng.normal(size=mean.shape) * sigma[:, :, None, :]
    if not prior:
        for chain in range(mean.shape[0]):
            for draw in range(mean.shape[1]):
                for block in data.blocks:
                    positions = block["positions"]
                    variance = np.tile(sigma[chain, draw] ** 2, len(positions))
                    current = latent[chain, draw, positions].reshape(-1)
                    simulated_noise = block["cholesky"] @ rng.normal(size=len(current))
                    correction = cho_solve(
                        cho_factor(block["covariance"] + np.diag(variance), lower=True),
                        data.observed[positions].reshape(-1)
                        - current
                        - simulated_noise,
                    )
                    latent[chain, draw, positions] = (
                        current + variance * correction
                    ).reshape(len(positions), len(data.basis))
    group["basis_effect"] = mean_array.copy(data=latent)
    effects = latent @ data.transform.T
    coords = {
        "chain": mean_array.chain,
        "draw": mean_array.draw,
        "cell": mean_array.cell,
        "contrast": data.contrasts,
    }
    group["cell_effect"] = xr.DataArray(
        effects, dims=("chain", "draw", "cell", "contrast"), coords=coords
    )
    group["panel_effect"] = (
        group["cell_effect"] * xr.DataArray(data.weights, dims="cell")
    ).sum("cell")
    return idata


def summarize_draws(values, tolerance):
    values = np.asarray(values)
    lo, med, hi = np.quantile(values, [0.025, 0.5, 0.975], axis=0)
    return {
        "posterior_lower_95": lo,
        "posterior_median": med,
        "posterior_upper_95": hi,
        "probability_benefit": (values > 0).mean(axis=0),
        "probability_meaningful_benefit": (values > tolerance).mean(axis=0),
        "probability_meaningful_harm": (values < -tolerance).mean(axis=0),
        "probability_practical_equivalence": (np.abs(values) <= tolerance).mean(axis=0),
    }


def fit_contrast_model(
    data,
    *,
    output_dir,
    prior_scale,
    random_effect_scale,
    slope_scale,
    practical_tolerance,
    nest_diseases=False,
    nest_outcomes=False,
    draws=2000,
    tune=2000,
    chains=4,
    cores=4,
    prior_draws=500,
    prior_only=False,
    target_accept=0.95,
    random_seed=2026,
    require_convergence=True,
    minimum_ess=400,
    max_rhat=1.01,
    minimum_bfmi=0.3,
    progressbar=True,
):
    """Persist priors, posterior, predictive checks, raw effects and audited weights."""
    from opera.analysis.bayesian_rarity import _require_bayesian_dependencies

    pm, az = _require_bayesian_dependencies()
    if (
        not np.isfinite(practical_tolerance)
        or practical_tolerance <= 0
        or prior_draws < 1
    ):
        raise ValueError(
            "Lock a positive metric-scale tolerance and prior predictive sample count"
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh immutable Bayesian output directory")
    model = build_contrast_model(
        data,
        prior_scale=prior_scale,
        random_effect_scale=random_effect_scale,
        slope_scale=slope_scale,
        nest_diseases=nest_diseases,
        nest_outcomes=nest_outcomes,
    )
    observed_names = [f"observed_{i}" for i in range(len(data.blocks))]
    with model:
        prior = pm.sample_prior_predictive(samples=prior_draws, random_seed=random_seed)
        add_conditional_cell_draws(prior, data, prior=True, random_seed=random_seed + 1)
        if prior_only:
            output.mkdir(parents=True)
            prior.to_netcdf(output / "prior_predictive.nc")
            values = (
                prior.prior["panel_effect"]
                .stack(sample=("chain", "draw"))
                .transpose("sample", "contrast")
                .values
            )
            summary = summarize_draws(values, practical_tolerance)
            pd.DataFrame(
                {
                    "contrast": data.contrasts,
                    **{
                        key.replace("posterior_", "prior_"): value
                        for key, value in summary.items()
                    },
                }
            ).to_csv(output / "prior_panel.csv", index=False)
            (output / "model_specification.json").write_text(
                json.dumps(
                    {
                        **data.specification,
                        "status": "prior_only_no_posterior_fitted",
                        "basis": data.basis,
                        "contrasts": data.contrasts,
                        "nest_diseases": nest_diseases,
                        "nest_outcomes": nest_outcomes,
                        "priors": {
                            "intercept_scale": prior_scale,
                            "random_effect_scale": random_effect_scale,
                            "slope_scale": slope_scale,
                        },
                        "practical_tolerance": practical_tolerance,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            return output
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )
        pm.sample_posterior_predictive(
            idata,
            var_names=observed_names,
            random_seed=random_seed,
            progressbar=progressbar,
            extend_inferencedata=True,
        )
    add_conditional_cell_draws(idata, data, random_seed=random_seed + 2)
    output.mkdir(parents=True)
    idata.to_netcdf(output / "posterior.nc")
    prior.to_netcdf(output / "prior_predictive.nc")
    diagnostics = az.summary(idata, kind="diagnostics").reset_index(names="parameter")
    diagnostics.to_csv(output / "diagnostics.csv", index=False)
    rhat, ess = diagnostics.r_hat.to_numpy(), diagnostics.ess_bulk.to_numpy()
    tail = diagnostics.ess_tail.to_numpy()
    reached = idata.sample_stats.get("reached_max_treedepth")
    max_depth_hits = (
        int(reached.sum())
        if reached is not None
        else int((idata.sample_stats.tree_depth >= 10).sum())
    )
    bfmi = np.asarray(az.bfmi(idata), dtype=float)
    convergence = bool(
        chains >= 2
        and np.isfinite(rhat).all()
        and np.isfinite(ess).all()
        and np.isfinite(tail).all()
        and rhat.max() <= max_rhat
        and min(ess.min(), tail.min()) >= minimum_ess
        and int(idata.sample_stats.diverging.sum()) == 0
        and max_depth_hits == 0
        and np.isfinite(bfmi).all()
        and (bfmi >= minimum_bfmi).all()
    )

    def samples(variable, dimensions):
        return (
            idata.posterior[variable]
            .stack(sample=("chain", "draw"))
            .transpose("sample", *dimensions)
            .values
        )

    theta = samples("cell_effect", ["cell", "contrast"])
    curve = samples("population_curve", ["grid", "contrast"])
    panel = samples("panel_effect", ["contrast"])
    summaries = summarize_draws(theta, practical_tolerance)
    cell_rows, curve_rows, panel_rows = [], [], []
    raw_draws = np.einsum("ckb,ok->cob", data.bootstrap, data.transform)
    raw_effect = data.observed @ data.transform.T
    for j, contrast in enumerate(data.contrasts):
        for i, cell in data.cells.iterrows():
            finite = raw_draws[i, j][np.isfinite(raw_draws[i, j])]
            lo, hi = np.quantile(finite, [0.025, 0.975])
            cell_rows.append(
                {
                    **cell.to_dict(),
                    "contrast": contrast,
                    "raw_gain": raw_effect[i, j],
                    "raw_lower_95": lo,
                    "raw_upper_95": hi,
                    "panel_weight": data.weights[i],
                    **{k: v[i, j] for k, v in summaries.items()},
                }
            )
        curve_summary = summarize_draws(curve[:, :, j], practical_tolerance)
        for i, support in enumerate(data.grid_support):
            curve_rows.append(
                {
                    "contrast": contrast,
                    "training_support": support,
                    **{k: v[i] for k, v in curve_summary.items()},
                }
            )
        raw_panel = data.weights @ raw_draws[:, j, :]
        finite = raw_panel[np.isfinite(raw_panel)]
        enough = len(finite) >= max(100, int(np.ceil(0.8 * len(raw_panel))))
        lo, hi = np.quantile(finite, [0.025, 0.975]) if enough else (np.nan, np.nan)
        panel_rows.append(
            {
                "contrast": contrast,
                "raw_gain": data.weights @ raw_effect[:, j],
                "raw_valid_draws": len(finite),
                "raw_interval_status": "paired_patient_bootstrap"
                if enough
                else "insufficient_joint_draws",
                "raw_lower_95": lo,
                "raw_upper_95": hi,
                **summarize_draws(panel[:, j], practical_tolerance),
            }
        )
    pd.DataFrame(cell_rows).to_csv(output / "posterior_cells.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "posterior_curve.csv", index=False)
    pd.DataFrame(panel_rows).to_csv(output / "posterior_panel.csv", index=False)
    predictive_checks = []
    for number, block in enumerate(data.blocks):
        actual = data.observed[block["positions"]].reshape(-1)
        chol = block["cholesky"]
        for label, group in (
            ("prior", prior.prior_predictive),
            ("posterior", idata.posterior_predictive),
        ):
            replicated = group[f"observed_{number}"].values.reshape(-1, len(actual))
            center = replicated.mean(axis=0)
            actual_discrepancy = np.mean(np.linalg.solve(chol, (actual - center)) ** 2)
            replicate_discrepancy = np.mean(
                np.linalg.solve(chol, (replicated - center).T) ** 2, axis=0
            )
            lo, hi = np.quantile(replicated, [0.025, 0.975], axis=0)
            predictive_checks.append(
                {
                    "cohort": block["cohort"],
                    "distribution": label,
                    "observed_inside_pointwise_95_fraction": float(
                        ((actual >= lo) & (actual <= hi)).mean()
                    ),
                    "whitened_discrepancy_tail_fraction": float(
                        (replicate_discrepancy >= actual_discrepancy).mean()
                    ),
                }
            )
    pd.DataFrame(predictive_checks).to_csv(
        output / "predictive_checks.csv", index=False
    )
    audit = {
        **data.specification,
        "runtime_versions": {
            "pymc": pm.__version__,
            "arviz": az.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "diagnostic_thresholds": {
            "minimum_bulk_and_tail_ess": minimum_ess,
            "max_rhat": max_rhat,
            "required_chains": 2,
            "allowed_divergences": 0,
            "allowed_max_tree_depth_hits": 0,
            "minimum_bfmi": minimum_bfmi,
        },
        "basis": data.basis,
        "contrasts": data.contrasts,
        "seeds": list(data.seeds),
        "contrast_transform": data.transform.tolist(),
        "nest_diseases": nest_diseases,
        "nest_outcomes": nest_outcomes,
        "priors": {
            "intercept_scale": prior_scale,
            "random_effect_scale": random_effect_scale,
            "slope_scale": slope_scale,
        },
        "practical_tolerance": practical_tolerance,
        "convergence_passed": convergence,
        "sampling_geometry": {
            "max_tree_depth_hits": max_depth_hits,
            "divergences": int(idata.sample_stats.diverging.sum()),
            "bfmi_by_chain": [
                float(value) if np.isfinite(value) else None for value in bfmi
            ],
        },
        "status": "convergence_passed_model_checks_still_require_review"
        if convergence
        else "diagnostic_only_not_reportable",
        "blocks": [
            {
                k: v
                for k, v in b.items()
                if k not in {"positions", "covariance", "cholesky"}
            }
            for b in data.blocks
        ],
    }
    (output / "model_specification.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    data.cells.assign(panel_weight=data.weights).to_csv(
        output / "model_cells.csv", index=False
    )
    np.savez_compressed(
        output / "model_inputs_INTERNAL.npz",
        observed=data.observed,
        bootstrap=data.bootstrap,
        design=data.design,
        grid_design=data.grid_design,
        **{f"covariance_{i}": b["covariance"] for i, b in enumerate(data.blocks)},
    )
    if require_convergence and not convergence:
        raise RuntimeError(
            f"Convergence gate failed; diagnostic artifacts retained in {output}"
        )
    return output
