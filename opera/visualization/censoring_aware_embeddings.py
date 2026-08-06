"""Censoring-aware diagnostics on a fixed two-dimensional embedding map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonData:
    eligible: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    status: np.ndarray


def _censor_survival(times: np.ndarray, events: np.ndarray):
    order = np.argsort(times)
    unique = np.unique(times[order])
    survival = 1.0
    values = []
    for time in unique:
        at_risk = np.sum(times >= time)
        censored = np.sum((times == time) & (events == 0))
        values.append((time, survival))  # left-continuous G(t-)
        if at_risk:
            survival *= max(0.0, 1.0 - censored / at_risk)
    knots = np.asarray([item[0] for item in values], dtype=float)
    left = np.asarray([item[1] for item in values], dtype=float)

    def evaluate(query):
        query = np.asarray(query, dtype=float)
        indices = np.searchsorted(knots, query, side="right") - 1
        return np.where(indices >= 0, left[np.maximum(indices, 0)], 1.0)

    return evaluate


def horizon_ipcw_data(
    times: np.ndarray,
    events: np.ndarray,
    horizon: float,
    *,
    min_censor_survival: float = 0.05,
) -> HorizonData:
    """Construct cumulative-incidence labels and IPCW weights at ``horizon``."""
    times = np.asarray(times, dtype=float)
    raw_events = np.asarray(events, dtype=float)
    valid = np.isfinite(times) & (times >= 0) & np.isin(raw_events, [0, 1, 2])
    events = np.where(np.isfinite(raw_events), raw_events, -1).astype(int)
    evaluate_g = _censor_survival(times[valid], events[valid])
    primary = valid & (events == 1) & (times <= horizon)
    competing = valid & (events == 2) & (times <= horizon)
    observed_through_horizon = valid & (times > horizon)
    eligible = primary | competing | observed_through_horizon
    target = primary.astype(float)
    evaluation_time = np.minimum(times, horizon)
    weights = np.zeros_like(times, dtype=float)
    weights[eligible] = 1.0 / np.clip(
        evaluate_g(evaluation_time[eligible]), min_censor_survival, None
    )
    status = np.full(times.shape, "ineligible/early censor", dtype=object)
    status[observed_through_horizon] = "known event-free"
    status[competing] = "competing death"
    status[primary] = "primary event"
    return HorizonData(eligible, target, weights, status)


def _default_bandwidth(coords: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors

    k = min(50, max(2, len(coords) - 1))
    distances = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors()[0][:, -1]
    positive = distances[distances > 0]
    return float(np.median(positive)) if positive.size else 1.0


def _kernel_surfaces(
    coords: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    bandwidth: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    padding = 0.04 * np.maximum(np.ptp(coords, axis=0), 1e-6)
    xs = np.linspace(coords[:, 0].min() - padding[0], coords[:, 0].max() + padding[0], grid_size)
    ys = np.linspace(coords[:, 1].min() - padding[1], coords[:, 1].max() + padding[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    numerator = np.zeros(len(grid))
    denominator = np.zeros(len(grid))
    squared = np.zeros(len(grid))
    scale = max(float(bandwidth), 1e-8)
    for start in range(0, len(grid), 512):
        stop = min(start + 512, len(grid))
        distance2 = ((grid[start:stop, None] - coords[None]) ** 2).sum(axis=2)
        kernel = np.exp(-0.5 * distance2 / (scale * scale)) * weights[None]
        denominator[start:stop] = kernel.sum(axis=1)
        numerator[start:stop] = kernel @ values
        squared[start:stop] = (kernel * kernel).sum(axis=1)
    surface = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    effective_support = np.divide(
        denominator**2, squared, out=np.zeros_like(denominator), where=squared > 0
    )
    return xx, yy, surface.reshape(xx.shape), effective_support.reshape(xx.shape), denominator.reshape(xx.shape)


def _cross_fitted_nuisance(
    covariates: pd.DataFrame | None,
    targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if covariates is None or covariates.shape[1] == 0 or np.unique(targets).size < 2:
        return np.full(len(targets), np.average(targets, weights=weights))
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = list(covariates.select_dtypes(include=[np.number]).columns)
    categorical = [column for column in covariates if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", make_pipeline(SimpleImputer(), StandardScaler()), numeric))
    if categorical:
        transformers.append(("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical))
    predictions = np.zeros(len(targets), dtype=float)
    minority = int(np.bincount(targets.astype(int)).min())
    folds = min(5, minority)
    if folds < 2:
        return np.full(len(targets), np.average(targets, weights=weights))
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=17).split(covariates, targets):
        model = make_pipeline(
            ColumnTransformer(transformers),
            LogisticRegression(max_iter=2000, solver="lbfgs"),
        )
        model.fit(covariates.iloc[train], targets[train], logisticregression__sample_weight=weights[train])
        predictions[test] = model.predict_proba(covariates.iloc[test])[:, 1]
    return predictions


def plot_censoring_aware_embedding_panel(
    coords: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    *,
    horizon: float,
    covariates: pd.DataFrame | None = None,
    covariate_names: Sequence[str] | None = None,
    bandwidth: float | None = None,
    grid_size: int = 80,
    min_effective_support: float = 25.0,
    title: str = "Outcome",
):
    """Plot status, time, IPCW risk, residual enrichment, and support panels."""
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    data = horizon_ipcw_data(times, events, horizon)
    eligible = data.eligible
    local_coords = coords[eligible]
    local_target = data.target[eligible]
    local_weights = data.weights[eligible]
    if bandwidth is None:
        bandwidth = _default_bandwidth(local_coords)
    local_covariates = None
    if covariates is not None:
        columns = list(covariate_names or covariates.columns)
        local_covariates = covariates.loc[eligible, columns].reset_index(drop=True)
    expected = _cross_fitted_nuisance(local_covariates, local_target, local_weights)
    xx, yy, risk, support, density = _kernel_surfaces(
        local_coords, local_target, local_weights,
        bandwidth=bandwidth, grid_size=grid_size,
    )
    _, _, enrichment, _, _ = _kernel_surfaces(
        local_coords, local_target - expected, local_weights,
        bandwidth=bandwidth, grid_size=grid_size,
    )
    mask = support < min_effective_support
    risk = np.ma.masked_where(mask, risk)
    enrichment = np.ma.masked_where(mask, enrichment)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    colors = {
        "primary event": "#b2182b", "competing death": "#2166ac",
        "known event-free": "#4d4d4d", "ineligible/early censor": "#bdbdbd",
    }
    for status, color in colors.items():
        selected = data.status == status
        axes[0].scatter(coords[selected, 0], coords[selected, 1], s=3, alpha=0.45, c=color, label=f"{status} (n={selected.sum():,})")
    axes[0].legend(markerscale=3, fontsize=7)
    axes[0].set_title("Event/status at horizon")
    markers = {0: "o", 1: "^", 2: "X"}
    labels = {0: "administrative censor", 1: "primary event", 2: "competing death"}
    for event, marker in markers.items():
        selected = np.isfinite(times) & (events == event)
        scatter = axes[1].scatter(coords[selected, 0], coords[selected, 1], c=np.minimum(times[selected], horizon), cmap="viridis", s=3, alpha=0.5, marker=marker, label=labels[event])
    fig.colorbar(scatter, ax=axes[1], label="Time to status (days; clipped)")
    axes[1].legend(markerscale=3, fontsize=7)
    axes[1].set_title("Time to observed status")
    risk_plot = axes[2].pcolormesh(xx, yy, risk, shading="auto", cmap="magma", vmin=0, vmax=1)
    fig.colorbar(risk_plot, ax=axes[2], label="IPCW local cumulative incidence")
    axes[2].set_title("IPCW kernel risk")
    bound = np.nanpercentile(np.abs(enrichment.compressed()), 98) if enrichment.count() else 1.0
    enrichment_plot = axes[3].pcolormesh(xx, yy, enrichment, shading="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    fig.colorbar(enrichment_plot, ax=axes[3], label="Observed − nuisance expected")
    axes[3].set_title("Nuisance-adjusted enrichment")
    support_plot = axes[4].pcolormesh(xx, yy, support, shading="auto", cmap="cividis")
    fig.colorbar(support_plot, ax=axes[4], label="Effective local sample size")
    axes[4].contour(xx, yy, support, levels=[min_effective_support], colors="white", linewidths=1)
    axes[4].set_title("Effective support")
    for axis in axes:
        axis.set_xlabel("Embedding 1")
        axis.set_ylabel("Embedding 2")
    fig.suptitle(f"{title} at {horizon:g} days · bandwidth={bandwidth:.3g}")
    diagnostics = {
        "n_total": int(len(coords)), "n_eligible": int(eligible.sum()),
        "n_primary": int(local_target.sum()), "bandwidth": float(bandwidth),
        "min_effective_support": float(min_effective_support),
        "max_kernel_density": float(np.nanmax(density)),
    }
    return fig, diagnostics
