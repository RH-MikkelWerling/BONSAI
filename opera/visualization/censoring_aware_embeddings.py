"""Censoring-aware diagnostics on a fixed two-dimensional embedding map."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from opera.visualization.style import (
    clean_2d_axes,
    PALETTE,
)


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
        # G(query-): include censoring at earlier knots, not at query itself.
        indices = np.searchsorted(knots, query, side="left")
        if not len(left):
            return np.ones_like(query)
        extended = np.append(left, survival)
        return extended[indices]

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

    if len(coords) < 2:
        return 1.0
    k = min(50, len(coords) - 1)
    distances = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors()[0][:, -1]
    positive = distances[distances > 0]
    return float(np.median(positive)) if positive.size else 1.0


def _display_grid(coords: np.ndarray, grid_size: int):
    padding = 0.04 * np.maximum(np.ptp(coords, axis=0), 1e-6)
    xs = np.linspace(
        coords[:, 0].min() - padding[0], coords[:, 0].max() + padding[0], grid_size
    )
    ys = np.linspace(
        coords[:, 1].min() - padding[1], coords[:, 1].max() + padding[1], grid_size
    )
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def neighbor_operator(embedding: np.ndarray, n_neighbors: int):
    """Row-stochastic kNN kernel built in the *full* embedding.

    The two-dimensional layout only decides where a grid point sits among
    patients; this operator decides which patients are its neighbours, in the
    space the model actually uses. Passing ``None`` downstream reproduces the
    plain two-dimensional kernel exactly.
    """
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    embedding = np.asarray(embedding, dtype=float)
    size = len(embedding)
    k = int(min(max(int(n_neighbors), 1), size))
    fitted = NearestNeighbors(n_neighbors=k).fit(embedding)
    distances, indices = fitted.kneighbors(embedding)  # includes the patient
    furthest = distances[:, -1]
    positive = furthest[furthest > 0]
    scale = max(float(np.median(positive)) if positive.size else 1.0, 1e-8)
    kernel = np.exp(-0.5 * (distances / scale) ** 2)
    kernel /= np.clip(kernel.sum(axis=1, keepdims=True), 1e-12, None)
    rows = np.repeat(np.arange(size), k)
    return csr_matrix((kernel.ravel(), (rows, indices.ravel())), shape=(size, size))


def _kernel_block(
    block: np.ndarray, coords: np.ndarray, scale: float, operator=None
) -> np.ndarray:
    """Grid-point x patient kernel weights, optionally routed through kNN."""
    distance2 = ((block[:, None] - coords[None]) ** 2).sum(axis=2)
    kernel = np.exp(-0.5 * distance2 / (scale * scale))
    if operator is not None:
        kernel = operator.T.dot(kernel.T).T
    return kernel


def _kernel_surfaces(
    coords: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    bandwidth: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, grid = _display_grid(coords, grid_size)
    numerator = np.zeros(len(grid))
    denominator = np.zeros(len(grid))
    squared = np.zeros(len(grid))
    scale = max(float(bandwidth), 1e-8)
    for start in range(0, len(grid), 512):
        stop = min(start + 512, len(grid))
        kernel = _kernel_block(grid[start:stop], coords, scale) * weights[None]
        denominator[start:stop] = kernel.sum(axis=1)
        numerator[start:stop] = kernel @ values
        squared[start:stop] = (kernel * kernel).sum(axis=1)
    surface = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    effective_support = np.divide(
        denominator**2, squared, out=np.zeros_like(denominator), where=squared > 0
    )
    return (
        xx,
        yy,
        surface.reshape(xx.shape),
        effective_support.reshape(xx.shape),
        denominator.reshape(xx.shape),
    )


def cause_specific_cif_from_hazards(
    cause_hazard: np.ndarray,
    *,
    interval_ends: np.ndarray,
    horizon: float,
    overall_hazard: np.ndarray | None = None,
) -> np.ndarray:
    """Convert discrete-time hazards into a cause-specific CIF at ``horizon``.

    Panel D compares like with like: the observed surface estimates the
    cause-specific cumulative incidence at the horizon, so a head that emits a
    hazard sequence or an all-cause survival curve has to pass through here
    before its output can be averaged. ``cause_hazard`` is
    ``(n_patients, n_intervals)`` on the grid ``interval_ends``;
    ``overall_hazard`` is the all-cause hazard and defaults to the
    cause-specific one when no competing risk is modelled.
    """
    cause = np.clip(np.atleast_2d(np.asarray(cause_hazard, dtype=float)), 0.0, 1.0)
    overall = (
        cause
        if overall_hazard is None
        else np.clip(np.atleast_2d(np.asarray(overall_hazard, dtype=float)), 0.0, 1.0)
    )
    if overall.shape != cause.shape:
        raise ValueError("cause_hazard and overall_hazard must have the same shape.")
    interval_ends = np.asarray(interval_ends, dtype=float)
    if interval_ends.shape[0] != cause.shape[1]:
        raise ValueError("interval_ends must have one entry per hazard interval.")
    survival = np.cumprod(1.0 - overall, axis=1)
    entering = np.concatenate([np.ones((len(cause), 1)), survival[:, :-1]], axis=1)
    within = interval_ends <= float(horizon)
    return (entering * cause)[:, within].sum(axis=1)


@dataclass(frozen=True)
class CalibrationResidual:
    """Spatial calibration residual D(x) = R_obs(x) - R_pred(x)."""

    observed: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    standard_error: np.ndarray
    z: np.ndarray
    support: np.ndarray
    blank: np.ndarray
    display: np.ma.MaskedArray
    label: str
    n_patients: int
    neighborhood: str


def spatial_calibration_residual(
    coords: np.ndarray,
    predicted_cif: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    bandwidth: float,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    grid_size: int = 80,
    embedding: np.ndarray | None = None,
    n_neighbors: int = 50,
    support_mask: np.ndarray | None = None,
    min_effective_support: float = 25.0,
    standardize: bool = True,
    n_bootstrap: int = 200,
    held_out: np.ndarray | None = None,
    random_state: int = 0,
) -> CalibrationResidual:
    """Local calibration of a risk model over its own embedding.

    For every grid point ``x`` of the display layout this smooths two
    per-patient quantities through *one* kernel pass, so the two surfaces
    differ only in what is averaged:

    ``R_obs(x)``
        the IPCW cause-specific cumulative incidence at the horizon, i.e. the
        surface panel C shows;
    ``R_pred(x)``
        the model's predicted cause-specific cumulative incidence at the same
        horizon, which must already be a CIF (see
        :func:`cause_specific_cif_from_hazards`).

    ``D(x) = R_obs(x) - R_pred(x)`` is positive where the model under-predicts
    risk in that region of its embedding and negative where it over-predicts.
    ``predicted_cif`` carries no sampling noise relative to the outcome labels,
    so the standard error of ``D`` is the standard error of the local IPCW
    estimate, taken here by a patient-level bootstrap that reuses the cached
    kernel.

    Neighbourhoods come from ``embedding`` (the full representation) when it is
    supplied; the two-dimensional ``coords`` then only place the answer for
    display. Where a grid point may be shown is still decided in two dimensions
    by ``support_mask`` -- panel E's effective-support mask -- because a kNN
    kernel is spread across its neighbours everywhere, including regions of the
    layout that hold no patients at all. With no ``embedding`` the pass reduces
    to panel C's own kernel and ``observed`` is panel C's surface exactly.

    ``held_out`` restricts the estimate to patients the model never saw, which
    is what makes this a calibration check rather than a re-reading of the
    training fit.
    """
    coords = np.asarray(coords, dtype=float)
    predicted_cif = np.asarray(predicted_cif, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(coords) == len(predicted_cif) == len(target) == len(weights):
        raise ValueError("coords, predicted_cif, target and weights must align.")
    finite = np.isfinite(predicted_cif)
    if finite.any() and (
        predicted_cif[finite].min() < -1e-9 or predicted_cif[finite].max() > 1 + 1e-9
    ):
        raise ValueError(
            "predicted_cif must be a cumulative incidence in [0, 1]; convert hazards "
            "or survival curves with cause_specific_cif_from_hazards() first."
        )
    if grid is None:
        xx, yy, points = _display_grid(coords, grid_size)
    else:
        xx, yy = grid
        points = np.column_stack([xx.ravel(), yy.ravel()])
    used = finite & np.isfinite(target) & np.isfinite(weights) & (weights > 0)
    if held_out is not None:
        used &= np.asarray(held_out, dtype=bool)
    if not used.any():
        raise ValueError("No patients remain for the calibration residual.")
    local_coords = coords[used]
    local_predicted = predicted_cif[used]
    local_target = target[used]
    local_weights = weights[used]
    operator = None
    neighborhood = "2D layout"
    if embedding is not None:
        operator = neighbor_operator(
            np.asarray(embedding, dtype=float)[used], n_neighbors
        )
        neighborhood = f"full embedding (k={min(int(n_neighbors), int(used.sum()))})"

    size = len(points)
    observed = np.full(size, np.nan)
    predicted = np.full(size, np.nan)
    support = np.zeros(size)
    standard_error = np.full(size, np.nan)
    draws = None
    if n_bootstrap and n_bootstrap > 0:
        generator = np.random.default_rng(random_state)
        share = np.full(len(local_coords), 1.0 / len(local_coords))
        draws = generator.multinomial(
            len(local_coords), share, size=int(n_bootstrap)
        ).T.astype(float)
    scale = max(float(bandwidth), 1e-8)
    for start in range(0, size, 512):
        stop = min(start + 512, size)
        kernel = (
            _kernel_block(points[start:stop], local_coords, scale, operator)
            * local_weights[None]
        )
        denominator = kernel.sum(axis=1)
        squared = (kernel * kernel).sum(axis=1)
        support[start:stop] = np.divide(
            denominator**2, squared, out=np.zeros_like(denominator), where=squared > 0
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            observed[start:stop] = np.where(
                denominator > 0, (kernel @ local_target) / denominator, np.nan
            )
            predicted[start:stop] = np.where(
                denominator > 0, (kernel @ local_predicted) / denominator, np.nan
            )
            if draws is not None:
                resampled_total = kernel @ draws
                # Bootstrap the paired observed-minus-predicted residual,
                # rather than only the outcome surface. Model and G stay fixed.
                resampled_event = (kernel * (local_target - local_predicted)[None]) @ draws
                replicates = np.where(
                    resampled_total > 0, resampled_event / resampled_total, np.nan
                )
                standard_error[start:stop] = np.nanstd(replicates, axis=1, ddof=1)

    shape = xx.shape
    observed = observed.reshape(shape)
    predicted = predicted.reshape(shape)
    support = support.reshape(shape)
    standard_error = standard_error.reshape(shape)
    residual = observed - predicted
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(standard_error > 0, residual / standard_error, np.nan)
    blank = ~np.isfinite(residual)
    if operator is None:
        blank = blank | (support < min_effective_support)
    if support_mask is not None:
        blank = blank | np.asarray(support_mask, dtype=bool)
    if standardize:
        blank = blank | ~np.isfinite(z)
        display = np.ma.masked_where(blank, z)
        label = "Observed − predicted (local z)"
    else:
        display = np.ma.masked_where(blank, residual)
        label = "Observed − predicted (CIF)"
    return CalibrationResidual(
        observed=observed,
        predicted=predicted,
        residual=residual,
        standard_error=standard_error,
        z=z,
        support=support,
        blank=blank,
        display=display,
        label=label,
        n_patients=int(used.sum()),
        neighborhood=neighborhood,
    )


def plot_censoring_aware_embedding_panel(
    coords: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    *,
    horizon: float,
    predicted_cif: np.ndarray | None = None,
    embedding: np.ndarray | None = None,
    held_out: np.ndarray | None = None,
    bandwidth: float | None = None,
    grid_size: int = 80,
    min_effective_support: float = 25.0,
    min_density_fraction: float = 0.01,
    n_neighbors: int = 50,
    residual_standardize: bool = False,
    residual_use_full_embedding: bool = False,
    residual_bootstrap: int = 200,
    residual_z_limit: float = 4.0,
    random_state: int = 0,
    title: str = "Outcome",
):
    """Plot status, time, IPCW risk, calibration residual, and support panels.

    Panel D needs ``predicted_cif``: the model's predicted cause-specific
    cumulative incidence at ``horizon``, one value per patient, from a model
    that did not train on them. ``held_out`` restricts ALL panels to the same
    evaluation patients. By default C and D use the same 2D kernel and D is
    observed minus predicted in CIF units. Full-embedding residual neighborhoods
    are an explicit alternative via ``residual_use_full_embedding``; then D's
    observed surface is not C. Standardized residuals are exploratory, not
    simultaneous significance tests: the bootstrap does not refit model or G.
    Support is Kish ESS, not a count of distinct neighbors; a separate relative
    kernel-mass threshold prevents displaying extrapolated Gaussian tails.
    """
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    if not 0 <= min_density_fraction < 1:
        raise ValueError("min_density_fraction must be in [0, 1).")
    if held_out is not None:
        selected = np.asarray(held_out, dtype=bool)
        coords, times, events = coords[selected], times[selected], events[selected]
        if predicted_cif is not None:
            predicted_cif = np.asarray(predicted_cif)[selected]
        if embedding is not None:
            embedding = np.asarray(embedding)[selected]
        held_out = None
    data = horizon_ipcw_data(times, events, horizon)
    eligible = data.eligible
    local_coords = coords[eligible]
    if not len(local_coords):
        raise ValueError("No eligible patients at this horizon.")
    local_target = data.target[eligible]
    local_weights = data.weights[eligible]
    if bandwidth is None:
        bandwidth = _default_bandwidth(local_coords)
    local_embedding = (
        None if not residual_use_full_embedding or embedding is None
        else np.asarray(embedding, dtype=float)[eligible]
    )
    xx, yy, risk, support, density = _kernel_surfaces(
        local_coords,
        local_target,
        local_weights,
        bandwidth=bandwidth,
        grid_size=grid_size,
    )
    mask = (support < min_effective_support) | (
        density < min_density_fraction * np.max(density)
    )
    density_margin = density - min_density_fraction * np.max(density)
    risk_margin = np.minimum(
        support / max(min_effective_support, 1e-12) - 1,
        density_margin / max(float(np.max(density)), 1e-12),
    )
    risk = np.ma.masked_where(mask, risk)
    local_held_out = (
        None if held_out is None else np.asarray(held_out, dtype=bool)[eligible]
    )
    residual = None
    if predicted_cif is not None:
        residual = spatial_calibration_residual(
            local_coords,
            np.asarray(predicted_cif, dtype=float)[eligible],
            local_target,
            local_weights,
            bandwidth=bandwidth,
            grid=(xx, yy),
            embedding=local_embedding,
            n_neighbors=n_neighbors,
            support_mask=mask,
            min_effective_support=min_effective_support,
            standardize=residual_standardize,
            n_bootstrap=residual_bootstrap,
            held_out=local_held_out,
            random_state=random_state,
        )

    # Reserve a separate, identical colorbar row for every map. Colorbars must
    # not shrink some projections while leaving others at a different size.
    fig = plt.figure(figsize=(10.5, 7.5))
    layout = fig.add_gridspec(
        2, 3, left=0.055, right=0.97, top=0.89, bottom=0.09, wspace=0.22, hspace=0.42
    )
    axes, color_axes = [], []
    for index in range(5):
        cell = layout[index // 3, index % 3].subgridspec(
            2, 1, height_ratios=[1, 0.055], hspace=0.10
        )
        axes.append(fig.add_subplot(cell[0]))
        color_axes.append(fig.add_subplot(cell[1]))
    color_axes[0].set_axis_off()
    legend_axis = fig.add_subplot(layout[1, 2])
    legend_axis.set_axis_off()

    def surface(axis, values, margin, *, cmap, vmin=None, vmax=None):
        """Interpolate for display only; retain the numerical support boundary.

        Interpolating the continuous support criterion rather than a boolean
        pixel mask removes coarse stair steps without expanding eligibility.
        """
        from scipy.ndimage import zoom
        from matplotlib.path import Path

        factor = max(1.0, 500 / max(xx.shape))
        raw = np.asarray(np.ma.getdata(values), dtype=float)
        valid = np.isfinite(raw)
        field = zoom(np.where(valid, raw, 0), factor, order=1)
        display_margin = zoom(margin, factor, order=1)
        if not valid.all():
            display_margin = np.where(
                zoom(valid.astype(float), factor, order=1) >= 1 - 1e-8,
                display_margin, -1.,
            )
        image = axis.imshow(
            field, origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            interpolation="bilinear", cmap=plt.get_cmap(cmap).with_extremes(bad="white"),
            vmin=vmin, vmax=vmax, aspect="equal", rasterized=True,
        )
        # A geometric contour clips the continuous image, avoiding the stair
        # steps of a boolean raster mask. Compound paths preserve islands/holes.
        if np.max(display_margin) >= 0:
            outline = axis.contourf(
                np.linspace(xx.min(), xx.max(), field.shape[1]),
                np.linspace(yy.min(), yy.max(), field.shape[0]),
                display_margin, levels=[0, max(float(display_margin.max()), 1e-12) + 1],
                colors="none",
            )
            paths = outline.get_paths()
            image.set_clip_path(Path.make_compound_path(*paths), axis.transData)
            outline.remove()
        else:
            image.set_visible(False)
        return image
    colors = {
        "primary event": PALETTE["positive"],
        "competing death": "#482878",
        "known event-free": "#168577",
        "ineligible/early censor": "#CDD3DC",
    }
    status_markers = {
        "primary event": "^", "competing death": "X",
        "known event-free": "o", "ineligible/early censor": "+",
    }
    for status in reversed(colors):
        color = colors[status]
        selected = data.status == status
        axes[0].scatter(
            coords[selected, 0],
            coords[selected, 1],
            s=7,
            alpha=0.68,
            linewidths=0.5 if status == "ineligible/early censor" else 0,
            marker=status_markers[status],
            rasterized=True,
            c=color,
            label=f"{status} (n={selected.sum():,})",
        )
    axes[0].set_title("Event/status at horizon")
    markers = {0: "o", 1: "^", 2: "X"}
    labels = {0: "administrative censor", 1: "primary event", 2: "competing death"}
    for event, marker in markers.items():
        selected = np.isfinite(times) & (events == event)
        scatter = axes[1].scatter(
            coords[selected, 0],
            coords[selected, 1],
            c=np.minimum(times[selected], horizon),
            cmap="plasma",
            norm=Normalize(0, horizon),
            s=7,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
            marker=marker,
            label=labels[event],
        )
    fig.colorbar(
        scatter,
        cax=color_axes[1],
        orientation="horizontal",
        label="Time to status (days; clipped)",
    )
    axes[1].set_title("Time to observed status")
    risk_plot = surface(
        axes[2], risk, risk_margin, cmap="plasma", vmin=0, vmax=1
    )
    fig.colorbar(
        risk_plot,
        cax=color_axes[2],
        orientation="horizontal",
        label="IPCW local cumulative incidence",
    )
    axes[2].set_title("IPCW kernel risk")
    if residual is not None:
        visible = residual.display.compressed()
        peak = float(np.max(np.abs(visible))) if visible.size else 0.0
        bound = float(residual_z_limit) if residual_standardize else max(peak, 1e-6)
        residual_plot = surface(
            axes[3],
            residual.display,
            np.minimum(risk_margin, residual.support / max(min_effective_support, 1e-12) - 1)
            if local_embedding is None else risk_margin,
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )
        fig.colorbar(
            residual_plot,
            cax=color_axes[3],
            orientation="horizontal",
            label=residual.label,
            extend="both" if peak > bound else "neither",
        )
    else:
        color_axes[3].set_axis_off()
        axes[3].text(
            0.5,
            0.5,
            "No predicted CIF supplied",
            transform=axes[3].transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=PALETTE["ink_secondary"],
        )
    axes[3].set_title("Calibration residual")
    support_plot = surface(
        axes[4], support, density_margin, cmap="plasma", vmin=0,
        vmax=float(np.max(support[density_margin >= 0])) if np.any(density_margin >= 0) else 1,
    )
    fig.colorbar(
        support_plot,
        cax=color_axes[4],
        orientation="horizontal",
        label="Effective local sample size",
    )
    axes[4].contour(
        xx, yy, support, levels=[min_effective_support], colors="white", linewidths=1
    )
    axes[4].set_title("Effective support")
    for letter, axis in zip("ABCDE", axes):
        axis.set_label(f"calibration-map:{letter}")
        clean_2d_axes(axis)
        axis.set(xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()), aspect="equal")
        panel_title = axis.get_title(loc="left")
        axis.set_title(f"{letter}   {panel_title}", loc="left", fontsize=10, pad=12)
    for index in (1, 2, 3, 4):
        color_axes[index].tick_params(labelsize=8, length=2.5, pad=3)
        color_axes[index].xaxis.label.set_size(8)
        color_axes[index].grid(False)
    if residual is not None:
        color_axes[3].text(
            0,
            1.55,
            "Over-prediction",
            transform=color_axes[3].transAxes,
            fontsize=7.5,
            ha="left",
            va="bottom",
            color="#336B8A",
        )
        color_axes[3].text(
            1,
            1.55,
            "Under-prediction",
            transform=color_axes[3].transAxes,
            fontsize=7.5,
            ha="right",
            va="bottom",
            color="#A14D58",
        )

    # An opaque legend key is easier to decode than translucent point-cloud
    # handles. Status colors and observation shapes have separate sections.
    def rail_text(x, y, text, **kwargs):
        legend_axis.text(x, y, text, transform=legend_axis.transAxes, **kwargs)

    rail_text(0, 1.02, "A   Status at horizon", fontsize=9, fontweight="semibold")
    rail_text(1, 1.02, "n", fontsize=8, ha="right", color=PALETTE["ink_muted"])
    for index, (status, color) in enumerate(colors.items()):
        y = 0.90 - index * 0.105
        legend_axis.plot(
            0.025,
            y,
            marker=status_markers[status],
            markersize=5,
            color=color,
            linestyle="none",
            transform=legend_axis.transAxes,
        )
        rail_text(0.10, y, status.capitalize(), fontsize=8, va="center")
        rail_text(
            1,
            y,
            f"{np.sum(data.status == status):,}",
            fontsize=8,
            ha="right",
            va="center",
            color=PALETTE["ink_secondary"],
        )
    legend_axis.plot(
        [0, 1],
        [0.51, 0.51],
        transform=legend_axis.transAxes,
        color=PALETTE["panel_border"],
        lw=0.6,
    )
    rail_text(0, 0.43, "B   Observed status", fontsize=9, fontweight="semibold")
    for index, (event, marker) in enumerate(markers.items()):
        y = 0.31 - index * 0.105
        legend_axis.plot(
            0.025,
            y,
            marker=marker,
            markersize=5,
            color=PALETTE["ink_secondary"],
            linestyle="none",
            transform=legend_axis.transAxes,
        )
        rail_text(0.10, y, labels[event].capitalize(), fontsize=8, va="center")
    import textwrap

    fig.text(
        0.055,
        0.965,
        textwrap.fill(f"{title} ({horizon:g} days)", width=80, break_long_words=False),
        fontsize=13,
        fontweight="semibold",
        va="top",
    )
    diagnostics = {
        "n_total": int(len(coords)),
        "n_eligible": int(eligible.sum()),
        "n_primary": int(local_target.sum()),
        "bandwidth": float(bandwidth),
        "min_effective_support": float(min_effective_support),
        "min_density_fraction": float(min_density_fraction),
        "support_definition": "Kish ESS: (sum kernel*IPCW)^2 / sum (kernel*IPCW)^2",
        "censoring_assumption": "Marginal independent censoring; G clipped at 0.05",
        "max_kernel_density": float(np.nanmax(density)),
        "n_neighbors": int(n_neighbors),
        "residual_z_limit": float(residual_z_limit),
    }
    if residual is not None:
        diagnostics.update(
            {
                "n_residual_patients": residual.n_patients,
                "residual_units": "local z" if residual_standardize else "CIF",
                "residual_bootstrap": int(residual_bootstrap),
                "residual_neighborhood": residual.neighborhood,
                "residual_max_abs": float(np.max(np.abs(visible)))
                if visible.size
                else 0.0,
                "residual_mean": float(np.mean(visible)) if visible.size else 0.0,
                "residual_cells": int(visible.size),
                # R_obs in panel D is panel C's surface, not a second estimator:
                # with every eligible patient admissible the two must coincide.
                "residual_sd": float(np.std(visible)) if visible.size else 0.0,
                "residual_fraction_beyond_2": (
                    float(np.mean(np.abs(visible) > 2))
                    if residual_standardize and visible.size
                    else None
                ),
                # With no separate embedding, D's observed surface is panel C's
                # surface rather than a second estimator; this proves it.
                "observed_surface_reuse_max_difference": (
                    float(
                        np.nanmax(
                            np.abs(residual.observed - np.ma.getdata(risk))[~mask]
                        )
                    )
                    if local_embedding is None
                    and (local_held_out is None or bool(local_held_out.all()))
                    and np.isfinite(np.asarray(predicted_cif, dtype=float)).all()
                    and (~mask).any()
                    else None
                ),
            }
        )
    return fig, diagnostics
