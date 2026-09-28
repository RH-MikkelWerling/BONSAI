import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from opera.run.render_style_gallery import synthetic_calibration_cohort
from opera.visualization.censoring_aware_embeddings import (
    cause_specific_cif_from_hazards,
    horizon_ipcw_data,
    plot_censoring_aware_embedding_panel,
    spatial_calibration_residual,
    _kernel_surfaces,
)

HORIZON = 365.0
BANDWIDTH = 0.45
GRID_SIZE = 35
N_NEIGHBORS = 40
MIN_SUPPORT = 10.0


def _residual(seed, **knobs):
    """Panel D's estimate on one synthetic held-out cohort."""
    cohort = synthetic_calibration_cohort(seed, **knobs)
    data = horizon_ipcw_data(cohort["times"], cohort["events"], HORIZON)
    eligible = data.eligible
    residual = spatial_calibration_residual(
        cohort["coords"][eligible],
        cohort["predicted_cif"][eligible],
        data.target[eligible],
        data.weights[eligible],
        bandwidth=BANDWIDTH,
        grid_size=GRID_SIZE,
        embedding=cohort["embedding"][eligible],
        n_neighbors=N_NEIGHBORS,
        min_effective_support=MIN_SUPPORT,
        n_bootstrap=150,
        held_out=cohort["held_out"][eligible],
        random_state=0,
    )
    return cohort, residual


def _grid_points(coords, residual):
    """Grid coordinates matching a residual surface, for region-wise checks."""
    from opera.visualization.censoring_aware_embeddings import _display_grid

    xx, yy, _ = _display_grid(coords, GRID_SIZE)
    assert xx.shape == residual.residual.shape
    return xx, yy


def test_horizon_labels_distinguish_primary_competing_and_early_censoring():
    data = horizon_ipcw_data(
        np.array([10.0, 20.0, 30.0, 100.0]),
        np.array([1, 2, 0, 0]),
        60.0,
    )
    assert data.eligible.tolist() == [True, True, False, True]
    assert data.target.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert data.status.tolist() == [
        "primary event",
        "competing death",
        "ineligible/early censor",
        "known event-free",
    ]
    assert np.all(data.weights[data.eligible] > 0)


def test_censoring_aware_panel_smoke():
    rng = np.random.default_rng(3)
    n = 80
    coords = rng.normal(size=(n, 2))
    times = rng.uniform(5, 200, size=n)
    events = rng.choice([0, 1, 2], size=n, p=[0.35, 0.5, 0.15])
    fig, diagnostics = plot_censoring_aware_embedding_panel(
        coords,
        times,
        events,
        horizon=90,
        predicted_cif=rng.uniform(0, 1, n),
        embedding=rng.normal(size=(n, 5)),
        grid_size=15,
        min_effective_support=2,
        n_neighbors=10,
        residual_bootstrap=25,
    )
    assert len(fig.axes) >= 5
    assert diagnostics["n_total"] == n
    assert diagnostics["residual_units"] == "CIF"
    maps = {ax.get_label(): ax for ax in fig.axes}
    assert maps["calibration-map:C"].images[0].get_array().shape[0] >= 499
    assert maps["calibration-map:C"].images[0].get_cmap().name == "plasma"
    assert maps["calibration-map:E"].images[0].get_cmap().name == "plasma"
    assert maps["calibration-map:B"].collections[0].get_cmap().name == "plasma"
    assert maps["calibration-map:C"].images[0].get_clip_path() is not None


def test_ipcw_includes_censoring_before_a_between_knot_horizon():
    data = horizon_ipcw_data(
        np.array([10., 20., 100., 120.]), np.array([0, 1, 0, 0]), 60.
    )
    assert data.weights[1:] == pytest.approx([4 / 3, 4 / 3, 4 / 3])


def test_all_map_panels_use_the_selected_evaluation_patients():
    rng = np.random.default_rng(8)
    selected = np.arange(40) % 2 == 0
    fig, diagnostics = plot_censoring_aware_embedding_panel(
        rng.normal(size=(40, 2)), np.full(40, 100.), np.zeros(40),
        horizon=60., predicted_cif=np.full(40, .2), held_out=selected,
        embedding=rng.normal(size=(40, 5)), grid_size=8,
        min_effective_support=1, residual_bootstrap=0,
    )
    assert diagnostics["n_total"] == 20
    assert diagnostics["n_residual_patients"] == 20
    assert diagnostics["observed_surface_reuse_max_difference"] == 0.
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_panel_d_still_draws_without_predictions():
    rng = np.random.default_rng(5)
    n = 60
    fig, diagnostics = plot_censoring_aware_embedding_panel(
        rng.normal(size=(n, 2)),
        rng.uniform(5, 200, n),
        rng.choice([0, 1, 2], n, p=[0.35, 0.5, 0.15]),
        horizon=90,
        grid_size=12,
        min_effective_support=2,
    )
    assert len(fig.axes) >= 5
    assert "residual_units" not in diagnostics


# --- panel C is reused, not re-estimated ------------------------------------


def test_kernel_surfaces_is_the_plain_two_dimensional_ipcw_kernel():
    """Pin panels C and E: a direct Gaussian-weighted IPCW average."""
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(40, 2))
    target = rng.integers(0, 2, 40).astype(float)
    weights = rng.uniform(1, 2, 40)
    xx, yy, risk, support, density = _kernel_surfaces(
        coords, target, weights, bandwidth=0.5, grid_size=9
    )
    point = np.array([xx[4, 4], yy[4, 4]])
    kernel = np.exp(-0.5 * ((point - coords) ** 2).sum(1) / 0.25) * weights
    assert risk[4, 4] == pytest.approx((kernel @ target) / kernel.sum())
    assert support[4, 4] == pytest.approx(kernel.sum() ** 2 / (kernel**2).sum())
    assert density[4, 4] == pytest.approx(kernel.sum())


def test_panel_d_observed_surface_is_panel_c_surface():
    """With no separate embedding, R_obs must be panel C's array, not a rerun."""
    cohort = synthetic_calibration_cohort(2026, blob_shift=0.3)
    _, diagnostics = plot_censoring_aware_embedding_panel(
        cohort["coords"],
        cohort["times"],
        cohort["events"],
        horizon=HORIZON,
        predicted_cif=cohort["predicted_cif"],
        held_out=cohort["held_out"],
        bandwidth=BANDWIDTH,
        grid_size=GRID_SIZE,
        min_effective_support=MIN_SUPPORT,
        residual_bootstrap=25,
    )
    assert diagnostics["observed_surface_reuse_max_difference"] == 0.0


# --- the residual is a calibration check ------------------------------------


def test_residual_lights_up_only_in_the_miscalibrated_region():
    shift = -0.30  # predictions deflated inside the blob -> observed above them
    cohort, residual = _residual(2026, blob_shift=shift)
    xx, yy = _grid_points(cohort["coords"], residual)
    centre = np.array([2.2, 0.0])
    distance = np.hypot(xx - centre[0], yy - centre[1])
    z = np.ma.masked_where(residual.blank, residual.z)
    inside = z[(distance < 1.0) & ~residual.blank]
    outside = z[(distance > 3.0) & ~residual.blank]
    assert inside.size and outside.size
    assert inside.mean() > 4.0, inside.mean()
    assert abs(outside.mean()) < 1.5, outside.mean()
    assert inside.mean() - outside.mean() > 4.0


def test_residual_flips_sign_with_the_injected_miscalibration():
    centre = np.array([2.2, 0.0])
    means = {}
    for shift in (-0.30, 0.30):
        cohort, residual = _residual(2026, blob_shift=shift)
        xx, yy = _grid_points(cohort["coords"], residual)
        near = (np.hypot(xx - centre[0], yy - centre[1]) < 1.0) & ~residual.blank
        means[shift] = np.ma.masked_where(residual.blank, residual.z)[near].mean()
    assert means[-0.30] > 4.0 > -4.0 > means[0.30]


def test_residual_is_flat_when_predictions_equal_the_truth():
    """Perfect predictions: the z field has unit spread and no systematic sign."""
    grid_means, spreads = [], []
    for seed in range(6):
        _, residual = _residual(seed)
        z = residual.display.compressed()
        grid_means.append(z.mean())
        spreads.append(z.std())
    grid_means, spreads = np.array(grid_means), np.array(spreads)
    assert np.all(np.abs(grid_means) < 1.5), grid_means
    assert np.all(spreads < 1.6), spreads
    assert abs(grid_means.mean()) < 0.5, grid_means.mean()


def test_predicted_and_observed_surfaces_share_one_smoothing_pass():
    """Feeding the outcome in as the prediction must cancel to exactly zero."""
    cohort = synthetic_calibration_cohort(4)
    data = horizon_ipcw_data(cohort["times"], cohort["events"], HORIZON)
    eligible = data.eligible
    residual = spatial_calibration_residual(
        cohort["coords"][eligible],
        data.target[eligible],  # prediction == observed label
        data.target[eligible],
        data.weights[eligible],
        bandwidth=BANDWIDTH,
        grid_size=GRID_SIZE,
        embedding=cohort["embedding"][eligible],
        n_neighbors=N_NEIGHBORS,
        min_effective_support=MIN_SUPPORT,
        n_bootstrap=0,
        random_state=0,
    )
    assert np.nanmax(np.abs(residual.residual)) < 1e-12


# --- parameters and guards ---------------------------------------------------


def test_raw_units_toggle_reports_cumulative_incidence():
    cohort = synthetic_calibration_cohort(2026, blob_shift=-0.30)
    data = horizon_ipcw_data(cohort["times"], cohort["events"], HORIZON)
    eligible = data.eligible
    shared = dict(
        bandwidth=BANDWIDTH,
        grid_size=GRID_SIZE,
        embedding=cohort["embedding"][eligible],
        n_neighbors=N_NEIGHBORS,
        min_effective_support=MIN_SUPPORT,
        n_bootstrap=60,
        random_state=0,
    )
    args = (
        cohort["coords"][eligible],
        cohort["predicted_cif"][eligible],
        data.target[eligible],
        data.weights[eligible],
    )
    raw = spatial_calibration_residual(*args, standardize=False, **shared)
    standardized = spatial_calibration_residual(*args, standardize=True, **shared)
    assert raw.label.endswith("(CIF)")
    assert standardized.label.endswith("(local z)")
    assert np.array_equal(raw.residual, standardized.residual, equal_nan=True)
    assert np.nanmax(np.abs(raw.display)) < 1.0  # a CIF difference, not a z
    assert np.nanmax(np.abs(standardized.display)) > 3.0


def test_held_out_flag_restricts_the_patients_used():
    cohort = synthetic_calibration_cohort(2026)
    data = horizon_ipcw_data(cohort["times"], cohort["events"], HORIZON)
    eligible = data.eligible
    held_out = np.zeros(int(eligible.sum()), dtype=bool)
    held_out[::2] = True
    residual = spatial_calibration_residual(
        cohort["coords"][eligible],
        cohort["predicted_cif"][eligible],
        data.target[eligible],
        data.weights[eligible],
        bandwidth=BANDWIDTH,
        grid_size=GRID_SIZE,
        min_effective_support=MIN_SUPPORT,
        n_bootstrap=0,
        held_out=held_out,
    )
    assert residual.n_patients == int(held_out.sum())


def test_predictions_must_be_a_cumulative_incidence():
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(30, 2))
    with pytest.raises(ValueError, match="cumulative incidence"):
        spatial_calibration_residual(
            coords,
            rng.uniform(1.0, 4.0, 30),  # a hazard rate, not a CIF
            rng.integers(0, 2, 30).astype(float),
            np.ones(30),
            bandwidth=0.5,
            grid_size=9,
            n_bootstrap=0,
        )


def test_cause_specific_cif_from_hazards_matches_the_discrete_definition():
    cause = np.array([[0.1, 0.2, 0.3]])
    overall = np.array([[0.3, 0.4, 0.5]])
    ends = np.array([100.0, 365.0, 700.0])
    expected = 1.0 * 0.1 + (1 - 0.3) * 0.2
    got = cause_specific_cif_from_hazards(
        cause, overall_hazard=overall, interval_ends=ends, horizon=365.0
    )
    assert got == pytest.approx([expected])
    # Without a competing cause the all-cause hazard is the cause-specific one.
    solo = cause_specific_cif_from_hazards(cause, interval_ends=ends, horizon=365.0)
    assert solo == pytest.approx([0.1 + 0.9 * 0.2])
    assert np.all(
        cause_specific_cif_from_hazards(cause, interval_ends=ends, horizon=1e9) <= 1.0
    )


def test_synthetic_cohort_predictions_are_calibrated_by_construction():
    cohort = synthetic_calibration_cohort(2026)
    assert np.array_equal(cohort["predicted_cif"], cohort["true_risk"])
    assert cohort["held_out"].all()
    shifted = synthetic_calibration_cohort(2026, blob_shift=0.25)
    moved = shifted["predicted_cif"] != shifted["true_risk"]
    assert 0.1 < moved.mean() < 0.5  # one region of the embedding, not all of it
    assert np.all(shifted["blob"][moved] > 0.01)
