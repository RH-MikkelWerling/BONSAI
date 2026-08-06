import numpy as np
import pandas as pd

from opera.visualization.censoring_aware_embeddings import (
    horizon_ipcw_data,
    plot_censoring_aware_embedding_panel,
)


def test_horizon_labels_distinguish_primary_competing_and_early_censoring():
    data = horizon_ipcw_data(
        np.array([10.0, 20.0, 30.0, 100.0]),
        np.array([1, 2, 0, 0]),
        60.0,
    )
    assert data.eligible.tolist() == [True, True, False, True]
    assert data.target.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert data.status.tolist() == [
        "primary event", "competing death", "ineligible/early censor", "known event-free"
    ]
    assert np.all(data.weights[data.eligible] > 0)


def test_censoring_aware_panel_smoke():
    rng = np.random.default_rng(3)
    n = 80
    coords = rng.normal(size=(n, 2))
    times = rng.uniform(5, 200, size=n)
    events = rng.choice([0, 1, 2], size=n, p=[0.35, 0.5, 0.15])
    covariates = pd.DataFrame({"age": rng.normal(60, 10, n), "cohort": rng.choice(["A", "B"], n)})
    fig, diagnostics = plot_censoring_aware_embedding_panel(
        coords, times, events, horizon=90, covariates=covariates,
        grid_size=15, min_effective_support=2,
    )
    assert len(fig.axes) >= 5
    assert diagnostics["n_total"] == n
