"""Headless smoke tests for OPERA visualization functions.

These tests only verify that figures build and return the expected object types
under the Agg backend. They deliberately do not assert pixel content. Functions
with optional heavy dependencies (umap-learn, scikit-learn manifold) are wrapped
so the test skips cleanly if the dependency is unavailable.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from opera.visualization.classification_plots import (  # noqa: E402
    plot_calibration,
    plot_full_evaluation,
    plot_roc_curve,
)
from opera.visualization.embedding_plots import (  # noqa: E402
    plot_embedding_projection,
    plot_embedding_stage_metadata_grid,
    plot_similarity_distributions,
)
from opera.visualization.analysis_plots import (  # noqa: E402
    plot_within_stratum_grid,
)
from opera.visualization.survival_plots import (  # noqa: E402
    plot_calibration_comparison,
    plot_timedep_auc_curves,
)


@pytest.fixture
def binary_data():
    """Deterministic separable-ish binary classification arrays."""
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n)
    # Probabilities correlated with the label so curves are well-defined.
    probabilities = np.clip(
        0.25 + 0.5 * labels + rng.normal(0, 0.15, size=n), 0.001, 0.999
    )
    return labels, probabilities


@pytest.fixture
def survival_data():
    """Deterministic survival arrays: times, events (0/1), and risk scores."""
    rng = np.random.default_rng(1)
    n = 200
    times = rng.uniform(10.0, 1500.0, size=n)
    events = rng.integers(0, 2, size=n)
    risk = np.clip(rng.uniform(0.0, 1.0, size=n), 0.001, 0.999)
    return times, events, risk


def teardown_function(_):
    plt.close("all")


def test_plot_full_evaluation_smoke(binary_data, tmp_path):
    labels, probabilities = binary_data

    figs = plot_full_evaluation(
        labels,
        probabilities,
        output_dir=str(tmp_path),
        prefix="binary",
    )

    assert isinstance(figs, dict)
    assert "panel" in figs and isinstance(figs["panel"], Figure)
    assert "roc" in figs and isinstance(figs["roc"], Figure)
    # Files were written for the saved figures.
    assert (tmp_path / "binary_roc_curve.png").exists()


def test_plot_full_evaluation_cox_mode(binary_data, survival_data, tmp_path):
    """Survival ('cox') path: passing times/events/window_days adds time-dep AUC."""
    labels, probabilities = binary_data
    times, events, risk = survival_data

    figs = plot_full_evaluation(
        labels,
        probabilities,
        output_dir=str(tmp_path),
        prefix="surv",
        times=times,
        events=events,
        survival_probabilities=risk,
        window_days=1500.0,
        outcome_name="overall_survival",
    )

    assert isinstance(figs, dict)
    assert "timedep_auc" in figs
    assert isinstance(figs["timedep_auc"], Figure)


def test_survival_timedep_auc_curves_smoke(tmp_path):
    """DataFrame-driven IPCW-AUC-over-time figure from survival_plots."""
    rng = np.random.default_rng(2)
    rows = []
    for model in ("opera", "tabular_rkkp"):
        for horizon in (30, 90, 365, 730):
            auc = float(np.clip(0.6 + rng.normal(0, 0.03), 0.5, 0.95))
            rows.append(
                {
                    "outcome": "os",
                    "model_family": model,
                    "horizon_days": horizon,
                    "ipcw_auc": auc,
                    "ci_lower": auc - 0.05,
                    "ci_upper": auc + 0.05,
                }
            )
    df = pd.DataFrame(rows)

    fig = plot_timedep_auc_curves(df, outcome="os", save_path=str(tmp_path / "auc.pdf"))

    assert isinstance(fig, Figure)
    assert (tmp_path / "auc.pdf").exists()


def test_survival_calibration_comparison_smoke(tmp_path):
    """Cohort calibration comparison from survival_plots."""
    rng = np.random.default_rng(3)
    rows = []
    for cohort in ("DLBCL", "FL"):
        predicted = np.linspace(0.05, 0.95, 8)
        observed = np.clip(predicted + rng.normal(0, 0.03, size=8), 0, 1)
        for pred, obs in zip(predicted, observed):
            rows.append(
                {
                    "outcome": "os",
                    "cohort": cohort,
                    "predicted": float(pred),
                    "observed": float(obs),
                    "ece": 0.04,
                    "hl_pvalue": 0.21,
                }
            )
    df = pd.DataFrame(rows)

    fig = plot_calibration_comparison(
        df, outcome="os", save_path=str(tmp_path / "cal.pdf")
    )

    assert isinstance(fig, Figure)


def test_embedding_projection_pca_smoke(tmp_path):
    """Embedding projection via PCA-free path.

    plot_embedding_projection supports method='umap' (optional dep) and 'tsne'.
    UMAP/t-SNE are heavy and may be unavailable, so skip on ImportError.
    """
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(60, 16)).astype(np.float32)
    labels = rng.integers(0, 2, size=60)

    try:
        fig = plot_embedding_projection(
            embeddings,
            labels,
            method="tsne",
            density_contours=False,
            save_path=str(tmp_path / "proj.pdf"),
        )
    except ImportError:
        pytest.skip("manifold reducer dependency not installed")

    assert isinstance(fig, Figure)


def test_plot_similarity_distributions_smoke(tmp_path):
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(120, 8)).astype(np.float32)
    labels = rng.integers(0, 2, size=120)

    fig = plot_similarity_distributions(
        embeddings,
        labels,
        n_pairs=2000,
        seed=0,
        save_path=str(tmp_path / "sim.pdf"),
    )

    assert isinstance(fig, Figure)
    assert (tmp_path / "sim.pdf").exists()


def test_within_stratum_grid_uses_shared_projection_and_explanatory_layout(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(17)
    n_per_stratum = [45, 70, 55]
    strata = np.concatenate(
        [np.repeat(value, n) for value, n in zip(["0-1", "2-3", "4-5"], n_per_stratum)]
    )
    labels = np.concatenate(
        [
            np.array([1] * n_events + [0] * (n - n_events))
            for n, n_events in zip(n_per_stratum, [5, 21, 29])
        ]
    )
    embeddings = rng.normal(size=(len(labels), 8))
    coords = rng.normal(size=(len(labels), 2))

    import opera.visualization.embedding_plots as embedding_plots

    monkeypatch.setattr(
        embedding_plots,
        "_reduce_embeddings",
        lambda values, method: coords,
    )
    output = tmp_path / "within_stratum.png"
    fig = plot_within_stratum_grid(
        embeddings,
        labels,
        strata,
        stratum_values=["0-1", "2-3", "4-5"],
        stratum_name="IPI",
        highlight_stratum="2-3",
        callout_text="Same IPI band; outcomes occupy different regions",
        talk_note="The middle panel is the prespecified clinical example.",
        save_path=str(output),
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    assert [axis.get_xlim() for axis in fig.axes].count(fig.axes[0].get_xlim()) == 3
    assert [axis.get_ylim() for axis in fig.axes].count(fig.axes[0].get_ylim()) == 3
    assert "event prevalence = 30%" in fig.axes[1].get_title()
    assert any("Same IPI band" in text.get_text() for text in fig.axes[1].texts)
    assert (
        fig.axes[1].spines["left"].get_linewidth()
        > fig.axes[0].spines["left"].get_linewidth()
    )
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend_labels == [
        "all patients (background context)",
        "within-stratum no event",
        "within-stratum event",
    ]
    figure_text = " ".join(text.get_text() for text in fig.texts)
    assert "Shared UMAP projection" in figure_text
    assert "held-out discrimination" in figure_text.replace("\n", " ")
    assert output.exists()
    assert output.with_suffix(".pdf").exists()


def test_within_stratum_grid_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="equal length"):
        plot_within_stratum_grid(
            np.zeros((4, 2)),
            np.zeros(3),
            np.zeros(4),
        )


def test_embedding_stage_metadata_grid_reuses_each_stage_projection(tmp_path):
    rng = np.random.default_rng(29)
    n = 84
    dapt = rng.normal(size=(n, 2))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    opera = np.column_stack([np.cos(theta), np.sin(theta)])
    metadata = {
        "Age": np.linspace(35, 85, n),
        "Sex": np.where(np.arange(n) % 2, "Female", "Male"),
        "Cohort (fine)": np.array(["DLBCL", "FL", "CLL"])[np.arange(n) % 3],
        "Treatment failure": (np.arange(n) % 5 == 0).astype(int),
    }
    output = tmp_path / "stage_metadata.png"
    fig = plot_embedding_stage_metadata_grid(
        {"dapt": dapt, "opera": opera},
        metadata,
        variable_kinds={"Age": "continuous"},
        stage_labels={"dapt": "DAPT\nembeddings", "opera": "OPERA\nembeddings"},
        category_labels={
            "Treatment failure": {0: "No failure", 1: "Treatment failure"}
        },
        annotations={
            ("opera", "Treatment failure"): [
                {"text": "Prespecified region", "xy": (1.0, 0.0)}
            ]
        },
        outcome_definition="Failure within the prespecified follow-up window.",
        save_path=str(output),
    )

    assert isinstance(fig, Figure)
    panel_axes = [axis for axis in fig.axes if axis.get_label().startswith("panel:")]
    assert len(panel_axes) == 8
    assert np.allclose(panel_axes[0].collections[0].get_offsets(), dapt)
    assert np.allclose(panel_axes[4].collections[0].get_offsets(), opera)
    assert len({axis.get_xlim() for axis in panel_axes[:4]}) == 1
    assert len({axis.get_ylim() for axis in panel_axes[:4]}) == 1
    assert len({axis.get_xlim() for axis in panel_axes[4:]}) == 1
    assert len({axis.get_ylim() for axis in panel_axes[4:]}) == 1
    assert [axis.get_title() for axis in panel_axes[:4]] == list(metadata)
    assert any(
        "Prespecified region" in text.get_text() for text in panel_axes[-1].texts
    )
    all_text = [*fig.texts, *(text for axis in panel_axes for text in axis.texts)]
    figure_text = " ".join(text.get_text() for text in all_text).replace("\n", " ")
    assert "DAPT embeddings" in figure_text
    assert "OPERA embeddings" in figure_text
    assert "held-out probes" in figure_text
    assert "coordinates are identical" in figure_text
    assert output.exists()
    assert output.with_suffix(".pdf").exists()


def test_embedding_stage_metadata_grid_rejects_misaligned_metadata():
    with pytest.raises(ValueError, match="one value per patient"):
        plot_embedding_stage_metadata_grid(
            {"opera": np.zeros((5, 2))},
            {"Age": np.zeros(4)},
        )


def test_plot_roc_curve_smoke(binary_data, tmp_path):
    labels, probabilities = binary_data

    fig = plot_roc_curve(
        labels,
        probabilities,
        title="ROC smoke",
        save_path=str(tmp_path / "roc.pdf"),
    )

    assert isinstance(fig, Figure)
    assert (tmp_path / "roc.pdf").exists()


def test_plot_calibration_curve_smoke(binary_data, tmp_path):
    labels, probabilities = binary_data

    fig = plot_calibration(
        labels,
        probabilities,
        n_bins=8,
        title="Calibration smoke",
        save_path=str(tmp_path / "cal.pdf"),
    )

    assert isinstance(fig, Figure)
    assert (tmp_path / "cal.pdf").exists()
