"""Checks for readable atlas legends and actual pipeline figure exports."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.run.plot_patient_embeddings import write_metadata_atlases
from opera.visualization.classification_plots import (
    survival_plot_horizon,
    plot_evaluation_panel,
)
from opera.visualization.embedding_plots import plot_embedding_stage_metadata_grid
from opera.visualization.training_report import export_training_report


def teardown_function():
    plt.close("all")


def test_atlas_legend_keeps_all_categories_and_removes_coordinate_ticks():
    coords = np.random.default_rng(0).normal(size=(40, 2))
    groups = np.resize([f"Group {i}" for i in range(10)], 40)
    fig = plot_embedding_stage_metadata_grid({"opera": coords}, {"Cohort": groups})
    panels = [ax for ax in fig.axes if ax.get_label().startswith("panel:")]
    assert len(panels) == 1
    assert not len(panels[0].get_xticks())
    labels = [
        text.get_text()
        for ax in fig.axes
        if ax.get_legend()
        for text in ax.get_legend().get_texts()
    ]
    assert len(labels) == 10
    assert all(" ·  4" in text for text in labels)


def test_patient_pipeline_writes_paginated_atlas_from_saved_coordinates(tmp_path):
    frame = pd.DataFrame(
        {
            "component_1": np.arange(30),
            "component_2": np.sin(np.arange(30)),
            **{f"metadata_{i}": np.resize(["a", "b", None], 30) for i in range(5)},
        }
    )
    outputs = write_metadata_atlases(
        frame, [f"metadata_{i}" for i in range(5)], method="pca", output_dir=tmp_path
    )
    assert len(outputs) == 2
    assert all(path.exists() and path.with_suffix(".pdf").exists() for path in outputs)


def test_training_report_includes_validation_only_metrics(tmp_path):
    pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train/loss": [0.8, 0.6, 0.4],
            "val/auroc": [0.6, 0.7, 0.8],
        }
    ).to_csv(tmp_path / "metrics.csv", index=False)
    logger = SimpleNamespace(log_dir=tmp_path, save=lambda: None)
    assert export_training_report(logger, tmp_path, is_global_zero=False) == []
    paths = export_training_report(logger, tmp_path)
    assert len(paths) == 1 and paths[0].exists()


def test_survival_horizon_uses_explicit_value_or_observed_followup():
    assert survival_plot_horizon([10, 20, np.nan], 365) == 365
    assert survival_plot_horizon([10, 20, np.nan]) == 19
    assert survival_plot_horizon([np.nan, -1]) is None


def test_panel_prefers_survival_even_without_window(monkeypatch):
    import opera.visualization.classification_plots as plots

    received = {}

    def fake_curve(times, events, risk, **kwargs):
        received.update(kwargs)
        kwargs["ax"].set_title(kwargs["title"])

    monkeypatch.setattr(plots, "plot_timedep_auc", fake_curve)
    plot_evaluation_panel(
        np.array([0, 1] * 20),
        np.tile([0.2, 0.8], 20),
        times=np.arange(1, 41),
        events=np.array([0, 1] * 20),
    )
    assert received["title"] == "Survival discrimination"
    assert received["window_days"] == np.percentile(np.arange(1, 41), 90)
