import pytest


def test_rarity_plot_helpers_generate_files(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    from opera.visualization.rarity_plots import (
        plot_real_rarity_delta,
        write_rarity_plots,
    )

    synthetic = pd.DataFrame(
        [
            {
                "model_family": "opera",
                "training_fraction": 0.1,
                "median_delta_auroc": 0.03,
                "lower_delta_auroc": 0.01,
                "upper_delta_auroc": 0.05,
            },
            {
                "model_family": "opera",
                "training_fraction": 1.0,
                "median_delta_auroc": 0.08,
                "lower_delta_auroc": 0.04,
                "upper_delta_auroc": 0.10,
            },
        ]
    )
    real = pd.DataFrame(
        [
            {
                "model_family": "opera",
                "n_train": 42,
                "n_events_train": 6,
                "delta_auroc_vs_baseline": 0.06,
                "auroc_lower": 0.62,
                "auroc_upper": 0.78,
                "baseline_auroc": 0.65,
                "supplement_only": True,
            }
        ]
    )

    write_rarity_plots(synthetic, real, str(tmp_path))

    assert (tmp_path / "synthetic_rarity_delta.png").exists()
    assert (tmp_path / "synthetic_rarity_delta.pdf").exists()
    assert (tmp_path / "real_rarity_delta.png").exists()
    assert (tmp_path / "real_rarity_delta.pdf").exists()
    assert (tmp_path / "rarity_delta_combined.png").exists()
    assert (tmp_path / "rarity_delta_combined.pdf").exists()

    fig = plot_real_rarity_delta(real)
    assert fig.axes[0].get_xlabel() == "Training events"


def test_real_rarity_trend_uses_stable_cells_only():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    from opera.visualization.rarity_plots import plot_real_rarity_delta

    real = pd.DataFrame(
        [
            {
                "model_family": "opera",
                "n_events_train": 5,
                "delta_auroc_vs_baseline": 0.10,
                "supplement_only": False,
            },
            {
                "model_family": "opera",
                "n_events_train": 15,
                "delta_auroc_vs_baseline": 0.06,
                "supplement_only": False,
            },
            {
                "model_family": "opera",
                "n_events_train": 45,
                "delta_auroc_vs_baseline": 0.03,
                "supplement_only": False,
            },
            {
                "model_family": "opera",
                "n_events_train": 2,
                "delta_auroc_vs_baseline": -0.25,
                "supplement_only": True,
            },
        ]
    )

    fig = plot_real_rarity_delta(real)

    line_labels = [line.get_label() for line in fig.axes[0].lines]
    assert any("trend (stable cells)" in label for label in line_labels)


def test_synthetic_rarity_plot_adds_smoothed_trend():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    from opera.visualization.rarity_plots import plot_synthetic_rarity_delta

    synthetic = pd.DataFrame(
        [
            {
                "model_family": "opera",
                "training_fraction": 0.05,
                "median_delta_auroc": 0.01,
            },
            {
                "model_family": "opera",
                "training_fraction": 0.10,
                "median_delta_auroc": 0.04,
            },
            {
                "model_family": "opera",
                "training_fraction": 0.25,
                "median_delta_auroc": 0.07,
            },
            {
                "model_family": "opera",
                "training_fraction": 1.00,
                "median_delta_auroc": 0.08,
            },
        ]
    )

    fig = plot_synthetic_rarity_delta(synthetic)

    line_labels = [line.get_label() for line in fig.axes[0].lines]
    assert any("smoothed trend" in label for label in line_labels)
