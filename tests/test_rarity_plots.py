import pytest


def test_rarity_plot_helpers_generate_files(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    from opera.visualization.rarity_plots import write_rarity_plots

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
