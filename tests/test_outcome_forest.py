from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from opera.run.plot_outcome_forest import _family_map, main as plot_outcome_forest_main
from opera.visualization.outcome_forest import (
    apply_outcome_family_map,
    family_macro_summary,
    normalize_contrast_table,
    normalize_estimate_table,
    paginate_outcomes,
    paired_comparison_to_forest_tables,
    plot_outcome_performance_forest,
)


def _paired() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome": ["anemia_g3plus", "infections_IV_based", "aki"],
            "outcome_family": ["Hematologic", "Infection", "Organ toxicity"],
            "cohort": ["DLBCL"] * 3,
            "horizon_days": [365] * 3,
            "n_paired": [900, 850, 875],
            "n_cases": [120, 75, 90],
            "opera_cif_auc": [0.79, 0.70, 0.76],
            "opera_cif_auc_lower": [0.76, 0.66, 0.72],
            "opera_cif_auc_upper": [0.82, 0.74, 0.80],
            "tabular_cif_auc": [0.82, 0.72, 0.75],
            "tabular_cif_auc_lower": [0.79, 0.68, 0.71],
            "tabular_cif_auc_upper": [0.85, 0.76, 0.79],
            "delta_cif_auc": [-0.03, -0.02, 0.01],
            "delta_cif_auc_lower": [-0.05, -0.04, -0.01],
            "delta_cif_auc_upper": [-0.01, 0.00, 0.03],
        }
    )


def test_paired_comparison_adapter_preserves_sign_and_denominators():
    estimates, contrasts = paired_comparison_to_forest_tables(_paired())
    assert set(estimates["model"]) == {"opera", "tabular_ehr"}
    assert len(estimates) == 6
    assert estimates["lower"].notna().all()
    anemia = contrasts.loc[contrasts["outcome"] == "anemia_g3plus"].iloc[0]
    assert anemia["estimate"] == pytest.approx(-0.03)
    assert anemia["lower"] == pytest.approx(-0.05)
    assert anemia["n"] == 900
    assert anemia["n_events"] == 120


def test_normalizers_accept_bayesian_style_columns_without_intervals():
    estimates = normalize_estimate_table(
        pd.DataFrame(
            {
                "outcome": ["aki"],
                "model_family": ["opera"],
                "cif_auc": [0.8],
            }
        ),
        metric="cif_auc",
    )
    assert estimates.iloc[0]["outcome_family"] == "Other"
    assert pd.isna(estimates.iloc[0]["lower"])

    contrasts = normalize_contrast_table(
        pd.DataFrame(
            {
                "outcome": ["aki"],
                "comparator": ["opera"],
                "baseline": ["tabular_ehr"],
                "delta_mean": [0.02],
                "ci_lower": [-0.01],
                "ci_upper": [0.05],
                "posterior_probability_superior": [0.91],
            }
        ),
        metric="cif_auc",
    )
    assert contrasts.iloc[0]["contrast"] == "opera_minus_tabular_ehr"
    assert contrasts.iloc[0]["probability_superior"] == pytest.approx(0.91)


def test_family_mapping_and_pagination_keep_small_families_together():
    estimates, _ = paired_comparison_to_forest_tables(_paired())
    estimates = apply_outcome_family_map(
        estimates,
        {
            "anemia_g3plus": "Cytopenia",
            "infections_IV_based": "Infection",
            "aki": "Organ toxicity",
        },
    )
    pages = paginate_outcomes(estimates, max_outcomes_per_page=2)
    assert sum(len(page) for page in pages) == 3
    assert len({outcome for page in pages for outcome in page}) == 3


def test_family_macro_summary_uses_equal_outcome_weights_without_fake_ci():
    estimates, contrasts = paired_comparison_to_forest_tables(_paired())
    estimates.loc[estimates["outcome"].eq("aki"), "outcome_family"] = "Hematologic"
    summary = family_macro_summary(estimates, contrasts)
    opera = summary.loc[
        summary["record_type"].eq("absolute")
        & summary["outcome_family"].eq("Hematologic")
        & summary["series"].eq("opera")
    ].iloc[0]
    assert opera["estimate"] == pytest.approx((0.79 + 0.76) / 2)
    assert opera["n_outcomes"] == 2
    assert "lower" not in summary.columns


def test_family_map_reads_hydra_config_without_resolving_unrelated_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "cross_outcome:\n"
        "  outcome_families:\n"
        "    Renal toxicity: [aki_first_documented]\n"
        "paths:\n"
        "  output: ${version:run.id}\n",
        encoding="utf-8",
    )
    assert _family_map(config) == {"aki_first_documented": "Renal toxicity"}


def test_forest_draws_aligned_absolute_delta_and_annotation_panels(tmp_path):
    estimates, contrasts = paired_comparison_to_forest_tables(_paired())
    output = tmp_path / "forest.png"
    fig = plot_outcome_performance_forest(
        estimates,
        contrasts,
        model_order=["opera", "tabular_ehr"],
        interval_label="95% CI",
        save_path=output,
    )
    assert len(fig.axes) == 3
    assert fig.axes[0].get_title(loc="left") == "Absolute held-out performance"
    assert fig.axes[1].get_title(loc="left") == "OPERA − Tabular EHR"
    assert any(line.get_linestyle() == "--" for line in fig.axes[1].lines)
    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert np.array_equal(
        fig.axes[0].collections[0].get_paths()[0].vertices,
        fig.axes[1].collections[0].get_paths()[0].vertices,
    )
    assert np.allclose(
        fig.axes[0].collections[0].get_facecolors(),
        fig.axes[1].collections[0].get_facecolors(),
    )
    plt.close(fig)


def test_forest_can_label_tuning_without_claiming_held_out():
    estimates, contrasts = paired_comparison_to_forest_tables(_paired())
    fig = plot_outcome_performance_forest(
        estimates, contrasts, evaluation_label="tuning",
    )
    assert fig.axes[0].get_title(loc="left") == "A  Absolute performance"
    assert any("Tuning only" in text.get_text() for text in fig.texts)
    plt.close(fig)


def test_forest_refuses_mixed_horizons():
    estimates, contrasts = paired_comparison_to_forest_tables(_paired())
    estimates.loc[0, "horizon_days"] = 90
    with pytest.raises(ValueError, match="mix horizons"):
        plot_outcome_performance_forest(estimates, contrasts)


def test_plot_cli_writes_figure_data_and_manifest(tmp_path, monkeypatch):
    paired = tmp_path / "paired.csv"
    output = tmp_path / "figures"
    _paired().to_csv(paired, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_outcome_forest",
            "--paired-comparison",
            str(paired),
            "--horizon",
            "365",
            "--max-outcomes-per-page",
            "2",
            "--output-dir",
            str(output),
        ],
    )
    plot_outcome_forest_main()
    assert (output / "forest_plot_data.csv").is_file()
    assert (output / "forest_plot_manifest.json").is_file()
    assert len(list(output.glob("*.png"))) == 2
    assert len(list(output.glob("*.pdf"))) == 2


def test_plot_cli_all_outcomes_one_page(tmp_path, monkeypatch):
    paired = tmp_path / "paired.csv"
    output = tmp_path / "all"
    _paired().to_csv(paired, index=False)
    monkeypatch.setattr(
        "sys.argv",
        ["plot_outcome_forest", "--paired-comparison", str(paired),
         "--horizon", "365", "--max-outcomes-per-page", "2",
         "--all-outcomes-one-page", "--output-dir", str(output)],
    )
    plot_outcome_forest_main()
    assert len(list(output.glob("*.png"))) == 1
    assert (output / "forest_family_macro_summary.csv").is_file()
