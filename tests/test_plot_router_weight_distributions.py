from pathlib import Path

import numpy as np
import pytest

from opera.run.plot_router_weight_distributions import (
    plot_bank, source_distribution_summary,
)


def test_distribution_summary_keeps_all_patients_and_home_source():
    weights = np.array([[.2, .38, .02], [.2, .02, .38]])
    summary = source_distribution_summary(
        weights, ["sepsis", "aki_first_documented", "bleeding"],
        home="sepsis", bank="outcome",
    ).set_index("source")
    assert len(summary) == 3
    assert summary.loc["sepsis", "is_home"]
    assert summary.loc["aki_first_documented", "mean_weight"] == pytest.approx(.2)
    assert summary.loc["bleeding", "mean_weight"] == pytest.approx(.2)
    assert summary.loc["aki_first_documented", "sd_weight"] > 0
    assert summary.loc["aki_first_documented", "n_patients"] == 2
    assert summary.loc["aki_first_documented", "display_name"] == "First documented AKI"


def test_distribution_plot_writes_each_bank_without_subsampling(tmp_path: Path):
    n = 30
    weights = np.tile(np.array([[.12, .08, .03]]), (n, 1))
    weights[:, 1] += np.linspace(0, .02, n)
    output = tmp_path / "outcome_sources.png"
    summary = plot_bank(weights, ["sepsis", "aki_first_documented", "bone_fracture"],
                        bank="outcome", disease="CLL", outcome="sepsis", output=output)
    assert len(summary) == 3
    assert output.stat().st_size > 0
    assert output.with_suffix(".pdf").stat().st_size > 0


def test_distribution_rejects_nonfinite_weights():
    with pytest.raises(ValueError, match="Invalid"):
        source_distribution_summary(np.array([[np.nan]]), ["CLL"],
                                    home="CLL", bank="disease")
