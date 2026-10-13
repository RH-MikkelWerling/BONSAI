from pathlib import Path

import pandas as pd
import pytest

from opera.run.plot_predictability_prototype import prepare_data


REGISTRY = Path("opera/configs/experiment_registry.yaml")


def test_predictability_prototype_rejects_heldout_rows(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({
        "outcome": ["overall_survival"], "cohort": ["ALL_HEMATOLOGY"],
        "split": ["held_out"], "evaluation_regime": ["fixed_horizon"],
        "auroc": [0.8], "n_positive": [10],
    }))
    with pytest.raises(ValueError, match="tuning rows only"):
        prepare_data(Path("unused.csv"), REGISTRY)


def test_predictability_prototype_maps_registered_family(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({
        "outcome": ["overall_survival"], "cohort": ["ALL_HEMATOLOGY"],
        "split": ["tuning"], "evaluation_regime": ["fixed_horizon"],
        "auroc": [0.8], "n_positive": [10],
    }))
    frame = prepare_data(Path("unused.csv"), REGISTRY)
    assert frame.iloc[0]["display_group"] == "Disease course and later malignancy"
