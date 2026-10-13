import json
from pathlib import Path

import pytest
import yaml

from opera.run.compare_approved_neural_validation_v1 import compare


def _make_run(root: Path, auc: float, *, cohort: str = "A") -> Path:
    root.mkdir()
    config = {
        "cohorts": {cohort: {"data_dir": "same"}},
        "outcomes": {"target": {"path": "same"}},
        "paths": {"sequence_data_dir": "same"},
        "competing_risk": {"objective": "horizon_bce", "interval_boundaries_days": [30, 365],
                           "readout_mode": "shared", "head_mode": "linear"},
        "training": {"evaluation_horizons_days": [30, 365],
                     "eval_monitor_metric": "val/cr_cif_auc_endpoint_group_macro",
                     "selection_min_cases": 20, "selection_min_controls": 20},
        "cross_outcome": {"endpoint_groups": {"target": ["target"]},
                          "outcome_families": {"family": ["target"]},
                          "aggregation": "endpoint_group_macro"},
        "seed": 42,
    }
    hydra = root / "hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    for name in ("best.ckpt", "checkpoint_metadata.json"):
        (root / name).write_text("present", encoding="utf-8")
    (root / "training_complete.json").write_text(json.dumps({"status": "completed"}),
                                                    encoding="utf-8")
    (root / "endpoint_selection_inventory.json").write_text(json.dumps([{"cell": 1}]),
                                                             encoding="utf-8")
    metrics = root / "contrastive_multicohort_runs" / "version_0"
    metrics.mkdir(parents=True)
    metrics.joinpath("metrics.csv").write_text(
        "epoch,val/cr_cif_auc_endpoint_group_macro,val/cr_cif_auc_selection_groups,val/cr_cif_selection_cells\n"
        f"0,{auc},1,1\n1,{auc + 0.01},1,1\n", encoding="utf-8")
    return root


def test_matched_tuning_contrast(tmp_path: Path) -> None:
    shared = _make_run(tmp_path / "shared", 0.70)
    trunks = _make_run(tmp_path / "trunks", 0.72)
    result = compare({"shared": shared, "family_trunks": trunks})
    assert result["selection_inventory_cells"] == 1
    assert result["contrasts"][0]["question"] == "readout_family_trunks"
    assert result["contrasts"][0]["best_auc_delta_candidate_minus_baseline"] == pytest.approx(0.02)
    assert result["common_completed_epoch"] == 2
    assert result["contrasts"][0]["common_epoch_auc_delta"] == pytest.approx(0.02)


def test_encoder_uses_run_identity_when_display_label_is_short(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "time2vec_router_interacting_scale02_e2e", 0.72)
    result = compare({"interactive_02": run})
    assert result["runs"]["interactive_02"]["encoder"] == "scaled_time2vec"


def test_rejects_different_cohort_contract(tmp_path: Path) -> None:
    shared = _make_run(tmp_path / "shared", 0.70)
    trunks = _make_run(tmp_path / "trunks", 0.72, cohort="B")
    with pytest.raises(ValueError, match="contract differs"):
        compare({"shared": shared, "family_trunks": trunks})


def test_requires_actual_hydra_directory(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "shared", 0.70)
    (run / "hydra" / "config.yaml").rename(run / "hydra" / "moved.yaml")
    with pytest.raises(ValueError, match="Missing resolved Hydra config:.*hydra"):
        compare({"shared": run})


def test_resumed_logger_versions_are_combined(tmp_path: Path) -> None:
    shared = _make_run(tmp_path / "shared", 0.70)
    resumed = _make_run(tmp_path / "resumed", 0.72)
    next_log = resumed / "contrastive_multicohort_runs" / "version_1"
    next_log.mkdir()
    next_log.joinpath("metrics.csv").write_text(
        "epoch,val/cr_cif_auc_endpoint_group_macro,val/cr_cif_auc_selection_groups,val/cr_cif_selection_cells\n"
        "2,0.75,1,1\n", encoding="utf-8"
    )
    result = compare({"shared": shared, "family_trunks": resumed})
    assert result["runs"]["family_trunks"]["validated_epochs"] == 3
    assert len(result["runs"]["family_trunks"]["metrics_files"]) == 2
    assert result["runs"]["family_trunks"]["best_epoch"] == 3
    assert result["common_completed_epoch"] == 2
    assert result["contrasts"][0]["common_epoch_auc_delta"] == pytest.approx(0.02)


def test_resumed_logger_later_version_wins_overlap(tmp_path: Path) -> None:
    shared = _make_run(tmp_path / "shared", 0.70)
    resumed = _make_run(tmp_path / "resumed", 0.72)
    next_log = resumed / "contrastive_multicohort_runs" / "version_1"
    next_log.mkdir()
    next_log.joinpath("metrics.csv").write_text(
        "epoch,val/cr_cif_auc_endpoint_group_macro,val/cr_cif_auc_selection_groups,val/cr_cif_selection_cells\n"
        "1,0.78,1,1\n2,0.76,1,1\n", encoding="utf-8"
    )
    result = compare({"shared": shared, "family_trunks": resumed})
    summary = result["runs"]["family_trunks"]
    assert summary["validated_epochs"] == 3
    assert summary["per_epoch_endpoint_group_auc"]["2"] == pytest.approx(0.78)
    assert summary["best_epoch"] == 2
