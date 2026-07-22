from pathlib import Path

import yaml

from opera.config_contracts import load_sweep_config
from opera.run.generate_sweep_configs import generate_configs, load_registry


REGISTRY = Path("opera/configs/experiment_registry.yaml")
SECOND_LINE_OUTCOMES = {
    "second_line_treatment",
    "treatment_failure",
    "treatment_failure_transformation",
}


def test_production_registry_has_locked_inventory() -> None:
    registry = load_registry(REGISTRY)

    assert len(registry["outcomes"]) == 87
    assert len(registry["cohort_groups"]) == 10
    assert sum(
        len(group["fine"]) for group in registry["cohort_groups"].values()
    ) == 24
    assert sum(
        count
        for group in registry["cohort_groups"].values()
        for count in group["fine"].values()
    ) == 38_905
    assert registry["horizons_days"] == [30, 90, 180, 365, 730]


def test_generated_sweeps_pass_contract_and_encode_availability(tmp_path: Path) -> None:
    paths = generate_configs(REGISTRY, tmp_path)
    sweep_paths = [
        path
        for path in paths
        if path.name.startswith(("fine_", "grouped_"))
    ]

    assert len(paths) == 15
    assert len(sweep_paths) == 12
    for path in sweep_paths:
        load_sweep_config(path)

    grouped = yaml.safe_load((tmp_path / "grouped_cox.yaml").read_text())
    assert set(grouped["cohorts"]["BL_LBL"]["exclude_outcomes"]) == SECOND_LINE_OUTCOMES
    assert set(grouped["cohorts"]["HCL"]["exclude_outcomes"]) == SECOND_LINE_OUTCOMES
    assert "exclude_outcomes" not in grouped["cohorts"]["MM"]
    assert grouped["outcomes"]["overall_survival"].get("competing_outcome_path") is None
    assert (
        grouped["outcomes"]["sepsis"]["competing_outcome_path"]
        == "${BONSAI_OUTCOMES_DIR}/overall_survival.parquet"
    )
    assert grouped["outcomes"]["sepsis"]["n_hours_end_include"] is None

    fine = yaml.safe_load((tmp_path / "fine_ipcw_30d.yaml").read_text())
    assert (
        fine["cohorts"]["TRANSFORMED_FL"]["data_dir"]
        == "${BONSAI_PROCESSED_DATA}/hematology_all"
    )
    assert fine["cohorts"]["MCL"]["population_file"] == "${BONSAI_COHORT_MEMBERSHIP}"
    assert fine["cohorts"]["SolM"]["cohort_fine_value"] == "SolM"
    assert set(fine["cohorts"]["BL"]["exclude_outcomes"]) == SECOND_LINE_OUTCOMES
    assert fine["outcomes"]["sepsis"]["n_hours_end_include"] == 30 * 24

    joint = yaml.safe_load((tmp_path / "joint_opera_full_panel.yaml").read_text())
    mol = yaml.safe_load((tmp_path / "multi_outcome_full_panel.yaml").read_text())
    assert len(joint["outcomes"]) == len(mol["outcomes"]) == 87
    assert "${BONSAI_" not in yaml.safe_dump(joint)
    assert "${BONSAI_" not in yaml.safe_dump(mol)
    assert set(joint["cohorts"]) == {
        "AMYLOIDOSIS", "BL_LBL", "CLL_SLL", "DLBCL_like", "HCL", "HL",
        "Indolent_B_NHL", "MCL", "MM", "T_NHL",
    }
