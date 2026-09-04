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
    assert sum(len(group["fine"]) for group in registry["cohort_groups"].values()) == 24
    assert (
        sum(
            count
            for group in registry["cohort_groups"].values()
            for count in group["fine"].values()
        )
        == 38_905
    )
    assert registry["horizons_days"] == [30, 90, 180, 365, 730]
    assert set(registry["outcomes_containing_death"]) == {
        "overall_survival",
        "treatment_failure",
        "treatment_failure_transformation",
    }


def test_generated_sweeps_pass_contract_and_encode_availability(tmp_path: Path) -> None:
    paths = generate_configs(REGISTRY, tmp_path)
    sweep_paths = [
        path for path in paths if path.name.startswith(("fine_", "grouped_"))
    ]

    assert len(paths) == 45
    assert len(sweep_paths) == 32
    for path in sweep_paths:
        load_sweep_config(path)

    grouped = yaml.safe_load((tmp_path / "grouped_cox.yaml").read_text())
    assert set(grouped["cohorts"]["BL_LBL"]["exclude_outcomes"]) == SECOND_LINE_OUTCOMES
    assert set(grouped["cohorts"]["HCL"]["exclude_outcomes"]) == SECOND_LINE_OUTCOMES
    assert (
        set(grouped["cohorts"]["AMYLOIDOSIS"]["exclude_outcomes"])
        == SECOND_LINE_OUTCOMES
    )
    assert "exclude_outcomes" not in grouped["cohorts"]["MM"]
    assert grouped["outcomes"]["overall_survival"].get("competing_outcome_path") is None
    assert grouped["outcomes"]["treatment_failure"].get("competing_outcome_path") is None
    assert (
        grouped["outcomes"]["treatment_failure_transformation"].get(
            "competing_outcome_path"
        )
        is None
    )
    assert (
        grouped["outcomes"]["sepsis"]["competing_outcome_path"]
        == "${BONSAI_OUTCOMES_DIR}/overall_survival.parquet"
    )
    assert grouped["outcomes"]["sepsis"]["n_hours_end_include"] is None

    fine = yaml.safe_load((tmp_path / "fine_ipcw_30d.yaml").read_text())
    assert (
        fine["cohorts"]["TRANSFORMED_FL"]["data_dir"]
        == "${BONSAI_PROCESSED_DATA}/daly_care"
    )
    assert fine["cohorts"]["MCL"]["population_file"] == "${BONSAI_COHORT_MEMBERSHIP}"
    assert fine["cohorts"]["SolM"]["cohort_fine_value"] == "SolM"
    assert set(fine["cohorts"]["BL"]["exclude_outcomes"]) == SECOND_LINE_OUTCOMES
    assert (
        set(fine["cohorts"]["AMYLOIDOSIS"]["exclude_outcomes"])
        == SECOND_LINE_OUTCOMES
    )
    assert fine["outcomes"]["sepsis"]["n_hours_end_include"] == 30 * 24
    assert "eligibility_file" not in fine["outcomes"]["sepsis"]

    fine_cif = yaml.safe_load((tmp_path / "fine_ipcw_cif_30d.yaml").read_text())
    assert fine_cif["model_variants"]["opera"]["training_mode"] == "ipcw_cif_bce"
    assert fine_cif["outcomes"]["sepsis"]["n_hours_end_include"] == 30 * 24
    random_overrides = set(
        fine_cif["model_variants"]["no_pretraining"]["extra_overrides"]
    )
    assert {
        "model.value_embedding_mode=film",
        "model.value_bin_vocab_size=0",
        "model.max_seqlen=3372",
        "model.hidden_size=64",
        "model.num_layers=4",
        "model.num_attention_heads=4",
        "model.abspos_encoding=fourier",
    } <= random_overrides

    fine_bce = yaml.safe_load((tmp_path / "fine_bce_90d.yaml").read_text())
    assert fine_bce["finetune_base_config"] == "opera/configs/finetune.yaml"
    assert fine_bce["outcomes"]["sepsis"]["n_hours_end_include"] == 90 * 24
    assert "training_mode" not in fine_bce["model_variants"]["no_pretraining"]

    joint = yaml.safe_load((tmp_path / "joint_opera_full_panel.yaml").read_text())
    mol = yaml.safe_load((tmp_path / "multi_outcome_full_panel.yaml").read_text())
    assert len(joint["outcomes"]) == len(mol["outcomes"]) == 87
    assert joint["training"]["batch_sampling"]["type"] == "random"
    assert joint["training"]["require_dapt_embedding_store"] is True
    assert joint["model"]["competing_event_handling"] == "exclude"
    assert joint["model"]["dapt_anchor_weight"] > 0
    assert joint["model"]["dapt_lambda_floor"] == 1.0
    assert joint["model"]["pooling"] == "mean_last_128"
    assert joint["competing_risk"]["loss_weight"] > 0
    assert joint["competing_risk"]["interval_boundaries_days"] == [
        3,
        7,
        14,
        30,
        60,
        90,
        180,
        365,
        730,
        1460,
    ]
    assert "${BONSAI_" not in yaml.safe_dump(joint)
    assert "${BONSAI_" not in yaml.safe_dump(mol)
    assert set(joint["cohorts"]) == {
        "AMYLOIDOSIS",
        "BL_LBL",
        "CLL_SLL",
        "DLBCL_like",
        "HCL",
        "HL",
        "Indolent_B_NHL",
        "MCL",
        "MM",
        "T_NHL",
    }

    direct = yaml.safe_load((tmp_path / "direct_cr_full_panel.yaml").read_text())
    family = yaml.safe_load((tmp_path / "direct_cr_family_trunks.yaml").read_text())
    curriculum = yaml.safe_load((tmp_path / "direct_cr_curriculum.yaml").read_text())
    combined = yaml.safe_load(
        (tmp_path / "direct_cr_family_trunks_curriculum.yaml").read_text()
    )
    for config in (direct, family, curriculum, combined):
        assert config["competing_risk"]["contrastive_loss_weight"] == 0.0
        assert config["cross_outcome"]["aggregation"] == "hierarchical_support"
        assert set(config["competing_risk"]["no_competing_outcomes"]) == {
            "overall_survival",
            "treatment_failure",
            "treatment_failure_transformation",
        }
    assert direct["competing_risk"]["head_mode"] == "linear"
    assert family["competing_risk"]["head_mode"] == "family_trunks"
    assert curriculum["competing_risk"]["curriculum"]["enabled"] is True
    assert combined["competing_risk"]["head_mode"] == "family_trunks"
    assert combined["competing_risk"]["curriculum"]["enabled"] is True

    uniform_cls = yaml.safe_load(
        (tmp_path / "direct_cr_uniform_cls.yaml").read_text()
    )
    kendall = yaml.safe_load((tmp_path / "direct_cr_kendall.yaml").read_text())
    kendall_null = yaml.safe_load(
        (tmp_path / "direct_cr_kendall_null.yaml").read_text()
    )
    hybrid = yaml.safe_load(
        (tmp_path / "opera_survival_kendall_null.yaml").read_text()
    )
    assert uniform_cls["competing_risk"]["weighter"] == "uniform"
    assert uniform_cls["model"]["pooling"] == "cls_last"
    assert "daly_care_t200_joined_bins" in uniform_cls["dapt_ckpt"]
    assert "daly_care_t200_joined_bins" in uniform_cls["paths"]["vocabulary"]
    assert all(
        "daly_care_t200_joined_bins" in cohort["data_dir"]
        for cohort in uniform_cls["cohorts"].values()
    )
    assert kendall["competing_risk"]["weighter"] == "kendall"
    assert kendall_null["competing_risk"]["weighter"] == "kendall_null"
    assert kendall_null["model"]["pooling"] == "cls_last"
    assert kendall_null["model"]["dapt_anchor_weight"] == 0.0
    assert kendall_null["training"]["require_dapt_embedding_store"] is False
    assert kendall_null["cross_outcome"]["aggregation"] == "macro"
    assert hybrid["competing_risk"]["contrastive_loss_weight"] == 0.1
