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
    assert registry["horizons_days"] == [30, 60, 90, 180, 365, 730]
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

    assert len(paths) == 71
    assert len(sweep_paths) == 38
    shared_bce = yaml.safe_load((tmp_path / "direct_cr_objective_horizon_bce.yaml").read_text())
    factorized_bce = yaml.safe_load((tmp_path / "direct_cr_objective_horizon_bce_factorized_cohort_outcome_prototypes.yaml").read_text())
    anchored = yaml.safe_load((tmp_path / "direct_cr_objective_horizon_bce_source_anchored_adapters.yaml").read_text())
    unanchored = yaml.safe_load((tmp_path / "direct_cr_objective_horizon_bce_unanchored_adapters.yaml").read_text())
    assert shared_bce["competing_risk"]["objective"] == "horizon_bce"
    assert factorized_bce["competing_risk"]["objective"] == "horizon_bce"
    assert factorized_bce["competing_risk"]["readout_mode"] == "factorized_cohort_outcome_prototypes"
    for config in (anchored, unanchored):
        assert config["competing_risk"]["objective"] == "horizon_bce"
        assert len(config["competing_risk"]["fine_names"]) == 24
        assert config["model"]["freeze_encoder"] is True
        assert config["source_opera_ckpt"] == "${oc.env:OPERA_SHARED_BCE_CKPT}"
        assert config["paths"]["sequence_data_dir"] == shared_bce["paths"]["sequence_data_dir"]
    fresh = yaml.safe_load((tmp_path / "direct_cr_bce_router_interacting_dapt_fresh_v1.yaml").read_text())
    assert fresh["adapter_initialization"] == "dapt_fresh"
    assert fresh["source_opera_ckpt"] is None
    assert fresh["model"]["freeze_encoder"] is False
    assert fresh["competing_risk"]["adapter_query_mode"] == "interacting"
    balanced = yaml.safe_load((tmp_path / "direct_cr_bce_lowrank_minority_event_v1.yaml").read_text())
    assert balanced["competing_risk"]["objective"] == "horizon_bce"
    assert balanced["training"]["batch_sampling"]["min_unique_events"] == 50
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
    assert combined["competing_risk"]["interval_boundaries_days"] == [
        30, 60, 90, 180, 365, 730
    ]
    assert combined["competing_risk"]["initialize_from_null_hazards"] is True
    assert combined["model"]["pooling"] == "cls_last"
    assert combined["training"]["eval_monitor_metric"] == (
        "val/cr_cif_auc_family_macro"
    )

    family_kendall_null = yaml.safe_load(
        (tmp_path / "direct_cr_family_trunks_curriculum_kendall_null.yaml").read_text()
    )
    assert family_kendall_null["competing_risk"]["weighter"] == "kendall_null"
    assert family_kendall_null["cross_outcome"]["aggregation"] == "macro"

    architecture_linear = yaml.safe_load(
        (tmp_path / "direct_cr_architecture_shared_linear.yaml").read_text()
    )
    architecture_mlp = yaml.safe_load(
        (tmp_path / "direct_cr_architecture_outcome_mlp.yaml").read_text()
    )
    architecture_family = yaml.safe_load(
        (tmp_path / "direct_cr_architecture_family_trunks.yaml").read_text()
    )
    architecture_attention = yaml.safe_load(
        (tmp_path / "direct_cr_architecture_outcome_attention.yaml").read_text()
    )
    architecture_prototypes = yaml.safe_load(
        (
            tmp_path / "direct_cr_architecture_shared_attention_prototypes.yaml"
        ).read_text()
    )
    architecture_factorized = yaml.safe_load(
        (
            tmp_path
            / "direct_cr_architecture_factorized_cohort_outcome_prototypes.yaml"
        ).read_text()
    )
    horizon_bce = yaml.safe_load(
        (tmp_path / "direct_cr_objective_horizon_bce.yaml").read_text()
    )
    horizon_bce_mlp = yaml.safe_load(
        (tmp_path / "direct_cr_objective_horizon_bce_outcome_mlp.yaml").read_text()
    )
    horizon_bce_attention = yaml.safe_load(
        (
            tmp_path / "direct_cr_objective_horizon_bce_outcome_attention.yaml"
        ).read_text()
    )
    horizon_bce_prototypes = yaml.safe_load(
        (
            tmp_path
            / "direct_cr_objective_horizon_bce_shared_attention_prototypes.yaml"
        ).read_text()
    )
    horizon_bce_factorized = yaml.safe_load(
        (
            tmp_path
            / "direct_cr_objective_horizon_bce_factorized_cohort_outcome_prototypes.yaml"
        ).read_text()
    )
    assert architecture_linear["competing_risk"]["head_mode"] == "linear"
    assert architecture_mlp["competing_risk"]["head_mode"] == "outcome_mlp"
    assert architecture_family["competing_risk"]["head_mode"] == "family_trunks"
    assert architecture_attention["competing_risk"]["head_mode"] == "linear"
    assert (
        architecture_attention["competing_risk"]["readout_mode"]
        == "outcome_attention"
    )
    assert architecture_prototypes["competing_risk"]["head_mode"] == "linear"
    assert (
        architecture_prototypes["competing_risk"]["readout_mode"]
        == "shared_attention_prototypes"
    )
    assert (
        architecture_prototypes["competing_risk"][
            "prototype_attention_num_prototypes"
        ]
        == 10
    )
    assert architecture_factorized["competing_risk"]["head_mode"] == "linear"
    assert (
        architecture_factorized["competing_risk"]["readout_mode"]
        == "factorized_cohort_outcome_prototypes"
    )
    assert (
        architecture_factorized["competing_risk"]["prototype_interaction_rank"]
        == 8
    )
    for architecture in (
        architecture_linear,
        architecture_mlp,
        architecture_family,
        architecture_attention,
        architecture_prototypes,
        architecture_factorized,
        horizon_bce,
        horizon_bce_mlp,
        horizon_bce_attention,
        horizon_bce_prototypes,
        horizon_bce_factorized,
    ):
        assert architecture["seed"] == 42
        assert architecture["model"]["pooling"] == "cls_last"
        assert architecture["paths"]["vocabulary"] == (
            "${paths.sequence_data_dir}/vocabulary.pt"
        )
        assert architecture["paths"]["sequence_data_dir"] == (
            "${oc.env:BONSAI_PROCESSED_DATA}/"
            "daly_care_t100_joined_bins_harmonized_v4"
        )
        assert architecture["dapt_ckpt"] == (
            "${oc.env:BONSAI_MODELS}/"
            "daly_care_t100_joined_bins_harmonized_v4/"
            "pretrain_harmonized_fourier_len8192_allwindows_20ep/"
            "version_0/best.ckpt"
        )
        assert set(
            cohort["data_dir"] for cohort in architecture["cohorts"].values()
        ) == {"${paths.sequence_data_dir}"}
        assert architecture["competing_risk"]["curriculum"]["enabled"] is True
        assert architecture["training"]["evaluation_horizons_days"] == [
            30,
            60,
            90,
            180,
            365,
            730,
        ]
    assert horizon_bce["competing_risk"]["objective"] == "horizon_bce"
    assert horizon_bce["competing_risk"]["head_mode"] == "linear"
    assert horizon_bce["competing_risk"]["readout_mode"] == "shared"
    for combined in (
        horizon_bce_mlp,
        horizon_bce_attention,
        horizon_bce_prototypes,
        horizon_bce_factorized,
    ):
        assert combined["competing_risk"]["objective"] == "horizon_bce"
    assert horizon_bce_mlp["competing_risk"]["head_mode"] == "outcome_mlp"
    assert horizon_bce_mlp["competing_risk"]["readout_mode"] == "shared"
    assert horizon_bce_attention["competing_risk"]["head_mode"] == "linear"
    assert (
        horizon_bce_attention["competing_risk"]["readout_mode"]
        == "outcome_attention"
    )
    assert horizon_bce_prototypes["competing_risk"]["head_mode"] == "linear"
    assert (
        horizon_bce_prototypes["competing_risk"]["readout_mode"]
        == "shared_attention_prototypes"
    )
    assert horizon_bce_factorized["competing_risk"]["head_mode"] == "linear"
    assert (
        horizon_bce_factorized["competing_risk"]["readout_mode"]
        == "factorized_cohort_outcome_prototypes"
    )
    assert (
        horizon_bce_factorized["competing_risk"]["prototype_interaction_rank"]
        == 8
    )

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
