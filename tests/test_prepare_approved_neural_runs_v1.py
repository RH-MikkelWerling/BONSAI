from pathlib import Path

import yaml

from opera.run.build_approved_outcome_release_v2 import _registry
from opera.functional.direct_cr_contract import validate_direct_cr_config
from opera.run.generate_sweep_configs import build_endpoint_weighting_pilot
from opera.run.prepare_approved_neural_runs_v1 import VARIANTS, candidate_configs


def test_candidate_architectures_preserve_objective_and_selection() -> None:
    source = yaml.safe_load(Path("opera/configs/experiment_registry.yaml").read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path("opera/configs/outcome_release_primary_v2.yaml").read_text(encoding="utf-8"))
    removed = {
        "AST_increased_g2plus", "AST_increased_g3plus", "dic_like_coagulopathy",
        "herpes_zoster", "hypermagnesemia_g3plus", "hypofibrinogenemia_g2plus",
        "hypofibrinogenemia_g3plus", "hypoglycemia_g2plus", "hypoglycemia_g3plus",
        "hypokalemia_g2plus", "hypokalemia_g3plus", "hypomagnesemia_g3plus",
        "hys_law_dili", "invasive_fungal", "leukocytosis_g2plus",
        "lymphopenia_g2plus", "lymphopenia_g3plus", "MACE_composite",
        "myocardial_infarction", "tumor_lysis_syndrome",
    }
    registry = _registry(source, set(source["outcomes"]) - removed, policy["rename"],
                         policy["add_from_lab_candidates"]
                         + policy["add_from_potassium_candidates"])
    base = build_endpoint_weighting_pilot(registry)
    variants = candidate_configs(base, registry)
    assert set(variants) == set(VARIANTS)
    for name, config in variants.items():
        assert list(config["outcomes"]) == registry["outcomes"]
        if "plain" in name:
            assert config["cross_outcome"]["aggregation"] == "macro"
            assert config["cross_outcome"]["endpoint_groups"] == base["cross_outcome"]["endpoint_groups"]
        else:
            assert config["cross_outcome"] == base["cross_outcome"]
        assert config["training"] == base["training"]
        assert config["competing_risk"]["objective"] == "horizon_bce"
        assert config.get("source_opera_ckpt") is None
        assert config["model"]["freeze_encoder"] is False
    assert variants["fourier_plain_shared_bce"] == variants["time2vec_plain_shared_bce"]
    assert variants["fourier_plain_factorized_bce"] == variants["time2vec_plain_factorized_bce"]
    assert variants["shared"] == variants["time2vec_grouped_shared_bce"]
    assert variants["family_trunks"] == variants["time2vec_family_trunks"]
    assert variants["router_additive_e2e"] == variants["time2vec_router_additive_e2e"]
    assert variants["router_interacting_e2e"] == variants["time2vec_router_interacting_e2e"]
    assert variants["fourier_plain_factorized_bce"]["competing_risk"]["readout_mode"] == (
        "factorized_cohort_outcome_prototypes"
    )
    assert variants["family_trunks"]["competing_risk"]["head_mode"] == "family_trunks"
    assert variants["router_additive_e2e"]["competing_risk"]["adapter_query_mode"] == "additive"
    assert variants["router_interacting_e2e"]["competing_risk"]["adapter_query_mode"] == "interacting"
    assert variants["router_additive_e2e"]["competing_risk"]["readout_mode"] == (
        "source_anchored_adapters"
    )
    for name in ("router_additive_e2e", "router_interacting_e2e",
                 "time2vec_router_additive_e2e", "time2vec_router_interacting_e2e",
                 "time2vec_router_disease_only_e2e", "time2vec_router_outcome_only_e2e",
                 "time2vec_router_no_sources_e2e",
                 "time2vec_router_interacting_scale02_e2e",
                 "time2vec_router_interacting_scale03_e2e"):
        assert variants[name]["adapter_initialization"] == "dapt_fresh"
        validate_direct_cr_config(variants[name])
    full = variants["time2vec_router_interacting_e2e"]
    for name, mode in (
        ("time2vec_router_disease_only_e2e", "disease_only"),
        ("time2vec_router_outcome_only_e2e", "outcome_only"),
        ("time2vec_router_no_sources_e2e", "none"),
    ):
        control = yaml.safe_load(yaml.safe_dump(variants[name]))
        assert control["competing_risk"].pop("adapter_source_mode") == mode
        assert control == full
    for name, scale in (
        ("time2vec_router_interacting_scale02_e2e", 0.2),
        ("time2vec_router_interacting_scale03_e2e", 0.3),
    ):
        control = yaml.safe_load(yaml.safe_dump(variants[name]))
        assert control["competing_risk"].pop("adapter_residual_scale") == scale
        baseline = yaml.safe_load(yaml.safe_dump(full))
        baseline["competing_risk"].pop("adapter_residual_scale")
        assert control == baseline
