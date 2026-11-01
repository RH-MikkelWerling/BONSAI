"""Stage 10 (1 November 2026): J and R_AB with the encoder fine-tuned harder;
each differs from its source fit only in the encoder learning-rate multiplier."""

from test_paper_run_bundle import base_config, reference

LOCK = {"source_mode": "both", "rank": 8, "query_mode": "interacting", "query_width": 512}


def test_stage_ten_copies_j_and_r_ab_with_only_the_encoder_rate_changed():
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_experiments import removal_bundle
    from opera.run.paper_prediction_queue import ROUTED_MODELS, canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle, external_routed
    from pathlib import Path

    args = dict(selected_router=LOCK, router_fix=True, router_own=True, sampling_ipw=True,
                router_pp=True, router_pt=True, router_pf=True, task_pooling=True)
    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, encoder_lr=True)
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    new = {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e for e in after[len(before):]}
    assert set(new) == {"J_ELR10_L", "R_AB_ELR10_L", "R_AB_ELR3_L"}
    sources = {"J_ELR10_L": "seed42/L/J_all_diseases_all_outcomes", "R_AB_ELR10_L": "seed42/L/fix_R_AB",
               "R_AB_ELR3_L": "seed42/L/fix_R_AB"}
    rates = {"J_ELR10_L": 1.0, "R_AB_ELR10_L": 1.0, "R_AB_ELR3_L": 0.3}
    for alias, entry in new.items():
        source = next(e["config"] for e in before if e["name"] == sources[alias])
        config = entry["config"]
        assert config["training"]["encoder_lr_multiplier"] == rates[alias]
        strip = lambda c: {**c, "training": {k: v for k, v in c["training"].items() if k != "encoder_lr_multiplier"}}
        assert strip(config) == strip(source), alias
        assert entry["queue_order"] < min(e.get("queue_order", 0) for e in before)
        validate_direct_cr_config(config)
    assert {"R_AB_ELR10_L", "R_AB_ELR3_L"} <= ROUTED_MODELS
    assert removal_bundle(Path("/r"), "R_AB_ELR10_L") == Path("/r/bundles/stage10")
    assert external_routed({"competing_risk": {}}, "R_AB_ELR10_L") == external_routed({"competing_risk": {}}, "R_AB_L")


def test_contrasts_reach_the_numbers():
    from opera.run.analyse_locked_predictions import ENCODER_LR
    from opera.run.build_paper_numbers import CODES

    assert set(ENCODER_LR) <= set(CODES)


def test_stage_eleven_adds_the_reading_models_at_the_full_encoder_rate():
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle

    args = dict(selected_router=LOCK, router_fix=True, router_own=True, sampling_ipw=True, router_pp=True,
                router_pt=True, router_pf=True, task_pooling=True, encoder_lr=True)
    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, encoder_lr_reading=True)
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    new = {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e for e in after[len(before):]}
    assert set(new) == {"J_FPF_PH_OWN_PH_ELR10_L", "J_OA_OWN_ELR10_L"}
    for alias, source_alias in (("J_FPF_PH_OWN_PH_ELR10_L", "J_FPF_PH_OWN_PH"), ("J_OA_OWN_ELR10_L", "J_OA_OWN")):
        source = next(e["config"] for e in before if e["name"].endswith(f"fix_{source_alias}"))
        config = new[alias]["config"]
        assert config["training"]["encoder_lr_multiplier"] == 1.0
        strip = lambda c: {**c, "training": {k: v for k, v in c["training"].items() if k != "encoder_lr_multiplier"}}
        assert strip(config) == strip(source)
        assert new[alias]["queue_order"] < min(e.get("queue_order", 0) for e in before)
        validate_direct_cr_config(config)
