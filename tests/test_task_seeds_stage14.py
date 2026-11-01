"""Stage 14 (1 November 2026): seed spread of the leading reading design and of J;
each fit is its source on another seed and nothing else."""

from test_paper_run_bundle import base_config, reference
from test_task_ablation_stage12 import ARGS
from test_task_pooling_stage9 import LOCK


def _seedless(config):
    """Config with every seed field removed (seeds are the only intended change)."""
    if isinstance(config, dict):
        return {k: _seedless(v) for k, v in config.items() if "seed" not in k}
    if isinstance(config, list):
        return [_seedless(v) for v in config]
    return config


def test_stage_fourteen_repeats_exactly_its_sources_on_new_seeds():
    from opera.functional.direct_cr_contract import validate_direct_cr_config
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle

    args = dict(ARGS, task_ablation=True, task_design=True)
    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, task_seeds=True)
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    new = {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e for e in after[len(before):]}
    assert set(new) == {"J_OA_OWN_s44_L", "J_OA_OWN_s45_L", "J_s43_L"}
    sources = {"J_OA_OWN": next(e for e in before if e["name"] == "seed42/L/fix_J_OA_OWN"),
               "J": next(e for e in before if e["name"] == "seed42/L/J_all_diseases_all_outcomes")}
    for alias, entry in new.items():
        validate_direct_cr_config(entry["config"])
        source = sources[alias.rsplit("_s", 1)[0]]
        assert entry["seed"] == int(alias.rsplit("_s", 1)[1].removesuffix("_L"))
        assert entry["config"] != source["config"]
        assert _seedless(entry["config"]) == _seedless(source["config"]), alias
        # claimed before every other fit still waiting
        assert entry["queue_order"] < min(e.get("queue_order", 0) for e in before)
    assert len({e["config"]["paper_run"]["seed"] for e in new.values()}) == 3


def test_seed_contrasts_reach_the_numbers():
    from opera.run.analyse_locked_predictions import TASK_SEEDS
    from opera.run.build_paper_numbers import CODES

    assert set(TASK_SEEDS) <= set(CODES)


def test_mechanism_ladder_models_are_scored_on_test():
    """Table 2 reads by mechanism; every model in those rows must be a locked
    paper model, so the held-out queue scores it (screening candidates are not)."""
    from opera.run.analyse_locked_predictions import TASK_ABLATION, TASK_POOLING, models_for
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle

    entries = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"},
                           **dict(ARGS, task_ablation=True, task_design=True), task_seeds=True)
    scored = {"J_L"} | {m for e in entries if e["phase"] == "router_fix"   # J_L: supervision fit
                        for m in canonical_models(e, e["config"], {"selected_router": LOCK})}
    ladder = {k: TASK_POOLING[k] for k in ("OFF_J", "OAOWN_OFF", "AB_OFF", "OAOWN_OA")}
    ladder |= {k: TASK_ABLATION[k] for k in ("OAOWN_DIS", "OAOWN_OUT")}
    needed = models_for(ladder) - {"R_L"}
    assert needed <= scored, needed - scored


def test_stage_fifteen_repeats_the_own_corrections_rung_on_seed_43():
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle

    args = dict(ARGS, task_ablation=True, task_design=True, task_seeds=True)
    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, ladder_seeds=True)
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    (new,) = after[len(before):]
    assert canonical_models(new, new["config"], {"selected_router": LOCK}) == ["R_PF_OFF_s43_L"]
    source = next(e for e in before if e["name"] == "seed42/L/fix_R_PF_OFF")
    assert new["seed"] == 43 and new["config"] != source["config"]
    assert _seedless(new["config"]) == _seedless(source["config"])
    assert new["queue_order"] < min(e.get("queue_order", 0) for e in before
                                    if not e["name"].split("/")[-1].startswith(("fix_J_OA_OWN_s", "fix_J_s")))


def test_held_fits_are_parked_and_released(tmp_path):
    from opera.run.paper_worker import HOLD_FITS, todo

    bundle = tmp_path / "paper" / "bundles" / "stage1"
    bundle.mkdir(parents=True)
    rows = [{"name": f"seed42/L/fix_{n}", "phase": "router_fix", "seed": 42, "scientific_hash": n}
            for n in ("R_AB_ELR3", "J_OA_OWN_s44")]
    (bundle / "bundle.json").write_text(__import__("json").dumps({"runs": rows}))
    assert all(gate is None for _, _, gate in todo([bundle], {}))
    (tmp_path / "paper" / HOLD_FITS).write_text("R_AB_ELR3  # cannot change a decision\n")
    gates = {row["name"]: gate for row, _, gate in todo([bundle], {})}
    assert gates == {"seed42/L/fix_R_AB_ELR3": "blocked_held:R_AB_ELR3", "seed42/L/fix_J_OA_OWN_s44": None}
