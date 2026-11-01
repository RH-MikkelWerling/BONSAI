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
