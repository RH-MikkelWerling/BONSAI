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


def test_seed_averages_weigh_each_seed_equally():
    from opera.evaluation.secondary_analysis import contrast_coefficients
    from opera.run.analyse_locked_predictions import TASK_SEEDS

    four = contrast_coefficients(TASK_SEEDS["OAOWN4_AB"])
    assert four["J_OA_OWN_s44_L"] == 0.25 and four["R_AB_s43_L"] == -0.5
    assert abs(sum(four.values())) < 1e-12           # a difference, not a level
    assert contrast_coefficients("gain(A,B)") == {"A": 1, "B": -1}   # unchanged
    s42 = contrast_coefficients(TASK_SEEDS["OAOWN4_S42"])
    assert s42["J_OA_OWN_L"] == 0.75 and s42["J_OA_OWN_s43_L"] == -0.25
    import pytest
    with pytest.raises(ValueError):
        contrast_coefficients("gain(A|A,B)")


def test_seed_average_contrast_is_the_mean_of_the_paired_gains():
    import sys
    sys.path.insert(0, "tests")
    from test_secondary_analysis import predictions
    from opera.evaluation.secondary_analysis import normalize_predictions, paired_analysis

    frame, _ = normalize_predictions(predictions())
    result = paired_analysis(frame, {"one": "gain(R_L,S_L)", "two": "gain(J_L,S_L)",
                                     "mean": "gain(R_L|J_L,S_L)"}, replicates=20)["contrasts"]
    gain = result.set_index(["contrast", "metric", "split", "outcome"]).gain.unstack(0).dropna()
    import numpy as np
    np.testing.assert_allclose(gain["mean"], (gain["one"] + gain["two"]) / 2)


def test_cross_seed_contrasts_pair_once_seeds_share_a_key():
    import sys
    sys.path.insert(0, "tests")
    import numpy as np
    from test_secondary_analysis import predictions
    from opera.evaluation.secondary_analysis import normalize_predictions, paired_analysis
    from opera.run.analyse_locked_predictions import collapse_seed_repeats

    frame = predictions()
    frame.loc[frame.model == "R_L", "seed"] = 43            # a seed repeat under its own name
    frame, _ = normalize_predictions(frame)
    contrasts = {"cross": "gain(R_L,J_L)"}
    import pytest
    with pytest.raises(ValueError, match="No requested contrast"):  # never pairs across seed keys
        paired_analysis(frame, contrasts, replicates=10)
    collapsed = collapse_seed_repeats(frame, {})
    after = paired_analysis(collapsed, contrasts, replicates=10)["contrasts"]
    same = paired_analysis(frame.assign(seed=42), contrasts, replicates=10)["contrasts"]
    np.testing.assert_allclose(after.query("contrast == 'cross'").gain, same.query("contrast == 'cross'").gain)
    assert not after.query("contrast == 'cross'").gain.dropna().empty
    both = frame.assign(seed=42)
    both = __import__("pandas").concat([both, both.assign(seed=43)])
    assert collapse_seed_repeats(both, {}).seed.nunique() == 2       # one name, two seeds: untouched


def test_stage_sixteen_repeats_the_runner_up_on_new_seeds():
    from opera.run.paper_prediction_queue import canonical_models
    from opera.run.prepare_paper_run_bundle import build_bundle

    args = dict(ARGS, task_ablation=True, task_design=True, task_seeds=True, ladder_seeds=True)
    before = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args)
    after = build_bundle(base_config(), reference(), {"L": "/x/best.ckpt"}, **args, selection_seeds=True)
    assert [e["config"] for e in after[: len(before)]] == [e["config"] for e in before]
    new = {canonical_models(e, e["config"], {"selected_router": LOCK})[0]: e for e in after[len(before):]}
    assert set(new) == {"J_OA_OWN_PH_s43_L", "J_OA_OWN_PH_s44_L"}
    source = next(e for e in before if e["name"] == "seed42/L/fix_J_OA_OWN_PH")
    for alias, entry in new.items():
        assert _seedless(entry["config"]) == _seedless(source["config"]), alias
    order = [e["name"].rsplit("_", 1)[-1] for e in sorted(after[len(before):], key=lambda e: e["queue_order"])]
    assert order == ["s43", "s44"]
