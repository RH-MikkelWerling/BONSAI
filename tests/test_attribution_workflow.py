import numpy as np
import pandas as pd
import pytest
import torch

from opera.interpretability.token_attribution import integrated_gradients, event_ids
from opera.run.explain_direct_competing_risk import (
    cached_explanation,
    _spearman,
    stratified_sample,
)


def test_linear_integrated_gradients_preserves_signed_credit_and_fixed_context():
    full = torch.tensor([[[3.0, 1.0], [1.0, 4.0], [7.0, 8.0]]])
    content = full.clone()
    baseline = torch.zeros_like(full)
    mask = torch.tensor([[True, True, False]])

    def target(batch):
        values = batch["token_embeddings"]
        return (values[..., 0] * 2 - values[..., 1]).sum(1, keepdim=True)

    attrs, prediction, reference = integrated_gradients(
        target, {}, full, content, baseline, mask, steps=1
    )
    torch.testing.assert_close(attrs[0, 0], torch.tensor([5.0, -2.0, 0.0]))
    torch.testing.assert_close(attrs.sum(-1), prediction - reference)


def test_event_ranking_has_no_artificial_events_at_special_token_timestamps():
    batch = {
        "code": torch.zeros((1, 6), dtype=torch.long),
        "abspos": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
    }
    mask = torch.tensor([[False, True, False, False, True, False]])
    assert event_ids(batch, mask).tolist() == [[-1, 0, -1, -1, 1, -1]]


def test_patient_cache_reuses_work_and_rejects_changed_exports(tmp_path):
    called = []

    def compute():
        called.append(True)
        return {
            "tokens": [pd.DataFrame({"subject_id": [5], "attribution": [0.2]})],
            "patients": [{"subject_id": 5, "prediction": 0.8}],
            "faithfulness": [],
            "sanity": [],
        }

    meta = {"subject_id": 5, "weight": 1.0}
    cached_explanation(tmp_path, 0, meta, compute)
    cached_explanation(tmp_path, 0, meta, compute)
    assert len(called) == 1
    path = tmp_path / "patient_cache_INTERNAL/patient-000000/tokens.parquet"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="export changed"):
        cached_explanation(tmp_path, 0, meta, compute)


def test_rank_checks_handle_ties_and_constant_maps():
    assert np.isnan(_spearman(torch.zeros(4), torch.arange(4)))
    assert _spearman(
        torch.tensor([0.0, 0.0, 1.0, 2.0]), torch.tensor([0.0, 0.0, 2.0, 3.0])
    ) == pytest.approx(1)


def test_sample_cap_is_not_silently_exceeded():
    with pytest.raises(ValueError, match="quotas exceed"):
        stratified_sample(
            [{"subtype": "a"}, {"subtype": "b"}], per_subtype=1, total=1, seed=1
        )


def test_whole_population_is_the_default_with_unit_weights():
    pool = [{"subtype": s} for s in "aabbbc"]
    for total in (0, 6, 100):
        chosen, weights = stratified_sample(pool, per_subtype=1, total=total, seed=1)
        assert sorted(chosen) == list(range(6)) and weights == [1.0] * 6


def test_resume_intent_rejects_data_or_configuration_changes(tmp_path):
    from argparse import Namespace
    from omegaconf import OmegaConf
    from opera.run.explain_direct_competing_risk import verify_intent

    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"weights")
    data = tmp_path / "input.csv"
    data.write_text("id\n1\n")
    output = tmp_path / "out"
    output.mkdir()
    args = Namespace(
        checkpoint=checkpoint,
        output_dir=output,
        resume=False,
        steps=32,
        token_budget=100,
    )
    cfg = OmegaConf.create({"input": str(data)})
    verify_intent(args, cfg, {"a": 0})
    args.resume = True
    args.token_budget = 10  # execution chunk can change without a scientific restart
    verify_intent(args, cfg, {"a": 0})
    data.write_text("id\n2\n")
    with pytest.raises(ValueError, match="inputs/settings changed"):
        verify_intent(args, cfg, {"a": 0})


def test_pilot_summaries_cannot_be_mistaken_for_completed_sample(tmp_path):
    from opera.run.summarise_attributions import run

    (tmp_path / "attribution_run.json").write_text('{"complete":false}')
    with pytest.raises(ValueError, match="pilot is incomplete"):
        run(tmp_path, {}, tmp_path / "summary")


def test_lab_bin_positions_support_zero_and_one_based_vocabularies():
    from opera.interpretability.attribution_summaries import bin_counts, _bin_position

    for start in (0, 1):
        vocab = {f"LABTEST//NPU123//BIN_{i}": i for i in range(start, start + 3)}
        counts = bin_counts(vocab)
        assert _bin_position(f"LABTEST//NPU123//BIN_{start}", counts) == 0
        assert _bin_position(f"LABTEST//NPU123//BIN_{start + 2}", counts) == 1


def test_latest_lab_condition_uses_latest_result_not_latest_matching_low_result():
    from opera.interpretability.attribution_summaries import concept_tokens

    frame = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "outcome": ["death"] * 2,
            "code": ["LABTEST//NPU123//BIN_0", "LABTEST//NPU123//BIN_2"],
            "position": [1, 2],
            "days_before_index": [10, 1],
        }
    )
    concept = {
        "codes": ["LABTEST//NPU123//BIN_"],
        "latest_only": True,
        "bin_position": [0, 0.3],
    }
    assert concept_tokens(frame, concept, {"NPU123": {"min": 0, "max": 2}}).empty


def test_zero_credit_patients_do_not_inflate_population_signed_contributions():
    from opera.interpretability.attribution_summaries import source_window_summary

    tokens = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "outcome": ["a", "a"],
            "code": ["OTHER", "OTHER"],
            "days_before_index": [1.0, 1.0],
            "attribution": [1.0, 0.0],
            "weight": [1.0, 1.0],
        }
    )
    row = source_window_summary(tokens, min_count=1, draws=10).iloc[0]
    assert row.signed_pp == pytest.approx(50.0)
    assert row.n_patients == 2 and row.n_share_patients == 1


def test_dispatcher_exposes_pilot_resume_and_horizon_specific_cpu_commands(tmp_path):
    from test_paper_workflow import args, env
    from opera.run.paper_experiments import build_commands
    import yaml

    settings = tmp_path / "attribution.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "run": {"outcomes": ["death"], "horizons": [365, 730]},
                "summary": {"horizon_days": 365},
            }
        )
    )
    values = env(tmp_path) | {
        "OPERA_PAPER_ATTRIBUTION_CONFIG": str(settings),
        "OPERA_PAPER_ATTRIBUTION_CHECKPOINT": str(tmp_path / "model.ckpt"),
        "OPERA_PAPER_ATTRIBUTION_CONFIG_NAME": "model",
        "OPERA_PAPER_ATTRIBUTION_CONFIG_DIR": str(tmp_path / "configs"),
    }
    command = build_commands(
        args("attributions", attribution_max_new_patients=5), values
    )[0]
    assert "--resume" in command
    assert command[command.index("--max-new-patients") + 1] == "5"
    summary = build_commands(
        args("attribution-summary", attribution_horizon=730), values
    )[0]
    assert summary[summary.index("--output-dir") + 1].endswith("summary_730d")
    figures = build_commands(
        args("attribution-figures", attribution_horizon=730), values
    )[0]
    assert figures[figures.index("--output-dir") + 1].endswith("summary_730d/figures")
    assert values["OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR"].endswith("summary_730d")


def test_input_restrictions_evaluate_the_locked_model_and_compare_every_arm(tmp_path):
    from pathlib import Path

    from test_paper_workflow import args, env
    from opera.run.paper_experiments import build_commands
    import yaml

    settings = tmp_path / "attribution.yaml"
    settings.write_text(yaml.safe_dump({"restrictions": [
        {"id": "timeline", "label": "Timeline only", "content_reference": True},
        {"id": "no_labs", "label": "Without laboratory results", "hide_prefixes": ["LABTEST"]},
        {"id": "last_year", "label": "Last year only", "keep_last_days": 365},
        {"id": "gap_7d", "label": "Last 7 days hidden", "drop_last_days": 7}]}))
    values = env(tmp_path) | {
        "OPERA_PAPER_ATTRIBUTION_CONFIG": str(settings),
        "OPERA_PAPER_ATTRIBUTION_CHECKPOINT": str(tmp_path / "model.ckpt"),
        "OPERA_PAPER_ATTRIBUTION_CONFIG_NAME": "model",
    }
    commands = build_commands(args("input-restrictions"), values)
    assert len(commands) == 6  # full, four restrictions, comparison
    full, timeline, labs, last_year, gap, compare = commands
    assert "--content-reference" not in full
    assert timeline[timeline.index("--content-reference") + 1].endswith(
        "attributions/baseline_INTERNAL.pt")
    assert labs[labs.index("--hide-prefixes") + 1] == "LABTEST"
    assert last_year[last_year.index("--keep-last-days") + 1] == "365"
    assert gap[gap.index("--drop-last-days") + 1] == "7"
    assert compare.count("--restricted") == 4
    completed = Path(gap[gap.index("--output-dir") + 1])
    completed.mkdir(parents=True)
    (completed / "direct_cr_predictions.parquet").touch()
    resumed = build_commands(args("input-restrictions"), values)
    assert len(resumed) == 5
    assert resumed[-1] == compare
    assert compare[compare.index("--full") + 1].endswith("input_restrictions/full")


def test_lab_curves_use_vocabulary_bin_ranges_in_either_format():
    import pandas as pd
    from opera.interpretability.attribution_summaries import lab_response

    tokens = pd.DataFrame({
        "subject_id": [1, 2, 3, 4], "outcome": ["os"] * 4, "position": [1, 1, 1, 1],
        "code": [f"LABTEST//NPU1//BIN_{k}" for k in (0, 2, 4, 4)],
        "days_before_index": [5.0] * 4, "attribution": [0.01, 0.0, -0.01, -0.02],
        "weight": [1.0] * 4})
    test = [{"npu": "NPU1", "label": "Test", "outcome": "os"}]
    zero_based = lab_response(tokens, test, {"NPU1": {"min": 0, "max": 4}}, draws=5,
                              min_count=1).set_index("bin")
    assert zero_based.bin_position.to_dict() == {0: 0.0, 2: 0.5, 4: 1.0}
    assert (zero_based.n_bins == 5).all()
    legacy = lab_response(tokens.assign(code=[f"LABTEST//NPU1//BIN_{k}" for k in (1, 3, 5, 5)]),
                          test, {"NPU1": 5}, draws=5, min_count=1).set_index("bin")
    assert legacy.bin_position.to_dict() == {1: 0.0, 3: 0.5, 5: 1.0}
    observed = lab_response(tokens, test, {}, draws=5, min_count=1).set_index("bin")
    assert observed.bin_position.to_dict() == {0: 0.0, 2: 0.5, 4: 1.0}
