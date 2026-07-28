"""Tests for the Part B multi-layer/multi-pooling extraction path.

Covers: (1) BonsaiBase.encode's additive output_hidden_states option does not
change default behavior and returns correct per-layer states; (2) the mask-
aware pooling functions in opera/functional/pooled_extraction.py; (3) an
end-to-end run of extract_pooled_variants against a tiny synthetic encoder
and subject pool, checked for schema, subject-id ordering, and exact
agreement between the "final"+"cls" variant and the existing cls_last
convention (bonsai_nets.BonsaiFinetune.get_pooled_representation).
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from bonsai.modules.networks.bonsai_nets import BonsaiBase, BonsaiFinetune
from opera.functional.pooled_extraction import (
    POOLING_NAMES,
    pool_variants,
    predict_token_mask,
    resolve_target_layers,
)
from opera.run.extract_pooled_patient_embeddings import extract_pooled_variants


def _small_config(num_layers: int = 4):
    return {
        "vocab_size": 8,
        "hidden_size": 8,
        "num_layers": num_layers,
        "num_attention_heads": 2,
        "max_seqlen": 32,
        "bias": False,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "causal": True,
        "attn_type": "sdpa",
    }


# ---------------------------------------------------------------------------
# BonsaiBase.encode(output_hidden_states=...)
# ---------------------------------------------------------------------------


def test_encode_default_behavior_is_unchanged():
    torch.manual_seed(0)
    model = BonsaiBase(**_small_config()).eval()
    batch = {
        "code": torch.tensor([[5, 6, 1]]),
        "age": torch.tensor([[40.0, 41.0, 41.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 2.0]]),
        "segment": torch.tensor([[0, 1, 1]]),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    with torch.no_grad():
        plain = model.encode(batch)
        explicit_false = model.encode(batch, output_hidden_states=False)
    assert torch.is_tensor(plain)
    assert torch.allclose(plain, explicit_false)


def test_encode_output_hidden_states_returns_one_tensor_per_layer():
    torch.manual_seed(1)
    num_layers = 4
    model = BonsaiBase(**_small_config(num_layers=num_layers)).eval()
    batch = {
        "code": torch.tensor([[5, 6, 7, 1]]),
        "age": torch.tensor([[40.0, 41.0, 42.0, 42.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 3.0, 3.0]]),
        "segment": torch.tensor([[0, 1, 2, 2]]),
        "attention_mask": torch.tensor([[True, True, True, True]]),
    }
    with torch.no_grad():
        final, hidden_states = model.encode(batch, output_hidden_states=True)

    assert len(hidden_states) == num_layers
    for layer_output in hidden_states:
        assert layer_output.shape == final.shape
    # The final returned tensor is the model's own final LayerNorm applied to
    # the last captured layer's output -- exactly what the existing cls_last
    # extraction reads from.
    with torch.no_grad():
        relaid = model.layernorm(hidden_states[-1])
    assert torch.allclose(final, relaid, atol=1e-6)
    # Intermediate layers are pre-final-layernorm, so they generally differ
    # from a layernormed version of themselves (sanity: not all zero delta).
    assert not torch.allclose(hidden_states[0], hidden_states[-1])


def test_encode_hidden_states_respect_right_padding_under_causal_attention():
    torch.manual_seed(2)
    model = BonsaiBase(**_small_config(num_layers=2)).eval()
    short = {
        "code": torch.tensor([[5, 6, 1]]),
        "age": torch.tensor([[40.0, 41.0, 41.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 2.0]]),
        "segment": torch.tensor([[0, 1, 1]]),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    padded = {
        "code": torch.tensor([[5, 6, 1, 0, 0]]),
        "age": torch.tensor([[40.0, 41.0, 41.0, 0.0, 0.0]]),
        "abspos": torch.tensor([[1.0, 2.0, 2.0, 0.0, 0.0]]),
        "segment": torch.tensor([[0, 1, 1, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, True, False, False]]),
    }
    with torch.no_grad():
        _, short_states = model.encode(short, output_hidden_states=True)
        _, padded_states = model.encode(padded, output_hidden_states=True)

    for short_layer, padded_layer in zip(short_states, padded_states):
        assert torch.allclose(short_layer, padded_layer[:, :3, :], atol=1e-6)


# ---------------------------------------------------------------------------
# pooled_extraction pure functions
# ---------------------------------------------------------------------------


def test_resolve_target_layers_matches_daly_care_depth_config():
    assert resolve_target_layers(4) == {"final": 4, "d75": 3, "d50": 2, "d25": 1}


def test_resolve_target_layers_clamps_shallow_stacks():
    layers = resolve_target_layers(1)
    assert layers == {"final": 1, "d75": 1, "d50": 1, "d25": 1}


def test_predict_token_mask_requires_exactly_one_hit():
    code = torch.tensor([[5, 1, 0], [1, 1, 0]])
    with pytest.raises(ValueError, match="exactly one predict/CLS token"):
        predict_token_mask(code, predict_token_id=1)


def test_pool_variants_cls_mean_last_max_are_correct_on_synthetic_states():
    # One patient, hidden_size=1 for readable arithmetic: content tokens are
    # [10, 20, 30], then an appended predict/CLS token valued 999.
    hidden = torch.tensor([[[10.0], [20.0], [30.0], [999.0]]])
    attention_mask = torch.tensor([[True, True, True, True]])
    predict_mask = torch.tensor([[False, False, False, True]])

    variants = pool_variants(
        {"final": hidden},
        attention_mask=attention_mask,
        predict_mask=predict_mask,
        last_k=2,
    )

    assert variants[("final", "cls")].item() == pytest.approx(999.0)
    assert variants[("final", "mean")].item() == pytest.approx(20.0)
    assert variants[("final", "last")].item() == pytest.approx(30.0)
    assert variants[("final", "max")].item() == pytest.approx(30.0)
    # mean of the last 2 content tokens (20, 30), excluding the CLS token.
    assert variants[("final", "mean_last_128")].item() == pytest.approx(25.0)


def test_pool_variants_rejects_sequence_with_no_content_tokens():
    hidden = torch.tensor([[[5.0]]])
    attention_mask = torch.tensor([[True]])
    predict_mask = torch.tensor([[True]])
    with pytest.raises(ValueError, match="no content tokens"):
        pool_variants(
            {"final": hidden},
            attention_mask=attention_mask,
            predict_mask=predict_mask,
        )


# ---------------------------------------------------------------------------
# End-to-end extract_pooled_variants
# ---------------------------------------------------------------------------


def _write_subject(subject_id, codes, ages, absposes, segments):
    return {
        "subject_id": subject_id,
        "code": torch.tensor(codes, dtype=torch.long),
        "age": torch.tensor(ages, dtype=torch.float32),
        "abspos": torch.tensor(absposes, dtype=torch.float32),
        "segment": torch.tensor(segments, dtype=torch.int64),
    }


def _build_fixture(tmp_path):
    # Every subject starts with one background token (segment 0), matching
    # DATA_FORMAT.md's "same background token count for all patients."
    subjects = {
        1: _write_subject(1, [7, 5, 6], [0.0, 40.0, 41.0], [0.0, 1.0, 2.0], [0, 1, 2]),
        2: _write_subject(2, [7, 5, 6, 5], [0.0, 30.0, 31.0, 32.0], [0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3]),
        3: _write_subject(3, [7, 6], [0.0, 50.0], [0.0, 1.0], [0, 1]),
        4: _write_subject(4, [7, 5, 5, 6], [0.0, 20.0, 21.0, 22.0], [0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3]),
        5: _write_subject(5, [7, 6, 5], [0.0, 60.0, 61.0], [0.0, 1.0, 2.0], [0, 1, 2]),
    }
    physical_a = tmp_path / "subject_data_shard_a.pt"
    physical_b = tmp_path / "subject_data_shard_b.pt"
    torch.save([subjects[1], subjects[2], subjects[3]], physical_a)
    torch.save([subjects[4], subjects[5]], physical_b)

    reference = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "split": ["train", "train", "train", "tuning", "held_out"],
            "censor_abspos": [2.0, 3.0, 1.0, 3.0, 2.0],
        }
    )
    vocabulary = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
        "CODE_A": 5,
        "CODE_B": 6,
        "BACKGROUND//sex_male": 7,
    }
    return {
        "paths": {"shard_a": physical_a, "shard_b": physical_b},
        "reference": reference,
        "vocabulary": vocabulary,
    }


def test_extract_pooled_variants_schema_and_ordering(tmp_path):
    torch.manual_seed(3)
    num_layers = 4
    encoder = BonsaiBase(**_small_config(num_layers=num_layers)).eval()
    fixture = _build_fixture(tmp_path)

    subject_ids, splits, embeddings, stats = extract_pooled_variants(
        encoder,
        reference=fixture["reference"],
        subject_split_paths=fixture["paths"],
        vocabulary=fixture["vocabulary"],
        split_keys={"train": "train", "tuning": "tuning", "held_out": "held_out"},
        max_len=32,
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
        last_k=2,
    )

    # Split order is fixed (train, tuning, held_out); train has 3 subjects
    # (1, 2, 3 in physical-pool order), then tuning (4), then held_out (5).
    assert subject_ids.tolist() == [1, 2, 3, 4, 5]
    assert splits.tolist() == ["train", "train", "train", "tuning", "held_out"]

    expected_layers = {"final": 4, "d75": 3, "d50": 2, "d25": 1}
    assert stats["resolved_layers"] == expected_layers
    assert set(embeddings) == {
        (label, pooling) for label in expected_layers for pooling in POOLING_NAMES
    }
    for key, values in embeddings.items():
        assert values.shape == (5, 8), key


def test_extract_pooled_variants_final_cls_matches_existing_cls_last_convention(
    tmp_path,
):
    torch.manual_seed(4)
    config = _small_config(num_layers=3)
    encoder = BonsaiBase(**config).eval()
    fixture = _build_fixture(tmp_path)

    _, _, embeddings, _ = extract_pooled_variants(
        encoder,
        reference=fixture["reference"],
        subject_split_paths=fixture["paths"],
        vocabulary=fixture["vocabulary"],
        split_keys={"train": "train", "tuning": "tuning", "held_out": "held_out"},
        max_len=32,
        batch_size=5,
        num_workers=0,
        device=torch.device("cpu"),
    )

    # Rebuild the same 5 sequences through BonsaiFinetune.get_pooled_representation
    # (the existing cls_last convention) and confirm exact numerical agreement
    # for the deepest resolved layer.
    finetune_model = BonsaiFinetune(**config, predict_token_id=1).eval()
    finetune_model.load_state_dict(encoder.state_dict(), strict=False)

    fixture_subjects = torch.load(fixture["paths"]["shard_a"], weights_only=False) + torch.load(
        fixture["paths"]["shard_b"], weights_only=False
    )
    from bonsai.functional.censoring import censor_subject
    from bonsai.functional.collate import dynamic_padding

    censor_by_id = dict(
        zip(
            fixture["reference"]["subject_id"],
            fixture["reference"]["censor_abspos"],
        )
    )
    prepared = []
    for subject in fixture_subjects:
        censored = censor_subject(
            subject,
            censor_date_abspos=float(censor_by_id[subject["subject_id"]]),
            predict_token_id=1,
        )
        censored["attention_mask"] = torch.ones(
            len(censored["code"]), dtype=torch.bool
        )
        prepared.append(censored)
    batch = dynamic_padding(prepared)
    with torch.no_grad():
        expected = finetune_model.get_pooled_representation(batch)

    # Reorder `expected` to the (1,2,3,4,5) subject order used by extraction.
    order = [
        int(sid) for sid in fixture["reference"]["subject_id"]
    ]  # 1..5 already in that order
    by_subject = {
        int(sid): row for sid, row in zip(order, expected)
    }
    expected_ordered = torch.stack([by_subject[i] for i in [1, 2, 3, 4, 5]])

    torch.testing.assert_close(
        torch.from_numpy(embeddings[("final", "cls")]),
        expected_ordered,
        atol=1e-5,
        rtol=1e-5,
    )


def test_extract_pooled_variants_reports_timing_and_zero_gpu_memory_on_cpu(tmp_path):
    encoder = BonsaiBase(**_small_config(num_layers=2)).eval()
    fixture = _build_fixture(tmp_path)

    _, _, _, stats = extract_pooled_variants(
        encoder,
        reference=fixture["reference"],
        subject_split_paths=fixture["paths"],
        vocabulary=fixture["vocabulary"],
        split_keys={"train": "train", "tuning": "tuning", "held_out": "held_out"},
        max_len=32,
        batch_size=5,
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert stats["peak_gpu_memory_bytes"] == 0
    assert stats["wall_time_seconds"] >= 0.0
    assert "scaled_dot_product_attention" in stats["attn_cls_skipped_reason"]
