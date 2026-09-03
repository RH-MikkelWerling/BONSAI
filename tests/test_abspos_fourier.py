from datetime import datetime, timezone

import pytest
import torch

from bonsai.modules.networks.components.embeddings import (
    AbsposFourierEncoding,
    EhrEmbeddings,
    Time2Vec,
)


def _epoch_hours(year: int, month: int, day: int = 1) -> float:
    value = datetime(year, month, day, tzinfo=timezone.utc)
    return value.timestamp() / 3600.0


def _monthly_abspos_grid() -> torch.Tensor:
    values = []
    year, month = 2003, 1
    while (year, month) <= (2026, 7):
        values.append(_epoch_hours(year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return torch.tensor(values)


def test_linear_channel_not_saturated():
    output = AbsposFourierEncoding(64)(_monthly_abspos_grid())
    assert output[:, 0].abs().max() < 5
    assert output[:, 0].min() > -1
    assert output[:, 0].max() < 1


def test_output_norm_comparable_to_code_embeddings():
    output = AbsposFourierEncoding(64)(_monthly_abspos_grid())
    ratio = output.norm(dim=-1).mean() / 7.76
    assert 0.2 < ratio < 2.0


@pytest.mark.parametrize("gap_years", [1.0 / 12.0, 1.0, 5.0])
def test_translation_invariance(gap_years):
    encoder = AbsposFourierEncoding(64)
    abspos = _monthly_abspos_grid()
    first = encoder(abspos)[:, 1 : 1 + 2 * encoder.num_pairs]
    second = encoder(abspos + gap_years * 8766.0)[:, 1 : 1 + 2 * encoder.num_pairs]
    similarities = (first * second).sum(dim=-1)
    assert similarities.std() < 1e-4


def test_slowest_period_exceeds_span():
    encoder = AbsposFourierEncoding(64)
    assert encoder.periods.max() > 25


def test_frequencies_are_buffers_not_parameters():
    encoder = AbsposFourierEncoding(64)
    assert "frequencies" in dict(encoder.named_buffers())
    assert "frequencies" not in dict(encoder.named_parameters())


def test_handles_negative_abspos():
    encoder = AbsposFourierEncoding(64)
    birth_abspos = torch.tensor(
        [_epoch_hours(1930, 1), _epoch_hours(1940, 1), _epoch_hours(1950, 1)]
    )
    output = encoder(birth_abspos)
    expected = (
        (birth_abspos - encoder.epoch_2000_hours) / encoder.hours_per_year
        - encoder.linear_ref
    ) / encoder.linear_scale
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, 0], expected)
    assert torch.unique(output[:, 0]).numel() == birth_abspos.numel()
    assert output[:, 0].min() < -1


def test_shape_and_dtype_match_legacy():
    abspos = _monthly_abspos_grid().to(torch.float64).reshape(1, -1)
    legacy = Time2Vec(64, clip_range=100)
    fourier = AbsposFourierEncoding(64)
    assert fourier(abspos).shape == legacy(abspos).shape
    assert fourier(abspos).dtype == legacy(abspos).dtype


def test_scaled_time2vec_matches_abs_pos_branch_thousands_of_hours():
    raw_hours = _monthly_abspos_grid().reshape(1, -1)
    torch.manual_seed(123)
    scaled = Time2Vec(64, clip_range=100, input_scale=1e-3)
    torch.manual_seed(123)
    branch_equivalent = Time2Vec(64, clip_range=100)

    torch.testing.assert_close(
        scaled(raw_hours),
        branch_equivalent(raw_hours / 1000.0),
        rtol=1e-4,
        atol=2e-4,
    )


def test_scaled_time2vec_is_selectable_without_changing_legacy():
    scaled = EhrEmbeddings(
        vocab_size=20,
        hidden_size=8,
        max_seqlen=10,
        abspos_encoding="scaled_time2vec",
    )
    legacy = EhrEmbeddings(
        vocab_size=20,
        hidden_size=8,
        max_seqlen=10,
        abspos_encoding="legacy",
    )

    assert isinstance(scaled.abspos_embedding, Time2Vec)
    assert scaled.abspos_embedding.input_scale == pytest.approx(1e-3)
    assert legacy.abspos_embedding.input_scale == pytest.approx(1.0)


def test_legacy_default_unchanged():
    torch.manual_seed(123)
    default = EhrEmbeddings(vocab_size=20, hidden_size=8, max_seqlen=10)
    torch.manual_seed(123)
    explicit = EhrEmbeddings(
        vocab_size=20,
        hidden_size=8,
        max_seqlen=10,
        abspos_encoding="legacy",
    )
    assert isinstance(default.abspos_embedding, Time2Vec)
    for default_value, explicit_value in zip(
        default.state_dict().values(), explicit.state_dict().values()
    ):
        assert torch.equal(default_value, explicit_value)
    abspos = torch.tensor([[400_000.0, 450_000.0]])
    assert torch.equal(
        default.abspos_embedding(abspos), explicit.abspos_embedding(abspos)
    )


def test_unknown_abspos_encoding_is_rejected():
    with pytest.raises(ValueError, match="Unknown abspos_encoding"):
        EhrEmbeddings(
            vocab_size=20,
            hidden_size=8,
            max_seqlen=10,
            abspos_encoding="mystery",
        )
