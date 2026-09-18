from datetime import datetime, timezone

import pytest
import torch

from bonsai.modules.networks.components.embeddings import (
    AbsposFourierEncoding,
    EhrEmbeddings,
    Time2Vec,
)
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain


def _epoch_hours(year: int, month: int, day: int = 1) -> float:
    value = datetime(year, month, day, tzinfo=timezone.utc)
    return value.timestamp() / 3600.0


def test_fourier_thousand_hour_input_matches_canonical_hour_input():
    hours = torch.tensor([[_epoch_hours(2003, 1), _epoch_hours(2026, 7)]])
    from_branch_data = AbsposFourierEncoding(64, input_unit="thousand_hours")
    from_canonical_data = AbsposFourierEncoding(64, input_unit="hours")

    torch.testing.assert_close(
        from_branch_data(hours / 1000.0),
        from_canonical_data(hours),
        rtol=1e-5,
        atol=1e-5,
    )


def test_scaled_time2vec_preserves_abs_pos_branch_behavior():
    embeddings = EhrEmbeddings(
        vocab_size=20,
        hidden_size=8,
        max_seqlen=10,
        abspos_encoding="scaled_time2vec",
    )
    assert isinstance(embeddings.abspos_embedding, Time2Vec)


def test_unknown_abspos_encoding_is_rejected():
    with pytest.raises(ValueError, match="Unknown abspos_encoding"):
        EhrEmbeddings(20, 8, 10, abspos_encoding="unknown")


def test_pretrain_hparams_record_abspos_encoding():
    model = BonsaiPretrain(
        vocab_size=20,
        max_seqlen=10,
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        bias=False,
        dropout=0.0,
        attention_dropout=0.0,
        causal=True,
        attn_type="sdpa",
        abspos_encoding="fourier",
    )
    assert model.hparams["abspos_encoding"] == "fourier"
    assert isinstance(model.embeddings.abspos_embedding, AbsposFourierEncoding)
