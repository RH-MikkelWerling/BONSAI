import torch
import torch.nn as nn
import torch.nn.functional as F

from opera.modules.networks.opera_nets import OperaContrastiveModel


class _StubEncoder(nn.Module):
    def forward(self, batch):
        return (batch["input_emb"],)


def test_dapt_anchor_defaults_to_production_value():
    model = OperaContrastiveModel(
        encoder=_StubEncoder(),
        outcome_names=["mortality"],
        hidden_size=8,
        projection_hidden_dim=8,
        projection_dim=4,
        outcome_sorted_event_times={"mortality": torch.tensor([10.0, 20.0, 30.0])},
        pooling="cls_last",
    )

    assert model.dapt_anchor_weight == 0.2


def _make_minimal_opera_model(dapt_anchor_weight=0.0, store=None):
    return OperaContrastiveModel(
        encoder=_StubEncoder(),
        outcome_names=["mortality"],
        hidden_size=8,
        projection_hidden_dim=8,
        projection_dim=4,
        outcome_sorted_event_times={"mortality": torch.tensor([10.0, 20.0, 30.0])},
        dapt_anchor_weight=dapt_anchor_weight,
        dapt_embedding_store=store,
        pooling="cls_last",
    )


def test_anchor_loss_is_zero_when_weight_is_zero():
    """Anchor should be a no-op when weight=0 regardless of store content."""
    model = _make_minimal_opera_model(dapt_anchor_weight=0.0)
    pooled = torch.randn(4, 8)
    subject_ids = torch.tensor([1, 2, 3, 4])
    model.dapt_embedding_store = {i: torch.randn(8) for i in range(1, 5)}

    result = model._compute_anchor_loss(pooled, subject_ids)

    assert result is None


def test_anchor_loss_is_non_negative():
    """Cosine anchor loss must always be >= 0."""
    model = _make_minimal_opera_model(dapt_anchor_weight=0.1)
    pooled = torch.randn(4, 8)
    subject_ids = torch.tensor([1, 2, 3, 4])
    model.dapt_embedding_store = {i: torch.randn(8) for i in range(1, 5)}

    result = model._compute_anchor_loss(pooled, subject_ids)

    assert result is not None
    assert result.item() >= 0.0


def test_anchor_loss_is_zero_when_embeddings_match_dapt():
    """If current embeddings equal DAPT embeddings, anchor loss should be ~0."""
    model = _make_minimal_opera_model(dapt_anchor_weight=0.1)
    raw = torch.randn(4, 8)
    pooled = F.normalize(raw, dim=-1)
    subject_ids = torch.tensor([1, 2, 3, 4])
    model.dapt_embedding_store = {i: raw[i - 1].clone() for i in range(1, 5)}

    result = model._compute_anchor_loss(pooled, subject_ids)

    assert result is not None
    assert result.item() < 1e-5


def test_anchor_loss_is_none_when_no_subjects_in_store():
    model = _make_minimal_opera_model(dapt_anchor_weight=0.1)
    pooled = torch.randn(4, 8)
    subject_ids = torch.tensor([99, 100, 101, 102])
    model.dapt_embedding_store = {i: torch.randn(8) for i in range(1, 5)}

    result = model._compute_anchor_loss(pooled, subject_ids)

    assert result is None


def test_forward_includes_anchor_loss_in_log_dict():
    """When anchor_weight > 0 and store populated, anchor_loss should appear."""
    input_emb = torch.randn(4, 3, 8)
    attention_mask = torch.ones(4, 3, dtype=torch.long)
    subject_ids = torch.tensor([1, 2, 3, 4])
    store = {i: input_emb[i - 1, -1].clone() for i in range(1, 5)}
    model = _make_minimal_opera_model(dapt_anchor_weight=0.1, store=store)
    batch = {
        "input_emb": input_emb,
        "attention_mask": attention_mask,
        "subject_id": subject_ids,
    }
    outcome_survival = {
        "mortality": {
            "times": torch.tensor([10.0, 20.0, 30.0, 40.0]),
            "events": torch.tensor([1, 0, 1, 0]),
        }
    }

    result = model(batch, outcome_survival)

    assert "anchor_loss" in result
    assert result["anchor_loss"].item() < 1e-5
    assert result["loss"].requires_grad
