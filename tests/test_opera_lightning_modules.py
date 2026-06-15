"""CPU-only unit tests for OPERA lightning modules.

These tests exercise construction and a single train/validation step for the
contrastive, joint-finetune, and multi-outcome-learning lightning modules.
They run on tiny random tensors with stub encoders so no real data, GPU, or
pretrained checkpoint is required.

The lightning modules call ``self.log(...)`` inside their steps. When a module
is not attached to a ``Trainer`` (as here), Lightning treats ``self.log`` as a
no-op that only emits a warning, so the steps can be invoked directly.
"""

import torch
import torch.nn as nn

from opera.modules.lightningmodules.OperaContrastiveModule import (
    OperaContrastiveModule,
)
from opera.modules.lightningmodules.JointFinetuneModule import JointFinetuneModule
from opera.modules.lightningmodules.MOLModule import MOLModule
from opera.modules.networks.opera_nets import OperaContrastiveModel
from opera.modules.networks.joint_finetune_net import JointFinetuneModel
from opera.modules.networks.mol_net import MultiOutcomeModel


HIDDEN = 8
SEQ_LEN = 3
BATCH = 4
OUTCOMES = ["mortality", "relapse"]


class _StubConfig:
    """Minimal encoder config exposing ``hidden_size`` and ``to_dict``.

    ``attach_model_config`` serialises ``encoder.config.to_dict()`` into the
    lightning module's hparams, and ``JointFinetuneModule`` reads
    ``encoder.config.hidden_size`` directly, so both must be present.
    """

    def __init__(self, hidden_size: int = HIDDEN):
        self.hidden_size = hidden_size

    def to_dict(self) -> dict:
        return {"hidden_size": self.hidden_size}


class _StubEncoder(nn.Module):
    """Minimal BonsaiEncoder stand-in.

    Returns the precomputed per-token embeddings carried in the batch under
    ``input_emb`` as the first element of a tuple, matching the
    ``outputs[0]`` access pattern of the real encoder. A trivial trainable
    ``nn.Linear(HIDDEN, HIDDEN)`` is registered (without altering the embeddings)
    so the encoder exposes ``encoder.<param>`` named parameters and the
    optimizer param-group split has something to act on.
    """

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.config = _StubConfig(hidden)

    def forward(self, batch):
        emb = batch["input_emb"]
        # Touch the parameter so it participates in the graph without changing
        # the (controlled) embedding values used by the contrastive geometry.
        return (emb + 0.0 * self.proj(emb),)


def _batch():
    """Tiny batch of BATCH subjects, SEQ_LEN tokens, HIDDEN-dim embeddings."""
    torch.manual_seed(0)
    input_emb = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    return {
        "input_emb": input_emb,
        "attention_mask": attention_mask,
        "subject_id": torch.arange(1, BATCH + 1),
        "code": torch.zeros(BATCH, SEQ_LEN, dtype=torch.long),
    }


# ── OperaContrastiveModule ──────────────────────────────────────────────────


def _make_opera_model(dapt_anchor_weight: float = 0.0, store=None):
    return OperaContrastiveModel(
        encoder=_StubEncoder(),
        outcome_names=OUTCOMES,
        hidden_size=HIDDEN,
        projection_hidden_dim=HIDDEN,
        projection_dim=4,
        outcome_sorted_event_times={
            name: torch.tensor([10.0, 20.0, 30.0]) for name in OUTCOMES
        },
        dapt_anchor_weight=dapt_anchor_weight,
        dapt_embedding_store=store,
        pooling="cls_last",
    )


def _opera_survival_batch():
    batch = _batch()
    batch["time_mortality"] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    batch["event_mortality"] = torch.tensor([1, 0, 1, 0])
    batch["time_relapse"] = torch.tensor([15.0, 25.0, 35.0, 45.0])
    batch["event_relapse"] = torch.tensor([0, 1, 0, 1])
    batch["outcome_mortality"] = torch.tensor([1, 0, 1, 0])
    batch["outcome_relapse"] = torch.tensor([0, 1, 0, 1])
    return batch


def test_opera_contrastive_module_imports():
    module = OperaContrastiveModule(
        model=_make_opera_model(),
        outcome_names=OUTCOMES,
    )
    assert module.outcome_names == OUTCOMES
    assert isinstance(module.model, OperaContrastiveModel)


def test_opera_contrastive_loss_shape():
    module = OperaContrastiveModule(
        model=_make_opera_model(),
        outcome_names=OUTCOMES,
    )
    loss = module.training_step(_opera_survival_batch(), 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_opera_contrastive_requires_dapt_ckpt_or_none():
    # Constructing with no DAPT store (dapt_ckpt=None equivalent) and an
    # explicit dapt_anchor_weight of zero must not crash.
    module = OperaContrastiveModule(
        model=_make_opera_model(dapt_anchor_weight=0.0, store=None),
        outcome_names=OUTCOMES,
        dapt_anchor_weight=0.0,
    )
    assert module.dapt_anchor_weight == 0.0
    loss = module.training_step(_opera_survival_batch(), 0)
    assert torch.isfinite(loss)


# ── JointFinetuneModule ─────────────────────────────────────────────────────


def _make_joint_model():
    return JointFinetuneModel(
        encoder=_StubEncoder(),
        outcome_names=OUTCOMES,
        hidden_size=HIDDEN,
        pooling="cls_last",
        freeze_encoder=False,
        dropout=0.1,
    )


def _joint_batch(mask_relapse: bool = False):
    batch = _batch()
    batch["outcome_mortality"] = torch.tensor([1, 0, 1, 0])
    if mask_relapse:
        batch["outcome_relapse"] = torch.tensor([-1, -1, -1, -1])
    else:
        batch["outcome_relapse"] = torch.tensor([0, 1, 0, 1])
    return batch


def test_joint_finetune_module_imports():
    module = JointFinetuneModule(
        model=_make_joint_model(),
        outcome_names=OUTCOMES,
    )
    assert module.outcome_names == OUTCOMES
    assert set(module.val_auroc.keys()) == set(OUTCOMES)


def test_joint_finetune_validation_step_returns_metrics():
    module = JointFinetuneModule(
        model=_make_joint_model(),
        outcome_names=OUTCOMES,
    )
    loss = module.validation_step(_joint_batch(), 0)
    assert torch.isfinite(loss)
    # The per-outcome AUROC metric objects should have accumulated state and
    # exist under the expected keys (the dict surfaced by validation logging).
    assert "mortality" in module.val_auroc
    assert module.val_auroc["mortality"].update_called


def test_joint_finetune_multi_outcome_masking():
    module = JointFinetuneModule(
        model=_make_joint_model(),
        outcome_names=OUTCOMES,
    )
    full_loss = module.training_step(_joint_batch(mask_relapse=False), 0)

    masked_module = JointFinetuneModule(
        model=_make_joint_model(),
        outcome_names=OUTCOMES,
    )
    # Copy weights so the only difference is the masked outcome.
    masked_module.model.load_state_dict(module.model.state_dict())
    masked_loss = masked_module.training_step(_joint_batch(mask_relapse=True), 0)

    # The fully labelled batch contributes two outcomes to the loss; masking
    # one outcome (all labels = -1) drops its contribution, so the losses differ.
    assert torch.isfinite(full_loss)
    assert torch.isfinite(masked_loss)
    assert not torch.isclose(full_loss, masked_loss)


# ── MOLModule ───────────────────────────────────────────────────────────────


def _make_mol_model():
    return MultiOutcomeModel(
        encoder=_StubEncoder(),
        outcome_names=OUTCOMES,
        hidden_size=HIDDEN,
        head_hidden_dim=HIDDEN,
        head_dropout=0.0,
        freeze_encoder=False,
        pooling="cls_last",
        weighting="equal",
    )


def _mol_batch():
    batch = _batch()
    batch["outcome_mortality"] = torch.tensor([1, 0, 1, 0])
    batch["outcome_relapse"] = torch.tensor([0, 1, 0, 1])
    return batch


def test_mol_module_imports():
    module = MOLModule(
        model=_make_mol_model(),
        outcome_names=OUTCOMES,
    )
    assert module.outcome_names == OUTCOMES
    assert set(module.val_aurocs.keys()) == set(OUTCOMES)


def test_mol_module_training_step_does_not_crash():
    module = MOLModule(
        model=_make_mol_model(),
        outcome_names=OUTCOMES,
    )
    loss = module.training_step(_mol_batch(), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
