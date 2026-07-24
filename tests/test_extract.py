"""Unit tests for opera.functional.extract.build_dapt_embedding_store.

build_dapt_embedding_store is implemented but was never exercised anywhere
in the codebase (no run script, no test) before opera.run.build_dapt_embedding_store
was added. These tests exist to catch signature drift and lock down the
expected {subject_id: tensor} contract the DAPT-prior mechanisms in
opera_nets.py depend on.
"""

from __future__ import annotations

import torch
from torch import nn

from opera.functional.extract import build_dapt_embedding_store


class _FakeDaptModel(nn.Module):
    """Minimal stand-in exposing the get_embeddings contract
    build_dapt_embedding_store actually calls."""

    def __init__(self, embedding_dim: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.calls: list[bool] = []

    def get_embeddings(self, batch, return_pre_projection: bool = False):
        self.calls.append(return_pre_projection)
        codes = batch["code"]
        # Deterministic per-patient embedding derived from the input so
        # different patients produce distinguishable, reproducible vectors.
        base = codes.float().mean(dim=1, keepdim=True)
        return base.expand(-1, self.embedding_dim).clone()


def _fake_dataloader(subject_ids: list[int], batch_size: int = 4):
    batches = []
    for start in range(0, len(subject_ids), batch_size):
        chunk = subject_ids[start : start + batch_size]
        batches.append(
            {
                "code": torch.arange(len(chunk) * 3, dtype=torch.float32).reshape(
                    len(chunk), 3
                )
                + torch.tensor(chunk, dtype=torch.float32).unsqueeze(1),
                "subject_id": torch.tensor(chunk, dtype=torch.long),
            }
        )
    return batches


def test_build_dapt_embedding_store_returns_one_embedding_per_subject():
    model = _FakeDaptModel(embedding_dim=4)
    subject_ids = list(range(10))
    dataloader = _fake_dataloader(subject_ids, batch_size=4)

    store = build_dapt_embedding_store(model, dataloader, device="cpu")

    assert set(store.keys()) == set(subject_ids)
    for sid, emb in store.items():
        assert emb.shape == (4,)
        assert isinstance(sid, int)


def test_build_dapt_embedding_store_uses_pre_projection_embeddings():
    model = _FakeDaptModel(embedding_dim=4)
    dataloader = _fake_dataloader([1, 2, 3])

    build_dapt_embedding_store(model, dataloader, device="cpu")

    assert model.calls and all(model.calls), (
        "build_dapt_embedding_store must request pre-projection embeddings "
        "-- the DAPT-prior mechanisms in opera_nets.py anchor to the pooled "
        "encoder state, not the (untrained, at store-build time) projection "
        "head output."
    )


def test_build_dapt_embedding_store_saves_to_disk(tmp_path):
    model = _FakeDaptModel(embedding_dim=4)
    dataloader = _fake_dataloader([5, 6, 7])
    save_path = tmp_path / "dapt_embeddings.pt"

    build_dapt_embedding_store(
        model, dataloader, device="cpu", save_path=str(save_path)
    )

    assert save_path.exists()
    reloaded = torch.load(save_path, weights_only=False)
    assert set(reloaded.keys()) == {5, 6, 7}
