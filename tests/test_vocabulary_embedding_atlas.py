from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from matplotlib.figure import Figure

from opera.evaluation.vocabulary_embeddings import (
    compute_token_movement,
    extract_vocabulary_embedding_frame,
    nearest_token_neighbors,
)
from opera.run.vocabulary_embedding_atlas import main as vocabulary_atlas_main
from opera.visualization.vocabulary_atlas import (
    plot_token_movement,
    plot_vocabulary_embedding_atlas,
    project_vocabulary_embeddings,
)


def _write_vocab(path):
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "LPR3//DC833": 3,
        "RKKP//ann_arbor_III": 4,
        "pathology//M9680": 5,
    }
    torch.save(vocabulary, path)
    return vocabulary


def _write_checkpoint(path, key: str, offset: float = 0.0):
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4) + offset
    torch.save({"state_dict": {key: values}}, path)


def teardown_function(_):
    plt.close("all")


def test_extract_vocabulary_embeddings_supports_bonsai_and_opera_keys(tmp_path):
    vocab_path = tmp_path / "vocabulary.pt"
    _write_vocab(vocab_path)
    bonsai_path = tmp_path / "bonsai.ckpt"
    opera_path = tmp_path / "opera.ckpt"
    _write_checkpoint(bonsai_path, "model.embeddings.code_embedding.weight")
    _write_checkpoint(opera_path, "model.encoder.embeddings.code_embedding.weight")

    bonsai = extract_vocabulary_embedding_frame(
        bonsai_path,
        vocab_path,
        stage="pretrain",
    )
    opera = extract_vocabulary_embedding_frame(opera_path, vocab_path, stage="opera")

    assert bonsai["embedding_stage"].unique().tolist() == ["pretrain"]
    assert opera["embedding_stage"].unique().tolist() == ["opera"]
    assert bonsai["token"].tolist() == [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "LPR3//DC833",
        "RKKP//ann_arbor_III",
        "pathology//M9680",
    ]
    assert bonsai.loc[bonsai["token"] == "[PAD]", "token_family"].item() == "special"
    assert (
        bonsai.loc[bonsai["token"] == "LPR3//DC833", "token_family"].item() == "LPR3"
    )
    assert bonsai.filter(like="embedding_").shape[1] >= 4


def test_vocabulary_projection_and_plot(tmp_path):
    vocab_path = tmp_path / "vocabulary.pt"
    _write_vocab(vocab_path)
    checkpoint = tmp_path / "opera.ckpt"
    _write_checkpoint(checkpoint, "model.encoder.embeddings.code_embedding.weight")
    frame = extract_vocabulary_embedding_frame(checkpoint, vocab_path, stage="opera")

    coordinates = project_vocabulary_embeddings(frame, method="pca")
    fig = plot_vocabulary_embedding_atlas(
        coordinates,
        min_group_n=1,
        highlight_tokens=["LPR3//DC833"],
        save_path=str(tmp_path / "vocab_atlas.png"),
    )

    assert isinstance(fig, Figure)
    assert {"atlas_x", "atlas_y", "token", "token_family"}.issubset(
        coordinates.columns
    )
    assert (tmp_path / "vocab_atlas.png").exists()
    assert (tmp_path / "vocab_atlas.pdf").exists()


def test_token_movement_and_neighbors(tmp_path):
    vocab_path = tmp_path / "vocabulary.pt"
    _write_vocab(vocab_path)
    pretrain_path = tmp_path / "pretrain.ckpt"
    dapt_path = tmp_path / "dapt.ckpt"
    _write_checkpoint(pretrain_path, "model.embeddings.code_embedding.weight")
    _write_checkpoint(dapt_path, "model.encoder.embeddings.code_embedding.weight", 1.0)
    pretrain = extract_vocabulary_embedding_frame(
        pretrain_path,
        vocab_path,
        stage="pretrain",
    )
    dapt = extract_vocabulary_embedding_frame(dapt_path, vocab_path, stage="dapt")

    movement = compute_token_movement(
        {"pretrain": pretrain, "dapt": dapt},
        reference_stage="pretrain",
    )
    neighbors = nearest_token_neighbors(dapt, ["LPR3//DC833"], top_k=2)
    fig = plot_token_movement(movement, top_n=3, save_path=str(tmp_path / "move.png"))

    assert not movement.empty
    assert set(movement["embedding_stage"]) == {"dapt"}
    assert {"cosine_distance", "l2_distance"}.issubset(movement.columns)
    assert neighbors["query_token"].unique().tolist() == ["LPR3//DC833"]
    assert len(neighbors) == 2
    assert isinstance(fig, Figure)
    assert (tmp_path / "move.png").exists()


def test_vocabulary_embedding_atlas_cli_writes_ready_outputs(tmp_path, monkeypatch):
    vocab_path = tmp_path / "vocabulary.pt"
    _write_vocab(vocab_path)
    pretrain_path = tmp_path / "pretrain.ckpt"
    opera_path = tmp_path / "opera.ckpt"
    _write_checkpoint(pretrain_path, "model.embeddings.code_embedding.weight")
    _write_checkpoint(opera_path, "model.encoder.embeddings.code_embedding.weight", 0.5)
    metadata = pd.DataFrame(
        {
            "token": ["LPR3//DC833", "RKKP//ann_arbor_III"],
            "curated_family": ["diagnosis", "quality_registry"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    output_dir = tmp_path / "atlas"

    monkeypatch.setattr(
        "sys.argv",
        [
            "vocabulary_embedding_atlas",
            "--checkpoint",
            f"pretrain={pretrain_path}",
            "--checkpoint",
            f"opera={opera_path}",
            "--vocabulary",
            str(vocab_path),
            "--metadata",
            str(metadata_path),
            "--projection",
            "pca",
            "--min_group_n",
            "1",
            "--highlight_tokens",
            "LPR3//DC833",
            "--neighbor_tokens",
            "LPR3//DC833",
            "--output_dir",
            str(output_dir),
        ],
    )

    vocabulary_atlas_main()

    expected = {
        "analysis_metadata.json",
        "vocabulary_embeddings_pretrain.csv",
        "vocabulary_embeddings_opera.csv",
        "vocabulary_atlas_coordinates.csv",
        "vocabulary_atlas.png",
        "vocabulary_atlas.pdf",
        "vocabulary_neighbors.csv",
        "token_movement.csv",
        "token_movement.png",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    with open(output_dir / "analysis_metadata.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["atlas_stage"] == "opera"
    coordinates = pd.read_csv(output_dir / "vocabulary_atlas_coordinates.csv")
    assert "curated_family" in coordinates.columns


def test_vocabulary_size_mismatch_is_rejected(tmp_path):
    vocab_path = tmp_path / "vocabulary.pt"
    _write_vocab(vocab_path)
    checkpoint = tmp_path / "bad.ckpt"
    values = torch.zeros((5, 4), dtype=torch.float32)
    torch.save({"state_dict": {"model.embeddings.code_embedding.weight": values}}, checkpoint)

    with pytest.raises(ValueError, match="Vocabulary size"):
        extract_vocabulary_embedding_frame(checkpoint, vocab_path)
