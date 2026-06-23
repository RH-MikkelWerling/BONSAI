from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from opera.evaluation.treatment_embeddings import (
    build_atlas_tables,
    compute_disease_treatment_geometry,
    merge_embeddings_with_metadata,
    normalize_regimens,
    probe_treatment_information,
)
from opera.run.treatment_embedding_atlas import _read_table
from opera.visualization.treatment_atlas import (
    plot_disease_treatment_atlas,
    plot_treatment_probe_performance,
    project_embedding_frame,
)


def _synthetic_treatment_data(seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    vectors = []
    subject_id = 0
    for disease_index, disease in enumerate(("DLBCL", "MM")):
        for treatment_index, treatment in enumerate(("regimen_a", "regimen_b")):
            for item in range(40):
                split = "train" if item < 30 else "held_out"
                rows.append(
                    {
                        "subject_id": subject_id,
                        "disease": disease,
                        "first_line_regimen": treatment,
                        "regimen_group": treatment,
                        "split": split,
                    }
                )
                vector = rng.normal(0, 0.25, size=8)
                vector[0] += 3.0 * treatment_index
                vector[1] += 2.0 * disease_index
                vectors.append(vector)
                subject_id += 1
    metadata = pd.DataFrame(rows)
    embeddings = pd.DataFrame(
        vectors,
        columns=[f"embedding_{index}" for index in range(8)],
    )
    embeddings.insert(0, "subject_id", metadata["subject_id"])
    return embeddings, metadata


def teardown_function(_):
    plt.close("all")


def test_normalize_regimens_is_disease_specific():
    metadata = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "disease": ["DLBCL", "DLBCL", "MM", "MM"],
            "treatment": ["R-CHOP", "CHOEP", "VRd", "Unknown"],
        }
    )
    mapping = {
        "DLBCL": {"R-CHOP-like": ["R-CHOP"], "other CHOP-like": ["CHOEP"]},
        "MM": {"VRd-like": ["VRD"]},
    }

    normalized = normalize_regimens(
        metadata,
        mapping,
        treatment_col="treatment",
        keep_unmapped=False,
    )

    assert normalized["regimen_group"].iloc[:3].tolist() == [
        "R-CHOP-like",
        "other CHOP-like",
        "VRd-like",
    ]
    assert pd.isna(normalized["regimen_group"].iloc[3])


def test_disease_conditioned_probe_uses_held_out_split():
    embeddings, metadata = _synthetic_treatment_data()
    merged = merge_embeddings_with_metadata(embeddings, metadata)

    results, predictions = probe_treatment_information(
        merged,
        evaluation_mode="held_out",
        min_disease_n=20,
        min_class_n=5,
    )

    assert set(results["disease"]) == {"DLBCL", "MM"}
    assert (results["evaluation"] == "held_out").all()
    assert (results["n_evaluated"] == 20).all()
    assert (results["balanced_accuracy"] > 0.9).all()
    assert len(predictions) == 40
    assert predictions["correct"].mean() > 0.9


def test_probe_cv_fallback_is_explicitly_exploratory():
    embeddings, metadata = _synthetic_treatment_data()
    merged = merge_embeddings_with_metadata(
        embeddings,
        metadata.drop(columns="split"),
    )

    results, _ = probe_treatment_information(
        merged,
        evaluation_mode="auto",
        min_disease_n=20,
        min_class_n=5,
        n_splits=4,
    )

    assert (results["evaluation"] == "stratified_cv_exploratory").all()


def test_atlas_tables_and_plot_share_coordinates(tmp_path):
    embeddings, metadata = _synthetic_treatment_data()
    merged = merge_embeddings_with_metadata(embeddings, metadata)
    coords = project_embedding_frame(merged, method="pca")
    patient, centroids = build_atlas_tables(merged, coords)

    assert len(patient) == len(merged)
    assert set(centroids["kind"]) == {
        "disease",
        "treatment",
        "disease_treatment",
    }
    assert len(centroids[centroids["kind"] == "disease_treatment"]) == 4

    fig = plot_disease_treatment_atlas(
        patient,
        centroids,
        min_disease_n=5,
        min_treatment_n=5,
        min_joint_n=5,
        save_path=str(tmp_path / "atlas.png"),
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    assert (tmp_path / "atlas.png").exists()
    assert (tmp_path / "atlas.pdf").exists()


def test_probe_performance_plot(tmp_path):
    results = pd.DataFrame(
        {
            "disease": ["DLBCL", "MM", "DLBCL", "MM"],
            "embedding_stage": ["dapt", "dapt", "opera", "opera"],
            "balanced_accuracy": [0.65, 0.70, 0.75, 0.80],
            "majority_balanced_accuracy": [0.5, 0.5, 0.5, 0.5],
        }
    )

    fig = plot_treatment_probe_performance(
        results,
        save_path=str(tmp_path / "probe.png"),
    )

    assert isinstance(fig, Figure)
    assert (tmp_path / "probe.png").exists()


def test_disease_treatment_geometry_uses_high_dimensional_embeddings():
    embeddings, metadata = _synthetic_treatment_data()
    merged = merge_embeddings_with_metadata(embeddings, metadata)

    geometry = compute_disease_treatment_geometry(merged, min_joint_n=5)

    assert not geometry.empty
    assert {
        "source_disease",
        "source_treatment",
        "target_disease",
        "target_treatment",
        "cosine_similarity",
        "neighbor_rank",
    }.issubset(geometry.columns)
    assert (
        geometry.groupby(["source_disease", "source_treatment"])["neighbor_rank"]
        .min()
        .eq(1)
        .all()
    )


def test_npz_embedding_loader(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        subject_ids=np.array([1, 2, 3]),
        embeddings=np.arange(12, dtype=float).reshape(3, 4),
    )

    frame = _read_table(str(path))

    assert frame["subject_id"].tolist() == [1, 2, 3]
    assert [column for column in frame if column.startswith("embedding_")] == [
        "embedding_0",
        "embedding_1",
        "embedding_2",
        "embedding_3",
    ]


def test_pt_embedding_store_loader(tmp_path):
    path = tmp_path / "embeddings.pt"
    torch.save(
        {
            11: torch.tensor([1.0, 2.0, 3.0]),
            12: torch.tensor([4.0, 5.0, 6.0]),
        },
        path,
    )

    frame = _read_table(str(path))

    assert frame["subject_id"].tolist() == [11, 12]
    assert frame.filter(like="embedding_").shape == (2, 3)
