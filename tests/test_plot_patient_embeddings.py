import numpy as np
import pandas as pd

from opera.run.plot_patient_embeddings import (
    load_embedding_frame,
    reduce_patient_embeddings,
)


def test_load_embedding_frame_preserves_npz_order_and_joins_metadata(tmp_path):
    artifact = tmp_path / "embeddings.npz"
    np.savez(
        artifact,
        subject_ids=np.array([3, 1, 2]),
        embeddings=np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"patientid": [1, 2, 3], "split": ["train", "held_out", "tuning"]}
    ).to_csv(metadata, index=False)

    embeddings, frame = load_embedding_frame(
        artifact, [str(metadata)], subject_col="patientid"
    )

    assert embeddings.shape == (3, 4)
    assert frame["subject_id"].tolist() == [3, 1, 2]
    assert frame["split"].tolist() == ["tuning", "train", "held_out"]


def test_pca_patient_projection_is_reproducible():
    embeddings = np.arange(40, dtype=np.float32).reshape(10, 4)
    first, method = reduce_patient_embeddings(
        embeddings,
        method="pca",
        seed=42,
        n_neighbors=5,
        min_dist=0.25,
        perplexity=3,
    )
    second, _ = reduce_patient_embeddings(
        embeddings,
        method="pca",
        seed=42,
        n_neighbors=5,
        min_dist=0.25,
        perplexity=3,
    )

    assert method == "pca"
    assert np.allclose(first, second)
