import numpy as np
import pandas as pd
import pytest

from opera.run.plot_patient_embeddings import (
    load_embedding_frame,
    load_outcome_time_columns,
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


def _outcome_frame():
    index_date = pd.Timestamp("2020-01-01")
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "index_date": [index_date] * 4,
            # subject 1: event, but after the 180d horizon.
            # subject 2: censored early (91d) -- unknown at the 180d horizon.
            # subject 3: censored late (400d), no event -- a real negative.
            # subject 4: event within the 180d horizon.
            "outcome_date": [
                index_date + pd.Timedelta(days=182),
                pd.NaT,
                pd.NaT,
                index_date + pd.Timedelta(days=31),
            ],
            "censor_date": [
                index_date + pd.Timedelta(days=366),
                index_date + pd.Timedelta(days=91),
                index_date + pd.Timedelta(days=400),
                index_date + pd.Timedelta(days=366),
            ],
        }
    )


def test_load_outcome_time_columns_derives_open_ended_time_and_event(tmp_path):
    path = tmp_path / "overall_survival.parquet"
    _outcome_frame().to_parquet(path)

    derived = load_outcome_time_columns(path)

    assert list(derived["subject_id"]) == [1, 2, 3, 4]
    assert derived["overall_survival_event"].tolist() == [True, False, False, True]
    assert derived["overall_survival_time_days"].tolist() == pytest.approx(
        [182.0, 91.0, 400.0, 31.0]
    )


def test_load_outcome_time_columns_horizon_binary_excludes_insufficient_followup(
    tmp_path,
):
    path = tmp_path / "overall_survival.parquet"
    _outcome_frame().to_parquet(path)

    derived = load_outcome_time_columns(
        path, outcome_name="os", horizons_days=(180.0,)
    )

    column = derived.set_index("subject_id")["os_within_180d"]
    assert column.loc[1] == False  # noqa: E712 -- event happened, after the horizon
    assert pd.isna(column.loc[2])  # censored at 91d, unknown at 180d
    assert column.loc[3] == False  # noqa: E712 -- censored at 400d without the event
    assert column.loc[4] == True  # noqa: E712 -- event at 31d, within the horizon


def test_load_outcome_time_columns_rejects_nonpositive_horizon(tmp_path):
    path = tmp_path / "overall_survival.parquet"
    _outcome_frame().to_parquet(path)

    with pytest.raises(ValueError, match="must be positive"):
        load_outcome_time_columns(path, horizons_days=(0.0,))


def test_load_embedding_frame_merges_outcome_derived_columns(tmp_path):
    artifact = tmp_path / "embeddings.npz"
    np.savez(
        artifact,
        subject_ids=np.array([4, 2, 1, 3]),
        embeddings=np.arange(16, dtype=np.float32).reshape(4, 4),
    )
    outcome_path = tmp_path / "overall_survival.parquet"
    _outcome_frame().to_parquet(outcome_path)
    outcome_frame = load_outcome_time_columns(
        outcome_path, horizons_days=(180.0,)
    )

    _, frame = load_embedding_frame(
        artifact, [], subject_col="subject_id", outcome_frames=[outcome_frame]
    )

    assert frame["subject_id"].tolist() == [4, 2, 1, 3]
    assert frame["overall_survival_event"].tolist() == [True, False, True, False]
    assert frame["overall_survival_time_days"].tolist() == pytest.approx(
        [31.0, 91.0, 182.0, 400.0]
    )


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
