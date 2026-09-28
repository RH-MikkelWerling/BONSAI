import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from opera.run.plot_patient_embeddings import (
    _kernel_probability_surface,
    _ordered_levels,
    categorical_colors,
    discover_outcome_paths,
    load_embedding_frame,
    load_outcome_time_columns,
    plot_outcome_probability_facets,
    reduce_patient_embeddings,
)
from opera.visualization.style import PALETTE


def test_outcome_boolean_colors_and_order_are_prevalence_invariant():
    values = np.array(["True"] * 8 + ["False"] * 2, dtype=object)
    valid = np.ones(len(values), dtype=bool)

    levels = _ordered_levels(values, valid)
    colors = categorical_colors(levels)

    assert levels == ["False", "True"]
    assert colors["False"] == PALETTE["negative"]
    assert colors["True"] == PALETTE["positive"]


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


def test_discover_outcome_paths_excludes_both_audit_spellings(tmp_path):
    for name in (
        "anemia_g3plus.parquet",
        "anemia_g3plus__audit.parquet",
        "infection_audit.parquet",
        "infections_IV_based__ungated.parquet",
    ):
        (tmp_path / name).touch()

    paths = discover_outcome_paths(tmp_path)

    assert [path.name for path in paths] == ["anemia_g3plus.parquet"]


def test_kernel_probability_surface_masks_low_support_and_tracks_signal():
    rng = np.random.default_rng(7)
    negative_x = rng.normal(-1.0, 0.18, 200)
    positive_x = rng.normal(1.0, 0.18, 200)
    x = np.concatenate([negative_x, positive_x])
    y = rng.normal(0.0, 0.18, len(x))
    labels = np.concatenate([np.zeros(200), np.ones(200)])

    _, _, probability = _kernel_probability_surface(
        x, y, labels, bins=50, bandwidth_bins=2.0, min_support=0.02
    )

    left = np.nanmean(probability[:15])
    right = np.nanmean(probability[-15:])
    assert left < 0.2
    assert right > 0.8
    assert np.isnan(probability).any()


def test_outcome_probability_facets_accept_nullable_boolean_labels(tmp_path):
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {
            "component_1": rng.normal(size=90),
            "component_2": rng.normal(size=90),
            "event_within_30d": pd.array(
                [True] * 30 + [False] * 30 + [None] * 30,
                dtype="boolean",
            ),
        }
    )

    paths = plot_outcome_probability_facets(
        frame,
        ["event_within_30d"],
        method="pca",
        output_dir=tmp_path,
        horizon_days=30,
        min_class_size=20,
    )

    assert len(paths) == 1
    assert paths[0].exists()
    assert plt.imread(paths[0]).shape[1] < 1500  # one facet, not four empty columns


def test_kde_facets_can_share_projection_grid_despite_different_known_patients():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    y = np.array([0.0, 1.0, -1.0, 0.5, -0.5])
    labels = np.array([0, 0, 1, 1, 1])
    common = {"bins": 25, "x_range": (-3.0, 3.0), "y_range": (-1.0, 1.0)}
    first_x, first_y, _ = _kernel_probability_surface(x, y, labels, **common)
    second_x, second_y, _ = _kernel_probability_surface(
        x[1:-1], y[1:-1], labels[1:-1], **common
    )
    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)


def test_kde_resolution_preserves_coordinate_smoothing_scale():
    rng = np.random.default_rng(72)
    x, y = rng.normal(size=(2, 3000))
    labels = (rng.random(len(x)) < 1 / (1 + np.exp(-x))).astype(float)
    _, _, coarse = _kernel_probability_surface(x, y, labels, bins=90)
    _, _, fine = _kernel_probability_surface(x, y, labels, bins=450)
    assert fine.shape == (450, 450)
    # Compare the same physical cell centers, allowing histogram discretization.
    sampled = fine[2::5, 2::5]
    valid = np.isfinite(coarse) & np.isfinite(sampled)
    assert valid.sum() > 100
    assert np.mean(np.abs(coarse[valid] - sampled[valid])) < 0.035
