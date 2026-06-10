import pytest

from opera.run.label_efficiency import (
    aggregate_task_results,
    compute_baseline_delta_tables,
    outcome_split_size_metadata,
    subsample_outcome_parquet,
)


def test_label_efficiency_outputs_pooled_curves_and_delta_tables():
    results = {
        "dlbcl:mortality": {
            "base": {0.1: [0.60, 0.62], 1.0: [0.70, 0.72]},
            "opera": {0.1: [0.66, 0.68], 1.0: [0.78, 0.80]},
        },
        "mm:infection": {
            "base": {0.1: [0.55, 0.57], 1.0: [0.65, 0.67]},
            "opera": {0.1: [0.61, 0.63], 1.0: [0.73, 0.75]},
        },
    }

    nested, task_df, pooled_df = aggregate_task_results(results)
    delta_df, pooled_delta_df = compute_baseline_delta_tables(task_df, "base")

    assert set(nested) == {"dlbcl:mortality", "mm:infection"}
    assert 0.1 in nested["dlbcl:mortality"]["base"]
    assert {"median_auroc", "lower", "upper", "n_tasks"}.issubset(pooled_df.columns)
    assert set(delta_df["model_family"]) == {"opera"}
    assert set(pooled_delta_df["model_family"]) == {"opera"}

    row = pooled_delta_df[pooled_delta_df["training_fraction"] == 0.1].iloc[0]
    assert row["median_delta_auroc"] == pytest.approx(0.06)
    assert row["n_tasks"] == 2


def test_tiny_fraction_subsampling_respects_requested_total(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    train = pd.DataFrame(
        {
            "subject_id": range(10),
            "split": ["train"] * 10,
            "label": [1] + [0] * 9,
        }
    )
    other = pd.DataFrame(
        {
            "subject_id": [100, 101],
            "split": ["tuning", "held_out"],
            "label": [0, 1],
        }
    )
    src = tmp_path / "outcome.parquet"
    dst = tmp_path / "subsampled.parquet"
    pd.concat([train, other]).to_parquet(src)

    subsample_outcome_parquet(str(src), fraction=0.1, seed=7, output_path=str(dst))

    result = pd.read_parquet(dst)
    assert len(result[result["split"] == "train"]) == 1
    assert len(result[result["split"] != "train"]) == 2


@pytest.mark.parametrize(
    ("labels", "fraction", "expected_n"),
    [
        ([1] + [0] * 9, 0.2, 2),
        ([1] + [0] * 9, 0.3, 3),
        ([1, 1, 1, 1, 1, 0], 0.5, 3),
        ([1, 0, 0, 0, 0, 0], 0.5, 3),
    ],
)
def test_tiny_fraction_subsampling_never_exceeds_requested_total(
    tmp_path,
    labels,
    fraction,
    expected_n,
):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    src = tmp_path / "outcome.parquet"
    dst = tmp_path / "subsampled.parquet"
    pd.DataFrame(
        {
            "subject_id": range(len(labels)),
            "split": ["train"] * len(labels),
            "label": labels,
        }
    ).to_parquet(src)

    subsample_outcome_parquet(str(src), fraction=fraction, seed=3, output_path=str(dst))

    result = pd.read_parquet(dst)
    assert len(result[result["split"] == "train"]) == expected_n


def test_n_sample_one_warns_and_falls_back_when_stratification_impossible(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    src = tmp_path / "outcome.parquet"
    dst = tmp_path / "subsampled.parquet"
    pd.DataFrame(
        {
            "subject_id": range(10),
            "split": ["train"] * 10,
            "label": [1] + [0] * 9,
        }
    ).to_parquet(src)

    with pytest.warns(RuntimeWarning, match="Exact stratified subsampling"):
        subsample_outcome_parquet(str(src), fraction=0.1, seed=7, output_path=str(dst))


def test_subsampling_derives_horizon_labels_from_event_time_outcomes(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    src = tmp_path / "outcome.parquet"
    dst = tmp_path / "subsampled.parquet"
    pd.DataFrame(
        {
            "subject_id": range(10),
            "split": ["train"] * 10,
            "index_date": pd.to_datetime(["2020-01-01"] * 10),
            "outcome_date": pd.to_datetime(["2020-01-10", "2020-01-20"] + [None] * 8),
            "censor_date": pd.to_datetime(["2021-01-01"] * 10),
        }
    ).to_parquet(src)

    subsample_outcome_parquet(
        str(src),
        fraction=0.4,
        seed=7,
        output_path=str(dst),
        n_hours_start_include=1,
        n_hours_end_include=24 * 30,
    )

    result = pd.read_parquet(dst)
    assert len(result[result["split"] == "train"]) == 4
    assert "label" not in result.columns


def test_outcome_split_size_metadata_counts_events(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    outcome = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "split": ["train", "train", "tuning", "held_out", "held_out"],
            "label": [1, 0, 1, 0, 1],
        }
    )
    path = tmp_path / "outcome.parquet"
    outcome.to_parquet(path)

    metadata = outcome_split_size_metadata(str(path))

    assert metadata["n_train"] == 2
    assert metadata["n_events_train"] == 1
    assert metadata["n_val"] == 1
    assert metadata["n_test"] == 2
    assert metadata["prevalence_test"] == pytest.approx(0.5)
