import pytest


def test_build_eval_frame_uses_labels_from_prediction_file(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from opera.run.evaluate_predictions import build_eval_frame

    outcome_path = tmp_path / "outcome.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "split": ["held_out", "held_out"],
            "index_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "outcome_date": pd.to_datetime([None, "2020-01-10"]),
            "censor_date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
        }
    ).to_parquet(outcome_path)
    predictions = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "probability": [0.2, 0.8],
            "label": [0, 1],
        }
    )

    frame = build_eval_frame(
        predictions=predictions,
        outcome_path=str(outcome_path),
        split="held_out",
        probability_col="probability",
        n_hours_start_include=1,
        n_hours_end_include=8760,
        require_min_followup=True,
    )

    assert frame["label"].tolist() == [0, 1]
    assert frame["probability"].tolist() == [0.2, 0.8]
    assert "time_days" in frame.columns


def test_prediction_input_validation_rejects_one_class_labels():
    np = pytest.importorskip("numpy")

    from opera.run.evaluate_predictions import validate_binary_inputs

    with pytest.raises(ValueError, match="both positive and negative"):
        validate_binary_inputs(np.array([0, 0]), np.array([0.1, 0.2]))


def test_prediction_input_validation_rejects_invalid_probabilities():
    np = pytest.importorskip("numpy")

    from opera.run.evaluate_predictions import validate_binary_inputs

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_binary_inputs(np.array([0, 1]), np.array([0.1, 1.2]))


def test_outcome_window_size_metadata_counts_splits(tmp_path):
    pd = pytest.importorskip("pandas")

    from opera.run.evaluate_predictions import outcome_window_size_metadata

    outcome_path = tmp_path / "outcome.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "split": ["train", "train", "tuning", "held_out"],
            "index_date": pd.to_datetime(["2020-01-01"] * 4),
            "outcome_date": pd.to_datetime(["2020-01-10", None, None, "2020-01-15"]),
            "censor_date": pd.to_datetime(
                ["2021-01-01", "2021-01-01", "2021-01-01", "2021-01-01"]
            ),
        }
    ).to_parquet(outcome_path)

    metadata = outcome_window_size_metadata(
        str(outcome_path),
        n_hours_start_include=1,
        n_hours_end_include=24 * 30,
    )

    assert metadata["n_train"] == 2
    assert metadata["n_events_train"] == 1
    assert metadata["n_val"] == 1
    assert metadata["n_test"] == 1
    assert metadata["n_events_test"] == 1
