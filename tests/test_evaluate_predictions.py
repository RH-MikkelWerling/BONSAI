import pytest
import json
import sys


def test_build_eval_frame_uses_canonical_outcome_labels_not_prediction_labels(tmp_path):
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
            "label": [1, 0],
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


def test_build_eval_frame_raises_on_missing_or_extra_prediction_ids(tmp_path):
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

    with pytest.raises(ValueError, match=r"missing_n=1 extra_n=1"):
        build_eval_frame(
            predictions=pd.DataFrame(
                {"subject_id": [1, 99], "probability": [0.2, 0.8]}
            ),
            outcome_path=str(outcome_path),
            split="held_out",
            probability_col="probability",
            n_hours_start_include=1,
            n_hours_end_include=8760,
            require_min_followup=True,
        )


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


@pytest.mark.parametrize("model_family", ["cox", "xgboost_aft"])
def test_survival_regime_writes_cindex_and_ipcw_metrics(
    tmp_path, monkeypatch, model_family
):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from opera.run.evaluate_predictions import main

    outcome_path = tmp_path / "outcome.parquet"
    prediction_path = tmp_path / "predictions.csv"
    output_dir = tmp_path / model_family
    subject_ids = list(range(1, 9))
    pd.DataFrame(
        {
            "subject_id": subject_ids,
            "split": ["held_out"] * 8,
            "index_date": pd.to_datetime(["2020-01-01"] * 8),
            "outcome_date": pd.to_datetime(
                [
                    "2020-01-05",
                    "2020-01-10",
                    "2020-01-20",
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            ),
            "censor_date": pd.to_datetime(
                [
                    "2020-03-01",
                    "2020-03-01",
                    "2020-03-01",
                    "2020-02-15",
                    "2020-02-20",
                    "2020-03-01",
                    "2020-03-15",
                    "2020-04-01",
                ]
            ),
        }
    ).to_parquet(outcome_path)
    pd.DataFrame(
        {
            "subject_id": subject_ids,
            "probability": [0.95, 0.85, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1],
        }
    ).to_csv(prediction_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_predictions",
            "--predictions",
            str(prediction_path),
            "--outcome",
            str(outcome_path),
            "--output_dir",
            str(output_dir),
            "--outcome_name",
            "synthetic_lab_outcome",
            "--model_family",
            model_family,
            "--evaluation_regime",
            "survival",
            "--n_hours_end_include",
            str(24 * 30),
            "--n_bootstrap",
            "5",
        ],
    )
    main()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    survival = metrics["survival"]
    assert survival["n_total"] == 8
    assert survival["n_events"] == 3
    assert "concordance_index" in survival
    endpoint = survival["per_horizon"]["30d"]
    assert endpoint["ipcw_auc"] == pytest.approx(1.0)
    assert endpoint["ipcw_brier"] == pytest.approx(0.0796875)
    assert endpoint["n_controls"] == 5
