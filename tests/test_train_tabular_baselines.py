import pandas as pd

from opera.run.train_tabular_baselines import (
    TUNING_GRIDS,
    build_arg_parser,
    high_dimensional_diagnostics,
    infer_feature_columns,
    missingness_report,
    outcome_labels,
    parse_columns,
    select_tabpfn_feature_columns,
    train_one_model,
    tune_estimator_params,
    validate_feature_matrix,
)


def test_tabular_tuning_is_default_but_can_be_explicitly_disabled():
    parser = build_arg_parser()
    required = [
        "--features",
        "features.parquet",
        "--outcome",
        "outcome.parquet",
        "--output_dir",
        "results",
        "--cohort",
        "dlbcl",
        "--outcome_name",
        "mortality",
    ]
    assert parser.parse_args(required).tune is True
    assert parser.parse_args([*required, "--no-tune"]).tune is False


def test_high_dimensional_diagnostics_flags_wide_training_cells():
    train = pd.DataFrame({"label": [0, 1] * 5})
    report = high_dimensional_diagnostics(train, [f"x{i}" for i in range(50)])
    assert report["low_n_high_p"] is True
    assert report["severe_low_n_high_p"] is True
    assert report["features_per_row"] == 5.0


def test_tuner_uses_real_preprocessing_for_missing_and_categorical_features():
    train = pd.DataFrame(
        {
            "numeric": [0.0, 1.0, None, 2.0] * 5,
            "category": ["a", "b", "a", None] * 5,
            "label": [0, 1, 0, 1] * 5,
        }
    )
    params = tune_estimator_params(
        "logistic",
        train,
        train.copy(),
        ["numeric", "category"],
        categorical_columns=["category"],
    )
    assert params in TUNING_GRIDS["logistic"]


def test_infer_feature_columns_excludes_reserved_and_explicit_columns():
    features = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "age": [60, 70],
            "sex": ["F", "M"],
            "label": [0, 1],
            "leaky_registry_field": [1.0, 2.0],
        }
    )

    columns = infer_feature_columns(
        features,
        exclude_columns=parse_columns("leaky_registry_field"),
    )

    assert columns == ["age", "sex"]


def test_missingness_report_tracks_train_and_test_availability():
    features = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "age": [60.0, None, 80.0],
            "lab": [None, 2.0, None],
        }
    )

    report = missingness_report(
        features,
        ["age", "lab"],
        train_subject_ids=[1, 2],
        test_subject_ids=[3],
    )

    age = report[report["feature"] == "age"].iloc[0]
    assert age["missing_fraction_all"] == 1 / 3
    assert age["missing_fraction_train"] == 0.5
    assert age["missing_fraction_test"] == 0.0


def test_tabpfn_feature_selector_prefers_observed_features():
    train = pd.DataFrame(
        {
            "dense": [1.0, 2.0, 3.0, 4.0],
            "sparse": [None, None, 1.0, None],
            "medium": [1.0, None, 2.0, 3.0],
        }
    )

    selected = select_tabpfn_feature_columns(
        train,
        ["dense", "sparse", "medium"],
        categorical_columns=None,
        max_features=2,
    )

    assert selected == ["dense", "medium"]


def test_validate_feature_matrix_rejects_duplicate_subjects_and_all_missing():
    features = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "age": [60.0, None, 70.0],
            "empty_lab": [None, None, None],
        }
    )

    report = validate_feature_matrix(features, ["age", "empty_lab"])

    assert not report["ok"]
    assert any("duplicate subject_id" in item for item in report["errors"])
    assert any("All-missing features" in item for item in report["errors"])


def test_validate_feature_matrix_warns_for_constant_features_when_allowed():
    features = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "age": [60.0, 60.0, 60.0],
            "empty_lab": [None, None, None],
        }
    )

    report = validate_feature_matrix(
        features,
        ["age", "empty_lab"],
        allow_all_missing_features=True,
    )

    assert report["ok"]
    assert report["n_all_missing_features"] == 1
    assert any("Constant or single-level" in item for item in report["warnings"])


def test_logistic_ipcw_bce_accepts_sample_weights_and_writes_probabilities():
    train = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "age": [60.0, 70.0, 80.0, 90.0],
            "label": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "subject_id": [5, 6],
            "age": [65.0, 85.0],
            "label": [0, 1],
        }
    )

    predictions, _ = train_one_model(
        model_name="logistic_ipcw_bce",
        train_df=train,
        test_df=test,
        feature_columns=["age"],
        categorical_columns=None,
        seed=42,
        sample_weight=pd.Series([1.0, 2.0, 1.0, 2.0]).to_numpy(),
    )

    assert predictions.columns.tolist() == ["subject_id", "probability"]
    assert predictions["probability"].between(0, 1).all()


def test_plain_fixed_labels_drop_early_censoring_while_survival_keeps_it(tmp_path):
    outcome_path = tmp_path / "outcome.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "split": ["train"] * 3,
            "index_date": pd.to_datetime(["2020-01-01"] * 3),
            "outcome_date": pd.to_datetime(["2020-01-10", None, None]),
            "censor_date": pd.to_datetime(["2020-03-01", "2020-03-01", "2020-01-15"]),
        }
    ).to_parquet(outcome_path)

    fixed = outcome_labels(
        str(outcome_path),
        split="train",
        n_hours_start_include=1,
        n_hours_end_include=24 * 30,
        require_min_followup=True,
    )
    survival = outcome_labels(
        str(outcome_path),
        split="train",
        n_hours_start_include=1,
        n_hours_end_include=24 * 30,
        require_min_followup=False,
        include_survival_fields=True,
    )

    assert set(fixed["subject_id"]) == {1, 2}
    assert set(survival["subject_id"]) == {1, 2, 3}
