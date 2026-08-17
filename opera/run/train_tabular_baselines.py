"""Train tabular baseline models from a locked cohort feature matrix.

The runner turns one feature matrix plus one outcome parquet into standard
prediction CSVs (`subject_id`, `probability`) that can be consumed by
`opera.run.evaluate_predictions` and the sweep. It is deliberately separate
from OPERA neural code so tabular baselines are easy to audit and rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from opera.evaluation.cohorts import (
    FIXED_HORIZON_REGIME,
    SURVIVAL_REGIME,
    build_evaluation_cohorts,
    population_subject_ids,
)
from opera.evaluation.comparison import fit_cox_survival, fit_xgboost_aft
from opera.functional.ipcw import compute_ipcw_train_weights


RESERVED_COLUMNS = {
    "subject_id",
    "split",
    "label",
    "target",
    "probability",
    "time_days",
    "event",
}
SURVIVAL_MODELS = frozenset({"cox", "xgboost_aft"})

# Validation-set tuning grids for binary classifiers.
# Each entry is a list of parameter dicts to evaluate on the validation split.
# Kept deliberately small so tuning adds little wall-clock overhead.
TUNING_GRIDS: dict[str, list[dict]] = {
    # Sparse/elastic-net solutions are substantially more stable than an
    # effectively unregularized fit when the encoded feature count exceeds n.
    "logistic": [
        {"C": c, "l1_ratio": ratio}
        for c in (0.001, 0.01, 0.1, 1.0)
        for ratio in (0.0, 0.5, 1.0)
    ],
    "xgboost": [
        {
            "learning_rate": 0.03,
            "max_depth": depth,
            "min_child_weight": child,
            "reg_alpha": alpha,
            "reg_lambda": reg_lambda,
            "colsample_bytree": colsample,
        }
        for depth, child, alpha, reg_lambda, colsample in (
            (2, 2, 0.5, 5.0, 0.5),
            (1, 10, 0.0, 10.0, 0.3),
            (2, 10, 0.5, 10.0, 0.3),
            (2, 5, 1.0, 10.0, 0.5),
            (3, 10, 1.0, 20.0, 0.3),
        )
    ],
}


def validate_feature_matrix(
    features: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: Optional[list[str]] = None,
    max_missing_fraction: Optional[float] = None,
    allow_all_missing_features: bool = False,
) -> dict:
    """Validate the tabular feature matrix contract before model training.

    Inputs are the full feature matrix and selected model columns. The returned
    dictionary contains errors, warnings, and summary counts suitable for JSON
    logging. Scientifically, this protects tabular baselines from leakage-prone
    reserved columns, duplicate patients, unusable all-missing predictors, and
    accidental train/test matrix drift before results enter the shared
    evaluation pipeline.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if "subject_id" not in features.columns:
        errors.append("Feature matrix must contain subject_id.")
    elif features["subject_id"].duplicated().any():
        n_dupes = int(features["subject_id"].duplicated().sum())
        errors.append(f"Feature matrix contains {n_dupes} duplicate subject_id rows.")

    unknown = [col for col in feature_columns if col not in features.columns]
    if unknown:
        errors.append(f"Selected feature columns are missing from matrix: {unknown}")

    reserved_used = sorted(set(feature_columns) & RESERVED_COLUMNS)
    if reserved_used:
        errors.append(f"Reserved columns cannot be used as features: {reserved_used}")

    if categorical_columns:
        missing_categorical = [
            col for col in categorical_columns if col not in features.columns
        ]
        if missing_categorical:
            errors.append(
                f"Categorical columns are missing from matrix: {missing_categorical}"
            )
        not_selected = [
            col for col in categorical_columns if col not in feature_columns
        ]
        if not_selected:
            warnings.append(
                f"Categorical columns not selected as features: {not_selected}"
            )

    if not feature_columns:
        errors.append("No feature columns remain after exclusions.")

    present_features = [col for col in feature_columns if col in features.columns]
    missing_fraction = (
        features[present_features].isna().mean().sort_values(ascending=False)
        if present_features
        else pd.Series(dtype=float)
    )
    all_missing = missing_fraction[missing_fraction >= 1.0].index.tolist()
    if all_missing and not allow_all_missing_features:
        errors.append(f"All-missing features are not allowed: {all_missing}")
    elif all_missing:
        warnings.append(
            f"All-missing features will be retained by request: {all_missing}"
        )

    if max_missing_fraction is not None:
        too_sparse = missing_fraction[
            missing_fraction > float(max_missing_fraction)
        ].index.tolist()
        if too_sparse:
            errors.append(
                "Features exceed --max_missing_fraction="
                f"{max_missing_fraction}: {too_sparse}"
            )

    constant_features = []
    for col in present_features:
        if features[col].nunique(dropna=True) <= 1:
            constant_features.append(col)
    if constant_features:
        warnings.append(
            f"Constant or single-level features detected: {constant_features}"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_rows": int(len(features)),
        "n_features": int(len(feature_columns)),
        "n_numeric_features": int(
            sum(
                col in features.columns
                and col not in set(categorical_columns or [])
                and str(features[col].dtype) not in {"object", "category", "bool"}
                for col in feature_columns
            )
        ),
        "n_categorical_features": int(
            sum(
                col in features.columns
                and (
                    col in set(categorical_columns or [])
                    or str(features[col].dtype) in {"object", "category", "bool"}
                )
                for col in feature_columns
            )
        ),
        "n_all_missing_features": int(len(all_missing)),
        "n_constant_features": int(len(constant_features)),
        "max_missing_fraction": float(missing_fraction.max())
        if not missing_fraction.empty
        else None,
    }


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV or parquet table from disk."""
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source)


def parse_columns(value: str) -> list[str]:
    """Parse a comma-separated CLI column list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def outcome_labels(
    outcome_path: str,
    split: str,
    n_hours_start_include: int,
    n_hours_end_include: Optional[int],
    require_min_followup: bool,
    competing_outcome_path: Optional[str] = None,
    include_survival_fields: bool = False,
    ipcw_horizon_hours: Optional[int] = None,
    ipcw_estimand: str = "net_risk",
    registry_start_date: Optional[str] = None,
    cohort: Optional[str] = None,
    outcome_name: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    allowed_subject_ids: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Derive labels and optional IPCW fields using the shared outcome pipeline.

    Inputs identify one outcome parquet split and prediction horizon. The output
    always includes `subject_id` and `label`; optionally it also includes
    `time_days`, `event`, and `ipcw_weight`. The scientific purpose is to let
    tabular baselines use the same horizon labels and censoring weights as
    OPERA survival finetuning.
    """
    cohorts = build_evaluation_cohorts(
        outcome_path,
        split=split,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        competing_outcomes=competing_outcome_path,
        eligibility=eligibility_path,
        registry_start_date=registry_start_date,
        cohort=cohort,
        outcome_name=outcome_name,
        allowed_subject_ids=allowed_subject_ids,
    )
    regime = FIXED_HORIZON_REGIME if require_min_followup else SURVIVAL_REGIME
    labels = {
        key: dict(value) for key, value in cohorts.for_regime(regime).records.items()
    }
    if ipcw_horizon_hours is not None:
        weights = compute_ipcw_train_weights(
            labels,
            horizon_hours=ipcw_horizon_hours,
            estimand=ipcw_estimand,
        )
        for subject_id, weight in weights.items():
            labels[subject_id]["ipcw_weight"] = float(weight)
    frame = pd.DataFrame.from_dict(labels, orient="index").reset_index(
        names="subject_id"
    )
    columns = ["subject_id", "label"]
    if include_survival_fields:
        columns.extend(
            col for col in ("time_days", "event", "ipcw_weight") if col in frame.columns
        )
    return frame[columns]


def merge_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join tabular features to labels and drop rows without labels."""
    if "subject_id" not in features.columns:
        raise ValueError("Feature matrix must contain subject_id.")
    if features["subject_id"].duplicated().any():
        raise ValueError("Feature matrix contains duplicate subject_id rows.")
    return features.merge(labels, on="subject_id", how="inner").dropna(subset=["label"])


def infer_feature_columns(
    features: pd.DataFrame,
    exclude_columns: Iterable[str] = (),
) -> list[str]:
    """Return model feature columns after removing reserved and explicit columns."""
    excluded = set(exclude_columns) | RESERVED_COLUMNS
    return [col for col in features.columns if col not in excluded]


def build_preprocessor(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: Optional[list[str]] = None,
    dense_output: bool = False,
) -> ColumnTransformer:
    """Create preprocessing for mixed numeric/categorical tabular features.

    Numeric variables use median imputation with explicit missingness
    indicators. Categorical variables keep missingness as its own level. This
    is important for EHR feature matrices where missingness is clinically and
    operationally informative rather than random noise.
    """
    if categorical_columns is None:
        categorical_columns = [
            col
            for col in feature_columns
            if str(train_df[col].dtype) in {"object", "category", "bool"}
        ]
    numeric_columns = [
        col for col in feature_columns if col not in set(categorical_columns)
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=not dense_output,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns after exclusions.")
    return ColumnTransformer(
        transformers,
        sparse_threshold=0.0 if dense_output else 0.3,
    )


def make_estimator(model_name: str, seed: int, tabpfn_device: str = "auto"):
    """Construct a supported tabular estimator."""
    base_model = model_name.removesuffix("_ipcw_cif_bce").removesuffix("_ipcw_bce")
    if base_model == "logistic":
        return LogisticRegression(
            C=0.1,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            max_iter=5000,
            tol=1e-3,
            class_weight="balanced",
            random_state=seed,
        )
    if base_model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "xgboost is required for --models xgboost. Install the project "
                "environment or use --models logistic."
            ) from exc
        return XGBClassifier(
            n_estimators=500,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.5,
            min_child_weight=2,
            reg_alpha=0.5,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
        )
    if base_model == "tabpfn":
        if model_name.endswith(("_ipcw_bce", "_ipcw_cif_bce")):
            raise ValueError("TabPFN IPCW-weighted training is not supported.")
        try:
            from tabpfn import TabPFNClassifier
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "tabpfn is optional and not installed. Use an environment with "
                "TabPFN available, or run --models xgboost/logistic."
            ) from exc
        kwargs = {}
        if tabpfn_device != "auto":
            kwargs["device"] = tabpfn_device
        try:
            return TabPFNClassifier(random_state=seed, **kwargs)
        except TypeError:
            return TabPFNClassifier(**kwargs)
    raise ValueError(f"Unknown model {model_name!r}.")


def tune_estimator_params(
    model_name: str,
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    feature_columns: list[str],
    categorical_columns: Optional[list[str]] = None,
    seed: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> dict:
    """Select hyperparameters for *model_name* using a held-out validation split.

    Iterates over ``TUNING_GRIDS[base_model_name]``, fits each candidate on
    *train_df*, evaluates AUROC on *val_df*, and returns the parameter dict that
    achieved the highest validation AUROC.  Falls back to an empty dict (default
    parameters) when:

    * fewer than two classes are present in either split,
    * the model is not in ``TUNING_GRIDS``, or
    * validation AUROC cannot be computed (e.g. all-positive labels).

    The function does NOT modify ``make_estimator``; callers must apply the
    returned params with ``estimator.set_params(**best_params)``.

    Parameters
    ----------
    model_name:
        Base model name, e.g. ``"logistic"`` or ``"xgboost"``.
        The ``_ipcw_bce`` suffix is stripped before grid lookup.
    train_df, val_df:
        DataFrames with columns for *feature_columns* plus a ``"label"``
        integer column.
    feature_columns:
        Ordered list of feature columns. The same imputation, scaling, and
        categorical encoding pipeline used by the final fit is applied here.
    seed:
        Random seed forwarded to estimators.

    Returns
    -------
    dict
        Best hyperparameter dict from ``TUNING_GRIDS``, or ``{}`` on fallback.
    """
    from sklearn.metrics import roc_auc_score

    base_model = model_name.removesuffix("_ipcw_cif_bce").removesuffix("_ipcw_bce")
    grid = TUNING_GRIDS.get(base_model)
    if not grid:
        return {}

    if (
        train_df["label"].nunique() < 2
        or val_df["label"].nunique() < 2
        or len(val_df) < 10
    ):
        return {}

    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)

    best_auroc = -1.0
    best_params: dict = {}
    for params in grid:
        try:
            estimator = make_estimator(base_model, seed)
            estimator.set_params(**params)
            pipeline = Pipeline(
                [
                    (
                        "preprocess",
                        build_preprocessor(
                            train_df,
                            feature_columns,
                            categorical_columns,
                            dense_output=False,
                        ),
                    ),
                    ("model", estimator),
                ]
            )
            fit_kwargs = (
                {"model__sample_weight": sample_weight}
                if sample_weight is not None
                else {}
            )
            pipeline.fit(train_df[feature_columns], y_train, **fit_kwargs)
            probs = pipeline.predict_proba(val_df[feature_columns])[:, 1]
            auroc = float(roc_auc_score(y_val, probs))
        except Exception:
            continue
        if auroc > best_auroc:
            best_auroc = auroc
            best_params = dict(params)

    return best_params


def high_dimensional_diagnostics(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Describe whether a baseline is operating in a low-n/high-p regime."""
    n_rows = int(len(train_df))
    n_features = int(len(feature_columns))
    ratio = float(n_features / n_rows) if n_rows else float("inf")
    minority = (
        int(train_df["label"].value_counts().min())
        if n_rows and "label" in train_df and train_df["label"].nunique() > 1
        else 0
    )
    return {
        "n_train_rows": n_rows,
        "n_raw_features": n_features,
        "features_per_row": ratio,
        "minority_class_rows": minority,
        "low_n_high_p": bool(n_features >= n_rows),
        "severe_low_n_high_p": bool(n_features >= 5 * max(n_rows, 1)),
    }


def missingness_report(
    features: pd.DataFrame,
    feature_columns: list[str],
    train_subject_ids: Iterable,
    test_subject_ids: Iterable,
) -> pd.DataFrame:
    """Summarize feature missingness overall and by train/test split."""
    train_ids = set(train_subject_ids)
    test_ids = set(test_subject_ids)
    train = features[features["subject_id"].isin(train_ids)]
    test = features[features["subject_id"].isin(test_ids)]
    rows = []
    for col in feature_columns:
        series = features[col]
        train_series = train[col]
        test_series = test[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique(dropna=True)),
                "missing_fraction_all": float(series.isna().mean()),
                "missing_fraction_train": float(train_series.isna().mean())
                if len(train_series)
                else float("nan"),
                "missing_fraction_test": float(test_series.isna().mean())
                if len(test_series)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_fraction_all", ascending=False)


def select_tabpfn_feature_columns(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: Optional[list[str]],
    max_features: int,
) -> list[str]:
    """Limit TabPFN to a stable subset when matrices are very wide.

    TabPFN is an optional strong tabular baseline, but it is less comfortable
    with extremely wide clinical feature matrices than XGBoost. This selector
    keeps features with more observed data first, then higher numeric variance
    or categorical cardinality.
    """
    if max_features <= 0 or len(feature_columns) <= max_features:
        return feature_columns
    categorical = set(categorical_columns or [])
    scores = []
    for col in feature_columns:
        observed = float(train_df[col].notna().mean())
        if col in categorical or str(train_df[col].dtype) in {
            "object",
            "category",
            "bool",
        }:
            signal = float(train_df[col].nunique(dropna=True))
        else:
            signal = float(
                pd.to_numeric(train_df[col], errors="coerce").var(skipna=True) or 0.0
            )
        scores.append((observed, signal, col))
    scores.sort(reverse=True)
    return [col for _, _, col in scores[:max_features]]


def train_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: Optional[list[str]],
    seed: int,
    tabpfn_device: str = "auto",
    sample_weight: Optional[np.ndarray] = None,
    tune_params: Optional[dict] = None,
) -> tuple[pd.DataFrame, Pipeline]:
    """Train one tabular model and return held-out prediction rows.

    `sample_weight` is used for IPCW-weighted binary training, making tabular
    logistic/XGBoost variants comparable to OPERA's horizon-specific IPCW-BCE
    objective while still emitting calibrated horizon probabilities.

    `tune_params` is an optional parameter dict returned by
    ``tune_estimator_params``; when provided the estimator's defaults are
    overridden before fitting.
    """
    dense_output = model_name == "tabpfn"
    preprocessor = build_preprocessor(
        train_df,
        feature_columns,
        categorical_columns,
        dense_output=dense_output,
    )
    estimator = make_estimator(model_name, seed, tabpfn_device=tabpfn_device)
    if tune_params:
        estimator.set_params(**tune_params)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    pipeline.fit(train_df[feature_columns], train_df["label"].astype(int), **fit_kwargs)
    probabilities = pipeline.predict_proba(test_df[feature_columns])[:, 1]
    predictions = pd.DataFrame(
        {
            "subject_id": test_df["subject_id"].to_numpy(),
            "probability": probabilities.astype(float),
        }
    )
    return predictions, pipeline


def train_one_survival_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    tau_days: float,
):
    """Train one censoring-aware survival model and return held-out predictions."""
    fit = fit_cox_survival if model_name == "cox" else fit_xgboost_aft
    predictor, backend, notes = fit(
        features=train_df[feature_columns],
        times=train_df["time_days"].to_numpy(dtype=float),
        events=train_df["event"].to_numpy(dtype=int),
        tau_days=tau_days,
        seed=seed,
    )
    probabilities = predictor.predict_proba(test_df[feature_columns])
    risk_scores = predictor.predict_risk(test_df[feature_columns])
    predictions = pd.DataFrame(
        {
            "subject_id": test_df["subject_id"].to_numpy(),
            "probability": np.asarray(probabilities, dtype=float),
            "risk_score": np.asarray(risk_scores, dtype=float),
        }
    )
    return predictions, predictor, backend, notes


def write_feature_importance(
    pipeline: Pipeline,
    output_path: Path,
) -> None:
    """Write model feature weights/importances when the estimator exposes them."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array(
            [f"feature_{i}" for i in range(getattr(model, "n_features_in_", 0))]
        )

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        kind = "gain_importance"
    elif hasattr(model, "coef_"):
        values = model.coef_.reshape(-1)
        kind = "coefficient"
    else:
        return
    frame = pd.DataFrame({"feature": names, kind: values.astype(float)})
    frame = frame.sort_values(kind, key=lambda s: s.abs(), ascending=False)
    frame.to_csv(output_path, index=False)


def train_tabular_baselines(args: argparse.Namespace) -> None:
    """Train requested tabular baselines and write prediction artifacts."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = read_table(args.features)
    allowed_subject_ids = population_subject_ids(
        args.population,
        cohort_fine_col=args.cohort_fine_col,
        cohort_fine_value=args.cohort_fine_value,
    )
    feature_columns = infer_feature_columns(
        features, parse_columns(args.exclude_columns)
    )
    categorical_columns = parse_columns(args.categorical_columns) or None
    contract = validate_feature_matrix(
        features,
        feature_columns,
        categorical_columns=categorical_columns,
        max_missing_fraction=args.max_missing_fraction,
        allow_all_missing_features=args.allow_all_missing_features,
    )
    structural_failure = (
        "subject_id" not in features.columns
        or not feature_columns
        or any(col not in features.columns for col in feature_columns)
    )
    if structural_failure and contract["errors"]:
        contract_path = (
            output_dir / f"{args.cohort}_{args.outcome_name}_feature_contract.json"
        )
        with open(contract_path, "w") as f:
            json.dump(contract, f, indent=2)
        raise ValueError(
            "Tabular feature matrix contract failed; see "
            f"{contract_path}: " + "; ".join(contract["errors"])
        )

    requested_models = (
        ["logistic", "xgboost", "cox", "xgboost_aft"]
        if args.models == "all"
        else parse_columns(args.models)
    )
    supported_models = {
        "logistic",
        "xgboost",
        "tabpfn",
        "logistic_ipcw_bce",
        "xgboost_ipcw_bce",
        "logistic_ipcw_cif_bce",
        "xgboost_ipcw_cif_bce",
        *SURVIVAL_MODELS,
    }
    unknown_models = sorted(set(requested_models) - supported_models)
    if unknown_models:
        raise ValueError(f"Unknown tabular baseline models: {unknown_models}.")
    uses_net_ipcw = any(model.endswith("_ipcw_bce") for model in requested_models)
    uses_cif_ipcw = any(model.endswith("_ipcw_cif_bce") for model in requested_models)
    uses_ipcw_training = uses_net_ipcw or uses_cif_ipcw
    uses_binary_training = any(
        model not in SURVIVAL_MODELS
        and not model.endswith(("_ipcw_bce", "_ipcw_cif_bce"))
        for model in requested_models
    )
    uses_tunable_training = any(
        model not in SURVIVAL_MODELS
        and model.removesuffix("_ipcw_cif_bce").removesuffix("_ipcw_bce")
        in TUNING_GRIDS
        for model in requested_models
    )
    uses_survival_training = any(model in SURVIVAL_MODELS for model in requested_models)
    if uses_ipcw_training and args.n_hours_end_include is None:
        raise ValueError("IPCW-BCE tabular training requires --n_hours_end_include.")

    train_labels = outcome_labels(
        args.outcome,
        split=args.train_split,
        n_hours_start_include=args.n_hours_start_include,
        n_hours_end_include=args.n_hours_end_include,
        require_min_followup=True,
        competing_outcome_path=args.competing_outcome,
        registry_start_date=args.registry_start_date,
        cohort=args.cohort,
        outcome_name=args.outcome_name,
        eligibility_path=args.eligibility,
        allowed_subject_ids=allowed_subject_ids,
    )
    ipcw_train_labels = (
        outcome_labels(
            args.outcome,
            split=args.train_split,
            n_hours_start_include=args.n_hours_start_include,
            n_hours_end_include=args.n_hours_end_include,
            require_min_followup=False,
            competing_outcome_path=args.competing_outcome,
            include_survival_fields=True,
            ipcw_horizon_hours=args.n_hours_end_include,
            registry_start_date=args.registry_start_date,
            cohort=args.cohort,
            outcome_name=args.outcome_name,
            eligibility_path=args.eligibility,
            allowed_subject_ids=allowed_subject_ids,
        )
        if uses_ipcw_training
        else None
    )
    cif_ipcw_train_labels = (
        outcome_labels(
            args.outcome,
            split=args.train_split,
            n_hours_start_include=args.n_hours_start_include,
            n_hours_end_include=args.n_hours_end_include,
            require_min_followup=False,
            competing_outcome_path=args.competing_outcome,
            include_survival_fields=True,
            ipcw_horizon_hours=args.n_hours_end_include,
            ipcw_estimand="cumulative_incidence",
            registry_start_date=args.registry_start_date,
            cohort=args.cohort,
            outcome_name=args.outcome_name,
            eligibility_path=args.eligibility,
            allowed_subject_ids=allowed_subject_ids,
        )
        if uses_cif_ipcw
        else None
    )
    test_labels = outcome_labels(
        args.outcome,
        split=args.test_split,
        n_hours_start_include=args.n_hours_start_include,
        n_hours_end_include=args.n_hours_end_include,
        require_min_followup=True,
        competing_outcome_path=args.competing_outcome,
        registry_start_date=args.registry_start_date,
        cohort=args.cohort,
        outcome_name=args.outcome_name,
        eligibility_path=args.eligibility,
        allowed_subject_ids=allowed_subject_ids,
    )
    survival_train_labels = (
        outcome_labels(
            args.outcome,
            split=args.train_split,
            n_hours_start_include=args.n_hours_start_include,
            n_hours_end_include=args.n_hours_end_include,
            require_min_followup=False,
            competing_outcome_path=args.competing_outcome,
            include_survival_fields=True,
            registry_start_date=args.registry_start_date,
            cohort=args.cohort,
            outcome_name=args.outcome_name,
            eligibility_path=args.eligibility,
            allowed_subject_ids=allowed_subject_ids,
        )
        if uses_survival_training
        else None
    )
    survival_test_labels = (
        outcome_labels(
            args.outcome,
            split=args.test_split,
            n_hours_start_include=args.n_hours_start_include,
            n_hours_end_include=args.n_hours_end_include,
            require_min_followup=False,
            competing_outcome_path=args.competing_outcome,
            include_survival_fields=True,
            registry_start_date=args.registry_start_date,
            cohort=args.cohort,
            outcome_name=args.outcome_name,
            eligibility_path=args.eligibility,
            allowed_subject_ids=allowed_subject_ids,
        )
        if uses_survival_training
        else None
    )
    train_df = merge_features_and_labels(features, train_labels)
    ipcw_train_df = (
        merge_features_and_labels(features, ipcw_train_labels)
        if ipcw_train_labels is not None
        else None
    )
    cif_ipcw_train_df = (
        merge_features_and_labels(features, cif_ipcw_train_labels)
        if cif_ipcw_train_labels is not None
        else None
    )
    test_df = merge_features_and_labels(features, test_labels)
    survival_train_df = (
        merge_features_and_labels(features, survival_train_labels)
        if survival_train_labels is not None
        else None
    )
    survival_test_df = (
        merge_features_and_labels(features, survival_test_labels)
        if survival_test_labels is not None
        else None
    )
    # Load the validation split for --tune when requested.
    tune_df: Optional[pd.DataFrame] = None
    if getattr(args, "tune", True) and uses_tunable_training:
        tune_labels = outcome_labels(
            args.outcome,
            split=args.tune_split,
            n_hours_start_include=args.n_hours_start_include,
            n_hours_end_include=args.n_hours_end_include,
            require_min_followup=True,
            competing_outcome_path=args.competing_outcome,
            registry_start_date=args.registry_start_date,
            cohort=args.cohort,
            outcome_name=args.outcome_name,
            eligibility_path=args.eligibility,
            allowed_subject_ids=allowed_subject_ids,
        )
        tune_df = merge_features_and_labels(features, tune_labels)
        if tune_df.empty or tune_df["label"].nunique() < 2:
            raise ValueError(
                f"Tuning split {args.tune_split!r} must contain both classes; "
                "held-out data cannot be used for model selection."
            )
    contract.update(
        {
            "n_train_labelled": int(len(train_df)),
            "n_ipcw_train_labelled": int(len(ipcw_train_df))
            if ipcw_train_df is not None
            else None,
            "n_test_labelled": int(len(test_df)),
            "n_survival_train": int(len(survival_train_df))
            if survival_train_df is not None
            else None,
            "n_survival_test": int(len(survival_test_df))
            if survival_test_df is not None
            else None,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "tune_split": args.tune_split if tune_df is not None else None,
            "n_tune_labelled": int(len(tune_df)) if tune_df is not None else None,
            "high_dimensional": high_dimensional_diagnostics(train_df, feature_columns),
        }
    )
    if uses_binary_training and train_df["label"].nunique() < 2:
        contract["errors"].append("Training labels contain fewer than two classes.")
        contract["ok"] = False
    if uses_ipcw_training and (
        ipcw_train_df is None or ipcw_train_df["label"].nunique() < 2
    ):
        contract["errors"].append(
            "IPCW training labels contain fewer than two classes."
        )
        contract["ok"] = False
    if (uses_binary_training or uses_ipcw_training) and test_df.empty:
        contract["errors"].append("No test labels overlap the feature matrix.")
        contract["ok"] = False
    if uses_binary_training and train_df.empty:
        contract["errors"].append("No train labels overlap the feature matrix.")
        contract["ok"] = False
    if uses_ipcw_training and (ipcw_train_df is None or ipcw_train_df.empty):
        contract["errors"].append("No IPCW train labels overlap the feature matrix.")
        contract["ok"] = False
    if uses_survival_training and (
        survival_train_df is None
        or survival_test_df is None
        or survival_train_df.empty
        or survival_test_df.empty
    ):
        contract["errors"].append(
            "No survival train/test labels overlap the feature matrix."
        )
        contract["ok"] = False
    if uses_survival_training and survival_train_df is not None:
        if int((survival_train_df["event"] == 1).sum()) < 2:
            contract["errors"].append(
                "Survival training cohort contains fewer than two observed events."
            )
            contract["ok"] = False

    miss = missingness_report(
        features,
        feature_columns,
        train_subject_ids=(
            pd.concat(
                [train_df["subject_id"], ipcw_train_df["subject_id"]],
                ignore_index=True,
            )
            if ipcw_train_df is not None
            else train_df["subject_id"]
        ),
        test_subject_ids=(
            pd.concat(
                [test_df["subject_id"], survival_test_df["subject_id"]],
                ignore_index=True,
            )
            if survival_test_df is not None
            else test_df["subject_id"]
        ),
    )
    missingness_path = (
        output_dir / f"{args.cohort}_{args.outcome_name}_feature_missingness.csv"
    )
    miss.to_csv(missingness_path, index=False)
    contract["missingness_report"] = str(missingness_path)
    contract_path = (
        output_dir / f"{args.cohort}_{args.outcome_name}_feature_contract.json"
    )
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2)
    if contract["errors"]:
        raise ValueError(
            "Tabular feature matrix contract failed; see "
            f"{contract_path}: " + "; ".join(contract["errors"])
        )
    if args.validate_only:
        print(f"Feature matrix contract passed: {contract_path}")
        return

    for model_name in requested_models:
        model_feature_columns = feature_columns
        is_survival_model = model_name in SURVIVAL_MODELS
        model_train_df = (
            survival_train_df
            if is_survival_model
            else (
                cif_ipcw_train_df
                if model_name.endswith("_ipcw_cif_bce")
                else (ipcw_train_df if model_name.endswith("_ipcw_bce") else train_df)
            )
        )
        model_test_df = survival_test_df if is_survival_model else test_df
        sample_weight = None
        if model_name.endswith(("_ipcw_bce", "_ipcw_cif_bce")):
            # IPCW preserves early-censored training patients; it does not change
            # the fixed-horizon evaluation cohort used for model comparisons.
            if "ipcw_weight" not in model_train_df.columns:
                raise ValueError("IPCW training requested but ipcw_weight is missing.")
            model_train_df = model_train_df[model_train_df["ipcw_weight"] > 0].copy()
            if model_train_df["label"].nunique() < 2:
                raise ValueError(
                    f"{model_name} has fewer than two classes after IPCW filtering."
                )
            sample_weight = model_train_df["ipcw_weight"].to_numpy(dtype=float)
        if model_name == "tabpfn":
            model_feature_columns = select_tabpfn_feature_columns(
                model_train_df,
                feature_columns,
                categorical_columns,
                max_features=args.tabpfn_max_features,
            )
            if (
                args.tabpfn_max_train_rows
                and len(model_train_df) > args.tabpfn_max_train_rows
            ):
                model_train_df = model_train_df.groupby(
                    "label", group_keys=False
                ).sample(
                    frac=min(1.0, args.tabpfn_max_train_rows / len(model_train_df)),
                    random_state=args.seed,
                )
        backend = model_name
        notes = []
        pipeline = None
        tune_params: Optional[dict] = None
        if is_survival_model:
            tau_days = (
                float(args.n_hours_end_include) / 24.0
                if args.n_hours_end_include is not None
                else 730.0
            )
            predictions, _predictor, backend, notes = train_one_survival_model(
                model_name=model_name,
                train_df=model_train_df,
                test_df=model_test_df,
                feature_columns=model_feature_columns,
                seed=args.seed,
                tau_days=tau_days,
            )
        else:
            if tune_df is not None and not tune_df.empty:
                tune_params = tune_estimator_params(
                    model_name,
                    train_df=model_train_df,
                    val_df=tune_df,
                    feature_columns=model_feature_columns,
                    categorical_columns=categorical_columns,
                    seed=args.seed,
                    sample_weight=sample_weight,
                )
            predictions, pipeline = train_one_model(
                model_name=model_name,
                train_df=model_train_df,
                test_df=model_test_df,
                feature_columns=model_feature_columns,
                categorical_columns=categorical_columns,
                seed=args.seed,
                tabpfn_device=args.tabpfn_device,
                sample_weight=sample_weight,
                tune_params=tune_params,
            )
        family = (
            f"{args.model_prefix}_{model_name}"
            if args.model_prefix and len(requested_models) > 1
            else (args.model_prefix or model_name)
        )
        stem = args.output_stem.format(
            model_family=family,
            model_name=model_name,
            cohort=args.cohort,
            outcome=args.outcome_name,
        )
        pred_path = output_dir / f"{stem}_predictions.csv"
        predictions.to_csv(pred_path, index=False)
        tune_pred_path = None
        if pipeline is not None and tune_df is not None:
            tune_pred_path = output_dir / f"{stem}_tuning_predictions.csv"
            pd.DataFrame(
                {
                    "subject_id": tune_df["subject_id"].to_numpy(),
                    "probability": pipeline.predict_proba(
                        tune_df[model_feature_columns]
                    )[:, 1].astype(float),
                }
            ).to_csv(tune_pred_path, index=False)
        if pipeline is not None:
            write_feature_importance(
                pipeline,
                output_dir / f"{stem}_feature_importance.csv",
            )
        metadata = {
            "model_family": family,
            "model_name": model_name,
            "training_mode": (
                "survival"
                if is_survival_model
                else (
                    "ipcw_cif_bce"
                    if model_name.endswith("_ipcw_cif_bce")
                    else ("ipcw_bce" if model_name.endswith("_ipcw_bce") else "binary")
                )
            ),
            "evaluation_regime": (
                SURVIVAL_REGIME if is_survival_model else FIXED_HORIZON_REGIME
            ),
            "backend": backend,
            "fit_notes": notes,
            "tune_params": tune_params,
            "cohort": args.cohort,
            "outcome_name": args.outcome_name,
            "features": args.features,
            "outcome": args.outcome,
            "n_features": len(feature_columns),
            "n_features_used": len(model_feature_columns),
            "feature_columns": model_feature_columns,
            "categorical_columns": categorical_columns or [],
            "n_train": int(len(model_train_df)),
            "n_train_events": int(
                (model_train_df["event"] == 1).sum()
                if is_survival_model
                else model_train_df["label"].sum()
            ),
            "n_train_weighted": float(sample_weight.sum())
            if sample_weight is not None
            else None,
            "n_test": int(len(model_test_df)),
            "n_test_events": int(
                (model_test_df["event"] == 1).sum()
                if is_survival_model
                else model_test_df["label"].sum()
            ),
            "seed": args.seed,
            "feature_contract": str(contract_path),
            "missingness_report": str(missingness_path),
            "prediction_file": str(pred_path),
            "tuning_prediction_file": str(tune_pred_path)
            if tune_pred_path is not None
            else None,
            "fit_split": args.train_split,
            "selection_split": args.tune_split if tune_df is not None else None,
            "evaluation_split": args.test_split,
            "registry_start_date": args.registry_start_date,
            "eligibility": args.eligibility,
        }
        with open(output_dir / f"{stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote {family} predictions: {pred_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OPERA tabular baselines")
    parser.add_argument("--features", required=True, help="Feature matrix CSV/parquet")
    parser.add_argument("--outcome", required=True, help="Outcome parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--outcome_name", required=True)
    parser.add_argument(
        "--models",
        default="xgboost",
        help="xgboost, logistic, tabpfn, cox, xgboost_aft, comma list, or all",
    )
    parser.add_argument("--model_prefix", default="tabular_ehr")
    parser.add_argument(
        "--output_stem",
        default="{model_family}_{cohort}_{outcome}",
        help="Filename stem template for prediction/metadata artifacts.",
    )
    parser.add_argument("--n_hours_start_include", type=int, default=1)
    parser.add_argument("--n_hours_end_include", type=int, default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="held_out")
    parser.add_argument("--competing_outcome", default=None)
    parser.add_argument(
        "--eligibility",
        default=None,
        help="Optional patient-level outcome eligibility CSV/parquet.",
    )
    parser.add_argument(
        "--population",
        default=None,
        help="Optional population CSV/parquet defining the study cohort.",
    )
    parser.add_argument("--cohort_fine_col", default=None)
    parser.add_argument("--cohort_fine_value", default=None)
    parser.add_argument(
        "--registry_start_date",
        default=None,
        help="Optional first date with reliable registry outcome/RKKP coverage.",
    )
    parser.add_argument("--exclude_columns", default="")
    parser.add_argument("--categorical_columns", default="")
    parser.add_argument("--tabpfn_device", default="auto")
    parser.add_argument("--tabpfn_max_features", type=int, default=500)
    parser.add_argument("--tabpfn_max_train_rows", type=int, default=10000)
    parser.add_argument(
        "--max_missing_fraction",
        type=float,
        default=None,
        help="Optional hard failure threshold for per-feature missingness.",
    )
    parser.add_argument(
        "--allow_all_missing_features",
        action="store_true",
        help="Permit all-missing feature columns instead of failing the contract.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Write feature contract and missingness reports without fitting models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable validation-based hyperparameter selection for logistic and "
            "xgboost. Uses --tune_split as the validation set. Only the "
            "parameters in TUNING_GRIDS are searched; other hyperparameters "
            "stay at their defaults. Ignored for survival models and tabpfn."
        ),
    )
    parser.add_argument(
        "--tune_split",
        default="tuning",
        help="Split name used as validation set when --tune is active.",
    )
    return parser


def main() -> None:
    train_tabular_baselines(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
