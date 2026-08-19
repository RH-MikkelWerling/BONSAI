"""Disease-conditioned treatment analyses for frozen patient embeddings.

Treatment assignment is strongly disease dependent.  A global treatment probe
can therefore look excellent simply because it recognizes the diagnosis.  The
helpers in this module fit one regimen classifier per disease and either assess
it on a locked held-out split or use explicitly labelled exploratory
cross-validation.

The resulting performance describes how much historical treatment-selection
information is accessible in an embedding.  It is not a treatment
recommendation and it does not estimate a causal treatment effect.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Return canonical numeric embedding columns in deterministic order."""
    columns = [
        column
        for column in frame.columns
        if column.startswith("embedding_")
        and column.removeprefix("embedding_").isdigit()
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise ValueError("No numeric embedding_* columns were found.")

    def sort_key(value: str) -> tuple[int, str]:
        suffix = value.removeprefix("embedding_")
        return (int(suffix), value)

    return sorted(columns, key=sort_key)


def merge_embeddings_with_metadata(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Join one embedding row to one metadata row per patient."""
    for name, frame in (("embeddings", embeddings), ("metadata", metadata)):
        if subject_col not in frame.columns:
            raise ValueError(f"{name} is missing {subject_col!r}.")
        if frame[subject_col].duplicated().any():
            duplicates = frame.loc[frame[subject_col].duplicated(), subject_col].head(
                10
            )
            raise ValueError(
                f"{name} must contain one row per patient; duplicate "
                f"{subject_col} values include {duplicates.tolist()}."
            )
    embedding_columns(embeddings)
    merged = embeddings.merge(
        metadata,
        on=subject_col,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Embedding and metadata tables have no overlapping patients.")
    return merged


def normalize_regimens(
    metadata: pd.DataFrame,
    regimen_mapping: Mapping[str, Mapping[str, Sequence[object]]],
    *,
    disease_col: str = "disease",
    treatment_col: str = "treatment",
    output_col: str = "regimen_group",
    keep_unmapped: bool = False,
) -> pd.DataFrame:
    """Apply disease-specific raw-treatment to regimen-group mappings.

    ``regimen_mapping`` has the form ``disease -> group -> raw values``.  Raw
    values are compared after whitespace trimming and case folding.  This keeps
    CHOP-like normalization independent from, for example, myeloma induction
    normalization.
    """
    required = {disease_col, treatment_col}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    normalized_lookup: dict[str, dict[str, str]] = {}
    for disease, groups in regimen_mapping.items():
        disease_key = str(disease).strip().casefold()
        disease_lookup: dict[str, str] = {}
        for group, raw_values in groups.items():
            for raw_value in raw_values:
                raw_key = str(raw_value).strip().casefold()
                previous = disease_lookup.get(raw_key)
                if previous is not None and previous != str(group):
                    raise ValueError(
                        f"Raw regimen {raw_value!r} in disease {disease!r} maps "
                        f"to both {previous!r} and {group!r}."
                    )
                disease_lookup[raw_key] = str(group)
        normalized_lookup[disease_key] = disease_lookup

    result = metadata.copy()
    groups: list[object] = []
    for disease, treatment in result[[disease_col, treatment_col]].itertuples(
        index=False, name=None
    ):
        if pd.isna(disease) or pd.isna(treatment):
            groups.append(np.nan)
            continue
        disease_key = str(disease).strip().casefold()
        treatment_key = str(treatment).strip().casefold()
        mapped = normalized_lookup.get(disease_key, {}).get(treatment_key)
        if mapped is not None:
            groups.append(mapped)
        elif keep_unmapped:
            groups.append(str(treatment).strip())
        else:
            groups.append(np.nan)
    result[output_col] = groups
    return result


def _probe_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "probe",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _macro_ovr_auc(
    truth: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> float:
    try:
        if len(classes) == 2:
            positive = classes[1]
            return float(roc_auc_score(truth == positive, probabilities[:, 1]))
        encoded = label_binarize(truth, classes=classes)
        return float(
            roc_auc_score(
                encoded,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        return float("nan")


def _metric_row(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    baseline_predicted: np.ndarray,
) -> dict[str, float]:
    balanced = float(balanced_accuracy_score(truth, predicted))
    baseline_balanced = float(balanced_accuracy_score(truth, baseline_predicted))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": balanced,
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "macro_ovr_auroc": _macro_ovr_auc(truth, probabilities, classes),
        "majority_accuracy": float(accuracy_score(truth, baseline_predicted)),
        "majority_balanced_accuracy": baseline_balanced,
        "balanced_accuracy_gain": balanced - baseline_balanced,
    }


def _held_out_predictions(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    *,
    split_col: str,
    train_splits: Sequence[str],
    test_split: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_values = frame[split_col].astype(str)
    train_mask = split_values.isin([str(value) for value in train_splits]).to_numpy()
    test_mask = (split_values == str(test_split)).to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Held-out probing requires train_splits={list(train_splits)!r} and "
            f"test_split={test_split!r} in {split_col!r}."
        )

    train_classes = np.unique(y[train_mask])
    supported_test = test_mask & np.isin(y, train_classes)
    if len(train_classes) < 2 or len(np.unique(y[supported_test])) < 2:
        raise ValueError(
            "Fewer than two treatment classes span train and held-out rows."
        )

    model = _probe_pipeline(seed)
    model.fit(x[train_mask], y[train_mask])
    predicted = model.predict(x[supported_test])
    probabilities = model.predict_proba(x[supported_test])
    classes = np.asarray(model.named_steps["probe"].classes_, dtype=object)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x[train_mask], y[train_mask])
    baseline = dummy.predict(x[supported_test])
    return supported_test, predicted, probabilities, classes, baseline


def _cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    folds = min(n_splits, int(counts.min()))
    if len(classes) < 2 or folds < 2:
        raise ValueError(
            "At least two treatment classes with two rows each are required."
        )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = _probe_pipeline(seed)
    predicted = cross_val_predict(model, x, y, cv=cv, method="predict")
    probabilities = cross_val_predict(model, x, y, cv=cv, method="predict_proba")
    dummy = DummyClassifier(strategy="most_frequent")
    baseline = cross_val_predict(dummy, x, y, cv=cv, method="predict")
    return np.ones(len(y), dtype=bool), predicted, probabilities, classes, baseline


def probe_treatment_information(
    frame: pd.DataFrame,
    *,
    disease_col: str = "disease",
    treatment_col: str = "regimen_group",
    subject_col: str = "subject_id",
    split_col: Optional[str] = "split",
    train_splits: Sequence[str] = ("train", "tuning"),
    test_split: str = "held_out",
    evaluation_mode: str = "auto",
    min_disease_n: int = 50,
    min_class_n: int = 10,
    min_test_class_n: int = 2,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict regimen from embeddings separately within every disease.

    Parameters
    ----------
    evaluation_mode:
        ``"held_out"`` requires explicit train/test split rows, ``"cv"`` uses
        stratified cross-validation and is exploratory, and ``"auto"`` uses a
        held-out split when available before falling back to CV.
    """
    if evaluation_mode not in {"auto", "held_out", "cv"}:
        raise ValueError("evaluation_mode must be one of: auto, held_out, cv.")
    required = {subject_col, disease_col, treatment_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Treatment probe frame is missing: {sorted(missing)}")
    embed_cols = embedding_columns(frame)

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for disease, disease_frame in frame.groupby(disease_col, dropna=False, sort=True):
        if pd.isna(disease):
            continue
        work = disease_frame.dropna(subset=[treatment_col]).copy()
        work = work.loc[np.isfinite(work[embed_cols].to_numpy(float)).all(axis=1)]
        class_counts = work[treatment_col].astype(str).value_counts()
        retained_classes = class_counts[class_counts >= min_class_n].index
        work = work[work[treatment_col].astype(str).isin(retained_classes)].copy()
        work[treatment_col] = work[treatment_col].astype(str)
        if len(work) < min_disease_n or work[treatment_col].nunique() < 2:
            continue

        can_hold_out = (
            split_col is not None
            and split_col in work.columns
            and work[split_col].astype(str).isin(train_splits).any()
            and (work[split_col].astype(str) == str(test_split)).any()
        )
        mode = evaluation_mode
        if mode == "auto":
            mode = "held_out" if can_hold_out else "cv"
        if mode == "held_out" and not can_hold_out:
            continue
        if mode == "held_out":
            split_values = work[str(split_col)].astype(str)
            train_counts = work.loc[
                split_values.isin(train_splits), treatment_col
            ].value_counts()
            test_counts = work.loc[
                split_values == str(test_split), treatment_col
            ].value_counts()
            supported_classes = train_counts[
                train_counts >= min_class_n
            ].index.intersection(test_counts[test_counts >= min_test_class_n].index)
            work = work[work[treatment_col].isin(supported_classes)].copy()
            if len(work) < min_disease_n or work[treatment_col].nunique() < 2:
                continue

        x = work[embed_cols].to_numpy(dtype=float)
        y = work[treatment_col].to_numpy(dtype=object)

        try:
            if mode == "held_out":
                mask, predicted, probabilities, classes, baseline = (
                    _held_out_predictions(
                        work,
                        x,
                        y,
                        split_col=str(split_col),
                        train_splits=train_splits,
                        test_split=test_split,
                        seed=seed,
                    )
                )
                evaluation = "held_out"
                n_train = int(work[split_col].astype(str).isin(train_splits).sum())
            else:
                mask, predicted, probabilities, classes, baseline = (
                    _cross_validated_predictions(
                        x,
                        y,
                        n_splits=n_splits,
                        seed=seed,
                    )
                )
                evaluation = "stratified_cv_exploratory"
                n_train = int(len(work))
        except ValueError:
            continue

        truth = y[mask]
        metrics = _metric_row(
            truth,
            predicted,
            probabilities,
            classes,
            baseline,
        )
        summary_rows.append(
            {
                disease_col: str(disease),
                "evaluation": evaluation,
                "n_available": int(len(work)),
                "n_train": n_train,
                "n_evaluated": int(mask.sum()),
                "n_classes": int(len(classes)),
                "classes": json.dumps([str(value) for value in classes]),
                **metrics,
            }
        )
        evaluated = work.loc[mask, [subject_col, disease_col, treatment_col]].copy()
        evaluated["predicted_regimen"] = predicted
        evaluated["prediction_confidence"] = probabilities.max(axis=1)
        evaluated["correct"] = predicted == truth
        evaluated["evaluation"] = evaluation
        prediction_rows.append(evaluated)

    summary = pd.DataFrame(summary_rows)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return summary, predictions


def build_atlas_tables(
    frame: pd.DataFrame,
    coordinates: np.ndarray,
    *,
    disease_col: str = "disease",
    treatment_col: str = "regimen_group",
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create patient-coordinate and disease/treatment centroid tables."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(frame), 2):
        raise ValueError(
            f"coordinates must have shape ({len(frame)}, 2), got {coordinates.shape}."
        )
    required = {subject_col, disease_col, treatment_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Atlas frame is missing: {sorted(missing)}")

    patient = (
        frame[[subject_col, disease_col, treatment_col]].copy().reset_index(drop=True)
    )
    patient["atlas_x"] = coordinates[:, 0]
    patient["atlas_y"] = coordinates[:, 1]
    valid = patient.dropna(subset=[disease_col, treatment_col]).copy()

    centroid_frames = []
    for kind, group_cols in (
        ("disease", [disease_col]),
        ("treatment", [treatment_col]),
        ("disease_treatment", [disease_col, treatment_col]),
    ):
        grouped = (
            valid.groupby(group_cols, dropna=False, sort=True)
            .agg(
                n=(subject_col, "size"),
                atlas_x=("atlas_x", "mean"),
                atlas_y=("atlas_y", "mean"),
            )
            .reset_index()
        )
        grouped["kind"] = kind
        centroid_frames.append(grouped)
    centroids = pd.concat(centroid_frames, ignore_index=True, sort=False)
    return patient, centroids


def compute_disease_treatment_geometry(
    frame: pd.DataFrame,
    *,
    disease_col: str = "disease",
    treatment_col: str = "regimen_group",
    min_joint_n: int = 10,
) -> pd.DataFrame:
    """Measure joint-cell similarity in the original embedding space.

    Patient embeddings are L2-normalized before cell centroids are calculated,
    and centroids are normalized again before cosine similarity.  The output is
    directed so every cell has a ranked nearest-neighbour list.
    """
    required = {disease_col, treatment_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Geometry frame is missing: {sorted(missing)}")
    embed_cols = embedding_columns(frame)
    work = frame.dropna(subset=[disease_col, treatment_col]).copy()
    vectors = work[embed_cols].to_numpy(dtype=float)
    finite = np.isfinite(vectors).all(axis=1)
    work = work.loc[finite].reset_index(drop=True)
    vectors = vectors[finite]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    cells = []
    for (disease, treatment), indices in work.groupby(
        [disease_col, treatment_col], sort=True
    ).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        if len(positions) < min_joint_n:
            continue
        centroid = vectors[positions].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        cells.append(
            {
                "disease": str(disease),
                "treatment": str(treatment),
                "n": int(len(positions)),
                "centroid": centroid,
            }
        )
    if len(cells) < 2:
        return pd.DataFrame()

    centroid_matrix = np.vstack([cell["centroid"] for cell in cells])
    similarities = centroid_matrix @ centroid_matrix.T
    rows = []
    for source_index, source in enumerate(cells):
        targets = [index for index in range(len(cells)) if index != source_index]
        targets.sort(key=lambda index: similarities[source_index, index], reverse=True)
        for rank, target_index in enumerate(targets, start=1):
            target = cells[target_index]
            rows.append(
                {
                    "source_disease": source["disease"],
                    "source_treatment": source["treatment"],
                    "source_n": source["n"],
                    "target_disease": target["disease"],
                    "target_treatment": target["treatment"],
                    "target_n": target["n"],
                    "cosine_similarity": float(
                        similarities[source_index, target_index]
                    ),
                    "same_disease": source["disease"] == target["disease"],
                    "same_treatment_label": source["treatment"] == target["treatment"],
                    "neighbor_rank": rank,
                }
            )
    return pd.DataFrame(rows)
