"""Comparison framework for OPERA outcome evaluation.

This module implements Task 1 of the OPERA evaluation pipeline: a common
cross-validated runner that compares clinical, tabular, and embedding-based
models on a shared survival-analysis benchmark.

The implementation is intentionally defensive:
- optional heavy dependencies (``lifelines``, ``xgboost``, ``torch``) are
  imported lazily and used when available;
- sklearn fallbacks keep the module importable and testable in lightweight
  environments;
- outputs are JSON-serializable to support caching and downstream table
  generation.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from opera.evaluation.metrics import compute_survival_metrics

LOGGER = logging.getLogger(__name__)
SEED = 42

DEFAULT_TAU_DAYS: Dict[str, float] = {
    "os": 730.0,
    "pfs": 730.0,
    "ttnt": 730.0,
    "grade3plus_infection": 90.0,
    "infection": 90.0,
    "grade3plus_neutropenia": 90.0,
    "neutropenia": 90.0,
    "aki": 30.0,
    "transfusion_required": 180.0,
    "treatment_failure_12m": 365.0,
    "treatment_failure": 365.0,
}

SCORE_SELECTION_RULES: Dict[str, Tuple[str, ...]] = {
    "dlbcl": ("ipi_score", "nccn_ipi", "age_adjusted_ipi", "ipi_risk_group"),
    "follicular": ("flipi_score", "flipi_risk_group"),
    "fl": ("flipi_score", "flipi_risk_group"),
    "mcl": ("mipi_score", "mipi_risk_group"),
    "mantle": ("mipi_score", "mipi_risk_group"),
    "hcl": ("hcl_score", "hcl_risk_group"),
    "hairy": ("hcl_score", "hcl_risk_group"),
}

AGE_CANDIDATES = ("age", "age_years", "age_at_index", "age_at_diagnosis")
SEX_CANDIDATES = ("sex", "gender", "sex_at_birth")


@dataclass
class PreparedDataset:
    """Merged dataset for one outcome and evaluation scope."""

    frame: pd.DataFrame
    evaluation_mask: pd.Series
    tau_days: float
    score_column: Optional[str]
    age_column: Optional[str]
    sex_column: Optional[str]


@dataclass
class PredictionPayload:
    """Out-of-fold predictions for one model."""

    model_name: str
    backend: str
    predictions: pd.DataFrame
    notes: List[str]


class ConstantPredictor:
    """Predict a fixed probability when a fold lacks trainable signal."""

    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.probability, len(features))

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(features)
        return np.log(probs / (1.0 - probs))


class SklearnBinaryPredictor:
    """Adapter exposing uniform probability and risk outputs."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features)[:, 1].astype(float)

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        estimator = self.pipeline[-1]
        if hasattr(estimator, "decision_function"):
            return self.pipeline.decision_function(features).astype(float)
        probs = self.predict_proba(features)
        return np.log(
            np.clip(probs, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probs, 1e-6, 1.0)
        )


class CoxSurvivalPredictor:
    """Thin wrapper around a fitted lifelines CoxPHFitter."""

    def __init__(
        self,
        cox_model: Any,
        feature_pipeline: ColumnTransformer,
        feature_columns: Sequence[str],
        tau_days: float,
    ) -> None:
        self.cox_model = cox_model
        self.feature_pipeline = feature_pipeline
        self.feature_columns = list(feature_columns)
        self.tau_days = float(tau_days)

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = self.feature_pipeline.transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return pd.DataFrame(
            np.asarray(transformed, dtype=float),
            columns=self.feature_columns,
            index=features.index,
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        surv = self.cox_model.predict_survival_function(
            self._transform(features),
            times=[self.tau_days],
        )
        values = np.asarray(surv.iloc[0], dtype=float)
        return np.clip(1.0 - values, 0.0, 1.0)

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        risk = self.cox_model.predict_partial_hazard(self._transform(features))
        return np.asarray(risk, dtype=float).reshape(-1)


class XGBoostAFTPredictor:
    """Survival AFT predictor with horizon-risk conversion."""

    def __init__(
        self,
        booster: Any,
        feature_pipeline: ColumnTransformer,
        tau_days: float,
        distribution: str,
        scale: float,
    ) -> None:
        self.booster = booster
        self.feature_pipeline = feature_pipeline
        self.tau_days = float(tau_days)
        self.distribution = distribution
        self.scale = float(scale)

    def _transform(self, features: pd.DataFrame) -> np.ndarray:
        transformed = self.feature_pipeline.transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float32)

    def _aft_cdf(self, z: np.ndarray) -> np.ndarray:
        if self.distribution == "normal":
            erf = np.vectorize(math.erf)
            return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
        if self.distribution == "logistic":
            return 1.0 / (1.0 + np.exp(-z))
        if self.distribution == "extreme":
            return 1.0 - np.exp(-np.exp(z))
        raise ValueError(f"Unsupported AFT distribution: {self.distribution}")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        matrix = xgb.DMatrix(self._transform(features))
        location = self.booster.predict(matrix)
        z = (np.log(np.maximum(self.tau_days, 1e-6)) - location) / self.scale
        return np.clip(self._aft_cdf(z), 0.0, 1.0)

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(features)
        return np.log(
            np.clip(probs, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probs, 1e-6, 1.0)
        )


class TorchMLPPredictor:
    """Small two-layer PyTorch MLP for OPERA outcome-head fine-tuning."""

    def __init__(self, model: Any, scaler: StandardScaler) -> None:
        self.model = model
        self.scaler = scaler

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        import torch

        x = self.scaler.transform(features)
        tensor = torch.tensor(x, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(float)

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(features)
        return np.log(
            np.clip(probs, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probs, 1e-6, 1.0)
        )


class ComparisonRunner:
    """Run the OPERA comparison benchmark for one outcome/cohort specification."""

    def __init__(
        self,
        outcome_name: str,
        cohort_specification: Optional[Any],
        mode: str,
        *,
        outcomes: pd.DataFrame,
        rkkp: Optional[pd.DataFrame],
        ehr_features: pd.DataFrame,
        embeddings_base: Mapping[Any, np.ndarray],
        embeddings_dapt: Mapping[Any, np.ndarray],
        embeddings_opera: Mapping[Any, np.ndarray],
        disease_cohorts: Mapping[str, Sequence[Any]],
        seed: int = SEED,
        n_splits: int = 5,
        n_bootstrap: int = 1000,
        tau_days: Optional[float] = None,
        evaluation_strategy: str = "prospective_holdout",
        enabled_models: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> None:
        self.outcome_name = outcome_name
        self.cohort_specification = cohort_specification
        self.mode = mode
        self.outcomes = outcomes.copy()
        self.rkkp = rkkp.copy() if rkkp is not None else None
        self.ehr_features = ehr_features.copy()
        self.embeddings_base = embeddings_base
        self.embeddings_dapt = embeddings_dapt
        self.embeddings_opera = embeddings_opera
        self.disease_cohorts = disease_cohorts
        self.seed = int(seed)
        self.n_splits = int(n_splits)
        self.n_bootstrap = int(n_bootstrap)
        self.tau_days = (
            float(tau_days) if tau_days is not None else infer_tau_days(outcome_name)
        )
        self.evaluation_strategy = evaluation_strategy
        self.enabled_models = (
            list(enabled_models)
            if enabled_models is not None
            else [
                "ClinicalScore",
                "XGBoost_specific",
                "XGBoost_all",
                "LinearProbe_base",
                "LinearProbe_dapt",
                "OPERA",
                "OPERA_mlp",
            ]
        )
        self.show_progress = show_progress

        if self.mode not in {"disease_specific", "pooled"}:
            raise ValueError("mode must be 'disease_specific' or 'pooled'.")
        if self.evaluation_strategy not in {"prospective_holdout", "cross_validation"}:
            raise ValueError(
                "evaluation_strategy must be 'prospective_holdout' or 'cross_validation'."
            )

    def run(self) -> Dict[str, Any]:
        prepared = self._prepare_dataset()
        folds = self._build_folds(prepared.frame)

        model_results: Dict[str, Any] = {}
        iterator = tqdm(
            self.enabled_models,
            desc=f"compare:{self.outcome_name}",
            disable=not self.show_progress,
        )
        for model_name in iterator:
            LOGGER.info(
                "Running comparison model %s for outcome=%s mode=%s",
                model_name,
                self.outcome_name,
                self.mode,
            )
            payload = self._cross_validated_predictions(
                model_name=model_name,
                prepared=prepared,
                folds=folds,
            )
            model_results[model_name] = self._summarize_predictions(payload, prepared)

        return make_jsonable(
            {
                "outcome_name": self.outcome_name,
                "mode": self.mode,
                "evaluation_strategy": self.evaluation_strategy,
                "cohort_specification": self._cohort_label(),
                "tau_days": prepared.tau_days,
                "n_patients": int(prepared.evaluation_mask.sum()),
                "n_binary_eligible": int(
                    prepared.frame.loc[
                        prepared.evaluation_mask, "binary_eligible"
                    ].sum()
                ),
                "score_column": prepared.score_column,
                "age_column": prepared.age_column,
                "sex_column": prepared.sex_column,
                "models": model_results,
            }
        )

    def _cohort_label(self) -> str:
        if self.mode == "pooled":
            return "all_hematology"
        if isinstance(self.cohort_specification, str):
            return self.cohort_specification
        return "custom_cohort"

    def _prepare_dataset(self) -> PreparedDataset:
        outcome_df = self.outcomes.loc[
            self.outcomes["outcome_name"] == self.outcome_name
        ].copy()
        if outcome_df.empty:
            raise ValueError(f"No rows found for outcome_name={self.outcome_name!r}.")

        if outcome_df["patient_id"].duplicated().any():
            LOGGER.warning(
                "Outcome %s has duplicated patient rows; keeping first occurrence per patient.",
                self.outcome_name,
            )
            outcome_df = outcome_df.drop_duplicates(subset=["patient_id"], keep="first")

        cohort_ids = set(self._resolve_cohort_ids(outcome_df))

        merged = outcome_df.merge(
            self.ehr_features,
            on="patient_id",
            how="left",
            suffixes=("", "_ehr"),
        )
        if self.rkkp is not None:
            merged = merged.merge(
                self.rkkp,
                on="patient_id",
                how="left",
                suffixes=("", "_rkkp"),
            )

        merged = merged.merge(
            self._embedding_frame(self.embeddings_base, "base"),
            on="patient_id",
            how="inner",
        )
        merged = merged.merge(
            self._embedding_frame(self.embeddings_dapt, "dapt"),
            on="patient_id",
            how="inner",
        )
        merged = merged.merge(
            self._embedding_frame(self.embeddings_opera, "opera"),
            on="patient_id",
            how="inner",
        )

        merged["event_indicator"] = merged["event_indicator"].fillna(0).astype(int)
        merged["time_to_event"] = pd.to_numeric(
            merged["time_to_event"], errors="coerce"
        )
        merged = merged.loc[np.isfinite(merged["time_to_event"])].copy()
        merged = merged.loc[merged["time_to_event"] >= 0].copy()

        binary = derive_horizon_labels(
            times=merged["time_to_event"].to_numpy(dtype=float),
            events=merged["event_indicator"].to_numpy(dtype=int),
            tau_days=self.tau_days,
        )
        merged["binary_label"] = binary["label"]
        merged["binary_eligible"] = binary["eligible"]

        age_column = find_first_column(merged, AGE_CANDIDATES)
        sex_column = find_first_column(merged, SEX_CANDIDATES)
        score_column = self._attach_selected_clinical_score(merged)

        if self.mode == "disease_specific":
            evaluation_mask = merged["patient_id"].isin(cohort_ids)
        else:
            evaluation_mask = pd.Series(True, index=merged.index)

        if (
            self.evaluation_strategy == "prospective_holdout"
            and "split" in merged.columns
            and (merged["split"] == "held_out").any()
        ):
            evaluation_mask &= merged["split"].eq("held_out")

        merged = merged.reset_index(drop=True)
        evaluation_mask = evaluation_mask.reset_index(drop=True)
        LOGGER.info(
            "Prepared dataset outcome=%s mode=%s strategy=%s n_total=%d n_eval=%d eligible_binary_eval=%d tau_days=%.1f",
            self.outcome_name,
            self.mode,
            self.evaluation_strategy,
            len(merged),
            int(evaluation_mask.sum()),
            int(merged.loc[evaluation_mask, "binary_eligible"].sum()),
            self.tau_days,
        )
        return PreparedDataset(
            frame=merged,
            evaluation_mask=evaluation_mask,
            tau_days=self.tau_days,
            score_column=score_column,
            age_column=age_column,
            sex_column=sex_column,
        )

    def _resolve_cohort_ids(self, outcome_df: pd.DataFrame) -> List[Any]:
        if self.mode == "pooled" or self.cohort_specification in (None, "all"):
            return self._all_patient_ids(outcome_df)
        if isinstance(self.cohort_specification, str):
            if self.cohort_specification in self.disease_cohorts:
                return list(self.disease_cohorts[self.cohort_specification])
            return outcome_df.loc[
                outcome_df["disease_subtype"] == self.cohort_specification,
                "patient_id",
            ].tolist()
        return list(self.cohort_specification)

    def _all_patient_ids(self, outcome_df: pd.DataFrame) -> List[Any]:
        return outcome_df["patient_id"].tolist()

    def _embedding_frame(
        self,
        embeddings: Mapping[Any, np.ndarray],
        prefix: str,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for patient_id, vector in embeddings.items():
            arr = np.asarray(vector, dtype=float).reshape(-1)
            row = {"patient_id": patient_id}
            row.update(
                {f"emb_{prefix}_{i}": float(value) for i, value in enumerate(arr)}
            )
            rows.append(row)
        if not rows:
            raise ValueError(f"Embedding mapping for prefix={prefix!r} is empty.")
        return pd.DataFrame(rows)

    def _attach_selected_clinical_score(self, frame: pd.DataFrame) -> Optional[str]:
        values: List[Any] = []
        has_any_value = False
        for _, row in frame.iterrows():
            subtype = str(row.get("disease_subtype", "")).lower()
            candidates: List[str] = []
            for pattern, columns in SCORE_SELECTION_RULES.items():
                if pattern in subtype:
                    candidates.extend(columns)
            if not candidates:
                candidates = [
                    "ipi_score",
                    "flipi_score",
                    "mipi_score",
                    "hcl_score",
                    "ipi_risk_group",
                ]
            chosen = np.nan
            for column in candidates:
                if column in frame.columns and pd.notna(row.get(column)):
                    chosen = row.get(column)
                    has_any_value = True
                    break
            values.append(chosen)
        frame["clinical_score_selected"] = values
        return "clinical_score_selected" if has_any_value else None

    def _build_folds(self, frame: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        if (
            self.evaluation_strategy == "prospective_holdout"
            and "split" in frame.columns
            and (frame["split"] == "held_out").any()
        ):
            train_mask = frame["split"].isin(["train", "tuning"])
            test_mask = frame["split"].eq("held_out")
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                raise ValueError(
                    "Prospective holdout evaluation requires non-empty train/tuning "
                    "and held_out splits."
                )
            return [
                (
                    np.flatnonzero(train_mask.to_numpy()),
                    np.flatnonzero(test_mask.to_numpy()),
                )
            ]

        y = frame["event_indicator"].fillna(0).astype(str)
        if self.mode == "pooled":
            strat = frame["disease_subtype"].fillna("unknown").astype(str) + "::" + y
        else:
            strat = y

        min_group = int(strat.value_counts().min()) if len(strat) else self.n_splits
        indices = np.arange(len(frame))
        if min_group >= 2 and len(np.unique(strat)) >= 2:
            n_splits = max(2, min(self.n_splits, min_group))
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.seed,
            )
            return list(splitter.split(indices, strat))

        n_splits = max(2, min(self.n_splits, len(frame)))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        LOGGER.warning(
            "Falling back to unstratified KFold for outcome=%s because at least "
            "one stratification bucket has fewer than two patients.",
            self.outcome_name,
        )
        return list(splitter.split(indices))

    def _cross_validated_predictions(
        self,
        model_name: str,
        prepared: PreparedDataset,
        folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> PredictionPayload:
        frame = prepared.frame
        prediction_rows: List[pd.DataFrame] = []
        notes: List[str] = []
        backend = "unknown"

        for fold_id, (train_idx, test_idx) in enumerate(folds):
            test_frame = frame.iloc[test_idx].copy()
            test_frame = test_frame.loc[
                prepared.evaluation_mask.iloc[test_idx].to_numpy()
            ].copy()
            if test_frame.empty:
                continue
            train_frame = self._training_frame_for_model(
                model_name,
                frame,
                prepared,
                train_idx,
                test_idx,
            )
            predictor, fold_backend, fold_notes = self._fit_model(
                model_name=model_name,
                train_frame=train_frame,
                prepared=prepared,
                seed=self.seed + fold_id,
            )
            backend = fold_backend
            notes.extend(fold_notes)
            eval_features = self._select_feature_frame(model_name, test_frame, prepared)
            test_frame["predicted_probability"] = predictor.predict_proba(eval_features)
            test_frame["predicted_risk"] = predictor.predict_risk(eval_features)
            test_frame["fold"] = fold_id
            prediction_rows.append(
                test_frame[
                    [
                        "patient_id",
                        "disease_subtype",
                        "time_to_event",
                        "event_indicator",
                        "binary_label",
                        "binary_eligible",
                        "predicted_probability",
                        "predicted_risk",
                        "fold",
                    ]
                ].copy()
            )

        if not prediction_rows:
            raise ValueError(
                f"No evaluation patients were available for model={model_name} "
                f"under mode={self.mode} strategy={self.evaluation_strategy}."
            )
        predictions = pd.concat(prediction_rows, axis=0, ignore_index=True)
        predictions = predictions.sort_values("patient_id").reset_index(drop=True)
        return PredictionPayload(
            model_name=model_name,
            backend=backend,
            predictions=predictions,
            notes=sorted(set(notes)),
        )

    def _training_frame_for_model(
        self,
        model_name: str,
        full_frame: pd.DataFrame,
        prepared: PreparedDataset,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> pd.DataFrame:
        train_frame = full_frame.iloc[train_idx].copy()
        if self.mode == "disease_specific" and model_name != "XGBoost_all":
            cohort_ids = set(self._resolve_cohort_ids(full_frame))
            train_frame = train_frame.loc[
                train_frame["patient_id"].isin(cohort_ids)
            ].copy()
        if (
            self.evaluation_strategy == "prospective_holdout"
            and "split" in full_frame.columns
            and model_name == "XGBoost_all"
        ):
            train_frame = full_frame.loc[
                full_frame["split"].isin(["train", "tuning"])
            ].copy()
        return train_frame

    def _select_feature_frame(
        self,
        model_name: str,
        frame: pd.DataFrame,
        prepared: PreparedDataset,
    ) -> pd.DataFrame:
        return frame[self._feature_columns(model_name, prepared)].copy()

    def _feature_columns(
        self,
        model_name: str,
        prepared: PreparedDataset,
    ) -> List[str]:
        if model_name == "ClinicalScore":
            columns = [
                column
                for column in (
                    prepared.score_column,
                    prepared.age_column,
                    prepared.sex_column,
                )
                if column is not None
            ]
            if not columns:
                raise ValueError(
                    "ClinicalScore requires at least one available feature among score/age/sex."
                )
            return columns
        if model_name == "LinearProbe_base":
            return [c for c in prepared.frame.columns if c.startswith("emb_base_")]
        if model_name == "LinearProbe_dapt":
            return [c for c in prepared.frame.columns if c.startswith("emb_dapt_")]
        if model_name in {"OPERA", "OPERA_mlp"}:
            return [c for c in prepared.frame.columns if c.startswith("emb_opera_")]
        if model_name in {"XGBoost_specific", "XGBoost_all"}:
            excluded = {
                "patient_id",
                "outcome_name",
                "disease_subtype",
                "event_indicator",
                "time_to_event",
                "index_date",
                "binary_label",
                "binary_eligible",
                "clinical_score_selected",
            }
            excluded.update({c for c in prepared.frame.columns if c.startswith("emb_")})
            return [c for c in prepared.frame.columns if c not in excluded]
        raise ValueError(f"Unsupported model_name={model_name!r}.")

    def _fit_model(
        self,
        model_name: str,
        train_frame: pd.DataFrame,
        prepared: PreparedDataset,
        seed: int,
    ) -> Tuple[Any, str, List[str]]:
        feature_columns = self._feature_columns(model_name, prepared)
        feature_frame = train_frame[feature_columns].copy()

        if model_name in {
            "ClinicalScore",
            "LinearProbe_base",
            "LinearProbe_dapt",
            "OPERA",
        }:
            return fit_logistic_probe(
                features=feature_frame,
                labels=train_frame["binary_label"].to_numpy(dtype=float),
                eligible=train_frame["binary_eligible"].to_numpy(dtype=bool),
                seed=seed,
            )
        if model_name in {"XGBoost_specific", "XGBoost_all"}:
            return fit_xgboost_aft(
                features=feature_frame,
                times=train_frame["time_to_event"].to_numpy(dtype=float),
                events=train_frame["event_indicator"].to_numpy(dtype=int),
                tau_days=prepared.tau_days,
                seed=seed,
            )
        if model_name == "OPERA_mlp":
            return fit_mlp_outcome_head(
                features=feature_frame,
                labels=train_frame["binary_label"].to_numpy(dtype=float),
                eligible=train_frame["binary_eligible"].to_numpy(dtype=bool),
                seed=seed,
            )
        raise ValueError(f"Unsupported model_name={model_name!r}.")

    def _summarize_predictions(
        self,
        payload: PredictionPayload,
        prepared: PreparedDataset,
    ) -> Dict[str, Any]:
        predictions = payload.predictions.copy()
        point_metrics = compute_comparison_metrics(
            predictions,
            tau_days=prepared.tau_days,
        )
        bootstrap = bootstrap_comparison_metrics(
            predictions,
            tau_days=prepared.tau_days,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed,
        )
        by_disease: Dict[str, Any] = {}
        for disease, disease_df in predictions.groupby("disease_subtype", dropna=False):
            label = "unknown" if pd.isna(disease) else str(disease)
            by_disease[label] = compute_comparison_metrics(
                disease_df.copy(),
                tau_days=prepared.tau_days,
            )
        return {
            "backend": payload.backend,
            "notes": payload.notes,
            "n_eval": int(len(predictions)),
            "n_binary_eligible": int(predictions["binary_eligible"].sum()),
            "metrics": point_metrics,
            "bootstrap_ci": bootstrap,
            "table_row": format_table_row(point_metrics, bootstrap),
            "by_disease_subtype": by_disease,
            "predictions": predictions.to_dict(orient="records"),
        }


def infer_tau_days(outcome_name: str) -> float:
    """Infer the task-specific prediction horizon from the outcome name."""
    lowered = outcome_name.lower()
    match = re.search(r"_(\d+)([dmy])\b", lowered)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "d":
            return value
        if unit == "m":
            return value * (365.0 / 12.0)
        if unit == "y":
            return value * 365.25
    for key, tau in DEFAULT_TAU_DAYS.items():
        if key in lowered:
            return tau
    return 365.0


def derive_horizon_labels(
    times: np.ndarray,
    events: np.ndarray,
    tau_days: float,
) -> Dict[str, np.ndarray]:
    """Create binary horizon labels with eligibility masking."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    eligible = (times > tau_days) | ((times <= tau_days) & (events == 1))
    labels = np.full(len(times), np.nan, dtype=float)
    labels[(times <= tau_days) & (events == 1)] = 1.0
    labels[times > tau_days] = 0.0
    return {"label": labels, "eligible": eligible.astype(bool)}


def find_first_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a mixed-type preprocessing pipeline."""
    numeric_columns = [
        c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])
    ]
    categorical_columns = [c for c in features.columns if c not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns were provided.")
    return ColumnTransformer(transformers=transformers)


def _fit_constant_from_labels(
    labels: np.ndarray, eligible: np.ndarray
) -> ConstantPredictor:
    eligible_labels = labels[eligible]
    if len(eligible_labels) == 0 or not np.isfinite(eligible_labels).any():
        return ConstantPredictor(0.5)
    probability = float(np.nanmean(eligible_labels))
    if not np.isfinite(probability):
        probability = 0.5
    return ConstantPredictor(probability)


def fit_logistic_probe(
    *,
    features: pd.DataFrame,
    labels: np.ndarray,
    eligible: np.ndarray,
    seed: int,
) -> Tuple[Any, str, List[str]]:
    """Fit an explicitly binary regularized logistic probe."""
    notes: List[str] = []
    train_mask = eligible & np.isfinite(labels)
    if train_mask.sum() < 10 or len(np.unique(labels[train_mask])) < 2:
        return (
            _fit_constant_from_labels(labels, eligible),
            "constant",
            ["Too few eligible labeled patients for logistic fit."],
        )

    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(features)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    try:
        pipeline.fit(features.loc[train_mask], labels[train_mask].astype(int))
        return SklearnBinaryPredictor(pipeline), "sklearn_logistic", notes
    except Exception as exc:
        notes.append(
            f"Logistic fit failed and constant fallback was used: {type(exc).__name__}."
        )
        return _fit_constant_from_labels(labels, eligible), "constant", notes


def fit_cox_survival(
    *,
    features: pd.DataFrame,
    times: np.ndarray,
    events: np.ndarray,
    tau_days: float,
    seed: int,
) -> Tuple[Any, str, List[str]]:
    """Fit a censoring-aware Cox PH model or raise a visible failure."""
    del seed
    if int((np.asarray(events) == 1).sum()) < 2:
        raise RuntimeError("Cox PH survival baseline requires at least two events.")
    preprocessor = build_preprocessor(features)
    try:
        from lifelines import CoxPHFitter

        transformed = preprocessor.fit_transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_columns = list(preprocessor.get_feature_names_out())
        design = pd.DataFrame(
            np.asarray(transformed, dtype=float),
            columns=feature_columns,
            index=features.index,
        )
        design["time_to_event"] = np.asarray(times, dtype=float)
        design["event_indicator"] = (np.asarray(events) == 1).astype(int)
        cox = CoxPHFitter(penalizer=0.1)
        cox.fit(
            design,
            duration_col="time_to_event",
            event_col="event_indicator",
            show_progress=False,
        )
        return (
            CoxSurvivalPredictor(
                cox,
                feature_pipeline=preprocessor,
                feature_columns=feature_columns,
                tau_days=tau_days,
            ),
            "lifelines_cox",
            [],
        )
    except Exception as exc:
        raise RuntimeError(
            "Cox PH survival baseline failed; no binary classifier fallback was used."
        ) from exc


def fit_linear_survival_or_logistic(
    *,
    features: pd.DataFrame,
    times: np.ndarray,
    events: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    tau_days: float,
    seed: int,
) -> Tuple[Any, str, List[str]]:
    """Compatibility wrapper for the now-strict Cox survival fit."""
    del labels, eligible
    return fit_cox_survival(
        features=features,
        times=times,
        events=events,
        tau_days=tau_days,
        seed=seed,
    )


def fit_xgboost_aft(
    *,
    features: pd.DataFrame,
    times: np.ndarray,
    events: np.ndarray,
    tau_days: float,
    seed: int,
) -> Tuple[Any, str, List[str]]:
    """Fit a censoring-aware XGBoost AFT model or raise a visible failure."""
    preprocessor = build_preprocessor(features)
    try:
        import xgboost as xgb

        transformed = preprocessor.fit_transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=np.float32)

        lower = times.astype(np.float32)
        upper = times.astype(np.float32)
        upper[events != 1] = np.inf
        dtrain = xgb.DMatrix(transformed)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)
        booster = xgb.train(
            params={
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "lambda": 1.0,
                "seed": seed,
                "tree_method": "hist",
            },
            dtrain=dtrain,
            num_boost_round=200,
            verbose_eval=False,
        )
        predictor = XGBoostAFTPredictor(
            booster=booster,
            feature_pipeline=preprocessor,
            tau_days=tau_days,
            distribution="normal",
            scale=1.0,
        )
        return predictor, "xgboost_aft", []
    except Exception as exc:
        raise RuntimeError(
            "XGBoost AFT survival baseline failed; no binary classifier fallback was used."
        ) from exc


def fit_xgboost_or_fallback(
    *,
    features: pd.DataFrame,
    times: np.ndarray,
    events: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    tau_days: float,
    seed: int,
) -> Tuple[Any, str, List[str]]:
    """Compatibility wrapper for the now-strict XGBoost AFT fit."""
    del labels, eligible
    return fit_xgboost_aft(
        features=features,
        times=times,
        events=events,
        tau_days=tau_days,
        seed=seed,
    )


def fit_mlp_outcome_head(
    *,
    features: pd.DataFrame,
    labels: np.ndarray,
    eligible: np.ndarray,
    seed: int,
) -> Tuple[Any, str, List[str]]:
    """Fit a simple two-layer outcome head on OPERA embeddings."""
    train_mask = eligible & np.isfinite(labels)
    if train_mask.sum() < 10 or len(np.unique(labels[train_mask])) < 2:
        return (
            _fit_constant_from_labels(labels, eligible),
            "constant",
            ["Too few eligible labeled patients for OPERA MLP fit."],
        )

    x_train = features.loc[train_mask].to_numpy(dtype=float)
    y_train = labels[train_mask].astype(int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    notes: List[str] = []

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(seed)

        class OutcomeHead(nn.Module):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        dataset = TensorDataset(
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)
        model = OutcomeHead(x_scaled.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(50):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        return TorchMLPPredictor(model, scaler), "torch_mlp", notes
    except Exception as exc:
        notes.append(
            "Fell back to sklearn MLP because torch was unavailable: "
            f"{type(exc).__name__}."
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=min(64, len(x_scaled)),
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(features.loc[train_mask], y_train)
    return SklearnBinaryPredictor(pipeline), "sklearn_mlp", notes


def compute_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Compute ICI and maximum calibration error from a smoothed calibration map."""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    mask = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[mask]
    probabilities = np.clip(probabilities[mask], 1e-6, 1.0 - 1e-6)
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return {"ici": float("nan"), "mce": float("nan")}

    observed = IsotonicRegression(out_of_bounds="clip").fit_transform(
        probabilities,
        labels,
    )
    abs_error = np.abs(observed - probabilities)
    return {"ici": float(abs_error.mean()), "mce": float(abs_error.max())}


def compute_comparison_metrics(
    predictions: pd.DataFrame,
    tau_days: float,
) -> Dict[str, Any]:
    """Compute discrimination, survival, and calibration metrics from OOF predictions."""
    eligible = predictions.loc[predictions["binary_eligible"].astype(bool)].copy()
    labels = eligible["binary_label"].to_numpy(dtype=float)
    probs = eligible["predicted_probability"].to_numpy(dtype=float)

    metrics: Dict[str, Any] = {
        "tau_days": float(tau_days),
        "n_total": int(len(predictions)),
        "n_binary_eligible": int(len(eligible)),
        "n_events_total": int((predictions["event_indicator"] == 1).sum()),
        "n_events_binary": int(np.nansum(labels)) if len(labels) else 0,
        "td_auroc": float("nan"),
        "brier_score": float("nan"),
        "ici": float("nan"),
        "mce": float("nan"),
        "c_index": float("nan"),
        "ipcw_auroc": float("nan"),
        "ipcw_brier": float("nan"),
    }

    if len(eligible) >= 2 and len(np.unique(labels)) >= 2:
        metrics["td_auroc"] = float(roc_auc_score(labels, probs))
        metrics["brier_score"] = float(brier_score_loss(labels, probs))
        metrics.update(compute_calibration_metrics(labels, probs))

    survival = compute_survival_metrics(
        times=predictions["time_to_event"].to_numpy(dtype=float),
        events=predictions["event_indicator"].to_numpy(dtype=int),
        predicted_risk=predictions["predicted_risk"].to_numpy(dtype=float),
        time_horizons=[float(tau_days)],
    )
    metrics["c_index"] = float(survival.get("concordance_index", float("nan")))
    horizon_key = f"{int(round(tau_days))}d"
    horizon_metrics = survival.get("per_horizon", {}).get(horizon_key, {})
    metrics["ipcw_auroc"] = float(horizon_metrics.get("ipcw_auc", float("nan")))
    metrics["ipcw_brier"] = float(horizon_metrics.get("ipcw_brier", float("nan")))
    return metrics


def bootstrap_comparison_metrics(
    predictions: pd.DataFrame,
    tau_days: float,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap confidence intervals for the comparison metrics."""
    rng = np.random.RandomState(seed)
    metric_names = [
        "td_auroc",
        "c_index",
        "ipcw_auroc",
        "ipcw_brier",
        "brier_score",
        "ici",
        "mce",
    ]
    values: Dict[str, List[float]] = {name: [] for name in metric_names}
    n = len(predictions)

    for _ in range(n_bootstrap):
        sampled = (
            predictions.iloc[rng.randint(0, n, size=n)].copy().reset_index(drop=True)
        )
        metrics = compute_comparison_metrics(sampled, tau_days=tau_days)
        for name in metric_names:
            value = metrics.get(name)
            if value is not None and np.isfinite(value):
                values[name].append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for name, metric_values in values.items():
        arr = np.asarray(metric_values, dtype=float)
        if len(arr) == 0:
            summary[name] = {
                "mean": float("nan"),
                "lower": float("nan"),
                "upper": float("nan"),
                "std": float("nan"),
            }
            continue
        summary[name] = {
            "mean": float(arr.mean()),
            "lower": float(np.quantile(arr, 0.025)),
            "upper": float(np.quantile(arr, 0.975)),
            "std": float(arr.std()),
        }
    return summary


def format_table_row(
    point_metrics: Mapping[str, Any],
    bootstrap_ci: Mapping[str, Mapping[str, float]],
) -> Dict[str, str]:
    """Format publication-friendly summary cells for LaTeX/table export."""

    def _fmt(metric_name: str) -> str:
        value = point_metrics.get(metric_name, float("nan"))
        ci = bootstrap_ci.get(metric_name, {})
        if not np.isfinite(value):
            return "NA"
        lower = ci.get("lower", float("nan"))
        upper = ci.get("upper", float("nan"))
        if np.isfinite(lower) and np.isfinite(upper):
            return f"{value:.3f} [{lower:.3f}, {upper:.3f}]"
        return f"{value:.3f}"

    return {
        "td_auroc": _fmt("td_auroc"),
        "c_index": _fmt("c_index"),
        "ipcw_auroc": _fmt("ipcw_auroc"),
        "ipcw_brier": _fmt("ipcw_brier"),
        "ici": _fmt("ici"),
        "mce": _fmt("mce"),
    }


def comparison_results_to_frame(results: Mapping[str, Any]) -> pd.DataFrame:
    """Convert nested comparison results into a flat table."""
    rows: List[Dict[str, Any]] = []
    for model_name, payload in results.get("models", {}).items():
        row = {
            "outcome_name": results.get("outcome_name"),
            "mode": results.get("mode"),
            "cohort_specification": results.get("cohort_specification"),
            "model": model_name,
            "backend": payload.get("backend"),
            "n_eval": payload.get("n_eval"),
            "n_binary_eligible": payload.get("n_binary_eligible"),
        }
        row.update(payload.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def make_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [make_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


__all__ = [
    "ComparisonRunner",
    "PreparedDataset",
    "PredictionPayload",
    "bootstrap_comparison_metrics",
    "comparison_results_to_frame",
    "compute_comparison_metrics",
    "derive_horizon_labels",
    "format_table_row",
    "infer_tau_days",
    "make_jsonable",
]
