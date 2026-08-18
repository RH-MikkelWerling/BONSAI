"""Canonical preparation of tabular feature matrices.

The feature export used by the DALY-CARE pipeline is commonly a pickle with
bookkeeping columns and feature families unavailable to sequence models.  This
module turns that export into a cached, auditable parquet suitable for matched
tabular comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEQUENCE_MATCHED_PROFILE = "sequence_matched"
SUPPORTED_PROFILES = frozenset({SEQUENCE_MATCHED_PROFILE})
BOOKKEEPING_COLUMNS = frozenset(
    {"timestamp", "prediction_time_uuid", "patientid"}
)
DEMOGRAPHIC_ALIASES = frozenset(
    {"age", "age_years", "age_at_index", "age_at_diagnosis", "sex", "matched_sex"}
)


def read_feature_table(path: str | Path) -> pd.DataFrame:
    """Read a pickle, parquet, or CSV feature table."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source, low_memory=False)
    raise ValueError(f"Unsupported feature-table format: {source}")


def _read_population(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source, low_memory=False)


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _align_subject_id_dtype(features: pd.DataFrame, population: pd.DataFrame) -> None:
    if pd.api.types.is_numeric_dtype(features["subject_id"]):
        population["subject_id"] = pd.to_numeric(
            population["subject_id"], errors="coerce"
        )
    else:
        features["subject_id"] = features["subject_id"].astype("string").str.strip()
        population["subject_id"] = (
            population["subject_id"].astype("string").str.strip()
        )


def _age_at_index(population: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in ("age_at_index", "age", "age_years", "age_at_diagnosis"):
        if column in population.columns:
            return pd.to_numeric(population[column], errors="coerce"), column
    if {"index_date", "birth_date"}.issubset(population.columns):
        index_date = pd.to_datetime(population["index_date"], errors="coerce")
        birth_date = pd.to_datetime(population["birth_date"], errors="coerce")
        return (index_date - birth_date).dt.days / 365.2425, "index_date-birth_date"
    raise ValueError(
        "Population metadata cannot supply age: expected age_at_index, age, "
        "age_years, age_at_diagnosis, or index_date plus birth_date."
    )


def _sex(population: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in ("matched_sex", "sex", "biological_sex", "gender"):
        if column in population.columns:
            return population[column].astype("string").str.strip(), column
    raise ValueError(
        "Population metadata cannot supply sex: expected matched_sex, sex, "
        "biological_sex, or gender."
    )


def _is_unmatched_sequence_feature(column: str) -> bool:
    name = str(column).lower()
    return "pred_rkkp" in name or "adverse_event" in name


def prepare_sequence_matched_features(
    source_path: str | Path,
    population_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Prepare or reuse a sequence-matched tabular feature parquet."""
    source = Path(source_path)
    population_source = Path(population_path)
    target = Path(output_path)
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    expected_inputs = {
        "profile": SEQUENCE_MATCHED_PROFILE,
        "source": _file_signature(source),
        "population": _file_signature(population_source),
    }
    if target.exists() and manifest_path.exists() and not overwrite:
        with open(manifest_path) as handle:
            cached = json.load(handle)
        if all(cached.get(key) == value for key, value in expected_inputs.items()):
            return target, cached

    features = read_feature_table(source)
    population = _read_population(population_source)
    original_shape = [int(features.shape[0]), int(features.shape[1])]
    if "subject_id" not in features.columns:
        if "patientid" not in features.columns:
            raise ValueError("Feature matrix contains neither subject_id nor patientid.")
        features = features.rename(columns={"patientid": "subject_id"})
    if "subject_id" not in population.columns:
        raise ValueError("Population metadata must contain subject_id.")
    _align_subject_id_dtype(features, population)
    features = features.dropna(subset=["subject_id"]).copy()
    population = population.dropna(subset=["subject_id"]).copy()
    if features["subject_id"].duplicated().any():
        raise ValueError("Feature matrix contains duplicate subject_id rows.")
    if population["subject_id"].duplicated().any():
        raise ValueError("Population metadata contains duplicate subject_id rows.")

    removed_bookkeeping = sorted(
        column for column in BOOKKEEPING_COLUMNS if column in features.columns
    )
    removed_unmatched = sorted(
        column
        for column in features.columns
        if column != "subject_id" and _is_unmatched_sequence_feature(column)
    )
    removed_demographics = sorted(
        column for column in DEMOGRAPHIC_ALIASES if column in features.columns
    )
    features = features.drop(
        columns=removed_bookkeeping + removed_unmatched + removed_demographics,
        errors="ignore",
    )

    age, age_source = _age_at_index(population)
    sex, sex_source = _sex(population)
    demographics = pd.DataFrame(
        {
            "subject_id": population["subject_id"],
            "age_at_index": age,
            "matched_sex": sex,
        }
    )
    demographics.loc[
        ~demographics["age_at_index"].between(0, 120), "age_at_index"
    ] = np.nan
    features = features.merge(
        demographics, on="subject_id", how="left", validate="one_to_one"
    )

    predictors = [column for column in features.columns if column != "subject_id"]
    removed_all_missing = sorted(
        column for column in predictors if features[column].isna().all()
    )
    features = features.drop(columns=removed_all_missing)
    predictors = [column for column in features.columns if column != "subject_id"]
    removed_constant = sorted(
        column for column in predictors if features[column].nunique(dropna=True) <= 1
    )
    features = features.drop(columns=removed_constant)
    if not any(column != "subject_id" for column in features.columns):
        raise ValueError("No predictors remain after sequence-matched preparation.")

    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(target, index=False)
    manifest: dict[str, Any] = {
        **expected_inputs,
        "output": str(target.resolve()),
        "original_shape": original_shape,
        "prepared_shape": [int(features.shape[0]), int(features.shape[1])],
        "age_source": age_source,
        "sex_source": sex_source,
        "removed_bookkeeping_columns": removed_bookkeeping,
        "removed_unmatched_columns": removed_unmatched,
        "removed_existing_demographic_columns": removed_demographics,
        "removed_all_missing_columns": removed_all_missing,
        "removed_constant_columns": removed_constant,
        "n_age_nonmissing": int(features["age_at_index"].notna().sum()),
        "n_sex_nonmissing": int(features["matched_sex"].notna().sum()),
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return target, manifest


def prepare_feature_matrix(
    source_path: str | Path,
    *,
    profile: str,
    population_path: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Prepare a source matrix under a named, cacheable feature profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unknown feature profile {profile!r}; expected one of "
            f"{sorted(SUPPORTED_PROFILES)}."
        )
    source = Path(source_path)
    directory = Path(output_dir) if output_dir else source.parent / "prepared"
    target = directory / f"{source.stem}__{profile}.parquet"
    return prepare_sequence_matched_features(
        source, population_path, target, overwrite=overwrite
    )
