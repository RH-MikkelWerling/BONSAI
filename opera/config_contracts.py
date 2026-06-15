"""Validated configuration contracts for OPERA orchestration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


class ConfigValidationError(ValueError):
    """Raised when an experiment config violates its declared contract."""

    def __init__(self, issues: list[str]):
        self.issues = tuple(issues)
        super().__init__("; ".join(issues))


def expand_config_values(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and platform environment variables."""
    if isinstance(value, str):
        value = re.sub(
            r"\$\{([^}]+)\}",
            lambda match: os.environ.get(match.group(1), match.group(0)),
            value,
        )
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_config_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config_values(item) for key, item in value.items()}
    return value


def _as_mapping(value: Any, path: str, issues: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        issues.append(f"{path} must be a mapping.")
        return {}
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    path: str,
    issues: list[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        issues.append(f"{path} has unknown fields: {unknown}")


def _optional_string(
    value: Any,
    path: str,
    issues: list[str],
) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    if not isinstance(value, str):
        issues.append(f"{path} must be a string or null.")
        return None
    return value


def _optional_date_string(
    value: Any,
    path: str,
    issues: list[str],
) -> Optional[str]:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return _optional_string(value, path, issues)


def _integer(value: Any, path: str, issues: list[str]) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(f"{path} must be an integer.")
        return None
    return value


@dataclass(frozen=True)
class CohortSpec:
    """One disease cohort participating in a sweep.

    ``training_cohort`` is optional and names the grouped disease cohort whose
    pre-trained checkpoint should be used when finetuning on this (possibly
    fine-grained) cohort.  When absent the cohort name itself is used as the
    training cohort key.  This supports the train-on-grouped / eval-on-fine
    paradigm where models are trained on, e.g., ``DLBCL_like`` and then
    evaluated separately on ``DLBCL``, ``BCL``, ``RT``, and ``RT_DERIVED``.
    """

    name: str
    data_dir: str
    ipi_score_col: Optional[str] = None
    registry_start_date: Optional[str] = None
    population_file: Optional[str] = None
    training_cohort: Optional[str] = None
    cohort_fine_col: Optional[str] = None
    cohort_fine_value: Optional[str] = None

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: Any,
        issues: list[str],
    ) -> "CohortSpec":
        path = f"cohorts.{name}"
        value = _as_mapping(raw, path, issues)
        _reject_unknown(
            value,
            {
                "data_dir",
                "ipi_score_col",
                "registry_start_date",
                "population_file",
                "training_cohort",
                "cohort_fine_col",
                "cohort_fine_value",
            },
            path,
            issues,
        )
        data_dir = value.get("data_dir")
        if not isinstance(data_dir, str) or not data_dir:
            issues.append(f"{path}.data_dir must be a non-empty string.")
            data_dir = ""
        training_cohort = _optional_string(
            value.get("training_cohort"),
            f"{path}.training_cohort",
            issues,
        )
        cohort_fine_col = _optional_string(
            value.get("cohort_fine_col"),
            f"{path}.cohort_fine_col",
            issues,
        )
        cohort_fine_value = _optional_string(
            value.get("cohort_fine_value"),
            f"{path}.cohort_fine_value",
            issues,
        )
        if (cohort_fine_col is None) != (cohort_fine_value is None):
            issues.append(
                f"{path}: cohort_fine_col and cohort_fine_value must both be "
                "set or both be absent."
            )
        return cls(
            name=name,
            data_dir=data_dir,
            ipi_score_col=_optional_string(
                value.get("ipi_score_col"),
                f"{path}.ipi_score_col",
                issues,
            ),
            registry_start_date=_optional_date_string(
                value.get("registry_start_date"),
                f"{path}.registry_start_date",
                issues,
            ),
            population_file=_optional_string(
                value.get("population_file"),
                f"{path}.population_file",
                issues,
            ),
            training_cohort=training_cohort,
            cohort_fine_col=cohort_fine_col,
            cohort_fine_value=cohort_fine_value,
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "data_dir": self.data_dir,
            "ipi_score_col": self.ipi_score_col,
            "registry_start_date": self.registry_start_date,
        }
        if self.population_file is not None:
            result["population_file"] = self.population_file
        if self.training_cohort is not None:
            result["training_cohort"] = self.training_cohort
        if self.cohort_fine_col is not None:
            result["cohort_fine_col"] = self.cohort_fine_col
        if self.cohort_fine_value is not None:
            result["cohort_fine_value"] = self.cohort_fine_value
        return result


@dataclass(frozen=True)
class OutcomeSpec:
    """One outcome task and its source event-time definition."""

    name: str
    outcome_file: str
    n_hours_start_include: int = 1
    n_hours_end_include: Optional[int] = None
    competing_outcome_file: Optional[str] = None
    competing_outcome_path: Optional[str] = None
    registry_start_date: Optional[str] = None
    registry_start_date_configured: bool = False
    eligibility_file: Optional[str] = None

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: Any,
        issues: list[str],
    ) -> "OutcomeSpec":
        path = f"outcomes.{name}"
        value = _as_mapping(raw, path, issues)
        _reject_unknown(
            value,
            {
                "outcome_file",
                "filename",
                "n_hours_start_include",
                "n_hours_end_include",
                "competing_outcome_file",
                "competing_outcome_path",
                "registry_start_date",
                "eligibility_file",
            },
            path,
            issues,
        )
        outcome_file = value.get("outcome_file", value.get("filename"))
        if outcome_file is None:
            outcome_file = f"{name}.parquet"
        if not isinstance(outcome_file, str) or not outcome_file:
            issues.append(f"{path}.outcome_file must be a non-empty string.")
            outcome_file = f"{name}.parquet"

        start = _integer(
            value.get("n_hours_start_include", 1),
            f"{path}.n_hours_start_include",
            issues,
        )
        if start is None:
            start = 1
        elif start < 0:
            issues.append(f"{path}.n_hours_start_include must be non-negative.")

        end_raw = value.get("n_hours_end_include")
        end = None
        if end_raw is not None:
            end = _integer(end_raw, f"{path}.n_hours_end_include", issues)
            if end is not None and end <= start:
                issues.append(
                    f"{path}.n_hours_end_include must be greater than "
                    "n_hours_start_include."
                )

        return cls(
            name=name,
            outcome_file=outcome_file,
            n_hours_start_include=start,
            n_hours_end_include=end,
            competing_outcome_file=_optional_string(
                value.get("competing_outcome_file"),
                f"{path}.competing_outcome_file",
                issues,
            ),
            competing_outcome_path=_optional_string(
                value.get("competing_outcome_path"),
                f"{path}.competing_outcome_path",
                issues,
            ),
            registry_start_date=_optional_date_string(
                value.get("registry_start_date"),
                f"{path}.registry_start_date",
                issues,
            ),
            registry_start_date_configured="registry_start_date" in value,
            eligibility_file=_optional_string(
                value.get("eligibility_file"),
                f"{path}.eligibility_file",
                issues,
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "outcome_file": self.outcome_file,
            "n_hours_start_include": self.n_hours_start_include,
            "n_hours_end_include": self.n_hours_end_include,
        }
        for key, value in (
            ("competing_outcome_file", self.competing_outcome_file),
            ("competing_outcome_path", self.competing_outcome_path),
            ("eligibility_file", self.eligibility_file),
        ):
            if value is not None:
                result[key] = value
        if self.registry_start_date_configured:
            result["registry_start_date"] = self.registry_start_date
        return result


@dataclass(frozen=True)
class VariantSpec:
    """One model or external-prediction variant in an evaluation sweep."""

    name: str
    values: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: Any,
        issues: list[str],
    ) -> "VariantSpec":
        path = f"model_variants.{name}"
        value = dict(_as_mapping(raw, path, issues))
        _reject_unknown(
            value,
            {
                "encoder_ckpt",
                "encoder_source",
                "results_file",
                "predictions_file",
                "training_mode",
                "training_stage",
                "pretraining_scope",
                "opera_scope",
                "extra_overrides",
                "seed",
                "model_family",
                "pos_weight",
                "include_outcomes",
                "exclude_outcomes",
            },
            path,
            issues,
        )

        encoder_source = value.get("encoder_source")
        allowed_sources = {
            "pretrain",
            "dapt",
            "contrastive",
            "joint",
            "mol",
            "random_init",
        }
        if encoder_source is not None and encoder_source not in allowed_sources:
            issues.append(
                f"{path}.encoder_source must be one of {sorted(allowed_sources)}."
            )

        training_mode = value.get("training_mode")
        if training_mode not in (None, "cox", "ipcw_bce"):
            issues.append(
                f"Variant {name!r} has invalid training_mode={training_mode!r}."
            )

        sources = [
            key
            for key in ("results_file", "predictions_file")
            if value.get(key) not in (None, "")
        ]
        if len(sources) > 1:
            issues.append(
                f"{path} must not configure both results_file and predictions_file."
            )
        if value.get("encoder_ckpt") and not encoder_source:
            issues.append(f"{path}.encoder_source is required with encoder_ckpt.")
        if not sources and not value.get("encoder_ckpt"):
            if encoder_source != "random_init":
                issues.append(
                    f"{path} must configure encoder_ckpt, results_file, "
                    "predictions_file, or encoder_source=random_init."
                )

        overrides = value.get("extra_overrides")
        if overrides is not None and (
            not isinstance(overrides, list)
            or any(not isinstance(item, str) for item in overrides)
        ):
            issues.append(f"{path}.extra_overrides must be a list of strings.")
        for key in ("include_outcomes", "exclude_outcomes"):
            names = value.get(key)
            if names is not None and (
                not isinstance(names, list)
                or any(not isinstance(item, str) for item in names)
            ):
                issues.append(f"{path}.{key} must be a list of outcome names.")

        seed = value.get("seed")
        if seed is not None:
            _integer(seed, f"{path}.seed", issues)
        return cls(name=name, values=value)

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class SweepConfig:
    """Canonical validated contract consumed by readiness and execution."""

    cohorts: Mapping[str, CohortSpec]
    outcomes: Mapping[str, OutcomeSpec]
    model_variants: Mapping[str, VariantSpec]
    output_dir: str = "./sweep_results"
    finetune_base_config: str = "opera/configs/finetune.yaml"
    seeds: tuple[int, ...] = (42,)
    rarity_mode: str = "none"
    baseline_model: Optional[str] = None
    rarity: Mapping[str, Any] = field(default_factory=dict)
    paths: Mapping[str, Any] = field(default_factory=dict)
    subgroups: Mapping[str, Any] = field(default_factory=dict)
    test_key: str = "held_out"
    run_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, raw: Any) -> "SweepConfig":
        issues: list[str] = []
        value = _as_mapping(raw, "config", issues)
        _reject_unknown(
            value,
            {
                "cohorts",
                "outcomes",
                "model_variants",
                "output_dir",
                "finetune_base_config",
                "seeds",
                "seed",
                "rarity_mode",
                "baseline_model",
                "rarity",
                "paths",
                "subgroups",
                "test_key",
                "run_id",
            },
            "config",
            issues,
        )

        raw_cohorts = _as_mapping(value.get("cohorts"), "cohorts", issues)
        raw_variants = _as_mapping(
            value.get("model_variants"),
            "model_variants",
            issues,
        )
        if not raw_cohorts:
            issues.append("Missing or empty required block: cohorts")
        if not raw_variants:
            issues.append("Missing or empty required block: model_variants")

        raw_outcomes = value.get("outcomes")
        if isinstance(raw_outcomes, list):
            raw_outcomes = {name: {} for name in raw_outcomes if isinstance(name, str)}
        raw_outcomes = _as_mapping(raw_outcomes, "outcomes", issues)
        if not raw_outcomes:
            issues.append("Missing or empty required block: outcomes")

        cohorts = {
            str(name): CohortSpec.from_mapping(str(name), item, issues)
            for name, item in raw_cohorts.items()
        }
        outcomes = {
            str(name): OutcomeSpec.from_mapping(str(name), item, issues)
            for name, item in raw_outcomes.items()
        }
        variants = {
            str(name): VariantSpec.from_mapping(str(name), item, issues)
            for name, item in raw_variants.items()
        }
        outcome_names = set(outcomes)
        for name, variant in variants.items():
            for key in ("include_outcomes", "exclude_outcomes"):
                configured = set(variant.values.get(key) or [])
                unknown = sorted(configured - outcome_names)
                if unknown:
                    issues.append(
                        f"model_variants.{name}.{key} references unknown "
                        f"outcomes: {unknown}"
                    )
            overlap = set(variant.values.get("include_outcomes") or []) & set(
                variant.values.get("exclude_outcomes") or []
            )
            if overlap:
                issues.append(
                    f"model_variants.{name} includes and excludes the same "
                    f"outcomes: {sorted(overlap)}"
                )
        for name, variant in variants.items():
            if (
                variant.values.get("training_mode") == "cox"
                and variant.values.get("pos_weight") is not None
            ):
                issues.append(
                    f"Variant {name!r} uses Cox training with pos_weight configured."
                )
        if any(
            variant.values.get("training_mode") == "ipcw_bce"
            for variant in variants.values()
        ):
            for variant_name, variant in variants.items():
                if variant.values.get("training_mode") != "ipcw_bce":
                    continue
                for name, outcome in outcomes.items():
                    if not variant_applies_to_outcome(variant.values, name):
                        continue
                    if outcome.n_hours_end_include is not None:
                        continue
                    issues.append(
                        f"Outcome {name!r} selected by variant "
                        f"{variant_name!r} has no n_hours_end_include; "
                        "IPCW-BCE variants require a fixed horizon."
                    )

        seeds_raw = value.get("seeds")
        if seeds_raw is None:
            seeds_raw = [value.get("seed", 42)]
        if not isinstance(seeds_raw, list) or not seeds_raw:
            issues.append("config.seeds must be a non-empty list of integers.")
            seeds = (42,)
        else:
            parsed_seeds = []
            for index, seed in enumerate(seeds_raw):
                parsed = _integer(seed, f"config.seeds[{index}]", issues)
                if parsed is not None:
                    parsed_seeds.append(parsed)
            seeds = tuple(parsed_seeds or [42])
            if len(set(seeds)) != len(seeds):
                issues.append("config.seeds must not contain duplicates.")

        rarity = dict(_as_mapping(value.get("rarity", {}), "rarity", issues))
        _reject_unknown(
            rarity,
            {"mode", "tier", "baseline_model", "size_metadata"},
            "rarity",
            issues,
        )
        rarity_mode = value.get("rarity_mode", rarity.get("mode", "none"))
        if rarity_mode not in {"none", "synthetic", "real"}:
            issues.append("config.rarity_mode must be none, synthetic, or real.")

        paths = dict(_as_mapping(value.get("paths", {}), "paths", issues))
        _reject_unknown(paths, {"subgroups"}, "paths", issues)
        subgroups = dict(_as_mapping(value.get("subgroups", {}), "subgroups", issues))
        _reject_unknown(subgroups, {"path", "columns"}, "subgroups", issues)
        columns = subgroups.get("columns", [])
        if not isinstance(columns, list) or any(
            not isinstance(item, str) for item in columns
        ):
            issues.append("subgroups.columns must be a list of strings.")

        output_dir = value.get("output_dir", "./sweep_results")
        base_config = value.get(
            "finetune_base_config",
            "opera/configs/finetune.yaml",
        )
        test_key = value.get("test_key", "held_out")
        for item, item_path in (
            (output_dir, "config.output_dir"),
            (base_config, "config.finetune_base_config"),
            (test_key, "config.test_key"),
        ):
            if not isinstance(item, str) or not item:
                issues.append(f"{item_path} must be a non-empty string.")

        baseline_model = _optional_string(
            value.get("baseline_model", rarity.get("baseline_model")),
            "config.baseline_model",
            issues,
        )
        run_id = _optional_string(value.get("run_id"), "config.run_id", issues)

        if issues:
            raise ConfigValidationError(issues)
        return cls(
            cohorts=cohorts,
            outcomes=outcomes,
            model_variants=variants,
            output_dir=output_dir,
            finetune_base_config=base_config,
            seeds=seeds,
            rarity_mode=rarity_mode,
            baseline_model=baseline_model,
            rarity=rarity,
            paths=paths,
            subgroups=subgroups,
            test_key=test_key,
            run_id=run_id,
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cohorts": {name: spec.to_mapping() for name, spec in self.cohorts.items()},
            "outcomes": {
                name: spec.to_mapping() for name, spec in self.outcomes.items()
            },
            "model_variants": {
                name: spec.to_mapping() for name, spec in self.model_variants.items()
            },
            "output_dir": self.output_dir,
            "finetune_base_config": self.finetune_base_config,
            "seeds": list(self.seeds),
            "rarity_mode": self.rarity_mode,
            "rarity": dict(self.rarity),
            "paths": dict(self.paths),
            "subgroups": dict(self.subgroups),
            "test_key": self.test_key,
        }
        if self.baseline_model is not None:
            result["baseline_model"] = self.baseline_model
        if self.run_id is not None:
            result["run_id"] = self.run_id
        return result


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load, expand, and validate a sweep YAML file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return SweepConfig.from_mapping(expand_config_values(raw))


_MANIFEST_TOP_LEVEL_KEYS = {
    "version",
    "primary_endpoints",
    "exploratory_outcomes",
    "primary_contrasts",
    "multiplicity",
    "min_events",
    "seeds",
    "cohort_analysis",
    "confirmatory_fine_cohorts",
    "calibration",
}

_MANIFEST_MULTIPLICITY_METHODS = {"benjamini_hochberg", "bonferroni", "holm"}


def validate_analysis_manifest(path: str | Path) -> dict:
    """Load and validate the locked analysis manifest.

    Returns the parsed manifest dict if valid.
    Raises ConfigValidationError if required fields are missing or invalid.
    """
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    issues: list[str] = []
    manifest = _as_mapping(raw, "manifest", issues)
    _reject_unknown(manifest, _MANIFEST_TOP_LEVEL_KEYS, "manifest", issues)

    if not isinstance(manifest.get("version"), str):
        issues.append("manifest.version must be a string.")

    primary_endpoints = manifest.get("primary_endpoints")
    if (
        not isinstance(primary_endpoints, list)
        or not primary_endpoints
        or any(not isinstance(item, str) for item in primary_endpoints)
    ):
        issues.append("manifest.primary_endpoints must be a non-empty list of strings.")

    primary_contrasts = manifest.get("primary_contrasts")
    if not isinstance(primary_contrasts, list) or not primary_contrasts:
        issues.append(
            "manifest.primary_contrasts must be a non-empty list of "
            "[reference, comparator] pairs."
        )
    else:
        for index, pair in enumerate(primary_contrasts):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(item, str) for item in pair)
            ):
                issues.append(
                    f"manifest.primary_contrasts[{index}] must be a pair of "
                    "two strings."
                )

    multiplicity = _as_mapping(
        manifest.get("multiplicity", {}),
        "manifest.multiplicity",
        issues,
    )
    method = multiplicity.get("method")
    if method not in _MANIFEST_MULTIPLICITY_METHODS:
        issues.append(
            "manifest.multiplicity.method must be one of "
            f"{sorted(_MANIFEST_MULTIPLICITY_METHODS)}."
        )
    alpha = multiplicity.get("alpha")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not (0 < float(alpha) < 1)
    ):
        issues.append("manifest.multiplicity.alpha must be a float in (0, 1).")

    min_events = _as_mapping(
        manifest.get("min_events", {}),
        "manifest.min_events",
        issues,
    )
    for key in ("test", "train"):
        value = _integer(
            min_events.get(key),
            f"manifest.min_events.{key}",
            issues,
        )
        if value is not None and value < 1:
            issues.append(f"manifest.min_events.{key} must be >= 1.")

    seeds = manifest.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)
    ):
        issues.append("manifest.seeds must be a non-empty list of integers.")
    elif len(set(seeds)) != len(seeds):
        issues.append("manifest.seeds must not contain duplicates.")

    if issues:
        raise ConfigValidationError(issues)
    return dict(manifest)


def variant_applies_to_outcome(
    variant: Mapping[str, Any],
    outcome_name: str,
) -> bool:
    """Return whether a configured model variant should run for an outcome."""
    included = variant.get("include_outcomes")
    if included is not None and outcome_name not in included:
        return False
    return outcome_name not in (variant.get("exclude_outcomes") or [])
