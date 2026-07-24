"""Label-only support preflight for the focused OPERA outcome-transfer study.

This module deliberately does *not* construct a model, load a checkpoint, or
run an evaluator.  It resolves the canonical transfer manifest and then
applies the exact fixed-horizon label rules used by the downstream binary
tasks.  The resulting report makes low-support outcomes visible before any
contrastive training is launched.

The shared-data contract is:

* one global outcome parquet per target;
* one shared membership table containing ``subject_id`` and
  ``cohort_grouped``;
* a global death outcome used as the competing event for non-death targets.

The report contains one row for all eligible hematology patients and one row
per canonical grouped cohort, for every transfer target/condition/horizon.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from opera.evaluation.cohort_flow import load_eligibility_frame
from opera.functional.outcomes import (
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
)
from opera.functional.outcome_transfer import REPOSITORY_ROOT
from opera.run.generate_sweep_configs import load_registry


DEFAULT_MANIFEST = Path("opera/configs/manifests/outcome_transfer.yaml")
DEFAULT_REGISTRY = Path("opera/configs/experiment_registry.yaml")
# Support counts are data-dependent and may be sensitive.  Keep them under the
# results root rather than beside checked-in generated configs.
DEFAULT_OUTPUT_DIR = "${BONSAI_RESULTS_ROOT}/outcome_transfer/preflight"
ALL_HEMATOLOGY = "all_hematology"
SUPPORT_CSV_NAME = "outcome_transfer_support.csv"
SUPPORT_JSON_NAME = "outcome_transfer_support.json"
# These are prespecified related-outcome primary targets, not a support-based
# selection rule.  A low count remains in the report and is flagged for data
# review rather than silently removed.
REQUIRED_PRIMARY_TARGETS = frozenset({"any_transfusion", "hospitalisation"})


class OutcomeTransferPreflightError(ValueError):
    """Raised when the label-only transfer support audit cannot be trusted."""


@dataclass(frozen=True)
class OutcomeSource:
    """Resolved shared-data sources and label settings for one endpoint."""

    outcome_path: Path
    competing_path: Path | None
    eligibility_path: Path | None
    registry_start_date: Any | None
    n_hours_start_include: int


def _expand_path(value: str | Path) -> Path:
    """Expand shell environment values while retaining normal ``Path`` use."""
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve_repository_or_cwd_path(value: str | Path) -> Path:
    """Prefer an explicit local path, then resolve checked-in registry paths."""
    candidate = _expand_path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPOSITORY_ROOT / candidate


def _has_unresolved_environment(value: str | Path) -> bool:
    text = str(value)
    return "${" in text or "$" in text or "%" in text


def _expand_output_dir(value: str | Path) -> Path:
    """Resolve a result destination without silently writing into the repo."""
    expanded = _expand_path(value)
    if _has_unresolved_environment(expanded):
        raise OutcomeTransferPreflightError(
            "Outcome-transfer support output_dir contains an unresolved environment "
            f"variable: {value}. Set BONSAI_RESULTS_ROOT or pass --output-dir."
        )
    return expanded


def _read_table(path: Path) -> pd.DataFrame:
    """Read a supported global table without assuming Parquet in tests."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise OutcomeTransferPreflightError(
        f"Unsupported table extension {path.suffix!r} for {path}; "
        "expected parquet or CSV."
    )


def _path_from_raw(raw: str | Path, root: Path) -> Path:
    path = _expand_path(raw)
    return path if path.is_absolute() else root / path


def _outcome_metadata(registry: Mapping[str, Any], outcome: str) -> Mapping[str, Any]:
    """Return optional registry metadata without maintaining a second taxonomy.

    The current production registry has a compact list of outcomes and common
    global paths.  Supporting the mapping spellings below keeps the preflight
    forward compatible with an explicit outcome-file/eligibility registry
    without duplicating its contents in Python.
    """
    for field in ("outcome_definitions", "outcome_configs", "outcome_metadata"):
        value = registry.get(field)
        if isinstance(value, Mapping) and isinstance(value.get(outcome), Mapping):
            return value[outcome]
    return {}


def _resolve_outcome_source(registry: Mapping[str, Any], outcome: str) -> OutcomeSource:
    """Resolve global outcome, competing-event, and optional eligibility paths."""
    paths = registry.get("paths")
    if not isinstance(paths, Mapping) or not paths.get("outcomes_dir"):
        raise OutcomeTransferPreflightError(
            "Registry paths.outcomes_dir is required for outcome-transfer preflight."
        )
    outcomes_dir = _expand_path(paths["outcomes_dir"])
    metadata = _outcome_metadata(registry, outcome)

    raw_outcome = metadata.get("outcome_file", f"{outcome}.parquet")
    outcome_path = _path_from_raw(raw_outcome, outcomes_dir)

    death_outcome = registry["death_outcome"]
    raw_competing = metadata.get("competing_outcome_path") or metadata.get(
        "competing_outcome_file"
    )
    if raw_competing in (None, "", "null") and outcome != death_outcome:
        raw_competing = f"{death_outcome}.parquet"
    competing_path = (
        None
        if raw_competing in (None, "", "null")
        else _path_from_raw(raw_competing, outcomes_dir)
    )

    raw_eligibility = metadata.get("eligibility_file")
    if raw_eligibility in (None, "", "null"):
        eligibility_files = registry.get("eligibility_files")
        if isinstance(eligibility_files, Mapping):
            raw_eligibility = eligibility_files.get(outcome)
    eligibility_path = (
        None
        if raw_eligibility in (None, "", "null")
        else _path_from_raw(raw_eligibility, outcomes_dir)
    )

    raw_start = metadata.get("n_hours_start_include", 1)
    if isinstance(raw_start, bool) or not isinstance(raw_start, int) or raw_start < 0:
        raise OutcomeTransferPreflightError(
            f"Outcome {outcome!r} has invalid n_hours_start_include={raw_start!r}."
        )
    return OutcomeSource(
        outcome_path=outcome_path,
        competing_path=competing_path,
        eligibility_path=eligibility_path,
        registry_start_date=metadata.get("registry_start_date"),
        n_hours_start_include=raw_start,
    )


def _load_split_keys(split_contract: str | Path) -> dict[str, str]:
    path = _expand_path(split_contract)
    if not path.exists() and not path.is_absolute():
        repository_candidate = Path(__file__).resolve().parents[2] / path
        if repository_candidate.exists():
            path = repository_candidate
    if not path.exists():
        raise OutcomeTransferPreflightError(
            f"Transfer split contract does not exist: {path}"
        )
    with path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle) or {}
    keys = {
        "train": contract.get("train_key", "train"),
        "tuning": contract.get("val_key", contract.get("tuning_key", "tuning")),
        "held_out": contract.get("test_key", "held_out"),
    }
    if len(set(keys.values())) != len(keys):
        raise OutcomeTransferPreflightError(
            f"Transfer split contract has non-distinct split keys: {keys}."
        )
    return {name: str(value) for name, value in keys.items()}


def _load_membership(
    registry: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, set[Any]]]:
    paths = registry.get("paths")
    columns = registry.get("cohort_columns")
    groups = registry.get("cohort_groups")
    if not isinstance(paths, Mapping) or not paths.get("cohort_membership_file"):
        raise OutcomeTransferPreflightError(
            "Registry paths.cohort_membership_file is required for transfer preflight."
        )
    if not isinstance(columns, Mapping) or not columns.get("grouped"):
        raise OutcomeTransferPreflightError(
            "Registry cohort_columns.grouped is required for transfer preflight."
        )
    if not isinstance(groups, Mapping) or not groups:
        raise OutcomeTransferPreflightError(
            "Registry cohort_groups is required for transfer preflight."
        )

    membership_path = _expand_path(paths["cohort_membership_file"])
    if not membership_path.exists():
        raise OutcomeTransferPreflightError(
            f"Cohort membership file does not exist: {membership_path}"
        )
    membership = _read_table(membership_path)
    grouped_col = str(columns["grouped"])
    required = {"subject_id", grouped_col}
    missing = required - set(membership.columns)
    if missing:
        raise OutcomeTransferPreflightError(
            f"Membership file {membership_path} is missing columns: {sorted(missing)}"
        )
    if membership["subject_id"].isna().any():
        raise OutcomeTransferPreflightError(
            f"Membership file {membership_path} contains missing subject_id values."
        )
    duplicate_count = int(membership["subject_id"].duplicated().sum())
    if duplicate_count:
        raise OutcomeTransferPreflightError(
            f"Membership file {membership_path} contains {duplicate_count} "
            "duplicate subject_id rows."
        )

    known_groups = list(groups)
    grouped_values = membership[grouped_col].astype(str)
    unknown = sorted(set(grouped_values) - set(known_groups))
    if unknown:
        raise OutcomeTransferPreflightError(
            "Membership contains cohort_grouped values not present in the canonical "
            f"registry: {unknown}."
        )
    if membership.empty:
        raise OutcomeTransferPreflightError(
            "Membership file contains no hematology patients."
        )

    ids_by_group = {
        group: set(membership.loc[grouped_values == group, "subject_id"].tolist())
        for group in known_groups
    }
    empty_groups = [group for group, ids in ids_by_group.items() if not ids]
    if empty_groups:
        raise OutcomeTransferPreflightError(
            f"Membership has no patients for canonical grouped cohorts: {empty_groups}."
        )
    return membership, ids_by_group


def _availability_by_group(registry: Mapping[str, Any]) -> dict[str, set[str]]:
    result = {str(group): set() for group in registry["cohort_groups"]}
    for rule in registry.get("availability_rules", {}).values():
        for group in rule.get("excluded_grouped", []):
            result[str(group)].update(str(value) for value in rule.get("outcomes", []))
    return result


def _plan_availability_by_group(
    plan: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, set[str]]:
    """Use resolver-expanded structural exclusions when available.

    The registry remains the fallback for focused synthetic tests and older
    resolved plans.  Production reports consume the explicit plan field so
    their availability matrix is recorded alongside transfer condition logic.
    """
    fallback = _availability_by_group(registry)
    raw = plan.get("structural_outcome_exclusions_by_group")
    if not isinstance(raw, Mapping):
        return fallback
    result = {group: set(values) for group, values in fallback.items()}
    for group, outcomes in raw.items():
        if group not in result:
            raise OutcomeTransferPreflightError(
                "Resolved transfer plan has structural exclusions for unknown "
                f"cohort_grouped {group!r}."
            )
        if not isinstance(outcomes, list) or not all(
            isinstance(value, str) for value in outcomes
        ):
            raise OutcomeTransferPreflightError(
                "Resolved transfer plan structural cohort exclusions must be "
                f"lists of outcome names; got {outcomes!r} for {group!r}."
            )
        result[group] = set(outcomes)
    return result


def _filtered_outcome_frame(
    *,
    source: OutcomeSource,
    outcome: str,
    all_ids: set[Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load and apply structural eligibility, leaving censoring for labels.

    The ascertainment eligibility scope intentionally retains rows that are
    structurally valid but censored before the horizon.  Those rows are needed
    to report ``n_*_censored_before_horizon`` and are then removed by the same
    fixed-horizon label rule as downstream probes.
    """
    if not source.outcome_path.exists():
        raise OutcomeTransferPreflightError(
            f"Outcome file for {outcome!r} does not exist: {source.outcome_path}"
        )
    frame = _read_table(source.outcome_path)
    required = {"subject_id", "split", "index_date", "censor_date"}
    missing = required - set(frame.columns)
    if missing:
        raise OutcomeTransferPreflightError(
            f"Outcome {outcome!r} is missing required columns: {sorted(missing)}"
        )
    frame = frame.loc[frame["subject_id"].isin(all_ids)].copy()
    if frame.empty:
        raise OutcomeTransferPreflightError(
            f"Outcome {outcome!r} has no rows for canonical hematology membership."
        )
    duplicate_count = int(frame["subject_id"].duplicated().sum())
    if duplicate_count:
        raise OutcomeTransferPreflightError(
            f"Outcome {outcome!r} has {duplicate_count} duplicate subject_id rows "
            "after membership filtering."
        )

    if source.eligibility_path is not None:
        if not source.eligibility_path.exists():
            raise OutcomeTransferPreflightError(
                f"Eligibility file for {outcome!r} does not exist: "
                f"{source.eligibility_path}"
            )
        eligibility = load_eligibility_frame(source.eligibility_path)
        frame = filter_outcome_eligibility(
            frame,
            eligibility,
            cohort=ALL_HEMATOLOGY,
            outcome_name=outcome,
            eligibility_scope="ascertainment",
        )
    frame = filter_registry_eligible_outcomes(
        frame,
        source.registry_start_date,
        cohort=ALL_HEMATOLOGY,
        outcome_name=outcome,
    )

    competing: pd.DataFrame | None = None
    if source.competing_path is not None:
        if not source.competing_path.exists():
            raise OutcomeTransferPreflightError(
                f"Competing-event file for {outcome!r} does not exist: "
                f"{source.competing_path}"
            )
        competing = _read_table(source.competing_path)
        missing = {"subject_id", "outcome_date"} - set(competing.columns)
        if missing:
            raise OutcomeTransferPreflightError(
                f"Competing-event file for {outcome!r} is missing columns: "
                f"{sorted(missing)}"
            )
        competing = competing.loc[competing["subject_id"].isin(all_ids)].copy()
    return frame, competing


def _label_counts(
    frame: pd.DataFrame,
    *,
    competing: pd.DataFrame | None,
    split_key: str,
    n_hours_start_include: int,
    n_hours_end_include: int,
) -> dict[str, int]:
    """Count labels with and without minimum follow-up at one split/horizon."""
    from opera.compat.bonsai import binarize_outcomes

    split_frame = frame.loc[frame["split"].astype(str) == split_key].copy()
    if split_frame.empty:
        return {
            "n": 0,
            "events": 0,
            "non_events": 0,
            "competing_events": 0,
            "censored_before_horizon": 0,
        }
    raw_records = binarize_outcomes(
        split_frame,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        require_min_followup=False,
        competing_event_df=competing,
    )
    retained_records = binarize_outcomes(
        split_frame,
        n_hours_start_include=n_hours_start_include,
        n_hours_end_include=n_hours_end_include,
        require_min_followup=True,
        competing_event_df=competing,
    )
    events = [record["event"] for record in retained_records.values()]
    # ``binarize_outcomes`` represents a competing death as ``event == 2``
    # and ``label == 0``.  It is therefore part of the fixed-horizon binary
    # probe denominator and must count as a non-event for the support
    # category, while remaining visible in its own competing-event column.
    labels = [record["label"] for record in retained_records.values()]
    return {
        "n": int(len(retained_records)),
        "events": int(sum(event == 1 for event in events)),
        "non_events": int(sum(label == 0 for label in labels)),
        "competing_events": int(sum(event == 2 for event in events)),
        "censored_before_horizon": int(len(raw_records) - len(retained_records)),
    }


def _support_category(events: int, non_events: int) -> str:
    minority = min(events, non_events)
    if minority >= 25:
        return "primary_candidate"
    if minority >= 10:
        return "partial_pooling_candidate"
    return "descriptive_only"


def _family_lookup(registry: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(outcome): str(family)
        for family, outcomes in registry["outcome_families"].items()
        for outcome in outcomes
    }


def _condition_target_requests(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand held-out conditions plus matched direct-supervision references.

    ``opera_full`` intentionally lists all 87 endpoints.  The support audit is
    focused on the transfer study's target union, so it emits full-OPERA rows
    only for targets that are actually compared to a held-out condition.
    """
    conditions = plan.get("conditions")
    if not isinstance(conditions, Mapping) or "opera_full" not in conditions:
        raise OutcomeTransferPreflightError(
            "Resolved transfer plan must contain conditions including opera_full."
        )
    requests: list[dict[str, Any]] = []
    direct_refs: dict[tuple[str, int], set[str]] = {}
    for condition_name, condition in conditions.items():
        if condition_name == "opera_full":
            continue
        horizon = condition.get("primary_horizon_days")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise OutcomeTransferPreflightError(
                f"Condition {condition_name!r} has invalid primary_horizon_days={horizon!r}."
            )
        for target in condition.get("evaluation_outcomes", []):
            target_name = str(target)
            requests.append(
                {
                    "condition_name": str(condition_name),
                    "condition": condition,
                    "target_outcome": target_name,
                    "primary_horizon_days": horizon,
                    "support_role": "held_out_condition",
                    "reference_for_conditions": None,
                }
            )
            direct_refs.setdefault((target_name, horizon), set()).add(
                str(condition_name)
            )

    full = conditions["opera_full"]
    for (target, horizon), source_conditions in sorted(direct_refs.items()):
        requests.append(
            {
                "condition_name": "opera_full",
                "condition": full,
                "target_outcome": target,
                "primary_horizon_days": horizon,
                "support_role": "direct_supervision_reference",
                "reference_for_conditions": ";".join(sorted(source_conditions)),
            }
        )

    union = set(plan.get("evaluation_target_union", []))
    emitted = {item["target_outcome"] for item in requests}
    if union and emitted != union:
        raise OutcomeTransferPreflightError(
            "Resolved plan evaluation_target_union does not match condition target "
            f"coverage: union={sorted(union)}, emitted={sorted(emitted)}."
        )
    return requests


def _row(
    *,
    request: Mapping[str, Any],
    target_family: str,
    evaluation_level: str,
    evaluation_group: str,
    availability: str,
    split_counts: Mapping[str, Mapping[str, int]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    condition = request["condition"]
    held_out = split_counts["held_out"]
    support_category = _support_category(held_out["events"], held_out["non_events"])
    required_primary = (
        request["target_outcome"] in REQUIRED_PRIMARY_TARGETS
        and evaluation_level == "all_hematology"
    )
    return {
        "target_outcome": request["target_outcome"],
        "target_family": target_family,
        "transfer_condition": request["condition_name"],
        "transfer_level": condition["transfer_level"],
        "support_role": request["support_role"],
        "reference_for_conditions": request["reference_for_conditions"],
        "primary_horizon_days": int(request["primary_horizon_days"]),
        "evaluation_level": evaluation_level,
        "evaluation_group": evaluation_group,
        "structural_availability": availability,
        "condition_blocked": bool(condition.get("launch_blocked", False)),
        "launch_blocked_reason": condition.get("launch_blocked_reason"),
        "dependency_resolution_status": condition.get(
            "dependency_resolution_status", "not_applicable"
        ),
        "n_train": split_counts["train"]["n"],
        "n_train_events": split_counts["train"]["events"],
        "n_train_non_events": split_counts["train"]["non_events"],
        "n_train_competing_events": split_counts["train"]["competing_events"],
        "n_train_censored_before_horizon": split_counts["train"][
            "censored_before_horizon"
        ],
        "n_tuning": split_counts["tuning"]["n"],
        "n_tuning_events": split_counts["tuning"]["events"],
        "n_tuning_non_events": split_counts["tuning"]["non_events"],
        "n_tuning_competing_events": split_counts["tuning"]["competing_events"],
        "n_tuning_censored_before_horizon": split_counts["tuning"][
            "censored_before_horizon"
        ],
        "n_held_out": held_out["n"],
        "n_held_out_events": held_out["events"],
        "n_held_out_non_events": held_out["non_events"],
        "n_competing_events": held_out["competing_events"],
        "n_censored_before_horizon": held_out["censored_before_horizon"],
        "held_out_minority_class": min(held_out["events"], held_out["non_events"]),
        "support_category": support_category,
        "required_primary_candidate": required_primary,
        "required_primary_candidate_met": (
            support_category == "primary_candidate" if required_primary else None
        ),
        "support_requirement_note": (
            None
            if not required_primary or support_category == "primary_candidate"
            else (
                "Prespecified related-outcome primary target lacks >=25 held-out "
                "events and >=25 held-out non-events; review the source labels and "
                "denominator before changing the experiment."
            )
        ),
        "target_seen_in_training_outcomes": request["target_outcome"]
        in set(condition.get("training_outcomes", [])),
        "registry_hash": plan.get("registry_hash"),
        "manifest_hash": plan.get("manifest_hash"),
        "base_contrastive_config_hash": plan.get("base_contrastive_config_hash"),
        "split_contract": plan.get("split_contract"),
        "split_contract_hash": plan.get("split_contract_hash"),
    }


def build_outcome_transfer_support_report(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    registry_path: str | Path = DEFAULT_REGISTRY,
    base_config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the label-only outcome-transfer support report in memory.

    No model, checkpoint, trainer, or embedding is created by this function.
    ``resolve_transfer_manifest`` is imported lazily so this narrowly scoped
    reporting module remains importable while a collaborator updates the
    transfer resolver itself.
    """
    from opera.functional.outcome_transfer import resolve_transfer_manifest

    plan = resolve_transfer_manifest(
        manifest_path=manifest_path,
        registry_path=registry_path,
        base_config_path=base_config_path,
    )
    registry = load_registry(_resolve_repository_or_cwd_path(registry_path))
    membership, ids_by_group = _load_membership(registry)
    del membership  # IDs are the only population input needed below.
    all_ids = set().union(*ids_by_group.values())
    split_keys = _load_split_keys(plan["split_contract"])
    family_by_outcome = _family_lookup(registry)
    availability_by_group = _plan_availability_by_group(plan, registry)

    target_requests = _condition_target_requests(plan)
    unknown_targets = sorted(
        {item["target_outcome"] for item in target_requests} - set(registry["outcomes"])
    )
    if unknown_targets:
        raise OutcomeTransferPreflightError(
            f"Resolved transfer plan has unknown registry targets: {unknown_targets}"
        )

    rows: list[dict[str, Any]] = []
    # Cache the raw shared labels by target.  The same target can occur in a
    # direct reference and an ablation condition, often at the same horizon.
    source_cache: dict[
        str, tuple[OutcomeSource, pd.DataFrame, pd.DataFrame | None]
    ] = {}
    for request in target_requests:
        target = str(request["target_outcome"])
        if target not in source_cache:
            source = _resolve_outcome_source(registry, target)
            source_cache[target] = (
                source,
                *_filtered_outcome_frame(
                    source=source,
                    outcome=target,
                    all_ids=all_ids,
                ),
            )
        source, target_frame, competing = source_cache[target]
        horizon_hours = int(request["primary_horizon_days"]) * 24
        target_family = family_by_outcome[target]

        condition_availability = request["condition"].get(
            "structural_cohort_exclusions", availability_by_group
        )
        if not isinstance(condition_availability, Mapping):
            raise OutcomeTransferPreflightError(
                f"Condition {request['condition_name']!r} has invalid "
                "structural_cohort_exclusions."
            )
        unavailable_groups = {
            group
            for group, excluded in condition_availability.items()
            if target in set(excluded)
        }
        all_target_ids = set().union(
            *[
                ids
                for group, ids in ids_by_group.items()
                if group not in unavailable_groups
            ]
        )
        all_availability = "partially_available" if unavailable_groups else "available"
        populations: Iterable[tuple[str, str, set[Any], str]] = [
            ("all_hematology", ALL_HEMATOLOGY, all_target_ids, all_availability)
        ]
        populations = [
            *populations,
            *[
                (
                    "cohort_grouped",
                    group,
                    ids,
                    "unavailable" if group in unavailable_groups else "available",
                )
                for group, ids in ids_by_group.items()
            ],
        ]
        for (
            evaluation_level,
            evaluation_group,
            population_ids,
            availability,
        ) in populations:
            if availability == "unavailable":
                zero = {
                    "n": 0,
                    "events": 0,
                    "non_events": 0,
                    "competing_events": 0,
                    "censored_before_horizon": 0,
                }
                split_counts = {name: zero.copy() for name in split_keys}
            else:
                scoped_frame = target_frame.loc[
                    target_frame["subject_id"].isin(population_ids)
                ].copy()
                scoped_competing = (
                    None
                    if competing is None
                    else competing.loc[
                        competing["subject_id"].isin(population_ids)
                    ].copy()
                )
                split_counts = {
                    logical_name: _label_counts(
                        scoped_frame,
                        competing=scoped_competing,
                        split_key=split_key,
                        n_hours_start_include=source.n_hours_start_include,
                        n_hours_end_include=horizon_hours,
                    )
                    for logical_name, split_key in split_keys.items()
                }
            rows.append(
                _row(
                    request=request,
                    target_family=target_family,
                    evaluation_level=evaluation_level,
                    evaluation_group=evaluation_group,
                    availability=availability,
                    split_counts=split_counts,
                    plan=plan,
                )
            )

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            [
                "transfer_condition",
                "target_family",
                "target_outcome",
                "primary_horizon_days",
                "evaluation_level",
                "evaluation_group",
            ],
            kind="stable",
        ).reset_index(drop=True)
    metadata = {
        "name": "outcome_transfer_support",
        "version": 1,
        "label_only": True,
        "registry": str(registry_path),
        "manifest": str(manifest_path),
        "registry_hash": plan.get("registry_hash"),
        "manifest_hash": plan.get("manifest_hash"),
        "base_contrastive_config_hash": plan.get("base_contrastive_config_hash"),
        "split_contract": plan.get("split_contract"),
        "split_contract_hash": plan.get("split_contract_hash"),
        "eligibility_scope": "ascertainment",
        "n_rows": int(len(report)),
        "evaluation_target_union": list(plan.get("evaluation_target_union", [])),
        "support_categories": {
            "primary_candidate": "held-out events >= 25 and held-out non-events >= 25",
            "partial_pooling_candidate": "held-out minority class between 10 and 24",
            "descriptive_only": "held-out minority class below 10",
        },
    }
    return report, metadata


def write_outcome_transfer_support_report(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    base_config_path: str | Path | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    """Write deterministic CSV and JSON support artifacts; never trains."""
    report, metadata = build_outcome_transfer_support_report(
        manifest_path=manifest_path,
        registry_path=registry_path,
        base_config_path=base_config_path,
    )
    destination = _expand_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / SUPPORT_CSV_NAME
    json_path = destination / SUPPORT_JSON_NAME
    report.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                # ``DataFrame.to_json`` converts NumPy scalar counts to native
                # JSON numbers instead of serialising them as strings.
                "rows": json.loads(report.to_json(orient="records")),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    csv_path, json_path, report = write_outcome_transfer_support_report(
        manifest_path=args.manifest,
        registry_path=args.registry,
        base_config_path=args.base_config,
        output_dir=args.output_dir,
    )
    print(
        "Wrote label-only transfer support report "
        f"({len(report)} rows): {csv_path}, {json_path}"
    )


if __name__ == "__main__":
    main()
