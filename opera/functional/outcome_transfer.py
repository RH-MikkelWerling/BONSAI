"""Resolve the narrow OPERA outcome-transfer experiment deterministically.

The transfer experiment holds out *outcome labels* only.  It deliberately
reuses the canonical joint OPERA configuration and does not alter the general
production sweep, temporal split contract, or cohort taxonomy.  This module
turns the small declarative manifest into an explicit, auditable plan which
can be consumed by config generation, support preflight, training launchers,
and downstream frozen-probe evaluation.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from opera.run.generate_sweep_configs import load_registry


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("opera/configs/manifests/outcome_transfer.yaml")
EXPECTED_CONDITIONS = (
    "opera_full",
    "opera_no_g3",
    "opera_no_transfusion_signal",
    "opera_no_hospitalisation_signal",
    "opera_no_infection_family",
    "opera_no_renal_family",
    "opera_no_cardiovascular_family",
)


def _repository_path(path: str | Path) -> Path:
    """Resolve repository-relative manifest paths without changing CWD."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _read_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _repository_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a YAML mapping: {resolved}")
    return loaded


def file_hash(path: str | Path) -> str:
    """Return a content hash for reproducible config/checkpoint provenance."""
    return hashlib.sha256(_repository_path(path).read_bytes()).hexdigest()


def load_transfer_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and minimally validate the declarative outcome-transfer manifest."""
    manifest = _read_mapping(path, label="Outcome-transfer manifest")
    required = {
        "name",
        "version",
        "registry",
        "base_contrastive_config",
        "split_contract",
        "seeds",
        "conditions",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Outcome-transfer manifest is missing keys: {missing}")
    if manifest["name"] != "outcome_transfer":
        raise ValueError("Outcome-transfer manifest name must be 'outcome_transfer'.")
    if manifest["version"] != 1:
        raise ValueError("Only outcome-transfer manifest version 1 is supported.")
    seeds = manifest["seeds"]
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("Outcome-transfer manifest seeds must be a list of integers.")
    if seeds != [42, 43, 44]:
        raise ValueError(
            "Outcome-transfer manifest must use the locked seeds [42, 43, 44]."
        )
    conditions = manifest["conditions"]
    if not isinstance(conditions, dict):
        raise ValueError("Outcome-transfer manifest conditions must be a mapping.")
    if tuple(conditions) != EXPECTED_CONDITIONS:
        raise ValueError(
            "Outcome-transfer manifest conditions must be exactly, and in order, "
            f"{list(EXPECTED_CONDITIONS)}."
        )
    return manifest


def _family_lookup(registry: Mapping[str, Any]) -> dict[str, str]:
    """Build the only family lookup, directly from the canonical registry."""
    return {
        outcome: family
        for family, members in registry["outcome_families"].items()
        for outcome in members
    }


def _structural_exclusions_by_group(
    registry: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Resolve registry-declared cohort availability without inventing cells."""
    outcomes = list(registry["outcomes"])
    excluded: dict[str, set[str]] = {
        group: set() for group in registry["cohort_groups"]
    }
    for rule in registry.get("availability_rules", {}).values():
        for group in rule.get("excluded_grouped", []):
            excluded[group].update(rule.get("outcomes", []))
    return {
        group: _ordered_unique(list(values), outcomes)
        for group, values in excluded.items()
        if values
    }


def _ordered_unique(values: Sequence[str], universe: Sequence[str]) -> list[str]:
    """Deduplicate values while retaining canonical registry ordering."""
    requested = set(values)
    return [value for value in universe if value in requested]


def _require_known_outcomes(
    outcomes: Sequence[str],
    registry_outcomes: Sequence[str],
    *,
    condition: str,
    field: str,
) -> None:
    unknown = sorted(set(outcomes) - set(registry_outcomes))
    if unknown:
        raise ValueError(
            f"Transfer condition {condition!r} has unknown {field}: {unknown}"
        )


def _require_known_families(
    families: Sequence[str], registry: Mapping[str, Any], *, condition: str
) -> None:
    unknown = sorted(set(families) - set(registry["outcome_families"]))
    if unknown:
        raise ValueError(
            f"Transfer condition {condition!r} has unknown outcome families: {unknown}"
        )


def _matched_g2_g3_pairs(registry_outcomes: Sequence[str]) -> list[dict[str, str]]:
    """Derive every valid G2/G3 pair rather than maintaining a manual list."""
    outcome_set = set(registry_outcomes)
    pairs = []
    for g3 in registry_outcomes:
        if not g3.endswith("_g3plus"):
            continue
        g2 = f"{g3[: -len('_g3plus')]}_g2plus"
        if g2 in outcome_set:
            pairs.append({"lower_grade_outcome": g2, "target_outcome": g3})
    if not pairs:
        raise ValueError("No matched G2/G3 outcomes were found in the registry.")
    return pairs


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is a syntactically valid SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _verify_hospitalisation_evidence(
    provenance: Mapping[str, Any] | None,
    requested_dependencies: Sequence[str],
) -> tuple[bool, str | None]:
    """Validate checked-in archival evidence for the direct-signal decision.

    The source R script belongs to the upstream outcome-construction workspace
    and is not available on every training server.  A checked-in evidence
    snapshot preserves its reviewed SHA-256, inspected line ranges, and the
    registered-component conclusion.  Both the snapshot content and its
    declared hash are checked here, so a manifest cannot merely assert a
    plausible-looking external hash.
    """
    if not isinstance(provenance, Mapping):
        return False, "dependency_provenance is missing"
    source_hash = provenance.get("source_sha256")
    evidence_file = provenance.get("evidence_file")
    evidence_hash = provenance.get("evidence_sha256")
    if not provenance.get("source") or not _is_sha256(source_hash):
        return False, "source and a valid source_sha256 are required"
    if not isinstance(evidence_file, str) or not evidence_file:
        return False, "evidence_file is required"
    if not _is_sha256(evidence_hash):
        return False, "a valid evidence_sha256 is required"
    path = _repository_path(evidence_file)
    if not path.is_file():
        return False, f"evidence_file does not exist: {path}"
    if file_hash(path).lower() != str(evidence_hash).lower():
        return False, "evidence_sha256 does not match the checked-in evidence file"
    try:
        evidence = _read_mapping(path, label="Hospitalisation dependency evidence")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return False, f"could not read evidence_file: {exc}"
    source = evidence.get("inspected_source")
    hospitalisation = evidence.get("hospitalisation")
    if not isinstance(source, Mapping) or not isinstance(hospitalisation, Mapping):
        return False, "evidence_file lacks inspected_source or hospitalisation mapping"
    if str(source.get("sha256", "")).lower() != str(source_hash).lower():
        return False, "source_sha256 does not match the archival evidence"
    evidence_dependencies = hospitalisation.get("direct_registered_outcomes")
    if not isinstance(evidence_dependencies, list) or not all(
        isinstance(value, str) for value in evidence_dependencies
    ):
        return False, "evidence_file direct_registered_outcomes must be a string list"
    if list(evidence_dependencies) != list(requested_dependencies):
        return False, (
            "manifest direct_dependencies do not match the archival evidence "
            f"({list(evidence_dependencies)!r} != {list(requested_dependencies)!r})"
        )
    return True, None


def _resolve_hospitalisation_dependencies(
    *,
    condition: str,
    exclusion: Mapping[str, Any],
    registry_outcomes: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Resolve direct hospitalisation components with explicit provenance.

    The target is always held out.  Additional registered outcomes can be
    excluded only when a manifest records an inspected construction source.
    This prevents an apparent transfer experiment from silently retaining a
    component of a composite target based merely on a name heuristic.
    """
    target = exclusion.get("outcome")
    if target != "hospitalisation":
        raise ValueError(
            f"{condition!r} must resolve direct dependencies for hospitalisation."
        )
    if target not in registry_outcomes:
        raise ValueError("hospitalisation is not present in the canonical registry.")

    requested = exclusion.get("direct_dependencies", [])
    if not isinstance(requested, list) or not all(
        isinstance(value, str) for value in requested
    ):
        raise ValueError(
            f"Transfer condition {condition!r} direct_dependencies must be a list."
        )
    _require_known_outcomes(
        requested,
        registry_outcomes,
        condition=condition,
        field="direct_dependencies",
    )
    provenance = exclusion.get("dependency_provenance")
    verified, evidence_problem = _verify_hospitalisation_evidence(
        provenance if isinstance(provenance, Mapping) else None,
        requested,
    )
    if requested and not verified:
        raise ValueError(
            f"Transfer condition {condition!r} lists direct dependencies without "
            f"verified dependency_provenance ({evidence_problem})."
        )

    # `direct_dependencies_excluded` intentionally means the complete signal
    # closure, including the target itself.  This aligns the metadata with the
    # actual training-exclusion list and avoids ambiguity for launch guards.
    direct_closure = _ordered_unique([target, *requested], registry_outcomes)
    status = (
        "verified_archival_source_evidence"
        if verified
        else "unresolved_source_not_found"
    )
    details = {
        "dependency_resolution_status": status,
        "dependency_provenance": deepcopy(dict(provenance)) if verified else None,
        "launch_blocked": not verified,
        "launch_blocked_reason": (
            None
            if verified
            else (
                "No verified hospitalisation construction provenance is recorded; "
                "only the target itself may be resolved, so launch is blocked. "
                f"Evidence issue: {evidence_problem}."
            )
        ),
    }
    return direct_closure, details


def _resolve_exclusion(
    *,
    condition: str,
    spec: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Resolve a declarative exclusion specification to registry outcomes."""
    outcomes = list(registry["outcomes"])
    exclusion = spec.get("exclude")
    if exclusion is None:
        return [], {
            "dependency_resolution_status": "not_applicable",
            "dependency_provenance": None,
            "launch_blocked": False,
            "launch_blocked_reason": None,
        }
    if not isinstance(exclusion, Mapping):
        raise ValueError(f"Transfer condition {condition!r} exclude must be a mapping.")
    exclusion_type = exclusion.get("type")
    if exclusion_type == "suffix":
        suffix = exclusion.get("value")
        if not isinstance(suffix, str) or not suffix:
            raise ValueError(
                f"Transfer condition {condition!r} needs a non-empty suffix."
            )
        return [outcome for outcome in outcomes if outcome.endswith(suffix)], {
            "dependency_resolution_status": "not_applicable",
            "dependency_provenance": None,
            "launch_blocked": False,
            "launch_blocked_reason": None,
        }
    if exclusion_type == "exact":
        exact = exclusion.get("outcomes", [])
        if not isinstance(exact, list) or not all(
            isinstance(value, str) for value in exact
        ):
            raise ValueError(
                f"Transfer condition {condition!r} exact exclusions must be a list."
            )
        _require_known_outcomes(
            exact, outcomes, condition=condition, field="exact excluded outcomes"
        )
        return _ordered_unique(exact, outcomes), {
            "dependency_resolution_status": "not_applicable",
            "dependency_provenance": None,
            "launch_blocked": False,
            "launch_blocked_reason": None,
        }
    if exclusion_type == "family":
        family = exclusion.get("family")
        if family not in registry["outcome_families"]:
            raise ValueError(
                f"Transfer condition {condition!r} names unknown family {family!r}."
            )
        return list(registry["outcome_families"][family]), {
            "dependency_resolution_status": "not_applicable",
            "dependency_provenance": None,
            "launch_blocked": False,
            "launch_blocked_reason": None,
        }
    if exclusion_type == "outcome_with_direct_dependencies":
        return _resolve_hospitalisation_dependencies(
            condition=condition,
            exclusion=exclusion,
            registry_outcomes=outcomes,
        )
    raise ValueError(
        f"Transfer condition {condition!r} has unsupported exclusion type "
        f"{exclusion_type!r}."
    )


def _resolve_evaluation_outcomes(
    *,
    condition: str,
    spec: Mapping[str, Any],
    excluded: Sequence[str],
    registry: Mapping[str, Any],
    matched_pairs: Sequence[Mapping[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Return all, primary, and secondary targets for a transfer condition."""
    outcomes = list(registry["outcomes"])
    exclusion = spec.get("exclude", {})
    exclusion_type = exclusion.get("type") if isinstance(exclusion, Mapping) else None
    if condition == "opera_full":
        return outcomes, outcomes, []
    if exclusion_type == "suffix" and exclusion.get("value") == "_g3plus":
        primary = [pair["target_outcome"] for pair in matched_pairs]
        secondary = [outcome for outcome in excluded if outcome not in set(primary)]
        return _ordered_unique([*primary, *secondary], outcomes), primary, secondary
    if exclusion_type == "family":
        return list(excluded), list(excluded), []
    configured = spec.get("evaluation_outcomes", [])
    if not isinstance(configured, list) or not all(
        isinstance(value, str) for value in configured
    ):
        raise ValueError(
            f"Transfer condition {condition!r} evaluation_outcomes must be a list."
        )
    _require_known_outcomes(
        configured, outcomes, condition=condition, field="evaluation_outcomes"
    )
    not_held_out = sorted(set(configured) - set(excluded))
    if not_held_out:
        raise ValueError(
            f"Transfer condition {condition!r} evaluates outcomes that remain in "
            f"contrastive adaptation: {not_held_out}"
        )
    resolved = _ordered_unique(configured, outcomes)
    return resolved, resolved, []


def _resolve_related_retained_outcomes(
    *,
    condition: str,
    spec: Mapping[str, Any],
    training_outcomes: Sequence[str],
    registry: Mapping[str, Any],
    matched_pairs: Sequence[Mapping[str, str]],
) -> list[str]:
    """Resolve explicitly declared retained proxy supervision from the registry."""
    if condition == "opera_no_g3":
        return [pair["lower_grade_outcome"] for pair in matched_pairs]
    exclusion = spec.get("exclude", {})
    # For a family holdout, *every* remaining registry endpoint is the
    # retained cross-system supervision.  Keeping this full list in the
    # resolved plan is more informative than an empty shorthand: it makes the
    # family-transfer contrast auditable without asking a reader to subtract
    # two separate panels by hand.
    if isinstance(exclusion, Mapping) and exclusion.get("type") == "family":
        return list(training_outcomes)
    families = spec.get("related_retained_families", [])
    if not isinstance(families, list) or not all(
        isinstance(value, str) for value in families
    ):
        raise ValueError(
            f"Transfer condition {condition!r} related_retained_families must be a list."
        )
    _require_known_families(families, registry, condition=condition)
    selected = [
        outcome
        for outcome in registry["outcomes"]
        if outcome in set(training_outcomes)
        and any(outcome in registry["outcome_families"][family] for family in families)
    ]
    return selected


def _validate_base_config(
    base_config: Mapping[str, Any], registry: Mapping[str, Any]
) -> None:
    """Fail if an ablation would not start from the canonical full panel."""
    base_outcomes = base_config.get("outcomes")
    if not isinstance(base_outcomes, Mapping):
        raise ValueError(
            "Canonical full OPERA config must contain an outcomes mapping."
        )
    registry_outcomes = list(registry["outcomes"])
    if list(base_outcomes) != registry_outcomes:
        missing = sorted(set(registry_outcomes) - set(base_outcomes))
        extra = sorted(set(base_outcomes) - set(registry_outcomes))
        order_matches = list(base_outcomes) == registry_outcomes
        raise ValueError(
            "Canonical full OPERA config outcomes must match the registry exactly; "
            f"missing={missing}, extra={extra}, order_matches={order_matches}."
        )


def resolve_transfer_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    registry_path: str | Path | None = None,
    base_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve all transfer conditions into a deterministic, explicit plan.

    No labels, checkpoints, or model code are loaded.  This is safe to call in
    validation and preflight commands.
    """
    manifest = load_transfer_manifest(manifest_path)
    selected_registry = registry_path or manifest["registry"]
    selected_base = base_config_path or manifest["base_contrastive_config"]
    registry = load_registry(_repository_path(selected_registry))
    base_config = _read_mapping(selected_base, label="Canonical full OPERA config")
    _validate_base_config(base_config, registry)

    registry_outcomes = list(registry["outcomes"])
    family_lookup = _family_lookup(registry)
    matched_pairs = _matched_g2_g3_pairs(registry_outcomes)
    structural_exclusions = _structural_exclusions_by_group(registry)
    conditions: dict[str, dict[str, Any]] = {}

    for condition, raw_spec in manifest["conditions"].items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Transfer condition {condition!r} must be a mapping.")
        transfer_level = raw_spec.get("transfer_level")
        if transfer_level not in {
            "direct_supervision",
            "severity_transfer",
            "related_outcome_transfer",
            "family_transfer",
        }:
            raise ValueError(
                f"Transfer condition {condition!r} has invalid transfer_level "
                f"{transfer_level!r}."
            )
        horizon = raw_spec.get("primary_horizon_days")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(
                f"Transfer condition {condition!r} needs a positive primary horizon."
            )

        excluded, dependency_details = _resolve_exclusion(
            condition=condition,
            spec=raw_spec,
            registry=registry,
        )
        training = [
            outcome for outcome in registry_outcomes if outcome not in set(excluded)
        ]
        if not training:
            raise ValueError(
                f"Transfer condition {condition!r} excludes every outcome."
            )
        evaluation, primary_evaluation, secondary_evaluation = (
            _resolve_evaluation_outcomes(
                condition=condition,
                spec=raw_spec,
                excluded=excluded,
                registry=registry,
                matched_pairs=matched_pairs,
            )
        )
        related_retained = _resolve_related_retained_outcomes(
            condition=condition,
            spec=raw_spec,
            training_outcomes=training,
            registry=registry,
            matched_pairs=matched_pairs,
        )

        conditions[condition] = {
            "name": condition,
            "transfer_level": transfer_level,
            "primary_horizon_days": horizon,
            "training_outcomes": training,
            "training_excluded_outcomes": excluded,
            "evaluation_outcomes": evaluation,
            "primary_evaluation_outcomes": primary_evaluation,
            "secondary_evaluation_outcomes": secondary_evaluation,
            "related_retained_outcomes": related_retained,
            "direct_dependencies_excluded": (
                excluded
                if raw_spec.get("exclude", {}).get("type")
                == "outcome_with_direct_dependencies"
                else []
            ),
            "matched_g2_g3_pairs": (
                [dict(pair) for pair in matched_pairs]
                if condition == "opera_no_g3"
                else []
            ),
            # The panel is global, but these cells remain intentionally absent
            # for the stated grouped cohorts.  Keeping this explicit makes it
            # impossible to mistake structural non-availability for a
            # transfer holdout or a label-loading failure.
            "structural_cohort_exclusions": {
                group: [outcome for outcome in unavailable if outcome in set(training)]
                for group, unavailable in structural_exclusions.items()
                if any(outcome in set(training) for outcome in unavailable)
            },
            **dependency_details,
        }

    target_union = _ordered_unique(
        [
            outcome
            for name, condition in conditions.items()
            if name != "opera_full"
            for outcome in condition["evaluation_outcomes"]
        ],
        registry_outcomes,
    )
    slots = [
        {"condition": condition, "seed": seed}
        for condition in EXPECTED_CONDITIONS
        for seed in manifest["seeds"]
    ]
    registry_hash = file_hash(selected_registry)
    manifest_hash = file_hash(manifest_path)
    return {
        "name": manifest["name"],
        "version": manifest["version"],
        "registry": str(selected_registry),
        "registry_hash": registry_hash,
        "manifest": str(manifest_path),
        "manifest_hash": manifest_hash,
        "base_contrastive_config": str(selected_base),
        "base_contrastive_config_hash": file_hash(selected_base),
        "split_contract": manifest["split_contract"],
        # The split contract is an input to both the support audit and frozen
        # probe labels.  A path alone is not sufficient provenance: changing
        # the file in place must invalidate generated configs/checkpoints.
        "split_contract_hash": file_hash(manifest["split_contract"]),
        "seeds": list(manifest["seeds"]),
        "checkpoint_reuse": deepcopy(manifest.get("checkpoint_reuse", {})),
        "checkpoint_slots": slots,
        "checkpoint_slot_count": len(slots),
        "new_contrastive_runs_if_full_reused": len(slots) - len(manifest["seeds"]),
        "evaluation_target_union": target_union,
        "outcome_families": family_lookup,
        "structural_outcome_exclusions_by_group": structural_exclusions,
        "conditions": conditions,
    }


def transfer_plan_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten an explicit plan into deterministic CSV-ready condition rows."""
    rows: list[dict[str, Any]] = []
    for condition in EXPECTED_CONDITIONS:
        resolved = plan["conditions"][condition]
        rows.append(
            {
                "condition": condition,
                "transfer_level": resolved["transfer_level"],
                "primary_horizon_days": resolved["primary_horizon_days"],
                "training_outcome_count": len(resolved["training_outcomes"]),
                "training_outcomes": json.dumps(resolved["training_outcomes"]),
                "training_excluded_outcome_count": len(
                    resolved["training_excluded_outcomes"]
                ),
                "training_excluded_outcomes": json.dumps(
                    resolved["training_excluded_outcomes"]
                ),
                "evaluation_outcome_count": len(resolved["evaluation_outcomes"]),
                "evaluation_outcomes": json.dumps(resolved["evaluation_outcomes"]),
                "primary_evaluation_outcomes": json.dumps(
                    resolved["primary_evaluation_outcomes"]
                ),
                "secondary_evaluation_outcomes": json.dumps(
                    resolved["secondary_evaluation_outcomes"]
                ),
                "related_retained_outcome_count": len(
                    resolved["related_retained_outcomes"]
                ),
                "related_retained_outcomes": json.dumps(
                    resolved["related_retained_outcomes"]
                ),
                "direct_dependencies_excluded": json.dumps(
                    resolved["direct_dependencies_excluded"]
                ),
                "dependency_resolution_status": resolved[
                    "dependency_resolution_status"
                ],
                "launch_blocked": resolved["launch_blocked"],
                "launch_blocked_reason": resolved["launch_blocked_reason"] or "",
                "matched_g2_g3_pairs": json.dumps(resolved["matched_g2_g3_pairs"]),
                "registry_hash": plan["registry_hash"],
                "manifest_hash": plan["manifest_hash"],
                "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
                "split_contract_hash": plan["split_contract_hash"],
            }
        )
    return rows


def write_resolved_transfer_plan(
    plan: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Write the exact JSON/CSV plan consumed by downstream transfer stages."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "resolved_transfer_plan.json"
    csv_path = destination / "resolved_transfer_plan.csv"
    json_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = transfer_plan_rows(plan)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_transfer_config(
    plan: Mapping[str, Any], condition: str, base_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Clone canonical OPERA settings while substituting only the outcome panel."""
    if condition not in plan["conditions"]:
        raise KeyError(f"Unknown transfer condition {condition!r}.")
    resolved = plan["conditions"][condition]
    config = deepcopy(dict(base_config))
    original_outcomes = config["outcomes"]
    config["outcomes"] = {
        outcome: deepcopy(original_outcomes[outcome])
        for outcome in resolved["training_outcomes"]
    }
    # The following fields make checkpoint provenance and leakage guards
    # inspectable without changing any architecture or optimizer setting.
    config.update(
        {
            "transfer_analysis": True,
            "transfer_condition": condition,
            "transfer_level": resolved["transfer_level"],
            # A generated condition is directly runnable for the first
            # manifest seed.  The narrow launcher overrides this field for
            # the other declared seeds and persists the actual value in the
            # checkpoint metadata.
            "seed": int(plan["seeds"][0]),
            "transfer_seeds": list(plan["seeds"]),
            "training_outcomes": list(resolved["training_outcomes"]),
            "training_excluded_outcomes": list(resolved["training_excluded_outcomes"]),
            "evaluation_outcomes": list(resolved["evaluation_outcomes"]),
            "primary_evaluation_outcomes": list(
                resolved["primary_evaluation_outcomes"]
            ),
            "secondary_evaluation_outcomes": list(
                resolved["secondary_evaluation_outcomes"]
            ),
            "related_retained_outcomes": list(resolved["related_retained_outcomes"]),
            "direct_dependencies_excluded": list(
                resolved["direct_dependencies_excluded"]
            ),
            "structural_cohort_exclusions": deepcopy(
                resolved["structural_cohort_exclusions"]
            ),
            "dependency_resolution_status": resolved["dependency_resolution_status"],
            "dependency_provenance": deepcopy(resolved["dependency_provenance"]),
            "launch_blocked": resolved["launch_blocked"],
            "launch_blocked_reason": resolved["launch_blocked_reason"],
            "registry_hash": plan["registry_hash"],
            "manifest_hash": plan["manifest_hash"],
            "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
            "split_contract": plan["split_contract"],
            "split_contract_hash": plan["split_contract_hash"],
            # Training/validation/checkpoint selection must see this exact
            # included panel only; evaluators load target labels later.
            "selection_outcomes": list(resolved["training_outcomes"]),
            "transfer_checkpoint_metadata": {
                "condition": condition,
                "transfer_level": resolved["transfer_level"],
                "seed": int(plan["seeds"][0]),
                "included_outcomes": list(resolved["training_outcomes"]),
                "excluded_outcomes": list(resolved["training_excluded_outcomes"]),
                "evaluation_outcomes": list(resolved["evaluation_outcomes"]),
                "related_retained_outcomes": list(
                    resolved["related_retained_outcomes"]
                ),
                "direct_dependencies_excluded": list(
                    resolved["direct_dependencies_excluded"]
                ),
                "structural_cohort_exclusions": deepcopy(
                    resolved["structural_cohort_exclusions"]
                ),
                "registry_hash": plan["registry_hash"],
                "manifest_hash": plan["manifest_hash"],
                "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
                "split_contract": plan["split_contract"],
                "split_contract_hash": plan["split_contract_hash"],
                "source_dapt_checkpoint": config.get("dapt_ckpt"),
                "selection_outcomes": list(resolved["training_outcomes"]),
            },
        }
    )
    return config
