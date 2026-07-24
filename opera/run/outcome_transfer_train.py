"""Launch the focused OPERA outcome-transfer contrastive conditions.

This is intentionally separate from the broad OPERA sweep.  It owns exactly
the seven manifest conditions and records every selected condition/seed slot in
``transfer_training_status.csv``.  By default it is a dry run; pass
``--execute`` only after the label-only support report has been reviewed.

Examples
--------
Validate command construction only::

    python -m opera.run.outcome_transfer_train --dry-run

Launch one ablation for the three canonical seeds::

    python -m opera.run.outcome_transfer_train \
      --conditions opera_no_g3 --execute

Reuse compatible full-OPERA checkpoints while launching selected conditions::

    python -m opera.run.outcome_transfer_train \
      --full-checkpoint-template '/checkpoints/opera_full/seed_{seed}/best.ckpt' \
      --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from opera.functional.outcome_transfer import (
    DEFAULT_MANIFEST,
    EXPECTED_CONDITIONS,
    REPOSITORY_ROOT,
    resolve_transfer_manifest,
)


DEFAULT_CONFIG_DIR = Path("opera/configs/generated/outcome_transfer")
DEFAULT_OUTPUT_DIR = Path("outputs/outcome_transfer")
STATUS_NAME = "transfer_training_status.csv"


class OutcomeTransferLaunchError(ValueError):
    """Raised when a requested transfer-training slot is unsafe to launch."""


def _parse_csv_values(
    raw: str | None,
    *,
    permitted: Iterable[str] | None = None,
) -> list[str]:
    values = [value.strip() for value in (raw or "").split(",") if value.strip()]
    if not values:
        return []
    if len(set(values)) != len(values):
        raise OutcomeTransferLaunchError(f"Values cannot contain duplicates: {values}")
    if permitted is not None:
        unknown = sorted(set(values) - set(permitted))
        if unknown:
            raise OutcomeTransferLaunchError(f"Unknown values: {unknown}")
    return values


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise OutcomeTransferLaunchError(f"Expected YAML mapping at {path}.")
    return data


def _validate_generated_config(
    config_path: Path,
    resolved: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Check the launch YAML remains an exact resolver product."""
    if not config_path.exists():
        raise OutcomeTransferLaunchError(
            f"Missing generated config {config_path}; run "
            "python -m opera.run.generate_outcome_transfer_configs first."
        )
    config = _load_yaml(config_path)
    outcome_mapping = config.get("outcomes")
    if not isinstance(outcome_mapping, Mapping):
        raise OutcomeTransferLaunchError(f"{config_path} has no outcomes mapping.")
    checks = {
        "transfer_analysis": True,
        "transfer_condition": resolved["name"],
        "transfer_level": resolved["transfer_level"],
        "training_outcomes": resolved["training_outcomes"],
        "training_excluded_outcomes": resolved["training_excluded_outcomes"],
        "evaluation_outcomes": resolved["evaluation_outcomes"],
        "selection_outcomes": resolved["training_outcomes"],
        "registry_hash": plan["registry_hash"],
        "manifest_hash": plan["manifest_hash"],
        "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
        "split_contract_hash": plan["split_contract_hash"],
    }
    for field, expected in checks.items():
        if config.get(field) != expected:
            raise OutcomeTransferLaunchError(
                f"{config_path} field {field!r} is not the resolved manifest value."
            )
    if list(outcome_mapping) != list(resolved["training_outcomes"]):
        raise OutcomeTransferLaunchError(
            f"{config_path} outcome mapping is not exactly training_outcomes."
        )
    held_out = set(resolved["training_excluded_outcomes"])
    if held_out & set(outcome_mapping):
        raise OutcomeTransferLaunchError(
            f"{config_path} routes held-out labels into contrastive training: "
            f"{sorted(held_out & set(outcome_mapping))}."
        )
    checkpoint_metadata = config.get("transfer_checkpoint_metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise OutcomeTransferLaunchError(
            f"{config_path} has no transfer_checkpoint_metadata mapping."
        )
    metadata_checks = {
        "condition": resolved["name"],
        "included_outcomes": resolved["training_outcomes"],
        "excluded_outcomes": resolved["training_excluded_outcomes"],
        "evaluation_outcomes": resolved["evaluation_outcomes"],
        "related_retained_outcomes": resolved["related_retained_outcomes"],
        "direct_dependencies_excluded": resolved["direct_dependencies_excluded"],
        "selection_outcomes": resolved["training_outcomes"],
        "registry_hash": plan["registry_hash"],
        "manifest_hash": plan["manifest_hash"],
        "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
        "split_contract": plan["split_contract"],
        "split_contract_hash": plan["split_contract_hash"],
    }
    for field, expected in metadata_checks.items():
        if checkpoint_metadata.get(field) != expected:
            raise OutcomeTransferLaunchError(
                f"{config_path} transfer_checkpoint_metadata field {field!r} "
                "is not the resolved manifest value."
            )
    return config


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Read a metadata sidecar first, then a Lightning checkpoint if needed."""
    sidecar = checkpoint_path.parent / "checkpoint_metadata.json"
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            metadata = payload.get("checkpoint_metadata", payload)
            if isinstance(metadata, Mapping):
                return dict(metadata)
    try:
        # Avoid importing torch for ordinary dry runs and sidecar-backed
        # checkpoint reuse checks.
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - data-dependent checkpoint I/O
        raise OutcomeTransferLaunchError(
            f"Unable to inspect checkpoint metadata for {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OutcomeTransferLaunchError(
            f"Checkpoint {checkpoint_path} is not a mapping."
        )
    hparams = payload.get("hyper_parameters", {})
    if not isinstance(hparams, Mapping):
        return {}
    metadata = hparams.get("checkpoint_metadata", {})
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def check_reusable_full_checkpoint(
    checkpoint_path: Path,
    plan: Mapping[str, Any],
    *,
    seed: int,
    allow_legacy_without_seed: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate a legacy or transfer-tagged full OPERA checkpoint for reuse.

    Earlier canonical full checkpoints predate this focused experiment and do
    not carry its manifest hash.  They are still reusable if their recorded
    adaptation outcome panel is exactly the canonical registry *and* their
    seed is recorded, or the caller explicitly supplies a seed-bound mapping.
    Tagged transfer checkpoints receive the stricter condition/seed/hash
    checks.  The default is deliberately fail-closed: an unseeded historical
    checkpoint must never silently stand in for an arbitrary seed.
    """
    if not checkpoint_path.exists():
        return False, "checkpoint_not_found", {}
    try:
        metadata = _checkpoint_metadata(checkpoint_path)
    except OutcomeTransferLaunchError:
        return False, "metadata_unreadable", {}
    full_outcomes = plan["conditions"]["opera_full"]["training_outcomes"]
    recorded = metadata.get("included_outcomes", metadata.get("outcome_set"))
    if (
        not isinstance(recorded, (list, tuple))
        or set(recorded) != set(full_outcomes)
        or len(recorded) != len(full_outcomes)
    ):
        return False, "outcome_panel_mismatch", metadata
    if metadata.get("condition") not in (None, "opera_full"):
        return False, "condition_mismatch", metadata
    recorded_seed = metadata.get("seed")
    if recorded_seed not in (None, seed):
        return False, "seed_mismatch", metadata
    if recorded_seed is None and not allow_legacy_without_seed:
        return False, "seed_provenance_required", metadata
    # A historic full panel may predate transfer provenance entirely.  Once a
    # checkpoint explicitly identifies itself as ``opera_full``, however, all
    # immutable inputs must be present and exact.  This prevents a stale full
    # config from being reused after the canonical base config or temporal
    # split contract changes in place.
    tagged_transfer_full = metadata.get("condition") == "opera_full"
    for field in (
        "registry_hash",
        "manifest_hash",
        "base_contrastive_config_hash",
        "split_contract_hash",
    ):
        recorded = metadata.get(field)
        if tagged_transfer_full:
            if recorded != plan[field]:
                return False, f"{field}_mismatch_or_missing", metadata
        elif recorded not in (None, plan[field]):
            return False, f"{field}_mismatch", metadata
    if metadata.get("training_stage") != "opera_contrastive_adaptation":
        return False, "training_stage_mismatch", metadata
    # Historic full checkpoints use ``source_checkpoint`` while focused
    # transfer checkpoints use the clearer ``source_dapt_checkpoint``.  Both
    # must prove the encoder was adapted from a DAPT source before a full
    # checkpoint can serve as the direct-supervision reference.
    source_dapt = metadata.get(
        "source_dapt_checkpoint", metadata.get("source_checkpoint")
    )
    if not source_dapt:
        return False, "source_dapt_checkpoint_missing", metadata
    return True, "compatible", metadata


def _template_binds_seed(template: str) -> bool:
    """Return whether a checkpoint template explicitly varies by ``seed``.

    Historic full-OPERA checkpoints may not contain a seed in their metadata.
    In that case a multi-seed transfer comparison remains reproducible only
    when the supplied path template explicitly selects a per-seed artefact.
    ``string.Formatter`` handles both ``{seed}`` and format-spec variants such
    as ``{seed:02d}`` without evaluating the template.
    """
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(template)]
    except ValueError as exc:
        raise OutcomeTransferLaunchError(
            f"Invalid --full-checkpoint-template: {exc}"
        ) from exc
    return "seed" in fields


def _format_full_checkpoint_template(template: str, seed: int) -> Path:
    """Format one explicitly seed-aware full-checkpoint template safely."""
    try:
        return Path(template.format(seed=seed))
    except (KeyError, IndexError, ValueError) as exc:
        raise OutcomeTransferLaunchError(
            "--full-checkpoint-template may use only a valid {seed} placeholder; "
            f"could not format it for seed={seed}: {exc}"
        ) from exc


def _check_ablation_checkpoint(
    checkpoint_path: Path,
    plan: Mapping[str, Any],
    *,
    condition: str,
    seed: int,
) -> tuple[bool, str, dict[str, Any]]:
    """Verify a previously launched ablation checkpoint before reusing it."""
    if not checkpoint_path.exists():
        return False, "checkpoint_not_found", {}
    try:
        metadata = _checkpoint_metadata(checkpoint_path)
    except OutcomeTransferLaunchError:
        return False, "metadata_unreadable", {}
    expected = plan["conditions"][condition]
    exact = {
        "training_stage": "opera_contrastive_adaptation",
        "condition": condition,
        "transfer_level": expected["transfer_level"],
        "seed": seed,
        "registry_hash": plan["registry_hash"],
        "manifest_hash": plan["manifest_hash"],
        "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
        "included_outcomes": expected["training_outcomes"],
        "excluded_outcomes": expected["training_excluded_outcomes"],
        "evaluation_outcomes": expected["evaluation_outcomes"],
        "related_retained_outcomes": expected["related_retained_outcomes"],
        "direct_dependencies_excluded": expected["direct_dependencies_excluded"],
        "selection_outcomes": expected["training_outcomes"],
        "split_contract": plan["split_contract"],
        "split_contract_hash": plan["split_contract_hash"],
    }
    for field, value in exact.items():
        if metadata.get(field) != value:
            return False, f"{field}_mismatch_or_missing", metadata
    if not metadata.get("source_dapt_checkpoint"):
        return False, "source_dapt_checkpoint_missing", metadata
    outcome_set = metadata.get("outcome_set")
    if (
        not isinstance(outcome_set, (list, tuple))
        or set(outcome_set) != set(expected["training_outcomes"])
        or len(outcome_set) != len(expected["training_outcomes"])
    ):
        return False, "outcome_set_mismatch_or_missing", metadata
    return True, "compatible", metadata


def _candidate_checkpoints(run_dir: Path) -> list[Path]:
    """Find checkpoints at both historic and CSVLogger-produced locations."""
    candidates = [run_dir / "best.ckpt"]
    # ``contrastive_multicohort`` uses CSVLogger(name=...), whose checkpoint
    # callback writes under this nested version directory rather than directly
    # in Hydra's run directory.
    candidates.extend(
        sorted(run_dir.glob("contrastive_multicohort_runs/version_*/best.ckpt"))
    )
    return [path for path in candidates if path.exists()]


def _find_reusable_ablation_checkpoint(
    run_dir: Path,
    plan: Mapping[str, Any],
    *,
    condition: str,
    seed: int,
) -> tuple[Path | None, str]:
    """Return a validated nested logger checkpoint or the reason none was safe."""
    candidates = _candidate_checkpoints(run_dir)
    if not candidates:
        return None, "checkpoint_not_found"
    compatible: list[Path] = []
    reasons: list[str] = []
    for candidate in candidates:
        valid, detail, _ = _check_ablation_checkpoint(
            candidate,
            plan,
            condition=condition,
            seed=seed,
        )
        if valid:
            compatible.append(candidate)
        else:
            reasons.append(f"{candidate}: {detail}")
    if compatible:
        return max(compatible, key=lambda path: path.stat().st_mtime), "compatible"
    return None, "; ".join(reasons)


def build_training_command(
    *,
    condition: str,
    seed: int,
    run_dir: Path,
) -> list[str]:
    """Return the exact subprocess command for one new contrastive slot."""
    return [
        sys.executable,
        "-m",
        "opera.run.contrastive_multicohort",
        f"--config-name=generated/outcome_transfer/{condition}",
        f"seed={seed}",
        f"hydra.run.dir={run_dir}",
    ]


def _status_row(
    *,
    condition: str,
    seed: int,
    status: str,
    run_dir: Path,
    plan: Mapping[str, Any],
    command: list[str] | None = None,
    checkpoint_path: Path | None = None,
    detail: str = "",
) -> dict[str, Any]:
    resolved = plan["conditions"][condition]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "seed": int(seed),
        "status": status,
        "transfer_level": resolved["transfer_level"],
        "included_outcome_count": len(resolved["training_outcomes"]),
        "excluded_outcome_count": len(resolved["training_excluded_outcomes"]),
        "run_dir": str(run_dir),
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "command": "" if command is None else json.dumps(command),
        "detail": detail,
        "registry_hash": plan["registry_hash"],
        "manifest_hash": plan["manifest_hash"],
    }


def _write_status(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STATUS_NAME
    fieldnames = (
        list(rows[0])
        if rows
        else [
            "timestamp_utc",
            "condition",
            "seed",
            "status",
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validate_preflight_report(
    report_path: str | Path,
    plan: Mapping[str, Any],
) -> Path:
    """Verify that execution follows the matching label-only support audit.

    The preflight is intentionally a human-review report rather than a hidden
    support filter: low-count targets remain in its rows and are not silently
    removed from contrastive adaptation.  This guard only proves that a
    current, complete label-only report was produced for this exact manifest
    before new GPU work starts.
    """
    source = Path(report_path)
    if not source.is_file():
        raise OutcomeTransferLaunchError(
            f"Preflight report does not exist: {source}. Run "
            "opera.run.outcome_transfer_preflight first."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeTransferLaunchError(
            f"Preflight report is not readable JSON: {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OutcomeTransferLaunchError("Preflight report must be a JSON mapping.")
    metadata = payload.get("metadata")
    rows = payload.get("rows")
    if not isinstance(metadata, Mapping) or not isinstance(rows, list):
        raise OutcomeTransferLaunchError(
            "Preflight report must contain mapping metadata and a rows list."
        )
    if metadata.get("label_only") is not True:
        raise OutcomeTransferLaunchError(
            "Preflight report is not labelled label_only=true; it cannot gate training."
        )
    for field in (
        "registry_hash",
        "manifest_hash",
        "base_contrastive_config_hash",
        "split_contract_hash",
    ):
        if metadata.get(field) != plan[field]:
            raise OutcomeTransferLaunchError(
                f"Preflight report {field} does not match the resolved transfer plan."
            )
    if set(metadata.get("evaluation_target_union", [])) != set(
        plan["evaluation_target_union"]
    ):
        raise OutcomeTransferLaunchError(
            "Preflight report evaluation_target_union does not match the resolved plan."
        )

    observed_cells = {
        (
            str(row.get("transfer_condition")),
            str(row.get("target_outcome")),
            int(row.get("primary_horizon_days")),
        )
        for row in rows
        if isinstance(row, Mapping)
        and row.get("evaluation_level") == "all_hematology"
        and row.get("evaluation_group") == "all_hematology"
        and isinstance(row.get("primary_horizon_days"), (int, float))
    }
    expected_cells = {
        (condition_name, str(target), int(condition["primary_horizon_days"]))
        for condition_name, condition in plan["conditions"].items()
        if condition_name != "opera_full"
        for target in condition["evaluation_outcomes"]
    }
    missing = sorted(expected_cells - observed_cells)
    if missing:
        raise OutcomeTransferLaunchError(
            "Preflight report is incomplete for the current transfer targets; "
            f"missing all-hematology cells such as {missing[:5]}."
        )
    return source


def launch_outcome_transfer_conditions(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    conditions: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    full_checkpoint_template: str | None = None,
    preflight_report: str | Path | None = None,
    execute: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Dry-run or execute selected transfer contrastive slots.

    Full OPERA is never retrained implicitly: an explicitly supplied,
    metadata-compatible checkpoint is reused; otherwise its status records why
    it could not be reused.  This keeps the expected new-run maximum at 18.
    """
    plan = resolve_transfer_manifest(manifest_path)
    selected_conditions = list(conditions or EXPECTED_CONDITIONS)
    unknown_conditions = sorted(set(selected_conditions) - set(EXPECTED_CONDITIONS))
    if unknown_conditions:
        raise OutcomeTransferLaunchError(f"Unknown conditions: {unknown_conditions}")
    if len(set(selected_conditions)) != len(selected_conditions):
        raise OutcomeTransferLaunchError("Conditions cannot contain duplicates.")
    selected_seeds = [int(seed) for seed in (seeds or plan["seeds"])]
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise OutcomeTransferLaunchError("Seeds must be a non-empty unique list.")
    invalid_seeds = sorted(set(selected_seeds) - set(plan["seeds"]))
    if invalid_seeds:
        raise OutcomeTransferLaunchError(
            f"Seeds are not declared in the manifest: {invalid_seeds}."
        )

    # A legacy full OPERA checkpoint can pre-date transfer-specific metadata.
    # Its path must then be explicitly seed-bound; never infer seed identity
    # from a manually typed one-off path or let one unlabelled file stand in
    # for multiple canonical seeds.
    full_checkpoint_paths: dict[int, Path] = {}
    full_template_binds_seed = False
    if "opera_full" in selected_conditions and full_checkpoint_template:
        full_template_binds_seed = _template_binds_seed(full_checkpoint_template)
        full_checkpoint_paths = {
            seed: _format_full_checkpoint_template(full_checkpoint_template, seed)
            for seed in selected_seeds
        }
        if (
            len(selected_seeds) > 1
            and full_template_binds_seed
            and len({str(path) for path in full_checkpoint_paths.values()})
            != len(full_checkpoint_paths)
        ):
            raise OutcomeTransferLaunchError(
                "--full-checkpoint-template resolves multiple canonical seeds to "
                "the same path. Provide one distinct full checkpoint per seed."
            )

    validated_preflight: Path | None = None
    if execute and any(condition != "opera_full" for condition in selected_conditions):
        if preflight_report is None:
            raise OutcomeTransferLaunchError(
                "--execute requires --preflight-report for new transfer ablations. "
                "Run the label-only support preflight and review its CSV/JSON first."
            )
        validated_preflight = _validate_preflight_report(preflight_report, plan)

    generated_dir = Path(config_dir)
    if not generated_dir.is_absolute() and not generated_dir.exists():
        generated_dir = REPOSITORY_ROOT / generated_dir
    output_root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for condition in selected_conditions:
        resolved = plan["conditions"][condition]
        config_path = generated_dir / f"{condition}.yaml"
        _validate_generated_config(config_path, resolved, plan)
        for seed in selected_seeds:
            run_dir = output_root / "runs" / condition / f"seed_{seed}"
            command = build_training_command(
                condition=condition,
                seed=seed,
                run_dir=run_dir,
            )
            if bool(resolved["launch_blocked"]):
                rows.append(
                    _status_row(
                        condition=condition,
                        seed=seed,
                        status="blocked",
                        run_dir=run_dir,
                        plan=plan,
                        command=command,
                        detail=str(resolved["launch_blocked_reason"] or ""),
                    )
                )
                continue

            if condition == "opera_full":
                if not full_checkpoint_template:
                    rows.append(
                        _status_row(
                            condition=condition,
                            seed=seed,
                            status="full_checkpoint_required",
                            run_dir=run_dir,
                            plan=plan,
                            command=command,
                            detail=(
                                "Provide --full-checkpoint-template to reuse a "
                                "compatible full OPERA checkpoint; full is never "
                                "retrained implicitly."
                            ),
                        )
                    )
                    continue
                checkpoint = full_checkpoint_paths[seed]
                compatible, detail, _ = check_reusable_full_checkpoint(
                    checkpoint,
                    plan,
                    seed=seed,
                    allow_legacy_without_seed=full_template_binds_seed,
                )
                rows.append(
                    _status_row(
                        condition=condition,
                        seed=seed,
                        status="reused"
                        if compatible
                        else "incompatible_full_checkpoint",
                        run_dir=run_dir,
                        plan=plan,
                        command=None,
                        checkpoint_path=checkpoint,
                        detail=detail,
                    )
                )
                continue

            existing, existing_detail = _find_reusable_ablation_checkpoint(
                run_dir,
                plan,
                condition=condition,
                seed=seed,
            )
            if existing is not None and not overwrite:
                rows.append(
                    _status_row(
                        condition=condition,
                        seed=seed,
                        status="already_present",
                        run_dir=run_dir,
                        plan=plan,
                        command=command,
                        checkpoint_path=existing,
                        detail="Use --overwrite to rerun this ablation slot.",
                    )
                )
                continue
            if (
                existing is None
                and existing_detail != "checkpoint_not_found"
                and not overwrite
            ):
                rows.append(
                    _status_row(
                        condition=condition,
                        seed=seed,
                        status="existing_checkpoint_incompatible",
                        run_dir=run_dir,
                        plan=plan,
                        command=command,
                        detail=(
                            "A checkpoint exists in the run directory but does not "
                            f"match this resolved transfer slot: {existing_detail}. "
                            "Use --overwrite only after reviewing it."
                        ),
                    )
                )
                continue

            if not execute:
                rows.append(
                    _status_row(
                        condition=condition,
                        seed=seed,
                        status="dry_run",
                        run_dir=run_dir,
                        plan=plan,
                        command=command,
                    )
                )
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, text=True, capture_output=True)
            (run_dir / "contrastive_stdout.log").write_text(
                completed.stdout or "", encoding="utf-8"
            )
            (run_dir / "contrastive_stderr.log").write_text(
                completed.stderr or "", encoding="utf-8"
            )
            discovered, discovery_detail = _find_reusable_ablation_checkpoint(
                run_dir,
                plan,
                condition=condition,
                seed=seed,
            )
            if completed.returncode != 0:
                status = "failed"
            elif discovered is None:
                status = "completed_checkpoint_not_found"
            else:
                status = "completed"
            rows.append(
                _status_row(
                    condition=condition,
                    seed=seed,
                    status=status,
                    run_dir=run_dir,
                    plan=plan,
                    command=command,
                    checkpoint_path=discovered,
                    detail=(
                        f"returncode={completed.returncode}; "
                        f"checkpoint_discovery={discovery_detail}"
                    ),
                )
            )

    for row in rows:
        row["preflight_report"] = (
            "" if validated_preflight is None else str(validated_preflight)
        )
    _write_status(rows, output_root)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--conditions",
        default=",".join(EXPECTED_CONDITIONS),
        help="Comma-separated subset of the seven transfer conditions.",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated manifest seed subset.",
    )
    parser.add_argument(
        "--full-checkpoint-template",
        default=None,
        help="Existing full OPERA checkpoint path, with optional {seed} placeholder.",
    )
    parser.add_argument(
        "--preflight-report",
        default=None,
        help=(
            "outcome_transfer_support.json for this manifest; required with "
            "--execute when launching new ablations."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch ablation subprocesses; default is a dry run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    # Explicit --dry-run makes command examples self-documenting; it is the
    # default and intentionally conflicts with --execute.
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run cannot be used together.")
    conditions = _parse_csv_values(args.conditions, permitted=EXPECTED_CONDITIONS)
    try:
        seeds = [int(value) for value in _parse_csv_values(args.seeds)]
    except ValueError as exc:
        parser.error(f"--seeds must contain integers: {exc}")
        return
    rows = launch_outcome_transfer_conditions(
        manifest_path=args.manifest,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        conditions=conditions,
        seeds=seeds,
        full_checkpoint_template=args.full_checkpoint_template,
        preflight_report=args.preflight_report,
        execute=args.execute,
        overwrite=args.overwrite,
    )
    print(
        f"Recorded {len(rows)} outcome-transfer training slots in "
        f"{Path(args.output_dir) / STATUS_NAME}."
    )


if __name__ == "__main__":
    main()
