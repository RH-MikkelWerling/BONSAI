"""
Lightweight preflight checks for OPERA experiment configs.

This does not validate private data contents. It catches common "press go"
problems such as missing model variants, placeholder checkpoint paths, and
rarity configs without a named baseline.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any

from opera.config_contracts import (
    ConfigValidationError,
    load_sweep_config,
    validate_analysis_manifest,
    variant_applies_to_outcome,
)
from opera.evaluation.cohort_flow import (
    eligibility_file_path,
    load_eligibility_frame,
    validate_eligibility_frame,
)
from opera.evaluation.split_contract import validate_cross_stage_split_contract
from opera.evaluation.tasks import (
    competing_outcome_file_path,
    normalize_outcome_config,
    outcome_file_path,
)
from opera.functional.outcomes import (
    filter_outcome_eligibility,
    filter_registry_eligible_outcomes,
    resolve_registry_start_date,
)

PLACEHOLDER_PREFIXES = ("/ckpts/", "/results/", "/data/")
_UNRESOLVED_ENVIRONMENT = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[^%]+%")


def _looks_like_placeholder(value: str) -> bool:
    return value.startswith(PLACEHOLDER_PREFIXES)


def _has_unresolved_environment(value: str) -> bool:
    """Return whether a path still contains an unexpanded environment token."""
    return bool(_UNRESOLVED_ENVIRONMENT.search(value))


def _expand_path(value: str | Path) -> Path:
    """Expand conventional environment/home markers before checking a path.

    ``load_sweep_config`` already expands config values.  Keeping this small
    second expansion here makes the preflight robust when callers construct
    mappings indirectly or use a platform-native ``%VAR%`` spelling.
    """
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _read_table(path: Path) -> Any:
    """Read a supported tabular file without assuming parquet everywhere."""
    import pandas as pd

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported tabular file extension {path.suffix!r}; expected parquet or CSV."
    )


def _membership_ids_for_cohort(
    *,
    cohort: str,
    cohort_cfg: dict[str, Any],
    require_existing_paths: bool,
    membership_cache: dict[Path, Any],
    issues: list[str],
) -> tuple[set[Any] | None, Any | None]:
    """Validate a shared membership table and return its cohort-restricted IDs.

    A shared outcome parquet contains all hematology patients.  The readiness
    event counts must therefore use precisely the same membership restriction
    as the train/evaluation commands.  This helper intentionally validates the
    stronger production invariant of one membership row per ``subject_id``.
    """
    raw_path = cohort_cfg.get("population_file")
    cohort_col = cohort_cfg.get("cohort_fine_col")
    cohort_value = cohort_cfg.get("cohort_fine_value")
    has_cohort_filter = bool(cohort_col or cohort_value)

    if raw_path in (None, "", "null"):
        if has_cohort_filter and require_existing_paths:
            issues.append(
                f"Cohort {cohort!r} configures cohort membership filtering but "
                "has no population_file."
            )
        return None, None

    raw_path = str(raw_path)
    if _has_unresolved_environment(raw_path):
        issues.append(
            f"Cohort {cohort!r} population_file contains an unresolved "
            f"environment variable: {raw_path}"
        )
        return None, None

    membership_path = _expand_path(raw_path)
    if membership_path.suffix.lower() not in {".parquet", ".pq", ".csv", ".txt"}:
        issues.append(
            f"Cohort {cohort!r} population_file must be parquet or CSV: "
            f"{membership_path}"
        )
        return None, None
    if not require_existing_paths:
        return None, None
    if not membership_path.exists():
        issues.append(
            f"Cohort {cohort!r} population_file does not exist: {membership_path}"
        )
        return None, None

    try:
        membership = membership_cache.get(membership_path)
        if membership is None:
            membership = _read_table(membership_path)
            membership_cache[membership_path] = membership
    except Exception as exc:
        issues.append(
            f"Could not read cohort {cohort!r} population_file {membership_path}: {exc}"
        )
        return None, None

    if "subject_id" not in membership.columns:
        issues.append(
            f"Cohort {cohort!r} population_file is missing required column "
            f"'subject_id': {membership_path}"
        )
        return None, membership
    if membership["subject_id"].isna().any():
        issues.append(
            f"Cohort {cohort!r} population_file contains missing subject_id values: "
            f"{membership_path}"
        )
        return None, membership
    duplicate_count = int(membership["subject_id"].duplicated().sum())
    if duplicate_count:
        issues.append(
            f"Cohort {cohort!r} population_file contains {duplicate_count} "
            f"duplicate subject_id rows: {membership_path}"
        )
        return None, membership

    if bool(cohort_col) != bool(cohort_value):
        # The config contract normally catches this.  Keep the readiness
        # check defensive for direct callers and future config migrations.
        issues.append(
            f"Cohort {cohort!r} must set cohort_fine_col and cohort_fine_value "
            "together."
        )
        return None, membership
    if cohort_col:
        if cohort_col not in membership.columns:
            issues.append(
                f"Cohort {cohort!r} membership column {cohort_col!r} is not in "
                f"{membership_path}; columns={list(membership.columns)}"
            )
            return None, membership
        scoped = membership[membership[cohort_col].astype(str) == str(cohort_value)]
        if scoped.empty:
            issues.append(
                f"Cohort {cohort!r} has no patients after filtering "
                f"{cohort_col}={cohort_value!r} in {membership_path}."
            )
            return set(), membership
        return set(scoped["subject_id"]), membership

    if membership.empty:
        issues.append(
            f"Cohort {cohort!r} population_file contains no patients: {membership_path}"
        )
        return set(), membership
    return set(membership["subject_id"]), membership


def check_sweep_config(
    config_path: str,
    require_existing_paths: bool = False,
    split_contract_path: str | None = None,
) -> list[str]:
    try:
        cfg = load_sweep_config(config_path).to_mapping()
    except ConfigValidationError as exc:
        return [f"Invalid sweep config: {issue}" for issue in exc.issues]

    issues: list[str] = []
    for key in ("cohorts", "outcomes", "model_variants"):
        if key not in cfg or not cfg[key]:
            issues.append(f"Missing or empty required block: {key}")

    variants = cfg.get("model_variants", {})
    if not variants:
        return issues

    if not any("encoder_ckpt" in item for item in variants.values()):
        issues.append("No foundation-model variants with encoder_ckpt were configured.")
    if _has_unresolved_environment(str(cfg.get("output_dir", ""))):
        issues.append(
            f"Sweep output_dir contains an unresolved environment variable: "
            f"{cfg.get('output_dir')}"
        )

    if cfg.get("rarity_mode") in {"synthetic", "real"} or cfg.get("rarity"):
        baseline = cfg.get("baseline_model") or cfg.get("rarity", {}).get(
            "baseline_model"
        )
        if not baseline:
            issues.append("Rarity config is missing baseline_model.")
        elif baseline not in variants:
            issues.append(
                f"Rarity baseline_model={baseline!r} is not in model_variants."
            )

    for name, variant in variants.items():
        training_mode = variant.get("training_mode")
        if training_mode is not None and training_mode not in {
            "cox",
            "cox_exact_cached",
            "ipcw_bce",
            "ipcw_cif_bce",
        }:
            issues.append(
                f"Variant {name!r} has invalid training_mode={training_mode!r}."
            )
        if (
            training_mode in {"cox", "cox_exact_cached"}
            and variant.get("pos_weight") is not None
        ):
            issues.append(
                f"Variant {name!r} uses Cox training with pos_weight configured."
            )
        for key in ("encoder_ckpt", "results_file", "predictions_file"):
            value = variant.get(key)
            if not value:
                continue
            if _looks_like_placeholder(str(value)):
                issues.append(
                    f"Variant {name!r} uses placeholder-looking {key}: {value}"
                )
            if _has_unresolved_environment(str(value)):
                issues.append(
                    f"Variant {name!r} {key} contains an unresolved environment "
                    f"variable: {value}"
                )
            if (
                require_existing_paths
                and key != "encoder_ckpt"
                and "{" not in str(value)
                and not _has_unresolved_environment(str(value))
                and not Path(value).exists()
            ):
                issues.append(f"Variant {name!r} {key} does not exist: {value}")
            if (
                require_existing_paths
                and key == "encoder_ckpt"
                and "{" not in str(value)
                and not _has_unresolved_environment(str(value))
                and not Path(value).exists()
            ):
                issues.append(f"Variant {name!r} {key} does not exist: {value}")

    has_ipcw_bce_variant = any(
        variant.get("training_mode") in {"ipcw_bce", "ipcw_cif_bce"}
        for variant in variants.values()
    )
    normalized_outcomes = normalize_outcome_config(cfg.get("outcomes") or {})
    if has_ipcw_bce_variant:
        for variant_name, variant in variants.items():
            if variant.get("training_mode") not in {"ipcw_bce", "ipcw_cif_bce"}:
                continue
            for outcome_name, outcome_cfg in normalized_outcomes.items():
                if not variant_applies_to_outcome(variant, outcome_name):
                    continue
                if outcome_cfg.get("n_hours_end_include") is None:
                    issues.append(
                        f"Outcome {outcome_name!r} selected by variant "
                        f"{variant_name!r} has no n_hours_end_include; "
                        "IPCW-BCE variants require a fixed horizon."
                    )

    # Membership is shared across all production fine/grouped configurations.
    # Cache it by path so a full 87-outcome preflight does not reread the same
    # large parquet once per cohort.
    membership_cache: dict[Path, Any] = {}
    for cohort, cohort_cfg in cfg.get("cohorts", {}).items():
        data_dir = cohort_cfg.get("data_dir")
        cohort_outcome_paths: list[Path] = []
        if not data_dir:
            issues.append(f"Cohort {cohort!r} is missing data_dir.")
        elif _has_unresolved_environment(str(data_dir)):
            issues.append(
                f"Cohort {cohort!r} data_dir contains an unresolved environment "
                f"variable: {data_dir}"
            )
        elif require_existing_paths and not _expand_path(data_dir).exists():
            issues.append(f"Cohort {cohort!r} data_dir does not exist: {data_dir}")

        membership_ids, membership_frame = _membership_ids_for_cohort(
            cohort=cohort,
            cohort_cfg=cohort_cfg,
            require_existing_paths=require_existing_paths,
            membership_cache=membership_cache,
            issues=issues,
        )
        membership_required = (
            cohort_cfg.get("population_file") not in (None, "", "null")
            or cohort_cfg.get("cohort_fine_col") not in (None, "", "null")
            or cohort_cfg.get("cohort_fine_value") not in (None, "", "null")
        )
        if cohort_cfg.get("registry_start_date") in (None, "", "null"):
            print(
                f"Cohort {cohort!r} has no registry_start_date; supervised "
                "analyses will include all prediction dates for now."
            )
        if (
            require_existing_paths
            and data_dir
            and not _has_unresolved_environment(str(data_dir))
        ):
            for outcome_name, outcome_cfg in normalized_outcomes.items():
                outcome_path = _expand_path(
                    outcome_file_path(data_dir, outcome_name, outcome_cfg)
                )
                outcome_df = None
                if _has_unresolved_environment(str(outcome_path)):
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} file contains "
                        f"an unresolved environment variable: {outcome_path}"
                    )
                elif not outcome_path.exists():
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} file "
                        f"does not exist: {outcome_path}"
                    )
                else:
                    if outcome_path not in cohort_outcome_paths:
                        cohort_outcome_paths.append(outcome_path)
                    try:
                        outcome_df = _read_table(outcome_path)
                        missing_columns = {"subject_id", "split"} - set(
                            outcome_df.columns
                        )
                        if missing_columns:
                            issues.append(
                                f"Cohort {cohort!r} outcome {outcome_name!r} is "
                                f"missing required columns: {sorted(missing_columns)}"
                            )
                            outcome_df = None
                        elif membership_required and membership_ids is None:
                            # Do not accidentally report whole-population counts as
                            # fine/grouped-cohort counts if membership validation
                            # already failed above.
                            print(
                                f"Skipping event-count audit for {cohort!r}/"
                                f"{outcome_name!r}: cohort membership could not be "
                                "resolved."
                            )
                            outcome_df = None
                        elif membership_ids is not None:
                            outcome_df = outcome_df[
                                outcome_df["subject_id"].isin(membership_ids)
                            ].copy()
                            if outcome_df.empty:
                                issues.append(
                                    f"Cohort {cohort!r} outcome {outcome_name!r} "
                                    "has no rows after membership filtering."
                                )
                    except Exception as exc:
                        print(
                            f"Could not inspect outcome data for {cohort!r}/"
                            f"{outcome_name!r}: {exc}"
                        )
                eligibility_path = eligibility_file_path(
                    data_dir,
                    cohort,
                    outcome_name,
                    outcome_cfg,
                )
                if eligibility_path is None:
                    print(
                        f"Cohort {cohort!r} outcome {outcome_name!r} has no "
                        "eligibility_file; cohort-flow denominators cannot be audited."
                    )
                    eligibility = None
                    eligibility_issues: list[str] = []
                elif _has_unresolved_environment(str(eligibility_path)):
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} eligibility "
                        f"file contains an unresolved environment variable: "
                        f"{eligibility_path}"
                    )
                    eligibility = None
                    eligibility_issues = ["unresolved path"]
                elif not eligibility_path.exists():
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} eligibility "
                        f"file does not exist: {eligibility_path}"
                    )
                    eligibility = None
                    eligibility_issues = ["missing path"]
                else:
                    try:
                        eligibility = load_eligibility_frame(eligibility_path)
                        eligibility_issues = validate_eligibility_frame(eligibility)
                        issues.extend(
                            f"Cohort {cohort!r} outcome {outcome_name!r} "
                            f"eligibility file {issue}."
                            for issue in eligibility_issues
                        )
                    except Exception as exc:
                        issues.append(
                            f"Could not inspect eligibility file for "
                            f"{cohort!r}/{outcome_name!r}: {exc}"
                        )
                        eligibility = None
                        eligibility_issues = ["unreadable"]

                # Count events on the exact supervised cohort: shared outcome
                # table -> membership restriction -> eligibility -> registry
                # coverage -> held-out split.  This prevents a large parent
                # cohort from masking an underpowered fine cohort at preflight.
                if outcome_df is not None:
                    try:
                        if eligibility is not None and not eligibility_issues:
                            outcome_df = filter_outcome_eligibility(
                                outcome_df,
                                eligibility,
                                cohort=cohort,
                                outcome_name=outcome_name,
                            )
                        registry_start_date = resolve_registry_start_date(
                            cohort_cfg,
                            outcome_cfg,
                        )
                        outcome_df = filter_registry_eligible_outcomes(
                            outcome_df,
                            registry_start_date,
                            cohort=cohort,
                            outcome_name=outcome_name,
                        )
                        test_df = outcome_df[
                            outcome_df["split"] == cfg.get("test_key", "held_out")
                        ].copy()
                        if "event" in test_df.columns:
                            n_events = int((test_df["event"] == 1).sum())
                            if n_events < 5:
                                issues.append(
                                    f"Cohort {cohort!r} outcome {outcome_name!r} "
                                    f"has only {n_events} held-out events after "
                                    "membership/eligibility filtering."
                                )
                            if int((test_df["event"] == 2).sum()) > 0:
                                issues.append(
                                    f"Cohort {cohort!r} outcome {outcome_name!r} "
                                    "contains competing events (event == 2)."
                                )

                        ipi_col = cohort_cfg.get("ipi_score_col")
                        if (
                            ipi_col
                            and membership_frame is not None
                            and ipi_col in membership_frame.columns
                            and not test_df.empty
                        ):
                            population = membership_frame[["subject_id", ipi_col]]
                            merged = test_df[["subject_id"]].merge(
                                population,
                                on="subject_id",
                                how="left",
                                validate="many_to_one",
                            )
                            coverage = float(merged[ipi_col].notna().mean())
                            if coverage < 0.5:
                                issues.append(
                                    f"Cohort {cohort!r} IPI coverage for {outcome_name!r} "
                                    f"is {coverage:.0%}; IPI rows will be skipped."
                                )
                            elif coverage < 0.8:
                                issues.append(
                                    f"Cohort {cohort!r} IPI coverage for {outcome_name!r} "
                                    f"is {coverage:.0%}; usable with caveats."
                                )
                            else:
                                print(
                                    f"Cohort {cohort!r} IPI coverage for {outcome_name!r} "
                                    f"is {coverage:.0%}; usable."
                                )
                        elif (
                            ipi_col
                            and membership_frame is not None
                            and ipi_col not in membership_frame.columns
                        ):
                            issues.append(
                                f"Cohort {cohort!r} IPI column {ipi_col!r} is not in "
                                "population_file."
                            )
                    except Exception as exc:
                        print(
                            f"Could not inspect filtered outcome data for {cohort!r}/"
                            f"{outcome_name!r}: {exc}"
                        )

                competing_path = competing_outcome_file_path(data_dir, outcome_cfg)
                if competing_path and _has_unresolved_environment(str(competing_path)):
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} competing "
                        f"outcome path contains an unresolved environment variable: "
                        f"{competing_path}"
                    )
                elif competing_path and not _expand_path(competing_path).exists():
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} competing "
                        f"outcome file does not exist: {competing_path}"
                    )
                for variant_name, variant in variants.items():
                    checkpoint_template = variant.get("encoder_ckpt")
                    if (
                        checkpoint_template
                        and "{" in str(checkpoint_template)
                        and not _has_unresolved_environment(str(checkpoint_template))
                    ):
                        checkpoint_path = Path(
                            str(checkpoint_template).format(
                                cohort=cohort,
                                outcome=outcome_name,
                                seed=cfg.get("seeds", [42])[0],
                            )
                        )
                        if not checkpoint_path.exists():
                            issues.append(
                                f"Variant {variant_name!r} encoder_ckpt does not "
                                f"exist: {checkpoint_path}"
                            )
                    template = variant.get("predictions_file")
                    if not template:
                        continue
                    pred_path = Path(
                        str(template).format(
                            cohort=cohort,
                            outcome=outcome_name,
                        )
                    )
                    if not pred_path.exists():
                        issues.append(
                            f"Variant {variant_name!r} prediction file does not exist: {pred_path}"
                        )
                    else:
                        try:
                            import pandas as pd

                            pred = (
                                pd.read_parquet(pred_path)
                                if pred_path.suffix.lower() in {".parquet", ".pq"}
                                else pd.read_csv(pred_path, nrows=5)
                            )
                            missing = {"subject_id", "probability"} - set(pred.columns)
                            if missing:
                                issues.append(
                                    f"Variant {variant_name!r} prediction file is missing "
                                    f"columns: {sorted(missing)}"
                                )
                        except Exception as exc:
                            print(
                                f"Could not inspect prediction file for "
                                f"{variant_name!r}: {exc}"
                            )

            if split_contract_path and cohort_outcome_paths:
                try:
                    split_report = validate_cross_stage_split_contract(
                        outcome_paths=cohort_outcome_paths,
                        subject_data_paths={
                            "ssl_train": str(Path(data_dir) / "subject_data_train.pt"),
                            "ssl_validation": str(
                                Path(data_dir) / "subject_data_tuning.pt"
                            ),
                        },
                        contract_path=split_contract_path,
                    )
                    issues.extend(
                        f"Cohort {cohort!r} split contract: {issue}"
                        for issue in split_report["issues"]
                    )
                except Exception as exc:
                    issues.append(
                        f"Cohort {cohort!r} split contract could not be validated: {exc}"
                    )

    try:
        import bonsai
        from bonsai.functional import checkpointing, outcomes

        version = getattr(bonsai, "__version__", "unknown")
        missing = [
            name
            for name in ("attach_checkpoint_metadata", "attach_model_config")
            if not hasattr(checkpointing, name)
        ]
        missing.extend(
            name for name in ("binarize_outcomes",) if not hasattr(outcomes, name)
        )
        if missing:
            issues.append(f"BONSAI import check missing symbols: {', '.join(missing)}")
        else:
            print(f"BONSAI import check passed (version={version}).")
    except Exception as exc:
        print(f"BONSAI import check failed: {exc}")

    return issues


def check_manifest_consistency(
    manifest_path: str | Path,
    sweep_config_path: str | Path,
) -> list[str]:
    """Cross-check the analysis manifest against a sweep config.

    Returns a list of human-readable issue strings. Empty list means consistent.

    Checks:
    1. Every outcome in manifest.primary_endpoints exists in the sweep outcomes block.
    2. Every model name appearing in manifest.primary_contrasts exists in model_variants.
    3. manifest.seeds is a subset of the sweep config seeds (or equal).
    4. manifest.min_events.test is a positive integer.
    """
    try:
        manifest = validate_analysis_manifest(manifest_path)
    except ConfigValidationError as exc:
        return [f"Invalid analysis manifest: {issue}" for issue in exc.issues]

    issues: list[str] = []

    try:
        cfg = load_sweep_config(sweep_config_path).to_mapping()
    except ConfigValidationError as exc:
        return [f"Invalid sweep config: {issue}" for issue in exc.issues]

    sweep_outcomes = set(normalize_outcome_config(cfg.get("outcomes") or {}))
    sweep_variants = set(cfg.get("model_variants") or {})
    sweep_seeds = cfg.get("seeds")

    # Check 1: primary endpoints exist in sweep outcomes.
    primary_endpoints = manifest.get("primary_endpoints")
    if isinstance(primary_endpoints, list):
        for endpoint in primary_endpoints:
            if endpoint not in sweep_outcomes:
                issues.append(
                    f"Manifest primary_endpoint {endpoint!r} is not present in the "
                    f"sweep outcomes block."
                )

    # Check 2: every model in primary_contrasts exists in model_variants.
    primary_contrasts = manifest.get("primary_contrasts")
    if isinstance(primary_contrasts, list):
        contrast_models: list[str] = []
        for pair in primary_contrasts:
            if isinstance(pair, (list, tuple)):
                contrast_models.extend(
                    model for model in pair if isinstance(model, str)
                )
        for model in dict.fromkeys(contrast_models):
            if model not in sweep_variants:
                issues.append(
                    f"Manifest primary_contrasts references model {model!r} that is "
                    f"not in the sweep model_variants block."
                )

    # Check 3: manifest seeds are a subset of (or equal to) the sweep seeds.
    manifest_seeds = manifest.get("seeds")
    if isinstance(manifest_seeds, list) and isinstance(sweep_seeds, list):
        extra_seeds = [seed for seed in manifest_seeds if seed not in set(sweep_seeds)]
        if extra_seeds:
            issues.append(
                f"Manifest seeds {sorted(extra_seeds)} are not present in the sweep "
                f"config seeds {sorted(sweep_seeds)}."
            )

    # Check 4: min_events.test is a positive integer.
    min_events = manifest.get("min_events")
    if isinstance(min_events, dict) and "test" in min_events:
        test_min = min_events.get("test")
        if isinstance(test_min, bool) or not isinstance(test_min, int) or test_min < 1:
            issues.append(
                f"Manifest min_events.test must be a positive integer, got "
                f"{test_min!r}."
            )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OPERA sweep readiness")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--require_existing_paths", action="store_true")
    parser.add_argument("--split_contract", default=None)
    parser.add_argument("--fail_on_issue", action="store_true")
    args = parser.parse_args()

    issues = check_sweep_config(
        args.config,
        require_existing_paths=args.require_existing_paths,
        split_contract_path=args.split_contract,
    )
    if issues:
        print("Readiness issues:")
        for issue in issues:
            print(f"  - {issue}")
        if args.fail_on_issue:
            raise SystemExit(1)
    else:
        print("Readiness check passed.")

    if args.manifest and Path(args.manifest).exists():
        manifest_issues = check_manifest_consistency(args.manifest, args.config)
        if manifest_issues:
            print(f"Manifest consistency check: {len(manifest_issues)} issues found.")
            for issue in manifest_issues:
                print(f"  - {issue}")
            issues.extend(manifest_issues)
            if args.fail_on_issue:
                raise SystemExit(1)
        else:
            print("Manifest consistency check passed.")


if __name__ == "__main__":
    main()
