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

from opera.evaluation.tasks import normalize_outcome_config, outcome_file_path

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in bare envs.
    raise SystemExit(
        "PyYAML is required for readiness checks. Install the repo with "
        '`python -m pip install -e ".[dev]"` first.'
    ) from exc


PLACEHOLDER_PREFIXES = ("/ckpts/", "/results/", "/data/")


def _looks_like_placeholder(value: str) -> bool:
    return value.startswith(PLACEHOLDER_PREFIXES)


def _expand_config_values(value):
    """Recursively expand environment variables in YAML config values."""
    if isinstance(value, str):
        value = re.sub(
            r"\$\{([^}]+)\}",
            lambda match: os.environ.get(match.group(1), match.group(0)),
            value,
        )
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_config_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_values(item) for key, item in value.items()}
    return value


def check_sweep_config(
    config_path: str, require_existing_paths: bool = False
) -> list[str]:
    with open(config_path) as f:
        cfg = _expand_config_values(yaml.safe_load(f) or {})

    issues: list[str] = []
    for key in ("cohorts", "outcomes", "model_variants"):
        if key not in cfg or not cfg[key]:
            issues.append(f"Missing or empty required block: {key}")

    variants = cfg.get("model_variants", {})
    if not variants:
        return issues

    if not any("encoder_ckpt" in item for item in variants.values()):
        issues.append("No foundation-model variants with encoder_ckpt were configured.")

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
        if training_mode is not None and training_mode not in {"cox", "ipcw_bce"}:
            issues.append(
                f"Variant {name!r} has invalid training_mode={training_mode!r}."
            )
        if training_mode == "cox" and variant.get("pos_weight") is not None:
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
            if (
                require_existing_paths
                and key != "encoder_ckpt"
                and "{" not in str(value)
                and not Path(value).exists()
            ):
                issues.append(f"Variant {name!r} {key} does not exist: {value}")

    has_ipcw_bce_variant = any(
        variant.get("training_mode") == "ipcw_bce" for variant in variants.values()
    )
    normalized_outcomes = normalize_outcome_config(cfg.get("outcomes") or {})
    if has_ipcw_bce_variant:
        for outcome_name, outcome_cfg in normalized_outcomes.items():
            if outcome_cfg.get("n_hours_end_include") is None:
                issues.append(
                    f"Outcome {outcome_name!r} has no n_hours_end_include; "
                    "IPCW-BCE variants require a fixed horizon."
                )

    for cohort, cohort_cfg in cfg.get("cohorts", {}).items():
        data_dir = cohort_cfg.get("data_dir")
        if not data_dir:
            issues.append(f"Cohort {cohort!r} is missing data_dir.")
        elif require_existing_paths and not Path(data_dir).exists():
            issues.append(f"Cohort {cohort!r} data_dir does not exist: {data_dir}")
        if cohort_cfg.get("registry_start_date") in (None, "", "null"):
            print(
                f"Cohort {cohort!r} has no registry_start_date; supervised "
                "analyses will include all prediction dates for now."
            )
        if require_existing_paths and data_dir:
            for outcome_name, outcome_cfg in normalized_outcomes.items():
                outcome_path = Path(
                    outcome_file_path(data_dir, outcome_name, outcome_cfg)
                )
                if not outcome_path.exists():
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} file "
                        f"does not exist: {outcome_path}"
                    )
                else:
                    try:
                        import pandas as pd

                        outcome_df = pd.read_parquet(outcome_path)
                        test_df = outcome_df[outcome_df.get("split") == "held_out"]
                        if "event" in test_df.columns:
                            n_events = int((test_df["event"] == 1).sum())
                            if n_events < 5:
                                issues.append(
                                    f"Cohort {cohort!r} outcome {outcome_name!r} "
                                    f"has only {n_events} held-out events."
                                )
                            if int((test_df["event"] == 2).sum()) > 0:
                                issues.append(
                                    f"Cohort {cohort!r} outcome {outcome_name!r} "
                                    "contains competing events (event == 2)."
                                )
                        ipi_col = cohort_cfg.get("ipi_score_col")
                        pop_path = Path(
                            cohort_cfg.get(
                                "population_file",
                                str(Path(data_dir) / "population_full.csv"),
                            )
                        )
                        if (
                            ipi_col
                            and pop_path.exists()
                            and "subject_id" in test_df.columns
                        ):
                            population = pd.read_csv(
                                pop_path, usecols=["subject_id", ipi_col]
                            )
                            merged = test_df[["subject_id"]].merge(
                                population,
                                on="subject_id",
                                how="left",
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
                    except Exception as exc:
                        print(
                            f"Could not inspect outcome data for {cohort!r}/"
                            f"{outcome_name!r}: {exc}"
                        )
                competing_path = outcome_cfg.get("competing_outcome_path")
                competing_file = outcome_cfg.get("competing_outcome_file")
                if competing_file and not competing_path:
                    competing_path = str(Path(data_dir) / "outcomes" / competing_file)
                if competing_path and not Path(competing_path).exists():
                    issues.append(
                        f"Cohort {cohort!r} outcome {outcome_name!r} competing "
                        f"outcome file does not exist: {competing_path}"
                    )
                for variant_name, variant in variants.items():
                    template = variant.get("predictions_file")
                    if not template:
                        continue
                    pred_path = Path(
                        str(template).format(cohort=cohort, outcome=outcome_name)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OPERA sweep readiness")
    parser.add_argument("--config", required=True)
    parser.add_argument("--require_existing_paths", action="store_true")
    parser.add_argument("--fail_on_issue", action="store_true")
    args = parser.parse_args()

    issues = check_sweep_config(
        args.config,
        require_existing_paths=args.require_existing_paths,
    )
    if issues:
        print("Readiness issues:")
        for issue in issues:
            print(f"  - {issue}")
        if args.fail_on_issue:
            raise SystemExit(1)
    else:
        print("Readiness check passed.")


if __name__ == "__main__":
    main()
