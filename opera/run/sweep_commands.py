"""
Pure argv-construction helpers for the OPERA evaluation sweep.

Each function returns the exact ``list[str]`` command line that the sweep would
hand to :func:`subprocess.run`. They contain no I/O and no subprocess calls, so
they can be unit-tested directly to lock down the CLI contract between the sweep
orchestrator and the ``finetune`` / ``evaluate`` / ``evaluate_predictions``
entry points.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Training modes that route to the survival finetune entry point and config.
SURVIVAL_TRAINING_MODES: frozenset[str] = frozenset({"cox", "ipcw_bce"})


def build_finetune_cmd(
    encoder_ckpt: str,
    encoder_source: str,
    cohort: str,
    cohort_data_dir: str,
    outcome_name: str,
    outcome_path: str,
    output_dir: Path,
    base_config: str,
    n_hours_start_include: int = 1,
    n_hours_end_include: Optional[int] = None,
    competing_outcome_path: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    training_mode: Optional[str] = None,
    extra_overrides: Optional[list[str]] = None,
) -> list[str]:
    """Return the argv list for a finetune subprocess call."""
    overrides = [
        f"encoder_ckpt={encoder_ckpt}",
        f"encoder_source={encoder_source}",
        f"dataset={cohort}",
        f"outcome={outcome_name}",
        f"paths.dir={cohort_data_dir}",
        f"paths.outcome={outcome_path}",
        f"labels.n_hours_start_include={n_hours_start_include}",
        f"labels.n_hours_end_include={'null' if n_hours_end_include is None else n_hours_end_include}",
        f"labels.registry_start_date={'null' if registry_start_date is None else registry_start_date}",
        f"hydra.run.dir={output_dir}",
    ]
    if competing_outcome_path:
        overrides.append(f"paths.competing_outcome={competing_outcome_path}")
    if eligibility_path:
        overrides.append(f"paths.eligibility={eligibility_path}")
    if training_mode is not None:
        overrides.append(f"training_mode={training_mode}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    is_survival = training_mode in SURVIVAL_TRAINING_MODES
    module = "opera.run.survival_finetune" if is_survival else "opera.run.finetune"
    config_name = "survival_finetune" if is_survival else Path(base_config).stem
    cmd = [
        sys.executable,
        "-m",
        module,
        f"--config-name={config_name}",
    ] + overrides
    return cmd


def build_evaluate_cmd(
    ckpt_path: Path,
    cohort: str,
    cohort_data_dir: str,
    outcome_name: str,
    outcome_path: str,
    output_dir: Path,
    encoder_source: str = "contrastive",
    n_hours_start_include: int = 1,
    n_hours_end_include: Optional[int] = None,
    competing_outcome_path: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    model_family: Optional[str] = None,
    training_stage: str = "evaluation",
    encoder_frozen: Optional[bool] = None,
    head_type: Optional[str] = None,
    cohort_fine_col: Optional[str] = None,
    cohort_fine_value: Optional[str] = None,
) -> list[str]:
    """Return the argv list for an evaluate subprocess call."""
    overrides = [
        f"ckpt_path={ckpt_path}",
        f"dataset={cohort}",
        f"outcome={outcome_name}",
        f"paths.dir={cohort_data_dir}",
        f"paths.outcome={outcome_path}",
        f"output_dir={output_dir}",
        f"labels.n_hours_start_include={n_hours_start_include}",
        f"labels.n_hours_end_include={'null' if n_hours_end_include is None else n_hours_end_include}",
        f"labels.registry_start_date={'null' if registry_start_date is None else registry_start_date}",
        f"training_stage={training_stage}",
    ]
    if model_family:
        overrides.append(f"model_family={model_family}")
    if encoder_frozen is not None:
        overrides.append(f"encoder_frozen={str(encoder_frozen).lower()}")
    if head_type is not None:
        overrides.append(f"head_type={head_type}")
    if competing_outcome_path:
        overrides.append(f"paths.competing_outcome={competing_outcome_path}")
    if eligibility_path:
        overrides.append(f"paths.eligibility={eligibility_path}")
    if cohort_fine_col and cohort_fine_value:
        overrides.append(f"cohort_fine_col={cohort_fine_col}")
        overrides.append(f"cohort_fine_value={cohort_fine_value}")

    module = (
        "opera.run.evaluate_joint"
        if encoder_source == "joint"
        else "opera.run.evaluate"
    )
    if encoder_source == "joint":
        overrides = [f"outcome_name={outcome_name}"] + overrides
    cmd = [sys.executable, "-m", module] + overrides
    return cmd


def build_prediction_evaluate_cmd(
    predictions_path: str,
    cohort: str,
    outcome_name: str,
    outcome_path: str,
    model_family: str,
    output_dir: Path,
    n_hours_start_include: int,
    n_hours_end_include,
    competing_outcome_path: Optional[str] = None,
    eligibility_path: Optional[str] = None,
    registry_start_date: Optional[str] = None,
    rarity_mode: str = "none",
    baseline_model: Optional[str] = None,
    ipi_coverage: Optional[float] = None,
    evaluation_subset: str = "full",
    seed: int = 42,
    subgroup_path: Optional[str] = None,
    subgroup_columns: Optional[list[str]] = None,
) -> list[str]:
    """Return the argv list for an evaluate_predictions subprocess call."""
    end_value = "null" if n_hours_end_include is None else str(n_hours_end_include)
    cmd = [
        sys.executable,
        "-m",
        "opera.run.evaluate_predictions",
        "--predictions",
        predictions_path,
        "--outcome",
        outcome_path,
        "--output_dir",
        str(output_dir),
        "--cohort",
        cohort,
        "--outcome_name",
        outcome_name,
        "--model_family",
        model_family,
        "--n_hours_start_include",
        str(n_hours_start_include),
        "--rarity_mode",
        rarity_mode,
        "--seed",
        str(seed),
    ]
    if baseline_model:
        cmd.extend(["--baseline_model", baseline_model])
    if ipi_coverage is not None:
        cmd.extend(["--ipi_coverage", str(ipi_coverage)])
    cmd.extend(["--evaluation_subset", evaluation_subset])
    if n_hours_end_include is not None:
        cmd.extend(["--n_hours_end_include", end_value])
    if competing_outcome_path:
        cmd.extend(["--competing_outcome", competing_outcome_path])
    if eligibility_path:
        cmd.extend(["--eligibility", eligibility_path])
    if registry_start_date is not None:
        cmd.extend(["--registry_start_date", registry_start_date])
    if subgroup_path and subgroup_columns:
        cmd.extend(["--subgroups", subgroup_path])
        cmd.extend(["--subgroup_columns", ",".join(subgroup_columns)])
    return cmd
