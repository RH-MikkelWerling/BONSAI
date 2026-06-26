import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import OmegaConf


REQUIRED_RESULT_FIELDS = [
    "run_id",
    "config_hash",
    "model_family",
    "training_stage",
    "cohort",
    "outcome",
    "outcome_window_hours",
    "split",
    "seed",
    "training_fraction",
    "rarity_mode",
    "rarity_tier",
    "checkpoint_path",
    "encoder_frozen",
    "head_type",
    "baseline_model",
    "n_train",
    "n_val",
    "n_test",
    "n_events_train",
    "n_events_val",
    "n_events_test",
    "prevalence_train",
    "prevalence_val",
    "prevalence_test",
    "n_total",
    "n_positive",
    "prevalence",
    "auroc",
    "auroc_lower",
    "auroc_upper",
    "auprc",
    "brier_score",
    "ipi_coverage",
    "evaluation_subset",
    "evaluation_regime",
    "n_competing_events_train",
    "n_competing_events_val",
    "n_competing_events_test",
    "enrichment_top5pct",
    "enrichment_top10pct",
    "n_top5pct_events",
    "n_top10pct_events",
    "calibration_slope",
    "calibration_intercept",
    "hl_statistic",
    "hl_pvalue",
]

CANONICAL_TRAINING_STAGES = {
    "joint_finetune": "joint_finetuning",
}


def config_hash(cfg: Any) -> str:
    container = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    payload = json.dumps(container, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def flatten_report_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    discrimination = report.get("discrimination", {})
    calibration = report.get("calibration", {})

    for key, value in discrimination.items():
        row[key] = value
    for key in (
        "brier_score",
        "ece",
        "mce",
        "hl_statistic",
        "hl_pvalue",
        "calibration_intercept",
        "calibration_slope",
    ):
        if key in calibration:
            row[key] = calibration[key]

    survival = report.get("survival", {})
    if survival:
        row["concordance_index"] = survival.get("concordance_index")
        row["survival_n_total"] = survival.get("n_total")
        row["survival_n_events"] = survival.get("n_events")
        for horizon, metrics in survival.get("per_horizon", {}).items():
            for key, value in metrics.items():
                row[f"{key}_{horizon}"] = value

    stratified = report.get("stratified_concordance", {})
    micro = stratified.get("micro", {})
    macro = stratified.get("macro", {})
    if micro:
        row["c_index_within_fine"] = micro.get("c_index")
        row["c_index_within_fine_lower"] = micro.get("lower")
        row["c_index_within_fine_upper"] = micro.get("upper")
        row["n_comparable_within_fine"] = micro.get("n_comparable")
        row["n_strata_fine"] = micro.get("n_strata")
        row["stratified_concordance_col"] = stratified.get("strata_col")
    if macro:
        row["c_index_macro_within_fine"] = macro.get("c_index")
        row["c_index_macro_within_fine_lower"] = macro.get("lower")
        row["c_index_macro_within_fine_upper"] = macro.get("upper")
        row["n_strata_fine_estimable"] = macro.get("n_strata_estimable")

    bootstrap_ci = report.get("bootstrap_ci", {})
    for metric, values in bootstrap_ci.items():
        if isinstance(values, dict):
            for key in ("mean", "lower", "upper", "std"):
                if key in values:
                    row[f"{metric}_{key}"] = values[key]

    enrichment = report.get("high_risk_enrichment")
    if enrichment is not None:
        enrichment_rows = (
            enrichment.to_dict("records")
            if hasattr(enrichment, "to_dict")
            else enrichment
        )
        for item in enrichment_rows:
            frac = item.get("top_fraction")
            if frac == 0.05:
                row["enrichment_top5pct"] = item.get("enrichment")
                row["n_top5pct_events"] = item.get("n_events")
            elif frac == 0.10:
                row["enrichment_top10pct"] = item.get("enrichment")
                row["n_top10pct_events"] = item.get("n_events")
    return row


def bootstrap_ci_rows(report: Dict[str, Any]) -> pd.DataFrame:
    """Return scalar bootstrap intervals in long, machine-readable form."""
    rows = []
    for family, payload in (
        ("binary", report.get("bootstrap_ci", {})),
        ("survival", report.get("survival_bootstrap_ci", {})),
    ):
        for metric, values in payload.items():
            if not isinstance(values, dict):
                continue
            row = {"metric_family": family, "metric": metric}
            for key in ("mean", "lower", "upper", "std"):
                row[key] = values.get(key)
            rows.append(row)
    return pd.DataFrame(rows)


def canonical_training_stage(stage: Optional[str]) -> Optional[str]:
    if stage is None:
        return None
    return CANONICAL_TRAINING_STAGES.get(stage, stage)


def _mapping_get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _rarity_metadata(cfg: Any, report: Dict[str, Any]) -> Dict[str, Any]:
    rarity_cfg = _mapping_get(cfg, "rarity", {}) or {}
    size_cfg = (
        _mapping_get(rarity_cfg, "size_metadata", None)
        or _mapping_get(cfg, "cohort_size_metadata", None)
        or {}
    )
    discrimination = report.get("discrimination", {})

    out = {
        "rarity_mode": _mapping_get(
            rarity_cfg, "mode", _mapping_get(cfg, "rarity_mode", "none")
        ),
        "rarity_tier": _mapping_get(
            rarity_cfg, "tier", _mapping_get(cfg, "rarity_tier", None)
        ),
        "baseline_model": _mapping_get(
            rarity_cfg,
            "baseline_model",
            _mapping_get(cfg, "baseline_model", None),
        ),
    }
    for split in ("train", "val", "test"):
        out[f"n_{split}"] = _mapping_get(size_cfg, f"n_{split}", None)
        out[f"n_events_{split}"] = _mapping_get(size_cfg, f"n_events_{split}", None)
        out[f"n_competing_events_{split}"] = _mapping_get(
            size_cfg, f"n_competing_events_{split}", None
        )
        out[f"prevalence_{split}"] = _mapping_get(size_cfg, f"prevalence_{split}", None)

    out["n_test"] = (
        out["n_test"] if out["n_test"] is not None else discrimination.get("n_total")
    )
    out["n_events_test"] = (
        out["n_events_test"]
        if out["n_events_test"] is not None
        else discrimination.get("n_positive")
    )
    out["prevalence_test"] = (
        out["prevalence_test"]
        if out["prevalence_test"] is not None
        else discrimination.get("prevalence")
    )
    return out


def build_result_row(
    cfg: Any,
    report: Dict[str, Any],
    checkpoint_path: str,
    split: str,
    model_family: Optional[str] = None,
    training_stage: Optional[str] = None,
    seed: Optional[int] = None,
    training_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    labels_cfg = cfg.get("labels", {})
    row = {
        "run_id": cfg.get("run_id", config_hash(cfg)),
        "config_hash": config_hash(cfg),
        "model_family": model_family
        or cfg.get("model_family")
        or cfg.get("encoder_source", "unknown"),
        "training_stage": canonical_training_stage(
            training_stage or cfg.get("training_stage", "evaluation")
        ),
        "cohort": cfg.get("dataset", cfg.get("cohort", "unknown")),
        "outcome": cfg.get("outcome", cfg.get("outcome_name", "unknown")),
        "outcome_window_hours": labels_cfg.get("n_hours_end_include"),
        "split": split,
        "seed": seed if seed is not None else cfg.get("seed"),
        "training_fraction": (
            training_fraction
            if training_fraction is not None
            else cfg.get("training_fraction")
        ),
        "checkpoint_path": str(checkpoint_path),
        "encoder_frozen": cfg.get("encoder_frozen"),
        "head_type": cfg.get("head_type"),
        "ipi_coverage": cfg.get("ipi_coverage"),
        "evaluation_subset": cfg.get("evaluation_subset", "full"),
        "evaluation_regime": cfg.get("evaluation_regime", "both"),
    }
    row.update(flatten_report_metrics(report))
    row.update(_rarity_metadata(cfg, report))
    for field in REQUIRED_RESULT_FIELDS:
        row.setdefault(field, None)
    return row


def write_result_artifacts(row: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "result.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    pd.DataFrame([row]).to_csv(output_dir / "result.csv", index=False)


def write_per_cohort_concordance_artifact(
    report: Dict[str, Any],
    result_row: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """Write optional per-stratum concordance rows for rarity analyses."""
    macro = report.get("stratified_concordance", {}).get("macro", {})
    per_stratum = macro.get("per_stratum", [])
    if not per_stratum:
        return None

    metadata_keys = (
        "run_id",
        "config_hash",
        "model_family",
        "training_stage",
        "cohort",
        "outcome",
        "outcome_window_hours",
        "split",
        "seed",
        "training_fraction",
        "rarity_mode",
        "rarity_tier",
    )
    common = {key: result_row.get(key) for key in metadata_keys}
    strata_col = report["stratified_concordance"].get("strata_col")
    common["strata_col"] = strata_col

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "per_cohort_concordance.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for item in per_stratum:
            row = {**common, **item}
            if strata_col:
                row[strata_col] = item.get("stratum")
            handle.write(json.dumps(row, default=str) + "\n")
    return path
