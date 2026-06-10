"""Run patient-level benefit analyses for configured model contrasts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from opera.evaluation.patient_transfer import (
    build_patient_transfer_table,
    compute_atypicality_scores,
)
from opera.visualization.patient_benefit_plots import (
    plot_benefit_contrast_ladder,
    plot_patient_benefit,
)

LOGGER = logging.getLogger(__name__)


def _read_table(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source)


def _resolve(template: str, cohort: str, outcome: str) -> str:
    return template.format(cohort=cohort, outcome=outcome)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(frame["x"], frame["y"])
        return float(rho), float(pval)
    except Exception:
        rho = frame["x"].rank().corr(frame["y"].rank())
        return float(rho), float("nan")


def _required_paths(
    contrast: dict[str, Any], cohort: str, outcome: str
) -> dict[str, str]:
    keys = [
        "baseline_predictions",
        "comparator_predictions",
        "baseline_embeddings",
        "metadata",
    ]
    return {key: _resolve(contrast[key], cohort, outcome) for key in keys}


def _cell_missing(paths: dict[str, str]) -> list[str]:
    return [f"{key}={path}" for key, path in paths.items() if not Path(path).exists()]


def _identity_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    subject_col: str,
    cohort_col: str,
) -> list[str]:
    keys = [subject_col]
    for col in (cohort_col, "outcome"):
        if col in left.columns and col in right.columns:
            keys.append(col)
    return keys


def run_patient_benefit_analysis(
    config_path: str,
    *,
    contrast_filter: str | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    contrasts = cfg.get("contrasts", [])
    if contrast_filter:
        contrasts = [item for item in contrasts if item.get("name") == contrast_filter]
    cohorts = cfg.get("cohorts", [])
    outcomes = cfg.get("outcomes", [])
    ladder_inputs = []
    summary_rows = []
    primary_outputs = []

    for contrast in contrasts:
        name = contrast["name"]
        contrast_dir = output_dir / name
        if not dry_run:
            contrast_dir.mkdir(parents=True, exist_ok=True)
        pooled_transfer = []
        pooled_atypicality = []
        pooled_embeddings = []

        for cohort in cohorts:
            for outcome in outcomes:
                paths = _required_paths(contrast, cohort, outcome)
                missing = _cell_missing(paths)
                cell_dir = contrast_dir / cohort / outcome
                if missing:
                    LOGGER.warning(
                        "Skipping %s/%s/%s; missing %s",
                        name,
                        cohort,
                        outcome,
                        "; ".join(missing),
                    )
                    if dry_run:
                        print(
                            f"[DRY RUN] missing {name}/{cohort}/{outcome}: {'; '.join(missing)}"
                        )
                    continue
                if dry_run:
                    print(f"[DRY RUN] would run {name}/{cohort}/{outcome}")
                    continue

                baseline_pred = _read_table(paths["baseline_predictions"])
                comparator_pred = _read_table(paths["comparator_predictions"])
                embeddings = _read_table(paths["baseline_embeddings"])
                metadata = _read_table(paths["metadata"])
                cohort_col = contrast.get("cohort_col", "cohort")
                subject_col = contrast.get("subject_col", "subject_id")
                if cohort_col not in metadata.columns:
                    metadata[cohort_col] = cohort
                if cohort_col not in embeddings.columns:
                    embeddings = embeddings.merge(
                        metadata[[subject_col, cohort_col]].drop_duplicates(
                            subject_col
                        ),
                        on=subject_col,
                        how="left",
                    )

                transfer = build_patient_transfer_table(
                    baseline_pred,
                    comparator_pred,
                    embeddings,
                    metadata,
                    subject_col=subject_col,
                    contrast_name=name,
                    feature_cols=contrast.get("feature_cols"),
                    k=int(contrast.get("k", 20)),
                )
                atypicality = compute_atypicality_scores(
                    embeddings,
                    metadata,
                    subject_col=subject_col,
                    cohort_col=cohort_col,
                    feature_cols=contrast.get("feature_cols"),
                    min_cohort_size=int(contrast.get("min_cohort_size", 10)),
                )
                transfer["cohort"] = cohort
                transfer["outcome"] = outcome
                atypicality["cohort"] = cohort
                atypicality["outcome"] = outcome
                cell_dir.mkdir(parents=True, exist_ok=True)
                transfer.to_csv(cell_dir / "patient_transfer.csv", index=False)
                atypicality.to_csv(cell_dir / "atypicality.csv", index=False)
                pooled_transfer.append(transfer)
                pooled_atypicality.append(atypicality)
                pooled_embeddings.append(
                    embeddings.assign(cohort=cohort, outcome=outcome)
                )

        if dry_run or not pooled_transfer:
            continue

        transfer_all = pd.concat(pooled_transfer, ignore_index=True)
        atypicality_all = pd.concat(pooled_atypicality, ignore_index=True)
        embeddings_all = pd.concat(pooled_embeddings, ignore_index=True)
        embedding_keys = [
            col
            for col in (
                contrast.get("subject_col", "subject_id"),
                contrast.get("cohort_col", "cohort"),
            )
            if col in embeddings_all.columns
        ]
        if embedding_keys:
            embeddings_all = embeddings_all.drop_duplicates(subset=embedding_keys)
        transfer_all.to_csv(contrast_dir / "pooled_patient_transfer.csv", index=False)
        atypicality_all.to_csv(contrast_dir / "pooled_atypicality.csv", index=False)

        fig = plot_patient_benefit(
            transfer_all,
            atypicality_all,
            embeddings_all,
            contrast_name=name,
            atypicality_mode=contrast.get("atypicality_mode", "own"),
            gain_col=contrast.get("gain_col", "brier_gain"),
            cohort_col=contrast.get("cohort_col", "cohort"),
            subject_col=contrast.get("subject_col", "subject_id"),
            save_path=str(contrast_dir / "patient_benefit.pdf"),
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        ladder_inputs.append(
            {
                "contrast_name": name,
                "patient_transfer_df": transfer_all,
                "atypicality_df": atypicality_all,
            }
        )
        if contrast.get("primary", False):
            primary_outputs.append(
                (contrast, transfer_all, atypicality_all, embeddings_all)
            )

        subject_col = contrast.get("subject_col", "subject_id")
        cohort_col = contrast.get("cohort_col", "cohort")
        merge_keys = _identity_keys(
            transfer_all, atypicality_all, subject_col, cohort_col
        )
        merged = transfer_all.merge(
            atypicality_all,
            on=merge_keys,
            how="inner",
            suffixes=("", "_atyp"),
        )
        rho_own, p_own = _spearman(
            merged.get("atypicality_own", pd.Series(dtype=float)),
            merged[contrast.get("gain_col", "brier_gain")],
        )
        rho_nearest, p_nearest = _spearman(
            merged.get("atypicality_nearest", pd.Series(dtype=float)),
            merged[contrast.get("gain_col", "brier_gain")],
        )
        summary_rows.append(
            {
                "contrast_name": name,
                "n_patients": int(len(merged)),
                "mean_brier_gain": float(
                    merged[contrast.get("gain_col", "brier_gain")].mean()
                ),
                "spearman_rho_own": rho_own,
                "spearman_p_own": p_own,
                "spearman_rho_nearest": rho_nearest,
                "spearman_p_nearest": p_nearest,
                "fraction_positive_gain": float(
                    (merged[contrast.get("gain_col", "brier_gain")] > 0).mean()
                ),
                "n_cohorts": int(transfer_all["cohort"].nunique()),
                "n_outcomes": int(transfer_all["outcome"].nunique()),
            }
        )

    if not dry_run and ladder_inputs:
        ladder = plot_benefit_contrast_ladder(
            ladder_inputs,
            save_path=str(output_dir / "contrast_ladder.pdf"),
        )
        import matplotlib.pyplot as plt

        plt.close(ladder)
    if not dry_run and primary_outputs:
        for i, (contrast, transfer, atypicality, embeddings) in enumerate(
            primary_outputs
        ):
            path = output_dir / (
                "main_figure_patient_benefit.pdf"
                if i == 0
                else f"main_figure_patient_benefit_{contrast['name']}.pdf"
            )
            fig = plot_patient_benefit(
                transfer,
                atypicality,
                embeddings,
                contrast_name=contrast["name"],
                atypicality_mode=contrast.get("atypicality_mode", "own"),
                gain_col=contrast.get("gain_col", "brier_gain"),
                cohort_col=contrast.get("cohort_col", "cohort"),
                subject_col=contrast.get("subject_col", "subject_id"),
                save_path=str(path),
            )
            import matplotlib.pyplot as plt

            plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    if not dry_run:
        summary.to_csv(output_dir / "contrast_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run patient-level OPERA benefit analysis"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--contrast", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_patient_benefit_analysis(
        args.config,
        contrast_filter=args.contrast,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
