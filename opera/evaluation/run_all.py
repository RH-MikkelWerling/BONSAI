"""Top-level orchestration for the OPERA evaluation pipeline.

This script starts with Task 1, the comparison framework, and is structured so
later retrieval and interpretability stages can plug into the same manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from opera.evaluation.comparison import (
    ComparisonRunner,
    comparison_results_to_frame,
    make_jsonable,
)

LOGGER = logging.getLogger(__name__)


def load_object(path: str | Path) -> Any:
    """Load a dataframe, mapping, or array-like object from disk."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format for {path}.")


def run_comparison_grid(
    *,
    outcomes: pd.DataFrame,
    rkkp: Optional[pd.DataFrame],
    ehr_features: pd.DataFrame,
    embeddings_base: Mapping[Any, Any],
    embeddings_dapt: Mapping[Any, Any],
    embeddings_opera: Mapping[Any, Any],
    disease_cohorts: Mapping[str, Sequence[Any]],
    outcome_names: Optional[Sequence[str]] = None,
    cohorts: Optional[Sequence[str]] = None,
    mode: str = "disease_specific",
    evaluation_strategy: str = "prospective_holdout",
    seed: int = 42,
    n_splits: int = 5,
    n_bootstrap: int = 1000,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the Task 1 comparison benchmark over an outcome × cohort grid."""
    selected_outcomes = (
        list(outcome_names)
        if outcome_names is not None
        else sorted(outcomes["outcome_name"].dropna().unique().tolist())
    )
    selected_cohorts = (
        list(cohorts) if cohorts is not None else sorted(disease_cohorts.keys())
    )
    if mode == "pooled":
        selected_cohorts = ["all_hematology"]

    results: Dict[str, Any] = {
        "stage": "comparison",
        "mode": mode,
        "evaluation_strategy": evaluation_strategy,
        "seed": seed,
        "n_splits": n_splits,
        "n_bootstrap": n_bootstrap,
        "results": {},
    }

    total = len(selected_outcomes) * len(selected_cohorts)
    progress = tqdm(total=total, desc="run_all", disable=not show_progress)
    try:
        for outcome_name in selected_outcomes:
            results["results"].setdefault(outcome_name, {})
            for cohort_name in selected_cohorts:
                runner = ComparisonRunner(
                    outcome_name=outcome_name,
                    cohort_specification=cohort_name if mode != "pooled" else "all",
                    mode=mode,
                    outcomes=outcomes,
                    rkkp=rkkp,
                    ehr_features=ehr_features,
                    embeddings_base=embeddings_base,
                    embeddings_dapt=embeddings_dapt,
                    embeddings_opera=embeddings_opera,
                    disease_cohorts=disease_cohorts,
                    seed=seed,
                    n_splits=n_splits,
                    n_bootstrap=n_bootstrap,
                    evaluation_strategy=evaluation_strategy,
                    show_progress=False,
                )
                LOGGER.info(
                    "Running comparison grid cell outcome=%s cohort=%s mode=%s",
                    outcome_name,
                    cohort_name,
                    mode,
                )
                results["results"][outcome_name][cohort_name] = runner.run()
                progress.update(1)
    finally:
        progress.close()

    return make_jsonable(results)


def flatten_grid_results(grid_results: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten nested grid outputs into one publication-friendly dataframe."""
    rows: List[pd.DataFrame] = []
    for outcome_name, cohort_payload in grid_results.get("results", {}).items():
        for cohort_name, result in cohort_payload.items():
            frame = comparison_results_to_frame(result)
            if not frame.empty:
                frame["grid_outcome_name"] = outcome_name
                frame["grid_cohort_name"] = cohort_name
                rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OPERA Task 1 comparison pipeline."
    )
    parser.add_argument("--outcomes", required=True, help="Outcome parquet/csv.")
    parser.add_argument("--rkkp", default=None, help="Optional RKKP parquet/csv.")
    parser.add_argument(
        "--ehr_features", required=True, help="EHR feature parquet/csv."
    )
    parser.add_argument(
        "--embeddings_base",
        required=True,
        help="Pickle/JSON mapping of base embeddings.",
    )
    parser.add_argument(
        "--embeddings_dapt",
        required=True,
        help="Pickle/JSON mapping of DAPT embeddings.",
    )
    parser.add_argument(
        "--embeddings_opera",
        required=True,
        help="Pickle/JSON mapping of OPERA embeddings.",
    )
    parser.add_argument(
        "--disease_cohorts",
        required=True,
        help="Pickle/JSON mapping of disease cohorts.",
    )
    parser.add_argument(
        "--mode", choices=["disease_specific", "pooled"], default="disease_specific"
    )
    parser.add_argument(
        "--evaluation_strategy",
        choices=["prospective_holdout", "cross_validation"],
        default="prospective_holdout",
    )
    parser.add_argument(
        "--outcome_name",
        action="append",
        default=None,
        help="Optional repeated filter for one or more outcomes.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=None,
        help="Optional repeated filter for one or more disease cohorts.",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_results = run_comparison_grid(
        outcomes=load_object(args.outcomes),
        rkkp=load_object(args.rkkp) if args.rkkp else None,
        ehr_features=load_object(args.ehr_features),
        embeddings_base=load_object(args.embeddings_base),
        embeddings_dapt=load_object(args.embeddings_dapt),
        embeddings_opera=load_object(args.embeddings_opera),
        disease_cohorts=load_object(args.disease_cohorts),
        outcome_names=args.outcome_name,
        cohorts=args.cohort,
        mode=args.mode,
        evaluation_strategy=args.evaluation_strategy,
        seed=args.seed,
        n_splits=args.n_splits,
        n_bootstrap=args.n_bootstrap,
        show_progress=True,
    )

    with (output_dir / "comparison_results.json").open("w", encoding="utf-8") as handle:
        json.dump(grid_results, handle, indent=2, default=str)

    flat = flatten_grid_results(grid_results)
    if not flat.empty:
        flat.to_csv(output_dir / "comparison_results.csv", index=False)

    LOGGER.info(
        "Wrote Task 1 comparison artifacts to %s",
        output_dir,
    )


if __name__ == "__main__":
    main()
