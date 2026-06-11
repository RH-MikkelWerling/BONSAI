"""Build a cohort-flow artifact from configured eligibility sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opera.config_contracts import ConfigValidationError, load_sweep_config
from opera.evaluation.cohort_flow import (
    eligibility_file_path,
    load_eligibility_frame,
    summarize_eligibility_frame,
)


def build_cohort_flow(config_path: str) -> pd.DataFrame:
    """Load all configured sidecars and return one tidy cohort-flow table."""
    config = load_sweep_config(config_path)
    rows: list[pd.DataFrame] = []
    missing_config: list[str] = []

    for cohort_name, cohort in config.cohorts.items():
        for outcome_name, outcome in config.outcomes.items():
            outcome_cfg = outcome.to_mapping()
            path = eligibility_file_path(
                cohort.data_dir,
                cohort_name,
                outcome_name,
                outcome_cfg,
            )
            if path is None:
                missing_config.append(f"{cohort_name}/{outcome_name}")
                continue
            if not path.exists():
                raise FileNotFoundError(
                    f"Eligibility file for {cohort_name}/{outcome_name} "
                    f"does not exist: {path}"
                )
            frame = load_eligibility_frame(path)
            rows.append(summarize_eligibility_frame(frame, cohort_name, outcome_name))

    if missing_config:
        raise ValueError(
            "Missing eligibility_file for configured cells: "
            + ", ".join(missing_config)
        )
    if not rows:
        raise ValueError("No configured eligibility sidecars were found.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize outcome-specific eligibility into cohort-flow counts."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        cohort_flow = build_cohort_flow(args.config)
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort_flow.to_csv(output, index=False)
    print(f"Cohort flow: {output}")


if __name__ == "__main__":
    main()
