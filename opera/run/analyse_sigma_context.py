"""Join OPERA sigmas with event/prevalence/effective-pair metadata."""

import argparse
from pathlib import Path

import pandas as pd

from opera.evaluation.sigma_context import (
    build_sigma_context_table,
    residualize_log_sigma,
)


def _read_table(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextualize learned OPERA sigmas")
    parser.add_argument(
        "--sigmas", required=True, help="CSV/parquet with outcome and sigma columns"
    )
    parser.add_argument(
        "--metadata", default=None, help="Optional outcome metadata table"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--outcome_col", default="outcome")
    parser.add_argument("--sigma_col", default="sigma")
    args = parser.parse_args()

    sigmas = _read_table(args.sigmas)
    metadata = _read_table(args.metadata) if args.metadata else None
    context = build_sigma_context_table(
        sigmas,
        metadata,
        outcome_col=args.outcome_col,
        sigma_col=args.sigma_col,
    )
    context = residualize_log_sigma(context, outcome_col=args.outcome_col)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(output, index=False)
    print(f"Wrote sigma context table to {output}")


if __name__ == "__main__":
    main()
