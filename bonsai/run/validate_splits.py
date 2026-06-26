"""Validate prospective outcome split boundaries and subject isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bonsai.functional.outcomes import resolve_split_contract, validate_split_integrity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate subject overlap and prospective split boundaries."
    )
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--train_end", default=None)
    parser.add_argument("--val_start", default=None)
    parser.add_argument("--val_end", default=None)
    parser.add_argument("--test_start", default=None)
    parser.add_argument("--test_end", default=None)
    parser.add_argument("--date_col", default="index_date")
    parser.add_argument("--train_key", default="train")
    parser.add_argument("--val_key", default="tuning")
    parser.add_argument("--test_key", default="held_out")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outcomes = pd.read_parquet(args.outcome)
    split_cfg = resolve_split_contract(
        {
            key: value
            for key, value in {
                "contract": args.contract,
                "train_end": args.train_end,
                "val_start": args.val_start,
                "val_end": args.val_end,
                "test_start": args.test_start,
                "test_end": args.test_end,
                "date_col": args.date_col,
                "train_key": args.train_key,
                "val_key": args.val_key,
                "test_key": args.test_key,
            }.items()
            if value is not None
        }
    )
    missing = [
        key
        for key in ("train_end", "val_start", "val_end", "test_start")
        if key not in split_cfg
    ]
    if missing:
        raise SystemExit(
            "Missing split boundary argument(s) and no complete --contract was "
            f"provided: {', '.join(missing)}"
        )
    report = validate_split_integrity(
        outcomes,
        **split_cfg,
    )
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    if args.fail_on_error and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
