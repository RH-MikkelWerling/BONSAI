"""Validate OPERA temporal split contract across stage inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opera.evaluation.split_contract import (
    DEFAULT_SPLIT_CONTRACT_PATH,
    validate_cross_stage_split_contract,
)


def _split_path_arg(values: list[str] | None) -> dict[str, str]:
    paths: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"Expected split=path, got {value!r}.")
        split, path = value.split("=", 1)
        paths[split] = path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate labels, subject_data, DAPT inputs, and embedding stores."
    )
    parser.add_argument("--contract", default=str(DEFAULT_SPLIT_CONTRACT_PATH))
    parser.add_argument("--outcome", action="append", required=True)
    parser.add_argument("--subject_data", action="append", default=[])
    parser.add_argument("--dapt_subject_data", action="append", default=[])
    parser.add_argument("--embedding_store", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_cross_stage_split_contract(
        outcome_paths=args.outcome,
        subject_data_paths=_split_path_arg(args.subject_data),
        dapt_subject_data_paths=_split_path_arg(args.dapt_subject_data),
        embedding_store_paths=args.embedding_store,
        contract_path=args.contract,
    )
    payload = json.dumps(report, indent=2, default=str)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    if args.fail_on_error and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
