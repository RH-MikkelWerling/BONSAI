"""Dry-run builder for OPERA adaptation breadth ladder experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opera.evaluation.cohort_scope import build_ladder_plan_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adaptation breadth ladder configs without training."
    )
    parser.add_argument(
        "--manifest",
        default="opera/configs/manifests/adaptation_breadth_ladder.yaml",
        help="YAML manifest defining target, cohort set, and neighbours.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. If omitted, the plan is printed.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Assert scope consistency and emit the plan; no training is launched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_ladder_plan_from_manifest(args.manifest)
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if not args.dry_run:
        print("Plan emitted only; launch training explicitly from the generated overrides.")


if __name__ == "__main__":
    main()
