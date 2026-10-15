"""Prepare one or more cacheable tabular feature matrices."""

from __future__ import annotations

import argparse

from opera.functional.tabular_features import (
    augment_with_registry_variables,
    prepare_feature_matrix,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="One or more source pickle/CSV/parquet feature matrices.",
    )
    parser.add_argument("--population", required=True)
    parser.add_argument(
        "--feature_profile",
        choices=["sequence_matched"],
        default="sequence_matched",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--registry-variables",
        help="Optional table of quality-registry variables at diagnosis; writes "
        "a second, registry-augmented matrix for the tabular reference arm.",
    )
    parser.add_argument("--registry-output")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.registry_variables and (len(args.features) != 1 or not args.registry_output):
        raise SystemExit("--registry-variables needs one --features source and --registry-output")
    for source in args.features:
        target, manifest = prepare_feature_matrix(
            source,
            profile=args.feature_profile,
            population_path=args.population,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        action = "prepared/reused"
        print(
            f"{action}: {source} -> {target} "
            f"({manifest['prepared_shape'][0]} rows, "
            f"{manifest['prepared_shape'][1] - 1} predictors)"
        )
        if args.registry_variables:
            augmented, extra = augment_with_registry_variables(
                target, args.registry_variables, args.registry_output)
            print(f"registry-augmented: {augmented} "
                  f"(+{len(extra['registry_columns'])} registry variables, "
                  f"{extra['patients_with_registry_record']} patients with a record)")


if __name__ == "__main__":
    main()
