"""Prepare one or more cacheable tabular feature matrices."""

from __future__ import annotations

import argparse

from opera.functional.tabular_features import SUPPORTED_PROFILES, prepare_feature_matrix


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="One or more source pickle/CSV/parquet feature matrices.",
    )
    parser.add_argument("--population", required=True)
    parser.add_argument("--demographics",
                        help="subject_id, sex and birth_date (the Table 1 demographics "
                             "export), when the membership file lacks them.")
    parser.add_argument(
        "--feature_profile",
        nargs="+",
        choices=sorted(SUPPORTED_PROFILES),
        default=["sequence_matched"],
        help="sequence_matched (primary) and/or registry_augmented (keeps RKKP "
        "predictors; tabular reference arm only).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for source in args.features:
        for profile in args.feature_profile:
            target, manifest = prepare_feature_matrix(
                source,
                profile=profile,
                population_path=args.population,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                demographics_path=args.demographics,
            )
            print(
                f"prepared/reused ({profile}): {source} -> {target} "
                f"({manifest['prepared_shape'][0]} rows, "
                f"{manifest['prepared_shape'][1] - 1} predictors)"
            )


if __name__ == "__main__":
    main()
