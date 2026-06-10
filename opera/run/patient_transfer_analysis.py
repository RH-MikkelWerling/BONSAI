"""Build patient-level transfer tables for OPERA model contrasts."""

import argparse
from pathlib import Path


from opera.evaluation.patient_transfer import (
    build_patient_transfer_table,
    summarize_beneficiary_profile,
)


def _feature_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-level OPERA transfer analysis"
    )
    parser.add_argument("--baseline_predictions", required=True)
    parser.add_argument("--comparator_predictions", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--contrast_name",
        default="joint_opera_minus_per_cohort_opera",
        help="Name stored in the patient-level gain table.",
    )
    parser.add_argument(
        "--feature_cols",
        default="",
        help="Comma-separated EHR or harmonized registry columns for beneficiary profiling.",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top_fraction", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_patient_transfer_table(
        args.baseline_predictions,
        args.comparator_predictions,
        args.embeddings,
        args.metadata,
        contrast_name=args.contrast_name,
        feature_cols=_feature_list(args.feature_cols),
        k=args.k,
    )
    table.to_csv(output_dir / "patient_transfer.csv", index=False)
    summary = summarize_beneficiary_profile(
        table,
        top_fraction=args.top_fraction,
    )
    summary.to_csv(output_dir / "beneficiary_profile.csv", index=False)

    if not table.empty:
        print(
            f"Wrote {len(table)} patient rows to {output_dir / 'patient_transfer.csv'} "
            f"(mean brier_gain={table['brier_gain'].mean():.4f})."
        )
    else:
        print("No overlapping patient predictions were found.")


if __name__ == "__main__":
    main()
