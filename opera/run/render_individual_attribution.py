"""Per-patient explanations from the cached attributions (INTERNAL).

For the check outcomes explained in absolute risk, writes inside the secure
environment, never for export:

``explanations_INTERNAL.parquet`` (``--all``)
    for every explained patient and outcome, the episodes of care that most
    raise and lower the predicted risk, with timing, a clinical description and
    the contribution in percentage points: the format a clinical tool would
    show beside each prediction.
``patient_<id>_INTERNAL.png`` and ``.csv`` (``--subject-id``)
    one patient's history read for each outcome: the episodes of care that move
    each predicted risk, at their time before treatment.

Patient-outcome pairs that failed the completeness check are left out.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from opera.run.explain_direct_competing_risk import parse_column, sha256
from opera.run.summarise_attributions import read_tokens
from opera.visualization.style import save_fig


def _passed(rows, tolerance):
    if "completeness_ok" in rows:
        return rows.completeness_ok.astype(bool)
    return rows.completeness_gap.abs() <= tolerance


def main():
    from opera.interpretability.attribution_patterns import (
        episode_contributions, explanation_table)
    from opera.interpretability.code_meaning import CodeDictionary
    from opera.visualization.attribution_overview import figure_patient_tracks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--subject-id", help="Render this patient's explanation figure.")
    parser.add_argument("--all", action="store_true",
                        help="Write the explanation table for every explained patient.")
    parser.add_argument("--horizon-days", type=float, default=365)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if not (args.all or args.subject_id):
        parser.error("Give --subject-id, --all or both")
    manifest = json.loads((args.input_dir / "attribution_run.json").read_text())
    for name, digest in manifest.get("output_sha256", {}).items():
        if sha256(args.input_dir / name) != digest:
            raise ValueError(f"Cached export changed: {name}")
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    settings = config.get("summary", {})
    labels = CodeDictionary.from_config(config)
    counts = json.loads((args.input_dir / "lab_bin_counts.json").read_text())
    gap = float(settings.get("episode_gap_days", 1.0))
    columns = {parse_column(c)[1]: c for c in manifest["columns"]
               if parse_column(c)[0] == "risk" and parse_column(c)[2] == args.horizon_days}
    if not columns:
        raise ValueError("No outcome was explained in absolute risk at this horizon")
    patients = pd.read_parquet(args.input_dir / "patient_attributions_INTERNAL.parquet")
    patients = patients.loc[_passed(patients, float(manifest.get("tolerance", 0.05)))
                            & patients.column.isin(columns.values())]
    tokens = read_tokens(args.input_dir, list(columns.values()))
    # Contributions of pairs that failed completeness are not used.
    for column in columns.values():
        accepted = patients.loc[patients.column == column, "subject_id"]
        tokens.loc[~tokens.subject_id.isin(accepted), column] = float("nan")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    written = []
    if args.all:
        table = explanation_table(tokens, patients, columns=columns, gap_days=gap,
                                  labels=labels, counts=counts,
                                  n=int(settings.get("explanation_episodes", 3)))
        table.to_parquet(args.output_dir / "explanations_INTERNAL.parquet", index=False)
        written.append("explanations_INTERNAL.parquet")
    if args.subject_id:
        subject = patients.loc[patients.subject_id.astype(str) == args.subject_id, "subject_id"]
        if subject.empty:
            raise ValueError("Patient not in the explained sample (or failed completeness)")
        subject = subject.iloc[0]
        mine = tokens.loc[tokens.subject_id == subject]
        episodes = episode_contributions(mine, patients, subject, columns=columns, gap_days=gap,
                                         labels=labels, counts=counts)
        fig = figure_patient_tracks(episodes, scale="risk")
        save_fig(fig, str(args.output_dir / f"patient_{args.subject_id}_INTERNAL.png"))
        episodes.to_csv(args.output_dir / f"patient_{args.subject_id}_INTERNAL.csv", index=False)
        written.append(f"patient_{args.subject_id}")
    (args.output_dir / "provenance_INTERNAL.json").write_text(json.dumps({
        "outputs": written, "horizon_days": args.horizon_days, "outcomes": list(columns),
        "attribution_manifest_sha256": sha256(args.input_dir / "attribution_run.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
