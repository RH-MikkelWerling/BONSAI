"""What each part of the record adds: AUC with restricted inputs vs the full input.

Compares ``direct_cr_predictions.parquet`` from the full evaluation with
evaluations of the same checkpoint on restricted inputs
(``evaluate_direct_competing_risk --content-reference``, ``--hide-prefixes``,
``--keep-last-days``). For every outcome, the held-out AUC at ``horizon``
(pooled over diseases) is compared with paired patient-bootstrap intervals,
using the same resamples for every outcome and restriction so that macro
averages and domain averages have valid intervals. The model is not
retrained: this measures what the fitted model relies on.

    python -m opera.run.compare_input_restrictions \\
        --full runs/eval_full \\
        --restricted "Timeline only=runs/eval_timeline" \\
        --restricted "Without laboratory results=runs/eval_no_labs" \\
        --restricted "Last year only=runs/eval_last365" \\
        --output-dir analysis/input_restrictions
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opera.run.compare_approved_neural_tuning_predictions import fast_cif_auc

KEY = ["subject_id", "cohort", "outcome"]


def read_predictions(path):
    path = Path(path)
    if path.is_dir():
        path = path / "direct_cr_predictions.parquet"
    return pd.read_parquet(path)


def _auc(frame, column, horizon):
    # Re-estimate censoring for each resample, as in the original evaluator.
    # Weighted ranks avoid materialising every case/control pair or computing
    # the unused Brier score. An empty resample has no estimable AUC.
    if frame.empty:
        return float("nan")
    return fast_cif_auc(
        frame.time_days.to_numpy(float), frame.event.to_numpy(int),
        frame[column].to_numpy(float), horizon)[0]


def compare(full, restricted, *, horizon=365, draws=200, seed=2026, domains=None,
            progress=None):
    """Per-outcome and macro AUC differences (restricted minus full)."""
    column = f"cif_{int(horizon)}d"
    patients = full.subject_id.unique()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)),
                             size=draws)
    position = pd.Series(np.arange(len(patients)), index=patients)
    rows, replicate_rows = [], []
    for restriction_number, (name, frame) in enumerate(restricted.items(), 1):
        merged = full[KEY + ["time_days", "event", column]].merge(
            frame[KEY + [column]], on=KEY, suffixes=("_full", "_restricted"),
            validate="one_to_one")
        if len(merged) != len(full):
            raise ValueError(f"{name}: restricted predictions do not cover the full rows")
        groups_by_outcome = merged.groupby("outcome")
        for outcome_number, (outcome, part) in enumerate(groups_by_outcome, 1):
            label = (f"Restriction {restriction_number}/{len(restricted)} ({name}), "
                     f"outcome {outcome_number}/{len(groups_by_outcome)} ({outcome})")
            if progress:
                progress(f"{label}: starting {draws} paired bootstrap draws")
            full_auc = _auc(part, f"{column}_full", horizon)
            restricted_auc = _auc(part, f"{column}_restricted", horizon)
            index = position[part.subject_id].to_numpy()
            deltas = []
            for r in range(draws):
                repeat = counts[r, index]
                sample = part.loc[np.repeat(part.index.to_numpy(), repeat)]
                deltas.append(_auc(sample, f"{column}_restricted", horizon)
                              - _auc(sample, f"{column}_full", horizon))
                if progress and ((r + 1) % max(1, draws // 4) == 0 or r + 1 == draws):
                    progress(f"{label}: {r + 1}/{draws} draws complete")
            deltas = np.asarray(deltas, float)
            replicate_rows.append(pd.Series(deltas, name=(name, outcome)))
            rows.append({"restriction": name, "outcome": outcome, "auc_full": full_auc,
                         "auc_restricted": restricted_auc,
                         "delta": restricted_auc - full_auc,
                         "lower": np.nanquantile(deltas, 0.025),
                         "upper": np.nanquantile(deltas, 0.975),
                         "n_rows": len(part)})
    table = pd.DataFrame(rows)
    replicates = pd.DataFrame(replicate_rows)
    summary = []
    groups = {"All outcomes": None}
    if domains:
        groups.update({d: [o for o, g in domains.items() if g == d]
                       for d in dict.fromkeys(domains.values())})
    for name in restricted:
        for label, members in groups.items():
            part = table.loc[(table.restriction == name)
                             & (table.outcome.isin(members) if members else True)]
            if part.empty:
                continue
            reps = replicates.loc[[(name, o) for o in part.outcome]].mean(0)
            summary.append({"restriction": name, "group": label,
                            "n_outcomes": len(part), "mean_delta": part.delta.mean(),
                            "lower": np.nanquantile(reps, 0.025),
                            "upper": np.nanquantile(reps, 0.975),
                            "mean_auc_full": part.auc_full.mean(),
                            "mean_auc_restricted": part.auc_restricted.mean()})
    return table, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--restricted", action="append", required=True,
                        help="label=path to a restricted evaluation (repeatable)")
    parser.add_argument("--horizon", type=float, default=365)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from opera.visualization.paper_taxonomy import outcome_endpoint_groups

    restricted = {}
    for item in args.restricted:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError("--restricted takes label=path")
        restricted[label] = read_predictions(path)
    table, summary = compare(read_predictions(args.full), restricted, horizon=args.horizon,
                             draws=args.draws, seed=args.seed,
                             domains=outcome_endpoint_groups(),
                             progress=lambda message: print(
                                 f"{datetime.now().isoformat(timespec='seconds')} {message}",
                                 flush=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "input_restriction_auc.csv", index=False)
    summary.to_csv(args.output_dir / "input_restriction_summary.csv", index=False)
    print(json.dumps(summary.loc[summary.group == "All outcomes"].to_dict("records"),
                     indent=2, default=float))


if __name__ == "__main__":
    main()
