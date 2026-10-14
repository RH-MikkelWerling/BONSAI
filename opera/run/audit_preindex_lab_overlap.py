"""Read-only baseline availability and prior-abnormality audit for lab outcomes.

Prior abnormality is a broad analyte signal, not the same CTCAE grade or proof
that a post-index event is incident. Patient-level inputs stay on the server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

OUTCOMES = {
    "neutrophils": ("neutropenia_g2plus", "neutropenia_g3plus"),
    "ALT": ("ALT_increased_g2plus", "ALT_increased_g3plus"),
    "AST": ("AST_increased_g2plus", "AST_increased_g3plus"),
    "bilirubin": ("bilirubin_increased_g2plus", "bilirubin_increased_g3plus"),
    "creatinine": ("creatinine_increased_g2plus", "creatinine_increased_g3plus"),
    "eosinophils": ("eosinophilia",),
}
WINDOWS = (30, 90, 365, 730)
HORIZONS = (30, 365, None)


def _outcome(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[
        "subject_id", "split", "index_date", "outcome_date", "censor_date",
    ])
    if frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
        raise ValueError(f"Missing or repeated subjects in {path}")
    for name in ("index_date", "outcome_date", "censor_date"):
        frame[name] = pd.to_datetime(frame[name], errors="coerce")
    if frame["index_date"].isna().any() or frame["censor_date"].isna().any():
        raise ValueError(f"Invalid index/censor date in {path}")
    return frame


def _preindex_signals(labs: pd.DataFrame, scaffold: pd.DataFrame) -> pd.DataFrame:
    joined = labs.merge(scaffold[["subject_id", "index_date"]],
                        on="subject_id", how="inner", validate="many_to_one")
    age = (joined["index_date"] - joined["sample_date"]).dt.days
    joined = joined.loc[age.between(1, max(WINDOWS))].copy()
    joined["days_before_index"] = age.loc[joined.index]
    joined["valid"] = joined["value"].notna()
    has_ref = joined["ref_upper"].notna() & joined["ref_upper"].gt(0)
    is_neutrophil = joined["analyte"].eq("neutrophils")
    joined["evaluable"] = joined["valid"] & (is_neutrophil | has_ref)
    joined["abnormal"] = joined["evaluable"] & (
        (is_neutrophil & joined["value"].lt(1.0)) |
        (~is_neutrophil & joined["value"].gt(joined["ref_upper"]))
    )
    return joined


def audit(labs: pd.DataFrame, outcomes_dir: Path) -> pd.DataFrame:
    survival = _outcome(outcomes_dir / "overall_survival.parquet")
    needed = {"subject_id", "analyte", "sample_date", "value", "ref_upper"}
    if not needed.issubset(labs):
        raise ValueError(f"Lab snapshot missing {sorted(needed - set(labs))}")
    labs = labs.loc[labs["analyte"].isin(OUTCOMES)].copy()
    labs["sample_date"] = pd.to_datetime(labs["sample_date"], errors="coerce")
    if labs["sample_date"].isna().any():
        raise ValueError("Lab snapshot contains invalid sample dates")
    for name in ("value", "ref_upper"):
        labs[name] = pd.to_numeric(labs[name], errors="coerce")
    prior = _preindex_signals(labs, survival)
    rows = []
    for analyte, names in OUTCOMES.items():
        source = prior.loc[prior["analyte"].eq(analyte)]
        for window in WINDOWS:
            band = source.loc[source["days_before_index"].le(window) &
                              source["evaluable"]]
            any_abnormal = band.groupby("subject_id")["abnormal"].any()
            # Most recent day's median avoids row-order-dependent baselines.
            daily = band.groupby(["subject_id", "sample_date"], as_index=False)[
                ["value", "ref_upper"]
            ].median()
            latest = daily.sort_values(["subject_id", "sample_date"]).drop_duplicates(
                "subject_id", keep="last"
            )
            if analyte == "neutrophils":
                latest_abnormal = latest["value"].lt(1.0)
            else:
                latest_abnormal = latest["value"].gt(latest["ref_upper"])
            baseline = pd.DataFrame({
                "subject_id": latest["subject_id"].to_numpy(),
                "latest_abnormal": latest_abnormal.to_numpy(),
            }).set_index("subject_id")
            baseline["ever_abnormal"] = any_abnormal.reindex(baseline.index).fillna(False)
            for outcome in names:
                path = outcomes_dir / f"{outcome}.parquet"
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = _outcome(path).set_index("subject_id")
                frame = frame.join(baseline, how="left")
                frame["has_baseline"] = frame["latest_abnormal"].notna()
                days = (frame["outcome_date"] - frame["index_date"]).dt.days
                for horizon in HORIZONS:
                    event = days.notna() if horizon is None else days.le(horizon) & days.notna()
                    for split, group in frame.groupby("split"):
                        indices = group.index
                        current_event = event.reindex(indices)
                        available = group["has_baseline"]
                        rows.append({
                            "analyte": analyte, "outcome": outcome,
                            "split": split, "baseline_window_days": window,
                            "event_horizon_days": "all" if horizon is None else horizon,
                            "n_outcome_patients": len(group),
                            "n_with_evaluable_preindex_lab": int(available.sum()),
                            "n_events": int(current_event.sum()),
                            "n_events_with_evaluable_preindex_lab": int((current_event & available).sum()),
                            "n_events_with_any_prior_abnormality": int((current_event & group["ever_abnormal"].fillna(False)).sum()),
                            "n_events_with_latest_prior_abnormality": int((current_event & group["latest_abnormal"].fillna(False)).sum()),
                        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", type=Path, required=True,
                        help="candidate_labs_v1_INTERNAL.parquet from the secure lab snapshot")
    parser.add_argument("--outcomes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(pd.read_parquet(args.labs), args.outcomes_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(result), "output": str(args.output),
                      "note": "Prior abnormality is analyte-level, not the same CTCAE grade"}))


if __name__ == "__main__":
    main()
