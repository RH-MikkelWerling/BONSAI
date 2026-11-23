"""Calibration across cells, per model (reporting addition of 22 Nov 2026;
docs/EVALUATION_SUMMARY_AMENDMENT_20261122.md).

Median and interquartile range of the calibration slope (ideal 1) and observed/expected ratio
(ideal 1) over cells with at least ``--min-events`` test events, overall and by training
event-rate band when ``primary_event_rate_train`` is in the file. Reads the downloaded
``calibration_summary`` only; runs locally.

  python scripts/summarise_cell_spread.py --calibration <calibration_summary.csv> --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RATE_BANDS, RATE_LABELS = [0, 0.05, 0.20, 1.0], ["<5%", "5-20%", ">=20%"]


def calibration_spread(calibration, min_events=100):
    cal = calibration[(calibration.status == "estimated") & (calibration.n_events >= min_events)].copy()
    cal["log_oe"] = np.log(cal.observed_expected_ratio.where(cal.observed_expected_ratio > 0))
    keys = [k for k in ("split", "seed", "model", "horizon_days") if k in cal]

    def spread(group):
        slope, log_oe = group.calibration_slope.dropna(), group.log_oe.dropna()
        return pd.Series({
            "n_cells": len(group),
            "slope_median": slope.median(), "slope_q25": slope.quantile(.25), "slope_q75": slope.quantile(.75),
            # O/E summarised on the log scale, where over- and under-prediction are symmetric.
            "oe_median": float(np.exp(log_oe.median())),
            "oe_q25": float(np.exp(log_oe.quantile(.25))), "oe_q75": float(np.exp(log_oe.quantile(.75))),
        })

    parts = [cal.groupby(keys)[["calibration_slope", "log_oe"]].apply(spread).reset_index()
             .assign(rate_band="all")]
    if "primary_event_rate_train" in cal:
        cal["rate_band"] = pd.cut(cal.primary_event_rate_train, RATE_BANDS, labels=RATE_LABELS,
                                  include_lowest=True).astype(str)
        parts.append(cal.groupby(keys + ["rate_band"])[["calibration_slope", "log_oe"]]
                     .apply(spread).reset_index())
    return pd.concat(parts, ignore_index=True).assign(min_events=min_events)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    calibration_spread(pd.read_csv(args.calibration), args.min_events).to_csv(
        args.out / "calibration_spread.csv", index=False)
    print("wrote", args.out / "calibration_spread.csv")


if __name__ == "__main__":
    main()
