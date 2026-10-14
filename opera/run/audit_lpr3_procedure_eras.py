"""Compare ungated procedure outcomes before/after the LPR2 key-join repair.

Outputs aggregate CSVs and a diagnostic PDF. This is an ascertainment check,
not a causal estimate or a reason by itself to move the coverage gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = (
    "RBC_transfusion", "platelet_transfusion", "any_transfusion", "antibiotics",
    "respiratory_support", "dialysis", "emergency_stay",
)
LPR3_START = pd.Timestamp("2019-03-05")
HORIZONS = (30, 365)
TIME_BINS = (-1, 7, 30, 90, 365, np.inf)
TIME_LABELS = ("day 1–7", "day 8–30", "day 31–90", "day 91–365", "day 366+")


def load_outcome(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[
        "subject_id", "index_date", "outcome_date", "censor_date",
    ])
    if frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
        raise ValueError(f"Missing or duplicate subjects in {path}")
    for column in ("index_date", "outcome_date", "censor_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if frame["index_date"].isna().any() or frame["censor_date"].isna().any():
        raise ValueError(f"Missing index/censor date in {path}")
    if (frame["censor_date"] < frame["index_date"]).any():
        raise ValueError(f"Censor before index in {path}")
    return frame.set_index("subject_id").sort_index()


def compare_outcome(old: pd.DataFrame, new: pd.DataFrame, outcome: str):
    if not old.index.equals(new.index):
        raise ValueError(f"Subject population changed for {outcome}; use ungated exports")
    for column in ("index_date", "censor_date"):
        if not old[column].equals(new[column]):
            raise ValueError(f"{column} changed for {outcome}; not a join-only comparison")
    joined = new.rename(columns={"outcome_date": "new_event"}).copy()
    joined["old_event"] = old["outcome_date"]
    joined["index_year"] = joined["index_date"].dt.year
    joined["index_era"] = np.where(joined["index_date"] < LPR3_START,
                                    "pre-LPR3 index", "post-LPR3 index")
    joined["outcome"] = outcome
    joined["followup_days"] = (joined["censor_date"] - joined["index_date"]).dt.days
    for version in ("old", "new"):
        joined[f"{version}_days"] = (
            joined[f"{version}_event"] - joined["index_date"]
        ).dt.days
        invalid = joined[f"{version}_days"].notna() & (
            (joined[f"{version}_days"] <= 0) |
            (joined[f"{version}_event"] > joined["censor_date"])
        )
        if invalid.any():
            raise ValueError(f"Invalid {version} event dates for {outcome}")
    return joined


def summarize(frame: pd.DataFrame):
    rates = []
    timing = []
    changes = []
    for (outcome, year, era), group in frame.groupby(
        ["outcome", "index_year", "index_era"], observed=True
    ):
        base = {"outcome": outcome, "index_year": int(year), "index_era": era,
                "n_patients": len(group)}
        old, new = group["old_event"], group["new_event"]
        changes.append({**base,
                        "n_new_event": int((old.isna() & new.notna()).sum()),
                        "n_earlier_event": int((old.notna() & new.notna() & (new < old)).sum()),
                        "n_later_event": int((old.notna() & new.notna() & (new > old)).sum()),
                        "n_lost_event": int((old.notna() & new.isna()).sum())})
        for version in ("old", "new"):
            days = group[f"{version}_days"]
            for horizon in HORIZONS:
                event = days.notna() & days.le(horizon)
                assessable = group["followup_days"].ge(horizon) | event
                rates.append({**base, "version": version, "horizon_days": horizon,
                              "n_full_horizon_followup": int(group["followup_days"].ge(horizon).sum()),
                              "n_assessable": int(assessable.sum()),
                              "n_events": int(event.sum()),
                              "event_fraction_assessable": float(event.sum() / assessable.sum())
                              if assessable.sum() else np.nan})
            bins = pd.cut(days.dropna(), TIME_BINS, labels=TIME_LABELS)
            counts = bins.value_counts()
            for label in TIME_LABELS:
                timing.append({**base, "version": version, "time_bin": label,
                               "n_events": int(counts.get(label, 0)),
                               "n_any_events": int(days.notna().sum())})
    return pd.DataFrame(rates), pd.DataFrame(timing), pd.DataFrame(changes)


def summarize_calendar_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version in ("old", "new"):
        event = frame.loc[frame[f"{version}_event"].notna(),
                          ["outcome", f"{version}_event"]].copy()
        event["event_year"] = event[f"{version}_event"].dt.year
        counts = event.groupby(["outcome", "event_year"]).size()
        rows.extend({"outcome": outcome, "event_year": int(year),
                     "version": version, "n_first_events": int(count)}
                    for (outcome, year), count in counts.items())
    return pd.DataFrame(rows)


def plot_rates(rates: pd.DataFrame, output: Path, max_full_index_year: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["TeX Gyre Pagella", "Palatino Linotype", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    # In recent calendar years, only early events can satisfy the 365-day
    # denominator. A complete-calendar-year restriction avoids 100% artifacts.
    subset = rates.loc[rates["horizon_days"].eq(365) &
                       rates["index_year"].le(max_full_index_year)]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for ax, outcome in zip(axes.flat, OUTCOMES):
        data = subset.loc[subset["outcome"].eq(outcome)]
        for version, color in (("old", "#9a9fa9"), ("new", "#263b80")):
            part = data.loc[data["version"].eq(version)].sort_values("index_year")
            ax.plot(part["index_year"], 100 * part["event_fraction_assessable"],
                    marker="o", markersize=3, linewidth=1.5, color=color,
                    label="Before fix" if version == "old" else "After fix")
        ax.axvline(2019, color="#b45a40", linestyle="--", linewidth=0.9)
        ax.set_title(outcome.replace("_", " "), loc="left", fontsize=10)
        ax.set_ylabel("365-day events / assessable (%)")
        ax.grid(axis="y", color="#e9e9ee", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("Index year")
    fig.suptitle("Procedure outcomes across the LPR2–LPR3 transition", fontsize=15)
    fig.text(0.5, 0.02,
             f"Dashed line: 2019 transition. Full calendar index years through {max_full_index_year} only; descriptive fractions, not competing-risk estimates.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output)
    plt.close(fig)


def plot_event_timing(timing: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["TeX Gyre Pagella", "Palatino Linotype", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    grouped = timing.groupby(["outcome", "index_era", "version", "time_bin"],
                             as_index=False)["n_events"].sum()
    colors = ("#263b80", "#5477af", "#8a9fc4", "#b79469", "#ddd0b7")
    fig, axes = plt.subplots(len(OUTCOMES), 2, figsize=(11, 14), sharex=True)
    for row, outcome in enumerate(OUTCOMES):
        for col, era in enumerate(("pre-LPR3 index", "post-LPR3 index")):
            ax = axes[row, col]
            data = grouped.loc[grouped["outcome"].eq(outcome) &
                               grouped["index_era"].eq(era)]
            for y, version in enumerate(("old", "new")):
                counts = data.loc[data["version"].eq(version)].set_index("time_bin")["n_events"]
                total = counts.sum()
                left = 0.0
                for label, color in zip(TIME_LABELS, colors):
                    width = 100 * counts.get(label, 0) / total if total else 0
                    ax.barh(y, width, left=left, height=0.55, color=color,
                            label=label if row == 0 and col == 0 else None)
                    left += width
                ax.text(101, y, f"n={total:,}", va="center", fontsize=7)
            ax.set_yticks((0, 1), ("Before", "After"))
            ax.invert_yaxis()
            ax.set_xlim(0, 120)
            ax.set_xticks((0, 25, 50, 75, 100))
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(outcome.replace("_", " "), rotation=0,
                              ha="right", va="center", labelpad=10, fontsize=9)
            if row == 0:
                ax.set_title(era, fontsize=11)
            if row == len(OUTCOMES) - 1:
                ax.set_xlabel("Share of recorded first events (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Time to first recorded procedure outcome", fontsize=15)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.old_dir.resolve() == args.new_dir.resolve():
        raise ValueError("Old and new outcome directories must differ")
    frames = []
    file_checks = []
    for outcome in OUTCOMES:
        old_path = args.old_dir / f"{outcome}.parquet"
        new_path = args.new_dir / f"{outcome}.parquet"
        if not old_path.is_file() or not new_path.is_file():
            raise FileNotFoundError(
                f"Missing parquet for {outcome}: old={old_path} "
                f"(exists={old_path.is_file()}), new={new_path} "
                f"(exists={new_path.is_file()})"
            )
        if old_path.resolve() == new_path.resolve():
            raise ValueError(f"Old and new files resolve to the same parquet: {old_path}")
        old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
        new_hash = hashlib.sha256(new_path.read_bytes()).hexdigest()
        file_checks.append({"outcome": outcome, "old_path": str(old_path.resolve()),
                            "new_path": str(new_path.resolve()),
                            "same_sha256": old_hash == new_hash,
                            "old_sha256": old_hash, "new_sha256": new_hash})
        frames.append(compare_outcome(load_outcome(old_path), load_outcome(new_path), outcome))
    full = pd.concat(frames, ignore_index=True)
    rates, timing, changes = summarize(full)
    calendar = summarize_calendar_events(full)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(file_checks).to_csv(args.output_dir / "input_file_checks.csv", index=False)
    rates.to_csv(args.output_dir / "event_rates_by_index_year.csv", index=False)
    timing.to_csv(args.output_dir / "time_to_event_bins_by_index_year.csv", index=False)
    changes.to_csv(args.output_dir / "changed_labels_by_index_year.csv", index=False)
    calendar.to_csv(args.output_dir / "first_events_by_calendar_year.csv", index=False)
    era_rates = rates.groupby(["outcome", "index_era", "version", "horizon_days"],
                             as_index=False)[["n_patients", "n_full_horizon_followup",
                                              "n_assessable", "n_events"]].sum()
    era_rates["event_fraction_assessable"] = (
        era_rates["n_events"] / era_rates["n_assessable"].replace(0, np.nan)
    )
    era_rates.to_csv(args.output_dir / "event_rates_by_index_era.csv", index=False)
    timing.groupby(["outcome", "index_era", "version", "time_bin"],
                   as_index=False)[["n_events", "n_any_events"]].sum().to_csv(
        args.output_dir / "time_to_event_bins_by_index_era.csv", index=False
    )
    max_index_with_365_days = full["censor_date"].max() - pd.Timedelta(days=365)
    max_full_index_year = max_index_with_365_days.year
    if (max_index_with_365_days.month, max_index_with_365_days.day) != (12, 31):
        max_full_index_year -= 1
    complete_year_rates = rates.loc[rates["index_year"].le(max_full_index_year)]
    complete_era_rates = complete_year_rates.groupby(
        ["outcome", "index_era", "version", "horizon_days"], as_index=False
    )[["n_patients", "n_full_horizon_followup", "n_assessable", "n_events"]].sum()
    complete_era_rates["event_fraction_assessable"] = (
        complete_era_rates["n_events"] /
        complete_era_rates["n_assessable"].replace(0, np.nan)
    )
    complete_era_rates.to_csv(
        args.output_dir / "event_rates_by_index_era_complete_years.csv", index=False
    )
    plot_rates(rates, args.output_dir / "lpr3_event_rates_by_index_year.pdf",
               max_full_index_year)
    plot_event_timing(timing, args.output_dir / "lpr3_time_to_event_by_index_era.pdf")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "outcomes": len(OUTCOMES),
                      "new_events": int(changes["n_new_event"].sum()),
                      "earlier_events": int(changes["n_earlier_event"].sum()),
                      "lost_events": int(changes["n_lost_event"].sum()),
                      "max_full_index_year_in_365d_plot": max_full_index_year,
                      "identical_file_hashes": int(sum(row["same_sha256"] for row in file_checks)),
                      "input_file_checks": str(args.output_dir / "input_file_checks.csv")}))


if __name__ == "__main__":
    main()
