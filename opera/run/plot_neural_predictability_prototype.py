"""Plot tuning CIF-AUC by horizon for a monitor-selected archived OPERA run."""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from opera.run.compare_approved_abspos_v2 import MONITOR, _epochs, _metrics_paths
from opera.visualization.outcome_names import outcome_display_name
from opera.visualization.style import PALETTE, despine, save_fig


HORIZON_COLORS = {
    30: "#343A88",
    60: "#6650A4",
    90: "#9460A5",
    180: "#BD648E",
    365: "#D47762",
    730: "#D9A52C",
}
DEFAULT_REGISTRY = Path("opera/configs/experiment_registry.yaml")
DEFAULT_GROUPS = Path("opera/configs/outcome_display_groups_v1.yaml")
DEFAULT_CANDIDATE_GROUPS = Path("opera/configs/outcome_display_groups_60_candidate_v1.yaml")


def _archive_csv(archive: tarfile.TarFile, run_token: str, suffix: str) -> pd.DataFrame:
    candidates = [
        name for name in archive.getnames()
        if run_token in name and name.endswith(suffix)
        and "/runs/" in name
    ]
    if suffix == "metrics.csv":
        versions = []
        for name in candidates:
            match = re.search(r"/version_(\d+)/metrics\.csv$", name)
            if match is None:
                raise ValueError(f"Unexpected Lightning metrics path: {name}")
            versions.append((int(match.group(1)), name))
        if not versions or len({version for version, _ in versions}) != len(versions):
            raise ValueError(f"Missing or duplicate metrics logger versions for {run_token!r}")
        candidates = [name for _, name in sorted(versions)]
    elif len(candidates) != 1:
        raise ValueError(f"Expected one {suffix} for {run_token!r}; found {len(candidates)}")
    frames = []
    for name in candidates:
        source = archive.extractfile(name)
        if source is None:
            raise ValueError(f"Could not read {name}")
        frames.append(pd.read_csv(io.BytesIO(source.read())))
    return pd.concat(frames, ignore_index=True)


def prepare_neural_data(
    bundle: Path,
    run_token: str = "weightinggroups__",
    horizons: tuple[int, ...] = (30, 60, 90, 180, 365, 730),
    registry_path: Path = DEFAULT_REGISTRY,
    groups_path: Path = DEFAULT_GROUPS,
) -> tuple[pd.DataFrame, int]:
    if not horizons or any(horizon not in HORIZON_COLORS for horizon in horizons):
        raise ValueError("Supported prototype horizons are 30, 60, 90, 180, 365, and 730 days")
    with tarfile.open(bundle, "r:gz") as archive:
        metrics = _archive_csv(archive, run_token, "metrics.csv")
        support = _archive_csv(archive, run_token, "competing_risk_interval_support.csv")
    monitor = "val/cr_cif_auc_endpoint_group_macro"
    if monitor not in metrics or not metrics[monitor].notna().any():
        raise ValueError("Bundle lacks the endpoint-group validation monitor")
    selected = metrics.loc[metrics[monitor].idxmax()]
    epoch = int(selected["epoch"])
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    groups = yaml.safe_load(groups_path.read_text(encoding="utf-8"))
    members = [outcome for outcomes in groups.values() for outcome in outcomes]
    if len(members) != len(set(members)) or set(members) != set(registry["outcomes"]):
        raise ValueError("Display-group membership does not match the registry")
    group_by_outcome = {
        outcome: group for group, outcomes in groups.items() for outcome in outcomes
    }
    rows = []
    for column, score in selected.items():
        match = re.fullmatch(r"val/cr_cif_auc/([^/]+)/([0-9]+)d", column)
        if not match or pd.isna(score):
            continue
        outcome, horizon_text = match.groups()
        horizon = int(horizon_text)
        if horizon not in horizons:
            continue
        if outcome not in group_by_outcome:
            raise ValueError(f"Unmapped neural outcome: {outcome}")
        n_events = support.loc[
            (support["outcome"] == outcome)
            & support["end_day"].notna()
            & (support["end_day"] <= horizon),
            "n_target",
        ].sum()
        rows.append({
            "outcome": outcome,
            "display_group": group_by_outcome[outcome],
            "horizon_days": horizon,
            "cif_auc": float(score),
            "training_primary_events": int(n_events),
        })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["outcome", "horizon_days"]).any():
        raise ValueError("Missing or duplicate per-outcome horizon metrics")
    if not frame["cif_auc"].between(0, 1).all():
        raise ValueError("CIF-AUC lies outside [0, 1]")
    return frame, epoch


def prepare_approved_run_data(
    run: Path,
    *,
    registry_path: Path,
    groups_path: Path,
    horizons: tuple[int, ...] = (30, 60, 90, 180, 365, 730),
) -> tuple[pd.DataFrame, int]:
    """Read a completed approved run directly, including resumed logger versions."""
    for name in ("training_complete.json", "endpoint_selection_inventory.json",
                 "competing_risk_interval_support.csv"):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    completion = json.loads((run / "training_complete.json").read_text(encoding="utf-8"))
    if completion.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run}")
    epochs = _epochs(_metrics_paths(run))
    best = max(epochs, key=lambda epoch: epochs[epoch][MONITOR])
    selected = epochs[best]
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    groups = yaml.safe_load(groups_path.read_text(encoding="utf-8"))
    expected = set(registry["outcomes"])
    members = [name for names in groups.values() for name in names]
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("Approved display groups must partition the registry exactly")
    by_outcome = {name: group for group, names in groups.items() for name in names}
    support = pd.read_csv(run / "competing_risk_interval_support.csv")
    inventory = pd.DataFrame(json.loads(
        (run / "endpoint_selection_inventory.json").read_text(encoding="utf-8")
    ))
    rows = []
    for name in sorted(expected):
        for horizon in horizons:
            score = selected.get(f"val/cr_cif_auc/{name}/{horizon}d")
            train_events = support.loc[
                support["outcome"].eq(name)
                & support["end_day"].notna()
                & support["end_day"].le(horizon), "n_target"
            ].sum()
            cell = inventory.loc[
                inventory["outcome"].eq(name)
                & inventory["horizon_days"].eq(horizon)
            ]
            if len(cell) != 1:
                raise ValueError(f"Missing or duplicate selection inventory cell: {name}, {horizon}d")
            rows.append({
                "outcome": name, "display_group": by_outcome[name],
                "horizon_days": horizon,
                "cif_auc": float(score) if score is not None and np.isfinite(score) else np.nan,
                "metric_available": score is not None and np.isfinite(score),
                "training_primary_events": int(train_events),
                "tuning_primary_events": int(cell.iloc[0]["n_cases"]),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["outcome", "horizon_days"]).any():
        raise ValueError("No unique outcome-horizon AUC cells found at selected epoch")
    if not frame["cif_auc"].notna().any():
        raise ValueError("No finite outcome-horizon AUC at selected epoch")
    return frame, best - 1


def filter_panel_candidates(
    frame: pd.DataFrame, decisions_path: Path, *, expected_count: int | None = None
) -> pd.DataFrame:
    """Display only selected endpoints; this does not retrain the archived model."""
    decisions = pd.read_csv(decisions_path)
    required = {"outcome", "disposition"}
    if not required.issubset(decisions.columns):
        raise ValueError(f"Panel decisions need columns {sorted(required)}")
    if decisions["outcome"].isna().any() or decisions["outcome"].duplicated().any():
        raise ValueError("Panel decisions contain missing or duplicate outcomes")
    candidates = set(decisions.loc[
        decisions["disposition"] == "primary_candidate", "outcome"
    ])
    if expected_count is not None and len(candidates) != expected_count:
        raise ValueError(f"Expected {expected_count} candidates; found {len(candidates)}")
    missing = candidates - set(frame["outcome"])
    if missing:
        raise ValueError(f"Candidate outcomes absent from neural metrics: {sorted(missing)}")
    result = frame.loc[frame["outcome"].isin(candidates)].copy()
    if result.empty:
        raise ValueError("No candidate outcomes remain in the plot")
    return result


def apply_display_groups(frame: pd.DataFrame, groups_path: Path) -> pd.DataFrame:
    """Remap the displayed outcomes, requiring an exact one-to-one partition."""
    groups = yaml.safe_load(groups_path.read_text(encoding="utf-8"))
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Display groups must be a nonempty mapping")
    members = [outcome for outcomes in groups.values() for outcome in outcomes]
    if len(members) != len(set(members)) or set(members) != set(frame["outcome"]):
        raise ValueError("Display groups must cover each displayed outcome exactly once")
    mapping = {outcome: group for group, outcomes in groups.items() for outcome in outcomes}
    result = frame.copy()
    result["display_group"] = result["outcome"].map(mapping)
    return result


def horizon_linear_trend(subset: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Descriptive OLS of AUC on log10 training-event count, at one horizon."""
    usable = subset.loc[
        subset["training_primary_events"].gt(0) & subset["cif_auc"].notna(),
        ["training_primary_events", "cif_auc"],
    ].copy()
    empty = pd.DataFrame(columns=["training_primary_events", "cif_auc"])
    if len(usable) < 3 or usable["training_primary_events"].nunique() < 2:
        return empty, {"n_outcomes": len(usable), "fit_available": False}
    x = np.log10(usable["training_primary_events"].to_numpy(dtype=float))
    y = usable["cif_auc"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1 - np.sum((y - fitted) ** 2) / total) if total > 0 else np.nan
    endpoints = np.quantile(x, [0.05, 0.95])
    points = pd.DataFrame({
        "training_primary_events": np.power(10.0, endpoints),
        "cif_auc": slope * endpoints + intercept,
    })
    return points, {
        "n_outcomes": len(usable), "fit_available": True,
        "auc_change_per_10x_events": float(slope), "r_squared": r_squared,
    }


def plot_neural_predictability(
    frame: pd.DataFrame, output: Path, *, run_label: str, selected_epoch: int,
    candidate_subset: bool = False, horizon_trends: bool = False,
) -> None:
    horizons = sorted(frame["horizon_days"].unique())
    fig, (ax, group_ax) = plt.subplots(
        1, 2, figsize=(12.0, 7.0), gridspec_kw={"width_ratios": [1.05, 1.1]}
    )
    for horizon in horizons:
        subset = frame.loc[
            (frame["horizon_days"] == horizon)
            & (frame["training_primary_events"] > 0)
            & frame["cif_auc"].notna()
        ]
        ax.scatter(
            subset["training_primary_events"], subset["cif_auc"],
            s=18, alpha=0.48, color=HORIZON_COLORS[horizon], linewidths=0,
            label=f"{horizon} days (n={len(subset)})", zorder=3,
        )
        if horizon_trends:
            trend, _ = horizon_linear_trend(subset)
            if len(trend):
                ax.plot(
                    trend["training_primary_events"], trend["cif_auc"],
                    color=HORIZON_COLORS[horizon], lw=1.7, alpha=0.95,
                    solid_capstyle="round", zorder=4,
                )
    ax.set_xscale("log")
    ax.set_xlabel("Primary events in training through horizon (log scale)")
    ax.set_ylabel("Validation CIF-AUC")
    ax.set_title("A  Outcome-level predictability", loc="left")
    # Early-horizon AUCs can genuinely fall below 0.45. A fixed lower limit
    # silently clipped valid outcomes (e.g. 30-day herpes zoster in the pilot).
    lower = min(0.45, max(0.0, float(frame["cif_auc"].min()) - 0.04))
    ax.set_ylim(lower, 1.01)
    despine(ax, "y")

    order = (
        frame.loc[frame["horizon_days"] == 365]
        .groupby("display_group")["cif_auc"].median().sort_values().index.tolist()
    )
    offsets = np.linspace(-0.29, 0.29, len(horizons))
    for horizon, offset in zip(horizons, offsets):
        subset = frame.loc[frame["horizon_days"] == horizon]
        for y, group in enumerate(order):
            values = subset.loc[
                subset["display_group"] == group, "cif_auc"
            ].dropna().to_numpy()
            if not len(values):
                continue
            median = float(np.median(values))
            if len(values) > 1:
                q25, q75 = np.quantile(values, [0.25, 0.75])
                group_ax.plot(
                    [q25, q75], [y + offset, y + offset],
                    color=HORIZON_COLORS[horizon], lw=1.25, alpha=0.38,
                    solid_capstyle="round", zorder=2,
                )
            group_ax.scatter(
                median, y + offset, s=24, marker="D",
                color=HORIZON_COLORS[horizon], zorder=3,
            )
    labels = order
    if candidate_subset:
        counts = frame.groupby("display_group")["outcome"].nunique()
        labels = [f"{group} (n={counts[group]})" for group in order]
    group_ax.set_yticks(range(len(order)), labels)
    group_ax.tick_params(axis="y", labelsize=7)
    group_ax.set_xlim(0.45, 1.01)
    group_ax.set_xlabel("Outcome CIF-AUC within group (diamond: median; bar: IQR)")
    group_ax.set_title("B  Clinical endpoint groups", loc="left")
    despine(group_ax, "x")
    horizon_handles = [
        Line2D([], [], linestyle="none", marker="D", markersize=5,
               color=HORIZON_COLORS[horizon],
               label=f"{horizon} days (n={sum(frame['horizon_days'].eq(horizon) & frame['cif_auc'].notna())})")
        for horizon in horizons
    ]
    group_ax.legend(
        handles=horizon_handles, title="Horizon (both panels)",
        loc="lower right", frameon=True, facecolor="white",
        edgecolor=PALETTE["grid"], framealpha=0.97,
        fontsize=7.2, title_fontsize=7.4,
    )
    subset_label = f"; {frame['outcome'].nunique()} candidate outcomes shown" if candidate_subset else ""
    fig.suptitle(
        f"OPERA validation predictability ({run_label}, epoch {selected_epoch + 1}{subset_label})",
        x=0.04, ha="left",
    )
    fig.text(
        0.04, 0.031 if horizon_trends else 0.015,
        "Censoring-aware CIF-AUC; not directly comparable with XGBoost binary AUROC. "
        "Bars show endpoint spread, not median uncertainty; unavailable horizon AUCs are omitted."
        + (" Full-panel model; display subset only." if candidate_subset else ""),
        fontsize=7.3, color=PALETTE["ink_secondary"],
    )
    if horizon_trends:
        fig.text(
            0.04, 0.012,
            "Lines: descriptive within-horizon OLS of AUC on log10 training events; "
            "slope and R-squared are exported separately.",
            fontsize=7.3, color=PALETTE["ink_secondary"],
        )
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.13, top=0.90, wspace=0.90)
    save_fig(fig, str(output))
    plt.close(fig)


def plot_neural_outcome_atlas(
    frame: pd.DataFrame, output: Path, *, groups: dict[str, list[str]],
    run_label: str, selected_epoch: int,
) -> None:
    """Full outcome-by-horizon tuning atlas, retaining unsupported cells."""
    outcomes = [name for members in groups.values() for name in members]
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("Atlas groups duplicate outcomes")
    horizons = sorted(frame["horizon_days"].unique())
    if not horizons:
        raise ValueError("No horizons for outcome atlas")
    values = np.full((len(outcomes), len(horizons)), np.nan)
    cases_365 = np.full(len(outcomes), np.nan)
    lookup = {name: i for i, name in enumerate(outcomes)}
    hlookup = {horizon: i for i, horizon in enumerate(horizons)}
    for row in frame.itertuples(index=False):
        if row.outcome not in lookup:
            raise ValueError(f"Atlas outcome absent from groups: {row.outcome}")
        values[lookup[row.outcome], hlookup[row.horizon_days]] = row.cif_auc
        if row.horizon_days == 365 and hasattr(row, "tuning_primary_events"):
            cases_365[lookup[row.outcome]] = row.tuning_primary_events
    fig = plt.figure(figsize=(12.8, max(11., .27 * len(outcomes) + 2.5)))
    grid = fig.add_gridspec(1, 3, width_ratios=[6, 1.15, 2.2],
                            left=.28, right=.97, top=.94, bottom=.07, wspace=.08)
    ax, count_ax, group_ax = [fig.add_subplot(grid[0, i]) for i in range(3)]
    cmap = plt.get_cmap("viridis").with_extremes(bad="#E9ECEF")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=Normalize(0, 1),
                      aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(horizons)), [f"{h}d" for h in horizons])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(outcomes)), [outcome_display_name(name) for name in outcomes],
                  fontsize=6.5)
    ax.tick_params(axis="both", length=0)
    for i in range(len(outcomes)):
        for j in range(len(horizons)):
            value = values[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "—",
                    ha="center", va="center", fontsize=6.1,
                    color="white" if np.isfinite(value) and value < .42 else "#141923")
    count_ax.set_xlim(0, 1)
    count_ax.set_ylim(len(outcomes) - .5, -.5)
    count_ax.set_axis_off()
    count_ax.text(.5, 1.01, "Events\n365d", transform=count_ax.transAxes,
                  ha="center", va="bottom", fontsize=8)
    for i, count in enumerate(cases_365):
        count_ax.text(.5, i, f"{int(count):,}" if np.isfinite(count) else "—",
                      ha="center", va="center", fontsize=6.5)
    group_ax.set_xlim(0, 1)
    group_ax.set_ylim(len(outcomes) - .5, -.5)
    group_ax.set_axis_off()
    group_ax.text(0, 1.01, "Clinical display group", transform=group_ax.transAxes,
                  ha="left", va="bottom", fontsize=8)
    start = 0
    for group, members in groups.items():
        end = start + len(members)
        for axis in (ax, count_ax, group_ax):
            axis.axhline(end - .5, color="#AAB4C0", lw=.65)
        group_ax.text(0, (start + end - 1) / 2, group,
                      va="center", fontsize=7.2, wrap=True)
        start = end
    fig.colorbar(image, ax=ax, location="bottom", shrink=.55, pad=.035,
                 label="Tuning CIF-AUC (0–1)")
    fig.suptitle(f"OPERA outcome predictability atlas — {run_label}, epoch {selected_epoch + 1}",
                 x=.28, ha="left", fontsize=14, fontweight="bold")
    fig.text(.28, .015, "Tuning only; selected checkpoint. Grey/— = unavailable AUC. "
             "365-day event counts are tuning support, not precision intervals.", fontsize=8)
    save_fig(fig, str(output))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--run", type=Path,
                        help="Completed approved run directory; reads resumed metrics directly.")
    parser.add_argument("--run-token", default="weightinggroups__")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="Exact outcome registry for the selected run; required for approved releases.")
    parser.add_argument("--panel-decisions", type=Path,
                        help="CSV from decide_outcome_panel.py; show primary_candidate outcomes only")
    parser.add_argument("--expect-candidates", type=int,
                        help="Fail unless panel-decisions contains exactly this many candidates")
    parser.add_argument("--display-groups", type=Path,
                        help="YAML mapping with exact membership of the displayed subset")
    parser.add_argument("--horizon-trends", action="store_true",
                        help="Overlay per-horizon OLS fits of AUC on log10 events in panel A")
    args = parser.parse_args()
    if args.registry != DEFAULT_REGISTRY and args.display_groups is None:
        parser.error("--display-groups is required with a nondefault --registry")
    if args.run:
        frame, selected_epoch = prepare_approved_run_data(
            args.run, registry_path=args.registry,
            groups_path=args.display_groups or DEFAULT_GROUPS,
        )
    else:
        frame, selected_epoch = prepare_neural_data(
            args.bundle, args.run_token,
            registry_path=args.registry,
            groups_path=args.display_groups or DEFAULT_GROUPS,
        )
    if args.panel_decisions:
        frame = filter_panel_candidates(
            frame, args.panel_decisions, expected_count=args.expect_candidates
        )
    elif args.expect_candidates is not None:
        parser.error("--expect-candidates requires --panel-decisions")
    groups_path = args.display_groups or (
        DEFAULT_CANDIDATE_GROUPS if args.panel_decisions and frame["outcome"].nunique() == 60
        else None
    )
    if groups_path:
        frame = apply_display_groups(frame, groups_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.output.with_name(args.output.stem + "_plot_data.csv")
    frame.sort_values(["display_group", "outcome", "horizon_days"]).to_csv(
        data_path, index=False
    )
    if args.horizon_trends:
        trends = []
        for horizon, subset in frame.groupby("horizon_days"):
            _, summary = horizon_linear_trend(subset)
            trends.append({"horizon_days": horizon, **summary})
        pd.DataFrame(trends).to_csv(
            args.output.with_name(args.output.stem + "_trend_summary.csv"), index=False
        )
    if args.run:
        atlas_groups = yaml.safe_load(
            (args.display_groups or DEFAULT_GROUPS).read_text(encoding="utf-8")
        )
        atlas_path = args.output.with_name(args.output.stem + "_outcome_atlas.png")
        plot_neural_outcome_atlas(
            frame, atlas_path, groups=atlas_groups, run_label=args.run.name,
            selected_epoch=selected_epoch,
        )
    plot_neural_predictability(
        frame, args.output, run_label=args.run.name if args.run else args.run_token.rstrip("_"),
        selected_epoch=selected_epoch, candidate_subset=bool(args.panel_decisions),
        horizon_trends=args.horizon_trends,
    )
    print(f"Wrote {len(frame)} tuning horizon-outcome cells: {args.output}; data: {data_path}"
          + (f"; atlas: {atlas_path}" if args.run else ""))


if __name__ == "__main__":
    main()
