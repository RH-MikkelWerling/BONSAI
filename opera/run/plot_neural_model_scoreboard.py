"""Plot descriptive tuning AUC aggregates from paired neural forest estimates.

This does not reproduce Lightning's 52-endpoint-group checkpoint-selection
metric. The dots are pooled-outcome AUCs on the same supported outcome panel;
macro summaries have no valid joint patient-bootstrap interval here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.visualization.style import CATEGORICAL, MODEL_DISPLAY, PALETTE, setup_style


def summarize_models(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    required = {"outcome", "outcome_family", "model", "estimate", "horizon_days"}
    if not required <= set(frame):
        raise ValueError(f"Missing estimate columns: {sorted(required - set(frame))}")
    if frame.duplicated(["model", "outcome", "horizon_days"]).any():
        raise ValueError("Duplicate model/outcome/horizon estimates")
    if frame.horizon_days.nunique() != 1 or frame.empty:
        raise ValueError("Select one nonempty horizon before plotting")
    models = list(dict.fromkeys(frame.model.astype(str)))
    if baseline not in models:
        raise ValueError(
            f"Baseline {baseline!r} is absent; available models: {models}. "
            "This estimates CSV cannot be relabeled or augmented with an "
            "unmatched run. Re-run the paired prediction comparison with the "
            "baseline prediction parquet included."
        )
    outcome_sets = [set(frame.loc[frame.model.eq(model), "outcome"]) for model in models]
    if any(values != outcome_sets[0] for values in outcome_sets[1:]):
        raise ValueError("Models do not cover the same outcomes")
    families = frame.groupby("outcome").outcome_family.nunique()
    if not families.eq(1).all():
        raise ValueError("An outcome has conflicting family assignments")
    pivot = frame.pivot(index="outcome", columns="model", values="estimate")
    if not np.isfinite(pivot.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite AUC estimate")
    base = pivot[baseline]
    rows = []
    for model in models:
        selected = frame.loc[frame.model.eq(model)]
        family_means = selected.groupby("outcome_family").estimate.mean()
        rows.append({
            "model": model,
            "outcome_macro_auc": float(pivot[model].mean()),
            "family_balanced_auc": float(family_means.mean()),
            "outcome_macro_delta_vs_baseline": float((pivot[model] - base).mean()),
            "n_outcomes": len(pivot),
            "n_families": len(family_means),
            "horizon_days": int(frame.horizon_days.iloc[0]),
        })
    return pd.DataFrame(rows)


def plot_scoreboard(frame: pd.DataFrame, summary: pd.DataFrame, baseline: str,
                    output: Path) -> None:
    setup_style()
    models = summary.sort_values("outcome_macro_auc").model.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, max(4.3, 0.48 * len(models) + 1.6)),
                             sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
    positions = np.arange(len(models))
    colors = [PALETTE.get(model, CATEGORICAL[i % len(CATEGORICAL)])
              for i, model in enumerate(models)]
    for y, (model, color) in enumerate(zip(models, colors)):
        values = frame.loc[frame.model.eq(model), "estimate"].to_numpy(float)
        axes[0].scatter(values, np.full(len(values), y), s=11, alpha=.17,
                        color=color, linewidths=0, rasterized=True)
        axes[0].scatter(summary.set_index("model").loc[model, "outcome_macro_auc"], y,
                        s=76, marker="D", color=color, edgecolor="white", linewidth=.8,
                        zorder=3)
        family_values = frame.loc[frame.model.eq(model)].groupby("outcome_family").estimate.mean()
        axes[1].scatter(family_values, np.full(len(family_values), y), s=27,
                        alpha=.42, color=color, linewidths=0)
        axes[1].scatter(summary.set_index("model").loc[model, "family_balanced_auc"], y,
                        s=76, marker="D", color=color, edgecolor="white", linewidth=.8,
                        zorder=3)
    for ax in axes:
        ax.set_yticks(positions, [MODEL_DISPLAY.get(model, model.replace("_", " "))
                                  for model in models])
        ax.set_ylim(-.55, len(models) - .45)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_title("A  Outcome-macro discrimination", loc="left", weight="bold")
    axes[1].set_title("B  Family-balanced discrimination", loc="left", weight="bold")
    axes[0].set_xlabel("Pooled outcome CIF-AUC (dots: outcomes; diamond: mean)")
    axes[1].set_xlabel("Mean within family (dots: families; diamond: mean)")
    fig.suptitle(f"OPERA tuning model comparison · {int(frame.horizon_days.iloc[0])}-day horizon",
                 x=.04, y=.99, ha="left", fontsize=13, color=PALETTE["ink"])
    fig.text(.04, .025,
             f"Matched supported outcomes across models; baseline: {MODEL_DISPLAY.get(baseline, baseline)}. "
             "Descriptive selected-checkpoint comparison; no macro confidence intervals.",
             ha="left", va="bottom", fontsize=8, color=PALETTE["ink_secondary"])
    fig.subplots_adjust(left=.29, right=.98, top=.89, bottom=.15, wspace=.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.estimates)
    summary = summarize_models(frame, args.baseline)
    plot_scoreboard(frame, summary, args.baseline, args.output)
    summary.to_csv(args.output.with_suffix(".csv"), index=False)
    print(summary.to_string(index=False))
    print("Figure:", args.output)


if __name__ == "__main__":
    main()
