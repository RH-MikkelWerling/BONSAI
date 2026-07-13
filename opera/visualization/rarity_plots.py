from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from opera.visualization.style import (
    ANNOT_SIZE,
    CATEGORICAL,
    FIG_FULL,
    LEGEND_SIZE,
    NOTE_SIZE,
    PALETTE,
    despine,
    model_color,
    model_label,
    save_fig,
    setup_style,
)


def _finish_delta_axis(ax: plt.Axes) -> None:
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=0.9, zorder=1)
    despine(ax, grid_axis="both")


def _split_main_supplement(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "supplement_only" not in group.columns:
        return group, group.iloc[0:0]
    supplement = group[group["supplement_only"].fillna(False)]
    main = group[~group["supplement_only"].fillna(False)]
    return main, supplement


def _real_yerr(group: pd.DataFrame):
    if {"auroc_lower", "auroc_upper", "baseline_auroc"}.issubset(group.columns):
        lower = group["delta_auroc_vs_baseline"] - (
            group["auroc_lower"] - group["baseline_auroc"]
        )
        upper = (group["auroc_upper"] - group["baseline_auroc"]) - group[
            "delta_auroc_vs_baseline"
        ]
        return [lower.clip(lower=0), upper.clip(lower=0)]
    return None


def _rarity_x_column(frame: pd.DataFrame) -> Optional[str]:
    """Prefer event-count rarity, with cohort-size fallback for old outputs."""
    for column in ("n_events_train", "n_train"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.gt(0).any():
                return column
    return None


def _rarity_x_label(column: Optional[str]) -> str:
    if column == "n_events_train":
        return "Training events"
    if column == "n_train":
        return "Training cohort size"
    return "Training rarity"


def _smoothed_trend(
    group: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    stable_only: bool,
    log_x: bool,
    min_points: int = 3,
    smooth_fraction: float = 0.6,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return a local weighted trend without assuming a linear effect shape."""
    fit_data = group.copy()
    if stable_only and "supplement_only" in fit_data.columns:
        fit_data = fit_data[~fit_data["supplement_only"].fillna(False)]
    if fit_data.empty:
        return None
    fit_data[x_col] = pd.to_numeric(fit_data[x_col], errors="coerce")
    fit_data[y_col] = pd.to_numeric(fit_data[y_col], errors="coerce")
    fit_data = fit_data.dropna(subset=[x_col, y_col])
    fit_data = fit_data[fit_data[x_col] > 0]
    if fit_data.empty:
        return None

    fit_data = (
        fit_data.groupby(x_col, as_index=False, dropna=False)[y_col]
        .median()
        .sort_values(x_col)
    )
    if len(fit_data) < min_points or fit_data[x_col].nunique() < min_points:
        return None

    x = fit_data[x_col].to_numpy(dtype=float)
    y = fit_data[y_col].to_numpy(dtype=float)
    grid = (
        np.geomspace(x.min(), x.max(), 120)
        if log_x
        else np.linspace(x.min(), x.max(), 120)
    )
    x_fit = np.log10(x) if log_x else x
    grid_fit = np.log10(grid) if log_x else grid
    n_points = len(x_fit)
    k_neighbors = min(n_points, max(2, int(np.ceil(smooth_fraction * n_points))))
    values: list[float] = []
    for grid_value in grid_fit:
        distances = np.abs(x_fit - grid_value)
        radius = np.partition(distances, k_neighbors - 1)[k_neighbors - 1]
        if radius <= 0:
            positive = distances[distances > 0]
            radius = positive.min() if positive.size else 1.0
        scaled = distances / radius
        weights = np.where(scaled < 1, (1 - scaled**3) ** 3, 0.0)
        if weights.sum() <= 0:
            weights[np.argmin(distances)] = 1.0
        values.append(float(np.average(y, weights=weights)))
    return grid, np.asarray(values)


def _plot_smoothed_trend(
    ax: plt.Axes,
    group: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color: str,
    label: str,
    stable_only: bool,
    log_x: bool,
) -> None:
    trend = _smoothed_trend(
        group,
        x_col=x_col,
        y_col=y_col,
        stable_only=stable_only,
        log_x=log_x,
    )
    if trend is None:
        return
    grid, values = trend
    ax.plot(
        grid,
        values,
        color=color,
        linestyle="-",
        linewidth=1.8,
        alpha=0.72,
        label=label,
        zorder=2,
    )


def _synthetic_trend_frame(group: pd.DataFrame) -> pd.DataFrame:
    trend_data = group.copy()
    trend_data["training_percent"] = pd.to_numeric(
        trend_data["training_fraction"], errors="coerce"
    ) * 100.0
    return trend_data


def plot_synthetic_rarity_delta(
    pooled: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Synthetic rarity: common disease cells with artificially reduced labels.

    The y-axis is model AUROC minus the named baseline AUROC.
    """
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    if pooled.empty:
        save_fig(fig, save_path)
        return fig

    for model, group in pooled.groupby("model_family"):
        group = group.sort_values("training_fraction")
        x = group["training_fraction"].astype(float) * 100.0
        y = group["median_delta_auroc"].astype(float)
        color = model_color(model)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=color,
            label=model_label(model),
            zorder=3,
        )
        if {"lower_delta_auroc", "upper_delta_auroc"}.issubset(group.columns):
            ax.fill_between(
                x,
                group["lower_delta_auroc"].astype(float),
                group["upper_delta_auroc"].astype(float),
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        _plot_smoothed_trend(
            ax,
            _synthetic_trend_frame(group),
            x_col="training_percent",
            y_col="median_delta_auroc",
            color=color,
            label=f"{model_label(model)} smoothed trend",
            stable_only=False,
            log_x=False,
        )

    _finish_delta_axis(ax)
    ax.set_xlabel("Training labels used (%)")
    ax.set_ylabel("Delta ROC-AUC vs baseline")
    ax.set_title("Synthetic rarity / label scarcity")
    ax.set_xlim(left=0)
    ax.legend(fontsize=LEGEND_SIZE, loc="best")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_real_rarity_delta(
    task_level: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Real rarity: genuinely small cohorts or cohort-outcome cells.

    Uses actual training-event count on a log-scaled x-axis when available,
    falling back to n_train for older result rows. Bootstrap uncertainty columns
    are kept when rows provide them.
    """
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    x_col = _rarity_x_column(task_level)
    if task_level.empty or x_col is None:
        save_fig(fig, save_path)
        return fig

    for model, group in task_level.groupby("model_family"):
        group = group.copy()
        group[x_col] = pd.to_numeric(group[x_col], errors="coerce")
        group = group.dropna(subset=[x_col, "delta_auroc_vs_baseline"])
        group = group[group[x_col] > 0]
        if group.empty:
            continue
        color = model_color(model)
        main, supplement = _split_main_supplement(group)
        for subset, marker, fill, alpha, label_suffix in (
            (main, "o", color, 0.92, ""),
            (supplement, "s", "white", 0.95, " (supplement)"),
        ):
            if subset.empty:
                continue
            ax.errorbar(
                subset[x_col].astype(float),
                subset["delta_auroc_vs_baseline"].astype(float),
                yerr=_real_yerr(subset),
                fmt=marker,
                color=color,
                markerfacecolor=fill,
                markeredgecolor=color,
                markeredgewidth=1.0,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.5,
                markersize=5.0,
                label=model_label(model) + label_suffix,
                alpha=alpha,
                zorder=3,
            )
        _plot_smoothed_trend(
            ax,
            group,
            x_col=x_col,
            y_col="delta_auroc_vs_baseline",
            color=color,
            label=f"{model_label(model)} smoothed trend (stable cells)",
            stable_only=True,
            log_x=True,
        )

    _finish_delta_axis(ax)
    ax.set_xscale("log")
    ax.set_xlabel(_rarity_x_label(x_col))
    ax.set_ylabel("Delta ROC-AUC vs baseline")
    ax.set_title("Real rare cohort-outcome cells")
    ax.legend(fontsize=LEGEND_SIZE, loc="best")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_combined_rarity_delta(
    synthetic_pooled: pd.DataFrame,
    real_task_level: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side overview keeping synthetic and real rarity visually distinct."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax_syn, ax_real = axes
    if not synthetic_pooled.empty:
        for model, group in synthetic_pooled.groupby("model_family"):
            group = group.sort_values("training_fraction")
            color = model_color(model)
            x = group["training_fraction"].astype(float) * 100.0
            ax_syn.plot(
                x,
                group["median_delta_auroc"].astype(float),
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                color=color,
                label=model_label(model),
            )
            if {"lower_delta_auroc", "upper_delta_auroc"}.issubset(group.columns):
                ax_syn.fill_between(
                    x,
                    group["lower_delta_auroc"].astype(float),
                    group["upper_delta_auroc"].astype(float),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            _plot_smoothed_trend(
                ax_syn,
                _synthetic_trend_frame(group),
                x_col="training_percent",
                y_col="median_delta_auroc",
                color=color,
                label=f"{model_label(model)} smoothed trend",
                stable_only=False,
                log_x=False,
            )
    _finish_delta_axis(ax_syn)
    ax_syn.set_title("Synthetic label scarcity")
    ax_syn.set_xlabel("Training labels used (%)")
    ax_syn.set_ylabel("Delta ROC-AUC vs baseline")

    real_x_col = _rarity_x_column(real_task_level)
    if not real_task_level.empty and real_x_col is not None:
        for model, group in real_task_level.groupby("model_family"):
            group = group.copy()
            group[real_x_col] = pd.to_numeric(group[real_x_col], errors="coerce")
            group = group.dropna(subset=[real_x_col, "delta_auroc_vs_baseline"])
            group = group[group[real_x_col] > 0]
            if group.empty:
                continue
            color = model_color(model)
            main, supplement = _split_main_supplement(group)
            if not main.empty:
                ax_real.scatter(
                    main[real_x_col].astype(float),
                    main["delta_auroc_vs_baseline"].astype(float),
                    color=color,
                    s=34,
                    alpha=0.88,
                    label=model_label(model),
                )
            if not supplement.empty:
                ax_real.scatter(
                    supplement[real_x_col].astype(float),
                    supplement["delta_auroc_vs_baseline"].astype(float),
                    facecolors="white",
                    edgecolors=color,
                    marker="s",
                    s=36,
                    alpha=0.95,
                    label=model_label(model) + " (supplement)",
                )
            _plot_smoothed_trend(
                ax_real,
                group,
                x_col=real_x_col,
                y_col="delta_auroc_vs_baseline",
                color=color,
                label=f"{model_label(model)} smoothed trend",
                stable_only=True,
                log_x=True,
            )
    _finish_delta_axis(ax_real)
    ax_real.set_xscale("log")
    ax_real.set_title("Real rare cohort-outcome cells")
    ax_real.set_xlabel(_rarity_x_label(real_x_col))
    ax_real.set_ylabel("Delta ROC-AUC vs baseline")

    handles, labels = ax_syn.get_legend_handles_labels()
    handles2, labels2 = ax_real.get_legend_handles_labels()
    by_label = dict(zip(labels + labels2, handles + handles2))
    if (
        "supplement_only" in real_task_level.columns
        and real_task_level["supplement_only"].fillna(False).any()
    ):
        by_label.setdefault(
            "Supplement-only real rare cell",
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor=PALETTE["ink_secondary"],
                markersize=5,
            ),
        )
    if by_label:
        fig.legend(
            by_label.values(),
            by_label.keys(),
            frameon=False,
            fontsize=LEGEND_SIZE,
            loc="lower center",
            ncol=min(4, len(by_label)),
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1))
    else:
        fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def write_rarity_plots(
    synthetic_pooled: pd.DataFrame,
    real_task_level: pd.DataFrame,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_synthetic_rarity_delta(
        synthetic_pooled,
        save_path=str(out / "synthetic_rarity_delta.png"),
    )
    plot_real_rarity_delta(
        real_task_level,
        save_path=str(out / "real_rarity_delta.png"),
    )
    plot_combined_rarity_delta(
        synthetic_pooled,
        real_task_level,
        save_path=str(out / "rarity_delta_combined.png"),
    )


def plot_synthetic_learning_curves(
    summary: pd.DataFrame,
    *,
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot task-macro learning curves against absolute target-label count."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_FULL)
    data = summary[summary["metric"] == metric].copy() if not summary.empty else summary
    if data.empty or not {"model", "sample_size", "macro_mean"}.issubset(data.columns):
        save_fig(fig, save_path)
        return fig
    for model, group in data.groupby("model"):
        group = group.sort_values("sample_size")
        color = model_color(model)
        ax.plot(
            group["sample_size"], group["macro_mean"], marker="o",
            color=color, label=model_label(model),
        )
        if {"ci_lower", "ci_upper"}.issubset(group.columns):
            ax.fill_between(
                group["sample_size"].astype(float),
                group["ci_lower"].astype(float),
                group["ci_upper"].astype(float),
                color=color, alpha=0.16, linewidth=0,
            )
    if not data.empty and (data["sample_size"] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel("Labelled target-diagnosis training patients")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("Controlled synthetic label scarcity (task-macro)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_relative_benefit_curve(
    summary: pd.DataFrame,
    *,
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot task-macro paired benefit over configured comparators."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_FULL)
    data = summary[summary["metric"] == metric].copy() if not summary.empty else summary
    if data.empty or not {"model", "comparator", "sample_size", "macro_mean"}.issubset(data.columns):
        save_fig(fig, save_path)
        return fig
    for (model, comparator), group in data.groupby(["model", "comparator"]):
        group = group.sort_values("sample_size")
        color = model_color(model)
        ax.plot(
            group["sample_size"], group["macro_mean"], marker="o", color=color,
            label=f"{model_label(model)} - {model_label(comparator)}",
        )
        if {"ci_lower", "ci_upper"}.issubset(group.columns):
            ax.fill_between(
                group["sample_size"].astype(float), group["ci_lower"].astype(float),
                group["ci_upper"].astype(float), color=color, alpha=0.16, linewidth=0,
            )
    _finish_delta_axis(ax)
    if not data.empty and (data["sample_size"] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel("Labelled target-diagnosis training patients")
    ax.set_ylabel(f"Paired {metric.replace('_', ' ')} benefit")
    ax.set_title("Relative benefit under target-label scarcity")
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_natural_viability_heatmap(
    eligibility: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show test-event counts and natural-task viability for every task."""
    setup_style()
    if eligibility.empty or not {"cohort_fine", "outcome"}.issubset(eligibility.columns):
        fig, _ = plt.subplots(figsize=FIG_FULL)
        save_fig(fig, save_path)
        return fig
    diagnoses = sorted(eligibility["cohort_fine"].dropna().unique())
    outcomes = sorted(eligibility["outcome"].dropna().unique())
    values = np.full((len(diagnoses), len(outcomes)), np.nan)
    annotations = np.full((len(diagnoses), len(outcomes)), "", dtype=object)
    tier_value = {"non_evaluable": 0, "aggregate_only": 1, "primary": 2}
    for row in eligibility.itertuples(index=False):
        i, j = diagnoses.index(row.cohort_fine), outcomes.index(row.outcome)
        values[i, j] = tier_value.get(row.natural_viability_tier, 0)
        annotations[i, j] = f"{int(row.n_test_positive)} events\n{row.natural_viability_tier}"
    width = max(5.5, 1.4 * len(outcomes))
    height = max(4.0, 0.42 * len(diagnoses))
    fig, ax = plt.subplots(figsize=(width, height))
    from matplotlib.colors import ListedColormap

    image = ax.imshow(values, aspect="auto", cmap=ListedColormap(["#EEEEEE", "#F2C66D", "#4C9F70"]), vmin=0, vmax=2)
    del image
    for i in range(len(diagnoses)):
        for j in range(len(outcomes)):
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=NOTE_SIZE)
    ax.set_xticks(range(len(outcomes)), [v.replace("_", " ") for v in outcomes], rotation=30, ha="right")
    ax.set_yticks(range(len(diagnoses)), diagnoses)
    ax.set_title("Natural rarity task viability (fixed 2023+ test set)")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Fine diagnosis")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_natural_rarity_scatter(
    differences: pd.DataFrame,
    *,
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot paired natural-task benefit without pooling task definitions."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_FULL)
    data = differences[differences["metric"] == metric].copy() if not differences.empty else differences
    required = {"outcome", "difference", "n_train_determinate", "natural_viability_tier"}
    if data.empty or not required.issubset(data.columns):
        save_fig(fig, save_path)
        return fig
    for outcome, group in data.groupby("outcome"):
        color = CATEGORICAL[hash(outcome) % len(CATEGORICAL)]
        marker = "o"
        main = group[group["natural_viability_tier"] == "primary"]
        aggregate = group[group["natural_viability_tier"] == "aggregate_only"]
        for subset, fill, suffix in ((main, color, ""), (aggregate, "white", " (aggregate only)")):
            if subset.empty:
                continue
            lower = subset["difference"] - subset["difference_ci_lower"]
            upper = subset["difference_ci_upper"] - subset["difference"]
            ax.errorbar(
                subset["n_train_determinate"], subset["difference"],
                yerr=[lower.clip(lower=0), upper.clip(lower=0)], fmt=marker,
                color=color, markerfacecolor=fill, capsize=2.5,
                label=outcome.replace("_", " ") + suffix,
            )
    _finish_delta_axis(ax)
    if not data.empty and (data["n_train_determinate"] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel("Natural target-diagnosis training patients")
    ax.set_ylabel(f"Paired {metric.replace('_', ' ')} benefit")
    ax.set_title("Naturally occurring fine-cohort rarity")
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_diagnosis_learning_curves(
    metrics: pd.DataFrame,
    *,
    cohort_fine: str,
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Diagnosis panel with outcome-specific curves retained as distinct lines."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIG_FULL)
    required = {"cohort_fine", "outcome", "model", "sample_size", "metric", "estimate"}
    if metrics.empty or not required.issubset(metrics.columns):
        save_fig(fig, save_path)
        return fig
    data = metrics[(metrics["cohort_fine"] == cohort_fine) & (metrics["metric"] == metric)]
    summary = data.groupby(["outcome", "model", "sample_size"], as_index=False)["estimate"].mean()
    for (outcome, model), group in summary.groupby(["outcome", "model"]):
        group = group.sort_values("sample_size")
        ax.plot(
            group["sample_size"], group["estimate"], marker="o",
            color=model_color(model),
            linestyle="-" if hash(outcome) % 2 else "--",
            label=f"{model_label(model)} | {outcome.replace('_', ' ')}",
        )
    if not summary.empty and (summary["sample_size"] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel("Labelled target-diagnosis training patients")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"{cohort_fine}: controlled label scarcity")
    ax.legend(fontsize=LEGEND_SIZE, ncol=2)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_combined_rarity_experiment(
    synthetic: pd.DataFrame,
    natural: pd.DataFrame,
    *,
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Adjacent panels; synthetic and natural observations are never pooled."""
    setup_style()
    fig, (ax_syn, ax_nat) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    syn = synthetic[synthetic["metric"] == metric] if not synthetic.empty and "metric" in synthetic else pd.DataFrame()
    if not syn.empty:
        for (model, comparator), group in syn.groupby(["model", "comparator"]):
            group = group.sort_values("sample_size")
            ax_syn.plot(group["sample_size"], group["macro_mean"], marker="o", color=model_color(model), label=f"{model_label(model)} - {model_label(comparator)}")
    _finish_delta_axis(ax_syn)
    ax_syn.set_xscale("log")
    ax_syn.set_title("Controlled synthetic rarity")
    ax_syn.set_xlabel("Sampled target labels")
    ax_syn.set_ylabel(f"Paired {metric.replace('_', ' ')} benefit")

    nat = natural[natural["metric"] == metric] if not natural.empty and "metric" in natural else pd.DataFrame()
    if not nat.empty:
        for outcome, group in nat.groupby("outcome"):
            color = CATEGORICAL[hash(outcome) % len(CATEGORICAL)]
            ax_nat.scatter(group["n_train_determinate"], group["difference"], color=color, alpha=0.82, label=outcome.replace("_", " "))
    _finish_delta_axis(ax_nat)
    ax_nat.set_xscale("log")
    ax_nat.set_title("Naturally occurring rarity")
    ax_nat.set_xlabel("Natural target labels")
    ax_nat.set_ylabel(f"Paired {metric.replace('_', ' ')} benefit")
    handles, labels = ax_syn.get_legend_handles_labels()
    handles2, labels2 = ax_nat.get_legend_handles_labels()
    if handles or handles2:
        fig.legend(handles + handles2, labels + labels2, frameon=False, fontsize=LEGEND_SIZE, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
    else:
        fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def write_rarity_experiment_plots(
    *,
    output_dir: str | Path,
    eligibility: pd.DataFrame,
    synthetic_metrics: pd.DataFrame,
    synthetic_summary: pd.DataFrame,
    synthetic_difference_summary: pd.DataFrame,
    natural_differences: pd.DataFrame,
    metrics: tuple[str, ...] = ("auroc", "auprc", "pr_skill", "brier_skill"),
) -> None:
    """Regenerate the publication figures entirely from cached tables."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_natural_viability_heatmap(eligibility, str(output / "natural_viability_heatmap.png"))
    diagnosis_dir = output / "diagnosis_learning_curves"
    if not synthetic_metrics.empty and "cohort_fine" in synthetic_metrics:
        for diagnosis in sorted(synthetic_metrics["cohort_fine"].dropna().unique()):
            plot_diagnosis_learning_curves(
                synthetic_metrics, cohort_fine=diagnosis,
                save_path=str(diagnosis_dir / f"{diagnosis}_auroc.png"),
            )
    for metric in metrics:
        plot_synthetic_learning_curves(
            synthetic_summary, metric=metric,
            save_path=str(output / f"synthetic_learning_curve_{metric}.png"),
        )
        plot_relative_benefit_curve(
            synthetic_difference_summary, metric=metric,
            save_path=str(output / f"synthetic_relative_benefit_{metric}.png"),
        )
        plot_natural_rarity_scatter(
            natural_differences, metric=metric,
            save_path=str(output / f"natural_rarity_scatter_{metric}.png"),
        )
        plot_combined_rarity_experiment(
            synthetic_difference_summary, natural_differences, metric=metric,
            save_path=str(output / f"combined_synthetic_natural_{metric}.png"),
        )


def plot_rarity_delta(
    results_df: pd.DataFrame,
    baseline_model: str = "tabular_ehr",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot OPERA minus tabular-EHR AUROC by cohort-outcome rarity.

    Input is long result-schema rows. The function filters to full-cohort rows
    because IPI-complete rows represent a different patient population. The
    scientific purpose is to show foundation-model gain over the universal
    tabular EHR baseline as training set size decreases.
    """
    setup_style()
    df = results_df.copy()
    if "evaluation_subset" in df.columns:
        df = df[df["evaluation_subset"].fillna("full") == "full"]
    key_cols = ["cohort", "outcome"]
    if "outcome_window_hours" in df.columns:
        key_cols.append("outcome_window_hours")
    wide = df.pivot_table(
        index=key_cols, columns="model_family", values="auroc", aggfunc="first"
    )
    meta_cols = [
        c for c in ["n_events_train", "n_train", "rarity_tier"] if c in df.columns
    ]
    meta = (
        df.groupby(key_cols, dropna=False)[meta_cols].first()
        if meta_cols
        else pd.DataFrame(index=wide.index)
    )
    plot_df = wide.join(meta).reset_index()
    plot_df["delta"] = plot_df.get("opera") - plot_df.get(baseline_model)
    plot_df["label"] = (
        plot_df["cohort"].astype(str).str.upper()
        + "-"
        + plot_df["outcome"].astype(str).str.replace("_", "")
    )
    rarity_col = _rarity_x_column(plot_df)
    sort_col = rarity_col if rarity_col is not None else "delta"
    plot_df = plot_df.dropna(subset=["delta"]).sort_values(sort_col)

    fig, ax = plt.subplots(figsize=FIG_FULL)
    ax.axhspan(
        plot_df["delta"].min() if not plot_df.empty else -0.01,
        0,
        color="#F3F3F3",
        zorder=0,
    )
    for i, row in plot_df.reset_index(drop=True).iterrows():
        tier = str(row.get("rarity_tier", "unknown"))
        color = CATEGORICAL[hash(tier) % len(CATEGORICAL)]
        ax.scatter(i, row["delta"], color=color, s=38, zorder=3)
        ax.text(i, row["delta"], row["label"], fontsize=ANNOT_SIZE, ha="center", va="bottom")
    ax.axhline(0, color=PALETTE["zero_line"], linewidth=0.9)
    ax.set_xlabel(
        f"Cohort-outcome cells sorted by {_rarity_x_label(rarity_col).lower()}"
    )
    ax.set_ylabel(f"AUROC(OPERA) - AUROC({baseline_model})")
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


def plot_rarity_invariance_panel(
    task_deltas: pd.DataFrame,
    *,
    baseline_model: str,
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot matched outcome deltas against event rate for each weighter."""
    setup_style()
    delta_column = f"delta_{metric}_vs_baseline"
    required = {"weighter", "event_rate", delta_column}
    missing = sorted(required.difference(task_deltas.columns))
    if missing:
        raise ValueError(
            "Rarity invariance plot is missing columns: " + ", ".join(missing)
        )
    task_columns = [
        column
        for column in ("cohort", "outcome", "outcome_window_hours")
        if column in task_deltas.columns
    ]
    plot_data = (
        task_deltas.groupby([*task_columns, "weighter"], as_index=False, dropna=False)
        .agg(
            event_rate=("event_rate", "first"),
            delta=(delta_column, "median"),
        )
        .sort_values("event_rate")
    )

    fig, ax = plt.subplots(figsize=FIG_FULL)
    for index, (weighter, group) in enumerate(plot_data.groupby("weighter")):
        color = CATEGORICAL[index % len(CATEGORICAL)]
        group = group.sort_values("event_rate")
        ax.scatter(
            group["event_rate"],
            group["delta"],
            color=color,
            alpha=0.82,
            s=34,
            label=weighter.title(),
            zorder=3,
        )
        _plot_smoothed_trend(
            ax,
            group,
            x_col="event_rate",
            y_col="delta",
            color=color,
            label=f"{weighter.title()} smoothed trend",
            stable_only=False,
            log_x=True,
        )
    ax.axhline(0.0, color=PALETTE["zero_line"], linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Held-out event rate")
    ax.set_ylabel(f"{metric.upper()}(OPERA) - {metric.upper()}({baseline_model})")
    ax.set_title("Rarity gradient by cross-outcome weighter")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig
