"""Patient-level benefit visualizations for OPERA contrasts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from opera.visualization.style import (
    ANNOT_SIZE,
    CATEGORICAL,
    FIG_FULL,
    LEGEND_SIZE,
    PALETTE,
    TITLE_SIZE,
    add_panel_label,
    despine,
    save_fig,
    setup_style,
)

LOGGER = logging.getLogger(__name__)


def _identity_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    subject_col: str,
    cohort_col: str,
) -> list[str]:
    keys = [subject_col]
    for col in (cohort_col, "outcome"):
        if col in left.columns and col in right.columns:
            keys.append(col)
    return keys


def _embedding_columns(
    frame: pd.DataFrame,
    subject_col: str,
    cohort_col: str,
) -> list[str]:
    excluded = {subject_col, cohort_col, "x", "y", "pacmap_x", "pacmap_y"}
    cols = [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not cols:
        raise ValueError("No numeric embedding columns found.")
    return cols


def _cohort_colors(values: pd.Series) -> dict:
    cohorts = sorted(values.dropna().astype(str).unique())
    return {
        cohort: CATEGORICAL[i % len(CATEGORICAL)] for i, cohort in enumerate(cohorts)
    }


def _load_coords(
    baseline_embeddings: pd.DataFrame,
    pacmap_coords: Optional[pd.DataFrame],
    subject_col: str,
    cohort_col: str,
    n_pacmap_samples: int,
) -> pd.DataFrame:
    if pacmap_coords is not None:
        coords = pacmap_coords.copy()
        if {"pacmap_x", "pacmap_y"}.issubset(coords.columns):
            coords = coords.rename(columns={"pacmap_x": "x", "pacmap_y": "y"})
        if not {subject_col, "x", "y"}.issubset(coords.columns):
            raise ValueError(
                "pacmap_coords must contain subject_id plus x/y or pacmap_x/pacmap_y."
            )
        cols = [subject_col, "x", "y"]
        if cohort_col in coords.columns:
            cols.append(cohort_col)
        out = baseline_embeddings[[subject_col, cohort_col]].merge(
            coords[cols],
            on=_identity_keys(baseline_embeddings, coords, subject_col, cohort_col),
            how="inner",
            suffixes=("", "_coord"),
        )
        if f"{cohort_col}_coord" in out.columns:
            out = out.drop(columns=[f"{cohort_col}_coord"])
        return out

    emb = baseline_embeddings.copy()
    embed_cols = _embedding_columns(emb, subject_col, cohort_col)
    if len(emb) > n_pacmap_samples:
        sampled = (
            emb.groupby(cohort_col, group_keys=False, dropna=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, int(round(n_pacmap_samples * len(g) / len(emb)))),
                    random_state=42,
                )
            )
            .head(n_pacmap_samples)
        )
    else:
        sampled = emb
    x = sampled[embed_cols].to_numpy(dtype=float)
    try:
        import pacmap

        reducer = pacmap.PaCMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(x)
    except Exception as exc:
        LOGGER.warning(
            "PaCMAP unavailable; falling back to UMAP for patient benefit plot: %s", exc
        )
        try:
            from umap import UMAP

            coords = UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.25,
                metric="cosine",
                random_state=42,
            ).fit_transform(x)
        except Exception as exc_umap:
            LOGGER.warning(
                "UMAP unavailable; using deterministic PCA fallback for tests/smoke runs: %s",
                exc_umap,
            )
            x_centered = x - np.nanmean(x, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
            coords = x_centered @ vt[:2].T
    out = sampled[[subject_col, cohort_col]].copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    return out


def _save_both(fig: plt.Figure, save_path: Optional[str]) -> None:
    save_fig(fig, save_path)
    if save_path is None:
        return
    path = Path(save_path)
    if path.suffix.lower() == ".pdf":
        save_fig(fig, str(path.with_suffix(".png")))


def _smooth_gain_surface(
    coords: pd.DataFrame,
    gain_col: str,
    grid_size: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if coords.empty or gain_col not in coords.columns:
        return None
    x = coords["x"].to_numpy(float)
    y = coords["y"].to_numpy(float)
    gain = coords[gain_col].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(gain)
    if valid.sum() < 5:
        return None
    x, y, gain = x[valid], y[valid], gain[valid]
    pad_x = 0.05 * max(float(x.max() - x.min()), 1e-6)
    pad_y = 0.05 * max(float(y.max() - y.min()), 1e-6)
    x_edges = np.linspace(x.min() - pad_x, x.max() + pad_x, grid_size + 1)
    y_edges = np.linspace(y.min() - pad_y, y.max() + pad_y, grid_size + 1)
    density, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    weighted, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=gain)
    try:
        from scipy.ndimage import gaussian_filter

        sigma = max(grid_size * 0.025, 1.0)
        density_s = gaussian_filter(density, sigma=sigma)
        weighted_s = gaussian_filter(weighted, sigma=sigma)
    except Exception:
        density_s = density
        weighted_s = weighted
    with np.errstate(invalid="ignore", divide="ignore"):
        surface = weighted_s / density_s
    surface[density_s < 3] = np.nan
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return np.meshgrid(x_centers, y_centers), surface.T


def _lowess_curve(
    x: np.ndarray, y: np.ndarray, frac: float, it: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return x, y
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        fitted = lowess(y, x, frac=frac, it=it, return_sorted=True)
        return fitted[:, 0], fitted[:, 1]
    except Exception:
        order = np.argsort(x)
        x_s = x[order]
        y_s = (
            pd.Series(y[order])
            .rolling(max(3, int(len(y) * frac)), center=True, min_periods=1)
            .mean()
        )
        return x_s, y_s.to_numpy(float)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan"), int(valid.sum())
    try:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(x[valid], y[valid])
        return float(rho), float(pval), int(valid.sum())
    except Exception:
        rx = pd.Series(x[valid]).rank().to_numpy(float)
        ry = pd.Series(y[valid]).rank().to_numpy(float)
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan"), int(valid.sum())


def _plot_gain_panel(
    ax: plt.Axes,
    merged: pd.DataFrame,
    x_col: str,
    gain_col: str,
    cohort_col: str,
    colors: dict,
    loess_frac: float,
    loess_it: int,
    alpha_scatter: float,
    point_size: float,
    annotate_extremes: int,
) -> None:
    for cohort, group in merged.groupby(cohort_col, dropna=False):
        ax.scatter(
            group[x_col],
            group[gain_col],
            s=point_size,
            color=colors.get(str(cohort), CATEGORICAL[-1]),
            alpha=alpha_scatter,
            rasterized=True,
            linewidths=0,
        )
    lx, ly = _lowess_curve(
        merged[x_col].to_numpy(float),
        merged[gain_col].to_numpy(float),
        loess_frac,
        loess_it,
    )
    ax.plot(lx, ly, color=PALETTE["zero_line"], linewidth=2.0)
    ax.axhline(0, color=PALETTE["zero_line"], linestyle="-", linewidth=0.7)
    ax.axvline(0, color=PALETTE["zero_line"], linestyle="-", linewidth=0.7)
    rho, pval, n = _spearman(
        merged[x_col].to_numpy(float), merged[gain_col].to_numpy(float)
    )
    ax.text(
        0.02,
        0.98,
        f"rho = {rho:.2f},  p = {pval:.2e},  n = {n}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOT_SIZE,
        color=PALETTE["ink_secondary"],
    )
    if annotate_extremes > 0 and not merged.empty:
        top = merged.nlargest(annotate_extremes, gain_col)
        labels = []
        for _, row in top.iterrows():
            labels.append(
                ax.annotate(
                    str(row[cohort_col]),
                    (row[x_col], row[gain_col]),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=ANNOT_SIZE,
                    color=PALETTE["ink_secondary"],
                )
            )
        try:
            from adjustText import adjust_text

            adjust_text(labels, ax=ax)
        except Exception:
            pass
    despine(ax, "both")


def _legend(fig: plt.Figure, colors: dict, n_cohorts: int) -> None:
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color=color, label=cohort, markersize=4
        )
        for cohort, color in colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(n_cohorts, 6),
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, -0.01),
    )


def plot_patient_benefit(
    patient_transfer_df: pd.DataFrame,
    atypicality_df: pd.DataFrame,
    baseline_embeddings: pd.DataFrame,
    *,
    contrast_name: str = "",
    atypicality_mode: str = "own",
    gain_col: str = "brier_gain",
    cohort_col: str = "cohort",
    subject_col: str = "subject_id",
    pacmap_coords: Optional[pd.DataFrame] = None,
    n_pacmap_samples: int = 5000,
    loess_frac: float = 0.4,
    loess_it: int = 2,
    alpha_scatter: float = 0.35,
    point_size: float = 6.0,
    annotate_extremes: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_style()
    coords = _load_coords(
        baseline_embeddings,
        pacmap_coords,
        subject_col,
        cohort_col,
        n_pacmap_samples,
    )
    colors = _cohort_colors(coords[cohort_col])
    fig = plt.figure(figsize=FIG_FULL)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.35)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1])

    gain_keys = _identity_keys(patient_transfer_df, coords, subject_col, cohort_col)
    gain_coords = patient_transfer_df[gain_keys + [gain_col]].merge(
        coords,
        on=gain_keys,
        how="inner",
    )
    surface = _smooth_gain_surface(gain_coords, gain_col)
    if surface is not None:
        (xx, yy), zz = surface
        vmax = np.nanmax(np.abs(zz)) if np.isfinite(zz).any() else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        ax_map.contourf(xx, yy, zz, levels=12, cmap="RdBu_r", norm=norm, alpha=0.35)
        ax_map.text(
            0.98,
            0.02,
            "Brier gain\n(comparator - baseline)",
            transform=ax_map.transAxes,
            ha="right",
            va="bottom",
            fontsize=ANNOT_SIZE,
            color=PALETTE["ink_secondary"],
        )

    for cohort, group in coords.groupby(cohort_col, dropna=False):
        color = colors.get(str(cohort), CATEGORICAL[-1])
        ax_map.scatter(
            group["x"],
            group["y"],
            s=4,
            color=color,
            alpha=0.5,
            rasterized=True,
            linewidths=0,
        )
        centroid = group[["x", "y"]].mean()
        ax_map.scatter(
            centroid["x"],
            centroid["y"],
            s=80,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax_map.text(
            centroid["x"],
            centroid["y"],
            str(cohort).upper(),
            ha="center",
            va="center",
            fontsize=ANNOT_SIZE,
            fontweight="bold",
            color=PALETTE["ink"],
            zorder=6,
        )
    ax_map.set_xlabel("PaCMAP 1")
    ax_map.set_ylabel("PaCMAP 2")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    despine(ax_map, "none")

    merge_keys = _identity_keys(
        patient_transfer_df, atypicality_df, subject_col, cohort_col
    )
    merged = patient_transfer_df.merge(
        atypicality_df,
        on=merge_keys,
        how="inner",
        suffixes=("", "_atyp"),
    )
    if cohort_col not in merged.columns and f"{cohort_col}_atyp" in merged.columns:
        merged[cohort_col] = merged[f"{cohort_col}_atyp"]
    x_col = (
        "atypicality_nearest" if atypicality_mode == "nearest" else "atypicality_own"
    )
    _plot_gain_panel(
        ax_gain,
        merged.dropna(subset=[x_col, gain_col]),
        x_col,
        gain_col,
        cohort_col,
        colors,
        loess_frac,
        loess_it,
        alpha_scatter,
        point_size,
        annotate_extremes,
    )
    if atypicality_mode == "nearest":
        ax_gain.set_xlabel("Atypicality from nearest disease centroid (SD)")
    elif atypicality_mode == "both":
        ax_gain.set_xlabel("Within-disease atypicality (SD from cohort centroid)")
        ax_top = ax_gain.twiny()
        finite = (
            merged[["atypicality_own", "atypicality_nearest"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(finite) >= 2:
            ticks = ax_gain.get_xticks()
            own = finite["atypicality_own"].to_numpy(float)
            nearest = finite["atypicality_nearest"].to_numpy(float)
            coeff = np.polyfit(own, nearest, deg=1)
            ax_top.set_xlim(np.polyval(coeff, ax_gain.get_xlim()))
            ax_top.set_xticks(np.polyval(coeff, ticks))
        ax_top.set_xlabel("Nearest-centroid atypicality (SD)")
        despine(ax_top, "none")
    else:
        ax_gain.set_xlabel("Within-disease atypicality (SD from cohort centroid)")
    ax_gain.set_ylabel("Prediction gain (Delta Brier score, baseline - comparator)")

    add_panel_label(ax_map, "A")
    add_panel_label(ax_gain, "B")
    if contrast_name:
        fig.suptitle(
            f"Patient-level benefit - {contrast_name}",
            fontsize=TITLE_SIZE,
            color=PALETTE["ink"],
            y=0.99,
        )
    _legend(fig, colors, len(colors))
    fig.subplots_adjust(bottom=0.18)
    _save_both(fig, save_path)
    return fig


def plot_benefit_contrast_ladder(
    contrast_results: list[dict],
    *,
    atypicality_mode: str = "own",
    gain_col: str = "brier_gain",
    cohort_col: str = "cohort",
    subject_col: str = "subject_id",
    loess_frac: float = 0.4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_style()
    n_rows = max(1, len(contrast_results))
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(FIG_FULL[0], 2.2 * n_rows), sharex=True
    )
    if n_rows == 1:
        axes = [axes]

    merged_rows = []
    for item in contrast_results:
        merge_keys = _identity_keys(
            item["patient_transfer_df"],
            item["atypicality_df"],
            subject_col,
            cohort_col,
        )
        merged = item["patient_transfer_df"].merge(
            item["atypicality_df"],
            on=merge_keys,
            how="inner",
            suffixes=("", "_atyp"),
        )
        if cohort_col not in merged.columns and f"{cohort_col}_atyp" in merged.columns:
            merged[cohort_col] = merged[f"{cohort_col}_atyp"]
        merged_rows.append(merged)
    all_df = (
        pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
    )
    x_col = (
        "atypicality_nearest" if atypicality_mode == "nearest" else "atypicality_own"
    )
    xlim = None
    ylim = None
    if not all_df.empty:
        x_vals = all_df[x_col].replace([np.inf, -np.inf], np.nan).dropna()
        y_vals = all_df[gain_col].replace([np.inf, -np.inf], np.nan).dropna()
        if not x_vals.empty:
            xlim = (float(x_vals.min()), float(x_vals.max()))
        if not y_vals.empty:
            ylim = (float(y_vals.min()), float(y_vals.max()))
    colors = _cohort_colors(all_df[cohort_col]) if cohort_col in all_df else {}

    for i, (ax, item, merged) in enumerate(zip(axes, contrast_results, merged_rows)):
        _plot_gain_panel(
            ax,
            merged.dropna(subset=[x_col, gain_col]),
            x_col,
            gain_col,
            cohort_col,
            colors,
            loess_frac,
            2,
            0.3,
            5.0,
            0,
        )
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel("Prediction gain")
        ax.text(
            -0.16,
            0.5,
            item.get("contrast_name", f"contrast_{i + 1}"),
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=ANNOT_SIZE,
            color=PALETTE["ink"],
        )
        add_panel_label(ax, chr(ord("A") + i), x=-0.08)
        if i < len(axes) - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(
                "Atypicality from nearest disease centroid (SD)"
                if atypicality_mode == "nearest"
                else "Within-disease atypicality (SD from cohort centroid)"
            )
    _legend(fig, colors, len(colors))
    fig.subplots_adjust(left=0.18, bottom=0.14, hspace=0.35)
    _save_both(fig, save_path)
    return fig
