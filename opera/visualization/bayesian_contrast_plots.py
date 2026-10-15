"""Auditable raw-versus-partially-pooled contrast curves and panel estimates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from opera.visualization.style import PALETTE, save_fig, setup_style


def plot_contrast_results(output_dir):
    root = Path(output_dir)
    audit = json.loads((root / "model_specification.json").read_text())
    cells = pd.read_csv(root / "posterior_cells.csv")
    curves = pd.read_csv(root / "posterior_curve.csv")
    panel = pd.read_csv(root / "posterior_panel.csv")
    contrasts = audit["contrasts"]
    setup_style()
    columns = min(3, len(contrasts))
    rows = int(np.ceil(len(contrasts) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.2 * rows),
        squeeze=False,
        layout="constrained",
    )
    for axis, contrast in zip(axes.flat, contrasts):
        raw = cells.loc[cells.contrast.eq(contrast)]
        curve = curves.loc[curves.contrast.eq(contrast)]
        x = np.log2(1 + raw[audit["rarity_column"]].to_numpy(float))
        gx = np.log2(1 + curve.training_support.to_numpy(float))
        axis.axhspan(
            -audit["practical_tolerance"],
            audit["practical_tolerance"],
            color=PALETTE["grid"],
            label="Practical equivalence",
        )
        axis.axhline(0, color=PALETTE["zero_line"], lw=0.8)
        axis.scatter(
            x,
            raw.raw_gain,
            c=PALETTE["pretrain"],
            s=12,
            alpha=0.55,
            label="Raw paired gain",
            rasterized=True,
        )
        axis.scatter(
            x,
            raw.posterior_median,
            c=PALETTE["opera"],
            s=9,
            alpha=0.6,
            label="Partially pooled cell estimate",
            rasterized=True,
        )
        axis.fill_between(
            gx,
            curve.posterior_lower_95,
            curve.posterior_upper_95,
            color=PALETTE["opera"],
            alpha=0.16,
        )
        axis.plot(
            gx,
            curve.posterior_median,
            color=PALETTE["opera"],
            lw=1.5,
            label="Mean trend and 95% interval",
        )
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{max(0, 2**value - 1):,.0f}")
        )
        axis.set(
            title=contrast,
            xlabel=f"Target-cell {audit['rarity_column']}",
            ylabel=f"Gain ({audit['metric']})",
        )
    for axis in list(axes.flat)[len(contrasts) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[0].legend(handles, labels, fontsize=6, frameon=False)
    warning = " — DIAGNOSTIC ONLY" if not audit["convergence_passed"] else ""
    synthetic = "SYNTHETIC REVIEW — " if audit.get("synthetic_review") else ""
    fig.suptitle(f"{synthetic}{audit['family']}: effect heterogeneity{warning}")
    save_fig(fig, str(root / "contrast_curves.pdf"))
    plt.close(fig)
    fig, axis = plt.subplots(
        figsize=(6, max(2.5, 0.55 * len(contrasts))), layout="constrained"
    )
    for i, contrast in enumerate(contrasts):
        row = panel.loc[panel.contrast.eq(contrast)].iloc[0]
        for offset, prefix, color, label in (
            (-0.1, "raw", PALETTE["pretrain"], "Raw paired average"),
            (0.1, "posterior", PALETTE["opera"], "Bayesian panel average"),
        ):
            center = row.raw_gain if prefix == "raw" else row.posterior_median
            lo, hi = row[f"{prefix}_lower_95"], row[f"{prefix}_upper_95"]
            if np.isfinite([lo, hi]).all():
                axis.plot([lo, hi], [i + offset, i + offset], c=color, lw=1.5)
            axis.scatter(
                [center], [i + offset], c=color, s=22, label=label if i == 0 else None
            )
    axis.axvspan(
        -audit["practical_tolerance"],
        audit["practical_tolerance"],
        color=PALETTE["grid"],
    )
    axis.axvline(0, c=PALETTE["zero_line"], lw=0.8)
    axis.set(
        yticks=np.arange(len(contrasts)),
        yticklabels=contrasts,
        xlabel=f"Gain ({audit['metric']}); {audit['weighting']}",
        title=f"{synthetic}{audit['family']}: finite-panel effects{warning}",
    )
    axis.invert_yaxis()
    axis.legend(frameon=False, fontsize=8)
    save_fig(fig, str(root / "contrast_panel.pdf"))
    plt.close(fig)
    return root / "contrast_curves.pdf", root / "contrast_panel.pdf"
