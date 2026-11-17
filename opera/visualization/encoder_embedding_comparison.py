"""Patient representations from different encoders, side by side (supplementary figure).

Redesign of ``paper_figures.figure_embedding_comparison`` (16 Nov 2026), in the thesis style:

* one column per model (for example the joint model or OPERA from the cohort, national and
  adapted encoders), each with its own two-dimensional projection of the same held-out patients;
* three rows:
    1. disease group: one point per patient, with a contour around each group's densest half;
    2. IPI category: one point per patient, for patients with the IPI computable, in the four
       standard IPI categories (no averaging of categories, no mixing of different indices),
       with outlines around the densest half of the Low and the High category, so the eye can
       judge whether an encoder separates them; patients without an IPI in light grey;
    3. observed one-year overall-survival risk relative to the average, as an IPCW kernel
       estimate on the same scale and colour map as the risk gallery (1/4x to 4x).

Recorded sequence length is no longer shown.

``coords``: ``model``, ``patient_id``, ``x``, ``y``, ``disease`` and ``ipi_group`` (one of
``IPI_LEVELS`` or missing), identical patients and metadata across models.
``survival``: ``patient_id``, ``time_days``, ``event`` (1 death, 0 censored) for overall survival.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from opera.visualization.censoring_aware_embeddings import (
    _default_bandwidth, _kernel_surfaces, horizon_ipcw_data)
from opera.visualization.paper_figures import _category_regions, _embedding_surface_mask, _map_axis
from opera.visualization.style import CATEGORICAL

IPI_LEVELS = ["Low", "Low-intermediate", "High-intermediate", "High"]
IPI_COLOURS = ["#9CC9BE", "#E3BC6A", "#C9705A", "#6E1A2E"]  # light to dark: low to high risk
DISEASE_EXTRA = ["#BD648E", "#697481"]
INK, INK2, MUTED, CLOUD = "#17212B", "#485463", "#657180", "#D5DAE1"


def _check(coords, survival, models):
    need = {"model", "patient_id", "x", "y", "disease", "ipi_group"}
    if not need <= set(coords):
        raise ValueError(f"Missing columns: {sorted(need - set(coords))}")
    if set(models) != set(coords.model) or len(models) != len(set(models)):
        raise ValueError("Model order must list each model once")
    ids = set(coords.loc[coords.model == models[0], "patient_id"])
    for m in models:
        if set(coords.loc[coords.model == m, "patient_id"]) != ids:
            raise ValueError("Every model must show the same patients")
    for col in ("disease", "ipi_group"):
        if (coords.groupby("patient_id")[col].nunique(dropna=False) > 1).any():
            raise ValueError(f"{col} must agree across models")
    unknown = set(coords.ipi_group.dropna()) - set(IPI_LEVELS)
    if unknown:
        raise ValueError(f"IPI categories outside the four standard ones: {sorted(unknown)}")
    if not ids <= set(survival.patient_id):
        raise ValueError("Survival must cover every patient")


def figure_encoder_embeddings(coords, survival, *, model_order=None, model_labels=None, horizon=365,
                              grid_size=150, min_effective_support=25.0, mark=None):
    models = list(model_order or coords.model.drop_duplicates())
    _check(coords, survival, models)
    labels = model_labels or {m: m for m in models}
    base = coords.loc[coords.model == models[0]].set_index("patient_id")
    diseases = sorted(base.disease.dropna().unique(), key=str)
    palette = CATEGORICAL + DISEASE_EXTRA
    if len(diseases) > len(palette):
        raise ValueError("Too many disease groups for the palette")
    disease_colours = {d: palette[i] for i, d in enumerate(diseases)}
    surv = survival.set_index("patient_id")

    ncol = len(models)
    size = 1.95
    fig = plt.figure(figsize=(size * ncol + 1.6, size * 3 + 0.8))
    grid = fig.add_gridspec(3, ncol + 1, width_ratios=[1] * ncol + [0.7], left=0.03, right=0.985,
                            bottom=0.03, top=1 - 0.62 / fig.get_figheight(), hspace=0.14, wspace=0.05)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = LogNorm(vmin=0.25, vmax=4)
    image = None
    for mi, model in enumerate(models):
        data = coords.loc[coords.model == model].set_index("patient_id").sort_index()
        pts = data[["x", "y"]].to_numpy(float)
        pad = 0.05 * np.ptp(pts, axis=0)
        extent = (pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0], pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])
        frame = data.reset_index()

        # 1: disease groups
        ax = fig.add_subplot(grid[0, mi])
        _category_regions(ax, frame, "disease", disease_colours, extent)
        # 2: IPI, observed category per patient
        bx = fig.add_subplot(grid[1, mi])
        bx.scatter(pts[:, 0], pts[:, 1], s=1.6, color=CLOUD, linewidths=0, rasterized=True, zorder=1)
        for level, colour in zip(IPI_LEVELS, IPI_COLOURS):  # high risk drawn last, on top
            part = frame[frame.ipi_group == level]
            bx.scatter(part.x, part.y, s=3.4, color=colour, alpha=0.55, linewidths=0, rasterized=True, zorder=2)
        # Where the lowest and highest IPI categories concentrate: the densest half of each.
        _category_regions(bx, frame, "ipi_group", {"Low": IPI_COLOURS[0], "High": IPI_COLOURS[-1]}, extent)
        # 3: observed one-year overall-survival risk relative to average
        cx = fig.add_subplot(grid[2, mi])
        s = surv.loc[data.index]
        hd = horizon_ipcw_data(s.time_days.to_numpy(float), s.event.to_numpy(float), horizon)
        used = hd.eligible
        overall = np.sum(hd.weights[used] * hd.target[used]) / max(np.sum(hd.weights[used]), 1e-12)
        if used.sum() and overall > 0:
            bw = 1.4 * _default_bandwidth(pts)
            xx, yy, surface, support, density = _kernel_surfaces(
                pts[used], hd.target[used], hd.weights[used], bandwidth=bw, grid_size=grid_size,
                bounds_coords=pts, min_padding=0.0)
            rel = np.clip(surface / overall, 0.25, 4)
            rel[_embedding_surface_mask(xx, yy, pts[used], support, density, min_effective_support)] = np.nan
            image = cx.imshow(rel, origin="lower", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=cmap,
                              norm=norm, interpolation="bilinear", aspect="auto", zorder=2, rasterized=True)
        cx.scatter(pts[:, 0], pts[:, 1], s=0.8, color=INK2, alpha=0.18, linewidths=0, rasterized=True, zorder=3)
        for a in (ax, bx, cx):
            _map_axis(a)
            a.set_xlim(extent[:2])
            a.set_ylim(extent[2:])
        ax.set_title(f"{chr(65 + mi)}   " + textwrap.fill(str(labels[model]), 22), loc="left", fontsize=9.6,
                     fontweight="bold", pad=6, color=INK)

    # legends, one per row, in the right-hand column
    def legend_axis(row):
        lax = fig.add_subplot(grid[row, -1])
        lax.axis("off")
        return lax

    n = len(base)
    legend_axis(0).legend(handles=[Line2D([], [], marker="o", ls="", color=disease_colours[d], label=str(d), markersize=4.5)
                                   for d in diseases], loc="center left", frameon=False, fontsize=7,
                          title="Disease group", title_fontsize=7.6, handletextpad=0.3)
    scored = int(base.ipi_group.notna().sum())
    legend_axis(1).legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l, markersize=4.5)
                                   for l, c in zip(IPI_LEVELS, IPI_COLOURS)]
                          + [Line2D([], [], marker="o", ls="", color=CLOUD, label="No IPI", markersize=4.5)],
                          loc="center left", frameon=False, fontsize=7, handletextpad=0.3,
                          title=f"IPI category\n({scored:,} of {n:,} patients;\noutlines: densest half of\nLow and of High)", title_fontsize=7.6)
    for lax_ in fig.axes:
        leg = lax_.get_legend()
        if leg is not None:
            leg._legend_box.align = "left"  # left-aligned titles on older matplotlib
    lax = legend_axis(2)
    if image is not None:
        cax = lax.inset_axes([0.08, 0.18, 0.1, 0.6])
        cb = fig.colorbar(image, cax=cax)
        cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=["¼×", "½×", "average", "2×", "4×"])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.6)
        cb.outline.set_visible(False)
        cax.set_title(textwrap.fill(f"Observed death within {horizon // 365 if horizon % 365 == 0 else horizon} "
                                    f"{'year' if horizon == 365 else 'days'}, relative to average", 18),
                      fontsize=7.6, loc="left", pad=6)
    if mark:
        fig.text(0.995, 0.004, mark, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    return fig


def synthetic_inputs(seed=23, n=1500):
    """Invented patients for layout checks only (never results)."""
    rng = np.random.default_rng(seed)
    names = ["DLBCL-like", "CLL/SLL", "Myeloma", "Indolent B-NHL", "Hodgkin", "T-cell NHL"]
    disease = rng.choice(len(names), n, p=[0.3, 0.17, 0.2, 0.17, 0.08, 0.08])
    angle = 2 * np.pi * np.arange(len(names)) / len(names)
    centres = np.c_[np.cos(angle), np.sin(angle)] * 3.2
    age = np.clip(rng.normal(66, 12, n) - 8 * (disease == 4), 18, 95)
    score = np.clip(np.round((age - 50) / 12 + rng.normal(0, 1, n)), 0, 3).astype(int)
    ipi = np.array(IPI_LEVELS, dtype=object)[score]
    ipi[~np.isin(disease, [0, 3, 4, 5])] = None  # IPI only for lymphomas
    ipi[rng.random(n) < 0.25] = None
    hazard = 0.04 * np.exp(0.6 * score + 0.4 * (disease == 2))
    death = rng.exponential(365 / hazard)
    censor = rng.uniform(200, 1500, n)
    survival = pd.DataFrame({"patient_id": np.arange(n), "time_days": np.minimum(death, censor),
                             "event": (death <= censor).astype(int)})
    within = np.c_[(score - 1.5) / 1.2, rng.normal(0, 0.6, n)]
    frames = []
    for model, sep in {"Cohort encoder": 0.85, "National encoder": 0.8, "Adapted encoder": 0.95}.items():
        noise = rng.normal(size=(n, 2))
        xy = sep * centres[disease] + 0.9 * within + (1.4 - sep) * noise
        frames.append(pd.DataFrame({"model": model, "patient_id": np.arange(n), "x": xy[:, 0], "y": xy[:, 1],
                                    "disease": np.array(names)[disease], "ipi_group": ipi}))
    return pd.concat(frames, ignore_index=True), survival
