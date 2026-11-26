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
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D

from opera.visualization.censoring_aware_embeddings import (
    _default_bandwidth, _kernel_surfaces, horizon_ipcw_data)
from opera.visualization.paper_figures import _category_regions, _embedding_surface_mask, _map_axis
from opera.visualization.style import CATEGORICAL

# Risk group by each disease's own index (Table 1): the recorded category of FLIPI, CLL-IPI, R-ISS, MIPI,
# IPSS-WM, IPS, R-IPI, MALT-IPI and the IPI, and the NCCN-IPI for DLBCL, collapsed to three groups.
IPI_LEVELS = ["Low", "Intermediate", "High"]
IPI_COLOURS = ["#9CC9BE", "#D99A62", "#6E1A2E"]  # light to dark: low to high risk
DISEASE_EXTRA = ["#BD648E", "#697481"]
# Two palette slots read alike on the disease maps (19 Nov): MCL's eucalyptus against indolent B-NHL's
# teal (OKLab distance 4.6; 3.8 deuteranopic) and T-cell NHL's lavender against Hodgkin's blue (6.1;
# 3.1). Fixed replacements, at least 10 from every other disease colour, also under simulated CVD.
DISEASE_COLOUR_OVERRIDES = {"MCL": "#6E1A2E", "T-cell NHL": "#D9A52C"}
INK, INK2, MUTED, CLOUD = "#17212B", "#485463", "#657180", "#D5DAE1"


def neighbourhood_metrics(coords, survival, *, horizon=365, k=50, columns=("x", "y"), space="map"):
    """Descriptive summaries of each map, so differences between encoders can be checked, not
    only seen. For every patient, its k nearest neighbours on the two-dimensional map:

    * disease agreement: the share of neighbours with the patient's own disease group;
    * neighbourhood mortality AUC: how well the neighbours' observed death rate within the
      horizon ranks the patient's own death (patients with known status at the horizon).

    Computed on the plotted projection, so they describe the maps, not the full representation.
    """
    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score
    surv = survival.set_index("patient_id")
    rows = []
    for model, data in coords.groupby("model", sort=False):
        data = data.set_index("patient_id")
        pts = data[list(columns)].to_numpy(float)
        if pts.shape[1] <= 3:
            _, idx = cKDTree(pts).query(pts, k=min(k + 1, len(pts)))
        else:  # full representation: cosine neighbours, exact
            from sklearn.neighbors import NearestNeighbors
            _, idx = NearestNeighbors(n_neighbors=min(k + 1, len(pts)), metric="cosine",
                                      algorithm="brute").fit(pts).kneighbors(pts)
        idx = idx[:, 1:]  # leave the patient out
        disease = data.disease.astype(str).to_numpy()
        agree = float(np.mean(disease[idx] == disease[:, None]))
        share = pd.Series(disease).value_counts(normalize=True)
        chance = float((share ** 2).sum())  # agreement expected if neighbours were random
        s = surv.loc[data.index]
        known = (s.event.to_numpy() == 1) & (s.time_days.to_numpy() <= horizon) | (s.time_days.to_numpy() > horizon)
        died = ((s.event.to_numpy() == 1) & (s.time_days.to_numpy() <= horizon)).astype(float)
        died_known = np.where(known, died, np.nan)
        neigh = np.nanmean(died_known[idx], axis=1)
        ok = known & np.isfinite(neigh)
        auc = float(roc_auc_score(died[ok], neigh[ok])) if len(np.unique(died[ok])) == 2 else float("nan")
        rows.append(dict(model=model, space=space, patients=len(data), k=k, horizon_days=horizon,
                         disease_agreement=agree, disease_agreement_chance=chance, neighbourhood_mortality_auc=auc,
                         known_status=int(ok.sum()), deaths=int(died[ok].sum())))
    return pd.DataFrame(rows)


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
        raise ValueError(f"index risk groups outside Low/Intermediate/High: {sorted(unknown)}")
    if not ids <= set(survival.patient_id):
        raise ValueError("Survival must cover every patient")


def figure_encoder_embeddings(coords, survival, *, model_order=None, model_labels=None, horizon=365,
                              grid_size=150, min_effective_support=25.0, mark=None, metrics=None,
                              row_labels=True):
    models = list(model_order or coords.model.drop_duplicates())
    _check(coords, survival, models)
    labels = model_labels or {m: m for m in models}
    base = coords.loc[coords.model == models[0]].set_index("patient_id")
    diseases = sorted(base.disease.dropna().unique(), key=str)
    palette = CATEGORICAL + DISEASE_EXTRA
    if len(diseases) > len(palette):
        raise ValueError("Too many disease groups for the palette")
    disease_colours = {d: palette[i] for i, d in enumerate(diseases)}
    disease_colours.update({d: c for d, c in DISEASE_COLOUR_OVERRIDES.items() if d in disease_colours})
    surv = survival.set_index("patient_id")

    ncol = len(models)
    size = 1.95
    fig = plt.figure(figsize=(size * ncol + 1.6 + (0.35 if row_labels else 0), size * 3 + 0.8))
    grid = fig.add_gridspec(3, ncol + 1, width_ratios=[1] * ncol + [0.7],
                            left=0.03 + (0.06 if row_labels else 0), right=0.985,
                            bottom=0.03, top=1 - 0.62 / fig.get_figheight(), hspace=0.14, wspace=0.05)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = LogNorm(vmin=0.25, vmax=4)
    image = None
    for mi, model in enumerate(models):
        data = coords.loc[coords.model == model].set_index("patient_id").sort_index()
        pts = data[["x", "y"]].to_numpy(float)
        # A few outlying patients (small detached clusters in UMAP) would otherwise set the range
        # and squeeze the map; frame the central 99% and let the rest fall outside the panel.
        lo_q, hi_q = np.percentile(pts, [0.5, 99.5], axis=0)
        pad = 0.06 * (hi_q - lo_q)
        extent = (lo_q[0] - pad[0], hi_q[0] + pad[0], lo_q[1] - pad[1], hi_q[1] + pad[1])
        outside = int(((pts[:, 0] < extent[0]) | (pts[:, 0] > extent[1]) |
                       (pts[:, 1] < extent[2]) | (pts[:, 1] > extent[3])).sum())
        if metrics is not None:
            metrics.loc[metrics.model == model, "outside_panel"] = outside
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
        if metrics is not None and model in set(metrics.model):
            r = metrics.set_index("model").loc[model]
            for a, text in ((ax, f"same-disease neighbours {100 * r.disease_agreement:.0f}%"),
                            (cx, f"mortality from neighbours, AUC {r.neighbourhood_mortality_auc:.2f}")):
                a.text(0.02, 0.02, text, transform=a.transAxes, fontsize=6.6, color=INK2, zorder=5,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        if row_labels and mi == 0:
            for a, text in ((ax, "Disease group"), (bx, "IPI category"),
                            (cx, "Observed one-year\nmortality")):
                a.text(-0.06, 0.5, text, transform=a.transAxes, rotation=90, ha="right", va="center",
                       fontsize=8, color=INK2)

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
                          + [Line2D([], [], marker="o", ls="", color=CLOUD, label="No index", markersize=4.5)],
                          loc="center left", frameon=False, fontsize=7, handletextpad=0.3,
                          title=f"Index risk group\n({scored:,} of {n:,} patients;\noutlines: densest half of\nLow and of High)", title_fontsize=7.6)
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
    ipi = np.array(["Low", "Intermediate", "Intermediate", "High"], dtype=object)[score]
    ipi[disease == 5] = None  # T-cell NHL: no established index recorded
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


# -- Aggregate export: binned map grids, drawn locally ----------------------------------------
def align_maps(coords, *, reference=None):
    """Rotate, reflect, scale and shift each model's map onto the reference model's, fitted on
    the same patients (orthogonal Procrustes). UMAP orientation is arbitrary, so without this
    the columns cannot be compared by eye. The rotation is the Procrustes fit; the scale makes
    each map's spread equal to the reference's. Distances within a map are preserved up to scale."""
    from scipy.linalg import orthogonal_procrustes
    models = list(coords.model.drop_duplicates())
    ref = reference or models[-1]
    target = coords.loc[coords.model == ref].set_index("patient_id")[["x", "y"]]
    t = target.to_numpy(float)
    t_mean = t.mean(0)
    t0 = t - t_mean
    out = []
    for model, data in coords.groupby("model", sort=False):
        d = data.set_index("patient_id")
        a = d.loc[target.index, ["x", "y"]].to_numpy(float)
        a_mean = a.mean(0)
        a0 = a - a_mean
        rot, _ = orthogonal_procrustes(a0, t0)
        # Match the spread (root-mean-square distance from the centre), not the least-squares
        # Procrustes scale: that scale shrinks a map in proportion to how poorly it agrees with the
        # reference, so weakly aligned encoders looked small and over-smoothed (18 Nov).
        scale = np.sqrt(np.sum(t0 ** 2) / max(np.sum(a0 ** 2), 1e-12))
        xy = (d[["x", "y"]].to_numpy(float) - a_mean) @ rot * scale + t_mean
        out.append(d.assign(x=xy[:, 0], y=xy[:, 1]).reset_index())
    return pd.concat(out, ignore_index=True)


def map_grids(coords, survival, *, horizon=365, grid=400, min_count=5, bandwidth=0.013, min_support=25.0,
              reference=None, frame=None, risk_support="density", export_support=False):
    """Binned, aggregate version of each encoder map, safe to export from the server.

    The maps are first aligned to the ``reference`` model (default: the last) by rotation,
    reflection, uniform scaling and translation fitted on the same patients (orthogonal
    Procrustes), so every column has the same orientation, and share one square frame over the
    central 99% of patients. Per model, a ``grid`` x ``grid`` lattice over that frame. Smoothing is set in map units (``bandwidth``, a fraction of the map's range), not
    in cells, so the grid can be made as fine as wanted without changing what is smoothed or
    suppressed; a finer grid only gives smoother edges. Returns ``(cells, meta)``:

    * ``cells``: long table ``model, layer, key, ix, iy, value`` with smoothed patient densities
      (about the number of patients within one kernel width of the cell) in total (``count``), by
      disease group (``disease``) and by IPI category (``ipi``), suppressed wherever that
      neighbourhood holds fewer than ``min_count`` patients; and observed death within the horizon relative
      to the average (``mortality``), an IPCW estimate smoothed over neighbouring cells, kept only
      where the smoothed support reaches ``min_support`` patients; and, when ``coords`` has a
      ``predicted`` column (the column's own model's predicted one-year risk), the smoothed mean
      predicted risk on the same scale (``predicted``).
    * ``meta``: the extent of each lattice and the number of patients outside it.
    No patient-level values leave this function.
    """
    from scipy.ndimage import gaussian_filter
    if risk_support not in {'density', 'ipcw_ess'}:
        raise ValueError('Choose density or ipcw_ess risk support')
    surv = survival.set_index("patient_id")
    rows, meta = [], []
    aligned = align_maps(coords, reference=reference)
    allpts = aligned[["x", "y"]].to_numpy(float)
    lo, hi = np.percentile(allpts, [0.5, 99.5], axis=0)
    side = (hi - lo).max() * 1.12  # one square frame shared by every map
    mid = (lo + hi) / 2
    x0, x1, y0, y1 = mid[0] - side / 2, mid[0] + side / 2, mid[1] - side / 2, mid[1] + side / 2
    if frame is not None:
        x0, x1, y0, y1 = map(float, frame)
        if not np.isfinite(frame).all() or x1 <= x0 or y1 <= y0:
            raise ValueError('Invalid shared map frame')
    for model, data in aligned.groupby("model", sort=False):
        data = data.set_index("patient_id")
        pts = data[["x", "y"]].to_numpy(float)
        ix = np.floor((pts[:, 0] - x0) / (x1 - x0) * grid).astype(int)
        iy = np.floor((pts[:, 1] - y0) / (y1 - y0) * grid).astype(int)
        inside = (ix >= 0) & (ix < grid) & (iy >= 0) & (iy < grid)
        meta.append(dict(model=model, x0=x0, x1=x1, y0=y0, y1=y1, grid=grid, patients=len(data),
                         outside_panel=int((~inside).sum()), min_count=min_count))

        sigma = bandwidth * grid  # kernel width in cells, fixed in map units
        scale = 2 * np.pi * sigma ** 2  # smoothed density -> patients within about one kernel width

        def smoothed(mask):
            counts = np.zeros((grid, grid))
            np.add.at(counts, (ix[mask & inside], iy[mask & inside]), 1)
            return gaussian_filter(counts, sigma) * scale

        total = smoothed(np.ones(len(pts), bool))
        keep = total >= min_count  # neighbourhoods of fewer than min_count patients are suppressed

        def add(layer, key, values):
            for (a, b) in zip(*np.nonzero(keep & (values >= 0.05))):
                rows.append((model, layer, key, int(a), int(b), round(float(values[a, b]), 2)))

        add("count", "", total)
        for d in sorted(data.disease.dropna().astype(str).unique()):
            add("disease", d, smoothed((data.disease.astype(str) == d).to_numpy()))
        for level in IPI_LEVELS:
            # Nullable registry categories yield pandas BooleanArray masks.
            # Missing categories belong to no index group, but remain in total,
            # disease and risk layers. NumPy indexing requires native bool.
            mask = data.ipi_group.eq(level).fillna(False).to_numpy(dtype=bool)
            add("ipi", level, smoothed(mask))
        # Observed death within the horizon: IPCW-weighted, smoothed over cells.
        s = surv.loc[data.index]
        hd = horizon_ipcw_data(s.time_days.to_numpy(float), s.event.to_numpy(float), horizon)
        use = hd.eligible & inside
        num, den, n = (np.zeros((grid, grid)) for _ in range(3))
        np.add.at(num, (ix[use], iy[use]), hd.weights[use] * hd.target[use])
        np.add.at(den, (ix[use], iy[use]), hd.weights[use])
        np.add.at(n, (ix[inside], iy[inside]), 1)
        sigma_m = 2 * sigma  # a rate needs a wider kernel than a density
        num, den, n = (gaussian_filter(a, sigma_m) for a in (num, den, n))
        overall = np.sum(hd.weights[use] * hd.target[use]) / max(np.sum(hd.weights[use]), 1e-12)
        support = n * 2 * np.pi * sigma_m ** 2
        with np.errstate(invalid='ignore', divide='ignore'):
            density_rule_risk = num / den
            density_rule_shown = (support >= min_support) & keep & np.isfinite(density_rule_risk)
        if risk_support == 'ipcw_ess':
            from opera.visualization.kernel_risk_support import binned_ipcw_surface
            local_risk, support = binned_ipcw_surface(
                ix[use], iy[use], hd.target[use], hd.weights[use], grid=grid, sigma=sigma_m)
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                local_risk = num / den
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = local_risk / overall
        shown = (support >= min_support) & keep
        if risk_support == 'ipcw_ess':
            # Add information support without relaxing the previous density mask.
            shown &= density_rule_shown
            if export_support:
                # Coarse aggregate bands cover the full density footprint,
                # including regions whose observed risk is not supported.
                bands = np.digitize(support, [5, 10, 20, 25, 50, 100])
                for a, b in zip(*np.nonzero(keep)):
                    rows.append((model, 'support_band', '', int(a), int(b), int(bands[a,b])))
            from opera.visualization.kernel_risk_support import support_coverage
            meta[-1].update(support_coverage(support, keep, density_rule_shown,
                                           ix[inside], iy[inside], minimum_count=min_count))
            meta[-1].update(risk_support_method='ipcw_kernel_kish_ess_v1', min_effective_support=min_support,
                           density_grid_cells=int(keep.sum()), supported_risk_grid_cells=int(shown.sum()),
                           informative_patients=int(use.sum()) if use.sum() >= min_count else np.nan,
                           early_censored_patients=int((inside & ~hd.eligible).sum())
                           if (inside & ~hd.eligible).sum() >= min_count else np.nan,
                           supported_max_risk=float(np.nanmax(local_risk[shown])) if shown.any() else np.nan,
                           supported_extreme_grid_cells=int((shown & (rel >= 4)).sum()),
                           density_rule_max_risk=float(np.nanmax(density_rule_risk[density_rule_shown]))
                           if density_rule_shown.any() else np.nan,
                           density_rule_extreme_grid_cells=int((density_rule_shown & (density_rule_risk >= 4*overall)).sum()),
                           max_effective_support_in_density_extremes=float(np.max(support[density_rule_shown & (density_rule_risk >= 4*overall)]))
                           if (density_rule_shown & (density_rule_risk >= 4*overall)).any() else np.nan)
        meta[-1]["observed_risk"] = round(float(overall), 4)  # aggregate: the average shown in titles
        for (a, b) in zip(*np.nonzero(shown & np.isfinite(rel))):
            rows.append((model, "mortality", "", int(a), int(b), round(float(rel[a, b]), 4)))
        # The column's own model: mean predicted risk, on the same smoothing, mask and scale (divided
        # by the same observed average), so a well-calibrated model shows the same map in both rows.
        if "predicted" in data and data.predicted.notna().any():
            p = data.predicted.to_numpy(float)
            ok = inside & np.isfinite(p)
            ps, pn = np.zeros((grid, grid)), np.zeros((grid, grid))
            np.add.at(ps, (ix[ok], iy[ok]), p[ok])
            np.add.at(pn, (ix[ok], iy[ok]), 1)
            meta[-1]["predicted_risk"] = round(float(p[ok].mean()), 4)
            boundary = 'constant' if risk_support == 'ipcw_ess' else 'reflect'
            ps, pn = gaussian_filter(ps, sigma_m, mode=boundary), gaussian_filter(pn, sigma_m, mode=boundary)
            with np.errstate(invalid="ignore", divide="ignore"):
                prel = (ps / pn) / overall
            for (a, b) in zip(*np.nonzero(shown & np.isfinite(prel))):
                rows.append((model, "predicted", "", int(a), int(b), round(float(prel[a, b]), 4)))
    cells = pd.DataFrame(rows, columns=["model", "layer", "key", "ix", "iy", "value"])
    return cells, pd.DataFrame(meta)


def figure_encoder_maps_from_grids(cells, meta, *, model_order=None, model_labels=None, metrics=None,
                                   horizon=365, smooth=1.2):
    """The encoder-map figure drawn from ``map_grids`` aggregates (no patient-level data)."""
    from scipy.ndimage import gaussian_filter
    # Layers without a key (count, mortality, predicted) read back from CSV as NaN; match them as "".
    cells = cells.assign(key=cells["key"].fillna("").astype(str))
    models = list(model_order or meta.model)
    labels = model_labels or {m: m for m in models}
    diseases = sorted(cells.loc[cells.layer == "disease", "key"].unique(), key=str)
    palette = CATEGORICAL + DISEASE_EXTRA
    disease_colours = {d: palette[i % len(palette)] for i, d in enumerate(diseases)}
    disease_colours.update({d: c for d, c in DISEASE_COLOUR_OVERRIDES.items() if d in disease_colours})
    ncol = len(models)
    fig = plt.figure(figsize=(1.95 * ncol + 1.95, 1.95 * 3 + 0.8))
    grid_spec = fig.add_gridspec(3, ncol + 1, width_ratios=[1] * ncol + [0.7], left=0.09, right=0.985,
                                 bottom=0.03, top=1 - 0.62 / fig.get_figheight(), hspace=0.14, wspace=0.05)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = LogNorm(vmin=0.25, vmax=4)
    image = None

    def lattice(model, layer, key=""):
        m = meta.set_index("model").loc[model]
        g = int(m.grid)
        a = np.zeros((g, g))
        part = cells[(cells.model == model) & (cells.layer == layer) & (cells.key == key)]
        a[part.ix.to_numpy(), part.iy.to_numpy()] = part.value.to_numpy()
        return a, (m.x0, m.x1, m.y0, m.y1)

    def densest_half(ax, counts, extent, colour, lw=1.1):
        dens = counts
        if dens.max() <= 0:
            return
        flat = np.sort(dens.ravel())[::-1]
        cum = np.cumsum(flat) / flat.sum()
        level = flat[np.searchsorted(cum, 0.5)]
        ax.contour(dens.T, levels=[level], extent=extent, colors=[colour], linewidths=lw, origin="lower")

    def grey_density(ax, model):
        total, extent = lattice(model, "count")
        dens = total
        ax.imshow(np.where(dens.T > 0.05, dens.T, np.nan), origin="lower", extent=extent, cmap="Greys",
                  vmin=0, vmax=np.nanpercentile(dens[dens > 0], 99) * 2.5, aspect="auto", zorder=1,
                  interpolation="bilinear")
        return extent

    for mi, model in enumerate(models):
        ax = fig.add_subplot(grid_spec[0, mi])
        total, extent = lattice(model, "count")
        stack = np.stack([lattice(model, "disease", d)[0] for d in diseases])
        lead = stack.argmax(0)
        purity = stack.max(0) / np.maximum(stack.sum(0), 1e-12)
        rgb = np.array([to_rgb(disease_colours[d]) for d in diseases])[lead]
        # colour: the most common disease; strength: how much it dominates and how many patients
        presence = np.clip(total / np.nanpercentile(total[total > 0], 60), 0, 1)
        alpha = np.where(total > 0, (0.25 + 0.75 * purity) * presence, 0)
        ax.imshow(np.dstack([rgb, alpha]).transpose(1, 0, 2), origin="lower", extent=extent, aspect="auto",
                  interpolation="bilinear", zorder=2)
        bx = fig.add_subplot(grid_spec[1, mi])
        grey_density(bx, model)
        for level, colour in zip(IPI_LEVELS, IPI_COLOURS):
            counts, _ = lattice(model, "ipi", level)
            if level in ("Low", "High"):
                densest_half(bx, counts, extent, colour, lw=1.4)
        # mean IPI category per cell, where the IPI is known
        stack = np.stack([lattice(model, "ipi", l)[0] for l in IPI_LEVELS])
        total = stack.sum(0)
        with np.errstate(invalid="ignore"):
            mean = (stack * np.arange(len(IPI_LEVELS))[:, None, None]).sum(0) / total
        mean[total < 0.5] = np.nan
        from matplotlib.colors import LinearSegmentedColormap
        bx.imshow(mean.T, origin="lower", extent=extent, aspect="auto", vmin=0, vmax=3, alpha=0.85, zorder=2,
                  cmap=LinearSegmentedColormap.from_list("ipi", IPI_COLOURS), interpolation="bilinear")
        cx = fig.add_subplot(grid_spec[2, mi])
        rel, _ = lattice(model, "mortality")
        rel[rel == 0] = np.nan
        image = cx.imshow(np.clip(rel.T, 0.25, 4), origin="lower", extent=extent, cmap=cmap, norm=norm,
                          aspect="auto", interpolation="bilinear", zorder=2)
        for a in (ax, bx, cx):
            _map_axis(a)
            a.set_xlim(extent[:2])
            a.set_ylim(extent[2:])
        ax.set_title(f"{chr(65 + mi)}   " + textwrap.fill(str(labels.get(model, model)), 22), loc="left",
                     fontsize=9.6, fontweight="bold", pad=6, color=INK)
        if metrics is not None and model in set(metrics.model):
            r = metrics.set_index("model").loc[model]
            for a, text in ((ax, f"same-disease neighbours {100 * r.disease_agreement:.0f}%"),
                            (cx, f"mortality from neighbours, AUC {r.neighbourhood_mortality_auc:.2f}")):
                a.text(0.02, 0.02, text, transform=a.transAxes, fontsize=6.6, color=INK2, zorder=5,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        if mi == 0:
            for a, text in ((ax, "Disease group"), (bx, "IPI category"), (cx, "Observed one-year\nmortality")):
                a.text(-0.06, 0.5, text, transform=a.transAxes, rotation=90, ha="right", va="center",
                       fontsize=8, color=INK2)

    def legend_axis(row):
        lax = fig.add_subplot(grid_spec[row, -1])
        lax.axis("off")
        return lax

    legend_axis(0).legend(handles=[Line2D([], [], marker="s", ls="", color=disease_colours[d], markersize=5,
                                          label=str(d)) for d in diseases],
                          loc="center left", frameon=False, fontsize=7, handletextpad=0.3,
                          title="Most common disease\n(faded where mixed)", title_fontsize=7.6)
    legend_axis(1).legend(handles=[Line2D([], [], marker="s", ls="", color=c, label=l, markersize=5)
                                   for l, c in zip(IPI_LEVELS, IPI_COLOURS)],
                          loc="center left", frameon=False, fontsize=7, handletextpad=0.3,
                          title="Mean index risk group\n(outlines: densest half\nof Low and of High)", title_fontsize=7.6)
    for lax_ in fig.axes:
        leg = lax_.get_legend()
        if leg is not None:
            leg._legend_box.align = "left"
    lax = legend_axis(2)
    if image is not None:
        cax = lax.inset_axes([0.08, 0.18, 0.1, 0.6])
        cb = fig.colorbar(image, cax=cax)
        cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=["¼×", "½×", "average", "2×", "4×"])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.6)
        cb.outline.set_visible(False)
        cax.set_title(textwrap.fill("Observed death within 1 year, relative to average", 18), fontsize=7.6,
                      loc="left", pad=6)
    return fig


# The paper's diverging scale (paper_figures.GAIN_CMAP), oriented for risk: lower = indigo, higher = sienna.
RISK_CMAP_COLOURS = ["#2D3A8C", "#AEB8DA", "#F7F7F5", "#E7B9A5", "#B5451B"]
ROW_TITLES = {"disease": "Disease group", "ipi": "Risk group by the disease's own index", "predicted": "Predicted\none-year mortality",
              "mortality": "Observed\none-year mortality"}
# Encoder identity colours as in the other figures (paper_figures.ORIGIN_STYLE: L, E, EA).
ENCODER_COLOURS = {"cohort": "#737D8C", "national": "#5C7AEA", "adapted": "#2D3A8C"}
_INK, _INK2, _MUTED = "#17212B", "#485463", "#657180"


def _encoder_colour(label):
    low = str(label).lower()
    for key in ("adapted", "national", "cohort"):
        if key in low:
            return ENCODER_COLOURS[key]
    return _INK2


def _side_legend(fig, gs_cell, title):
    """A slim key in the right-hand column, titled like the figures' legends."""
    holder = fig.add_subplot(gs_cell)
    holder.axis("off")
    bar = holder.inset_axes([0.12, 0.12, 0.22, 0.62])
    holder.text(0.0, 0.84, title, transform=holder.transAxes, fontsize=7.8, color=_INK2, va="bottom",
                ha="left", linespacing=1.1)
    return bar


def _blocks(a, factor, how="sum"):
    """Coarsen a lattice by ``factor`` (display pixels, as in the outcome gallery)."""
    g = a.shape[0] // factor * factor
    b = a[:g, :g].reshape(g // factor, factor, g // factor, factor)
    if how == "sum":
        return b.sum((1, 3))
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, (1, 3))


MIXED_COLOUR = "#E3E6EA"  # cells where no disease group dominates; lighter than "Other"


def figure_encoder_maps_v2(cells, meta, *, model_order=None, model_labels=None, metrics=None,
                           rows=("disease", "ipi", "predicted", "mortality"), width=6.9, horizon=365,
                           pixels=800, other_below=0.02, row_rules=False, panel_notes=False,
                           row_titles=None, risk_label=None, mixed_below=0.4):
    """Encoder maps from ``map_grids`` aggregates, drawn like the outcome gallery (Figures S24-S29).

    One column per encoder (maps aligned to the adapted encoder's, one shared frame) and one row per
    layer, each headed by a labelled rule as the gallery heads its clinical domains. The lattice is
    shown as ``pixels`` x ``pixels`` square cells without interpolation. Disease: the most common
    disease in each cell, or neutral grey ("mixed") where no group reaches ``mixed_below`` of the
    cell's patients. IPI: the mean IPI category where one is recorded.
    Predicted and observed one-year mortality: relative to the observed average, on the gallery's
    plasma scale (log, 1/4x to 4x). Call after ``paper_story_figures.thesis_style()``.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    # Layers without a key (count, mortality, predicted) read back from CSV as NaN; match them as "".
    cells = cells.assign(key=cells["key"].fillna("").astype(str))
    models = list(model_order or meta.model)
    labels = model_labels or {m: m for m in models}
    # At most eight disease colours: groups under ``other_below`` of patients (hairy cell leukaemia,
    # amyloidosis) are drawn together as a grey "Other"; their densities are summed, nothing is lost.
    sizes = cells[cells.layer == "disease"].groupby("key").value.sum()
    sizes = sizes / sizes.sum()
    small = set(sizes[sizes < other_below].index)
    if len(sizes) - len(small) > 8:
        small |= set(sizes.sort_values().index[:len(sizes) - 8])
    if small:
        cells = cells.copy()
        hit = (cells.layer == "disease") & cells.key.isin(small)
        cells.loc[hit, "key"] = "Other"
        cells = cells.groupby(["model", "layer", "key", "ix", "iy"], as_index=False).value.sum()
    # Colour follows the disease, not its rank in this figure: a fixed order (cohort sizes in the joint
    # family's maps) so every encoder figure gives each disease the same colour.
    fixed = ["Myeloma", "DLBCL-like", "Indolent B-NHL", "CLL/SLL", "Hodgkin", "T-cell NHL", "MCL", "BL/LBL"]
    ranked = [d for d in sizes.sort_values(ascending=False).index if d not in small]
    diseases = [d for d in fixed if d in ranked] + [d for d in ranked if d not in fixed] + (["Other"] if small else [])
    palette = [c for c in CATEGORICAL if c != "#9E9E9E"] + DISEASE_EXTRA[:1]  # grey is kept for Other
    disease_colours = {d: palette[i % len(palette)] for i, d in enumerate(d for d in diseases if d != "Other")}
    disease_colours.update({d: c for d, c in DISEASE_COLOUR_OVERRIDES.items() if d in disease_colours})
    disease_colours["Other"] = "#B8BEC6"
    risk_cmap = plt.get_cmap("plasma").copy()
    risk_cmap.set_bad((1, 1, 1, 0))
    ipi_cmap = LinearSegmentedColormap.from_list("ipi", IPI_COLOURS)
    ipi_cmap.set_bad((1, 1, 1, 0))
    norm = LogNorm(vmin=0.25, vmax=4)
    rows = [r for r in rows if r != "ipi" or (cells.layer == "ipi").any()]
    ncol, nrow = len(models), len(rows)
    m_meta = meta.set_index("model")
    metric_rows = metrics.set_index("model") if metrics is not None else None
    left, right, wspace = 0.02, 0.98, 0.05
    panel = width * (right - left) / (ncol + wspace * (ncol - 1))
    # Vertical layout in inches from the top: names, then per row a header, the maps and, under the
    # disease and IPI rows, their own key; the shared mortality key goes under the last row.
    top_names, header, sub_h, key_row, last_key = 0.20, 0.27, 0.17, 0.40, 0.62
    slots, cursor = {}, top_names
    # Rows with a note under each map heading (average risk, "grey: no index recorded"). Off by default:
    # those facts belong in the caption (Mikkel, 18 Nov).
    has_sub = {r: (panel_notes and r != "disease") for r in rows}
    for layer in rows:
        cursor += header + (sub_h if has_sub.get(layer, True) else 0.0)
        slots[layer] = cursor
        cursor += panel + (key_row if layer in ("disease", "ipi") else 0.0)
        if layer == "support_band":
            cursor += last_key
        if layer in ("predicted", "mortality") and layer == [r for r in rows if r in ("predicted", "mortality")][-1] and "support_band" in rows:
            cursor += last_key
    risk_rows = [l for l in rows if l in ("predicted", "mortality")]
    if risk_rows and "support_band" not in rows:
        cursor += last_key
    fig = plt.figure(figsize=(width, cursor + 0.05))
    H = fig.get_figheight()
    pw = panel / width

    def place(ri, mi):
        top = slots[rows[ri]]
        return fig.add_axes([left + mi * pw * (1 + wspace), 1 - (top + panel) / H, pw, panel / H])

    def lattice(model, layer, key=""):
        g = int(m_meta.loc[model, "grid"])
        a = np.full((g, g), np.nan) if layer in ("mortality", "predicted", "support_band") else np.zeros((g, g))
        part = cells[(cells.model == model) & (cells.layer == layer) & (cells.key == key)]
        a[part.ix.to_numpy(), part.iy.to_numpy()] = part.value.to_numpy()
        return a

    def pct(model, column):
        v = m_meta.loc[model][column] if column in m_meta else np.nan
        return f"{100 * v:.1f}%" if np.isfinite(v) else None

    images, axes = {}, {}
    for mi, model in enumerate(models):
        r = m_meta.loc[model]
        extent = (r.x0, r.x1, r.y0, r.y1)
        factor = max(1, int(round(int(r.grid) / pixels)))  # whole cells: 400 / 3 = 133 across
        total = _blocks(lattice(model, "count"), factor) / factor ** 2
        # No specks or pinholes: drop isolated cells, fill small gaps inside the patient area.
        from opera.visualization.paper_figures import _tidy_surface_mask
        empty = _tidy_surface_mask(total <= 0)
        for ri, layer in enumerate(rows):
            ax = place(ri, mi)
            axes[ri, mi] = ax
            sub = None
            if layer == "disease":
                stack = np.stack([_blocks(lattice(model, "disease", d), factor) for d in diseases])
                lead = stack.argmax(0)
                purity = stack.max(0) / np.maximum(stack.sum(0), 1e-12)
                rgb = np.array([to_rgb(disease_colours[d]) for d in diseases])[lead]
                # Mixed cells are grey, not a paler disease colour: pale myeloma navy read as Hodgkin
                # (19 Nov), so a tint could pass for another group.
                rgb = np.where((purity < mixed_below)[..., None], np.array(to_rgb(MIXED_COLOUR)), rgb)
                alpha = np.where(empty, 0.0, 1.0)
                ax.imshow(np.dstack([rgb, alpha]).transpose(1, 0, 2), origin="lower", extent=extent,
                          aspect="equal", interpolation="nearest", zorder=1)
                if metric_rows is not None and model in metric_rows.index:
                    sub = f"same-disease neighbours {100 * metric_rows.loc[model].disease_agreement:.0f}%"
            elif layer == "ipi":
                stack = np.stack([_blocks(lattice(model, "ipi", l), factor) / factor ** 2 for l in IPI_LEVELS])
                tot = stack.sum(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = (stack * np.arange(len(IPI_LEVELS))[:, None, None]).sum(0) / tot
                    share = tot / total
                recorded = (tot >= 3) & (share >= 0.2) & ~empty
                recorded = ~_tidy_surface_mask(~recorded, min_fraction=0.03)
                mean[~recorded] = np.nan
                ax.imshow(np.where(empty, 0, 1).T * 0 + 1, origin="lower", extent=extent, aspect="equal",
                          cmap=LinearSegmentedColormap.from_list("bg", ["#ECEEF1", "#ECEEF1"]),
                          alpha=np.where(empty, 0, 1).T.astype(float), interpolation="nearest", zorder=0)
                images["ipi"] = ax.imshow(mean.T, origin="lower", extent=extent, aspect="equal", vmin=0,
                                          vmax=len(IPI_LEVELS) - 1, cmap=ipi_cmap, interpolation="nearest", zorder=1)
                sub = "grey: no index recorded"
            elif layer == "support_band":
                from matplotlib.colors import BoundaryNorm
                support = _blocks(lattice(model, layer), factor, how="mean")
                support[empty] = np.nan
                cmap = plt.get_cmap("YlGnBu", 7).copy()
                cmap.set_bad((1, 1, 1, 0))
                images["support"] = ax.imshow(support.T, origin="lower", extent=extent,
                    cmap=cmap, norm=BoundaryNorm(np.arange(8) - .5, 7),
                    aspect="equal", interpolation="nearest", zorder=1)
            else:
                rel = _blocks(lattice(model, layer), factor, how="mean")
                rel[empty] = np.nan
                images["risk"] = ax.imshow(np.clip(rel.T, 0.25, 4), origin="lower", extent=extent, cmap=risk_cmap,
                                           norm=norm, aspect="equal", interpolation="nearest", zorder=1)
                avg = pct(model, "predicted_risk" if layer == "predicted" else "observed_risk")
                sub = f"{avg} at {horizon} days" if avg else None
                if layer == "mortality" and metric_rows is not None and model in metric_rows.index:
                    sub = (sub + "; " if sub else "") + \
                        f"neighbourhood AUC {metric_rows.loc[model].neighbourhood_mortality_auc:.2f}"
            _map_axis(ax)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            if sub and panel_notes:
                ax.set_title(sub, fontsize=7.2, pad=2, color=_INK2, loc="center")
    fig.canvas.draw()
    # Encoder names once, above the columns.
    for mi, model in enumerate(models):
        pos = axes[0, mi].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 1 - 0.06 / H, str(labels.get(model, model)), ha="center", va="top",
                 fontsize=9.0, fontweight="bold", color=_INK)
    # Row headings in the paper's panel style ("A   Disease group"); the comparison runs across the
    # columns, so no rule cuts between rows. ``row_rules=True`` restores the gallery's labelled rules.
    for ri, layer in enumerate(rows):
        first, last = axes[ri, 0].get_position(), axes[ri, ncol - 1].get_position()
        y = first.y1 + ((sub_h if has_sub.get(layer, True) else 0.0) + 0.04) / H
        name = (row_titles or {}).get(layer, "Effective follow-up support" if layer == "support_band"
                                     else ROW_TITLES.get(layer, layer)).replace("\n", " ")
        if row_rules:
            fig.add_artist(Line2D([first.x0, last.x1], [y, y], transform=fig.transFigure, color=_INK2, lw=0.6))
            fig.text(first.x0, y + 0.03 / H, name, fontsize=8.0, fontweight="bold", color=_INK, va="bottom",
                     ha="left")
        else:
            fig.text(first.x0, y - 0.02 / H, f"{chr(65 + ri)}   {name}", fontsize=9.6, fontweight="bold",
                     color=_INK, va="bottom", ha="left")
    # Each key sits under the row it explains; the mortality key, shared by both risk rows, at the bottom.
    def key_axes(y_top):
        return fig.add_axes([0.30, 1 - (y_top + 0.17) / H, 0.40, 0.07 / H])

    if "disease" in rows:
        y = slots["disease"] + panel + 0.06
        handles = [Patch(fc=disease_colours[d], ec="none", label=str(d)) for d in diseases]
        handles.append(Patch(fc=MIXED_COLOUR, ec="none", label=f"Mixed (none ≥{100 * mixed_below:.0f}%)"))
        fig.legend(handles=handles,
                   loc="upper center", bbox_to_anchor=(0.5, 1 - y / H),
                   ncol=len(handles) if len(handles) <= 6 else int(np.ceil(len(handles) / 2)), frameon=False,
                   fontsize=7.2, handlelength=1.0, handleheight=0.8, columnspacing=1.2, borderaxespad=0)
    for key in ("ipi", "risk", "support"):
        if key not in images:
            continue
        y = (slots["ipi"] + panel - 0.02 if key == "ipi" else
             slots["support_band"] + panel + 0.02 if key == "support" else
             slots[risk_rows[-1]] + panel + 0.02)
        cb = fig.colorbar(images[key], cax=key_axes(y), orientation="horizontal")
        if key == "risk":
            cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=["¼×", "½×", "average", "2×", "4×"])
            cb.set_label(risk_label or "One-year mortality relative to the observed average (both rows above)", fontsize=7.2,
                         color=_INK2, labelpad=2)
        elif key == "support":
            cb.set_ticks(range(7), labels=["<5", "5–9", "10–19", "20–24", "25–49", "50–99", "≥100"])
            cb.set_label("Effective sample size for observed one-year mortality", fontsize=7.2,
                         color=_INK2, labelpad=2)
        else:
            cb.set_ticks(range(len(IPI_LEVELS)), labels=IPI_LEVELS)
            cb.set_label("Mean risk group (FLIPI, CLL-IPI, R-ISS, NCCN-IPI, ...)", fontsize=7.2, color=_INK2,
                         labelpad=2)
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.8, length=0, colors=_INK2, pad=2)
        cb.outline.set_visible(False)
    return fig


DENSITY_COLOURS = ["#EEF0F8", "#AEB8DA", "#5C6BB5", "#2D3A8C", "#141C4F"]  # one hue, light to dark


def figure_encoder_disease_grid(cells, meta, *, model_order=None, model_labels=None, width=6.9, pixels=800,
                                blocks=2):
    """Where each disease's patients sit under each encoder: small multiples from ``map_grids``.

    One panel per disease and encoder. Colour is the share of that disease's patients in each square
    (its smoothed density divided by its total), on one scale per disease shared by its encoders, so
    small diseases are as visible as large ones and the encoders can be compared within a disease.
    The light grey area is where any patients are. Diseases run largest first, in ``blocks`` side-by-side
    columns of encoder triplets, each headed by a labelled rule as in the outcome gallery.
    """
    from matplotlib.colors import LinearSegmentedColormap
    # Layers without a key (count, mortality, predicted) read back from CSV as NaN; match them as "".
    cells = cells.assign(key=cells["key"].fillna("").astype(str))
    models = list(model_order or meta.model)
    labels = model_labels or {m: m for m in models}
    m_meta = meta.set_index("model")
    dis = cells[cells.layer == "disease"]
    diseases = list(dis.groupby("key").value.sum().sort_values(ascending=False).index)
    cmap = LinearSegmentedColormap.from_list("density", DENSITY_COLOURS)
    cmap.set_bad((1, 1, 1, 0))
    ncol = len(models) * blocks
    nrow = int(np.ceil(len(diseases) / blocks))
    left, right, wspace, gap = 0.02, 0.98, 0.04, 0.35  # gap: extra space between blocks, in panels
    panel = width * (right - left) / (ncol + wspace * (ncol - 1) + gap * (blocks - 1))
    top_names, header, bar_h = 0.24, 0.26, 0.55
    fig = plt.figure(figsize=(width, top_names + nrow * (header + panel) + bar_h))
    H = fig.get_figheight()
    pw = panel / width

    def lattice(model, layer, key=""):
        g = int(m_meta.loc[model, "grid"])
        a = np.zeros((g, g))
        part = cells[(cells.model == model) & (cells.layer == layer) & (cells.key == key)]
        a[part.ix.to_numpy(), part.iy.to_numpy()] = part.value.to_numpy()
        return a

    from opera.visualization.paper_figures import _tidy_surface_mask
    frames = {}
    for model in models:
        r = m_meta.loc[model]
        factor = max(1, int(round(int(r.grid) / pixels)))
        total = _blocks(lattice(model, "count"), factor)
        frames[model] = (factor, (r.x0, r.x1, r.y0, r.y1), _tidy_surface_mask(total <= 0))
    image = None
    for di, disease in enumerate(diseases):
        block, row = divmod(di, nrow)
        shares = {}
        for model in models:
            factor, _, empty = frames[model]
            d = _blocks(lattice(model, "disease", disease), factor)
            share = d / max(d.sum(), 1e-12)
            share[empty | (d <= 0)] = np.nan
            shares[model] = share
        top = np.nanpercentile(np.concatenate([s[np.isfinite(s)] for s in shares.values()] or [[1.0]]), 99)
        top = top if np.isfinite(top) and top > 0 else 1.0
        y_top = top_names + row * (header + panel) + header
        x_block = left + block * (len(models) * pw * (1 + wspace) - pw * wspace + gap * pw + pw * wspace)
        for mi, model in enumerate(models):
            factor, extent, empty = frames[model]
            ax = fig.add_axes([x_block + mi * pw * (1 + wspace), 1 - (y_top + panel) / H, pw, panel / H])
            ax.imshow(np.where(empty, np.nan, 1.0).T, origin="lower", extent=extent, aspect="equal",
                      cmap=LinearSegmentedColormap.from_list("bg", ["#E6E9ED", "#E6E9ED"]),
                      interpolation="nearest", zorder=0)
            image = ax.imshow(np.clip(shares[model].T / top, 0, 1), origin="lower", extent=extent,
                              aspect="equal", cmap=cmap, vmin=0, vmax=1, interpolation="nearest", zorder=1)
            _map_axis(ax)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            if mi == 0:
                x1 = x_block + len(models) * pw * (1 + wspace) - pw * wspace
                y = 1 - (y_top - 0.05) / H
                fig.add_artist(Line2D([x_block, x1], [y, y], transform=fig.transFigure, color=_INK2, lw=0.6))
                fig.text(x_block, y + 0.03 / H, str(disease), fontsize=7.8, fontweight="bold", color=_INK,
                         va="bottom", ha="left")
        if row == 0:
            for mi, model in enumerate(models):
                fig.text(x_block + (mi + 0.5) * pw * (1 + wspace) - 0.5 * pw * wspace, 1 - 0.05 / H,
                         str(labels.get(model, model)).replace(" encoder", "\nencoder"), ha="center", va="top",
                         fontsize=7.6, fontweight="bold", color=_INK, linespacing=1.1)
    if image is not None:
        bar = fig.add_axes([0.30, 0.28 / H, 0.40, 0.07 / H])
        cb = fig.colorbar(image, cax=bar, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1], labels=["none", "", "most"])
        cb.set_label("Share of the disease's patients in each square (scaled within each disease)", fontsize=7.2,
                     color=_INK2, labelpad=2)
        cb.ax.tick_params(labelsize=6.8, length=0, colors=_INK2, pad=2)
        cb.outline.set_visible(False)
    return fig
