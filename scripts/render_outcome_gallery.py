"""Outcome gallery (supp_embedding_gallery_1..6) from the server's outcome grids (19 Nov):
for each of the 69 outcomes, adapted OPERA's predicted and observed (IPCW) one-year risk on the
adapted OPERA map, both relative to that outcome's observed average, on the encoder maps' log
scale (1/4x to 4x). Aggregate lattices only (cells with fewer than five patients suppressed on the
server).

Usage:
  python -m scripts.render_outcome_gallery scratch/download_outcome_grids_20261119 \
      paper/figures/results_thesis_overlay
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from opera.run.paper_story_figures import thesis_style, _save, INK, INK2, MUTED
from opera.visualization import opera_design_figures as od
from opera.visualization.paper_figures import _tidy_surface_mask

PER_PAGE = 12
COLUMNS = 3  # outcome pairs per row


def _crop(grids, margin=0.03):
    """One square frame around the occupied cells of every outcome (same frame on every page)."""
    occ = pd.concat([g.loc[g.layer.eq("count"), ["ix", "iy"]] for g in grids])
    lo_x, hi_x, lo_y, hi_y = occ.ix.min(), occ.ix.max(), occ.iy.min(), occ.iy.max()
    side = int(max(hi_x - lo_x, hi_y - lo_y) * (1 + 2 * margin)) + 1
    return int((lo_x + hi_x) / 2 - side / 2), int((lo_y + hi_y) / 2 - side / 2), side


def _surface(grid, layer, frame):
    ox, oy, side = frame
    a = np.full((side, side), np.nan) if layer != "count" else np.zeros((side, side))
    part = grid.loc[grid.layer.eq(layer)]
    ix, iy = part.ix.to_numpy() - ox, part.iy.to_numpy() - oy
    keep = (ix >= 0) & (ix < side) & (iy >= 0) & (iy < side)
    a[ix[keep], iy[keep]] = part.value.to_numpy()[keep]
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grids", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    thesis_style()
    meta = pd.read_csv(a.grids / "outcome_grid_meta.csv")
    if not meta.status.eq("complete").all():
        raise ValueError("Incomplete outcome grids")
    domains = od._domain_order()
    meta["d"] = meta.domain.map({d: i for i, d in enumerate(domains)}).fillna(len(domains))
    meta = meta.sort_values(["d", "label"]).reset_index(drop=True)
    grids = {r.file: pd.read_csv(a.grids / r.file) for r in meta.itertuples()}
    frame = _crop(list(grids.values()))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = LogNorm(vmin=0.25, vmax=4)
    pages = [meta.iloc[i:i + PER_PAGE] for i in range(0, len(meta), PER_PAGE)]
    for number, page in enumerate(pages, start=1):
        rows = int(np.ceil(len(page) / COLUMNS))
        width, gap = 6.9, 0.18
        pair = (width - gap * (COLUMNS - 1)) / COLUMNS
        panel = pair / 2 - 0.03
        # As the earlier gallery: each clinical domain named above a rule spanning its outcomes in the
        # row (repeated where a domain continues), and each outcome's name and average centred above
        # its pair of maps.
        head, row_h, key_h = 0.05, panel + 0.86, 0.62
        height = head + rows * row_h + key_h
        fig = plt.figure(figsize=(width, height))
        image = None
        items = list(page.itertuples())
        for row in range(rows):
            members = items[row * COLUMNS:(row + 1) * COLUMNS]
            top_row = head + row * row_h
            start = 0
            while start < len(members):
                end = start
                while end + 1 < len(members) and members[end + 1].domain == members[start].domain:
                    end += 1
                x_left = start * (pair + gap) / width
                x_right = (end * (pair + gap) + pair) / width
                y = 1 - (top_row + 0.20) / height
                name = members[start].domain if end > start else od._short_domain(members[start].domain)
                fig.text(x_left, y + 0.03 / height, name, fontsize=8.2, fontweight="bold",
                         color=INK, ha="left", va="bottom")
                fig.add_artist(plt.Line2D([x_left, x_right], [y, y], transform=fig.transFigure,
                                          color=INK2, lw=0.7))
                start = end + 1
            for col, r in enumerate(members):
                x0 = col * (pair + gap)
                top = top_row + 0.68
                centre = (x0 + pair / 2) / width
                fig.text(centre, 1 - (top - 0.25) / height, r.label, fontsize=7.6, color=INK,
                         ha="center", va="bottom")
                fig.text(centre, 1 - (top - 0.09) / height, f"{100 * r.observed_risk:.1f}% at 365 days",
                         fontsize=6.6, color=INK2, ha="center", va="bottom")
                count = _surface(grids[r.file], "count", frame)
                empty = _tidy_surface_mask(count <= 0)
                for j, (layer, name) in enumerate((("predicted", "predicted"), ("mortality", "observed"))):
                    ax = fig.add_axes([(x0 + j * (panel + 0.06)) / width, 1 - (top + panel) / height,
                                       panel / width, panel / height])
                    values = _surface(grids[r.file], layer, frame)
                    values[empty] = np.nan
                    image = ax.imshow(np.clip(values.T, 0.25, 4), origin="lower", cmap=cmap, norm=norm,
                                      interpolation="nearest", aspect="equal")
                    ax.set_axis_off()
                    ax.text(0.5, -0.02, name, transform=ax.transAxes, ha="center", va="top",
                            fontsize=6.4, color=MUTED)
        cax = fig.add_axes([0.30, 0.35 / height, 0.40, 0.07 / height])
        cb = fig.colorbar(image, cax=cax, orientation="horizontal")
        cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=["¼×", "½×", "average", "2×", "4×"])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.8, length=0)
        cb.outline.set_visible(False)
        cb.set_label("One-year risk relative to the outcome's observed average", fontsize=7.2, color=INK2)
        _save(fig, a.out / f"supp_embedding_gallery_{number}")


if __name__ == "__main__":
    main()
