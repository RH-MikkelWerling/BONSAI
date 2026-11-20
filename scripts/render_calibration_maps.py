"""Local calibration maps (supp_calibration_maps) from the adapted joint model's outcome grids
(19 Nov): for the most common outcome of each clinical domain, observed (IPCW) minus predicted
one-year risk, in percentage points, in each square of the Fig. 5 map. A difference rather than a
ratio, so that common and rare outcomes are on one scale (a ratio cannot exceed 1/risk and swings
widely on a few events). Squares with fewer than five expected events (kernel-weighted patients nearby times predicted
risk) are left blank, and domains whose most common outcome is below 15% are not shown. Both grid layers
are relative to the outcome's observed average, so absolute risk = layer x observed risk.
Aggregate lattices only (squares with fewer than five patients suppressed on the server).

Usage:
  python -m scripts.render_calibration_maps scratch/download_outcome_grids_joint_20261119 \
      paper/figures/results_thesis_overlay
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from opera.run.paper_story_figures import thesis_style, _save, INK, INK2, MUTED
from opera.visualization import opera_design_figures as od
from opera.visualization.paper_figures import _tidy_surface_mask
from scripts.render_outcome_gallery import WIDEN, _crop, _surface, widen

# Blue where fewer events were observed than predicted, claret where more; a pale grey at 1 so that
# well-calibrated squares stay visible against the white page.
CMAP = LinearSegmentedColormap.from_list("calibration", ["#2B5C8A", "#E3E6EA", "#8E2C48"])
LIMIT = 15.0  # percentage points
MIN_EVENTS = 10.0  # squares with fewer expected events (under the wider kernel) are left blank
MIN_RISK = 0.0  # every domain; squares without enough expected events are blank


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
    chosen = (meta.sort_values("observed_risk", ascending=False).drop_duplicates("domain")
              .sort_values("d").reset_index(drop=True))
    skipped = chosen.loc[chosen.observed_risk < MIN_RISK, "domain"].tolist()
    chosen = chosen.loc[chosen.observed_risk >= MIN_RISK].reset_index(drop=True)
    print("domains too rare to map:", skipped)
    grids = {r.file: pd.read_csv(a.grids / r.file) for r in chosen.itertuples()}
    frame = _crop(list(grids.values()))
    norm = Normalize(vmin=-LIMIT, vmax=LIMIT)
    cols, W = 4, 6.9
    rows = int(np.ceil(len(chosen) / cols))
    panel, gap_x, head, row_h, foot = 1.5, 0.27, 0.1, 2.12, 0.62
    H = head + rows * row_h + foot
    fig = plt.figure(figsize=(W, H))
    left = (W - cols * panel - (cols - 1) * gap_x) / 2
    for i, r in enumerate(chosen.itertuples()):
        row, col = divmod(i, cols)
        x0, top = left + col * (panel + gap_x), head + row * row_h + 0.5
        grid = grids[r.file]
        count = _surface(grid, "count", frame)
        empty = _tidy_surface_mask(count <= 0)
        observed, covered = widen(count, _surface(grid, "mortality", frame) * r.observed_risk)
        predicted, _ = widen(count, _surface(grid, "predicted", frame) * r.observed_risk)
        diff = 100 * (observed - predicted)
        few = np.nan_to_num(covered * predicted) < MIN_EVENTS
        rgba = CMAP(norm(np.clip(np.nan_to_num(diff), -LIMIT, LIMIT)))
        rgba[..., 3] = np.where(empty | few | ~np.isfinite(diff), 0.0, 1.0)
        ax = fig.add_axes([x0 / W, 1 - (top + panel) / H, panel / W, panel / H])
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower", interpolation="nearest", aspect="equal")
        ax.set_axis_off()
        cx = (x0 + panel / 2) / W
        fig.text(cx, 1 - (top - 0.36) / H, od._short_domain(r.domain), fontsize=6.6, color=MUTED, style="italic",
                 ha="center", va="bottom")
        fig.text(cx, 1 - (top - 0.2) / H, r.label, fontsize=7.8, color=INK, fontweight="bold", ha="center",
                 va="bottom")
        fig.text(cx, 1 - (top - 0.05) / H,
                 f"observed {100 * r.observed_risk:.1f}%, predicted {100 * r.predicted_risk:.1f}%",
                 fontsize=6.3, color=INK2, ha="center", va="bottom")
    cax = fig.add_axes([0.3, 0.3 / H, 0.4, 0.07 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([-15, -10, -5, 0, 5, 10, 15], labels=["−15", "", "", "0", "", "", "+15"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=6.8, length=0)
    cb.outline.set_visible(False)
    cb.set_label("Observed − predicted one-year risk (percentage points)",
                 fontsize=7.2, color=INK2)
    fig.text(0.3, 0.42 / H, "fewer events than predicted", fontsize=6.3, color="#2B5C8A", style="italic",
             ha="left", va="bottom")
    fig.text(0.7, 0.42 / H, "more events than predicted", fontsize=6.3, color="#8E2C48", style="italic",
             ha="right", va="bottom")
    _save(fig, a.out / "supp_calibration_maps")


if __name__ == "__main__":
    main()
