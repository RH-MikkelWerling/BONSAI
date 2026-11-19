"""Patient-map figure (fig_patient_map) on the adapted joint model's map, from aggregate grids only:
A, disease groups (as Fig. 5: most common group per square, grey where none has 40%), with each
group's centre numbered; B-F, predicted one-year risk of five outcomes from different domains,
relative to each outcome's observed average; G, local calibration for overall survival (observed
IPCW risk divided by predicted risk).

Usage:
  python -m scripts.render_patient_map scratch/download_story_20261118c/tables \
      scratch/download_outcome_grids_joint_20261119 paper/figures/results_thesis_overlay
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm, to_rgb
from matplotlib.patches import Patch

from opera.run.paper_story_figures import thesis_style, _save, INK, INK2, MUTED
from opera.visualization import encoder_embedding_comparison as e
from opera.visualization.paper_figures import _tidy_surface_mask

MODEL = "Adapted joint"
OUTCOMES = [("serious_infection_composite", "B"), ("neutropenia_g3plus", "C"),
            ("aki_first_documented", "D"), ("hypercalcemia_g2plus", "E"), ("VTE", "F")]
FIXED = ["Myeloma", "DLBCL-like", "Indolent B-NHL", "CLL/SLL", "Hodgkin", "T-cell NHL", "MCL", "BL/LBL"]


def _crop(counts, margin=0.03):
    occ = np.argwhere(counts > 0)
    (lo_x, lo_y), (hi_x, hi_y) = occ.min(0), occ.max(0)
    side = int(max(hi_x - lo_x, hi_y - lo_y) * (1 + 2 * margin)) + 1
    ox, oy = int((lo_x + hi_x) / 2 - side / 2), int((lo_y + hi_y) / 2 - side / 2)
    return slice(max(ox, 0), ox + side), slice(max(oy, 0), oy + side)


def _lattice(frame, layer, key=None, grid=800, fill=0.0):
    a = np.full((grid, grid), fill)
    part = frame.loc[frame.layer.eq(layer)]
    if key is not None:
        part = part.loc[part.key.eq(key)]
    a[part.ix.to_numpy(), part.iy.to_numpy()] = part.value.to_numpy()
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("story_tables", type=Path)
    p.add_argument("outcome_grids", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    thesis_style()
    cells = pd.read_csv(a.story_tables / "embedding_grids_joint.csv.gz", low_memory=False)
    cells = cells.loc[cells.model.eq(MODEL)].assign(key=lambda d: d.key.fillna("").astype(str))
    meta = pd.read_csv(a.outcome_grids / "outcome_grid_meta.csv").set_index("outcome")
    story_meta = pd.read_csv(a.story_tables / "embedding_grids_joint_meta.csv").set_index("model")
    if not np.allclose(story_meta.loc[MODEL, ["x0", "x1", "y0", "y1"]].to_numpy(float),
                       meta[["x0", "x1", "y0", "y1"]].iloc[0].to_numpy(float)):
        raise ValueError("Outcome grids are not on the adapted joint map's frame")
    counts = _lattice(cells, "count")
    sx, sy = _crop(counts)
    empty = _tidy_surface_mask(counts[sx, sy] <= 0)

    # A: disease groups, as Fig. 5 (groups under 2% of patients drawn together as Other).
    disease = cells.loc[cells.layer.eq("disease")]
    sizes = disease.groupby("key").value.sum()
    sizes = sizes / sizes.sum()
    small = set(sizes[sizes < 0.02].index)
    groups = [d for d in FIXED if d in sizes.index and d not in small] + \
             [d for d in sizes.sort_values(ascending=False).index if d not in FIXED and d not in small]
    palette = [c for c in e.CATEGORICAL if c != "#9E9E9E"] + e.DISEASE_EXTRA[:1]
    colours = {d: palette[i % len(palette)] for i, d in enumerate(groups)}
    colours.update({d: c for d, c in e.DISEASE_COLOUR_OVERRIDES.items() if d in colours})
    layers = [_lattice(disease, "disease", d)[sx, sy] for d in groups]
    if small:
        layers.append(sum(_lattice(disease, "disease", d)[sx, sy] for d in small))
    stack = np.stack(layers)
    names = groups + (["Other"] if small else [])
    colours["Other"] = "#B8BEC6"
    lead = stack.argmax(0)
    purity = stack.max(0) / np.maximum(stack.sum(0), 1e-12)
    rgb = np.array([to_rgb(colours[d]) for d in names])[lead]
    rgb = np.where((purity < 0.4)[..., None], np.array(to_rgb(e.MIXED_COLOUR)), rgb)
    alpha = np.where(empty, 0.0, 1.0)

    fig = plt.figure(figsize=(6.9, 7.6))
    big = fig.add_axes([0.02, 0.53, 0.50, 0.44])
    big.imshow(np.dstack([rgb, alpha]).transpose(1, 0, 2), origin="lower", interpolation="nearest")
    # Each group's number at its densest square where it leads: a mean position fell in mixed
    # territory for groups spread over several regions (DLBCL-like).
    for number, (name, layer) in enumerate(zip(names, stack), start=1):
        if name == "Other" or layer.sum() <= 0:
            continue
        mine = np.where((lead == number - 1) & ~empty, layer, -np.inf)
        if not np.isfinite(mine).any():
            mine = np.where(~empty, layer, -np.inf)
        cx, cy = np.unravel_index(np.argmax(mine), mine.shape)
        big.text(cx, cy, str(number), ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color="white", bbox=dict(boxstyle="circle,pad=0.18", fc=colours[name], ec="white", lw=0.8))
    big.set_axis_off()
    big.set_title("A   Disease group", loc="left", fontsize=9.6, fontweight="bold", color=INK)
    handles = [Patch(fc=colours[d], ec="none", label=(f"{i}  {d}" if d != "Other" else "Other"))
               for i, d in enumerate(names, start=1)]
    handles.append(Patch(fc=e.MIXED_COLOUR, ec="none", label="Mixed (none ≥40%)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.56, 0.93), frameon=False, fontsize=7.4,
               handlelength=1.0, handleheight=0.8, labelspacing=0.5)

    # B-F: predicted risk relative to the outcome's average; G: observed / predicted, overall survival.
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad((1, 1, 1, 0))
    norm = LogNorm(vmin=0.25, vmax=4)
    small_w, small_h = 0.30, 0.20
    slots = [(0.02, 0.30), (0.35, 0.30), (0.68, 0.30), (0.02, 0.08), (0.35, 0.08), (0.68, 0.08)]
    image = None
    for (outcome, letter), (x, y) in zip(OUTCOMES, slots):
        r = meta.loc[outcome]
        grid = pd.read_csv(a.outcome_grids / r.file)
        values = _lattice(grid, "predicted", fill=np.nan)[sx, sy]
        values[empty] = np.nan
        ax = fig.add_axes([x, y, small_w, small_h])
        image = ax.imshow(np.clip(values.T, 0.25, 4), origin="lower", cmap=cmap, norm=norm,
                          interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(f"{letter}   {r.label}", loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=2)
    survival = pd.read_csv(a.outcome_grids / meta.loc["overall_survival"].file)
    observed = _lattice(survival, "mortality", fill=np.nan)[sx, sy]
    predicted = _lattice(survival, "predicted", fill=np.nan)[sx, sy]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = observed / predicted
    ratio[empty] = np.nan
    x, y = slots[5]
    gx = fig.add_axes([x, y, small_w, small_h])
    bias_cmap = plt.get_cmap("RdBu_r").copy()
    bias_cmap.set_bad((1, 1, 1, 0))
    calib = gx.imshow(np.clip(ratio.T, 0.25, 4), origin="lower", cmap=bias_cmap,
                      norm=LogNorm(vmin=0.25, vmax=4), interpolation="nearest")
    gx.set_axis_off()
    gx.set_title("G   Calibration, overall survival", loc="left", fontsize=8.6, fontweight="bold", color=INK,
                 pad=2)
    for mappable, box, ticks, labels, text in (
            (image, [0.08, 0.035, 0.40, 0.009], [0.25, 0.5, 1, 2, 4], ["¼×", "½×", "average", "2×", "4×"],
             "B–F: predicted one-year risk relative to the outcome's observed average"),
            (calib, [0.68, 0.035, 0.26, 0.009], [0.25, 0.5, 1, 2, 4], ["¼×", "½×", "1", "2×", "4×"],
             "G: observed ÷ predicted")):
        cb = fig.colorbar(mappable, cax=fig.add_axes(box), orientation="horizontal")
        cb.set_ticks(ticks, labels=labels)
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.6, length=0)
        cb.outline.set_visible(False)
        cb.set_label(text, fontsize=7.0, color=INK2)
    _save(fig, a.out / "fig_patient_map")


if __name__ == "__main__":
    main()
