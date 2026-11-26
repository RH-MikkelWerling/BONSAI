"""Final encoder maps from the 800-grid aggregates: one shared square crop around the occupied cells of
all three maps (same frame and scale for every column), then the standard renderers."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path.cwd()))
from opera.run.paper_story_figures import thesis_style, _save
from opera.visualization import encoder_embedding_comparison as e

def shared_crop(cells, meta, margin=0.03):
    occ = cells[cells.layer == "count"]
    lo_x, hi_x, lo_y, hi_y = occ.ix.min(), occ.ix.max(), occ.iy.min(), occ.iy.max()
    side = int(max(hi_x - lo_x, hi_y - lo_y) * (1 + 2 * margin)) + 1
    ox = int((lo_x + hi_x) / 2 - side / 2); oy = int((lo_y + hi_y) / 2 - side / 2)
    c = cells.assign(ix=cells.ix - ox, iy=cells.iy - oy)
    c = c[(c.ix >= 0) & (c.ix < side) & (c.iy >= 0) & (c.iy < side)]
    m = meta.copy()
    dx = (m.x1 - m.x0) / m.grid; dy = (m.y1 - m.y0) / m.grid
    m["x0"] = meta.x0 + ox * dx; m["x1"] = m.x0 + side * dx
    m["y0"] = meta.y0 + oy * dy; m["y1"] = m.y0 + side * dy; m["grid"] = side
    return c, m

tables = Path(sys.argv[1]); out = Path(sys.argv[2])
thesis_style()
for family, stem in [("joint", "fig_encoder_maps"), ("opera", "supp_embedding_opera")]:
    cells = pd.read_csv(tables / f"embedding_grids_{family}.csv.gz", low_memory=False)
    meta = pd.read_csv(tables / f"embedding_grids_{family}_meta.csv")
    cells, meta = shared_crop(cells, meta)
    print(family, "crop side", int(meta.grid.iloc[0]))
    labels = {m: m.split()[0].replace("Local", "Cohort") + " encoder" for m in meta.model}
    support = "--support" in sys.argv[3:]
    rows = ("disease", "ipi", "predicted", "mortality", "support_band") if support else ("disease", "ipi", "predicted", "mortality")
    if support and not cells.layer.eq("support_band").any():
        raise ValueError("Support panel requested but support_band aggregates are absent")
    _save(e.figure_encoder_maps_v2(cells, meta, model_labels=labels, rows=rows), out / stem)
    if family == "joint":
        _save(e.figure_encoder_disease_grid(cells, meta, model_labels=labels), out / "supp_embedding_diseases")
