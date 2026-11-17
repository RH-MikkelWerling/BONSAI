"""Re-render registered paper figures in the thesis style, from their existing recipes.

Each recipe directory has ``figures.yaml`` (id, function, tables, json_inputs, options) and
``plot_tables/``. The figure functions are unchanged; only the style differs: TeX Gyre
Pagella (the manuscript's body face), muted ink, and the thesis panel headings (bold
letter and title). Output goes to ``paper/figures/results_thesis/`` so the registered
figures in ``paper/figures/results/`` are untouched.

    python -m opera.run.render_thesis_style --recipes scratch/recovered_final_figures_20261116_v4 \
        --ids supp_horizons supp_coverage supp_support --out paper/figures/results_thesis
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from opera.run.paper_story_figures import thesis_style  # noqa: E402
from opera.visualization import attribution_overview, opera_design_figures, paper_figures  # noqa: E402
from opera.visualization.style import setup_style  # noqa: E402

MODULES = (paper_figures, opera_design_figures, attribution_overview)
HEADING = re.compile(r"^([A-H])\s{1,3}(\S.*)$", re.S)


def function(name):
    for module in MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise KeyError(name)


def thesis_headings(fig):
    """Panel titles 'A  Title' become bold thesis headings; other text takes the body face."""
    for ax in fig.axes:
        t = ax.title if ax.get_title(loc="left") == "" else ax._left_title
        text = t.get_text()
        m = HEADING.match(text)
        if m:
            t.set_text(f"{m.group(1)}   {m.group(2)}")
            t.set_fontweight("bold")
            t.set_fontsize(min(t.get_fontsize(), 9.6))
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontfamily("serif")


def render(recipe_dir: Path, ids, out: Path, mark=None):
    spec = yaml.safe_load((recipe_dir / "figures.yaml").read_text())
    done = []
    for entry in spec["figures"]:
        if ids and entry["id"] not in ids:
            continue
        kwargs = {k: pd.read_csv(recipe_dir / v, low_memory=False) for k, v in (entry.get("tables") or {}).items()}
        kwargs.update({k: json.loads((recipe_dir / v).read_text()) for k, v in (entry.get("json_inputs") or {}).items()})
        kwargs.update(entry.get("options") or {})
        fig = function(entry["function"])(**kwargs, mark=mark)
        thesis_headings(fig)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{entry['id']}.pdf")
        fig.savefig(out / f"{entry['id']}.png", dpi=200)
        plt.close(fig)
        done.append(entry["id"])
        print(f"wrote {out / entry['id']}.pdf")
    missing = set(ids or []) - set(done)
    if missing:
        raise SystemExit(f"not in {recipe_dir}: {sorted(missing)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipes", type=Path, required=True)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    setup_style()
    thesis_style()
    render(a.recipes, a.ids, a.out)


if __name__ == "__main__":
    main()
