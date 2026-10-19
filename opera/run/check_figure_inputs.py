"""Is every manuscript figure ready to render? Run on the server after the models.

For each figure the manuscript includes (``\\resultfigure``), finds the recipe
or renderer that makes it, checks that every input exists in the analysis
root, and names the pipeline action that produces each missing input (with the
reason the plot-table step recorded, when it recorded one). Static diagrams
(TikZ) need no data. Exit status 1 while anything is missing, so the check can
gate rendering.

    bash scripts/run_paper_experiments.sh figure-readiness --execute
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
RESULTS_RECIPE = REPO / "opera/configs/paper_results_figures.full.example.yaml"
ATTRIBUTION_RECIPE = REPO / "opera/configs/attribution_figures.example.yaml"
# Rendered by the clinical-scores action into <analysis root>/clinical_scores.
CLINICAL_FIGURES = ("fig_risk_groups", "fig_clinical_score", "supp_clinical_score_age",
                    "supp_score_overview")
EMBEDDING_TABLES = ("gallery_coordinates.csv", "embedding_gallery.parquet",
                    "patient_coordinates.csv", "matched_patient_embeddings.csv",
                    "calibration_map_outcomes.json")


def manuscript_text(path: Path) -> str:
    """The manuscript with every ``\\input{...}`` expanded (body files are shared
    with the thesis)."""
    path = Path(path)

    def expand(match):
        name = match.group(1)
        child = path.parent / (name if name.endswith(".tex") else f"{name}.tex")
        return manuscript_text(child) if child.is_file() else match.group(0)

    return re.sub(r"\\input\{([^}]+)\}", expand, path.read_text())


def manuscript_figures(path: Path) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\\resultfigure(?:\[[^\]]*\])?\{([A-Za-z0-9_]+)\}",
                                         manuscript_text(path))))


def producer(path: Path, analysis_root: Path) -> str:
    """Pipeline action (or script) that writes ``path``."""
    text = str(path)
    if path.name in EMBEDDING_TABLES:
        return "embedding-tables"
    if "/plot_tables/" in text:
        return "plot-tables"
    if "/clinical_scores/" in text:
        return "clinical-scores"
    if "/attributions/" in text:
        return "attributions, then attribution-summary"
    if "/input_restrictions/" in text:
        return "input-restrictions (after attributions)"
    if "synthetic_patient_episodes" in text:
        return "python -m opera.run.render_paper_figure_prototypes (synthetic illustration)"
    return "unknown"


def recipe_inputs(recipe: Path, environment: dict[str, str]) -> dict[str, list[Path]]:
    """Figure id -> resolved input paths, as the renderer resolves them."""
    spec = yaml.safe_load(recipe.read_text())
    saved = dict(os.environ)
    os.environ.update(environment)
    try:
        out = {}
        for plate in spec.get("figures", []):
            paths = []
            for source in {**plate.get("tables", {}), **plate.get("json_inputs", {})}.values():
                expanded = os.path.expandvars(source)
                path = Path(expanded)
                paths.append(path if path.is_absolute() else recipe.resolve().parent / path)
            out[plate["id"]] = paths
        return out
    finally:
        os.environ.clear()
        os.environ.update(saved)


def check(analysis_root: Path, manuscript: Path, *, horizon: int = 365,
          recipes: tuple[Path, ...] = (RESULTS_RECIPE, ATTRIBUTION_RECIPE)) -> list[dict]:
    analysis_root = analysis_root.resolve()
    plot_tables = analysis_root / "figure_inputs" / "plot_tables"
    environment = {
        "OPERA_PAPER_ANALYSIS_ROOT": str(analysis_root),
        "OPERA_PAPER_PLOT_TABLE_ROOT": str(plot_tables),
        "OPERA_PAPER_ATTRIBUTION_SUMMARY_DIR": str(analysis_root / "attributions"
                                                   / f"summary_{float(horizon):g}d"),
        "OPERA_PAPER_INPUT_RESTRICTION_DIR": str(analysis_root / "input_restrictions"),
    }
    inputs = {}
    for recipe in recipes:
        inputs.update(recipe_inputs(recipe, environment))
    manifest = plot_tables / "plot_tables_manifest.json"
    reasons = (json.loads(manifest.read_text()).get("tables", {})
               if manifest.is_file() else {})
    rows = []
    for figure in manuscript_figures(manuscript):
        if figure in CLINICAL_FIGURES:
            needed = [analysis_root / "clinical_scores" / f"{figure}.pdf"]
        elif figure in inputs:
            needed = inputs[figure]
        else:
            rows.append({"figure": figure, "status": "no recipe", "missing": []})
            continue
        missing = []
        for path in needed:
            if path.is_file():
                continue
            reason = reasons.get(path.name, {})
            missing.append({"input": str(path), "run": producer(path, analysis_root),
                            **({"reason": reason.get("input") or reason.get("status")}
                               if reason else {})})
        rows.append({"figure": figure, "status": "ready" if not missing else "missing",
                     "missing": missing})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, default=REPO / "paper/opera_manuscript.tex")
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = parser.parse_args()
    rows = check(args.analysis_root, args.manuscript, horizon=args.horizon)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        ready = [r for r in rows if r["status"] == "ready"]
        print(f"{len(ready)} of {len(rows)} manuscript figures ready to render")
        actions = {}
        for row in rows:
            if row["status"] == "ready":
                continue
            print(f"  {row['figure']}: {row['status']}")
            for item in row["missing"]:
                note = f" ({item['reason']})" if item.get("reason") else ""
                print(f"      {Path(item['input']).name}  <- {item['run']}{note}")
                actions.setdefault(item["run"], set()).add(row["figure"])
        if actions:
            print("Actions still to run:")
            for action, figures in actions.items():
                print(f"  {action}: {len(figures)} figure(s)")
    sys.exit(0 if all(r["status"] == "ready" for r in rows) else 1)


if __name__ == "__main__":
    main()
