"""Every manuscript figure can be built from server outputs, and nothing else."""

import json
from pathlib import Path

import yaml

from opera.run.check_figure_inputs import check

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "paper/opera_manuscript.tex"


def test_every_manuscript_figure_has_a_recipe_and_every_input_a_producer(tmp_path):
    rows = check(tmp_path, MANUSCRIPT)
    assert rows, "the manuscript includes rendered figures"
    assert not [r["figure"] for r in rows if r["status"] == "no recipe"]
    unknown = [(r["figure"], m["input"]) for r in rows for m in r["missing"]
               if m["run"] == "unknown"]
    assert not unknown, unknown


def test_all_figures_are_ready_once_their_inputs_exist(tmp_path):
    for row in check(tmp_path, MANUSCRIPT):
        for item in row["missing"]:
            path = Path(item["input"])
            if "synthetic_patient_episodes" in str(path):
                continue  # written by the prototype renderer into the repository
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
    rows = check(tmp_path, MANUSCRIPT)
    waiting = [r for r in rows if r["status"] != "ready"]
    synthetic = (REPO / "paper/figures/synthetic_patient_episodes.csv").is_file()
    assert all("synthetic" in m["input"] for r in waiting for m in r["missing"]) or not waiting
    if synthetic:
        assert not waiting


def test_figure_workspace_keeps_analysis_outputs_where_their_step_writes_them(tmp_path):
    import subprocess
    import sys

    workspace = tmp_path / "analysis" / "figure_inputs"
    subprocess.run([sys.executable, "-m", "opera.run.prepare_paper_figure_inputs",
                    "--output-dir", str(workspace), "--execute"], check=True,
                   capture_output=True)
    recipe = yaml.safe_load((workspace / "figures.yaml").read_text())
    headline = next(p for p in recipe["figures"] if p["id"] == "fig_headline")
    assert headline["tables"]["scores"] == str(
        (tmp_path / "analysis" / "clinical_scores" / "headline_scores.csv").resolve())
    assert headline["tables"]["cells"].endswith("figure_inputs/plot_tables/paired_cells.csv")
    assert json.loads((workspace / "required_inputs.json").read_text())


def test_relative_recipe_inputs_resolve_against_the_recipe(tmp_path):
    from opera.run.render_paper_results import resolve_input

    recipe = tmp_path / "configs" / "figures.yaml"
    assert resolve_input("../data/t.csv", recipe) == (tmp_path / "data" / "t.csv").resolve()
    assert resolve_input("${OPERA_SURELY_UNSET_VARIABLE}/t.csv", recipe) is None


def test_readiness_checks_extra_registered_figures_when_requested(tmp_path):
    manuscript = tmp_path / 'paper.tex'
    manuscript.write_text('No figures placed yet.')
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump({'figures':[{'id':'fig_extra', 'tables':{'cells':'missing.csv'}}]}))
    assert check(tmp_path, manuscript, recipes=(recipe,)) == []
    rows = check(tmp_path, manuscript, recipes=(recipe,), include_registered=True)
    assert rows[0]['figure'] == 'fig_extra'
    assert rows[0]['status'] == 'missing'
