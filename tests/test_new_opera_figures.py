"""New-OPERA figures (1 Nov 2026): builders, aggregate-only exports and renders.

Synthetic inputs come from ``opera.run.render_opera_design_proofs``, which runs
them through the real plot-table builders, task-structure export and arrow
aggregation."""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from opera.run import render_opera_design_proofs as proofs
from opera.visualization import opera_design_figures as design
from opera.visualization.layout_audit import audit_layout


def _rendered(fig):
    report = audit_layout(fig)
    plt.close(fig)
    return report


def test_design_search_contrasts_cover_both_references_and_skip_identities():
    from opera.run.analyse_locked_predictions import design_search_contrasts

    contrasts = design_search_contrasts({"J_L", "J_OA_OWN_L", "R_L", "R_PF_OFF_L", "UNLISTED_L"})
    assert contrasts["DS_J_L__OAOWN"] == "gain(J_L,J_OA_OWN_L)"
    assert contrasts["DS_J_OA_OWN_L__AB"] == "gain(J_OA_OWN_L,R_L)"
    assert contrasts["DS_R_L__OAOWN"] == "gain(R_L,J_OA_OWN_L)"
    assert "DS_J_OA_OWN_L__OAOWN" not in contrasts and "DS_R_L__AB" not in contrasts
    assert not any("UNLISTED" in name for name in contrasts)
    assert design_search_contrasts({"J_L", "R_L"}) == {"DS_J_L__AB": "gain(J_L,R_L)"}


def test_every_design_in_the_search_has_a_label():
    from opera.run.analyse_locked_predictions import DESIGN_SEARCH_MODELS
    from opera.run.build_paper_plot_tables import DESIGN_LABELS

    assert set(DESIGN_SEARCH_MODELS) <= set(DESIGN_LABELS)


def test_gain_sources_flip_ablations_into_removals():
    cells = proofs.synthetic_gain_sources(np.random.default_rng(0))
    assert set(cells.step) == {"corrections", "reading", "routing", "no_disease", "no_outcome", "total"}
    removal = cells.loc[cells.step == "no_disease"]
    # Synthetic "what the diagnosis correction adds" is +0.004: removing it costs.
    assert removal.panel_mean.iloc[0] < 0
    assert (removal.lower <= removal.gain).all() and (removal.gain <= removal.upper).all()
    assert (removal.panel_lower <= removal.panel_mean).all()
    assert removal.group_mean.notna().all()


def test_design_search_table_reads_tuning_only_and_labels_designs():
    from opera.run.build_paper_plot_tables import Missing, design_search_table

    table = proofs.synthetic_design_search(np.random.default_rng(0))
    assert set(table.reference) == {"J_OA_OWN", "R_AB"}
    selected = table.loc[table.selected]
    assert set(selected.reference) == {"R_AB"} and (selected.code == "J_OA_OWN").all()
    assert table.loc[table.model == "R_L", "code"].eq("R_AB").all()
    assert table.seed_of_selected.sum() == 6  # three seeds x two references
    held_out = table.assign(split="held_out", stratum="all_cells", contrast="DS_J_L__AB",
                            horizon_days=0)
    with pytest.raises(Missing):
        design_search_table(held_out)


def test_arrow_aggregates_never_write_small_bins_or_patients():
    from opera.run.export_opera_correction_arrows import arrow_aggregates

    rng = np.random.default_rng(1)
    before = rng.normal(size=(500, 2))
    after = {("o", "total"): before + 0.1, ("o", "reading"): (before, before + 0.05)}
    arrows, density = arrow_aggregates(before, after, bins=6)
    assert (arrows["count"] >= 20).all() and (density["count"] >= 20).all()
    assert len(arrows) <= 2 * 36 and len(density) < len(before)
    assert set(arrows.columns) == {"outcome", "component", "center_x", "center_y", "dx", "dy",
                                   "count", "coherence"}
    np.testing.assert_allclose(arrows.loc[arrows.component == "total", ["dx", "dy"]], 0.1)
    with pytest.raises(ValueError, match="at least 20"):
        arrow_aggregates(before, after, min_count=5)


def test_representation_parts_decompose_the_model_path():
    from opera.modules.networks.attention_own_corrections import AttentionOwnCorrections
    from opera.modules.networks.opera_nets import OutcomeAttentionReadout
    from opera.run.export_opera_correction_arrows import representation_parts

    torch.manual_seed(0)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.outcome_attention_readout = OutcomeAttentionReadout(16, ["a", "b", "c"], num_heads=2)
            self.attention_own_corrections = AttentionOwnCorrections(16, ["d1", "d2"], ["a", "b", "c"], rank=2)
            with torch.no_grad():
                self.outcome_attention_readout.residual_gates.fill_(0.5)
                self.attention_own_corrections.outcome_up.normal_()

        def _encode_hidden(self, batch):
            return batch["hidden"]

        def _pool_hidden(self, hidden, batch):
            return hidden[:, -1]

    model = Stub().eval()
    batch = {"hidden": torch.randn(4, 5, 16), "attention_mask": torch.ones(4, 5),
             "cohort_fine_index": torch.tensor([0, 1, 0, -1])}
    parts = representation_parts(model, batch, [2, 0])
    with torch.no_grad():
        read = model.outcome_attention_readout(batch["hidden"], batch["hidden"][:, -1], batch["attention_mask"])
        final = model.attention_own_corrections(read, batch["cohort_fine_index"])
    torch.testing.assert_close(parts["shared"], batch["hidden"][:, -1])
    torch.testing.assert_close(parts["read"], read[:, [2, 0]])
    torch.testing.assert_close(parts["final"], final[:, [2, 0]])
    assert not torch.allclose(parts["read"], parts["final"])


def test_task_structure_csv_round_trip_renders(tmp_path):
    from opera.run.analyse_locked_predictions import read_table

    tables, summary = proofs.synthetic_task_structure()
    for name, table in tables.items():
        table.to_csv(tmp_path / f"{name}.csv", index=name.endswith("similarity"))
    read = {name: read_table(tmp_path / f"{name}.csv") for name in tables}
    assert "training_events" in read["outcome_specialisation"]
    fig = design.figure_task_organisation(
        read["outcome_query_similarity"], read["disease_correction_similarity"],
        read["outcome_specialisation"], read["disease_specialisation"], summary,
        outcome_correction_similarity=read["outcome_correction_similarity"])
    images = [ax.images[0] for ax in fig.axes if ax.images and ax.get_label() != "<colorbar>"]
    images = [image for image in images if image.get_array().ndim == 2 and image.get_array().shape[0] > 1][:3]
    assert images[1].norm.vmax == images[2].norm.vmax  # correction maps share a scale
    assert len(fig.axes) == 5  # three matrices, two colourbars
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "Overall survival" in labels  # registry ids shown by display name
    assert not _rendered(fig)["text_collision_candidates"]
    assert not _rendered(design.figure_specialisation(read["outcome_specialisation"],
                                                      read["disease_specialisation"]))["text_collision_candidates"]


@pytest.mark.parametrize("name", ["fig_gain_sources", "supp_design_search", "supp_correction_arrows",
                                  "supp_xgboost_gap"])
def test_new_figures_render_without_text_collisions(name):
    rng = np.random.default_rng(3)
    if name == "fig_gain_sources":
        fig = design.figure_gain_sources(proofs.synthetic_gain_sources(rng), mark="SYNTHETIC")
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert "+ Learn whom to borrow from" in labels and "− Remove outcome correction" in labels
    elif name == "supp_design_search":
        fig = design.figure_design_search(proofs.synthetic_design_search(rng), mark="SYNTHETIC")
        assert len(fig.axes) == 2 and fig.axes[0].patches  # seed band drawn
    elif name == "supp_correction_arrows":
        arrows, density, meta = proofs.synthetic_correction_arrows(n=3000)
        fig = design.figure_correction_arrows(arrows, density, meta, mark="SYNTHETIC")
        assert fig.opera_arrow_scale > 0
    else:
        fig = design.figure_xgboost_gap(proofs.synthetic_gap_cells(rng), mark="SYNTHETIC")
    assert not _rendered(fig)["text_collision_candidates"]


def test_gain_sources_without_the_pending_ablations():
    cells = proofs.synthetic_gain_sources(np.random.default_rng(4))
    fig = design.figure_gain_sources(cells.loc[~cells.step.str.startswith("no_")])
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(labels) == 4 and not any("Remove" in label for label in labels)
    plt.close(fig)


def test_recipe_registers_the_new_figures_and_the_renderer_finds_them(tmp_path, monkeypatch):
    from opera.run.render_paper_results import main

    recipe = yaml.safe_load(open("opera/configs/paper_results_figures.full.example.yaml"))
    new = {"fig_gain_sources", "supp_design_search", "supp_task_organisation",
           "supp_specialisation", "supp_correction_arrows", "supp_xgboost_gap"}
    assert new <= set(recipe["required_figures"]) and new <= {p["id"] for p in recipe["figures"]}
    proofs.synthetic_gap_cells(np.random.default_rng(0)).to_csv(tmp_path / "paired_cells.csv", index=False)
    config = tmp_path / "recipe.yaml"
    config.write_text(yaml.safe_dump({"figures": [{"id": "supp_xgboost_gap", "function": "figure_xgboost_gap",
                                                   "tables": {"cells": "paired_cells.csv"}}]}))
    monkeypatch.setattr(sys, "argv", ["render", "--config", str(config), "--output-dir",
                                      str(tmp_path / "out"), "--execute"])
    main()
    manifest = json.loads((tmp_path / "out" / "figure_manifest.json").read_text())
    assert [f["id"] for f in manifest["figures"]] == ["supp_xgboost_gap"]
    assert "design_figures" in manifest["code_sha256"]


def test_figure_readiness_ignores_commented_out_figures(tmp_path):
    from opera.run.check_figure_inputs import manuscript_figures, producer

    tex = tmp_path / "m.tex"
    tex.write_text("% \\resultfigure{supp_router_arrows}{x}{y}\n\\resultfigure{supp_xgboost_gap}{x}{y} 5\\% more\n")
    assert manuscript_figures(tex) == ["supp_xgboost_gap"]
    assert producer(tmp_path / "plot_tables" / "outcome_query_similarity.csv", tmp_path) == "task-structure"


def test_pipeline_actions_for_the_exports(tmp_path):
    from test_paper_workflow import args, env
    from opera.run import paper_experiments

    environment = {**env(tmp_path), "OPERA_PAPER_ATTRIBUTION_CHECKPOINT": str(tmp_path / "run/best.ckpt"),
                   "OPERA_PAPER_ATTRIBUTION_CONFIG_NAME": "cfg",
                   "OPERA_PAPER_EMBEDDING_CONFIG": str(tmp_path / "e.yaml")}
    for action, module in (("task-structure", "opera.run.export_opera_task_structure"),
                           ("correction-arrows", "opera.run.export_opera_correction_arrows")):
        command = [str(c) for c in paper_experiments.build_commands(args(action), environment)[0]]
        assert module in command and command[command.index("--output-dir") + 1].endswith("plot_tables")
