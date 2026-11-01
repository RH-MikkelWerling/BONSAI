"""Task-structure export and figures for the new OPERA (synthetic checkpoint
with planted family structure)."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from opera.run.export_opera_task_structure import task_structure, within_between
from opera.visualization.opera_design_figures import figure_specialisation, figure_task_organisation


def _state(seed=0, planted=True):
    g = torch.Generator().manual_seed(seed)
    h, r, n_out, n_dis = 16, 4, 8, 6
    family = torch.randn(2, h, generator=g)
    queries = torch.stack([family[o // 4] * (3 if planted else 0) + torch.randn(h, generator=g) for o in range(n_out)])
    lineage = torch.randn(3, r, h, generator=g)
    down = torch.stack([lineage[d // 2] * (3 if planted else 0) + torch.randn(r, h, generator=g) for d in range(n_dis)])
    return {"outcome_attention_readout.queries": queries,
            "outcome_attention_readout.attention.in_proj_weight": torch.eye(h).repeat(3, 1),
            "outcome_attention_readout.residual_gates": torch.linspace(0, 0.3, n_out),
            "attention_own_corrections.outcome_down": torch.randn(n_out, r, h, generator=g),
            "attention_own_corrections.outcome_up": torch.randn(n_out, h, r, generator=g),
            "attention_own_corrections.disease_down": down,
            "attention_own_corrections.disease_up": torch.ones(n_dis, h, r)}


def _names():
    outcomes = [f"o{i}" for i in range(8)]
    diseases = [f"d{i}" for i in range(6)]
    return outcomes, diseases, {o: f"F{int(o[1:]) // 4}" for o in outcomes}, {d: f"L{int(d[1:]) // 2}" for d in diseases}


def test_planted_family_structure_is_found_and_absent_structure_is_not():
    outcomes, diseases, family, lineage = _names()
    tables, summary = task_structure(_state(planted=True), outcomes, diseases, outcome_family=family,
                                     disease_lineage=lineage, outcome_cases={o: 20 + 10 * i for i, o in enumerate(outcomes)})
    assert summary["outcome_query_similarity"]["difference"] > 0.3
    assert summary["outcome_query_similarity"]["p_permutation"] < 0.05
    assert summary["disease_correction_similarity"]["difference"] > 0.2
    assert set(tables) == {"outcome_query_similarity", "outcome_correction_similarity",
                           "disease_correction_similarity", "outcome_specialisation", "disease_specialisation"}
    _, flat = task_structure(_state(planted=False), outcomes, diseases, outcome_family=family, disease_lineage=lineage)
    assert flat["outcome_query_similarity"]["p_permutation"] > 0.05
    spec = tables["outcome_specialisation"]
    np.testing.assert_allclose(spec.gate.to_numpy(), np.tanh(np.linspace(0, 0.3, 8)), rtol=1e-6)


def test_within_between_is_exact_on_a_block_matrix():
    labels = ["a", "a", "b", "b"]
    s = np.array([[1, .8, .1, .1], [.8, 1, .1, .1], [.1, .1, 1, .6], [.1, .1, .6, 1]])
    out = within_between(s, labels, permutations=200)
    assert abs(out["within"] - 0.7) < 1e-9 and abs(out["between"] - 0.1) < 1e-9


def test_figures_render(tmp_path):
    outcomes, diseases, family, lineage = _names()
    tables, summary = task_structure(_state(), outcomes, diseases, outcome_family=family, disease_lineage=lineage,
                                     outcome_cases={o: 20 + 10 * i for i, o in enumerate(outcomes)})
    fig = figure_task_organisation(tables["outcome_query_similarity"], tables["disease_correction_similarity"],
                                   family, lineage, summary)
    fig.savefig(tmp_path / "organisation.png")
    fig = figure_specialisation(tables["outcome_specialisation"])
    fig.savefig(tmp_path / "specialisation.png")
    assert (tmp_path / "organisation.png").stat().st_size > 10_000


def test_task_matrices_share_the_colour_scale_of_their_single_colourbar():
    import matplotlib.pyplot as plt

    outcome = pd.DataFrame([[1, .8], [.8, 1]], index=["o1", "o2"], columns=["o1", "o2"])
    disease = pd.DataFrame([[1, -.2], [-.2, 1]], index=["d1", "d2"], columns=["d1", "d2"])
    fig = figure_task_organisation(outcome, disease,
                                   {"o1": "F", "o2": "F"}, {"d1": "L", "d2": "L"})
    first, second = [axis.images[0] for axis in fig.axes[:2]]
    assert first.norm.vmin == second.norm.vmin == -.8
    assert first.norm.vmax == second.norm.vmax == .8
    np.testing.assert_allclose(first.cmap(first.norm(.2)), second.cmap(second.norm(.2)))
    plt.close(fig)
