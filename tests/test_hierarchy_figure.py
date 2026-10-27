import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from opera.run.analyse_locked_predictions import router_decomposition
from opera.run.build_paper_plot_tables import Missing, hierarchy_cells
from opera.visualization.paper_figures import figure_hierarchy


def test_decomposition_only_for_analysed_designs():
    assert router_decomposition({"J_L", "R_L"}) == {}
    contrasts = router_decomposition({"R_O_L", "RBH_O_L", "N_A_L", "R_AB_L"})
    assert contrasts == {"router_own_O": "gain(RBH_O_L,N_A_L)",
                         "router_borrow_O": "gain(R_O_L,RBH_O_L)",
                         "router_both_O": "gain(R_O_L,N_A_L)"}


def _contrasts(design):
    rows = []
    for cohort in ("DLBCL", "MCL"):
        for split, code in (("tuning", "P5"), ("held_out", f"router_own_{design}"),
                            ("held_out", f"router_borrow_{design}"),
                            ("held_out", f"router_both_{design}")):
            rows.append({"cohort": cohort, "outcome": "sepsis", "horizon_days": 365,
                         "metric": "auc", "split": split, "seed": 42, "contrast": code,
                         "gain": 0.01, "lower": 0.0, "upper": 0.02})
    return pd.DataFrame(rows)


def test_hierarchy_cells_table_and_missing_design():
    metadata = pd.DataFrame({"cohort": ["DLBCL", "MCL"], "outcome": "sepsis",
                             "horizon_days": 365, "n_events_train": [500, 12]})
    both = pd.concat([_contrasts("O"), _contrasts("AB").query("contrast != 'P5'")])
    table = hierarchy_cells(both, metadata)
    assert set(table.router_design) == {"AB"} and len(table) == 2
    assert {"tuning_joint_vs_specialist", "own", "borrow_lower", "both_upper"} <= set(table)
    with pytest.raises(Missing):
        hierarchy_cells(_contrasts("O").query("contrast == 'P5'"), metadata)


def test_figure_hierarchy_renders():
    rng = np.random.default_rng(0)
    n = 300
    events = rng.integers(2, 3000, n)
    cells = pd.DataFrame({"metric": "auc", "horizon_days": 365, "n_events_train": events,
                          "tuning_joint_vs_specialist": rng.normal(0.01, 0.03, n),
                          "own": rng.normal(0, 0.02, n), "borrow": rng.normal(0, 0.02, n)})
    cells["both"] = cells.own + cells.borrow
    for c in ("own", "borrow", "both"):
        cells[f"{c}_lower"], cells[f"{c}_upper"] = cells[c] - 0.05, cells[c] + 0.05
    fig = figure_hierarchy(cells)
    assert len(fig.axes) == 3  # A, B, C (the band bars are an inset of C)


def test_opera_alias_copies_the_chosen_model():
    from opera.run.analyse_locked_predictions import with_opera_alias
    frame = pd.DataFrame({"model": ["J_L", "R_L", "X"], "risk": [0.1, 0.2, 0.3]})
    out, contrasts = with_opera_alias(frame, "R_L")
    assert out.loc[out.model == "OPERA", "risk"].tolist() == [0.2]
    assert contrasts == {"O1": "gain(OPERA,X)", "O2": "gain(OPERA,J_L)"}  # no S_L
    assert with_opera_alias(frame, "R_O_L")[1] == {}
    assert with_opera_alias(frame, None)[0] is frame


def test_hierarchy_cells_follow_the_chosen_design():
    metadata = pd.DataFrame({"cohort": ["DLBCL", "MCL"], "outcome": "sepsis",
                             "horizon_days": 365, "n_events_train": [500, 12]})
    both = pd.concat([_contrasts("O"), _contrasts("AB").query("contrast != 'P5'")])
    assert set(hierarchy_cells(both, metadata).router_design) == {"AB"}  # default order
    assert set(hierarchy_cells(both, metadata, designs=("O", "AB")).router_design) == {"O"}
