from pathlib import Path

import numpy as np
import pytest
import torch

from opera.run.export_router_borrowing_atlas import (
    RouteAccumulator, budget_matrices, plot_atlas, source_flows,
)


def test_accumulator_conditions_on_fine_disease_and_target_eligibility():
    acc = RouteAccumulator(n_disease=2, n_outcome=2)
    weights = torch.tensor([
        [[.4, .2, .1, .2, .1], [.3, .2, .2, .1, .2]],
        [[.5, .1, .1, .2, .1], [.3, .1, .2, .2, .2]],
        [[.2, .1, .3, .2, .2], [.2, .1, .3, .2, .2]],
    ])
    acc.update(weights, torch.tensor([0, 0, 1]),
               torch.tensor([[True, False], [True, True], [True, True]]))
    assert acc.counts.tolist() == [[2, 1], [1, 1]]
    means = acc.mean()
    assert means[0, 0, 0] == pytest.approx(.45)
    assert means[0, 1, 0] == pytest.approx(.3)
    budgets = budget_matrices(means, acc.counts, min_patients=1)
    assert budgets["borrowed_disease"][0, 0] == pytest.approx(.1)
    assert budgets["borrowed_outcome"][0, 0] == pytest.approx(.1)
    assert budgets["own_disease"][1, 0] == pytest.approx(.3)
    dflow, fflow = source_flows(means, acc.counts, {"A": ["a"], "B": ["b"]},
                               ["a", "b"], min_patients=1)
    assert np.isnan(dflow[0, 0])
    assert dflow[0, 1] == pytest.approx(.15)
    assert fflow[0, 1] == pytest.approx(.15)


def test_low_support_is_masked_without_changing_source_cube():
    means = np.ones((2, 2, 5), dtype=float) / 5
    counts = np.array([[20, 1], [0, 30]])
    budgets = budget_matrices(means, counts, min_patients=20)
    assert np.isnan(budgets["shared"][0, 1])
    assert np.isnan(budgets["borrowed_outcome"][1, 0])
    assert means[0, 1, 0] == pytest.approx(.2)


def test_atlas_plot_writes_both_pages(tmp_path: Path):
    values = np.array([[.1, .2], [.3, .4]])
    flow = np.array([[np.nan, .1], [.2, np.nan]])
    plot_atlas({"borrowed_disease": values, "borrowed_outcome": values},
               flow, flow, ["CLL", "SLL"], ["a", "b"], ["A", "B"], tmp_path)
    assert (tmp_path / "router_full_task_borrowing_atlas.pdf").stat().st_size > 0
    assert (tmp_path / "router_full_source_flow.pdf").stat().st_size > 0
