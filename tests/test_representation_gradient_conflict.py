import json

import numpy as np
import pytest
import torch

from opera.diagnostics.representation_gradient_conflict import (
    _batched_gradient_cosines,
    _gradient_cosine,
    _write_outputs,
)
from opera.diagnostics.conflict_verdict import run_conflict_verdict
from opera.modules.networks.opera_nets import outcome_eligibility_mask
from opera.diagnostics.endpoint_dependencies import (
    classify_endpoint_pair,
    classify_endpoint_pairs,
)


def test_gradient_cosine_uses_only_jointly_eligible_rows():
    first = torch.tensor([[1.0, 0.0], [100.0, 0.0], [0.0, 1.0]])
    second = torch.tensor([[1.0, 0.0], [-100.0, 0.0], [0.0, 1.0]])
    joint = torch.tensor([True, False, True])

    cosine = _gradient_cosine(first, second, joint)

    assert cosine == pytest.approx(1.0)


def test_batched_gradient_cosines_match_scalar_implementation():
    first = [
        torch.tensor([[1.0, 0.0], [9.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 1.0], [9.0, 0.0], [1.0, -1.0]]),
    ]
    second = [
        torch.tensor([[1.0, 0.0], [-9.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, -1.0], [-9.0, 0.0], [-1.0, -1.0]]),
    ]
    joint = torch.tensor([True, False, True])

    observed = _batched_gradient_cosines(first, second, joint)
    expected = [_gradient_cosine(a, b, joint) for a, b in zip(first, second)]

    assert observed == pytest.approx(expected)


def test_endpoint_dependencies_keep_nested_thresholds_as_ineligible_controls():
    relationship = classify_endpoint_pair("anemia_g2plus", "anemia_g3plus")
    assert relationship.relationship == "nested_threshold_control"
    assert not relationship.scaffold_eligible


def test_composite_is_removed_but_atomic_pair_remains_eligible():
    rows = classify_endpoint_pairs(
        ["sepsis", "infections_IV_based", "serious_infection_composite"],
        composites={"serious_infection_composite": ["sepsis", "infections_IV_based"]},
    )
    indexed = {(row["outcome_a"], row["outcome_b"]): row for row in rows}
    assert indexed[("sepsis", "infections_IV_based")]["scaffold_eligible"]
    assert not indexed[("sepsis", "serious_infection_composite")]["scaffold_eligible"]


def test_diagnostic_joint_support_excludes_sentinel_rows():
    times_a = torch.tensor([1.0, -1.0, 3.0, 4.0])
    events_a = torch.tensor([1, -1, 0, 1])
    times_b = torch.tensor([1.0, 2.0, -1.0, 4.0])
    events_b = torch.tensor([1, 0, -1, 1])
    joint = outcome_eligibility_mask(times_a, events_a) & outcome_eligibility_mask(
        times_b,
        events_b,
    )
    first = torch.tensor([[1.0], [100.0], [100.0], [1.0]])
    second = torch.tensor([[1.0], [-100.0], [-100.0], [1.0]])

    cosine = _gradient_cosine(first, second, joint)

    assert joint.tolist() == [True, False, False, True]
    assert cosine == pytest.approx(1.0)


def test_gradient_conflict_outputs_include_ranked_pairs(tmp_path):
    names = ["a", "b", "c"]
    cosine_sum = np.array(
        [
            [2.0, -1.0, 1.5],
            [-1.0, 2.0, 0.5],
            [1.5, 0.5, 2.0],
        ]
    )
    cosine_count = np.full((3, 3), 2)
    support_sum = np.full((3, 3), 24.0)
    supported_batches = np.full((3, 3), 2)

    summary = _write_outputs(
        tmp_path,
        tmp_path / "model.ckpt",
        names,
        cosine_sum,
        cosine_count,
        support_sum,
        supported_batches,
        np.array([2.0, 1.0, 3.0]),
        np.array([10.0, 10.0, 10.0]),
        n_batches=2,
        min_overlap=8,
        negative_threshold=-0.1,
    )

    assert summary["fraction_below_zero"] == pytest.approx(1.0 / 3.0)
    assert summary["most_conflicting"][0]["outcome_a"] == "a"
    assert summary["most_conflicting"][0]["outcome_b"] == "b"
    assert (tmp_path / "gradient_cosine.npy").exists()
    assert (tmp_path / "gradient_cosine.csv").exists()
    assert (tmp_path / "joint_support.csv").exists()
    assert (tmp_path / "gradient_pair_batches.csv").exists()
    assert (tmp_path / "gradient_cosine_heatmap.png").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["n_supported_pairs"] == 3


def _synthetic_gradient_pair_batches() -> list[dict]:
    rng = np.random.default_rng(23)
    records = []
    outcomes = ["conflict_a", "conflict_b", "aligned_a", "aligned_b", "orthogonal"]
    for batch_index in range(36):
        base = rng.normal(size=24)
        gradients = {
            "conflict_a": base + rng.normal(scale=0.04, size=24),
            "conflict_b": -base + rng.normal(scale=0.04, size=24),
            "aligned_a": base + rng.normal(scale=0.05, size=24),
            "aligned_b": base + rng.normal(scale=0.05, size=24),
            "orthogonal": rng.normal(size=24),
        }
        for first_index, first in enumerate(outcomes):
            for second in outcomes[first_index + 1 :]:
                first_gradient = gradients[first]
                second_gradient = gradients[second]
                cosine = float(
                    np.dot(first_gradient, second_gradient)
                    / (np.linalg.norm(first_gradient) * np.linalg.norm(second_gradient))
                )
                records.append(
                    {
                        "batch_index": batch_index,
                        "outcome_a": first,
                        "outcome_b": second,
                        "cosine": cosine,
                        "joint_support": 16,
                    }
                )
        records.append(
            {
                "batch_index": batch_index,
                "outcome_a": "low_support_a",
                "outcome_b": "low_support_b",
                "cosine": -0.99,
                "joint_support": 3,
            }
        )
    return records


def test_conflict_verdict_recovers_planted_gradient_structure(tmp_path):
    import pandas as pd

    summary = run_conflict_verdict(
        pd.DataFrame(_synthetic_gradient_pair_batches()),
        tmp_path,
        min_mean_support=8,
        n_bootstrap=500,
        seed=4,
    )

    pairs = pd.read_csv(tmp_path / "conflict_pair_verdicts.csv").set_index(
        ["outcome_a", "outcome_b"]
    )
    assert (
        pairs.loc[
            ("conflict_a", "conflict_b"),
            "classification",
        ]
        == "genuine_conflict"
    )
    assert (
        pairs.loc[
            ("aligned_a", "aligned_b"),
            "classification",
        ]
        == "genuine_alignment"
    )
    assert (
        pairs.loc[
            ("low_support_a", "low_support_b"),
            "classification",
        ]
        == "indeterminate"
    )
    assert not pairs.loc[
        ("low_support_a", "low_support_b"),
        "adequate_support",
    ]
    assert summary["n_adequately_supported_pairs"] == 10
    assert summary["conflict_structure"]["coherent_conflict_group"]
    assert (tmp_path / "conflict_verdict.json").exists()
    assert (tmp_path / "conflict_verdict.txt").exists()
    assert (tmp_path / "conflict_pair_intervals.png").exists()
    assert (tmp_path / "conflict_pair_intervals.pdf").exists()
