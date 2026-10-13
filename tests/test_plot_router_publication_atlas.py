import numpy as np
import pytest

from opera.run.plot_router_publication_atlas import summarize


def test_equal_offhome_weights_have_unit_preference_despite_unequal_bank_sizes():
    diseases = ["A", "B", "C"]
    outcomes = ["one", "two", "three", "four"]
    means = np.zeros((3, 4, 1 + 3 + 4))
    means[:, :, 0] = .2
    means[:, :, 1:4] = .1
    means[:, :, 4:] = .125
    counts = np.full((3, 4), 25)
    result = summarize(means, counts, diseases, outcomes,
                       {"Early": ["one", "two"], "Late": ["three", "four"]}, 20)
    assert np.allclose(result["other_disease_mass"], .2)
    assert np.allclose(result["other_outcome_mass"], .375)
    assert np.allclose(result["disease_source_lift"][~np.eye(3, dtype=bool)], 1)
    assert np.allclose(result["outcome_source_lift"][~np.eye(4, dtype=bool)], 1)
    assert np.allclose(result["outcome_family_source_lift"], 1)


def test_unavailable_tasks_are_excluded_and_family_partition_is_checked():
    means = np.full((2, 2, 5), .2)
    counts = np.array([[25, 0], [25, 25]])
    result = summarize(means, counts, ["A", "B"], ["one", "two"],
                       {"Family": ["one", "two"]}, 20)
    assert np.isfinite(result["other_disease_mass"][0, 0])
    with pytest.raises(ValueError, match="partition"):
        summarize(means, counts, ["A", "B"], ["one", "two"],
                  {"Family": ["one"]}, 20)
