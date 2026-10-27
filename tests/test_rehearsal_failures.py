import pandas as pd
import pytest
import torch

from opera.run.explain_direct_competing_risk import Explainer
from opera.run.render_clinical_score_panels import expand_index_scores


def test_missing_computed_score_is_skipped_not_fatal():
    metadata = pd.DataFrame({"subject_id": [1, 2], "cohort_fine": ["DLBCL", "FL"]})
    config = {"scores": [{"id": "ipi_computed_all", "cohorts": "all",
                          "group_column": "ipi_computed_group"}],
              "all_indices": {"index_column": "index", "template": {"group_column": "IPI"}}}
    specs = expand_index_scores(config, metadata)
    assert [s["id"] for s in specs] == ["ipi_computed_all"] and specs[0]["cohorts"] == []


def test_attribution_halves_its_token_budget_on_cuda_oom():
    explainer = Explainer.__new__(Explainer)  # only the budget logic is exercised
    explainer.token_budget = 4096
    calls = []

    def compute():
        calls.append(explainer._chunk(100))
        if explainer.token_budget > 1024:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "done"

    assert explainer._on_oom(compute) == "done"
    assert calls == [40, 20, 10] and explainer.token_budget == 1024
    explainer.token_budget = 600
    with pytest.raises(torch.cuda.OutOfMemoryError):
        explainer._on_oom(lambda: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")))


def test_gap_before_prediction_hides_only_the_last_days():
    from opera.evaluation.input_restriction import restrict_sample
    hours = 24.0
    # background (age) at birth, events 40, 20, 10 and 3 days before the index,
    # and the prediction token ([CLS], id 1) at the index.
    sample = {"code": torch.tensor([5, 10, 11, 12, 13, 1]),
              "abspos": torch.tensor([0.0, 60.0, 80.0, 90.0, 97.0, 100.0]) * hours,
              "outcome_x": torch.tensor(1)}
    kept = restrict_sample(sample, protected_ids={1, 5}, drop_last_days=14)
    assert kept["code"].tolist() == [5, 10, 11, 1]  # 10 and 3 days before: hidden
    assert kept["outcome_x"].item() == 1
    kept = restrict_sample(sample, protected_ids={1, 5}, drop_last_days=28)
    assert kept["code"].tolist() == [5, 10, 1]
