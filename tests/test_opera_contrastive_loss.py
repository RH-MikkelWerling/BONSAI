import torch

from opera.modules.networks.opera_nets import MultiOutcomeSurvivalLoss

MOCK_SORTED_EVENTS = {"mortality": torch.tensor([5.0, 10.0, 20.0, 30.0, 40.0])}


def test_missing_dapt_embeddings_are_neutral_not_moderate_similarity():
    loss = MultiOutcomeSurvivalLoss(
        ["mortality"],
        outcome_sorted_event_times=MOCK_SORTED_EVENTS,
        dapt_lambda_floor=0.3,
    )
    store = {
        1: torch.tensor([1.0, 0.0]),
        2: torch.tensor([0.0, 1.0]),
    }
    subject_ids = torch.tensor([1, 2, 3])

    weights, known = loss._compute_dapt_weights(subject_ids, store)

    assert known.tolist() == [True, True, False]
    assert weights[0, 2].item() == 1.0
    assert weights[2, 0].item() == 1.0
    assert weights[1, 2].item() == 1.0
    assert weights[2, 1].item() == 1.0
    assert weights[0, 1].item() < 1.0


def test_dapt_diagnostics_are_logged_when_prior_is_active():
    loss = MultiOutcomeSurvivalLoss(
        ["mortality"],
        outcome_sorted_event_times=MOCK_SORTED_EVENTS,
        dapt_lambda_floor=0.3,
    )
    embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    outcome_survival = {
        "mortality": {
            "times": torch.tensor([10.0, 20.0, 30.0]),
            "events": torch.tensor([1, 0, 1]),
        }
    }
    store = {
        10: torch.tensor([1.0, 0.0]),
        11: torch.tensor([0.0, 1.0]),
        12: torch.tensor([1.0, 1.0]),
    }

    logs = loss(
        embeddings,
        outcome_survival,
        subject_ids=torch.tensor([10, 11, 12]),
        dapt_embedding_store=store,
    )

    assert "dapt/weight_mean" in logs
    assert "dapt/weight_std" in logs
    assert "dapt/coverage" in logs
    assert logs["dapt/coverage"].item() == 1.0
    assert logs["loss"].requires_grad
