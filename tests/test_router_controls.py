"""Routing decomposition controls: no borrowing and disease identity only."""

import torch

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout

FINE = ["DLBCL", "FL", "MM"]
OUTCOMES = ["death", "infection"]


def readout(**kwargs):
    torch.manual_seed(0)
    module = SourceAnchoredReadout(8, FINE, OUTCOMES, bottleneck=2, residual_scale=0.5,
                                   home_floor=0.05, **kwargs)
    module.set_training_support([5] * 5, [50] * 5, [50] * 5, event_tau=10)
    return module


def test_defaults_are_parameter_identical_to_the_existing_router():
    assert set(readout().state_dict()) == set(
        readout(borrowing=True, disease_embedding=False).state_dict())
    assert "disease_embedding.weight" in readout(disease_embedding=True).state_dict()


def test_without_borrowing_only_own_disease_and_own_outcome_are_used():
    module = readout(source_mode="both", borrowing=False)
    patient, fine = torch.randn(4, 8), torch.tensor([0, 1, 2, 1])
    weights = module.route_weights(patient, fine)  # (batch, outcome, shared + sources)
    for b, f in enumerate(fine.tolist()):
        for o in range(len(OUTCOMES)):
            used = (weights[b, o, 1:] > 1e-6).nonzero().flatten().tolist()
            assert set(used) <= {f, len(FINE) + o}
    with_borrowing = readout(source_mode="both").route_weights(patient, fine)
    assert (with_borrowing[0, 0, 1:] > 1e-6).sum() > 2


def test_disease_embedding_control_adds_identity_and_nothing_else():
    module = readout(source_mode="none", disease_embedding=True)
    patient = torch.randn(3, 8)
    with torch.no_grad():
        module.disease_embedding.weight.copy_(torch.randn(3, 8))
    output = module(patient, torch.tensor([0, 1, -1]))
    expected = patient + module.disease_embedding.weight[[0, 1, 0]] * torch.tensor(
        [[1.0], [1.0], [0.0]])  # unknown disease gets no shift
    torch.testing.assert_close(output[:, 0], expected)
    torch.testing.assert_close(output[:, 1], expected)
