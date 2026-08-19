import torch

from bonsai.modules.networks.components.embeddings import EhrEmbeddings


def _inputs():
    return dict(
        code=torch.tensor([[1, 2, 0]]),
        age=torch.zeros((1, 3)),
        abspos=torch.tensor([[400000.0, 401000.0, 0.0]]),
        segment=torch.tensor([[0, 1, 0]]),
    )


def test_none_abspos_is_invariant_to_calendar_shift():
    model = EhrEmbeddings(5, 8, 8, abspos_encoding="none")
    inputs = _inputs()
    shifted = dict(inputs, abspos=inputs["abspos"] + 5 * 8766)
    assert torch.allclose(model(**inputs), model(**shifted))


def test_sequence_relative_fourier_is_invariant_to_global_calendar_shift():
    model = EhrEmbeddings(5, 8, 8, abspos_encoding="sequence_relative_fourier")
    inputs = _inputs()
    shifted_abspos = inputs["abspos"].clone()
    shifted_abspos[:, :2] += 5 * 8766
    shifted = dict(inputs, abspos=shifted_abspos)
    assert torch.allclose(model(**inputs)[:, :2], model(**shifted)[:, :2], atol=1e-5)


def test_gap_encoding_responds_to_event_spacing_not_calendar_era():
    model = EhrEmbeddings(
        5, 8, 8, abspos_encoding="sequence_relative_fourier_with_gaps"
    )
    inputs = dict(
        code=torch.tensor([[1, 2, 3, 0]]),
        age=torch.zeros((1, 4)),
        abspos=torch.tensor([[399000.0, 400000.0, 401000.0, 0.0]]),
        segment=torch.tensor([[0, 1, 2, 0]]),
    )
    wider = dict(inputs, abspos=torch.tensor([[399000.0, 400000.0, 402000.0, 0.0]]))
    assert not torch.allclose(model(**inputs)[:, 2], model(**wider)[:, 2])
