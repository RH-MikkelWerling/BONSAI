import torch

from opera.functional.linear_probe import freeze_encoder_for_linear_probe


class TinyProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.finetune_head = torch.nn.Linear(2, 1)


class TinyStrictLinearProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.classifier = torch.nn.Linear(2, 1)


def test_freeze_encoder_for_linear_probe_leaves_only_head_trainable():
    model = TinyProbeModel()

    metadata = freeze_encoder_for_linear_probe(model)

    assert metadata["encoder_frozen"] is True
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.finetune_head.parameters())
    assert metadata["n_trainable_parameters"] == 2
    assert metadata["n_frozen_parameters"] == 2


def test_freeze_encoder_for_strict_linear_probe_uses_classifier_prefix():
    model = TinyStrictLinearProbe()

    metadata = freeze_encoder_for_linear_probe(
        model,
        trainable_prefixes=("classifier.",),
    )

    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert metadata["trainable_parameter_names"] == [
        "classifier.weight",
        "classifier.bias",
    ]
