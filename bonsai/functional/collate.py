import torch
from typing import List, Dict


def _padding_value(key: str):
    if key in {"target", "target_value_bin"}:
        return -100
    if key == "target_value_mask":
        return False
    if key in {"numeric_value", "numeric_target"}:
        return float("nan")
    if key == "code_loss_weight":
        return 0.0
    return 0


def dynamic_padding(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collected = {key: [] for key in batch[0]}
    for sample in batch:
        for key, val in sample.items():
            collected[key].append(val)

    output = {}
    for key, values in collected.items():
        if key == "subject_id":
            output[key] = torch.tensor(values)
            continue
        first = values[0]
        if isinstance(first, torch.Tensor) and first.ndim > 0:
            output[key] = torch.nn.utils.rnn.pad_sequence(
                values,
                batch_first=True,
                padding_value=_padding_value(key),
            )
        elif isinstance(first, torch.Tensor):
            output[key] = torch.stack(values)
        else:
            output[key] = torch.tensor(values)

    return output
