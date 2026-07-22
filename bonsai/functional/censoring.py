from bisect import bisect_left, bisect_right
from typing import Dict, Optional
import torch
from bonsai.functional.subject_data import clone_subject
from bonsai.functional.truncation import sequence_tensor_fields


def censor_subject(
    subject: Dict[str, torch.Tensor],
    censor_date_abspos: float,
    predict_token_id: Optional[int] = None,
    inclusive: bool = True,
) -> Dict:
    """
    Censors a subject's data by truncating all attributes at the censor date,
    OPTIONALLY: then appends a CLS token with the censoring information.
    """
    subject = clone_subject(subject)

    # Find the position where censor_date fits in the sorted abspos list
    boundary = subject["abspos"].numpy()
    idx = (
        bisect_right(boundary, censor_date_abspos)
        if inclusive
        else bisect_left(boundary, censor_date_abspos)
    )

    # Slice everything up to idx
    for embed_name in sequence_tensor_fields(subject):
        subject[embed_name] = subject[embed_name][:idx]

    if predict_token_id is not None:
        subject = append_predict_token(subject, censor_date_abspos, predict_token_id)

    return subject


def append_predict_token(
    subject: Dict, censor_date_abspos: float, predict_token_id: int
) -> Dict:
    extra_sequence_fields = [
        field
        for field in sequence_tensor_fields(subject)
        if field not in {"code", "abspos", "segment", "age"}
    ]
    subject["code"] = torch.cat(
        (
            subject["code"],
            torch.tensor([predict_token_id], dtype=subject["code"].dtype),
        )
    )
    subject["abspos"] = torch.cat(
        (
            subject["abspos"],
            torch.tensor([censor_date_abspos], dtype=subject["abspos"].dtype),
        )
    )
    subject["segment"] = torch.cat(
        (
            subject["segment"],
            torch.tensor(
                [subject["segment"][-1] + 1 if len(subject["segment"]) > 0 else 0],
                dtype=subject["segment"].dtype,
            ),
        )
    )

    age_in_years = float((censor_date_abspos - subject["abspos"][0]) / (365.25 * 24))
    subject["age"] = torch.cat(
        (
            subject["age"],
            torch.tensor(
                [age_in_years],
                dtype=subject["age"].dtype,
            ),
        )
    )
    for field in extra_sequence_fields:
        value = subject[field]
        subject[field] = torch.cat(
            (
                value,
                torch.zeros(1, dtype=value.dtype, device=value.device),
            )
        )
    return subject
