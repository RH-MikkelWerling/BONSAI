from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from bonsai.functional.censoring import censor_subject
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.subject_data import clone_subject
from bonsai.functional.truncation import infer_background_length, truncate_subject
from bonsai.functional.input_contract import validate_numeric_value_control


class FinetuneDataset(Dataset):
    def __init__(
        self,
        subjects: List[Dict],
        outcomes: Dict[int, dict],
        predict_token_id: int,
        background_length: Optional[int],
        max_len: int,
        numeric_value_control: str = "observed",
    ):
        self.subjects = subjects
        self.outcomes = outcomes
        self.predict_token_id = predict_token_id
        self.background_length = background_length
        self.max_len = max_len
        self.numeric_value_control = validate_numeric_value_control(
            numeric_value_control
        )

    def __getitem__(self, index: int) -> dict:
        subject = clone_subject(self.subjects[index])
        background_length = (
            infer_background_length(subject)
            if self.background_length is None
            else int(self.background_length)
        )
        if (
            self.numeric_value_control == "masked"
            and "numeric_value" in subject
        ):
            subject["numeric_value"] = torch.full_like(
                subject["numeric_value"], float("nan")
            )
        subject_outcome = self.outcomes[subject["subject_id"]]

        subject["target"] = torch.tensor([subject_outcome["label"]], dtype=torch.long)

        if subject_outcome["censor_abspos"] is not None:
            subject = censor_subject(
                subject,
                censor_date_abspos=subject_outcome["censor_abspos"],
                predict_token_id=self.predict_token_id,
            )
        subject = truncate_subject(
            subject, max_len=self.max_len, background_length=background_length
        )

        subject["segment"] = normalize_segments(subject["segment"])
        subject["attention_mask"] = torch.ones(len(subject["code"]), dtype=torch.bool)

        return subject

    def __len__(self):
        return len(self.subjects)
