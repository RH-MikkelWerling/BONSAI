"""
Dataset for OPERA contrastive learning.

Each sample is a standard BONSAI subject dict augmented with survival
fields per outcome:
    ``time_<n>``   float  time-to-event or censoring in days (-1 = missing)
    ``event_<n>``  int    1 = observed event, 0 = censored, -1 = missing

The binary ``outcome_<n>`` label is also retained for compatibility
with the ablation (binary SupCon) loss.
"""

from typing import List, Dict
import torch
import pandas as pd
from torch.utils.data import Dataset
from bonsai.functional.censoring import censor_subject
from bonsai.functional.truncation import truncate_subject
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.subject_data import clone_subject


class ContrastiveDataset(Dataset):
    """
    Parameters
    ----------
    subjects : list of dicts
        Standard BONSAI subject data (code, abspos, segment, age, subject_id).
    outcome_dicts : dict[str, dict[int, dict]]
        Mapping from outcome name -> {subject_id: outcome_record}.

        Each outcome_record must contain:
            "label"           : int  (0 | 1)
            "censor_abspos"   : float  (absolute position of censor/index date)

        And *should* contain for survival loss (falls back to label if absent):
            "time_days"       : float  time from index date to event or censoring
            "event"           : int    1 = observed, 0 = censored

        Subjects not present in a given outcome dict get missing sentinel
        values (-1) and are excluded from that outcome's loss.

    predict_token_id : int
    background_length : int
    max_len : int
    """

    def __init__(
        self,
        subjects: List[Dict],
        outcome_dicts: Dict[str, Dict[int, dict]],
        predict_token_id: int,
        background_length: int,
        max_len: int = 8192,
    ):
        self.subjects = subjects
        self.outcome_dicts = outcome_dicts
        self.outcome_names = sorted(outcome_dicts.keys())
        self.predict_token_id = predict_token_id
        self.background_length = background_length
        self.max_len = max_len
        self._validate_shared_prediction_origins()

    def _validate_shared_prediction_origins(self) -> None:
        """Require one prediction origin per patient across all outcomes."""
        origins: Dict[int, tuple[str, float]] = {}
        for outcome_name, records in self.outcome_dicts.items():
            for subject_id, record in records.items():
                value = record.get("censor_abspos")
                if value is None or pd.isnull(value):
                    continue
                origin = float(value)
                previous = origins.get(subject_id)
                if previous is None:
                    origins[subject_id] = (outcome_name, origin)
                    continue
                previous_name, previous_origin = previous
                if abs(previous_origin - origin) > 1e-6:
                    raise ValueError(
                        f"Patient {subject_id} has inconsistent prediction "
                        f"origins across outcomes {previous_name!r} "
                        f"({previous_origin}) and {outcome_name!r} ({origin})."
                    )

    def __getitem__(self, index: int) -> dict:
        subject = clone_subject(self.subjects[index])
        sid = subject["subject_id"]

        # Use the first available outcome's censor_abspos for sequence censoring
        censor_abspos = None
        for name in self.outcome_names:
            if sid in self.outcome_dicts[name]:
                c = self.outcome_dicts[name][sid].get("censor_abspos")
                if c is not None and not pd.isnull(c):
                    censor_abspos = c
                    break

        if censor_abspos is not None:
            subject = censor_subject(
                subject,
                censor_date_abspos=censor_abspos,
                predict_token_id=self.predict_token_id,
            )

        subject = truncate_subject(
            subject, max_len=self.max_len, background_length=self.background_length
        )
        subject["segment"] = normalize_segments(subject["segment"])
        subject["attention_mask"] = torch.ones(len(subject["code"]), dtype=torch.bool)

        # Attach survival fields per outcome
        for name in self.outcome_names:
            if sid in self.outcome_dicts[name]:
                rec = self.outcome_dicts[name][sid]
                label = int(rec["label"])
                time_days = float(rec.get("time_days", -1.0))
                event = int(rec.get("event", label))  # fallback: label IS event

                subject[f"outcome_{name}"] = torch.tensor(label, dtype=torch.long)
                subject[f"time_{name}"] = torch.tensor(time_days, dtype=torch.float)
                subject[f"event_{name}"] = torch.tensor(event, dtype=torch.long)
            else:
                subject[f"outcome_{name}"] = torch.tensor(-1, dtype=torch.long)
                subject[f"time_{name}"] = torch.tensor(-1.0, dtype=torch.float)
                subject[f"event_{name}"] = torch.tensor(-1, dtype=torch.long)

        # Dummy target for collate compatibility
        subject["target"] = torch.tensor([0], dtype=torch.long)

        return subject

    def __len__(self):
        return len(self.subjects)
