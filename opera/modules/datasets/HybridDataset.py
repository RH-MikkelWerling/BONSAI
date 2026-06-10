"""
Dataset for the hybrid model that pairs EHR sequences with tabular features.

The tabular features (e.g. RKKP variables) are loaded from a separate
CSV/parquet file keyed by subject_id.
"""

from typing import List, Dict
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from bonsai.functional.censoring import censor_subject
from bonsai.functional.truncation import truncate_subject
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.subject_data import clone_subject


class HybridDataset(Dataset):
    """
    Parameters
    ----------
    subjects : list of dicts
    outcomes : dict[int, dict]
    tabular_df : pd.DataFrame
        Must have 'subject_id' column + feature columns.
        Missing values are filled with 0.0.
    feature_columns : list of str
        Columns in tabular_df to use as features.
    predict_token_id : int
    background_length : int
    max_len : int
    """

    def __init__(
        self,
        subjects: List[Dict],
        outcomes: Dict[int, dict],
        tabular_df: pd.DataFrame,
        feature_columns: List[str],
        predict_token_id: int,
        background_length: int,
        max_len: int = 8192,
    ):
        self.subjects = subjects
        self.outcomes = outcomes
        self.predict_token_id = predict_token_id
        self.background_length = background_length
        self.max_len = max_len
        self.feature_columns = feature_columns

        # Build lookup: subject_id → numpy array of features
        tabular_df = tabular_df.set_index("subject_id")
        tabular_df = tabular_df[feature_columns].fillna(0.0)
        self.tabular_lookup = {
            sid: row.values.astype(np.float32) for sid, row in tabular_df.iterrows()
        }
        self.default_tabular = np.zeros(len(feature_columns), dtype=np.float32)

    def __getitem__(self, index: int) -> dict:
        subject = clone_subject(self.subjects[index])
        sid = subject["subject_id"]
        subject_outcome = self.outcomes[sid]

        subject["target"] = torch.tensor([subject_outcome["label"]], dtype=torch.long)

        if not pd.isnull(subject_outcome["censor_abspos"]):
            subject = censor_subject(
                subject,
                censor_date_abspos=subject_outcome["censor_abspos"],
                predict_token_id=self.predict_token_id,
            )
        subject = truncate_subject(
            subject, max_len=self.max_len, background_length=self.background_length
        )
        subject["segment"] = normalize_segments(subject["segment"])
        subject["attention_mask"] = torch.ones(len(subject["code"]), dtype=torch.long)

        # Attach tabular features
        tab = self.tabular_lookup.get(sid, self.default_tabular)
        subject["tabular"] = torch.from_numpy(tab)

        return subject

    def __len__(self):
        return len(self.subjects)
