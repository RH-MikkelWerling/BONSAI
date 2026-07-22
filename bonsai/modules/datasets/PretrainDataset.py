from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from bonsai.functional.censoring import censor_subject
from bonsai.functional.features import compute_abspos
from bonsai.functional.normalization import normalize_segments
from bonsai.functional.subject_data import clone_subject
from bonsai.functional.truncation import sequence_tensor_fields, truncate_subject


class PretrainDataset(Dataset):
    def __init__(
        self,
        subjects: List[Dict],
        max_len: int,
        background_length: int,
        cutoff_date: Optional[dict] = None,
        truncation_strategy: str = "tail",
        tail_window_probability: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ):
        self.subjects = subjects
        self.max_len = max_len
        self.background_length = background_length
        self.truncation_strategy = truncation_strategy
        self.tail_window_probability = tail_window_probability
        self.generator = generator
        self.cutoff_date = (
            compute_abspos(datetime(**cutoff_date)) if cutoff_date is not None else None
        )

    def _prepare_subject(self, index: int) -> tuple[dict, dict]:
        subject = clone_subject(self.subjects[index])
        if self.cutoff_date is not None:
            subject = censor_subject(subject, self.cutoff_date, inclusive=False)
        truncated_subject, truncation_metadata = truncate_subject(
            subject,
            self.max_len,
            self.background_length,
            strategy=self.truncation_strategy,
            tail_window_probability=self.tail_window_probability,
            generator=self.generator,
            return_metadata=True,
        )
        truncated_subject["attention_mask"] = torch.ones(
            len(truncated_subject["code"]), dtype=torch.bool
        )
        truncated_subject["segment"] = normalize_segments(truncated_subject["segment"])
        return truncated_subject, truncation_metadata

    def __getitem__(self, index: int) -> dict:
        truncated_subject, _ = self._prepare_subject(index)
        return truncated_subject

    def __len__(self):
        return len(self.subjects)


class MLMPretrainDataset(PretrainDataset):
    def __init__(
        self,
        subjects: List[Dict],
        max_len: int,
        background_length: int,
        vocabulary: Dict[str, int],
        masking_select_ratio: float,
        masking_mask_ratio: float = 0.8,
        masking_random_ratio: float = 0.1,
        masking_ignore_special_tokens: bool = True,
        cutoff_date: Optional[dict] = None,
        truncation_strategy: str = "tail",
        tail_window_probability: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(
            subjects,
            max_len,
            background_length,
            cutoff_date=cutoff_date,
            truncation_strategy=truncation_strategy,
            tail_window_probability=tail_window_probability,
            generator=generator,
        )
        self.vocabulary = vocabulary

        self.masking_select_ratio = masking_select_ratio
        self.masking_mask_ratio = masking_mask_ratio
        self.masking_random_ratio = masking_random_ratio
        self.masking_n_special_tokens = (
            len([token for token in vocabulary if token.startswith("[")])
            if masking_ignore_special_tokens
            else 0
        )

    def __getitem__(self, index: int) -> dict:
        subject, _ = self._prepare_subject(index)
        masked_codes, target, selected_indices = self.mask_patient_codes(
            subject["code"]
        )
        self._prepare_value_targets(subject, selected_indices)
        subject["code"] = masked_codes
        subject["target"] = target
        return subject

    def mask_patient_codes(
        self, codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = codes.clone()
        probability_vector = torch.full(target.shape, self.masking_select_ratio)

        # Ignore special tokens
        special_token_mask = codes < self.masking_n_special_tokens
        probability_vector.masked_fill_(special_token_mask, value=0.0)

        # Get MLM mask
        selected_indices = torch.bernoulli(probability_vector).bool()
        target[~selected_indices] = -100

        # Replace with [MASK]
        indices_mask = (
            torch.bernoulli(torch.full(target.shape, self.masking_mask_ratio)).bool()
            & selected_indices
        )
        codes[indices_mask] = self.vocabulary["[MASK]"]

        # Replace with random word and Account for already masked tokens
        random_ratio = self.masking_random_ratio / (1 - self.masking_mask_ratio)
        indicies_random = (
            torch.bernoulli(torch.full(target.shape, random_ratio)).bool()
            & selected_indices
            & ~indices_mask
        )
        random_words = torch.randint(
            self.masking_n_special_tokens,
            len(self.vocabulary),
            target.shape,
            dtype=codes.dtype,
        )
        codes[indicies_random] = random_words[indicies_random]
        return codes, target, selected_indices

    def _prepare_value_targets(
        self,
        subject: dict,
        selected_indices: torch.Tensor,
    ) -> None:
        if not {
            "value_bin",
            "value_normalized",
            "value_present",
        }.issubset(subject):
            return
        value_mask = selected_indices & subject["value_present"].bool()
        subject["target_value_mask"] = value_mask
        subject["target_value_bin"] = subject["value_bin"].clone()
        subject["target_value_bin"][~value_mask] = -100
        subject["target_value_normalized"] = subject["value_normalized"].clone()
        subject["target_value_normalized"][~value_mask] = 0.0
        subject["value_bin"] = subject["value_bin"].clone()
        subject["value_normalized"] = subject["value_normalized"].clone()
        subject["value_present"] = subject["value_present"].clone()
        subject["value_bin"][value_mask] = 0
        subject["value_normalized"][value_mask] = 0.0
        subject["value_present"][value_mask] = False


class ARPretrainDataset(PretrainDataset):
    def __init__(
        self,
        subjects: List[Dict],
        max_len: int,
        background_length: int,
        cutoff_date: Optional[dict] = None,
        truncation_strategy: str = "tail",
        tail_window_probability: float = 0.5,
        generator: Optional[torch.Generator] = None,
        vocabulary: Optional[Dict[str, int]] = None,
        value_embedding_mode: str = "legacy",
    ):
        super().__init__(
            subjects,
            max_len + 1,
            background_length,
            cutoff_date=cutoff_date,
            truncation_strategy=truncation_strategy,
            tail_window_probability=tail_window_probability,
            generator=generator,
        )  # +1 because we shift by one token in __getitem__
        self.val_token_id = None if vocabulary is None else vocabulary.get("[VAL]")
        self.value_embedding_mode = value_embedding_mode

    def __getitem__(self, index: int) -> dict:
        subject, truncation_metadata = self._prepare_subject(index)
        subject["target"] = subject["code"][1:]
        subject["target"] = subject["target"].masked_fill(subject["target"] == 0, -100)
        has_values = {
            "value_bin",
            "value_normalized",
            "value_present",
        }.issubset(subject)
        if has_values:
            value_mask = subject["value_present"][1:].bool()
            subject["target_value_mask"] = value_mask
            subject["target_value_bin"] = subject["value_bin"][1:].clone()
            subject["target_value_bin"][~value_mask] = -100
            subject["target_value_normalized"] = subject["value_normalized"][1:].clone()
            subject["target_value_normalized"][~value_mask] = 0.0
            if self.value_embedding_mode == "combined_binning":
                # In causal training, the state at the owning lab event
                # predicts the following [VAL] scalar. Avoid an additional,
                # nearly trivial CE target for the shared marker token.
                if self.val_token_id is None:
                    raise ValueError(
                        "combined_binning requires [VAL] in the vocabulary."
                    )
                value_mask = value_mask & (subject["code"][1:] == self.val_token_id)
                subject["target_value_mask"] = value_mask
                subject["target_value_bin"][~value_mask] = -100
                subject["target"][subject["code"][1:] == self.val_token_id] = -100
        if truncation_metadata["clinical_window_started_mid_history"]:
            boundary_target = self.background_length - 1
            if 0 <= boundary_target < len(subject["target"]):
                subject["target"][boundary_target] = -100
                if has_values:
                    subject["target_value_mask"][boundary_target] = False
                    subject["target_value_bin"][boundary_target] = -100
                    subject["target_value_normalized"][boundary_target] = 0.0
        for key in sequence_tensor_fields(subject):
            subject[key] = subject[key][:-1]
        return subject
