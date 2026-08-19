import numpy as np
import pandas as pd
import torch
from torch import nn

from opera.evaluation.vocabulary_learning import (
    count_token_exposure,
    neighbour_coherence,
    stratified_token_sample,
    token_geometry,
    vocabulary_coverage,
)
from opera.evaluation.treatment_embeddings import embedding_columns
from opera.run.diagnose_vocabulary_learning import (
    contextual_probes,
    contextual_sensitivity,
    token_losses,
)


def _embedding_frame(values):
    return pd.DataFrame(
        {
            "token_id": [0, 1, 2],
            "token": ["[PAD]", "LAB//A", "LAB//B"],
            "token_family": ["special", "LAB", "LAB"],
            "embedding_0": [row[0] for row in values],
            "embedding_1": [row[1] for row in values],
            "embedding_norm": [np.linalg.norm(row) for row in values],
        }
    )


def test_exposure_and_coverage_include_unseen_tokens():
    subjects = [
        {"code": torch.tensor([0, 1, 1]), "subject_id": 1},
        {"code": torch.tensor([0, 1]), "subject_id": 2},
    ]
    exposure = count_token_exposure(subjects, {0: "[PAD]", 1: "LAB//A", 2: "LAB//B"})
    assert exposure.set_index("token").loc["LAB//A", "n_occurrences"] == 3
    assert exposure.set_index("token").loc["LAB//A", "n_subjects"] == 2
    assert exposure.set_index("token").loc["LAB//B", "n_occurrences"] == 0

    coverage = vocabulary_coverage(exposure, [1, 3, 4]).set_index("minimum_occurrences")
    assert coverage.loc[1, "retained_tokens"] == 2
    assert coverage.loc[4, "retained_tokens"] == 0


def test_geometry_reports_exact_optional_movement():
    exposure = pd.DataFrame(
        {
            "token_id": [0, 1, 2],
            "token": ["[PAD]", "LAB//A", "LAB//B"],
            "n_occurrences": [10, 3, 0],
            "n_subjects": [2, 2, 0],
            "subject_fraction": [1.0, 1.0, 0.0],
        }
    )
    initial = _embedding_frame([[1, 0], [0, 1], [1, 1]])
    trained = _embedding_frame([[1, 0], [1, 1], [1, 1]])
    result = token_geometry(trained, exposure, initial).set_index("token")
    assert result.loc["LAB//A", "l2_from_initial"] == 1.0
    assert result.loc["LAB//B", "l2_from_initial"] == 0.0
    assert str(result.loc["LAB//B", "frequency_stratum"]) == "unseen"


def test_embedding_columns_excludes_derived_embedding_statistics():
    frame = _embedding_frame([[1, 0], [0, 1], [1, 1]])
    assert embedding_columns(frame) == ["embedding_0", "embedding_1"]


def test_neighbour_coherence_summarizes_by_frequency():
    geometry = pd.DataFrame(
        {
            "token_id": [1, 2],
            "n_occurrences": [3, 120],
            "frequency_stratum": pd.Categorical(["2-4", "100-499"]),
        }
    )
    neighbours = pd.DataFrame(
        {
            "token_id": [1, 1, 2, 2],
            "same_family": [True, False, True, True],
            "cosine_similarity": [0.9, 0.8, 0.7, 0.6],
        }
    )
    result = neighbour_coherence(neighbours, geometry).set_index("frequency_stratum")
    assert result.loc["2-4", "same_family_precision_at_k"] == 0.5
    assert result.loc["100-499", "same_family_precision_at_k"] == 1.0


def test_stratified_token_sample_preserves_frequency_groups():
    frame = pd.DataFrame(
        {
            "token_id": range(12),
            "frequency_stratum": pd.Categorical(
                ["unseen"] * 6 + ["1000+"] * 6
            ),
        }
    )
    sampled = stratified_token_sample(frame, max_tokens=4, seed=3)
    assert len(sampled) == 4
    assert set(sampled["frequency_stratum"].astype(str)) == {"unseen", "1000+"}


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.code_embedding = nn.Embedding.from_pretrained(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )

    def forward(self, batch):
        hidden = self.embeddings.code_embedding(batch["code"]).clone()
        values = torch.nan_to_num(batch["numeric_value"], nan=0.0)
        hidden[..., 0] += values
        hidden[..., 1] += batch["abspos"] / 1000.0
        return hidden


def _tiny_batch():
    return {
        "subject_id": torch.tensor([10, 11]),
        "code": torch.tensor([[1, 2], [2, 1]]),
        "target": torch.tensor([[2, -100], [1, -100]]),
        "numeric_value": torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
        "abspos": torch.tensor([[1000.0, 2000.0], [3000.0, 4000.0]]),
        "segment": torch.ones((2, 2), dtype=torch.long),
        "age": torch.zeros((2, 2)),
        "attention_mask": torch.ones((2, 2), dtype=torch.bool),
    }


def test_contextual_diagnostics_emit_counterfactuals_and_event_samples():
    details, summary, events = contextual_sensitivity(
        _TinyEncoder(), [_tiny_batch()], device=torch.device("cpu"), max_batches=1
    )
    assert set(summary["perturbation"]) == {
        "mask_numeric_values",
        "calendar_plus_5y",
        "reverse_event_context",
    }
    assert len(details) == 6
    assert len(events) == 4
    assert {"contextual_0", "raw_0", "numeric_value", "abspos"}.issubset(events)


def test_token_losses_are_grouped_by_target_token():
    result = token_losses(
        _TinyEncoder(),
        [_tiny_batch()],
        device=torch.device("cpu"),
        max_batches=1,
        logit_chunk_size=1,
    )
    assert set(result["token_id"]) == {1, 2}
    assert result["n_loss_targets"].sum() == 2


def test_contextual_probes_skip_too_small_samples():
    events = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "numeric_value": [1.0, 2.0],
            "abspos": [1000.0, 2000.0],
            "raw_0": [0.0, 1.0],
            "contextual_0": [1.0, 2.0],
        }
    )
    assert contextual_probes(events, seed=42).empty
