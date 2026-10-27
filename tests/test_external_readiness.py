import json

import numpy as np
import pandas as pd
import pytest
import torch

from bonsai.modules.networks.bonsai_nets import BonsaiPretrain
from opera.run.build_paper_patient_metadata import national_coverage
from opera.run.check_external_checkpoint import check
from opera.run.prepare_external_sequences import audit_sequences

VOCAB = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4, "BG_SEX//F": 5,
         "DIAGNOSIS//A": 6, "MEDICATION//B": 7}


def _checkpoint(tmp_path, *, hidden_size=256, vocab=VOCAB, rows=None):
    model = BonsaiPretrain(vocab_size=rows or len(vocab), max_seqlen=16,
                           hidden_size=hidden_size, num_layers=1, num_attention_heads=2,
                           bias=False, dropout=0.0, attention_dropout=0.0, causal=True,
                           attn_type="sdpa", abspos_encoding="fourier")
    path, vocab_path = tmp_path / "national.ckpt", tmp_path / "vocabulary.pt"
    torch.save({"hyper_parameters": {"model_config": model.hparams},
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
    torch.save(vocab, vocab_path)
    return path, vocab_path


def test_preflight_accepts_a_matching_checkpoint(tmp_path):
    report = check(*_checkpoint(tmp_path), "daly_care_joined_bins_data")
    assert report["problems"] == []
    assert report["encoder"]["hidden_size"] == 256 and report["numeric_value_mode"] == "joined_binning"


def test_preflight_names_wrong_width_and_vocabulary(tmp_path):
    path, _ = _checkpoint(tmp_path, hidden_size=8)
    other = tmp_path / "other.pt"
    torch.save({**VOCAB, "LABTEST//C": 8}, other)
    problems = " ".join(check(path, other, "daly_care_joined_bins_data")["problems"])
    assert "hidden size 8" in problems and "wrong vocabulary file" in problems


def _sequences(tmp_path, subjects, vocab=VOCAB):
    sequences = tmp_path / "sequences"
    sequences.mkdir()
    torch.save(vocab, sequences / "vocabulary.pt")
    torch.save(subjects, sequences / "subject_data_train.pt")
    torch.save([], sequences / "subject_data_tuning.pt")
    (tmp_path / "filter_report.json").write_text(json.dumps({"latest_event": "2023-06-30"}))
    membership = tmp_path / "membership.csv"
    return sequences, membership


def test_audit_fails_on_missing_patients_vocabulary_and_short_extract(tmp_path):
    sequences, membership = _sequences(tmp_path, [{"subject_id": 1, "code": [6, 3, 7]}])
    pd.DataFrame({"subject_id": [1, 2]}).to_csv(membership, index=False)
    with pytest.raises(SystemExit, match="1 paper patients have no national sequence"):
        audit_sequences(tmp_path, membership)
    report = audit_sequences(tmp_path, membership, allow_missing=True)
    assert report["unknown_token_share"] == pytest.approx(1 / 3)
    other = tmp_path / "other.pt"
    torch.save({**VOCAB, "DIAGNOSIS//A": 7, "MEDICATION//B": 6}, other)  # same size, reordered
    with pytest.raises(SystemExit, match="vocabulary differs"):
        audit_sequences(tmp_path, membership, external_vocab=other, allow_missing=True)
    with pytest.raises(SystemExit, match="before 2024-12-31"):
        audit_sequences(tmp_path, membership, required_latest="2024-12-31", allow_missing=True)


def test_national_coverage_counts_unknown_pre_index_tokens_only():
    subjects = [{"subject_id": 1, "code": [1, 5, 6, 3, 3, 7, 3],
                 "abspos": [0, 0, 1, 2, 3, 4, 9]}]
    coverage = national_coverage(subjects, {1: 5.0}, VOCAB)
    # Before the index: DIAGNOSIS, [UNK], [UNK], MEDICATION (CLS and background aside).
    assert coverage.national_vocabulary_coverage.iloc[0] == pytest.approx(0.5)
    assert np.isnan(national_coverage(subjects, {1: -1.0}, VOCAB)
                    .national_vocabulary_coverage.iloc[0])
