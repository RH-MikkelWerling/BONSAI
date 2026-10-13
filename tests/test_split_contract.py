import pandas as pd
import torch

from bonsai.functional.outcomes import apply_prospective_split
from opera.evaluation.split_contract import (
    load_temporal_split_contract,
    validate_cross_stage_split_contract,
)


def _subject(subject_id: int) -> dict:
    return {
        "subject_id": subject_id,
        "code": torch.tensor([1, 2, 3]),
        "abspos": torch.tensor([0.0, 1.0, 2.0]),
        "segment": torch.tensor([0, 1, 1]),
    }


def test_canonical_temporal_contract_assigns_2021_2022_2023plus_splits():
    contract = load_temporal_split_contract()
    outcomes = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "index_date": pd.to_datetime(
                ["2021-12-31", "2022-06-01", "2022-12-31", "2023-01-01"]
            ),
        }
    )

    split = apply_prospective_split(outcomes, **contract)

    assert split.set_index("subject_id")["split"].to_dict() == {
        1: "train",
        2: "tuning",
        3: "tuning",
        4: "held_out",
    }


def test_cross_stage_split_contract_treats_subject_files_as_physical_partitions(
    tmp_path,
):
    outcome_path = tmp_path / "mortality.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "index_date": pd.to_datetime(["2021-01-01", "2022-01-01", "2024-01-01"]),
            "split": ["train", "tuning", "held_out"],
        }
    ).to_parquet(outcome_path, index=False)
    train_subjects = tmp_path / "subject_data_train.pt"
    tuning_subjects = tmp_path / "subject_data_tuning.pt"
    held_out_subjects = tmp_path / "subject_data_held_out.pt"
    embedding_store = tmp_path / "dapt_embeddings.pt"
    torch.save([_subject(1)], train_subjects)
    torch.save([_subject(2)], tuning_subjects)
    torch.save([_subject(3)], held_out_subjects)
    torch.save({3: torch.ones(4)}, embedding_store)

    report = validate_cross_stage_split_contract(
        outcome_paths=[outcome_path],
        subject_data_paths={
            "train": train_subjects,
            "tuning": tuning_subjects,
            "held_out": held_out_subjects,
        },
        embedding_store_paths=[embedding_store],
    )

    assert report["ok"] is True
    assert report["issues"] == []
    assert (
        report["details"]["embedding_stores"][str(embedding_store)][
            "n_held_out_subjects"
        ]
        == 1
    )


def test_cross_stage_split_contract_flags_missing_pooled_subjects(tmp_path):
    outcome_path = tmp_path / "mortality.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "index_date": pd.to_datetime(["2021-01-01", "2022-01-01", "2024-01-01"]),
            "split": ["train", "tuning", "held_out"],
        }
    ).to_parquet(outcome_path, index=False)
    physical_train = tmp_path / "subject_data_train.pt"
    torch.save([_subject(1), _subject(2)], physical_train)

    report = validate_cross_stage_split_contract(
        outcome_paths=[outcome_path],
        subject_data_paths={"ssl_train": physical_train},
    )

    assert report["ok"] is False
    assert any("missing 1 outcome subjects" in issue for issue in report["issues"])
    assert all("[3]" not in issue for issue in report["issues"])


def test_cross_stage_split_contract_reports_overlap_counts_without_ids(tmp_path):
    outcome_path = tmp_path / "mortality.parquet"
    pd.DataFrame(
        {
            "subject_id": [1001, 1001],
            "index_date": pd.to_datetime(["2021-01-01", "2023-01-01"]),
            "split": ["train", "held_out"],
        }
    ).to_parquet(outcome_path, index=False)
    train_path = tmp_path / "train.pt"
    held_path = tmp_path / "held.pt"
    torch.save([_subject(1001)], train_path)
    torch.save([_subject(1001)], held_path)

    report = validate_cross_stage_split_contract(
        outcome_paths=[outcome_path],
        subject_data_paths={"train": train_path, "held_out": held_path},
    )

    assert not report["ok"]
    assert any("1 subjects" in issue for issue in report["issues"])
    assert "1001" not in str(report)


def test_cross_stage_split_contract_aggregate_errors_in_memory(monkeypatch):
    from opera.evaluation import split_contract

    frame = pd.DataFrame(
        {
            "subject_id": [1001, 1001],
            "index_date": pd.to_datetime(["2021-01-01", "2023-01-01"]),
            "split": ["train", "held_out"],
        }
    )
    monkeypatch.setattr(split_contract, "_read_outcome", lambda path: frame)
    monkeypatch.setattr(
        split_contract, "_load_subject_ids", lambda path: {1001}
    )
    report = validate_cross_stage_split_contract(
        outcome_paths=["synthetic.parquet"],
        subject_data_paths={"train": "train.pt", "held_out": "held.pt"},
    )
    assert not report["ok"]
    assert any("1 subjects" in issue for issue in report["issues"])
    assert "1001" not in str(report)
