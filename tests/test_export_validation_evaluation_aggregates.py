import json

from opera.run.export_validation_evaluation_aggregates import collect


def test_collect_exports_only_whitelisted_validation_scalars(tmp_path):
    root = tmp_path / "evaluation"
    for outcome, split in (("anemia_g3plus", "tuning"), ("overall_survival", "held_out")):
        directory = root / outcome
        directory.mkdir(parents=True)
        row = {
            "outcome": outcome,
            "split": split,
            "auroc": 0.75,
            "cif_auc_365d": 0.72,
            "subject_id": "sensitive_id",
            "predictions": [0.1, 0.9],
        }
        (directory / "result.jsonl").write_text(json.dumps(row) + "\n")
    rows, inventory = collect(root)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "anemia_g3plus"
    assert rows[0]["auroc"] == 0.75
    assert "subject_id" not in rows[0]
    assert "predictions" not in rows[0]
    assert {x["status"] for x in inventory} == {
        "included_validation", "excluded_nonvalidation"
    }


def test_collect_fails_closed_when_split_is_missing_or_outcome_mismatched(tmp_path):
    directory = tmp_path / "evaluation" / "anemia_g3plus"
    directory.mkdir(parents=True)
    path = directory / "result.jsonl"
    path.write_text(json.dumps({"outcome": "anemia_g3plus", "auroc": 0.8}) + "\n")
    assert collect(directory.parent)[0] == []
    path.write_text(json.dumps({"outcome": "other", "split": "tuning"}) + "\n")
    assert collect(directory.parent)[0] == []
