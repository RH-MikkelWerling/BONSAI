from opera.evaluation.results_schema import (
    REQUIRED_RESULT_FIELDS,
    bootstrap_ci_rows,
    build_result_row,
    canonical_training_stage,
)


class TinyCfg(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_build_result_row_contains_required_fields():
    cfg = TinyCfg(
        {
            "dataset": "dlbcl",
            "outcome": "mortality_1y",
            "labels": {"n_hours_end_include": 24 * 365},
            "encoder_source": "contrastive",
            "seed": 42,
            "rarity": {
                "mode": "real",
                "tier": "small_cell",
                "baseline_model": "tabular_ehr",
                "size_metadata": {
                    "n_train": 12,
                    "n_events_train": 3,
                    "prevalence_train": 0.25,
                },
            },
        }
    )
    report = {
        "discrimination": {
            "auroc": 0.8,
            "auprc": 0.4,
            "n_total": 100,
            "n_positive": 20,
            "prevalence": 0.2,
        },
        "calibration": {
            "brier_score": 0.12,
            "ece": 0.03,
        },
        "bootstrap_ci": {
            "auroc": {
                "mean": 0.79,
                "lower": 0.70,
                "upper": 0.86,
                "std": 0.04,
            },
        },
    }

    row = build_result_row(
        cfg,
        report,
        checkpoint_path="/tmp/model.ckpt",
        split="held_out",
    )

    for field in REQUIRED_RESULT_FIELDS:
        assert field in row
    assert row["model_family"] == "contrastive"
    assert row["cohort"] == "dlbcl"
    assert row["outcome_window_hours"] == 24 * 365
    assert row["brier_score"] == 0.12
    assert row["rarity_mode"] == "real"
    assert row["baseline_model"] == "tabular_ehr"
    assert row["encoder_frozen"] is None
    assert row["head_type"] is None
    assert row["n_train"] == 12
    assert row["n_test"] == 100
    assert row["n_events_test"] == 20
    assert row["auroc_lower"] == 0.70


def test_joint_training_stage_is_canonicalized():
    assert canonical_training_stage("joint_finetune") == "joint_finetuning"
    assert canonical_training_stage("joint_finetuning") == "joint_finetuning"
    assert canonical_training_stage("linear_probe") == "linear_probe"


def test_build_result_row_keeps_linear_probe_metadata():
    cfg = TinyCfg(
        {
            "dataset": "dlbcl",
            "outcome": "mortality_1y",
            "model_family": "opera_linear_probe",
            "training_stage": "linear_probe",
            "encoder_frozen": True,
            "head_type": "linear_probe",
            "labels": {},
        }
    )
    row = build_result_row(cfg, {"discrimination": {}}, "/tmp/best.ckpt", "held_out")

    assert row["training_stage"] == "linear_probe"
    assert row["encoder_frozen"] is True
    assert row["head_type"] == "linear_probe"


def test_bootstrap_ci_rows_returns_structured_intervals():
    report = {
        "bootstrap_ci": {
            "auroc": {"mean": 0.8, "lower": 0.7, "upper": 0.9, "std": 0.05}
        },
        "survival_bootstrap_ci": {
            "concordance_index": {
                "mean": 0.75,
                "lower": 0.65,
                "upper": 0.85,
                "std": 0.04,
            }
        },
    }

    rows = bootstrap_ci_rows(report)

    assert set(rows["metric_family"]) == {"binary", "survival"}
    assert set(rows["metric"]) == {"auroc", "concordance_index"}
