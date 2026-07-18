"""Synthetic, CPU-only tests for the focused frozen-probe transfer path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

import opera.evaluation.outcome_transfer_evaluation as transfer_evaluation
from opera.evaluation.outcome_transfer_aggregation import (
    aggregate_transfer_predictions,
)
from opera.evaluation.outcome_transfer_evaluation import (
    DenominatorParityError,
    EmbeddingArtifact,
    OutcomeTransferEvaluationError,
    TRANSFER_FAILURE_COLUMNS,
    TRANSFER_PREDICTION_COLUMNS,
    TRANSFER_PROBE_STATUS_COLUMNS,
    TRANSFER_RESULT_COLUMNS,
    _split_keys,
    assert_prediction_denominator_parity,
    evaluate_frozen_transfer_probes,
    resolve_registry,
    write_frozen_probe_outputs,
)
from opera.visualization.outcome_transfer import write_outcome_transfer_figure


def _synthetic_transfer_fixture(tmp_path):
    """Create clearly synthetic labels/embeddings; no production data or GPU."""
    subject_ids = np.arange(1, 25)
    split = np.repeat(["train", "tuning", "held_out"], 8)
    membership = pd.DataFrame(
        {
            "subject_id": subject_ids,
            # Each grouped cohort has positives and negatives in every split.
            "cohort_grouped": np.where(
                np.isin(subject_ids % 4, [0, 1]), "GROUP_A", "GROUP_B"
            ),
        }
    )
    membership_path = tmp_path / "synthetic_membership.csv"
    membership.to_csv(membership_path, index=False)
    index_date = pd.Timestamp("2020-01-01")
    outcome = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "split": split,
            "index_date": index_date,
            "censor_date": pd.Timestamp("2021-01-01"),
            "outcome_date": [
                index_date + pd.Timedelta(days=10) if subject_id % 2 == 0 else pd.NaT
                for subject_id in subject_ids
            ],
        }
    )
    outcome.to_csv(tmp_path / "synthetic_target.csv", index=False)
    pd.DataFrame(
        {"subject_id": subject_ids, "outcome_date": [pd.NaT] * len(subject_ids)}
    ).to_csv(tmp_path / "overall_survival.csv", index=False)
    split_contract = tmp_path / "synthetic_split.yaml"
    split_contract.write_text(
        yaml.safe_dump(
            {"train_key": "train", "val_key": "tuning", "test_key": "held_out"}
        ),
        encoding="utf-8",
    )
    registry = {
        "death_outcome": "overall_survival",
        "outcomes": ["synthetic_target", "overall_survival"],
        "paths": {
            "cohort_membership_file": str(membership_path),
            "outcomes_dir": str(tmp_path),
        },
        "cohort_columns": {"grouped": "cohort_grouped"},
        "outcome_metadata": {
            "synthetic_target": {
                "outcome_file": "synthetic_target.csv",
                "competing_outcome_file": "overall_survival.csv",
            },
            "overall_survival": {"outcome_file": "overall_survival.csv"},
        },
    }
    plan = {
        "seeds": [42],
        "registry_hash": "synthetic-registry-hash",
        "manifest_hash": "synthetic-manifest-hash",
        "base_contrastive_config_hash": "synthetic-base-config-hash",
        "split_contract": str(split_contract),
        "split_contract_hash": "synthetic-split-contract-hash",
        "outcome_families": {
            "synthetic_target": "Synthetic family",
            "overall_survival": "Disease control & survival",
        },
        "conditions": {
            "opera_full": {
                "training_outcomes": ["synthetic_target", "overall_survival"],
                "training_excluded_outcomes": [],
                "evaluation_outcomes": ["synthetic_target"],
                "transfer_level": "direct_supervision",
                "related_retained_outcomes": [],
                "direct_dependencies_excluded": [],
            },
            "opera_no_target": {
                "training_outcomes": ["overall_survival"],
                "training_excluded_outcomes": ["synthetic_target"],
                "evaluation_outcomes": ["synthetic_target"],
                "transfer_level": "family_transfer",
                "primary_horizon_days": 90,
                "launch_blocked": False,
                "related_retained_outcomes": [],
                "direct_dependencies_excluded": [],
            },
        },
    }
    rng = np.random.default_rng(7)
    artifacts = {}
    for representation, strength in (("dapt", 0.2), ("opera_no_target", 0.45), ("opera_full", 0.8)):
        signal = (subject_ids % 2 == 0).astype(float)
        frame = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "embedding_0": strength * signal + rng.normal(0, 0.15, len(subject_ids)),
                "embedding_1": rng.normal(0, 1, len(subject_ids)),
                "_subject_key": [str(item) for item in subject_ids],
            }
        )
        if representation == "dapt":
            metadata = {
                "checkpoint_hash": f"checkpoint-{representation}",
                "condition": "dapt",
                "training_stage": "hematology_domain_adaptation",
                "source_checkpoint": "/synthetic/pretraining.ckpt",
                "registry_hash": plan["registry_hash"],
                "manifest_hash": plan["manifest_hash"],
                "base_contrastive_config_hash": plan[
                    "base_contrastive_config_hash"
                ],
                "split_contract": plan["split_contract"],
                "split_contract_hash": plan["split_contract_hash"],
            }
        else:
            condition = plan["conditions"][representation]
            metadata = {
                "checkpoint_hash": f"checkpoint-{representation}",
                "training_stage": "opera_contrastive_adaptation",
                "condition": representation,
                "seed": 42,
                "transfer_level": condition["transfer_level"],
                "registry_hash": plan["registry_hash"],
                "manifest_hash": plan["manifest_hash"],
                "base_contrastive_config_hash": plan[
                    "base_contrastive_config_hash"
                ],
                "included_outcomes": condition["training_outcomes"],
                "excluded_outcomes": condition["training_excluded_outcomes"],
                "evaluation_outcomes": condition["evaluation_outcomes"],
                "related_retained_outcomes": condition.get("related_retained_outcomes", []),
                "direct_dependencies_excluded": condition.get(
                    "direct_dependencies_excluded", []
                ),
                "selection_outcomes": condition["training_outcomes"],
                "split_contract": plan["split_contract"],
                "split_contract_hash": plan["split_contract_hash"],
                "source_dapt_checkpoint": "/synthetic/dapt.ckpt",
                "outcome_set": condition["training_outcomes"],
            }
        artifacts[(representation, 42)] = EmbeddingArtifact(
            representation=representation,
            seed=42,
            path=tmp_path / f"synthetic_{representation}.npz",
            frame=frame,
            artifact_hash=f"artifact-{representation}",
            checkpoint_hash=f"checkpoint-{representation}",
            checkpoint_hash_source="synthetic_fixture",
            metadata=metadata,
            metadata_path=tmp_path / f"synthetic_{representation}.metadata.json",
        )
    return registry, plan, artifacts


def test_frozen_probe_is_pan_hematology_and_grouped_rows_reuse_predictions(tmp_path, monkeypatch):
    registry, plan, artifacts = _synthetic_transfer_fixture(tmp_path)
    original_fit = transfer_evaluation.fit_standardized_linear_probe
    calls = []

    def counted_fit(*args, **kwargs):
        calls.append(args[0].representation)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(transfer_evaluation, "fit_standardized_linear_probe", counted_fit)
    results, predictions, status, failures = evaluate_frozen_transfer_probes(
        plan,
        registry=registry,
        artifacts=artifacts,
        c_grid=(0.1, 1.0),
    )

    assert calls == ["dapt", "opera_no_target", "opera_full"]
    assert failures.empty
    assert set(status["status"]) == {"completed"}
    assert set(results["evaluation_level"]) == {"pan_hematology", "cohort_grouped"}
    assert set(results.loc[results["evaluation_level"] == "cohort_grouped", "evaluation_group"]) == {
        "GROUP_A",
        "GROUP_B",
    }
    # Stored patient rows occur once per representation, not once per cohort:
    # the cohort metrics are a re-stratification of these exact predictions.
    assert len(predictions) == 3 * 8
    assert results["encoder_frozen"].all()
    assert set(results["probe_type"]) == {"standardized_logistic_regression"}


def test_paired_aggregation_fails_closed_and_writes_figure(tmp_path):
    registry, plan, artifacts = _synthetic_transfer_fixture(tmp_path)
    results, predictions, _, failures = evaluate_frozen_transfer_probes(
        plan,
        registry=registry,
        artifacts=artifacts,
        c_grid=(0.1, 1.0),
    )
    assert failures.empty
    deltas, cohorts, family_summary, aggregation_failures = aggregate_transfer_predictions(
        plan,
        predictions,
        n_bootstrap=20,
    )
    assert len(deltas) == 9  # 3 contrasts x AUROC/AUPRC/Brier for one seed/task.
    assert len(cohorts) == 18  # the same three contrasts/metrics for two groups.
    assert aggregation_failures.empty
    assert not family_summary.empty
    assert set(deltas["evaluation_level"]) == {"pan_hematology"}
    assert {
        "condition_a_checkpoint_hash",
        "condition_b_checkpoint_hash",
        "condition_a_direct_target_seen",
        "condition_b_direct_target_seen",
        "condition_a_included_outcome_count",
        "condition_b_excluded_outcome_count",
    } <= set(deltas.columns)
    assert set(deltas["condition_a_checkpoint_hash"]) <= {
        "checkpoint-dapt",
        "checkpoint-opera_no_target",
        "checkpoint-opera_full",
    }
    # The generic frozen-probe fixture does not model a Grade 2/3 panel, but
    # the primary figure now deliberately requires an explicit resolved plan
    # rather than inferring severity targets from result names.
    severity_plot_plan = {
        "conditions": {
            "opera_no_g3": {
                "transfer_level": "severity_transfer",
                "matched_g2_g3_pairs": [
                    {
                        "lower_grade_outcome": "synthetic_g2plus",
                        "target_outcome": "synthetic_g3plus",
                    }
                ],
                "primary_evaluation_outcomes": ["synthetic_g3plus"],
                "secondary_evaluation_outcomes": [],
                "evaluation_outcomes": ["synthetic_g3plus"],
            }
        }
    }
    paths = write_outcome_transfer_figure(
        tmp_path,
        results=results,
        deltas=deltas,
        plan=severity_plot_plan,
    )
    assert paths["png"].exists()
    assert paths["pdf"].exists()
    plt.close("all")

    mismatched = {
        "dapt": predictions.loc[predictions["condition"] == "dapt"].copy(),
        "opera_full": predictions.loc[predictions["condition"] == "opera_full"].copy(),
    }
    mismatched["opera_full"].loc[mismatched["opera_full"].index[0], "label"] = 1 - int(
        mismatched["opera_full"].iloc[0]["label"]
    )
    with pytest.raises(DenominatorParityError, match="label mismatch"):
        assert_prediction_denominator_parity(mismatched, context="synthetic parity test")


def test_opera_embedding_requires_a_plan_matched_metadata_sidecar(tmp_path):
    registry, plan, artifacts = _synthetic_transfer_fixture(tmp_path)
    bad = dict(artifacts)
    bad[("opera_no_target", 42)] = replace(
        bad[("opera_no_target", 42)], metadata_path=None
    )
    with pytest.raises(OutcomeTransferEvaluationError, match="no embedding metadata sidecar"):
        evaluate_frozen_transfer_probes(
            plan,
            registry=registry,
            artifacts=bad,
            c_grid=(0.1,),
        )


def test_opera_embedding_rejects_missing_dapt_source_provenance(tmp_path):
    registry, plan, artifacts = _synthetic_transfer_fixture(tmp_path)
    target = artifacts[("opera_no_target", 42)]
    incomplete = dict(target.metadata)
    incomplete.pop("source_dapt_checkpoint")
    bad = dict(artifacts)
    bad[("opera_no_target", 42)] = replace(target, metadata=incomplete)
    with pytest.raises(OutcomeTransferEvaluationError, match="source_dapt_checkpoint"):
        evaluate_frozen_transfer_probes(
            plan,
            registry=registry,
            artifacts=bad,
            c_grid=(0.1,),
        )


def test_dapt_embedding_cannot_be_an_opera_checkpoint_relabelled_as_dapt(tmp_path):
    registry, plan, artifacts = _synthetic_transfer_fixture(tmp_path)
    dapt = artifacts[("dapt", 42)]
    invalid = dict(dapt.metadata)
    invalid["training_stage"] = "opera_contrastive_adaptation"
    bad = dict(artifacts)
    bad[("dapt", 42)] = replace(dapt, metadata=invalid)
    with pytest.raises(OutcomeTransferEvaluationError, match="hematology_domain_adaptation"):
        evaluate_frozen_transfer_probes(
            plan,
            registry=registry,
            artifacts=bad,
            c_grid=(0.1,),
        )


def test_dapt_embedding_requires_matching_extraction_context_hashes(tmp_path):
    registry, plan, artifacts = _synthetic_transfer_fixture(tmp_path)
    dapt = artifacts[("dapt", 42)]
    stale = dict(dapt.metadata)
    stale["split_contract_hash"] = "stale-split"
    bad = dict(artifacts)
    bad[("dapt", 42)] = replace(dapt, metadata=stale)
    with pytest.raises(OutcomeTransferEvaluationError, match="split_contract_hash"):
        evaluate_frozen_transfer_probes(
            plan,
            registry=registry,
            artifacts=bad,
            c_grid=(0.1,),
        )


def test_checked_in_split_and_registry_paths_do_not_depend_on_launch_cwd(monkeypatch):
    """The transfer CLIs are safe to invoke from a server results directory."""
    # ``tests/`` has no ``opera/configs`` child, so this exercises the
    # repository-root fallback without creating another temporary directory.
    monkeypatch.chdir(Path(__file__).resolve().parent)
    assert _split_keys("opera/configs/manifests/temporal_split.yaml") == {
        "train": "train",
        "tuning": "tuning",
        "held_out": "held_out",
    }
    assert (
        resolve_registry("opera/configs/experiment_registry.yaml")["death_outcome"]
        == "overall_survival"
    )


def test_empty_frozen_probe_outputs_keep_a_stable_schema(tmp_path):
    paths = write_frozen_probe_outputs(
        tmp_path,
        results=pd.DataFrame(),
        predictions=pd.DataFrame(),
        probe_status=pd.DataFrame(),
        failures=pd.DataFrame(),
    )
    assert set(TRANSFER_RESULT_COLUMNS) <= set(
        pd.read_csv(paths["transfer_results"]).columns
    )
    assert set(TRANSFER_PROBE_STATUS_COLUMNS) <= set(
        pd.read_csv(paths["transfer_probe_status"]).columns
    )
    assert set(TRANSFER_FAILURE_COLUMNS) <= set(pd.read_csv(paths["transfer_failures"]).columns)
    assert set(TRANSFER_PREDICTION_COLUMNS) <= set(
        pd.read_parquet(paths["transfer_predictions"]).columns
    )
