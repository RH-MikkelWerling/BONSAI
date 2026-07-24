"""Focused synthetic tests for the label-only outcome-transfer preflight."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from opera.run.generate_sweep_configs import load_registry
from opera.run.outcome_transfer_preflight import (
    OutcomeTransferPreflightError,
    build_outcome_transfer_support_report,
    write_outcome_transfer_support_report,
)


REGISTRY_PATH = Path("opera/configs/experiment_registry.yaml")


def _synthetic_outcomes(subject_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return synthetic target/death labels with event, competing, and censor rows.

    This fixture is intentionally synthetic.  It exercises the fixed-horizon
    label contract only; it is not an estimate of the production population.
    """
    rows: list[dict[str, object]] = []
    death_rows: list[dict[str, object]] = []
    for offset, subject_id in enumerate(subject_ids):
        split = "train" if offset < 30 else "tuning" if offset < 60 else "held_out"
        index_date = pd.Timestamp("2020-01-01")
        censor_date = index_date + pd.Timedelta(days=200)
        # Thirty held-out events, then one competing death and one early
        # censor.  The retained held-out denominator is 59 = 30 events + 28
        # ordinary non-events + 1 competing event.  The competing death is
        # retained with binary label 0 and is also counted separately.
        target_event = (
            (split == "train" and offset % 2 == 0)
            or (split == "tuning" and offset % 2 == 0)
            or (split == "held_out" and 60 <= offset < 90)
        )
        if offset == 91:
            censor_date = index_date + pd.Timedelta(days=30)
        rows.append(
            {
                "subject_id": subject_id,
                "split": split,
                "index_date": index_date,
                "censor_date": censor_date,
                "outcome_date": (
                    index_date + pd.Timedelta(days=10) if target_event else pd.NaT
                ),
            }
        )
        death_rows.append(
            {
                "subject_id": subject_id,
                "split": split,
                "index_date": index_date,
                "censor_date": censor_date,
                "outcome_date": (
                    index_date + pd.Timedelta(days=15) if offset == 90 else pd.NaT
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(death_rows)


def _patch_resolved_plan(monkeypatch: pytest.MonkeyPatch, split_contract: Path) -> None:
    """Patch only the resolver boundary; the label preflight stays real."""
    from opera.functional import outcome_transfer

    plan = {
        "registry_hash": "synthetic-registry-hash",
        "manifest_hash": "synthetic-manifest-hash",
        "split_contract": str(split_contract),
        "evaluation_target_union": ["any_transfusion", "hospitalisation"],
        "conditions": {
            "opera_full": {
                "name": "opera_full",
                "transfer_level": "direct_supervision",
                "primary_horizon_days": 90,
                "training_outcomes": ["any_transfusion", "hospitalisation"],
                "training_excluded_outcomes": [],
                "evaluation_outcomes": ["any_transfusion", "hospitalisation"],
                "related_retained_outcomes": [],
                "direct_dependencies_excluded": [],
                "launch_blocked": False,
                "launch_blocked_reason": None,
            },
            "opera_no_transfusion_signal": {
                "name": "opera_no_transfusion_signal",
                "transfer_level": "related_outcome_transfer",
                "primary_horizon_days": 90,
                "training_outcomes": ["anemia_g2plus"],
                "training_excluded_outcomes": [
                    "any_transfusion",
                    "RBC_transfusion",
                    "platelet_transfusion",
                ],
                "evaluation_outcomes": ["any_transfusion"],
                "related_retained_outcomes": ["anemia_g2plus"],
                "direct_dependencies_excluded": [],
                "launch_blocked": False,
                "launch_blocked_reason": None,
            },
            "opera_no_hospitalisation_signal": {
                "name": "opera_no_hospitalisation_signal",
                "transfer_level": "related_outcome_transfer",
                "primary_horizon_days": 90,
                "training_outcomes": ["emergency_stay", "respiratory_support"],
                "training_excluded_outcomes": ["hospitalisation"],
                "evaluation_outcomes": ["hospitalisation"],
                "related_retained_outcomes": [
                    "emergency_stay",
                    "respiratory_support",
                ],
                "direct_dependencies_excluded": ["hospitalisation"],
                "launch_blocked": True,
                "launch_blocked_reason": "synthetic launch guard",
                "dependency_resolution_status": "synthetic_unresolved",
            },
        },
    }
    monkeypatch.setattr(
        outcome_transfer,
        "resolve_transfer_manifest",
        lambda **_: plan,
    )


def _write_shared_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = load_registry(REGISTRY_PATH)
    groups = list(registry["cohort_groups"])
    subject_ids = list(range(1, 121))
    membership = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "cohort_grouped": [groups[offset % len(groups)] for offset in range(120)],
        }
    )
    membership_path = tmp_path / "cohort_membership.parquet"
    membership.to_parquet(membership_path, index=False)
    outcomes_dir = tmp_path / "outcomes"
    outcomes_dir.mkdir()
    target, death = _synthetic_outcomes(subject_ids)
    target.to_parquet(outcomes_dir / "any_transfusion.parquet", index=False)
    target.to_parquet(outcomes_dir / "hospitalisation.parquet", index=False)
    death.to_parquet(outcomes_dir / "overall_survival.parquet", index=False)

    split_contract = tmp_path / "temporal_split.yaml"
    split_contract.write_text(
        yaml.safe_dump(
            {
                "train_key": "train",
                "val_key": "tuning",
                "test_key": "held_out",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BONSAI_COHORT_MEMBERSHIP", str(membership_path))
    monkeypatch.setenv("BONSAI_OUTCOMES_DIR", str(outcomes_dir))
    _patch_resolved_plan(monkeypatch, split_contract)


def test_transfer_support_preflight_counts_global_and_grouped_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_shared_synthetic_data(tmp_path, monkeypatch)

    report, metadata = build_outcome_transfer_support_report(
        manifest_path=tmp_path / "synthetic_manifest.yaml",
        registry_path=REGISTRY_PATH,
    )

    required = {
        "target_outcome",
        "target_family",
        "transfer_condition",
        "primary_horizon_days",
        "n_train",
        "n_train_events",
        "n_tuning",
        "n_tuning_events",
        "n_held_out",
        "n_held_out_events",
        "n_held_out_non_events",
        "n_competing_events",
        "n_censored_before_horizon",
        "support_category",
    }
    assert required <= set(report.columns)
    assert metadata["label_only"] is True
    assert set(report["evaluation_level"]) == {"all_hematology", "cohort_grouped"}

    transfusion_all = report.loc[
        (report["target_outcome"] == "any_transfusion")
        & (report["transfer_condition"] == "opera_no_transfusion_signal")
        & (report["evaluation_group"] == "all_hematology")
    ].iloc[0]
    assert transfusion_all["n_held_out"] == 59
    assert transfusion_all["n_held_out_events"] == 30
    assert transfusion_all["n_held_out_non_events"] == 29
    assert transfusion_all["n_competing_events"] == 1
    assert transfusion_all["n_censored_before_horizon"] == 1
    assert transfusion_all["support_category"] == "primary_candidate"
    assert transfusion_all["required_primary_candidate"]
    assert transfusion_all["required_primary_candidate_met"]
    assert not transfusion_all["target_seen_in_training_outcomes"]

    direct_reference = report.loc[
        (report["target_outcome"] == "any_transfusion")
        & (report["transfer_condition"] == "opera_full")
        & (report["evaluation_group"] == "all_hematology")
    ].iloc[0]
    assert direct_reference["target_seen_in_training_outcomes"]
    assert direct_reference["n_held_out"] == transfusion_all["n_held_out"]

    hospital = report.loc[
        (report["target_outcome"] == "hospitalisation")
        & (report["transfer_condition"] == "opera_no_hospitalisation_signal")
        & (report["evaluation_group"] == "all_hematology")
    ].iloc[0]
    assert hospital["condition_blocked"]
    assert hospital["launch_blocked_reason"] == "synthetic launch guard"
    assert hospital["dependency_resolution_status"] == "synthetic_unresolved"

    expected_groups = set(load_registry(REGISTRY_PATH)["cohort_groups"])
    grouped = report.loc[report["evaluation_level"] == "cohort_grouped"]
    assert set(grouped["evaluation_group"]) == expected_groups


def test_transfer_support_preflight_writes_csv_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_shared_synthetic_data(tmp_path, monkeypatch)
    output_dir = tmp_path / "report"

    csv_path, json_path, report = write_outcome_transfer_support_report(
        manifest_path=tmp_path / "synthetic_manifest.yaml",
        registry_path=REGISTRY_PATH,
        output_dir=output_dir,
    )

    assert csv_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["n_rows"] == len(report)
    assert len(payload["rows"]) == len(report)
    assert isinstance(payload["rows"][0]["n_held_out"], int)


def test_transfer_support_preflight_rejects_nonunique_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_shared_synthetic_data(tmp_path, monkeypatch)
    membership_path = Path(os.environ["BONSAI_COHORT_MEMBERSHIP"])
    membership = pd.read_parquet(membership_path)
    pd.concat([membership, membership.iloc[[0]]], ignore_index=True).to_parquet(
        membership_path, index=False
    )

    with pytest.raises(OutcomeTransferPreflightError, match="duplicate subject_id"):
        build_outcome_transfer_support_report(
            manifest_path=tmp_path / "synthetic_manifest.yaml",
            registry_path=REGISTRY_PATH,
        )
