"""Focused tests for matched Grade 2/3 severity transfer reporting."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from opera.evaluation.outcome_transfer_aggregation import (
    OutcomeTransferAggregationError,
    TRANSFER_AGGREGATION_FAILURE_COLUMNS,
    TRANSFER_COHORT_SUMMARY_COLUMNS,
    TRANSFER_DELTA_COLUMNS,
    TRANSFER_FAMILY_SUMMARY_COLUMNS,
    summarize_severity_deltas,
    write_transfer_aggregation_outputs,
)
from opera.visualization.outcome_transfer import plot_outcome_transfer_figure


def _severity_plan() -> dict:
    """A resolver-shaped plan with two matched and one supplemental G3 target."""
    return {
        "conditions": {
            "opera_no_g3": {
                "transfer_level": "severity_transfer",
                "primary_horizon_days": 90,
                "matched_g2_g3_pairs": [
                    {
                        "lower_grade_outcome": "matched_alpha_g2plus",
                        "target_outcome": "matched_alpha_g3plus",
                    },
                    {
                        "lower_grade_outcome": "matched_beta_g2plus",
                        "target_outcome": "matched_beta_g3plus",
                    },
                ],
                "primary_evaluation_outcomes": [
                    "matched_alpha_g3plus",
                    "matched_beta_g3plus",
                ],
                "secondary_evaluation_outcomes": ["unmatched_g3plus"],
                "evaluation_outcomes": [
                    "matched_alpha_g3plus",
                    "matched_beta_g3plus",
                    "unmatched_g3plus",
                ],
            }
        }
    }


def _severity_deltas() -> pd.DataFrame:
    rows: list[dict] = []
    targets = {
        "matched_alpha_g3plus": ("matched_alpha_g2plus", "primary_matched_g2_g3", 0.10),
        "matched_beta_g3plus": ("matched_beta_g2plus", "primary_matched_g2_g3", 0.30),
        # This deliberately extreme supplemental result must not leak into
        # the primary matched G2/G3 summary or panel.
        "unmatched_g3plus": (None, "secondary_unmatched_g3", 0.95),
    }
    for target, (lower, role, transfer_delta) in targets.items():
        for seed, adjustment in ((42, 0.00), (43, 0.10)):
            for contrast, condition_a, condition_b, value in (
                (
                    "transfer_vs_dapt",
                    "opera_no_g3",
                    "dapt",
                    transfer_delta + adjustment,
                ),
                ("full_vs_transfer", "opera_full", "opera_no_g3", 0.05 + adjustment),
                (
                    "full_vs_dapt",
                    "opera_full",
                    "dapt",
                    transfer_delta + 0.05 + adjustment,
                ),
            ):
                rows.append(
                    {
                        "comparison_condition": "opera_no_g3",
                        "target_outcome": target,
                        "target_family": "Synthetic",
                        "transfer_level": "severity_transfer",
                        "primary_horizon_days": 90,
                        "evaluation_role": role,
                        "matched_lower_grade_outcome": lower,
                        "seed": seed,
                        "evaluation_level": "pan_hematology",
                        "evaluation_group": "all_hematology",
                        "contrast": contrast,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "metric": "auroc",
                        "estimate": value,
                        "status": "completed",
                    }
                )
    return pd.DataFrame(rows)


def test_primary_severity_aggregate_is_resolver_matched_and_secondary_is_separate(
    tmp_path,
):
    plan = _severity_plan()
    deltas = _severity_deltas()

    primary = summarize_severity_deltas(plan, deltas, scope="primary")
    secondary = summarize_severity_deltas(plan, deltas, scope="secondary")

    transfer = primary.loc[primary["contrast"] == "transfer_vs_dapt"].iloc[0]
    assert transfer["severity_scope"] == "primary_matched_g2_g3"
    assert transfer["n_targets"] == 2
    assert transfer["n_target_seed_cells"] == 4
    assert json.loads(transfer["target_outcomes"]) == [
        "matched_alpha_g3plus",
        "matched_beta_g3plus",
    ]
    assert transfer["macro_estimate"] == pytest.approx(0.25)
    assert transfer["aggregation"] == (
        "paired_patient_delta_macro_matched_g2_g3_after_seed_mean"
    )
    assert set(primary["contrast"]) == {
        "transfer_vs_dapt",
        "full_vs_transfer",
        "full_vs_dapt",
    }

    supplemental = secondary.loc[secondary["contrast"] == "transfer_vs_dapt"].iloc[0]
    assert supplemental["severity_scope"] == "secondary_unmatched_g3"
    assert supplemental["n_targets"] == 1
    assert json.loads(supplemental["target_outcomes"]) == ["unmatched_g3plus"]
    assert supplemental["macro_estimate"] == pytest.approx(1.0)

    paths = write_transfer_aggregation_outputs(
        tmp_path,
        deltas=deltas,
        cohort_results=pd.DataFrame(),
        family_summary=pd.DataFrame(),
        failures=pd.DataFrame(),
        severity_primary_summary=primary,
        severity_secondary_summary=secondary,
    )
    assert paths["transfer_severity_primary_summary"].exists()
    assert paths["transfer_severity_secondary_summary"].exists()
    assert set(TRANSFER_DELTA_COLUMNS) <= set(
        pd.read_csv(paths["transfer_deltas"]).columns
    )
    assert set(TRANSFER_FAMILY_SUMMARY_COLUMNS) <= set(
        pd.read_csv(paths["transfer_family_summary"]).columns
    )
    assert set(TRANSFER_COHORT_SUMMARY_COLUMNS) <= set(
        pd.read_csv(paths["transfer_cohort_summary"]).columns
    )
    assert set(TRANSFER_AGGREGATION_FAILURE_COLUMNS) <= set(
        pd.read_csv(paths["transfer_failures"]).columns
    )


def test_primary_severity_figure_uses_paired_deltas_and_excludes_unmatched_target():
    plan = _severity_plan()
    deltas = _severity_deltas()
    empty_results = pd.DataFrame(
        columns=[
            "condition",
            "comparison_condition",
            "target_outcome",
            "metric",
            "value",
            "evaluation_level",
            "evaluation_group",
        ]
    )

    figure = plot_outcome_transfer_figure(empty_results, deltas, plan=plan)
    severity_axis = figure.axes[0]
    assert severity_axis.get_title() == "Severity transfer (matched Grade 2/3 targets)"
    assert severity_axis.get_xlabel() == "Paired Delta ROC-AUC"
    y_labels = [label.get_text() for label in severity_axis.get_yticklabels()]
    assert "matched_alpha_g3plus" in y_labels
    assert "matched_beta_g3plus" in y_labels
    assert "unmatched_g3plus" not in y_labels
    assert {text.get_text() for text in severity_axis.get_legend().get_texts()} == {
        "No-G3 OPERA - DAPT",
        "Full OPERA - No-G3 OPERA",
        "Full OPERA - DAPT",
    }
    point_x = np.concatenate(
        [
            collection.get_offsets()[:, 0]
            for collection in severity_axis.collections
            if len(collection.get_offsets())
        ]
    )
    assert point_x.max() < 0.5
    plt.close(figure)


def test_primary_severity_rejects_a_manually_expanded_target_list():
    plan = _severity_plan()
    plan["conditions"]["opera_no_g3"]["primary_evaluation_outcomes"].append(
        "unmatched_g3plus"
    )
    with pytest.raises(
        OutcomeTransferAggregationError,
        match="programmatically resolved matched G2/G3 target order",
    ):
        summarize_severity_deltas(plan, _severity_deltas(), scope="primary")
