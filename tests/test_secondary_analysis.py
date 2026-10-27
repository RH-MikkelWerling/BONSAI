"""Synthetic locked-prediction analyses, including leakage/parity backstops."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from opera.evaluation.secondary_analysis import (
    audit_router_identity,
    align_models,
    contrast_coefficients,
    coverage_inventory,
    bootstrap_metric_values,
    negative_transfer_strata,
    normalize_predictions,
    paired_analysis,
    panel_subgroup_gains,
    patient_bootstrap,
    seed_averaged_effects,
    route_summaries,
    subgroup_patient_gains,
    tuning_selected_strategies,
)
from opera.run.analyse_locked_predictions import read_artifact, read_route_artifact, run


def predictions():
    rows = []
    for split in ("tuning", "held_out"):
        for outcome in ("a", "b"):
            for patient in range(12):
                label = patient % 2
                for model, strength in {
                    "S_L": 0.15,
                    "D_L": 0.25,
                    "O_L": 0.2,
                    "J_L": -0.05,
                    "R_L": 0.3,
                }.items():
                    rows.append(
                        {
                            "split": split,
                            "seed": 42,
                            "cohort": "CLL",
                            "subject_id": f"{split}_{patient}",
                            "outcome": outcome,
                            "horizon_days": 365,
                            "model": model,
                            "label": label,
                            "probability": 0.5
                            + (2 * label - 1) * strength * (1 + 0.1 * (patient % 3)),
                            "data_contract": "synthetic_contract",
                        }
                    )
    return pd.DataFrame(rows)


class SecondaryAnalysisTest(unittest.TestCase):
    def test_same_test_patients_are_not_independent_replicates_across_seeds(self):
        frame = predictions()
        frame = pd.concat([frame, frame.assign(seed=43)], ignore_index=True)
        frame, _ = normalize_predictions(frame)
        result = paired_analysis(frame, {"R1": "gain(R_L,J_L)"}, replicates=30)
        samples = result["bootstrap_draws"].query(
            "split == 'held_out' and outcome == 'a' and metric == 'brier'"
        )
        np.testing.assert_allclose(
            samples.query("seed == 42").gain_draws.iloc[0],
            samples.query("seed == 43").gain_draws.iloc[0],
        )
        means = seed_averaged_effects(result["contrasts"], result["bootstrap_draws"])
        mean = means.query(
            "split == 'held_out' and outcome == 'a' and metric == 'brier'"
        ).iloc[0]
        single = (
            result["contrasts"]
            .query(
                "split == 'held_out' and outcome == 'a' and metric == 'brier' and seed == 42"
            )
            .iloc[0]
        )
        self.assertAlmostEqual(mean.difference_se, single.difference_se)
        self.assertEqual(mean.n_training_seeds, 2)

    def test_identity_audit_flags_retrospective_labels_without_assuming_clinical_ignorance(
        self,
    ):
        metadata = pd.DataFrame(
            {
                "subject_id": ["a", "b", "c"],
                "index_date": ["2023-01-01"] * 3,
                "cohort_identity_evidence_date": [
                    "2022-12-01",
                    "2023-02-01",
                    "2023-02-01",
                ],
                "baseline_disease_clinically_confirmed": [False, False, True],
                "baseline_confirmation_reference": [
                    None,
                    None,
                    "baseline_chart_review_hash",
                ],
                "cohort_source": ["pass1", "pass2_fwd_dx", "pass2_fwd_dx"],
            }
        )
        audit = audit_router_identity(metadata)
        self.assertEqual(
            set(audit.status),
            {
                "evidence_preindex",
                "same_or_postindex_evidence_needs_review",
                "clinically_confirmed_at_index",
            },
        )
        self.assertEqual(audit.n_patients.sum(), 3)
        self.assertEqual(
            audit_router_identity(
                metadata.drop(columns="cohort_identity_evidence_date")
            ).status.iloc[0],
            "missing_identity_time_provenance",
        )
        with self.assertRaises(ValueError):
            audit_router_identity(
                metadata.drop(columns="baseline_confirmation_reference")
            )

    def test_parser_does_not_execute_expressions_and_handles_interactions(self):
        self.assertEqual(
            contrast_coefficients("gain(J_L,D_L)-gain(O_L,S_L)"),
            {"J_L": 1, "D_L": -1, "O_L": -1, "S_L": 1},
        )
        for expression in ("__import__('os')", "gain(A,B)gain(C,D)", "gain(A,B);1"):
            with self.assertRaises(ValueError):
                contrast_coefficients(expression)

    def test_invalid_labels_duplicates_probabilities_and_split_overlap_fail(self):
        for column, value in (
            ("label", 0.5),
            ("probability", np.nan),
            ("probability", 1.1),
            ("known", "false"),
        ):
            frame = predictions()
            if column not in frame:
                frame[column] = True
            frame[column] = frame[column].astype(object)
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                normalize_predictions(frame)
        with self.assertRaises(ValueError):
            normalize_predictions(pd.concat([predictions(), predictions().iloc[:1]]))
        frame = predictions()
        frame.loc[frame.split == "held_out", "subject_id"] = frame.loc[
            frame.split == "held_out", "subject_id"
        ].str.replace("held_out", "tuning")
        with self.assertRaisesRegex(ValueError, "overlap"):
            normalize_predictions(frame)

    def test_alignment_rejects_missing_patients_labels_and_contracts(self):
        cell = predictions().query("split == 'tuning' and outcome == 'a'")
        base = cell.query("model == 'S_L'")
        for changed in (
            base.iloc[1:],
            base.assign(label=1 - base.label),
            base.assign(data_contract="other_release"),
        ):
            with self.assertRaises(ValueError):
                align_models({"base": base, "changed": changed})

    def test_paired_gains_orientation_single_class_and_correlated_draws(self):
        frame, _ = normalize_predictions(predictions())
        result = paired_analysis(
            frame, {"P5": "gain(J_L,S_L)", "R1": "gain(R_L,J_L)"}, replicates=30
        )
        rescue = result["contrasts"].query("contrast == 'R1' and metric == 'brier'")
        self.assertTrue((rescue.gain > 0).all())
        self.assertTrue(
            (
                result["contrasts"].query("contrast == 'P5' and metric == 'brier'").gain
                < 0
            ).all()
        )
        draw = result["bootstrap_draws"].query("contrast == 'R1' and metric == 'brier'")
        first = draw.query("split == 'held_out' and outcome == 'a'").gain_draws.iloc[0]
        second = draw.query("split == 'held_out' and outcome == 'b'").gain_draws.iloc[0]
        np.testing.assert_allclose(first, second)
        self.assertGreater(np.ptp(first), 0)
        single = frame.loc[frame.label == 0]
        scores = paired_analysis(single, {"R1": "gain(R_L,J_L)"}, replicates=3)[
            "contrasts"
        ]
        self.assertTrue(scores.query("metric == 'auc'").gain.isna().all())
        self.assertTrue(np.isfinite(scores.query("metric == 'brier'").gain).all())

    def test_vectorized_bootstrap_auc_matches_weighted_sklearn_with_ties(self):
        labels = np.array([0, 1, 0, 1, 1, 0])
        probabilities = np.array([0.2, 0.2, 0.5, 0.8, 0.5, 0.8])
        weights = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 3, 1, 2, 0], [2, 0, 1, 0, 0, 2]])
        result = bootstrap_metric_values(labels, probabilities, weights)
        for index in (0, 1):
            self.assertAlmostEqual(
                result["auc"][index],
                roc_auc_score(labels, probabilities, sample_weight=weights[index]),
            )
        self.assertTrue(np.isnan(result["auc"][2]))

    def test_selection_and_negative_transfer_never_use_test_to_define_groups(self):
        frame, _ = normalize_predictions(predictions())
        selected, decisions = tuning_selected_strategies(frame)
        self.assertEqual(set(decisions.selected_model), {"D_L"})
        changed = frame.copy()
        changed.loc[
            (changed.split == "held_out") & (changed.model == "S_L"), "probability"
        ] = changed.loc[
            (changed.split == "held_out") & (changed.model == "S_L"), "label"
        ]
        _, changed_decisions = tuning_selected_strategies(changed)
        pd.testing.assert_frame_equal(decisions, changed_decisions)
        self.assertTrue(selected.split.eq("held_out").all())
        table = paired_analysis(
            frame, {"P5": "gain(J_L,S_L)", "R1": "gain(R_L,J_L)"}, replicates=3
        )["contrasts"]
        strata = negative_transfer_strata(table)
        self.assertEqual(set(strata.stratum), {"tuning_negative_pooling"})

    def test_subgroups_require_complete_preindex_metadata(self):
        frame, _ = normalize_predictions(predictions())
        result = paired_analysis(frame, {"R1": "gain(R_L,J_L)"}, replicates=3)
        metadata = pd.DataFrame({"subject_id": frame.subject_id.unique(), "age": 60})
        definitions = {"age": {"provenance": "pre_index", "bins": [0, 50, 80, 120]}}
        groups = subgroup_patient_gains(
            result["patient_gains"],
            metadata,
            definitions,
            draws=patient_bootstrap(frame, 3, 2026),
        )
        self.assertTrue((groups.gain > 0).all())
        with self.assertRaises(ValueError):
            subgroup_patient_gains(
                result["patient_gains"],
                metadata.iloc[1:],
                definitions,
                draws=patient_bootstrap(frame, 3, 2026),
            )
        with self.assertRaises(ValueError):
            subgroup_patient_gains(
                result["patient_gains"],
                metadata,
                {"age": {"provenance": "post_index"}},
                draws={},
            )

    def test_panel_subgroups_pool_patients_across_cells(self):
        frame, _ = normalize_predictions(predictions())
        result = paired_analysis(frame, {"R1": "gain(R_L,J_L)"}, replicates=30)
        subjects = frame.subject_id.unique()
        metadata = pd.DataFrame({"subject_id": subjects,
                                 "age": np.where(np.arange(len(subjects)) % 2, 70, 40)})
        definitions = {"age": {"provenance": "pre_index", "bins": [0, 50, 120]}}
        draws = patient_bootstrap(frame, 30, 2026)
        panel = panel_subgroup_gains(result["patient_gains"], metadata, definitions,
                                     draws=draws)
        cells = subgroup_patient_gains(result["patient_gains"], metadata, definitions,
                                       draws=draws)
        panel = panel.loc[panel.metric == "brier_gain"]
        # One panel row per split/seed/horizon/contrast/subgroup, never per cell.
        self.assertFalse(panel.duplicated(
            ["split", "seed", "horizon_days", "contrast", "subgroup"]).any())
        per_horizon = panel.loc[panel.horizon_days != 0]
        self.assertLessEqual(len(per_horizon), len(cells.loc[cells.metric == "brier_gain"]))
        gains = result["patient_gains"].merge(metadata, on="subject_id")
        held = gains.loc[(gains.split == "held_out") & (gains.age > 50)]
        row = panel.loc[(panel.split == "held_out") & (panel.subgroup.str.startswith("(50"))]
        self.assertIn(0, set(row.horizon_days))  # pooled over every horizon
        for horizon, value in row.set_index("horizon_days").gain.items():
            rows = held if horizon == 0 else held.loc[held.horizon_days == horizon]
            self.assertAlmostEqual(value, rows.brier_gain.mean())
        self.assertTrue((panel.lower <= panel.upper).all())

    def test_coverage_preserves_failed_models_and_route_normalization(self):
        frame, _ = normalize_predictions(predictions())
        roster = pd.DataFrame(
            [
                {
                    "model": model,
                    "seed": 42,
                    "cohort": "CLL",
                    "outcome": "a",
                    "horizon_days": 365,
                    "status": status,
                }
                for model, status in [("S_L", "complete"), ("S_FAILED", "failed")]
            ]
        )
        coverage = coverage_inventory(frame, roster)
        self.assertEqual(coverage.iloc[1].prediction_status, "absent")
        self.assertEqual(coverage.iloc[1].status, "failed")
        routes = pd.DataFrame(
            [
                {
                    "model": "R_L",
                    "seed": 42,
                    "split": "held_out",
                    "cohort": "CLL",
                    "outcome": "a",
                    "subject_id": "x",
                    "source": source,
                    "bank": source,
                    "weight": weight,
                }
                for source, weight in [("shared", 0.2), ("disease", 0.8)]
            ]
        )
        self.assertEqual(len(route_summaries(routes)), 2)
        with self.assertRaises(ValueError):
            route_summaries(routes.iloc[1:])

    def test_native_cif_adapter_masks_censoring_and_competing_death(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "native.csv"
            pd.DataFrame(
                {
                    "subject_id": ["001", "002", "003"],
                    "cohort": "CLL",
                    "outcome": "a",
                    "split": "held_out",
                    "time_days": [5, 10, 365],
                    "event": [0, 2, 1],
                    "cif_365d": [0.2, 0.1, 0.7],
                }
            ).to_csv(path, index=False)
            frame, audit = read_artifact(
                {
                    "path": str(path),
                    "format": "direct_cif",
                    "model": "J_L",
                    "seed": 42,
                    "data_contract": "test",
                }
            )
            self.assertEqual(audit["excluded_unknown_rows"], 1)
            self.assertEqual(frame.subject_id.tolist(), ["002", "003"])
            self.assertEqual(frame.label.tolist(), [0, 1])

    def test_native_routes_and_source_removal_probabilities_are_supported(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pd.DataFrame(
                {
                    "subject_id": ["001"],
                    "split": ["held_out"],
                    "outcome": ["a"],
                    "shared_weight": [0.2],
                    "disease_weights": ['{"disease:CLL": 0.8}'],
                    "outcome_weights": ['{"outcome:a": 0.0}'],
                }
            ).to_csv(root / "routes.csv", index=False)
            routes = read_route_artifact(
                {
                    "route_weights": str(root / "routes.csv"),
                    "route_model": "R_L",
                    "route_seed": 42,
                },
                pd.DataFrame({"subject_id": ["001"], "cohort_fine": ["CLL"]}),
            )
            self.assertEqual(len(route_summaries(routes)), 3)
            pd.DataFrame(
                {
                    "subject_id": ["001"],
                    "cohort": ["CLL"],
                    "outcome": ["a"],
                    "split": ["held_out"],
                    "time_days": [365],
                    "event": [0],
                    "cif_365d": [0.2],
                    "removed_source_cif_365d": [0.4],
                }
            ).to_csv(root / "removed.csv", index=False)
            removed, _ = read_artifact(
                {
                    "path": str(root / "removed.csv"),
                    "format": "direct_cif",
                    "model": "REMOVED",
                    "probability_prefix": "removed_source_cif_",
                    "seed": 42,
                    "data_contract": "test",
                }
            )
            self.assertEqual(removed.probability.iloc[0], 0.4)

    def test_router_null_equivalence_gates_bank_effects(self):
        from opera.evaluation.secondary_analysis import (
            annotate_router_gate,
            router_null_equivalence,
        )

        base = {"split": "held_out", "seed": 42, "horizon_days": 365,
                "stratum": "all_cells", "stratum_value": "all"}
        summaries = pd.DataFrame(
            [
                {**base, "contrast": "R0", "metric": "auc", "gain": 0.001,
                 "lower": -0.004, "upper": 0.006},
                {**base, "contrast": "R0", "metric": "brier", "gain": 0.0,
                 "lower": -0.003, "upper": 0.003},
                {**base, "contrast": "R0", "metric": "log_loss", "gain": 0.0,
                 "lower": np.nan, "upper": np.nan},
                {**base, "contrast": "R2", "metric": "auc", "gain": 0.02,
                 "lower": 0.01, "upper": 0.03},
                {**base, "contrast": "R2", "metric": "brier", "gain": 0.001,
                 "lower": 0.0, "upper": 0.002},
                {**base, "contrast": "P1", "metric": "auc", "gain": 0.01,
                 "lower": 0.0, "upper": 0.02},
            ]
        )
        gate = router_null_equivalence(
            summaries, {"auc": 0.005, "brier": 0.005, "log_loss": 0.01}
        ).set_index("metric")
        # AUC interval exceeds +0.005: inconclusive is not equivalence.
        self.assertEqual(gate.loc["auc", "status"], "equivalence_not_shown")
        self.assertEqual(gate.loc["brier", "status"], "equivalent_within_tolerance")
        self.assertEqual(
            gate.loc["log_loss", "status"], "insufficient_bootstrap_draws"
        )
        unlocked = router_null_equivalence(summaries, None)
        self.assertTrue(unlocked.status.eq("tolerance_not_locked").all())
        annotated = annotate_router_gate(summaries, gate.reset_index()).set_index(
            ["contrast", "metric"]
        )
        self.assertEqual(
            annotated.loc[("R2", "auc"), "router_null_gate"], "equivalence_not_shown"
        )
        self.assertEqual(
            annotated.loc[("R2", "brier"), "router_null_gate"],
            "equivalent_within_tolerance",
        )
        self.assertTrue(pd.isna(annotated.loc[("P1", "auc"), "router_null_gate"]))
        missing = router_null_equivalence(
            summaries.loc[summaries.contrast != "R0"], {"auc": 0.01}
        )
        self.assertTrue(missing.status.eq("missing_null_router_contrast").all())

    def test_cli_backbone_writes_all_core_tables_without_overwriting(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            frame = predictions()
            frame.to_csv(root / "predictions.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "cohort": "CLL",
                        "outcome": outcome,
                        "horizon_days": 365,
                        "n_events_train": 20,
                        "outcome_family": "test",
                        "endpoint_group": outcome,
                    }
                    for outcome in ["a", "b"]
                ]
            ).to_csv(root / "cells.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": frame.subject_id.unique(),
                    "age": [40, 70] * (frame.subject_id.nunique() // 2)
                    + [40] * (frame.subject_id.nunique() % 2),
                    "predominant_region": ["Hovedstaden", "Sjælland"] * (
                        frame.subject_id.nunique() // 2) + ["Hovedstaden"] * (
                        frame.subject_id.nunique() % 2),
                    "index_date": "2023-01-01",
                    "cohort_identity_evidence_date": "2022-12-01",
                }
            ).to_csv(root / "patients.csv", index=False)
            config = {
                "estimand": "binary_known_status_unweighted",
                "artifacts": [{"path": str(root / "predictions.csv")}],
                "contrast_manifest": "opera/configs/manifests/final_contrasts_20261014.yaml",
                "cell_metadata": str(root / "cells.csv"),
                "patient_metadata": str(root / "patients.csv"),
                "patient_groups": {"age": {"provenance": "pre_index", "bins": [0, 50, 80]}},
                "stratified_reanalysis": {"predominant_region": {}, "age": {}},
                "bootstrap_replicates": 3,
                # Tiny fixture exercises schema/CLI, not production eligibility.
                "performance_min_cases": 0,
                "performance_min_controls": 0,
            }
            manifest = run(config, root / "output")
            for name in [
                "contrasts",
                "effect_summaries",
                "late_diagnosis_exclusion_sensitivity",
                "negative_transfer_summary",
                "negative_transfer",
                "calibration",
                "strategy_selection",
                "patient_subgroups",
                "contrast_coverage",
                "stratified_predominant_region_effects",
                "stratified_age_effects",
            ]:
                self.assertGreater(manifest["tables"][name], 0)
                self.assertTrue((root / "output" / f"{name}.csv").is_file())
            age_effects = pd.read_csv(root / "output" / "stratified_age_effects.csv")
            self.assertEqual(set(age_effects.stratum_value), {"(-0.001, 50.0]", "(50.0, 80.0]"})
            self.assertTrue((root / "output" / "patient_gains_INTERNAL.csv").is_file())
            with self.assertRaises(FileExistsError):
                run(config, root / "output")


if __name__ == "__main__":
    unittest.main()


class CifIpcwEstimandTest(unittest.TestCase):
    def _frame(self, n=400, seed=3):
        rng = np.random.default_rng(seed)
        times = rng.exponential(300, n).round(1)
        events = rng.choice([0, 1, 2], n, p=[0.45, 0.35, 0.20])
        rows = []
        for model, noise in (("A", 0.15), ("B", 0.35)):
            risk = np.clip(1 / (1 + np.exp((times - 250) / 120)) + rng.normal(0, noise, n), 0.01, 0.99)
            for i in range(n):
                rows.append(
                    dict(model=model, seed=42, split="held_out", cohort="CLL",
                         outcome="event", horizon_days=365, subject_id=f"p{i:04d}",
                         data_contract="c1", probability=risk[i],
                         time_days=times[i], event=int(events[i]))
                )
        return pd.DataFrame(rows), times, events

    def test_ipcw_metrics_reproduce_the_native_cif_evaluator(self):
        from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
        from opera.evaluation.secondary_analysis import metric_values, normalize_predictions

        frame, times, events = self._frame()
        normalized, audit = normalize_predictions(frame, "cif_ipcw")
        self.assertEqual(audit["estimand"], "cif_ipcw")
        self.assertGreater(audit["zero_weight_censored_rows"], 0)
        for model in ("A", "B"):
            rows = normalized.loc[normalized.model == model].sort_values("subject_id")
            ours = metric_values(rows.label, rows.probability, ipcw=rows.ipcw_weight)
            native = compute_competing_risk_metrics_at_horizon(
                rows.time_days.to_numpy(), rows.event.to_numpy(),
                rows.probability.to_numpy(), 365.0,
            )
            self.assertAlmostEqual(ours["auc"], native["cif_auc"], places=10)
            self.assertAlmostEqual(ours["brier"], native["cif_brier"], places=10)

    def test_ipcw_paired_bootstrap_and_zero_weight_placeholders(self):
        from opera.evaluation.secondary_analysis import (
            normalize_predictions,
            paired_analysis,
        )

        frame, _, _ = self._frame(n=200)
        censored_early = (frame.time_days < 365) & (frame.event == 0)
        tabular = frame.loc[frame.model == "A"].copy()
        tabular["model"] = "X"
        tabular["prediction_absent_zero_weight"] = censored_early[tabular.index]
        tabular.loc[tabular.prediction_absent_zero_weight, "probability"] = 0.0
        normalized, _ = normalize_predictions(pd.concat([frame, tabular]), "cif_ipcw")
        tables = paired_analysis(
            normalized, {"T": "gain(A,X)", "AB": "gain(A,B)"}, replicates=30
        )
        gains = tables["contrasts"].set_index(["contrast", "metric"]).gain
        # X equals A wherever weights are positive: the placeholders are inert.
        self.assertAlmostEqual(gains[("T", "auc")], 0.0, places=12)
        self.assertAlmostEqual(gains[("T", "brier")], 0.0, places=12)
        self.assertNotEqual(gains[("AB", "auc")], 0.0)
        broken = tabular.copy()
        known = ~censored_early[broken.index]
        broken.loc[known.idxmax(), "prediction_absent_zero_weight"] = True
        with self.assertRaises(ValueError):
            normalize_predictions(pd.concat([frame, broken]), "cif_ipcw")

    def test_estimand_must_be_declared_and_not_mixed(self):
        from opera.evaluation.secondary_analysis import normalize_predictions

        frame, _, _ = self._frame(n=50)
        positive = (frame.event == 1) & (frame.time_days <= frame.horizon_days)
        labelled = frame.assign(
            label=positive.astype(int),
            known=positive
            | (frame.time_days >= frame.horizon_days)
            | ((frame.event == 2) & (frame.time_days <= frame.horizon_days)),
        )
        known_only, _ = normalize_predictions(labelled)
        full, _ = normalize_predictions(frame, "cif_ipcw")
        self.assertLess(len(known_only), len(full))
        self.assertNotIn("ipcw_weight", known_only)
        with self.assertRaises(ValueError):
            normalize_predictions(frame, "ipcw_net_risk")


class CalibrationSummaryTest(unittest.TestCase):
    def test_slope_and_calibration_in_the_large_recover_known_miscalibration(self):
        from opera.evaluation.secondary_analysis import calibration_summary

        rng = np.random.default_rng(0)
        n = 20000
        truth = rng.uniform(0.02, 0.6, n)
        label = rng.binomial(1, truth)
        logit = np.log(truth / (1 - truth))
        overconfident = 1 / (1 + np.exp(-(2.0 * logit)))  # slope 0.5 expected
        rows = []
        for model, prob in (("calibrated", truth), ("overconfident", overconfident)):
            rows.append(pd.DataFrame({
                "model": model, "split": "held_out", "seed": 42, "cohort": "CLL",
                "outcome": "event", "horizon_days": 365, "subject_id": np.arange(n),
                "label": label, "probability": prob}))
        summary = calibration_summary(pd.concat(rows)).set_index("model")
        self.assertAlmostEqual(summary.loc["calibrated", "calibration_slope"], 1.0, delta=0.06)
        self.assertAlmostEqual(summary.loc["overconfident", "calibration_slope"], 0.5, delta=0.04)
        self.assertAlmostEqual(summary.loc["calibrated", "calibration_in_the_large"], 0.0, delta=0.01)
        self.assertAlmostEqual(summary.loc["calibrated", "observed_expected_ratio"], 1.0, delta=0.04)


def test_ipa_is_brier_relative_to_the_event_rate_and_matches_bootstrap():
    import numpy as np

    from opera.evaluation.secondary_analysis import bootstrap_metric_values, metric_values

    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.2, 400)
    p = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.1, 400), 0.01, 0.99)
    ipcw = rng.uniform(0.8, 1.5, 400) * (rng.uniform(size=400) > 0.1)
    for weights in (None, ipcw):
        point = metric_values(y, p, ipcw=weights)
        rate = (y * (1 if weights is None else weights)).sum() / len(y)
        null = (((y - rate) ** 2) * (1 if weights is None else weights)).sum() / len(y)
        assert np.isclose(point["ipa"], 1 - point["brier"] / null)
        draws = bootstrap_metric_values(y, p, np.ones((2, 400)), ipcw=weights)
        assert np.allclose(draws["ipa"], point["ipa"])
    # The event rate itself scores IPA 0.
    assert np.isclose(metric_values(y, np.full(400, y.mean()))["ipa"], 0)
