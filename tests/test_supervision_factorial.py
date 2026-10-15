"""Supervision scopes and full-split selection, without GPU dependencies."""

import copy
import unittest

import numpy as np

from opera.functional.checkpoint_selection import horizon_bce_values
from opera.run.prepare_supervision_factorial import SELECTION_METRIC, factorial_configs


def base_config():
    return {
        "dapt_ckpt": "/old64/best.ckpt",
        "source_opera_ckpt": None,
        "model": {
            "freeze_encoder": False,
            "pooling": "cls_last",
            "dapt_anchor_weight": 0,
        },
        "cohorts": {
            "CLL_SLL": {
                "cohort_fine_col": "cohort_grouped",
                "cohort_fine_value": "CLL_SLL",
            },
            "HCL": {"exclude_outcomes": ["second_line_treatment"]},
        },
        "outcomes": {
            "overall_survival": {"outcome_file": "survival.parquet"},
            "second_line_treatment": {"competing_outcome_file": "survival.parquet"},
        },
        "competing_risk": {
            "objective": "horizon_bce",
            "readout_mode": "shared",
            "head_mode": "linear",
            "loss_weight": 1,
            "contrastive_loss_weight": 0,
            "interval_boundaries_days": [30, 365],
            "curriculum": {"enabled": True},
        },
        "cross_outcome": {
            "aggregation": "endpoint_group_macro",
            "weighter": "uniform",
            "endpoint_groups": {
                "survival": ["overall_survival"],
                "treatment": ["second_line_treatment"],
            },
        },
        "training": {"evaluation_horizons_days": [30, 365], "learning_rate": 0.0005},
    }


class SupervisionFactorialTest(unittest.TestCase):
    def test_scopes_exclusions_initialization_and_immutability(self):
        base = base_config()
        original = copy.deepcopy(base)
        configs = factorial_configs(base, encoder_checkpoint="/new256/best.ckpt")
        self.assertEqual(base, original)
        self.assertNotIn("S_HCL__second_line_treatment", configs)
        disease = configs["O_CLL__all_outcomes"]
        self.assertEqual(disease["cohorts"]["CLL_SLL"]["cohort_fine_value"], "CLL")
        self.assertEqual(
            disease["cohorts"]["CLL_SLL"]["cohort_fine_col"], "cohort_fine"
        )
        self.assertEqual(
            set(configs["D_all_diseases__second_line_treatment"]["cohorts"]),
            {"CLL_SLL"},
        )
        self.assertEqual(
            set(configs["O_HCL__all_outcomes"]["outcomes"]), {"overall_survival"}
        )
        for config in configs.values():
            self.assertEqual(config["dapt_ckpt"], "/new256/best.ckpt")
            self.assertEqual(
                config["training"]["eval_monitor_metric"], SELECTION_METRIC
            )
            self.assertFalse(config["competing_risk"]["curriculum"]["enabled"])
            self.assertEqual(config["competing_risk"]["readout_mode"], "shared")
            if "second_line_treatment" in config["outcomes"]:
                self.assertEqual(
                    config["outcomes"]["second_line_treatment"],
                    original["outcomes"]["second_line_treatment"],
                )

    def test_router_or_supervised_initialization_rejected(self):
        for field, value in (
            ("readout_mode", "source_anchored_adapters"),
            ("objective", "piecewise_exponential"),
        ):
            base = base_config()
            base["competing_risk"][field] = value
            with self.assertRaises(ValueError):
                factorial_configs(base, encoder_checkpoint="/new256/best.ckpt")
        base = base_config()
        base["source_opera_ckpt"] = "/joint/best.ckpt"
        with self.assertRaises(ValueError):
            factorial_configs(base, encoder_checkpoint="/new256/best.ckpt")

    def test_single_class_censoring_and_competing_death_selection(self):
        # Early-censored patient is unknown; death is known negative;
        # primary event at day 60 is negative at 30 and positive at 365.
        values = horizon_bce_values(
            [5, 10, 60, 365],
            [0, 2, 1, 0],
            [[0.99, 0.99], [0.1, 0.2], [0.2, 0.8], [0.3, 0.4]],
            [30, 365],
        )
        expected = [-np.log([0.9, 0.8, 0.7]).mean(), -np.log([0.8, 0.8, 0.6]).mean()]
        np.testing.assert_allclose(values, expected)
        self.assertEqual(horizon_bce_values([5], [0], [[0.2]], [30]), [])
        np.testing.assert_allclose(
            horizon_bce_values([365], [0], [[0.2]], [30]), [-np.log(0.8)]
        )

    def test_selection_uses_all_patients_and_rejects_nonfinite_predictions(self):
        values = horizon_bce_values([365] * 3, [0] * 3, [[0.1], [0.1], [0.9]], [365])
        self.assertAlmostEqual(values[0], -np.log([0.9, 0.9, 0.1]).mean())
        with self.assertRaises(ValueError):
            horizon_bce_values([365], [0], [[float("nan")]], [365])


if __name__ == "__main__":
    unittest.main()
