import json

import numpy as np
import pandas as pd
import yaml

from opera.diagnostics.outcome_trunk_discovery import run_trunk_discovery


def _gradient_batches() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    groups = [
        ["anemia_g2plus", "anemia_g3plus", "red_1", "red_2", "red_composite"],
        ["infection_1", "infection_2", "infection_3", "infection_4", "infection_5"],
        ["renal_1", "renal_2", "renal_3", "renal_4", "renal_5"],
    ]
    family = {outcome: index for index, group in enumerate(groups) for outcome in group}
    outcomes = [outcome for group in groups for outcome in group]
    records = []
    for batch in range(24):
        for i, first in enumerate(outcomes):
            for second in outcomes[i + 1 :]:
                mean = 0.72 if family[first] == family[second] else -0.18
                records.append(
                    {
                        "batch_index": batch,
                        "gradient_component": "full_likelihood",
                        "outcome_a": first,
                        "outcome_b": second,
                        "cosine": mean + rng.normal(scale=0.035),
                        "null_cosine": 0.04,
                        "excess_alignment": mean - 0.04 + rng.normal(scale=0.035),
                        "joint_support": 32,
                    }
                )
    return pd.DataFrame(records)


def test_discovery_recovers_groups_and_writes_reusable_config(tmp_path):
    dependencies = {
        "endpoint_dependencies": {
            "composites": {"red_composite": ["red_1", "red_2"]},
            "aliases": {},
        }
    }
    summary = run_trunk_discovery(
        _gradient_batches(),
        tmp_path,
        dependency_cfg=dependencies,
        min_size=5,
        max_size=5,
        n_clusters=3,
        attempts=16,
        n_bootstrap=12,
        seed=4,
    )

    membership = pd.read_csv(tmp_path / "learned_trunk_membership.csv")
    trunk_by_outcome = membership.set_index("outcome")["trunk"].to_dict()
    assert trunk_by_outcome["anemia_g2plus"] == trunk_by_outcome["anemia_g3plus"]
    assert trunk_by_outcome["red_composite"] == trunk_by_outcome["red_1"]
    assert trunk_by_outcome["red_composite"] == trunk_by_outcome["red_2"]
    assert membership.groupby("trunk").size().tolist() == [5, 5, 5]
    assert summary["affinity_value"] == "excess_alignment"
    assert summary["mean_within_trunk_coassignment"] > 0.95

    config = yaml.safe_load((tmp_path / "learned_outcome_families.yaml").read_text())
    assert set(config) == {"outcome_families"}
    assert len(config["outcome_families"]) == 3
    saved = json.loads((tmp_path / "trunk_discovery_summary.json").read_text())
    assert saved["n_outcomes"] == 15
    assert (tmp_path / "outcome_pair_evidence.csv").exists()
    assert (tmp_path / "bootstrap_coassignment.csv").exists()
    assert (tmp_path / "learned_trunk_affinity_heatmap.png").exists()


def test_dependency_pairs_do_not_drive_partition_score(tmp_path):
    summary = run_trunk_discovery(
        _gradient_batches(),
        tmp_path,
        dependency_cfg={"endpoint_dependencies": {}},
        min_size=5,
        max_size=5,
        n_clusters=3,
        attempts=8,
        n_bootstrap=3,
        seed=2,
    )
    # Three size-five trunks contain 30 pairs, minus the deterministic G2+/G3+
    # pair, which is a must-link control rather than independent evidence.
    assert summary["selected_candidate"]["n_within_atomic_pairs"] == 29
