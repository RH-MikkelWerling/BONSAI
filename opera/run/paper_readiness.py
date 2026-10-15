"""Read-only paper-input checks; never print unrelated .env values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from opera.functional.cohort_groups import ALL_GROUPED
from opera.run.paper_experiments import load_environment, path_value
from opera.run.prepare_paper_run_bundle import load_resolved_config


def check(env):
    paths = {
        key: path_value(env, key)
        for key in (
            "OPERA_RELEASE_DIR",
            "OPERA_SEQUENCE_DATA_DIR",
            "BONSAI_COHORT_MEMBERSHIP",
            "BONSAI_OUTCOMES_DIR",
            "OPERA_PAPER_BASE_CONFIG",
            "OPERA_PAPER_ROUTER_REFERENCE",
            "OPERA_PAPER_LOCAL_CKPT",
        )
    }
    release = paths["OPERA_RELEASE_DIR"]
    if paths["BONSAI_OUTCOMES_DIR"] != (release / "outcomes").resolve():
        raise ValueError(
            "BONSAI_OUTCOMES_DIR must point to the selected release's outcomes"
        )
    registry = release / "experiment_registry_primary_v2.yaml"
    required = [
        registry,
        paths["BONSAI_COHORT_MEMBERSHIP"],
        paths["OPERA_PAPER_BASE_CONFIG"],
        paths["OPERA_PAPER_ROUTER_REFERENCE"],
    ]
    required += [
        paths["OPERA_SEQUENCE_DATA_DIR"] / name
        for name in ("vocabulary.pt", "subject_data_train.pt", "subject_data_tuning.pt")
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing paper inputs: " + ", ".join(missing))
    roster = yaml.safe_load(registry.read_text())["outcomes"]
    base = load_resolved_config(paths["OPERA_PAPER_BASE_CONFIG"])
    if (
        len(roster) != 69
        or set(base["outcomes"]) != set(roster)
        or set(base["cohorts"]) != set(ALL_GROUPED)
    ):
        raise ValueError("Expected the approved 69 outcomes and all ten disease groups")
    expected = paths["OPERA_SEQUENCE_DATA_DIR"] / "vocabulary.pt"
    if Path(base["paths"]["vocabulary"]).resolve() != expected.resolve():
        raise ValueError("Base config vocabulary differs from selected sequence data")
    for cohort, config in base["cohorts"].items():
        if (
            Path(config["population_file"]).resolve()
            != paths["BONSAI_COHORT_MEMBERSHIP"]
        ):
            raise ValueError(f"Base metadata differs from sealed metadata for {cohort}")
    checkpoint = paths["OPERA_PAPER_LOCAL_CKPT"]
    return {
        "status": "paper_inputs_checked_not_gpu_smoke_tested",
        "n_outcomes": 69,
        "n_disease_groups": 10,
        "local_encoder": "present_completion_audit_required"
        if checkpoint.exists()
        else "pending_pretraining_finish",
        "supervisor_signoff": "record_separately",
        "environment_values": "unrelated_values_not_printed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    env = load_environment(args.env_file)
    import os

    # Hydra oc.env must see the same explicit dotenv settings as these checks.
    os.environ.update(env)
    print(json.dumps(check(env), indent=2))


if __name__ == "__main__":
    main()
