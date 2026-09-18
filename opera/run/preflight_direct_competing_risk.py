"""Compose and validate a direct competing-risk run before requesting a GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from opera.functional.direct_cr_contract import validate_direct_cr_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--require-existing-paths", action="store_true")
    return parser


def _required_paths(config: dict) -> dict[str, Path]:
    paths = {
        "DAPT checkpoint": Path(str(config["dapt_ckpt"])),
        "vocabulary": Path(str(config["paths"]["vocabulary"])),
    }
    for cohort, cohort_config in config["cohorts"].items():
        paths[f"cohort {cohort} data_dir"] = Path(str(cohort_config["data_dir"]))
        population = cohort_config.get("population_file")
        if population:
            paths[f"cohort {cohort} population_file"] = Path(str(population))
    for outcome, outcome_config in config["outcomes"].items():
        path = outcome_config.get("path") or outcome_config.get("outcome_file")
        if path:
            paths[f"outcome {outcome}"] = Path(str(path))
    return paths


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base="1.2"
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    validate_direct_cr_config(resolved)
    if args.require_existing_paths:
        missing = [
            f"{label}: {path}"
            for label, path in _required_paths(resolved).items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))
    print(
        "Direct competing-risk preflight passed: "
        f"{len(resolved['cohorts'])} cohorts, {len(resolved['outcomes'])} outcomes, "
        f"boundaries={resolved['competing_risk']['interval_boundaries_days']}, "
        f"head={resolved['competing_risk'].get('head_mode', 'linear')}, "
        f"weighter={resolved['competing_risk'].get('weighter', 'uniform')}."
    )


if __name__ == "__main__":
    main()
