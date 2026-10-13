"""Compose and validate a direct competing-risk run before requesting a GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from opera.functional.direct_cr_contract import validate_direct_cr_config
from opera.modules.networks.competing_risk import validate_competing_risk_sampling


def _register_resolvers() -> None:
    # ``run_id`` only determines an output-directory suffix and is irrelevant
    # to this read-only preflight. Training registers the production resolver;
    # use a stable sentinel here so fully resolving the same config cannot fail.
    OmegaConf.register_new_resolver(
        "version",
        lambda: 0,
        use_cache=True,
        replace=True,
    )


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
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--borrowed-update-reference", type=Path,
                        help="Require scientific settings to match this saved anchored config except gradient ownership.")
    reference.add_argument("--curriculum-off-reference", type=Path,
                           help="Require scientific settings to match this saved config except disabling curriculum.")
    reference.add_argument("--adapter-epsilon-reference", type=Path,
                           help="Require scientific settings to match except lower adapter-matrix epsilon.")
    return parser


def validate_borrowed_update_match(config: dict, reference: dict) -> None:
    import copy

    actual = copy.deepcopy(config)
    expected = copy.deepcopy(reference)
    if actual["competing_risk"].get("readout_mode") != "unanchored_adapters":
        raise ValueError("Borrowed-update comparison must use unanchored_adapters")
    if expected["competing_risk"].get("readout_mode") != "source_anchored_adapters":
        raise ValueError("Borrowed-update reference must use source_anchored_adapters")
    actual["competing_risk"]["readout_mode"] = "source_anchored_adapters"
    _validate_scientific_settings_match(actual, expected)


def validate_curriculum_off_match(config: dict, reference: dict) -> None:
    import copy

    actual, expected = copy.deepcopy(config), copy.deepcopy(reference)
    if actual["competing_risk"].get("curriculum", {}).get("enabled") is not False:
        raise ValueError("Curriculum-off comparison must explicitly disable curriculum")
    if expected["competing_risk"].get("curriculum", {}).get("enabled") is not True:
        raise ValueError("Curriculum-off reference must explicitly enable curriculum")
    actual["competing_risk"]["curriculum"]["enabled"] = True
    _validate_scientific_settings_match(actual, expected)


def _validate_scientific_settings_match(actual: dict, expected: dict) -> None:
    for item in (actual, expected):
        item["competing_risk"].setdefault("adapter_source_mode", "both")
    keys = ("seed", "model", "training", "hardware", "cohorts", "outcomes",
            "cross_outcome", "competing_risk", "adapter_initialization",
            "source_opera_ckpt", "dapt_ckpt", "dapt_embedding_store")
    differences = [key for key in keys if actual.get(key) != expected.get(key)]
    for key in ("sequence_data_dir", "vocabulary"):
        if actual.get("paths", {}).get(key) != expected.get("paths", {}).get(key):
            differences.append(f"paths.{key}")
    if differences:
        raise ValueError("Matched router settings differ from the reference in: " + ", ".join(differences))


def validate_adapter_epsilon_match(config: dict, reference: dict) -> None:
    import copy
    import math

    actual, expected = copy.deepcopy(config), copy.deepcopy(reference)
    epsilon = float(actual['training'].get('adapter_optimizer_epsilon', float('nan')))
    prior = expected['training'].get('adapter_optimizer_epsilon')
    previous = float(expected['training']['optimizer_epsilon'] if prior is None else prior)
    if not math.isfinite(epsilon) or not 0 < epsilon < previous:
        raise ValueError('Adapter epsilon must be positive and lower than the reference')
    if prior is None:
        actual['training'].pop('adapter_optimizer_epsilon', None)
        expected['training'].pop('adapter_optimizer_epsilon', None)
    else:
        actual['training']['adapter_optimizer_epsilon'] = prior
    _validate_scientific_settings_match(actual, expected)


def _required_paths(config: dict) -> dict[str, Path]:
    paths = {
        "DAPT checkpoint": Path(str(config["dapt_ckpt"])),
        "vocabulary": Path(str(config["paths"]["vocabulary"])),
    }
    if config.get("source_opera_ckpt"):
        paths["shared BCE source checkpoint"] = Path(str(config["source_opera_ckpt"]))
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
    _register_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base="1.2"
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    validate_direct_cr_config(resolved)
    validate_competing_risk_sampling(
        resolved.get("competing_risk"),
        resolved.get("training", {}).get("batch_sampling"),
    )
    reference_path = args.borrowed_update_reference or args.curriculum_off_reference or args.adapter_epsilon_reference
    if reference_path:
        # Resolve scientific sections individually; unrelated runtime/output
        # fields in a saved Hydra config do not need to resolve in preflight.
        saved = OmegaConf.load(reference_path)
        keys = ("seed", "model", "training", "hardware", "cohorts", "outcomes",
                "cross_outcome", "competing_risk", "adapter_initialization",
                "source_opera_ckpt", "dapt_ckpt", "dapt_embedding_store")
        reference = {}
        for key in keys:
            value = saved.get(key)
            reference[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        reference["paths"] = {key: saved.get("paths", {}).get(key)
                              for key in ("sequence_data_dir", "vocabulary")}
        if args.borrowed_update_reference:
            validate_borrowed_update_match(resolved, reference)
            print("Matched anchored reference: only adapter parameter gradient ownership changes.")
        elif args.curriculum_off_reference:
            validate_curriculum_off_match(resolved, reference)
            print("Matched curriculum-on reference: only curriculum.enabled changes to false.")
        else:
            validate_adapter_epsilon_match(resolved, reference)
            print("Matched optimizer reference: only source down/up matrix epsilon changes.")
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
        f"objective={resolved['competing_risk'].get('objective', 'piecewise_exponential')}, "
        f"head={resolved['competing_risk'].get('head_mode', 'linear')}, "
        f"weighter={resolved['competing_risk'].get('weighter', 'uniform')}."
    )


if __name__ == "__main__":
    main()
