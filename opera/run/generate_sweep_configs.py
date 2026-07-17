"""Generate production Cox/IPCW sweep configs from one locked registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = Path("opera/configs/experiment_registry.yaml")
DEFAULT_OUTPUT_DIR = Path("opera/configs/generated")


class _NoAliasDumper(yaml.SafeDumper):
    """Keep generated YAML reviewable by avoiding anchors for repeated lists."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    required = {
        "horizons_days",
        "death_outcome",
        "cohort_groups",
        "outcomes",
        "outcome_families",
        "model_variants",
    }
    missing = required - set(registry)
    if missing:
        raise ValueError(f"Experiment registry is missing: {sorted(missing)}")

    outcomes = list(registry["outcomes"])
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("Experiment registry contains duplicate outcome names.")
    if registry["death_outcome"] not in outcomes:
        raise ValueError("death_outcome must name an outcome in the registry.")

    family_members = [
        outcome
        for members in registry["outcome_families"].values()
        for outcome in members
    ]
    duplicate_family_members = sorted(
        {name for name in family_members if family_members.count(name) > 1}
    )
    if duplicate_family_members:
        raise ValueError(
            "Outcomes occur in multiple families: "
            f"{duplicate_family_members}"
        )
    missing_families = sorted(set(outcomes) - set(family_members))
    unknown_family_members = sorted(set(family_members) - set(outcomes))
    if missing_families or unknown_family_members:
        raise ValueError(
            "Outcome-family coverage mismatch; "
            f"missing={missing_families}, unknown={unknown_family_members}"
        )

    horizons = registry["horizons_days"]
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(not isinstance(value, int) or value <= 0 for value in horizons)
        or horizons != sorted(set(horizons))
    ):
        raise ValueError("horizons_days must be unique increasing positive integers.")

    groups = registry["cohort_groups"]
    fine_names: list[str] = []
    for grouped, config in groups.items():
        if not config.get("slug") or not config.get("fine"):
            raise ValueError(f"Cohort group {grouped!r} needs slug and fine mappings.")
        fine_names.extend(config["fine"])
    if len(fine_names) != len(set(fine_names)):
        raise ValueError("A fine cohort is assigned to more than one grouped cohort.")

    for rule_name, rule in registry.get("availability_rules", {}).items():
        unknown_outcomes = sorted(set(rule.get("outcomes", [])) - set(outcomes))
        unknown_groups = sorted(set(rule.get("excluded_grouped", [])) - set(groups))
        if unknown_outcomes or unknown_groups:
            raise ValueError(
                f"Availability rule {rule_name!r} has unknown outcomes/groups: "
                f"{unknown_outcomes}/{unknown_groups}"
            )


def _excluded_outcomes_by_group(registry: dict[str, Any]) -> dict[str, set[str]]:
    result = {name: set() for name in registry["cohort_groups"]}
    for rule in registry.get("availability_rules", {}).values():
        for grouped in rule.get("excluded_grouped", []):
            result[grouped].update(rule.get("outcomes", []))
    return result


def _cohorts(registry: dict[str, Any], level: str) -> dict[str, dict[str, Any]]:
    if level not in {"grouped", "fine"}:
        raise ValueError("level must be grouped or fine")
    data_dir = registry["paths"]["shared_data_dir"]
    population_file = registry["paths"]["cohort_membership_file"]
    exclusions = _excluded_outcomes_by_group(registry)
    result: dict[str, dict[str, Any]] = {}
    for grouped, group_cfg in registry["cohort_groups"].items():
        common = {
            "data_dir": data_dir,
            "population_file": population_file,
            "training_cohort": grouped,
            "ipi_score_col": None,
            "registry_start_date": None,
        }
        if exclusions[grouped]:
            common["exclude_outcomes"] = sorted(exclusions[grouped])
        if level == "grouped":
            result[grouped] = {
                **common,
                "cohort_fine_col": registry["cohort_columns"]["grouped"],
                "cohort_fine_value": grouped,
            }
            continue
        for fine in group_cfg["fine"]:
            result[fine] = {
                **common,
                "cohort_fine_col": registry["cohort_columns"]["fine"],
                "cohort_fine_value": fine,
            }
    return result


def _outcomes(
    registry: dict[str, Any], *, training_mode: str, horizon_days: int | None
) -> dict[str, dict[str, Any]]:
    if training_mode == "cox" and horizon_days is not None:
        raise ValueError("Cox generation does not take a fixed horizon.")
    if training_mode == "ipcw_bce" and horizon_days is None:
        raise ValueError("IPCW-BCE generation requires a fixed horizon.")
    death = registry["death_outcome"]
    result = {}
    for outcome in registry["outcomes"]:
        config: dict[str, Any] = {
            "outcome_file": f"{registry['paths']['outcomes_dir']}/{outcome}.parquet",
            "n_hours_start_include": 1,
            "n_hours_end_include": (
                None if horizon_days is None else int(horizon_days) * 24
            ),
            "registry_start_date": None,
        }
        if outcome != death:
            config["competing_outcome_path"] = (
                f"{registry['paths']['outcomes_dir']}/{death}.parquet"
            )
        result[outcome] = config
    return result


def build_sweep_config(
    registry: dict[str, Any], *, level: str, training_mode: str, horizon_days: int | None
) -> dict[str, Any]:
    objective = "cox" if training_mode == "cox" else f"ipcw_{horizon_days}d"
    variants = {
        name: {**config, "training_mode": training_mode}
        for name, config in registry["model_variants"].items()
    }
    return {
        "output_dir": f"{registry['paths']['output_root']}/{level}/{objective}",
        "finetune_base_config": "opera/configs/survival_finetune.yaml",
        "seeds": list(registry["seeds"]),
        "rarity_mode": "real",
        "cohorts": _cohorts(registry, level),
        "outcomes": _outcomes(
            registry,
            training_mode=training_mode,
            horizon_days=horizon_days,
        ),
        "model_variants": variants,
    }


def generate_configs(
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    registry = load_registry(registry_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    objectives = [("cox", None)] + [
        ("ipcw_bce", horizon) for horizon in registry["horizons_days"]
    ]
    for level in ("grouped", "fine"):
        for training_mode, horizon in objectives:
            label = "cox" if horizon is None else f"ipcw_{horizon}d"
            path = output / f"{level}_{label}.yaml"
            config = build_sweep_config(
                registry,
                level=level,
                training_mode=training_mode,
                horizon_days=horizon,
            )
            path.write_text(
                yaml.dump(
                    config,
                    Dumper=_NoAliasDumper,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            written.append(path)

    family_path = output / "outcome_families.yaml"
    family_path.write_text(
        yaml.dump(
            {"outcome_families": registry["outcome_families"]},
            Dumper=_NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    written.append(family_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    written = generate_configs(args.registry, args.output_dir)
    print(f"Generated {len(written)} files in {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
