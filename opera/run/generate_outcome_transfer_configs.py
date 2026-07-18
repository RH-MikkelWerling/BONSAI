"""Resolve the focused OPERA outcome-transfer manifest and generate configs.

Usage
-----
Validate and write the deterministic plan/configs::

    python -m opera.run.generate_outcome_transfer_configs

Resolve only (useful in CI/preflight)::

    python -m opera.run.generate_outcome_transfer_configs --validate-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from opera.functional.outcome_transfer import (
    DEFAULT_MANIFEST,
    EXPECTED_CONDITIONS,
    REPOSITORY_ROOT,
    build_transfer_config,
    resolve_transfer_manifest,
    write_resolved_transfer_plan,
)


DEFAULT_OUTPUT_DIR = Path("opera/configs/generated/outcome_transfer")


class _NoAliasDumper(yaml.SafeDumper):
    """Keep generated configs readable and stable under review."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = REPOSITORY_ROOT / source
    with source.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}.")
    return loaded


def generate_outcome_transfer_configs(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    registry_path: str | Path | None = None,
    base_config_path: str | Path | None = None,
) -> list[Path]:
    """Generate seven condition templates plus resolved JSON/CSV provenance."""
    plan = resolve_transfer_manifest(
        manifest_path,
        registry_path=registry_path,
        base_config_path=base_config_path,
    )
    base_path = base_config_path or plan["base_contrastive_config"]
    base_config = _load_yaml(base_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for condition in EXPECTED_CONDITIONS:
        config = build_transfer_config(plan, condition, base_config)
        config["checkpoint_reuse_policy"] = plan["checkpoint_reuse"].get(
            condition, "new_contrastive_training_required"
        )
        path = destination / f"{condition}.yaml"
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

    json_path, csv_path = write_resolved_transfer_plan(plan, destination)
    written.extend([json_path, csv_path])
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve and validate the manifest without writing generated files.",
    )
    args = parser.parse_args()
    plan = resolve_transfer_manifest(
        args.manifest,
        registry_path=args.registry,
        base_config_path=args.base_config,
    )
    if args.validate_only:
        print(
            "Validated outcome-transfer manifest: "
            f"{plan['checkpoint_slot_count']} checkpoint slots, "
            f"{len(plan['evaluation_target_union'])} evaluation targets."
        )
        return
    written = generate_outcome_transfer_configs(
        args.manifest,
        output_dir=args.output_dir,
        registry_path=args.registry,
        base_config_path=args.base_config,
    )
    print(f"Generated {len(written)} outcome-transfer files in {args.output_dir}")


if __name__ == "__main__":
    main()
