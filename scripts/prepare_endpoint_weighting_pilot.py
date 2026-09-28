"""Generate only the two matched pilot configs, leaving other configs untouched."""
from pathlib import Path
import yaml
from opera.run.generate_sweep_configs import (
    DEFAULT_REGISTRY, load_registry, build_endpoint_weighting_pilot,
    _hydra_environment, _NoAliasDumper,
)
from opera.functional.direct_cr_contract import validate_direct_cr_config


def main():
    registry = load_registry(DEFAULT_REGISTRY)
    for revised, name in ((True, "endpoint_groups"), (False, "weighting_reference")):
        config = _hydra_environment(build_endpoint_weighting_pilot(registry, revised=revised))
        validate_direct_cr_config(config)
        path = Path("opera/configs/generated") / f"direct_cr_bce_lowrank_{name}_v1.yaml"
        path.write_text("# @package _global_\n" + yaml.dump(
            config, Dumper=_NoAliasDumper, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
