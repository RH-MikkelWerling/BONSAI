import importlib.util
from pathlib import Path


def test_versioned_numeric_aliases_remain_in_their_endpoint_groups() -> None:
    path = Path("opera/functional/endpoint_groups.py")
    spec = importlib.util.spec_from_file_location("release_endpoint_groups", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [
        "alp_elevation_legacy_tier2plus", "alp_elevation_legacy_tier3plus",
        "hyponatremia_g2plus", "sodium_below_125_mmol_l",
        "potassium_below_3p0_mmol_l",
    ]
    groups = module.endpoint_groups_v1(names)
    assert groups["alp_elevation_legacy"] == names[:2]
    assert groups["hyponatremia"] == names[2:4]
    module.validate_endpoint_groups(groups, names)
