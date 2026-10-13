"""Version-1 endpoint groups: grade variants together, distinct endpoints apart."""
import re
import numpy as np


def endpoint_groups_v1(outcomes):
    groups = {}
    for name in outcomes:
        group = re.sub(r"_g[234]plus$", "", name)
        # Release-v2 aliases preserve two nested historical ALP thresholds and
        # two nested numerical sodium thresholds. Keep each pair weighted as
        # one endpoint when constructing endpoint-group-macro experiments.
        if name in {"alp_elevation_legacy_tier2plus", "alp_elevation_legacy_tier3plus"}:
            group = "alp_elevation_legacy"
        elif name == "sodium_below_125_mmol_l":
            group = "hyponatremia"
        groups.setdefault(group, []).append(name)
    return groups


def validate_endpoint_groups(groups, outcomes):
    members = [name for values in groups.values() for name in values]
    if any(not values for values in groups.values()) or sorted(members) != sorted(outcomes):
        raise ValueError("endpoint_groups must cover each configured outcome exactly once.")


def endpoint_group_score(values, groups):
    """Horizon mean -> grade mean -> group mean; return score and coverage."""
    means = [np.mean([np.mean(values[name]) for name in members if values.get(name)])
             for members in groups.values() if any(values.get(name) for name in members)]
    return (float(np.mean(means)) if means else float("nan")), len(means)
