"""Version-1 endpoint groups: grade variants together, distinct endpoints apart."""
import re
import numpy as np


def endpoint_groups_v1(outcomes):
    groups = {}
    for name in outcomes:
        group = re.sub(r"_g[234]plus$", "", name)
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
