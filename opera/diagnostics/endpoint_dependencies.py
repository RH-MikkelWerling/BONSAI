"""Auditable endpoint relationships for scaffold-discovery diagnostics.

The dependency graph is deliberately separate from outcome families.  Families
describe clinical similarity; dependencies describe label construction and are
therefore exclusions/controls when estimating evidence for transfer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Mapping, Sequence


_GRADE_RE = re.compile(r"^(?P<stem>.+)_g(?P<grade>[23])plus$", re.IGNORECASE)


@dataclass(frozen=True)
class EndpointRelationship:
    outcome_a: str
    outcome_b: str
    relationship: str
    scaffold_eligible: bool
    reason: str


def nested_threshold_stem(name: str) -> tuple[str, int] | None:
    match = _GRADE_RE.match(name)
    if match is None:
        return None
    return match.group("stem").lower(), int(match.group("grade"))


def classify_endpoint_pair(
    outcome_a: str,
    outcome_b: str,
    *,
    composites: Mapping[str, Sequence[str]] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> EndpointRelationship:
    """Classify a pair without using observed gradients or event rates."""
    composites = composites or {}
    aliases = aliases or {}
    a_nested = nested_threshold_stem(outcome_a)
    b_nested = nested_threshold_stem(outcome_b)
    if a_nested and b_nested and a_nested[0] == b_nested[0]:
        return EndpointRelationship(
            outcome_a, outcome_b, "nested_threshold_control", False,
            "Deterministically nested G2+/G3+ definitions are retained as a control, "
            "but cannot establish cross-outcome transfer.",
        )

    for composite, components in composites.items():
        members = set(components)
        if (outcome_a == composite and outcome_b in members) or (
            outcome_b == composite and outcome_a in members
        ):
            return EndpointRelationship(
                outcome_a, outcome_b, "component_composite", False,
                "One endpoint is an explicit component of the other.",
            )
    if outcome_a in composites or outcome_b in composites:
        return EndpointRelationship(
            outcome_a, outcome_b, "contains_composite", False,
            "Composite endpoints are excluded from scaffold discovery.",
        )

    for canonical, variants in aliases.items():
        group = {canonical, *variants}
        if outcome_a in group and outcome_b in group:
            return EndpointRelationship(
                outcome_a, outcome_b, "alternate_definition", False,
                "Alternate definitions of the same construct are not independent scaffolds.",
            )
    return EndpointRelationship(
        outcome_a, outcome_b, "atomic_candidate", True,
        "No configured deterministic label dependency was found.",
    )


def classify_endpoint_pairs(
    outcomes: Iterable[str],
    *,
    composites: Mapping[str, Sequence[str]] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[dict]:
    names = list(outcomes)
    return [
        asdict(
            classify_endpoint_pair(
                first, second, composites=composites, aliases=aliases
            )
        )
        for i, first in enumerate(names)
        for second in names[i + 1 :]
    ]


def dependency_config(config: Mapping | None) -> tuple[dict, dict]:
    """Read the optional ``endpoint_dependencies`` diagnostic config block."""
    block = dict((config or {}).get("endpoint_dependencies", config or {}))
    return dict(block.get("composites", {})), dict(block.get("aliases", {}))
