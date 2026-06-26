"""Cohort-scope contracts for adaptation breadth experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


FROZEN_GENERAL_PRETRAINING = "frozen"
SCOPE_TARGET_ONLY = "target_only"
SCOPE_TARGET_PLUS_NEIGHBOURS = "target_plus_neighbours"
SCOPE_FULL_SPECTRUM = "full_spectrum"
DEFAULT_SCOPE_ORDER = (
    SCOPE_TARGET_ONLY,
    SCOPE_TARGET_PLUS_NEIGHBOURS,
    SCOPE_FULL_SPECTRUM,
)


@dataclass(frozen=True)
class CohortScope:
    """One rung in the adaptation breadth ladder.

    DAPT and contrastive stages use ``adaptation_cohorts``. Finetuning must
    always use ``finetune_cohort`` so breadth changes do not silently change the
    target data used for the supervised comparison.
    """

    name: str
    target_cohort: str
    adaptation_cohorts: tuple[str, ...]
    finetune_cohort: str
    general_pretraining: str = FROZEN_GENERAL_PRETRAINING

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adaptation_cohorts"] = list(self.adaptation_cohorts)
        return payload

    def hydra_overrides(self) -> dict[str, list[str]]:
        cohorts = ",".join(self.adaptation_cohorts)
        return {
            "dapt": [
                f"cohort_scope.name={self.name}",
                f"cohort_scope.general_pretraining={self.general_pretraining}",
                f"cohort_scope.adaptation_cohorts=[{cohorts}]",
            ],
            "contrastive": [
                f"cohort_scope.name={self.name}",
                f"cohort_scope.general_pretraining={self.general_pretraining}",
                f"cohort_scope.adaptation_cohorts=[{cohorts}]",
            ],
            "finetune": [
                f"cohort_scope.name={self.name}",
                f"cohort_scope.general_pretraining={self.general_pretraining}",
                f"cohort_scope.adaptation_cohorts=[{cohorts}]",
                f"cohort_scope.finetune_cohort={self.finetune_cohort}",
                f"dataset={self.finetune_cohort}",
            ],
        }


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return tuple(unique)


def _cohort_names_from_config(cohorts: Mapping[str, Any] | Sequence[str]) -> tuple[str, ...]:
    if isinstance(cohorts, Mapping):
        return tuple(str(name) for name in cohorts.keys())
    return tuple(str(name) for name in cohorts)


def build_adaptation_breadth_ladder(
    *,
    target_cohort: str,
    cohorts: Mapping[str, Any] | Sequence[str],
    neighbours: Mapping[str, Sequence[str]] | None = None,
    scope_order: Sequence[str] = DEFAULT_SCOPE_ORDER,
) -> list[CohortScope]:
    """Build the target/neighbour/full adaptation ladder for one target cohort."""

    cohort_names = _cohort_names_from_config(cohorts)
    if target_cohort not in cohort_names:
        raise ValueError(
            f"target_cohort={target_cohort!r} is not present in configured cohorts"
        )

    neighbour_names = tuple(str(item) for item in (neighbours or {}).get(target_cohort, ()))
    unknown_neighbours = sorted(set(neighbour_names) - set(cohort_names))
    if unknown_neighbours:
        raise ValueError(
            f"Neighbour cohorts are not configured: {', '.join(unknown_neighbours)}"
        )

    rungs: list[CohortScope] = []
    for scope_name in scope_order:
        if scope_name == SCOPE_TARGET_ONLY:
            adaptation_cohorts = (target_cohort,)
        elif scope_name == SCOPE_TARGET_PLUS_NEIGHBOURS:
            adaptation_cohorts = _ordered_unique((target_cohort, *neighbour_names))
        elif scope_name == SCOPE_FULL_SPECTRUM:
            adaptation_cohorts = cohort_names
        else:
            raise ValueError(f"Unknown adaptation scope {scope_name!r}")
        rungs.append(
            CohortScope(
                name=scope_name,
                target_cohort=target_cohort,
                adaptation_cohorts=adaptation_cohorts,
                finetune_cohort=target_cohort,
            )
        )
    validate_adaptation_breadth_ladder(rungs)
    return rungs


def validate_adaptation_breadth_ladder(scopes: Sequence[CohortScope]) -> None:
    if not scopes:
        raise ValueError("Adaptation breadth ladder is empty")
    names = [scope.name for scope in scopes]
    if len(names) != len(set(names)):
        raise ValueError("Adaptation breadth ladder has duplicate scope names")
    target = scopes[0].target_cohort
    finetune = scopes[0].finetune_cohort
    for scope in scopes:
        if scope.target_cohort != target:
            raise ValueError("All ladder rungs must share one target_cohort")
        if scope.finetune_cohort != finetune or finetune != target:
            raise ValueError("All ladder rungs must keep finetune_cohort fixed")
        if scope.general_pretraining != FROZEN_GENERAL_PRETRAINING:
            raise ValueError("General pretraining must remain frozen for the ladder")
        if target not in scope.adaptation_cohorts:
            raise ValueError(f"Scope {scope.name!r} excludes the target cohort")


def build_ladder_plan(scopes: Sequence[CohortScope]) -> dict[str, Any]:
    validate_adaptation_breadth_ladder(scopes)
    return {
        "target_cohort": scopes[0].target_cohort,
        "general_pretraining": FROZEN_GENERAL_PRETRAINING,
        "finetune_cohort": scopes[0].finetune_cohort,
        "scopes": [
            {
                **scope.to_dict(),
                "hydra_overrides": scope.hydra_overrides(),
            }
            for scope in scopes
        ],
    }


def load_ladder_manifest(path: str | Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest {path} must contain a mapping")
    return payload


def build_ladder_plan_from_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_ladder_manifest(path)
    if manifest.get("general_pretraining", FROZEN_GENERAL_PRETRAINING) != (
        FROZEN_GENERAL_PRETRAINING
    ):
        raise ValueError("Adaptation breadth ladder requires frozen general pretraining")
    scopes = build_adaptation_breadth_ladder(
        target_cohort=str(manifest["target_cohort"]),
        cohorts=manifest["cohorts"],
        neighbours=manifest.get("neighbours", {}),
        scope_order=manifest.get("scope_order", DEFAULT_SCOPE_ORDER),
    )
    return build_ladder_plan(scopes)
