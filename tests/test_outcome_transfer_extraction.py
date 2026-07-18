"""Pure provenance-contract tests for transfer embedding extraction."""

from __future__ import annotations

from copy import deepcopy

import pytest

from opera.functional.outcome_transfer import resolve_transfer_manifest
from opera.run.extract_outcome_transfer_embeddings import (
    OutcomeTransferExtractionError,
    _validate_checkpoint_identity,
)


def _tagged_checkpoint_metadata(plan, condition: str, seed: int) -> dict:
    expected = plan["conditions"][condition]
    return {
        "training_stage": "opera_contrastive_adaptation",
        "condition": condition,
        "transfer_level": expected["transfer_level"],
        "seed": seed,
        "registry_hash": plan["registry_hash"],
        "manifest_hash": plan["manifest_hash"],
        "base_contrastive_config_hash": plan["base_contrastive_config_hash"],
        "included_outcomes": list(expected["training_outcomes"]),
        "excluded_outcomes": list(expected["training_excluded_outcomes"]),
        "evaluation_outcomes": list(expected["evaluation_outcomes"]),
        "related_retained_outcomes": list(expected["related_retained_outcomes"]),
        "direct_dependencies_excluded": list(
            expected["direct_dependencies_excluded"]
        ),
        "selection_outcomes": list(expected["training_outcomes"]),
        "split_contract": plan["split_contract"],
        "split_contract_hash": plan["split_contract_hash"],
        # Production contrastive modules sort their head names internally.
        "outcome_set": sorted(expected["training_outcomes"]),
        "source_dapt_checkpoint": "/synthetic/dapt.ckpt",
    }


def test_extractor_requires_complete_tagged_opera_provenance() -> None:
    plan = resolve_transfer_manifest()
    metadata = _tagged_checkpoint_metadata(plan, "opera_no_g3", 42)
    assert (
        _validate_checkpoint_identity(
            representation="opera_no_g3",
            seed=42,
            metadata=metadata,
            plan=plan,
        )
        == "/synthetic/dapt.ckpt"
    )

    incomplete = deepcopy(metadata)
    incomplete.pop("evaluation_outcomes")
    with pytest.raises(OutcomeTransferExtractionError, match="incomplete provenance"):
        _validate_checkpoint_identity(
            representation="opera_no_g3",
            seed=42,
            metadata=incomplete,
            plan=plan,
        )

    stale = deepcopy(metadata)
    stale["split_contract_hash"] = "stale-split"
    with pytest.raises(OutcomeTransferExtractionError, match="split_contract_hash"):
        _validate_checkpoint_identity(
            representation="opera_no_g3",
            seed=42,
            metadata=stale,
            plan=plan,
        )


def test_legacy_full_maps_source_checkpoint_to_dapt_provenance() -> None:
    plan = resolve_transfer_manifest()
    metadata = {
        "training_stage": "opera_contrastive_adaptation",
        "outcome_set": sorted(plan["conditions"]["opera_full"]["training_outcomes"]),
        "source_checkpoint": "/synthetic/legacy-dapt.ckpt",
    }
    assert (
        _validate_checkpoint_identity(
            representation="opera_full",
            seed=42,
            metadata=metadata,
            plan=plan,
            checkpoint_path="/synthetic/full/seed_42/best.ckpt",
            legacy_full_checkpoint_template="/synthetic/full/seed_{seed}/best.ckpt",
        )
        == "/synthetic/legacy-dapt.ckpt"
    )


def test_legacy_full_rejects_mismatched_recorded_seed() -> None:
    plan = resolve_transfer_manifest()
    metadata = {
        "training_stage": "opera_contrastive_adaptation",
        "seed": 43,
        "outcome_set": sorted(plan["conditions"]["opera_full"]["training_outcomes"]),
        "source_checkpoint": "/synthetic/legacy-dapt.ckpt",
    }
    with pytest.raises(OutcomeTransferExtractionError, match="does not match requested"):
        _validate_checkpoint_identity(
            representation="opera_full",
            seed=42,
            metadata=metadata,
            plan=plan,
        )


def test_extractor_rejects_opera_checkpoint_as_dapt() -> None:
    plan = resolve_transfer_manifest()
    with pytest.raises(OutcomeTransferExtractionError, match="not 'dapt'"):
        _validate_checkpoint_identity(
            representation="dapt",
            seed=42,
            metadata={
                "condition": "opera_full",
                "training_stage": "opera_contrastive_adaptation",
                "source_checkpoint": "/synthetic/dapt.ckpt",
            },
            plan=plan,
        )
