import json

import pytest

from opera.modules.networks.outcome_scaling import resolve_outcome_reference_scales


def test_resolve_outcome_reference_scales_loads_diagnostic_artifact(tmp_path):
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps({"outcome_reference_scales": {"mortality": 0.75}}),
        encoding="utf-8",
    )

    result = resolve_outcome_reference_scales(
        {"outcome_scale_mode": "initial_kl", "outcome_reference_scale_file": str(path)}
    )

    assert result["outcome_reference_scales"] == {"mortality": 0.75}
    assert "outcome_reference_scale_file" not in result


def test_resolve_outcome_reference_scales_rejects_two_sources(tmp_path):
    path = tmp_path / "references.json"
    path.write_text(json.dumps({"mortality": 0.75}), encoding="utf-8")

    with pytest.raises(ValueError, match="either"):
        resolve_outcome_reference_scales(
            {
                "outcome_reference_scale_file": str(path),
                "outcome_reference_scales": {"mortality": 1.0},
            }
        )
