import pytest
import torch
from opera.functional.endpoint_groups import endpoint_groups_v1, endpoint_group_score, validate_endpoint_groups
from opera.modules.networks.competing_risk import PiecewiseExponentialCompetingRiskLoss
from opera.run.generate_sweep_configs import build_endpoint_weighting_pilot, load_registry, DEFAULT_REGISTRY


def test_grade_groups_do_not_merge_distinct_or_composite_endpoints():
    names = ["anemia_g2plus", "anemia_g3plus", "severe_cytopenia", "treatment_failure"]
    groups = endpoint_groups_v1(names)
    assert groups["anemia"] == names[:2]
    assert len(groups) == 3
    validate_endpoint_groups(groups, names)
    with pytest.raises(ValueError):
        validate_endpoint_groups({"bad": names + names[:1]}, names)


def test_selection_hierarchy_equalizes_groups_not_grade_counts():
    score, coverage = endpoint_group_score(
        {"g2": [.6, .8], "g3": [.9], "other": [.5]},
        {"grades": ["g2", "g3"], "other": ["other"], "unsupported": ["absent"]},
    )
    assert score == pytest.approx(.65)
    assert coverage == 2


def test_pilot_loss_equalizes_horizons_then_grades_and_groups():
    names = ["a_g2plus", "a_g3plus", "b"]
    objective = PiecewiseExponentialCompetingRiskLoss(
        names, [30., 90.], objective="horizon_bce", weighter="uniform", no_competing_outcomes=[],
        cross_outcome_config={"aggregation": "endpoint_group_macro", "endpoint_groups": endpoint_groups_v1(names)},
    )
    hazards = torch.full((3, 3, 2, 3), -2., requires_grad=True)
    survival = {name: {"times": torch.tensor([10., 50., 100.]),
                       "events": torch.tensor([1, 0, 0])} for name in names}
    total, _, terms = objective(hazards, survival, return_per_outcome=True)
    expected = ((terms[names[0]]["loss"] + terms[names[1]]["loss"]) / 2 + terms["b"]["loss"]) / 2
    assert torch.allclose(total, expected)
    targets, known = objective._horizon_targets(survival["b"]["times"], survival["b"]["events"])
    cells = objective._horizon_bce_cells(objective._horizon_cif(hazards[:, 2], "b"), targets)
    expected_b = torch.stack([cells[known[:, h], h].mean() for h in range(2)]).mean()
    assert torch.allclose(terms["b"]["loss"], expected_b)
    total.backward()
    assert torch.isfinite(hazards.grad).all()


def test_matched_configs_share_selection_and_all_nonweighting_settings():
    registry = load_registry(DEFAULT_REGISTRY)
    new = build_endpoint_weighting_pilot(registry)
    reference = build_endpoint_weighting_pilot(registry, revised=False)
    assert new["training"] == reference["training"]
    assert new["model"] == reference["model"]
    assert new["competing_risk"] == reference["competing_risk"]
    assert new["cross_outcome"]["endpoint_groups"] == reference["cross_outcome"]["endpoint_groups"]
    assert len(new["outcomes"]) == 87
    assert len(new["cross_outcome"]["endpoint_groups"]) < 87


def test_validation_logs_group_score_and_label_only_inventory(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from opera.modules.lightningmodules.OperaContrastiveModule import OperaContrastiveModule
    logs, exports = {}, []
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "write_text", lambda _, text: exports.append(json.loads(text)))
    store = {"times": [torch.tensor([10.] * 25 + [100.] * 25)],
             "events": [torch.tensor([1] * 25 + [0] * 25)],
             "cif": [torch.tensor([[.8]] * 25 + [[.2]] * 25)]}
    module = SimpleNamespace(
        endpoint_groups={"a": ["a_g2plus", "a_g3plus"]},
        trainer=SimpleNamespace(sanity_checking=False, default_root_dir="unused"),
        _val_cr_predictions={"a_g2plus": store, "a_g3plus": store},
        evaluation_horizons_days=[30.], outcome_family={},
        selection_min_cases=20, selection_min_controls=20,
        checkpoint_selection_metric='val/cr_cif_auc_endpoint_group_macro',
        log=lambda key, value, **kwargs: logs.update({key: value}),
    )
    OperaContrastiveModule._log_competing_risk_validation(module)
    assert logs["val/cr_cif_auc_endpoint_group_macro"] == pytest.approx(1.)
    assert logs["val/cr_cif_selection_cells"] == 2
    assert len(exports[0]) == 2
    assert exports[0][0]["n_cases"] == 25
    assert not module._val_cr_predictions
