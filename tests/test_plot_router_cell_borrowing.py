import json
from pathlib import Path

import pandas as pd
import pytest

from opera.run.plot_router_cell_borrowing import (
    compare_cells, parse_cell, plot_cells, route_similarity, summarize_cell,
    verify_route_provenance,
)
from opera.visualization.outcome_names import outcome_display_name


def _config():
    return {
        "competing_risk": {"fine_names": ["CLL", "SLL"]},
        "outcomes": {"sepsis": {}, "aki": {}},
    }


def _frame(n=25):
    return pd.DataFrame([{
        "subject_id": i, "split": "tuning", "outcome": "sepsis", "fine_index": 0,
        "shared_weight": .4,
        "disease_weights": json.dumps({"disease:CLL": .2, "disease:SLL": .1}),
        "outcome_weights": json.dumps({"outcome:aki": .1, "outcome:sepsis": .2}),
    } for i in range(n)])


def test_cell_budget_separates_home_and_off_home():
    report = summarize_cell(_frame(), _config(), disease="CLL", outcome="sepsis", n_boot=20)
    assert report["n_patients"] == 25
    assert report["budget"] == pytest.approx({
        "shared": .4, "own_disease": .2, "other_diseases": .1,
        "own_outcome": .2, "other_outcomes": .1,
    })
    assert set(report["sources"]["source"]) == {"SLL", "aki"}
    assert report["sources"]["mean_weight"].tolist() == pytest.approx([.1, .1])
    assert report["sources"]["uniform_bank_enrichment"].tolist() == pytest.approx([1, 1])


def test_outcome_display_names_preserve_definition_caveats():
    assert outcome_display_name("hospitalisation") == "Hospital admission"
    assert outcome_display_name("alp_elevation_legacy_tier3plus") == "ALP elevation (legacy tier 3+)"
    assert outcome_display_name("prescription_antimicrobial_use_proxy") == "Antimicrobial-prescription proxy"
    assert outcome_display_name("sepsis") == "Sepsis"


def test_source_enrichment_accounts_for_unequal_bank_sizes():
    config = {"competing_risk": {"fine_names": ["CLL", "SLL"]},
              "outcomes": {name: {} for name in ("sepsis", "aki", "bleeding", "fracture")}}
    frame = _frame()
    frame["shared_weight"] = .2
    frame["disease_weights"] = json.dumps({"disease:CLL": .1, "disease:SLL": .1})
    frame["outcome_weights"] = json.dumps({"outcome:sepsis": .2, "outcome:aki": .2,
                                            "outcome:bleeding": .1, "outcome:fracture": .1})
    report = summarize_cell(frame, config, disease="CLL", outcome="sepsis", n_boot=10)
    by_source = report["sources"].set_index("source")
    assert report["budget"]["other_outcomes"] == pytest.approx(.4)
    assert by_source.loc["aki", "mean_weight"] == pytest.approx(.2)
    assert by_source.loc["aki", "off_home_bank_share"] == pytest.approx(.5)
    assert by_source.loc["aki", "uniform_bank_enrichment"] == pytest.approx(1.5)
    assert by_source.loc["SLL", "uniform_bank_enrichment"] == pytest.approx(1)


def test_flat_mean_can_hide_strong_patient_outcome_selection():
    config = {"competing_risk": {"fine_names": ["CLL", "SLL"]},
              "outcomes": {name: {} for name in ("sepsis", "aki", "bleeding")}}
    frame = _frame(n=24)
    frame["shared_weight"] = .2
    frame["disease_weights"] = json.dumps({"disease:CLL": .1, "disease:SLL": .1})
    frame["outcome_weights"] = [json.dumps({"outcome:sepsis": .2,
                                             "outcome:aki": .38 if i % 2 else .02,
                                             "outcome:bleeding": .02 if i % 2 else .38})
                                for i in range(len(frame))]
    report = summarize_cell(frame, config, disease="CLL", outcome="sepsis", n_boot=10)
    outcome = report["selectivity"].set_index("kind").loc["outcome"]
    assert outcome["mean_route_peak_enrichment"] == pytest.approx(1)
    assert outcome["median_patient_peak_enrichment"] == pytest.approx(1.9)


def test_cell_validation_rejects_wrong_or_duplicate_rows():
    assert parse_cell("CLL:sepsis", _config()) == ("CLL", "sepsis", 0)
    with pytest.raises(ValueError, match="absent"):
        parse_cell("CLL:unknown", _config())
    frame = _frame()
    frame.loc[0, "split"] = "held_out"
    with pytest.raises(ValueError, match="tuning-only"):
        summarize_cell(frame, _config(), disease="CLL", outcome="sepsis")
    frame = _frame()
    frame.loc[1, "subject_id"] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_cell(frame, _config(), disease="CLL", outcome="sepsis")


def test_cell_plot_writes_bitmap_and_pdf(tmp_path: Path):
    report = summarize_cell(_frame(), _config(), disease="CLL", outcome="sepsis", n_boot=10)
    plot_cells([report], tmp_path, top_k=2)
    assert (tmp_path / "router_cell_borrowing.png").stat().st_size > 0
    assert (tmp_path / "router_cell_borrowing.pdf").stat().st_size > 0


def test_contrast_excludes_home_comparators_and_detects_flat_routing():
    first = summarize_cell(_frame(), _config(), disease="CLL", outcome="sepsis", n_boot=10)
    second_frame = _frame().assign(outcome="aki", fine_index=1)
    second_frame["disease_weights"] = json.dumps({"disease:CLL": .1, "disease:SLL": .2})
    second_frame["outcome_weights"] = json.dumps({"outcome:aki": .2, "outcome:sepsis": .1})
    second = summarize_cell(second_frame, _config(), disease="SLL", outcome="aki", n_boot=10)
    assert compare_cells([first, second]).empty  # every source is home in one cell
    assert route_similarity([first, second]).empty  # no common off-home sources
    third = summarize_cell(_frame(), _config(), disease="CLL", outcome="sepsis", n_boot=10)
    comparison = compare_cells([first, third])
    assert comparison["excess_weight"].abs().max() == pytest.approx(0)
    similarity = route_similarity([first, third])
    assert (similarity["off_home_total_variation"] == 0).all()


def test_route_export_must_match_saved_run_and_checkpoint(tmp_path: Path):
    run = tmp_path / "run"
    hydra = run / "hydra"
    export = tmp_path / "export"
    hydra.mkdir(parents=True)
    export.mkdir()
    checkpoint = run / "best.ckpt"
    checkpoint.touch()
    config = hydra / "config.yaml"
    config.touch()
    routes = export / "source_route_weights.parquet"
    routes.touch()
    summary = {"split": "tuning", "checkpoint": str(checkpoint),
               "route_removal_diagnostic": []}
    (export / "direct_cr_summary.json").write_text(json.dumps(summary))
    assert verify_route_provenance(routes, config, checkpoint) == summary
    summary["split"] = "held_out"
    (export / "direct_cr_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="not tuning-only"):
        verify_route_provenance(routes, config, checkpoint)
