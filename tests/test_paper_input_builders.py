"""Builders for patient metadata, embeddings, clinical scores and subgroups."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from opera.run.build_paper_embedding_tables import (
    calibration_map_inputs,
    embedding_comparison,
    project,
)
from opera.run.build_paper_patient_metadata import build_patient_metadata, history_summary
from opera.run.build_paper_plot_tables import SUBGROUP_FIGURES, subgroup_table
from opera.run.render_clinical_score_panels import high_score_threshold, score_patients
from opera.visualization import paper_figures as figures


def test_history_counts_only_clinical_tokens_strictly_before_index():
    vocabulary = {"[CLS]": 0, "BG_GENDER_F": 1, "D/C83": 2, "M/L01": 3, "LAB/HB": 4}
    subjects = [
        {"subject_id": 1, "code": [1, 2, 3, 4, 2], "abspos": [0, 10, 20, 30, 40]},
        {"subject_id": 2, "code": [1, 4], "abspos": [0, 5]},
        {"subject_id": 3, "code": [2], "abspos": [0]},  # not in the population
    ]
    history = history_summary(subjects, {1: 30.0, 2: 1.0}, vocabulary,
                              national_vocabulary={"D/C83": 0, "M/L01": 1})
    rows = history.set_index("subject_id")
    # Patient 1: tokens at 10 and 20 precede the index at 30; the event at 30 does not.
    assert rows.loc[1, "history_event_count"] == 2
    assert rows.loc[1, "national_vocabulary_coverage"] == 1.0
    assert rows.loc[2, "history_event_count"] == 0
    assert np.isnan(rows.loc[2, "national_vocabulary_coverage"])
    assert 3 not in rows.index

    population = pd.DataFrame({"subject_id": [1, 2], "index_date": ["2023-02-01"] * 2,
                               "age_at_index": [61, 74], "sex": ["F", "M"],
                               "cohort_fine": ["DLBCL", "FL"],
                               "index": ["IPI", None], "IPI": ["high", None]})
    metadata = build_patient_metadata(population, history)
    assert metadata.index_year.tolist() == [2023, 2023]
    assert metadata.loc[0, "IPI"] == "high"


def test_projection_is_fitted_on_tuning_patients_only():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"subject_id": range(60),
                          "split": ["tuning"] * 40 + ["held_out"] * 20,
                          "embedding": list(rng.normal(size=(60, 8)))})
    coords = project(frame, method="pca")
    shifted = frame.copy()
    shifted["embedding"] = list(np.stack(frame.embedding)
                                + np.where((frame.split == "held_out").to_numpy()[:, None],
                                           5.0, 0.0))
    # Changing held-out patients cannot move the map they are projected onto.
    moved = project(shifted, method="pca")
    assert set(coords.subject_id) == set(range(40, 60))
    reference = project(pd.concat([frame.loc[frame.split == "tuning"],
                                   shifted.loc[shifted.split == "held_out"]]),
                        method="pca")
    np.testing.assert_allclose(moved[["x", "y"]], reference[["x", "y"]])


def test_embedding_and_calibration_map_tables_render(tmp_path):
    rng = np.random.default_rng(1)
    ids = np.arange(80)
    splits = np.where(ids < 50, "tuning", "held_out")
    stages = {}
    for label in ("Local fine-tuned", "National fine-tuned"):
        path = tmp_path / f"{label}.npz"
        np.savez(path, subject_ids=ids, embeddings=rng.normal(size=(80, 6)), splits=splits)
        stages[label] = [str(path)]
    metadata = pd.DataFrame({"subject_id": ids, "cohort_grouped": "DLBCL_like",
                             "IPI": rng.choice(["low", "medium", "high", None], 80),
                             "age_at_index": rng.uniform(30, 90, 80),
                             "history_event_count": rng.integers(1, 5000, 80)})
    comparison, coordinates = embedding_comparison(
        stages, metadata, method="pca",
        group_map={"low": "Low", "medium": "Intermediate", "high": "High"})
    assert comparison.patient_id.nunique() == 30
    assert set(comparison.disease) == {"DLBCL-like"}
    fig = figures.figure_embedding_comparison(comparison)
    plt.close(fig)

    predictions = pd.DataFrame([
        {"subject_id": s, "outcome": o, "time_days": rng.uniform(10, 800),
         "event": rng.integers(0, 3), "cif_365d": rng.uniform(0, 0.6)}
        for s in ids[50:] for o in ("treatment_failure", "sepsis") if not (s == 60 and o == "sepsis")])
    coords, payload = calibration_map_inputs(coordinates["Local fine-tuned"], predictions,
                                             ["treatment_failure", "sepsis"])
    assert len(coords) == 29  # patient 60 lacks one outcome and is left out
    assert all(len(values[0]) == 29 for values in payload.values())
    fig = figures.figure_calibration_maps(coords, payload, grid_size=30)
    plt.close(fig)


def three_group_spec():
    return dict(model="J_L", seed=42, outcome="failure", horizon_days=730,
                cohorts=["DLBCL"], index_column="index", index_values=["IPI"],
                group_column="IPI", group_map={"low": "Low", "medium": "Intermediate",
                                               "high": "High"},
                group_order=["Low", "Intermediate", "High"], score_high_group="High",
                age_column="age_at_index", score_timing="registered at diagnosis")


def test_clinical_scores_read_index_and_category_columns():
    predictions = pd.DataFrame(dict(
        subject_id=["a", "b", "c", "d"], model="J_L", seed=42, split="held_out",
        outcome="failure", horizon_days=730, cohort="DLBCL",
        probability=[0.05, 0.2, 0.7, 0.4], time_days=[800, 300, 100, 900],
        event=[0, 1, 1, 0]))
    metadata = pd.DataFrame(dict(subject_id=["a", "b", "c", "d"],
                                 index=["IPI", "IPI", "IPI", "CLL-IPI"],
                                 IPI=["low", "Medium ", "high", "high"],
                                 age_at_index=[55, 66, 77, 70]))
    spec = three_group_spec()
    frame, audit = score_patients(predictions, metadata, spec, "held_out")
    assert frame.subject_id.tolist() == ["a", "b", "c"]  # d was scored with another index
    assert frame.ipi_group.tolist() == ["Low", "Intermediate", "High"]
    assert frame.ipi_score.tolist() == [1, 2, 3]
    assert audit["unavailable_or_late_n"] == 1
    assert high_score_threshold(spec) == 3
    metadata.loc[0, "IPI"] = "very low"
    with pytest.raises(ValueError, match="Unmapped"):
        score_patients(predictions, metadata, spec, "held_out")


def test_three_group_clinical_figures_render():
    rng = np.random.default_rng(3)
    n = 600
    risk = rng.uniform(0, 1, n)
    group = np.digitize(risk + rng.normal(0, 0.2, n), [0.35, 0.65])
    patients = pd.DataFrame({
        "time_days": rng.uniform(30, 1500, n), "event": rng.integers(0, 2, n),
        "model_risk": risk, "age": rng.uniform(30, 90, n), "ipi_score": group + 1,
        "ipi_group": np.array(["Low", "Intermediate", "High"])[group]})
    common = dict(outcome_label="Treatment failure", thresholds=(0.1, 0.4),
                  ipi_label="IPI", ipi_order=("Low", "Intermediate", "High"))
    for fig in (figures.figure_risk_groups(patients, **common),
                figures.figure_clinical_score(patients, ipi_high=3, **common),
                figures.figure_clinical_score_age(patients, **common)):
        plt.close(fig)


def test_subgroup_tables_have_readable_labels_and_render():
    panels = pd.DataFrame([
        {"split": "held_out", "seed": 42, "horizon_days": 365, "contrast": contrast,
         "covariate": covariate, "subgroup": subgroup, "metric": "brier_gain",
         "gain": 0.004, "lower": 0.001, "upper": 0.007}
        for contrast in ("P5", "T2", "R1")
        for covariate, subgroup in (("age_at_index", "(-0.001, 50.0]"),
                                    ("age_at_index", "(50.0, 65.0]"),
                                    ("sex", "F"), ("IPI", "missing"),
                                    ("history_event_count", "(99.0, 999.0]"))])
    table, styles, order = subgroup_table(panels, None,
                                          SUBGROUP_FIGURES["patient_subgroups"])
    assert set(table.contrast) == {"P5", "T2"}
    assert {"≤ 50", "51–65", "Female", "Not recorded", "100–999"} <= set(table.subgroup)
    assert list(order) == ["history_event_count", "age_at_index", "sex", "IPI"]
    fig = figures.figure_subgroup_effects(table, contrast_styles=styles,
                                          covariate_order=order, xlabel="Brier")
    plt.close(fig)


def test_identity_evidence_date_and_assignment_source_are_carried():
    population = pd.DataFrame({"subject_id": [1, 2], "index_date": ["2023-02-01"] * 2,
                               "age_at_index": [61, 74], "sex": ["F", "M"],
                               "cohort_fine": ["DLBCL", "FL"],
                               "cohort_source": ["pass1", "pass2_fwd_dx"],
                               "dx_date": ["2023-01-20", "2023-03-15"]})
    history = pd.DataFrame({"subject_id": [1, 2], "history_event_count": [10, 20]})
    metadata = build_patient_metadata(population, history, identity_date_column="dx_date")
    late = metadata.cohort_identity_evidence_date > metadata.index_date
    assert late.tolist() == [False, True]
    assert metadata.cohort_source.tolist() == ["pass1", "pass2_fwd_dx"]
    assert metadata.cohort_identity_post_index.tolist() == [False, True]


def test_model_cutoffs_are_chosen_on_tuning_patients():
    from opera.run.render_clinical_score_panels import select_model_cutoffs
    from opera.visualization.paper_figures import _known_status

    rng = np.random.default_rng(5)
    n = 1500
    risk = rng.uniform(0, 1, n)
    group = np.digitize(risk + rng.normal(0, 0.25, n), [0.35, 0.7])
    event_time = np.where(rng.uniform(size=n) < risk, rng.uniform(1, 700, n), 2000.0)
    tuning = pd.DataFrame({
        "model_risk": risk, "ipi_group": np.array(["Low", "Intermediate", "High"])[group],
        "time_days": np.minimum(event_time, 1000.0),
        "event": (event_time <= 1000).astype(int)})
    spec = dict(three_group_spec(), horizon_days=730)
    size = select_model_cutoffs(tuning, {**spec, "cutoff_rule": "size_matched"})
    shares = tuning.ipi_group.value_counts(normalize=True)
    assert abs((risk < size[0]).mean() - shares["Low"]) < 0.01
    low = select_model_cutoffs(tuning, {**spec, "cutoff_rule": "risk_matched_low"})
    known, labels = _known_status(tuning, 730)
    assert labels[(known.model_risk < low[0]).to_numpy()].mean() <= labels[
        (known.ipi_group == "Low").to_numpy()].mean()
    assert low == sorted(low)
    # High cutoff: the model's high-risk group has the index's specificity.
    controls = known.loc[labels.to_numpy() == 0]
    index_specificity = (controls.ipi_group != "High").mean()
    model_specificity = (controls.model_risk < low[-1]).mean()
    assert abs(model_specificity - index_specificity) < 0.01

    from opera.run.render_clinical_score_panels import stratification_summary
    rows = pd.DataFrame(stratification_summary(tuning, {**spec, "id": "ipi"}, low))
    assert set(rows.system) == {"index", "model"}
    assert {"top 5%", "top 10%"} <= set(rows.group)
    top = rows.set_index("group").loc["top 5%"]
    assert top.observed_risk > rows.query("system == 'index' and group == 'High'").observed_risk.iloc[0]


def test_every_recorded_index_becomes_a_score_entry():
    from opera.run.render_clinical_score_panels import expand_index_scores

    metadata = pd.DataFrame({
        "subject_id": range(400),
        "cohort_fine": ["DLBCL"] * 200 + ["CLL"] * 150 + ["FL"] * 50,
        "index": ["IPI"] * 120 + ["aaIPI"] * 80 + ["CLL-IPI"] * 150 + ["FLIPI"] * 10 + [None] * 40,
        "IPI": ["low", "medium", "high", "medium"] * 100})
    template = dict(three_group_spec(), outcome="failure", lock_provenance="x",
                    applicability_review="y")
    config = {"all_indices": {"minimum_patients": 50, "template": template, "pooled": False,
                              "manuscript": {"index": "IPI", "cohort": "DLBCL"}},
              "scores": [{"id": "explicit", "cohorts": ["DLBCL"], "index_values": ["aaIPI"]}]}
    specs = expand_index_scores(config, metadata)
    generated = {s["id"]: s for s in specs if s["id"] != "explicit"}
    assert set(generated) == {"ipi_dlbcl", "cll_ipi_cll"}  # aaIPI explicit, FLIPI too few
    assert generated["ipi_dlbcl"]["manuscript_figures"]
    assert not generated["cll_ipi_cll"]["manuscript_figures"]
    assert generated["cll_ipi_cll"]["index_values"] == ["CLL-IPI"]


def test_registry_profile_keeps_only_the_rkkp_predictors(tmp_path):
    from opera.functional.tabular_features import prepare_feature_matrix
    from opera.run.paper_tabular_queue import job_models

    source = tmp_path / "export.parquet"
    pd.DataFrame({"subject_id": [1, 2, 3], "pred_lab_mean": [1.0, 2.0, 3.0],
                  "pred_rkkp_ps": [0, 2, 1], "pred_adverse_event_x": [0, 1, 0]}
                 ).to_parquet(source)
    population = tmp_path / "population.csv"
    pd.DataFrame({"subject_id": [1, 2, 3], "index_date": ["2020-01-01"] * 3,
                  "birth_date": ["1950-01-01", "1960-01-01", "1970-01-01"], "sex": ["F", "M", "F"]}
                 ).to_csv(population, index=False)
    matched, _ = prepare_feature_matrix(source, profile="sequence_matched",
                                        population_path=population, output_dir=tmp_path)
    augmented, _ = prepare_feature_matrix(source, profile="registry_augmented",
                                          population_path=population, output_dir=tmp_path)
    matched, augmented = pd.read_parquet(matched), pd.read_parquet(augmented)
    assert "pred_rkkp_ps" not in matched and "pred_rkkp_ps" in augmented
    assert "pred_adverse_event_x" not in matched and "pred_adverse_event_x" in augmented
    assert job_models({}, ("xgboost",)) == ("xgboost",)

def test_outcome_and_cohort_columns_never_become_predictors(tmp_path):
    from opera.functional.tabular_features import prepare_feature_matrix

    source = tmp_path / "all.parquet"
    pd.DataFrame({"subject_id": [1, 2, 3], "pred_lab_mean": [1.0, 2.0, 3.0],
                  "outc_death_within_365_days": [0, 1, 0],
                  "cohort_fine": ["DLBCL", "CLL", "MM"], "cohort_grouped": ["A", "B", "C"]}
                 ).to_parquet(source)
    population = tmp_path / "population.csv"
    pd.DataFrame({"subject_id": [1, 2, 3], "age_at_index": [60, 70, 80], "sex": ["F", "M", "F"]}
                 ).to_csv(population, index=False)
    path, manifest = prepare_feature_matrix(source, profile="sequence_matched",
                                            population_path=population, output_dir=tmp_path)
    assert set(pd.read_parquet(path)) == {"subject_id", "pred_lab_mean", "age_at_index", "matched_sex"}
    assert manifest["removed_outcome_or_cohort_columns"] == [
        "cohort_fine", "cohort_grouped", "outc_death_within_365_days"]


def test_an_index_recorded_in_several_subtypes_is_also_compared_across_them():
    from opera.run.render_clinical_score_panels import expand_index_scores

    metadata = pd.DataFrame({
        "subject_id": range(300),
        "cohort_fine": ["DLBCL"] * 200 + ["PTCL"] * 70 + ["BL"] * 30,
        "index": "IPI", "IPI": ["low", "medium", "high"] * 100})
    template = dict(three_group_spec(), outcome="failure")
    specs = {s["id"]: s for s in expand_index_scores(
        {"all_indices": {"minimum_patients": 50, "template": template}}, metadata)}
    assert set(specs) == {"ipi_dlbcl", "ipi_ptcl", "ipi_all_subtypes"}
    assert specs["ipi_all_subtypes"]["cohorts"] == ["BL", "DLBCL", "PTCL"]  # BL too small alone


def test_ipi_and_nccn_ipi_from_registry_components():
    from opera.run.compute_prognostic_scores import compute_scores

    registry = pd.DataFrame({
        "pid": [1, 2, 3],
        "alder": [80, 50, 65],
        "ldh_ratio_col": [3.5, 0.8, 1.2],
        "ann_arbor": ["IVB", "1", None],
        "ecog": [2, 0, 1],
        "n_extranodal": [2, 0, 1],
        "en_site": ["ja", "nej", "nej"]})
    scores = compute_scores(registry, {
        "subject_id": "pid", "age": "alder", "ldh_ratio": "ldh_ratio_col",
        "stage": "ann_arbor", "ps": "ecog", "extranodal_sites": "n_extranodal",
        "nccn_extranodal": "en_site"}).set_index("subject_id")
    # Patient 1: every IPI factor; NCCN 3 (age) + 2 (LDH) + 1 + 1 + 1 = 8.
    assert scores.loc[1, "ipi_computed_score"] == 5
    assert scores.loc[1, "nccn_ipi_score"] == 8 and scores.loc[1, "nccn_ipi_group"] == "high"
    # Patient 2: NCCN 1 point (age 41-60) -> low; IPI 0 -> low.
    assert scores.loc[2, "nccn_ipi_score"] == 1 and scores.loc[2, "nccn_ipi_group"] == "low"
    assert scores.loc[2, "ipi_computed_group"] == "low"
    # Missing stage: no score rather than a partial one.
    assert pd.isna(scores.loc[3, "nccn_ipi_score"]) and pd.isna(scores.loc[3, "ipi_computed_group"])


def test_explicit_score_over_all_scored_subtypes():
    from opera.run.render_clinical_score_panels import expand_index_scores

    metadata = pd.DataFrame({"subject_id": range(4), "cohort_fine": ["DLBCL", "PTCL", "CLL", "BL"],
                             "ipi_computed_group": ["low", "high", None, "low"]})
    (spec,) = expand_index_scores({"scores": [{"id": "x", "cohorts": "all",
                                               "group_column": "ipi_computed_group"}]}, metadata)
    assert spec["cohorts"] == ["BL", "DLBCL", "PTCL"]


def test_nccn_ipi_matches_the_lyfo_definition():
    """Age-specific LDH limit (205, 255 from 70) and site-based extranodal flag."""
    from opera.run.compute_prognostic_scores import NCCN_EXTRANODAL_SITES, compute_scores

    sites = {s: ["No", "No", None] for s in NCCN_EXTRANODAL_SITES}
    sites["lung_diagnosis"] = ["Yes", "No", None]
    registry = pd.DataFrame({"subject_id": [1, 2, 3], "age": [72, 55, 50],
                             "ldh": [600, 210, 100], "stage": [3, 2, 1], "ps": [1, 2, 0],
                             **sites})
    scores = compute_scores(registry, {"nccn_extranodal": NCCN_EXTRANODAL_SITES}
                            ).set_index("subject_id")
    # 72 y: age 2; LDH 600/255 = 2.4 -> 1; stage III 1; lung 1; PS 1 -> 0 => 5.
    assert scores.loc[1, "nccn_ipi_score"] == 5
    # 55 y: age 1; LDH 210/205 > 1 -> 1; stage II 0; no sites 0; PS 2 -> 1 => 3.
    assert scores.loc[2, "nccn_ipi_score"] == 3
    # All sites missing: no score.
    assert pd.isna(scores.loc[3, "nccn_ipi_score"])


def test_history_summary_records_the_predominant_region():
    from opera.run.build_paper_patient_metadata import history_summary

    vocabulary = {"[CLS]": 0, "BG_x": 1, "DX//A": 2, "REGION//Hovedstaden": 3,
                  "REGION//Sjaelland": 4}
    subjects = [
        {"subject_id": 1, "code": [0, 2, 3, 3, 4, 2], "abspos": [0, 1, 2, 3, 4, 100]},
        {"subject_id": 2, "code": [0, 3, 4], "abspos": [0, 1, 2]},
    ]
    table = history_summary(subjects, {1: 50, 2: 50}, vocabulary).set_index("subject_id")
    assert table.loc[1, "predominant_region"] == "Hovedstaden"
    assert pd.isna(table.loc[2, "predominant_region"])  # tie
    assert table.loc[1, "history_event_count"] == 4  # DX + 3 region tokens, post-index DX dropped
