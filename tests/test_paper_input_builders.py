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
