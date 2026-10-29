import numpy as np
import pandas as pd

from opera.interpretability.attribution_patterns import (
    REFERENCE, RESIDUAL, episode_ids, explanation_table,
    outcome_similarity, patient_episodes, variance_decomposition)
from opera.interpretability.attribution_summaries import label_tokens


def test_episodes_join_events_within_the_gap_and_keep_background_apart():
    tokens = pd.DataFrame({
        "subject_id": [1] * 6,
        "code": ["BG_SEX//F", "DIAGNOSIS//DJ18", "LABTEST//NPU1//BIN_2", "MEDICATION//J01",
                 "LABTEST//NPU1//BIN_1", "DIAGNOSIS//DC83"],
        "days_before_index": [20000.0, 200.0, 199.5, 198.8, 30.0, 2.0],
    })
    assert episode_ids(tokens, gap_days=1.0).tolist() == [-1, 0, 0, 0, 1, 2]


def _population(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows, patients = [], []
    for subject in range(n):
        lab = rng.normal(0, 1.0)
        medication = rng.normal(0, 0.5)
        base = -2.0 + rng.normal(0, 0.3)
        rows += [{"subject_id": subject, "code": "LABTEST//NPU1//BIN_1",
                  "days_before_index": 10.0, "attribution": lab, "weight": 1.0},
                 {"subject_id": subject, "code": "MEDICATION//J01",
                  "days_before_index": 400.0, "attribution": medication, "weight": 1.0}]
        patients.append({"subject_id": subject, "prediction": base + lab + medication,
                         "baseline_prediction": base, "weight": 1.0})
    return label_tokens(pd.DataFrame(rows)), pd.DataFrame(patients)


def test_variance_shares_are_exact_and_recover_the_planted_split():
    tokens, patients = _population()
    table = variance_decomposition(tokens, patients, by="source", draws=200)
    shares = table.set_index("group").share
    assert abs(shares.sum() - 1) < 1e-9
    assert abs(shares[RESIDUAL]) < 1e-9
    # Planted variances 1 (labs), 0.25 (medication) and 0.09 (reference).
    assert abs(shares["Laboratory"] - 1 / 1.34) < 0.08
    assert abs(shares["Medication"] - 0.25 / 1.34) < 0.05
    assert abs(shares[REFERENCE] - 0.09 / 1.34) < 0.03
    lab = table.set_index("group").loc["Laboratory"]
    assert lab.share_lower < lab.share < lab.share_upper


def test_outcome_similarity_separates_shared_and_opposite_use_of_history():
    rng = np.random.default_rng(1)
    rows = []
    for subject in range(40):
        for episode in range(8):
            a = rng.normal()
            rows.append({"subject_id": subject, "code": "DIAGNOSIS//DJ18",
                         "days_before_index": 10.0 * episode, "weight": 1.0,
                         "same_a": a, "same_b": 2 * a, "opposite": -a,
                         "unrelated": rng.normal()})
    wide = pd.DataFrame(rows)
    table = outcome_similarity(wide, ["same_a", "same_b", "opposite", "unrelated"],
                               min_count=10).set_index(["column_a", "column_b"]).spearman
    assert table["same_a", "same_b"] > 0.99
    assert table["same_a", "opposite"] < -0.99
    assert abs(table["same_a", "unrelated"]) < 0.15


def test_patient_episodes_list_chronological_episodes_with_main_codes():
    tokens = pd.DataFrame({
        "subject_id": [1] * 4,
        "code": ["DIAGNOSIS//DJ181", "MEDICATION//J01CA04", "LABTEST//NPU1//BIN_1",
                 "BG_SEX//F"],
        "days_before_index": [60.0, 59.5, 3.0, 30000.0],
        "attribution": [0.02, 0.01, -0.005, 0.0],
    })
    episodes = patient_episodes(tokens, labels={"DIAGNOSIS//DJ18": "Pneumonia"})
    timed = episodes.loc[episodes.episode >= 0]
    assert timed.start_days.tolist() == [60.0, 3.0]
    assert timed.iloc[0].main_codes.startswith("Pneumonia")
    assert abs(timed.iloc[0].attribution - 0.03) < 1e-12


def test_input_restriction_comparison_is_paired_and_shares_resamples():
    from opera.run.compare_input_restrictions import compare

    rng = np.random.default_rng(3)
    rows = []
    for subject in range(300):
        for outcome in ("a", "b"):
            risk = rng.random()
            event = int(rng.random() < risk)
            rows.append({"subject_id": subject, "cohort": "X", "outcome": outcome,
                         "time_days": 100.0 if event else 400.0, "event": event,
                         "cif_365d": risk})
    full = pd.DataFrame(rows)
    scrambled = full.assign(cif_365d=rng.permutation(full.cif_365d.to_numpy()))
    table, summary = compare(full, {"Same": full, "Scrambled": scrambled}, draws=50,
                             domains={"a": "D1", "b": "D2"})
    same = table.loc[table.restriction == "Same"]
    assert (same.delta.abs() < 1e-12).all() and (same.upper.abs() < 1e-12).all()
    worse = summary.set_index(["restriction", "group"]).loc[("Scrambled", "All outcomes")]
    assert worse.mean_delta < -0.15 and worse.upper < 0
    assert set(summary.group) == {"All outcomes", "D1", "D2"}


def test_every_attribution_number_in_the_manuscript_is_produced_and_formatted(tmp_path):
    import json
    import re
    from pathlib import Path

    from opera.run.build_paper_numbers import attribution_numbers, restriction_numbers

    root = Path(__file__).resolve().parents[1]
    from opera.run.check_figure_inputs import manuscript_text

    text = manuscript_text(root / "paper/opera_manuscript.tex")
    keys = set(re.findall(r"\\result\{((?:ig|restrict)_[a-z0-9_]+)\}", text))
    code = "".join((root / p).read_text() for p in (
        "opera/interpretability/attribution_summaries.py",
        "opera/run/summarise_attributions.py", "opera/run/build_paper_numbers.py"))
    sanity_checks = {"random_head", "random_model", "baseline_source_mean"}
    for key in keys:
        produced = f'"{key}"' in code or f"{key}=" in code or (
            key.startswith("ig_rho_") and key[len("ig_rho_"):] in sanity_checks)
        assert produced, f"{key} is cited in the manuscript but never produced"
    numbers = {k: 0.42 for k in keys if k.startswith("ig_")}
    numbers.update(ig_patients=1500, ig_outcomes=69, ig_expect_total=15,
                       ig_similarity_within=.42)
    path = tmp_path / "attribution_numbers.json"
    path.write_text(json.dumps({"numbers": numbers}))
    restrictions = tmp_path / "input_restriction_summary.csv"
    pd.DataFrame([{"restriction": r, "group": "All outcomes", "mean_delta": -0.02,
                   "lower": -0.03, "upper": -0.01}
                  for r in ("Only the timing of care", "Without laboratory results",
                            "Without medication", "Without diagnoses",
                            "Without region and specialty",
                            "Only the last year")]).to_csv(restrictions)
    sources = {}
    formatted = attribution_numbers(path, sources) | restriction_numbers(restrictions, sources)
    assert keys <= set(formatted)
    assert formatted["ig_var_reference"] == "42\\%"
    assert formatted["ig_similarity_within"] == "0.42"
    assert formatted["ig_patients"] == "1,500"
    assert formatted["restrict_timeline"].startswith("$-$0.020 (95\\% CI")


def test_concept_labels_keep_acronyms_direction_and_readable_codes():
    from opera.interpretability.attribution_patterns import concept_label

    labels = {"LABTEST//NPU19658//BIN": "LDH", "LABTEST//NPU19673//BIN": "Albumin",
              "DIAGNOSIS//DI50": "Heart failure"}
    assert concept_label("LABTEST//NPU19658//high", labels) == "LDH, upper bins"
    assert concept_label("LABTEST//NPU19673//low", labels) == "Albumin, lower bins"
    assert concept_label("LABTEST//NPU19673//mid", labels) == "Albumin, middle bins"
    assert concept_label("DIAGNOSIS//DI50", labels) == "Heart failure"
    assert concept_label("PROCEDURE//KTSF", labels) == "Procedure KTSF"
    assert concept_label("CONTACT_TYPE//ACUTE", labels) == "Contact type acute"


def _patient_frame():
    columns = {"a": "attr|risk|a|365", "b": "attr|risk|b|365"}
    tokens = pd.DataFrame({
        "subject_id": [7] * 5,
        "code": ["DIAGNOSIS//DJ189", "MEDICATION//J01CA04", "LABTEST//NPU02902//BIN_1",
                 "DIAGNOSIS//DI50", "BG_SEX//F"],
        "days_before_index": [60.0, 59.6, 4.0, 900.0, 30000.0],
        columns["a"]: [0.05, 0.02, 0.0, 0.01, 0.0],
        columns["b"]: [0.0, 0.0, 0.04, -0.01, 0.0],
    })
    patients = pd.DataFrame([
        {"subject_id": 7, "column": columns["a"], "prediction": 0.18, "baseline_prediction": 0.10},
        {"subject_id": 7, "column": columns["b"], "prediction": 0.12, "baseline_prediction": 0.09}])
    labels = {"DIAGNOSIS//DJ18": "Pneumonia", "LABTEST//NPU02902//BIN": "Neutrophils",
              "MEDICATION//J01C": "Penicillins",
              "DIAGNOSIS//DI50": "Heart failure"}
    return tokens, patients, columns, labels


def test_explanation_table_lists_what_raises_and_lowers_each_risk():
    tokens, patients, columns, labels = _patient_frame()
    table = explanation_table(tokens, patients, columns=columns, labels=labels, n=2)
    raises_a = table.loc[(table.outcome == "a") & (table.direction == "raises")]
    assert raises_a.description.iloc[0].startswith("Pneumonia") and raises_a["rank"].iloc[0] == 1
    lowers_b = table.loc[(table.outcome == "b") & (table.direction == "lowers")]
    assert lowers_b.description.tolist() == ["Heart failure"]
    assert lowers_b.contribution_pp.iloc[0] < 0
    assert set(table.columns) >= {"risk", "reference_risk", "when", "contribution_pp"}
