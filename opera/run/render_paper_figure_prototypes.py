"""Render every manuscript figure from synthetic inputs as a design proof.

The synthetic tables have the same columns as the analysis outputs, so the
same figure functions render the real figures from locked results. Nothing
here is evidence of model performance; every artifact is watermarked.

    python -m opera.run.render_paper_figure_prototypes --output-dir paper/render/prototypes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.visualization import paper_figures as figures
from opera.visualization.outcome_names import outcome_display_name
from opera.visualization.style import save_fig

HORIZONS = (30, 60, 90, 180, 365, 730)
REPO = Path(__file__).resolve().parents[2]


def taxonomy():
    registry = yaml.safe_load((REPO / "opera/configs/experiment_registry.yaml").read_text())
    group_names = {"AMYLOIDOSIS": "Amyloidosis", "BL_LBL": "BL/LBL", "CLL_SLL": "CLL/SLL",
                   "DLBCL_like": "DLBCL-like", "HCL": "HCL", "HL": "Hodgkin",
                   "Indolent_B_NHL": "Indolent B-NHL", "MCL": "MCL", "MM": "Myeloma",
                   "T_NHL": "T-cell NHL"}
    fine_names = {"AMYLOIDOSIS": "Amyloidosis", "TRANSFORMED_FL": "Transformed FL",
                  "SolM": "Solitary plasmacytoma"}
    diseases = [
        {"cohort": fine_names.get(fine, fine), "cohort_group": group_names.get(group, group),
         "patients": count}
        for group, spec in registry["cohort_groups"].items()
        for fine, count in spec["fine"].items()
    ]
    groups = yaml.safe_load(
        (REPO / "opera/configs/outcome_display_groups_60_candidate_v1.yaml").read_text()
    )
    outcomes = [
        {"outcome": outcome_display_name(name), "endpoint_group": group}
        for group, members in groups.items()
        for name in members
    ]
    return pd.DataFrame(diseases), pd.DataFrame(outcomes)


def synthetic_cells(rng):
    diseases, outcomes = taxonomy()
    rows = []
    prevalence = dict(zip(outcomes.outcome, rng.uniform(0.01, 0.25, len(outcomes))))
    for disease in diseases.itertuples():
        for outcome in outcomes.itertuples():
            for horizon in HORIZONS:
                share = 0.8 * disease.patients * prevalence[outcome.outcome]
                events = rng.poisson(share * min(1.0, (horizon / 365) ** 0.6))
                scarcity = 1 / (1 + np.log10(max(events, 1)))
                base = {
                    "D-S": 0.045 * scarcity + rng.normal(0, 0.012),
                    "O-S": 0.025 * scarcity + rng.normal(0, 0.012),
                    "interaction": -0.01 * scarcity + rng.normal(0, 0.008),
                    "S-X": rng.normal(0.004, 0.012),
                    "X-LR": rng.normal(0.012, 0.01),
                    "R1": rng.normal(0.002, 0.006),
                }
                base["J-S"] = base["D-S"] + base["O-S"] + base["interaction"]
                base["J-X"] = base["J-S"] + base["S-X"]
                base["XP-X"] = 0.6 * base["D-S"] + rng.normal(0, 0.008)
                base["J-XP"] = base["J-X"] - base["XP-X"]
                base["R-X"] = base["J-X"] + base["R1"]
                base["EA-L:J"] = rng.normal(0.008, 0.01) + 0.01 * scarcity
                estimable = events >= 5
                for contrast, gain in base.items():
                    for metric, scale in (("auc", 1.0), ("brier", 0.08)):
                        value = gain * scale if estimable else np.nan
                        half = (0.02 + 0.08 * scarcity) * scale
                        rows.append({
                            "cohort": disease.cohort, "cohort_group": disease.cohort_group,
                            "outcome": outcome.outcome,
                            "endpoint_group": outcome.endpoint_group,
                            "horizon_days": horizon, "contrast": contrast,
                            "metric": metric, "gain": value, "lower": value - half,
                            "upper": value + half, "n_events_train": events,
                        })
    return pd.DataFrame(rows)


def synthetic_curve(cells):
    rows = []
    data = cells.loc[(cells.metric == "auc") & (cells.horizon_days == 365)]
    grid = np.logspace(0.7, 3.6, 60)
    for contrast in ("D-S", "O-S", "J-S", "J-X"):
        part = data.loc[(data.contrast == contrast) & data.gain.notna()]
        x = np.log10(part.n_events_train.clip(lower=1))
        for point in grid:
            weights = np.exp(-0.5 * ((x - np.log10(point)) / 0.35) ** 2)
            mean = float(np.average(part.gain, weights=weights))
            spread = 0.006 + 0.02 / np.sqrt(weights.sum())
            rows.append({"contrast": contrast, "training_support": point, "mean": mean,
                         "lower": mean - spread, "upper": mean + spread})
    return pd.DataFrame(rows)


def synthetic_external(rng):
    estimates = pd.DataFrame([
        {"origin": origin, "readout": readout, "metric": "auc", "horizon_days": 365,
         "estimate": value, "lower": value - 0.008, "upper": value + 0.008}
        for (origin, readout), value in {
            ("L", "J"): 0.781, ("L", "R"): 0.785, ("E", "J"): 0.788,
            ("E", "R"): 0.790, ("EA", "J"): 0.796, ("EA", "R"): 0.797,
        }.items()
    ])
    contrasts = pd.DataFrame([
        {"contrast": f"{name}:{readout}", "metric": "auc", "horizon_days": 365,
         "gain": gain, "lower": gain - 0.006, "upper": gain + 0.006}
        for (name, readout), gain in {
            ("E-L", "J"): 0.007, ("E-L", "R"): 0.005, ("EA-E", "J"): 0.008,
            ("EA-E", "R"): 0.007, ("EA-L", "J"): 0.015, ("EA-L", "R"): 0.012,
        }.items()
    ])
    return estimates, contrasts


def synthetic_gate():
    values = {"R0": 0.0005, "R2": 0.004, "R3": 0.0, "R4": 0.001, "R5": 0.005,
              "R6": 0.005, "R7": 0.001}
    return pd.DataFrame([{"contrast": k, "gain": v, "lower": v - 0.003,
                          "upper": v + 0.003} for k, v in values.items()])


def synthetic_atlas(cells, rng):
    base = cells.loc[(cells.contrast == "J-S") & (cells.metric == "auc")]
    outcomes = base.groupby(["outcome", "endpoint_group", "horizon_days"]).n_events_train.sum()
    level = {group: rng.uniform(0.62, 0.9) for group in base.endpoint_group.unique()}
    rows = []
    for (outcome, group, horizon), events in outcomes.items():
        auc = level[group] - 0.02 * np.log(horizon / 30) / np.log(24) + rng.normal(0, 0.035)
        rows.append({"outcome": outcome, "endpoint_group": group, "horizon_days": horizon,
                     "n_events_train": events, "cif_auc": float(np.clip(auc, 0.45, 0.99))})
    return pd.DataFrame(rows)


def synthetic_maps(rng, n=1500):
    centres = np.array([[0, 0], [4, 0.6], [1.6, 3.4]])
    labels = rng.integers(0, 3, n)
    coords = centres[labels] + rng.normal(0, 0.8, (n, 2))
    outcomes = {}
    specs = {
        "Treatment failure": lambda c: 0.15 + 0.35 * (c[:, 0] > 2.5),
        "Serious infection": lambda c: 0.08 + 0.3 * np.exp(-((c[:, 1] - 3.4) ** 2) / 2),
        "Neutropenia ≥ grade 3": lambda c: 0.25 + 0.3 * (c[:, 1] < 1) * (c[:, 0] < 2),
        "Acute hospitalisation": lambda c: 0.3 + 0.15 * np.tanh(c[:, 0] - 2),
    }
    for name, risk in specs.items():
        p = np.clip(risk(coords), 0.02, 0.95)
        event_time = rng.exponential(365 / -np.log1p(-p))
        death_time = rng.exponential(365 / 0.06, n)
        censor_time = rng.uniform(150, 1500, n)
        time = np.minimum.reduce([event_time, death_time, censor_time])
        events = np.select([time == event_time, time == death_time], [1, 2], 0)
        predicted = np.clip(p * (1 + rng.normal(0, 0.05, n)), 0, 1)
        routed = None
        if name != "Treatment failure":
            # Synthetic routed projection: pulled towards that outcome's risk region.
            centre = coords[np.argsort(p)[-200:]].mean(axis=0)
            pull = (centre - coords) * 0.12 * (p[:, None] / p.max())
            routed = coords + pull + rng.normal(0, 0.05, (n, 2))
        outcomes[name] = (time, events, predicted, routed)
    return coords, outcomes


def synthetic_routes(rng):
    diseases, outcomes = taxonomy()
    disease_groups = dict(zip(diseases.cohort, diseases.cohort_group))
    outcome_groups = dict(zip(outcomes.outcome, outcomes.endpoint_group))
    rows = []
    for d in diseases.cohort:
        for o in outcomes.outcome:
            shared = rng.uniform(0.35, 0.8)
            disease_raw = np.array([
                (6.0 if s == d else 1.6 if disease_groups[s] == disease_groups[d] else 0.25)
                * rng.lognormal(0, 0.35) for s in diseases.cohort])
            outcome_raw = np.array([
                (5.0 if s == o else 1.4 if outcome_groups[s] == outcome_groups[o] else 0.2)
                * rng.lognormal(0, 0.35) for s in outcomes.outcome])
            lean = 1 / (1 + np.exp(-(rng.normal(0, 0.6)
                                     + (0.8 if outcome_groups[o].startswith("Disease") else 0)
                                     - (0.6 if "Infection" in outcome_groups[o] else 0))))
            rest = 1 - shared
            disease_w = rest * lean * disease_raw / disease_raw.sum()
            outcome_w = rest * (1 - lean) * outcome_raw / outcome_raw.sum()
            rows.append({"cohort": d, "outcome": o, "bank": "shared", "source": "shared",
                         "mean": shared})
            rows += [{"cohort": d, "outcome": o, "bank": "disease", "source": s, "mean": w}
                     for s, w in zip(diseases.cohort, disease_w)]
            rows += [{"cohort": d, "outcome": o, "bank": "outcome", "source": s, "mean": w}
                     for s, w in zip(outcomes.outcome, outcome_w)]
    return pd.DataFrame(rows), disease_groups, outcome_groups


def synthetic_calibration(cells, rng):
    base = cells.loc[(cells.contrast == "J-S") & (cells.metric == "auc")
                     & (cells.horizon_days == 365)]
    rows = []
    shift = {"J": (1.0, 0.0), "R": (0.98, 0.001), "S": (0.82, -0.004), "X": (0.9, 0.003)}
    for cell in base.itertuples():
        if cell.n_events_train < 10:
            continue
        for model, (slope, bias) in shift.items():
            rows.append({"model": model, "cohort": cell.cohort, "outcome": cell.outcome,
                         "endpoint_group": cell.endpoint_group, "horizon_days": 365,
                         "status": "estimated",
                         "calibration_slope": slope + rng.normal(0, 0.12),
                         "calibration_in_the_large": bias + rng.normal(0, 0.008)})
    summary = pd.DataFrame(rows)
    # One example outcome from each of four endpoint groups, in a common disease.
    groups = base.drop_duplicates("outcome").groupby("endpoint_group").outcome.first()
    preferred = ("Disease course and later malignancy", "Infection and antimicrobial use",
                 "Neutrophil, white-cell and platelet findings", "Renal injury and support")
    picks = [groups[g] for g in preferred if g in groups.index]
    picks += [o for o in groups if o not in picks][: 4 - len(picks)]
    examples = {outcome: ("DLBCL", outcome) for outcome in picks[:4]}
    rel = []
    for cohort, outcome in examples.values():
        for model, distortion in (("J", 1.0), ("X", 1.35)):
            for b in range(8):
                predicted = (b + 0.5) / 8 * 0.6
                observed = 0.6 * (predicted / 0.6) ** distortion + rng.normal(0, 0.012)
                rel.append({"model": model, "cohort": cohort, "outcome": outcome,
                            "horizon_days": 365, "probability_bin": b,
                            "predicted_weighted_mean": predicted,
                            "observed_fraction": max(0.0, observed),
                            "n": int(400 * np.exp(-b / 2.5)) + 15})
    return summary, pd.DataFrame(rel), examples


IPI_ORDER = ("Low", "Intermediate", "High")
RISK_THRESHOLDS = (0.1, 0.4)


def synthetic_risk_patients(rng, n=1600):
    """Patients with the registry's three IPI categories (low/medium/high)."""
    ipi_order = list(IPI_ORDER)
    latent = rng.normal(0, 1, n)
    age = np.clip(62 + 9 * latent + rng.normal(0, 11, n), 19, 96)
    # The index leans on age; the model sees more of the history.
    score = np.clip(np.round(2.6 + 1.0 * latent + 0.06 * (age - 62)
                             + rng.normal(0, 1.1, n)), 0, 8).astype(int)
    ipi = np.digitize(score, [2, 4])
    risk = 1 / (1 + np.exp(-(latent * 1.4 - 1.2)))
    hazard = -np.log1p(-np.clip(risk, 0.01, 0.95)) / 730
    event_time = rng.exponential(1 / hazard)
    censor = rng.uniform(300, 3000, n)
    time = np.minimum(event_time, censor)
    available = rng.random(n) < 0.55
    return pd.DataFrame({
        "time_days": time, "event": (event_time <= censor).astype(int),
        "model_risk": np.clip(risk + rng.normal(0, 0.04, n), 0, 1), "age": age,
        "ipi_score": np.where(available, ipi + 1, np.nan),  # category rank
        "ipi_group": np.where(available, np.array(ipi_order)[ipi], None)})


def synthetic_subgroups(rng):
    who = []
    for covariate, groups in {
        "history": ["< 20 records", "20–99", "100–499", "≥ 500"],
        "age": ["< 50", "50–64", "65–79", "≥ 80"],
        "sex": ["Female", "Male"],
        "ipi": ["Score available", "Score not available"],
    }.items():
        for index, group in enumerate(groups):
            for contrast, base in (("J-S", 0.004), ("J-X", 0.002)):
                trend = 0.003 * (len(groups) - index) / len(groups) if covariate == "history" else 0
                gain = base + trend + rng.normal(0, 0.0008)
                who.append({"covariate": covariate, "subgroup": group, "contrast": contrast,
                            "gain": gain, "lower": gain - 0.0015, "upper": gain + 0.0015})
    national = []
    for covariate, groups in {
        "coverage": ["< 80% of history in vocabulary", "80–95%", "≥ 95%"],
        "support": ["0–19 training events", "20–99", "100–499", "≥ 500"],
        "year": ["2023", "2024", "2025"],
    }.items():
        for index, group in enumerate(groups):
            for contrast, base in (("EA-L:J", 0.003), ("E-L:J", 0.001)):
                tilt = -0.001 * index if covariate == "support" else 0.001 * index
                gain = base + tilt + rng.normal(0, 0.0006)
                national.append({"covariate": covariate, "subgroup": group,
                                 "contrast": contrast, "gain": gain,
                                 "lower": gain - 0.0014, "upper": gain + 0.0014})
    return pd.DataFrame(who), pd.DataFrame(national)


def synthetic_coverage(cells, rng):
    base = cells.loc[(cells.contrast == "J-S") & (cells.metric == "auc")
                     & (cells.horizon_days == 365)].copy()
    status = np.where(base.n_events_train < 5, "no_training_events",
                      np.where(rng.random(len(base)) < 0.04, "single_class_test",
                               np.where(rng.random(len(base)) < 0.01, "fit_failed",
                                        "evaluated")))
    base["status"] = status
    structural = base.cohort.isin(["Amyloidosis", "BL", "LBL", "HCL"]) & \
        base.outcome.str.contains("treatment|Second", case=False)
    base.loc[structural, "status"] = "not_applicable"
    return base


def synthetic_rescue(cells, rng):
    base = cells.loc[(cells.contrast == "J-S") & (cells.metric == "auc")
                     & (cells.horizon_days == 365) & cells.gain.notna()].copy()
    base["tuning_joint_vs_specialist"] = base.gain + rng.normal(0, 0.02, len(base))
    base["joint_vs_specialist"] = base.gain + rng.normal(0, 0.01, len(base))
    harmed = base.tuning_joint_vs_specialist < 0
    base["routed_vs_specialist"] = base.joint_vs_specialist + np.where(
        harmed, rng.normal(0.012, 0.01, len(base)), rng.normal(0.001, 0.006, len(base)))
    return base


def synthetic_strategy(cells, rng):
    base = cells.loc[(cells.contrast == "J-S") & (cells.metric == "auc")
                     & (cells.horizon_days == 365) & cells.gain.notna()].copy()
    base["specialist"] = np.clip(rng.uniform(0.6, 0.9, len(base)), 0.5, 0.99)
    base["joint"] = np.clip(base.specialist + base.gain, 0.5, 0.99)
    base["best_tuning"] = np.clip(base.specialist + base.gain * rng.uniform(0.4, 1.1, len(base))
                                  + rng.normal(0, 0.01, len(base)), 0.5, 0.99)
    selection = base[["cohort", "outcome", "endpoint_group"]].copy()
    selection["selected_model"] = rng.choice(["S", "D", "O"], len(base), p=[0.3, 0.45, 0.25])
    return base, selection


def synthetic_embedding_comparison(seed=23):
    """Invented coordinates for layout, never interpreted as learned geometry.

    Disease groups separate more after fine-tuning, and risk follows age
    within each group, so the proof shows what the real figure should reveal.
    """
    rng = np.random.default_rng(seed)
    n = 1500
    names = ["DLBCL-like", "CLL/SLL", "Myeloma", "Indolent B-NHL", "Hodgkin", "T-cell NHL"]
    disease = rng.choice(len(names), n, p=[0.3, 0.17, 0.2, 0.17, 0.08, 0.08])
    angle = 2 * np.pi * np.arange(len(names)) / len(names)
    centres = np.c_[np.cos(angle), np.sin(angle)] * 3.2
    age = np.clip(rng.normal(66, 12, n) - 8 * (disease == 4), 18, 95)
    groups = np.array(["Low", "Intermediate", "High"], dtype=object)[
        np.digitize(age + rng.normal(0, 8, n), [60, 75])]
    groups[rng.random(n) < 0.5] = None
    stages = {"Local pretrained": 0.35, "Local fine-tuned": 0.85,
              "National pretrained": 0.3, "National + local": 0.5,
              "National fine-tuned": 0.9}
    noise = rng.normal(size=(n, 2))
    within = np.c_[(age - 66) / 18, rng.normal(0, 0.6, n)]
    frames = []
    for stage, separation in stages.items():
        xy = separation * centres[disease] + 0.9 * within + (1.4 - separation) * noise
        frames.append(pd.DataFrame({
            "model": stage, "patient_id": np.arange(n), "x": xy[:, 0], "y": xy[:, 1],
            "disease": np.array(names)[disease], "ipi_group": groups, "age": age,
            "sequence_length": np.maximum(1, np.exp(5 + rng.normal(0, 0.7, n))).astype(int)}))
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/render/prototypes"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = synthetic_cells(rng)
    curve = synthetic_curve(cells)
    estimates, external = synthetic_external(rng)
    mark = "SYNTHETIC DESIGN PROOF"
    risk_patients = synthetic_risk_patients(np.random.default_rng(11), n=2400)
    tuning_patients = synthetic_risk_patients(np.random.default_rng(19), n=2400)
    known, labels = figures._known_status(tuning_patients.loc[tuning_patients.ipi_group.notna()], 730)
    specificity = float((known.loc[labels == 0, "ipi_score"] < len(IPI_ORDER)).mean())
    locked_threshold = float(np.quantile(known.loc[labels == 0, "model_risk"], specificity))
    renders = {
        "supp_clinical_score_age": figures.figure_clinical_score_age(
            risk_patients, outcome_label="Treatment failure", ipi_label="IPI", ipi_order=IPI_ORDER, thresholds=RISK_THRESHOLDS,
            event_free=True, mark=mark),
        "supp_embedding_comparison": figures.figure_embedding_comparison(
            synthetic_embedding_comparison(), mark=mark),
        "fig_atlas": figures.figure_atlas(synthetic_atlas(cells, rng), mark=mark),
        "fig_pooling": figures.figure_pooling(cells, curve, mark=mark),
        "supp_pooling_cells": figures.figure_pooling_cells(cells, mark=mark),
        "fig_tabular": figures.figure_tabular(cells, curve, mark=mark),
        "fig_external": figures.figure_external(
            estimates, pd.concat([external, cells.loc[cells.contrast == "EA-L:J"]]),
            mark=mark),
        "supp_routing": figures.figure_routing(cells, synthetic_gate(), mark=mark),
        "supp_router_usage": figures.figure_router_usage(*synthetic_routes(rng), mark=mark),
        "supp_horizons": figures.figure_horizons(cells, mark=mark),
        "fig_calibration": figures.figure_calibration(*synthetic_calibration(cells, rng),
                                                      mark=mark),
        "fig_risk_groups": figures.figure_risk_groups(
            risk_patients, outcome_label="Treatment failure",
            ipi_label="IPI", ipi_order=IPI_ORDER, thresholds=RISK_THRESHOLDS, mark=mark, event_free=True),
        "fig_clinical_score": figures.figure_clinical_score(
            risk_patients,
            outcome_label="Treatment failure", ipi_label="IPI", ipi_order=IPI_ORDER, thresholds=RISK_THRESHOLDS, mark=mark,
            event_free=True, model_threshold=locked_threshold, ipi_high=len(IPI_ORDER)),
        "supp_score_overview": figures.figure_score_overview([
            {"title": title, "endpoint": endpoint, "horizon_days": 730,
             "patients": synthetic_risk_patients(np.random.default_rng(seed), n=n),
             "thresholds": RISK_THRESHOLDS, "ipi_order": IPI_ORDER, "event_free": True}
            for seed, (title, endpoint, n) in enumerate((
                ("IPI · DLBCL", "Treatment-failure-free survival", 2400),
                ("CLL-IPI · CLL/SLL", "Overall survival", 1600),
                ("R-ISS · Myeloma", "Overall survival", 1800),
                ("FLIPI · Follicular lymphoma", "Treatment-failure-free survival", 1100),
                ("MIPI · Mantle-cell lymphoma", "Overall survival", 700),
                ("IPS · Hodgkin lymphoma", "Treatment-failure-free survival", 900)),
                start=31)], mark=mark),
        "supp_coverage": figures.figure_coverage(synthetic_coverage(cells, rng), mark=mark),
        "supp_support": figures.figure_support(cells, mark=mark),
        "supp_rescue": figures.figure_rescue(synthetic_rescue(cells, rng), mark=mark),
        "supp_strategy": figures.figure_strategy(*synthetic_strategy(cells, rng), mark=mark),
        "supp_calibration_maps": figures.figure_calibration_maps(
            *synthetic_maps(rng), mark=mark),
    }
    who, national = synthetic_subgroups(rng)
    renders["fig_who_benefits"] = figures.figure_subgroup_effects(
        who, contrast_styles={"J-S": (figures.MODEL_STYLE["J"][0], "Joint vs specialist"),
                              "J-X": (figures.MODEL_STYLE["X"][0], "Joint vs XGBoost")},
        covariate_order={"history": "Recorded history before treatment", "age": "Age",
                         "sex": "Sex", "ipi": "Clinical risk score"},
        xlabel="Brier score improvement (×100, 95% CI)", mark=mark)
    renders["supp_national_where"] = figures.figure_subgroup_effects(
        national, contrast_styles={
            "EA-L:J": (figures.ORIGIN_STYLE["EA"][0], "Adapted national vs local"),
            "E-L:J": (figures.ORIGIN_STYLE["E"][0], "National vs local")},
        covariate_order={"coverage": "History covered by national vocabulary",
                         "support": "Target training events", "year": "Year of treatment"},
        xlabel="Brier score improvement (×100, 95% CI)", mark=mark)
    for name, fig in renders.items():
        save_fig(fig, str(args.output_dir / f"{name}.png"))
        print(args.output_dir / f"{name}.png")


if __name__ == "__main__":
    main()
