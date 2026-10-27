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

from opera.evaluation.decision_curves import figure_decision_curve

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
                base["J-XR"] = base["J-X"] - 0.012 + rng.normal(0, 0.008)
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
                            "n_patients_train": int(0.8 * disease.patients),
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


def synthetic_borrowing(rng):
    """Borrowing tables with modest, plausible structure, through the real
    pooling and claim code: about zero across lineages, a little within a
    lineage, the prespecified pairs a few tenths of an AUC point in their
    domain; noise and estimable cells set by each target's size; leaving a
    source out costs less within a lineage (related sources stand in). The true
    sizes are unknown until the tuning run. Returns the map, the prespecified
    pairs, the three levels (with all diseases) and the aggregate test."""
    from opera.run.source_removal import (
        borrowing_map, disease_groups, disease_sizes, hypothesis_table, hypothesis_test,
        lineage_summary)

    groups, sizes = disease_groups(), disease_sizes()
    plan = yaml.safe_load((REPO / "opera/configs/borrowing_hypotheses.yaml").read_text())
    boost = {(h["target"], h["source"], h["domain"]): rng.uniform(0.2, 0.5)
             for h in plan["hypotheses"]}
    domains = list(yaml.safe_load(
        (REPO / "opera/configs/outcome_display_groups_paper_v1.yaml").read_text()))
    rows, levels, effects, draws = [], [], {}, {}
    shared = {}  # one patient resample moves all of a target's pairs alike
    for target, n in sizes.items():
        cells = min(n // 8, 400)
        if not cells:
            continue  # nothing estimable in the test period
        se = (0.03 + 0.4 / np.sqrt(n / 50))  # per domain
        for source in sizes:
            if target == source:
                continue
            base = 0.10 if groups[target] == groups[source] else 0.01
            infection_across = 0.04 if groups[target] != groups[source] else 0.0
            masked = 0.4 if groups[target] == groups[source] else 0.9
            alone, left_out = [], []
            for domain in [*domains, "All outcomes"]:
                if domain == "All outcomes":
                    a, o = np.mean(alone), np.mean(left_out)
                    err = se / np.sqrt(len(domains))
                else:
                    true = base + boost.get((target, source, domain), 0.0) \
                        + 0.8 * boost.get((target, source, "All outcomes"), 0.0) \
                        + (infection_across if domain.startswith("Infection") else 0.0)
                    a = true + rng.normal(0, se)
                    o = masked * true + rng.normal(0, se)
                    alone.append(a)
                    left_out.append(o)
                    err = se
                noise = shared.setdefault((target, domain), rng.normal(0, 1, 400))
                effects[(target, source, domain)] = a
                draws[(target, source, domain)] = (a + err * (0.6 * noise
                                                   + 0.8 * rng.normal(0, 1, 400)))
                rows.append({"target": target, "source": source, "endpoint_group": domain,
                             "in_effect": a, "in_lower": a - 1.96 * err,
                             "in_upper": a + 1.96 * err, "in_se": err, "in_p_value": np.nan,
                             "in_n_cells": cells if domain == "All outcomes"
                             else max(1, cells // len(domains)),
                             "out_effect": o, "out_lower": o - 1.96 * err,
                             "out_upper": o + 1.96 * err})
        dependence = 0.3 + 1.2 * (50 / n) ** 0.35
        for contrast, true, factor in (("dependence", dependence, 1.0),
                                       ("trained_borrowing", 0.6 * dependence, 1.6),
                                       ("routing", 0.4 * dependence, 1.6)):
            err = factor * se / 2
            est = true + rng.normal(0, err)
            levels.append({"contrast": contrast, "metric": "auc", "target": target,
                           "endpoint_group": "All outcomes", "effect": est,
                           "lower": est - 1.96 * err, "upper": est + 1.96 * err,
                           "target_group": groups[target], "n_patients": n})
    raw = pd.DataFrame(rows)
    levels = pd.DataFrame(levels)
    overall = (levels.groupby("contrast")
               .agg(effect=("effect", "mean"),
                    half=("upper", lambda u: 0.0))  # filled below
               .reset_index())
    for row in overall.itertuples():
        part = levels.loc[levels.contrast == row.contrast]
        err = np.sqrt((((part.upper - part.lower) / 3.92) ** 2).sum()) / len(part)
        levels = pd.concat([levels, pd.DataFrame([{
            "contrast": row.contrast, "metric": "auc", "target": "All diseases",
            "endpoint_group": "All outcomes", "effect": row.effect,
            "lower": row.effect - 1.96 * err, "upper": row.effect + 1.96 * err,
            "target_group": None, "n_patients": np.nan}])], ignore_index=True)
    # By outcome domain, all diseases: borrowing helps most where an outcome's
    # meaning differs between diseases (disease course), least where the biology
    # is shared (cytopenia).
    overall = levels.loc[(levels.contrast == "dependence")
                         & (levels.target == "All diseases")].iloc[0]
    shift = {"Disease course and later malignancy": 0.35, "Infection and antimicrobial use": 0.15,
             "Neutrophil, white-cell and platelet findings": -0.3,
             "Erythroid findings and transfusion": -0.2}
    for domain in domains:
        est = overall.effect + shift.get(domain, 0.0) + rng.normal(0, 0.08)
        err = (overall.upper - overall.lower) / 3.92 * 2.2
        levels = pd.concat([levels, pd.DataFrame([{
            "contrast": "dependence", "metric": "auc", "target": "All diseases",
            "endpoint_group": domain, "effect": est, "lower": est - 1.96 * err,
            "upper": est + 1.96 * err, "target_group": None, "n_patients": np.nan}])],
            ignore_index=True)
    # IPA (secondary): the same pattern, noisier and on its own scale.
    ipa = levels.assign(metric="ipa")
    noise = rng.normal(0, 0.15, len(ipa)) * (ipa.upper - ipa.lower)
    ipa = ipa.assign(effect=1.5 * ipa.effect + noise, lower=1.5 * ipa.lower + noise - 0.2,
                     upper=1.5 * ipa.upper + noise + 0.2)
    levels = pd.concat([levels, ipa], ignore_index=True)
    sparse = {t for t, n in sizes.items() if min(n // 8, 400) < 24}
    return (borrowing_map(raw, groups), hypothesis_table(raw, plan), levels,
            hypothesis_test(raw, plan, permutations=2000),
            lineage_summary(effects, draws, groups, exclude=sparse))


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


def synthetic_decision_curves(rng):
    """Decision curves from synthetic patients with an IPI recorded, through
    the real analysis (IPCW net benefit, tuning-observed index risks)."""
    from opera.evaluation.decision_curves import decision_curves

    def patients(n):
        frame = synthetic_risk_patients(rng, n=n)
        frame = frame.loc[frame.ipi_group.notna()].reset_index(drop=True)
        # A tuned tabular model: the same signal, a little noisier.
        frame["xgboost"] = np.clip(frame.model_risk + rng.normal(0, 0.07, len(frame)), 0, 1)
        return frame

    return decision_curves(patients(3200), patients(1200), 730,
                           sorted(set(np.round(np.arange(0.05, 0.401, 0.01), 2))
                                  | set(SYNTHETIC_DCA_POINTS.values())),
                           comparators=["xgboost"], n_boot=300)


SYNTHETIC_DCA_POINTS = {"Low-risk cutoff": 0.083, "High-risk cutoff": 0.317}


def synthetic_subgroups(rng):
    who = []
    for covariate, groups in {
        "history": ["< 20 records", "20–99", "100–499", "≥ 500"],
        "age": ["< 50", "50–64", "65–79", "≥ 80"],
        "sex": ["Female", "Male"],
        "predominant_region": ["Hovedstaden", "Sjælland", "Syddanmark", "Midtjylland",
                               "Nordjylland"],
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
    # More recorded history with age, and least in Hodgkin lymphoma.
    sequence_length = np.maximum(1, np.exp(5 + 0.02 * (age - 66) - 0.5 * (disease == 4)
                                           + rng.normal(0, 0.6, n))).astype(int)
    within = np.c_[(age - 66) / 18, rng.normal(0, 0.6, n)]
    frames = []
    for stage, separation in stages.items():
        xy = separation * centres[disease] + 0.9 * within + (1.4 - separation) * noise
        frames.append(pd.DataFrame({
            "model": stage, "patient_id": np.arange(n), "x": xy[:, 0], "y": xy[:, 1],
            "disease": np.array(names)[disease], "ipi_group": groups, "age": age,
            "sequence_length": sequence_length}))
    return pd.concat(frames, ignore_index=True)


def synthetic_model_cells(cells, rng):
    """Absolute one-year AUCs consistent with the synthetic gains."""
    keys = ["cohort", "outcome", "horizon_days"]
    gains = (cells.loc[cells.metric == "auc"].pivot_table(index=keys, columns="contrast",
                                                          values="gain").reset_index())
    info = cells.drop_duplicates(keys).set_index(keys)
    index = pd.MultiIndex.from_frame(gains[keys])
    # Outcomes differ in how predictable they are; diseases shift that a little.
    level = {o: rng.uniform(0.6, 0.86) for o in gains.outcome.unique()}
    shift = {c: rng.normal(0, 0.025) for c in gains.cohort.unique()}
    xgb = np.clip([level[o] + shift[c] + rng.normal(0, 0.02)
                   for o, c in zip(gains.outcome, gains.cohort)], 0.5, 0.95)
    return pd.DataFrame({**{k: gains[k] for k in keys}, "X": xgb, "J": xgb + gains["J-X"],
                         "n_events_train": info.n_events_train.reindex(index).to_numpy(),
                         "cohort_group": info.cohort_group.reindex(index).to_numpy(),
                         "endpoint_group": info.endpoint_group.reindex(index).to_numpy(),
                         }).dropna(subset=["J"])


def synthetic_efficiency(models, cells, rng):
    """Joint models on a share of training patients: national loses less."""
    models = models.copy()
    models["J_EA"] = models.J + 0.012 + rng.normal(0, 0.006, len(models))
    rows = []
    for fraction, loss_local, loss_national in ((10, 0.07, 0.03), (25, 0.04, 0.017),
                                                (50, 0.018, 0.008)):
        models[f"J_L_F{fraction}"] = models.J - loss_local + rng.normal(0, 0.01, len(models))
        models[f"J_EA_F{fraction}"] = (models.J_EA - loss_national
                                       + rng.normal(0, 0.01, len(models)))
        gain = models[f"J_EA_F{fraction}"] - models[f"J_L_F{fraction}"]
        rows.append(models[["cohort", "outcome", "horizon_days"]].assign(
            contrast=f"EA-L:F{fraction}", metric="auc", gain=gain))
    return models, pd.concat([cells, *rows], ignore_index=True)


def synthetic_regions(rng):
    """Per-region effects, absolute metrics and data coverage."""
    regions = ["Hovedstaden", "Sjælland", "Syddanmark", "Midtjylland", "Nordjylland"]
    effects, models, coverage = [], [], []
    for region in regions:
        for contrast, base in (("J-S", 0.022), ("J-X", 0.025)):
            gain = base + rng.normal(0, 0.006)
            effects.append({"region": region, "contrast": contrast, "metric": "auc",
                            "horizon_days": 0, "gain": gain, "lower": gain - 0.012,
                            "upper": gain + 0.012})
        joint = 0.79 + rng.normal(0, 0.01)
        for model, value, bias in (("J", joint, rng.normal(0, 0.004)),
                                   ("X", joint - 0.025, rng.normal(0.004, 0.004))):
            models.append({"region": region, "model": model, "metric": "auc",
                           "horizon_days": 0, "estimate": value,
                           "lower": value - 0.015, "upper": value + 0.015})
            models.append({"region": region, "model": model, "metric": "calibration_bias",
                           "horizon_days": 0, "estimate": bias,
                           "lower": bias - 0.006, "upper": bias + 0.006})
        history = rng.uniform(300, 900)
        labs = history * rng.uniform(0.25, 0.5)
        coverage.append({"region": region, "history_event_count_median": history,
                         "history_event_count_q1": history * 0.5,
                         "history_event_count_q3": history * 1.8,
                         "laboratory_event_count_median": labs,
                         "laboratory_event_count_q1": labs * 0.4,
                         "laboratory_event_count_q3": labs * 2})
    return pd.DataFrame(effects), pd.DataFrame(models), pd.DataFrame(coverage)


def synthetic_headline_scores(rng):
    labels = ["IPI · DLBCL", "NCCN-IPI · DLBCL", "IPI · all lymphomas", "CLL-IPI · CLL/SLL",
              "R-ISS · Myeloma", "FLIPI · Follicular", "MIPI · Mantle cell", "IPS · Hodgkin"]
    index = rng.uniform(0.42, 0.6, len(labels))
    return pd.DataFrame({"label": labels, "index_sensitivity": index,
                         "model_sensitivity": index + rng.uniform(0.05, 0.16, len(labels)),
                         "specificity": rng.uniform(0.65, 0.8, len(labels))})


def synthetic_gallery(rng, n=2500):
    """Patients in six regions; each outcome's risk concentrates in one."""
    import yaml
    from opera.run.build_paper_embedding_tables import embedding_gallery_inputs
    from opera.visualization.paper_taxonomy import DISPLAY_GROUPS

    angle = 2 * np.pi * np.arange(6) / 6
    centers = np.c_[np.cos(angle), np.sin(angle)] * 4
    region = rng.integers(0, 6, n)
    xy = centers[region] + rng.normal(0, 1.3, (n, 2))
    coords = pd.DataFrame({"subject_id": range(n), "x": xy[:, 0], "y": xy[:, 1]})
    groups = yaml.safe_load(Path(DISPLAY_GROUPS).read_text())
    rows = []
    for k, outcome in enumerate(o for members in groups.values() for o in members[:6]):
        risk = np.clip((0.04 + 0.1 * rng.uniform())
                       * np.where(region == k % 6, 3.0, 0.7), 0, 0.95)
        event_time = np.where(rng.uniform(size=n) < risk, rng.uniform(1, 365, n), 2000)
        censor = rng.uniform(200, 1500, n)
        rows.append(pd.DataFrame({"subject_id": range(n), "outcome": outcome,
                                  "time_days": np.minimum(event_time, censor),
                                  "event": (event_time <= censor).astype(int),
                                  "cif_365d": np.clip(risk * rng.lognormal(0, 0.25, n),
                                                      0, 0.99)}))
    coords, outcomes = embedding_gallery_inputs(coords, pd.concat(rows))
    names = np.array(["DLBCL-like", "CLL/SLL", "Myeloma", "Indolent B-NHL", "Hodgkin",
                      "T-cell NHL"])
    return coords.assign(disease=names[region]), outcomes


FACTOR_CODES = {
    "frailty": ["DIAGNOSIS//DI50", "DIAGNOSIS//DN18", "CONTACT_TYPE//ACUTE",
                "MEDICATION//C03CA01", "LABTEST//NPU19673//BIN_1"],
    "infection": ["DIAGNOSIS//DJ189", "MEDICATION//J01CA04", "MEDICATION//J02AC01",
                  "LABTEST//NPU19748//BIN_5"],
    "marrow": ["LABTEST//NPU02902//BIN_1", "LABTEST//NPU03568//BIN_1",
               "MEDICATION//L01XC02", "LABTEST//NPU02319//BIN_1"],
    "renal": ["LABTEST//NPU18016//BIN_6", "DIAGNOSIS//DN185"],
    "disease": ["PATHOLOGY//M96803", "LABTEST//NPU19123//BIN_6", "PROCEDURE//KTSF",
                "PATHOLOGY//SYNTH_TP53", "PATHOLOGY//SYNTH_MYC", "PATHOLOGY//SYNTH_IGHV"],
    "none": ["CONTACT_SPECIALTY//HAEM", "MEDICATION//N02BE01", "DIAGNOSIS//DZ00",
             "REGION//HOVEDSTADEN"],
}
# Stand-ins for configured findings whose real codes are not yet known
# (molecular pathology codes, found on the server by the vocabulary audit).
SYNTHETIC_CONCEPT_CODES = {"tp53_aberration": ["PATHOLOGY//SYNTH_TP53"],
                           "myc_rearrangement": ["PATHOLOGY//SYNTH_MYC"],
                           "ighv_unmutated": ["PATHOLOGY//SYNTH_IGHV"]}
CONCEPT_LABELS = {
    "PATHOLOGY//SYNTH_TP53": "TP53 aberration", "PATHOLOGY//SYNTH_MYC": "MYC rearrangement",
    "PATHOLOGY//SYNTH_IGHV": "Unmutated IGHV",
    "DIAGNOSIS//DI50": "Heart failure", "DIAGNOSIS//DN18": "Chronic kidney disease",
    "CONTACT_TYPE//ACUTE": "Acute admission", "MEDICATION//C03C": "Loop diuretics",
    "LABTEST//NPU19673//low": "Low albumin", "DIAGNOSIS//DJ18": "Pneumonia",
    "MEDICATION//J01C": "Penicillins", "MEDICATION//J02A": "Antimycotics",
    "LABTEST//NPU19748//high": "High CRP", "LABTEST//NPU02902//low": "Low neutrophils",
    "LABTEST//NPU03568//low": "Low platelets", "MEDICATION//L01X": "Antineoplastics",
    "LABTEST//NPU02319//low": "Low haemoglobin", "LABTEST//NPU18016//high": "High creatinine",
    "PATHOLOGY//M96": "Lymphoma morphology", "LABTEST//NPU19123//high": "High LDH",
    "PROCEDURE//KTSF": "Biopsy",
    "LABTEST//NPU19673//BIN": "Albumin", "LABTEST//NPU19748//BIN": "CRP",
    "LABTEST//NPU02902//BIN": "Neutrophils", "LABTEST//NPU03568//BIN": "Platelets",
    "LABTEST//NPU02319//BIN": "Haemoglobin", "LABTEST//NPU18016//BIN": "Creatinine",
    "LABTEST//NPU19123//BIN": "LDH",
}
DOMAIN_FACTORS = {
    "Disease course": {"disease": 1.0, "frailty": 0.6},
    "Infection": {"infection": 1.0, "marrow": 0.3, "frailty": 0.3},
    "Erythroid": {"marrow": 0.8, "disease": 0.3},
    "Neutrophil": {"marrow": 1.0},
    "Lymphocyte": {"marrow": 0.5, "disease": 0.4},
    "Liver": {"frailty": 0.3, "disease": 0.5},
    "Renal": {"renal": 1.0, "frailty": 0.4},
    "Cardiovascular": {"frailty": 1.0},
    "Acute care": {"frailty": 0.7, "infection": 0.4},
}


SYNTHETIC_BINS = {npu: {"min": 1, "max": 6} for npu in (
    "NPU19673", "NPU19748", "NPU02902", "NPU03568", "NPU02319", "NPU18016", "NPU19123")}


def synthetic_patient(weights, factors):
    """A curated synthetic history with a clinical story (no real patient).

    Long-standing heart failure on loop diuretics, a pneumonia admission two
    months before treatment, the diagnostic work-up and, days before treatment,
    low neutrophils and albumin. Each event carries one latent tendency, and
    every outcome reads it with its domain's weights, as in the synthetic
    population. Returns the wide token table (log-odds per outcome).
    """
    events = [("BG_SEX//M", 25000, "none", 0.0)]
    events += [("DIAGNOSIS//DI50", 1460, "frailty", 1.0),
               ("CONTACT_TYPE//ACUTE", 1460, "frailty", 0.3)]
    events += [("MEDICATION//C03CA01", d, "frailty", 0.15) for d in range(1400, 100, -90)]
    events += [("LABTEST//NPU02319//BIN_4", 420, "marrow", -0.3),
               ("LABTEST//NPU19673//BIN_4", 420, "frailty", -0.3)]
    events += [("CONTACT_TYPE//ACUTE", 71, "infection", 0.4),
               ("DIAGNOSIS//DJ189", 68, "infection", 1.2),
               ("LABTEST//NPU19748//BIN_5", 71, "infection", 0.6)]
    events += [("MEDICATION//J01CA04", d, "infection", 0.4) for d in (71, 70, 69)]
    events += [("PATHOLOGY//M96803", 21, "disease", 0.8), ("PROCEDURE//KTSF", 22, "disease", 0.3),
               ("LABTEST//NPU19123//BIN_6", 20, "disease", 0.7)]
    events += [("LABTEST//NPU02902//BIN_1", 5, "marrow", 1.6),
               ("LABTEST//NPU19673//BIN_1", 5, "frailty", 0.7),
               ("LABTEST//NPU02319//BIN_4", 5, "marrow", -0.4),
               ("CONTACT_SPECIALTY//HAEM", 3, "none", 0.0)]
    tokens = pd.DataFrame(events, columns=["code", "days_before_index", "factor", "size"])
    tokens = tokens.assign(subject_id=-1, weight=1.0, position=range(len(tokens)))
    index = tokens.factor.map(factors.index).to_numpy()
    for outcome, w in weights.items():
        tokens[f"attr|logit|{outcome}|365"] = 0.25 * w[index] * tokens["size"].to_numpy()
    return tokens.drop(columns=["factor", "size"])


# Direction each synthetic code carries: abnormal findings raise risk,
# mid-range results and routine contacts barely move it.
CODE_EFFECT = {"LABTEST//NPU02319//BIN_4": -0.5, "LABTEST//NPU19673//BIN_4": -0.5,
               "CONTACT_SPECIALTY//HAEM": 0.0, "MEDICATION//N02BE01": 0.0,
               "DIAGNOSIS//DZ00": 0.0, "REGION//HOVEDSTADEN": 0.0}


def synthetic_attribution_patterns(seed=41, n=500):
    """Structured synthetic histories run through the real summaries.

    Latent patient tendencies (frailty, infection, marrow, renal, disease
    activity) generate events, abnormal findings raise risk, and each clinical
    domain reads the tendencies with different weights. Every table is made by
    the pipeline's own functions, so the figures show what the analysis
    produces, not drawn-on patterns.
    """
    import yaml

    from opera.interpretability.attribution_patterns import (
        episode_contributions, explanation_table, outcome_similarity, variance_decomposition)
    from opera.interpretability.attribution_summaries import (
        concept_effects, label_tokens, long_tokens)
    from opera.visualization.paper_taxonomy import outcome_endpoint_groups

    rng = np.random.default_rng(seed)
    domains = outcome_endpoint_groups()
    outcomes = list(domains)
    factors = list(FACTOR_CODES)
    weights = {}
    for outcome, domain in domains.items():
        spec = next((v for k, v in DOMAIN_FACTORS.items() if domain.startswith(k)),
                    {"disease": 0.5, "renal": 0.3})
        weights[outcome] = np.array([spec.get(f, 0.0) for f in factors]) \
            * rng.uniform(0.7, 1.3, len(factors))
    rows, columns = [], {o: f"attr|logit|{o}|365" for o in outcomes}
    latent = rng.normal(0.3, 1, (n, len(factors)))
    latent[:, factors.index("none")] = 0
    for subject in range(n):
        k = rng.integers(25, 90)
        factor = rng.choice(len(factors), size=k, p=[0.2, 0.15, 0.15, 0.08, 0.12, 0.3])
        recent = rng.random(k) < 0.4
        days = np.where(recent, rng.exponential(12, k), rng.uniform(30, 1500, k))
        recency = np.where(days <= 30, 2.0, 1.0)
        row = {"subject_id": subject, "weight": 1.0}
        for j in range(k):
            codes = FACTOR_CODES[factors[factor[j]]]
            code = codes[rng.integers(len(codes))]
            rows.append({**row, "code": code, "days_before_index": float(days[j]),
                         "position": j, "_f": factor[j], "_r": recency[j],
                         "_e": CODE_EFFECT.get(code, 1.0),
                         "_noise": rng.normal(0, 1, len(outcomes))})
        rows.append({**row, "code": "BG_SEX//F", "days_before_index": 25000.0,
                     "position": -1, "_f": factors.index("none"), "_r": 1.0, "_e": 0.0,
                     "_noise": rng.normal(0, 1, len(outcomes))})
    frame = pd.DataFrame(rows)
    f_index, r, effect = frame._f.to_numpy(), frame._r.to_numpy(), frame._e.to_numpy()
    noise = np.stack(frame._noise)
    lat = latent[frame.subject_id.to_numpy(), f_index]
    for i, outcome in enumerate(outcomes):
        loading = weights[outcome][f_index]
        frame[columns[outcome]] = 0.05 * loading * (effect * (0.6 + 0.4 * lat)
                                                    + 0.2 * noise[:, i]) * r
    wide = frame.drop(columns=["_f", "_r", "_e", "_noise"])
    patient_rows = []
    for outcome in outcomes:
        sums = wide.groupby("subject_id")[columns[outcome]].sum()
        base = np.log(0.08 / 0.92) + rng.normal(0, 0.25, n)
        prediction = base + sums.to_numpy()
        event = rng.random(n) < 1 / (1 + np.exp(-prediction))
        patient_rows.append(pd.DataFrame({
            "subject_id": sums.index, "outcome": outcome, "column": columns[outcome],
            "prediction": prediction, "baseline_prediction": base, "weight": 1.0,
            "time_days": np.where(event, rng.uniform(10, 360, n), 400.0),
            "event": event.astype(int)}))
    patients = pd.concat(patient_rows, ignore_index=True)

    config = yaml.safe_load((REPO / "opera/configs/attribution_analysis.example.yaml").read_text())
    variance = {"source": [], "window": []}
    concepts = []
    for outcome in outcomes:
        long = label_tokens(long_tokens(wide, columns[outcome], outcome=outcome))
        rows_o = patients.loc[patients.outcome == outcome]
        for by in variance:
            variance[by].append(variance_decomposition(long, rows_o, by=by, draws=60)
                                .assign(outcome=outcome))
        specs = [{**c, "codes": c["codes"] or SYNTHETIC_CONCEPT_CODES.get(c["name"], []),
                  "outcome": outcome, "expected": (c.get("expected") or {}).get(outcome)}
                 for c in config["concepts"]]
        table = concept_effects(long, specs, SYNTHETIC_BINS, draws=60, min_count=20)
        for name in ("mean_pp", "lower_pp", "upper_pp", "median_pp"):
            table[name] = table[name] / 100  # log-odds, as the summariser reports them
        concepts.append(table)
    similarity = outcome_similarity(wide, list(columns.values()), min_count=30)
    reverse = {v: k for k, v in columns.items()}
    similarity = similarity.assign(outcome_a=similarity.column_a.map(reverse),
                                   outcome_b=similarity.column_b.map(reverse))
    restrictions = []
    effects = {item["label"]: value for item, value in zip(
        config["restrictions"], (-0.06, -0.025, -0.012, -0.015, -0.003, -0.008))}
    lab_heavy = ("Neutrophil", "Renal", "Erythroid", "Liver", "Sodium", "Calcium", "Metabolic")
    for name, effect in effects.items():
        for group in ["All outcomes", *dict.fromkeys(domains.values())]:
            scale = 1.0 if group == "All outcomes" else rng.uniform(0.5, 1.5)
            if name == "Without laboratory results" and group.startswith(lab_heavy):
                scale *= 2.5
            if name == "Without medication" and group.startswith("Infection"):
                scale *= 2.5
            if name == "Without diagnoses" and group.startswith(("Cardiovascular", "Disease")):
                scale *= 2.0
            mean = effect * scale
            half = 0.004 if group == "All outcomes" else 0.01
            restrictions.append({"restriction": name, "group": group, "mean_delta": mean,
                                 "lower": mean - half, "upper": mean + half})

    # One curated patient, on the absolute-risk scale, for the clinical figure.
    curated = synthetic_patient(weights, factors)
    check = ["overall_survival", "serious_infection_composite", "neutropenia_g3plus"]
    risk_columns = {o: f"attr|risk|{o}|365" for o in check}
    rows = []
    for outcome in check:
        base = float(patients.loc[patients.outcome == outcome, "baseline_prediction"].median())
        logit = curated[columns[outcome]]
        p, p0 = 1 / (1 + np.exp(-(base + logit.sum()))), 1 / (1 + np.exp(-base))
        curated[risk_columns[outcome]] = logit * (p - p0) / logit.sum()
        rows.append({"subject_id": -1, "column": risk_columns[outcome], "outcome": outcome,
                     "prediction": p, "baseline_prediction": p0})
    patient_rows = pd.DataFrame(rows)
    explanation = explanation_table(curated, patient_rows, columns=risk_columns,
                                    labels=CONCEPT_LABELS, counts=SYNTHETIC_BINS, n=3)
    episodes_risk = episode_contributions(curated, patient_rows, -1, columns=risk_columns,
                                          labels=CONCEPT_LABELS, counts=SYNTHETIC_BINS)
    # Six complications across domains, on log-odds, for the supplementary tracks.
    across = ["overall_survival", "serious_infection_composite", "neutropenia_g3plus",
              "any_transfusion", "heart_failure", "aki_first_documented"]
    logit_columns = {o: columns[o] for o in across if o in columns}
    logit_rows = []
    for outcome, column in logit_columns.items():
        base = float(patients.loc[patients.column == column, "baseline_prediction"].median())
        logit_rows.append({"subject_id": -1, "column": column, "outcome": outcome,
                           "prediction": base + curated[column].sum(), "baseline_prediction": base})
    episodes_logit = episode_contributions(curated, pd.DataFrame(logit_rows), -1,
                                           columns=logit_columns, labels=CONCEPT_LABELS,
                                           counts=SYNTHETIC_BINS, scale=1.0)
    return dict(variance_sources=pd.concat(variance["source"], ignore_index=True),
                variance_windows=pd.concat(variance["window"], ignore_index=True),
                concepts=pd.concat(concepts, ignore_index=True), similarity=similarity,
                restrictions=pd.DataFrame(restrictions), explanation=explanation,
                episodes_risk=episodes_risk, episodes_logit=episodes_logit,
                wide=wide, patients=patients)


def synthetic_hierarchy(rng, n=300):
    events = rng.integers(2, 3000, n)
    own = rng.normal(.005, .02, n)
    borrow = rng.normal(.01, .025, n)
    return pd.DataFrame({'metric': 'auc', 'horizon_days': 365,
                         'n_events_train': events,
                         'tuning_joint_vs_specialist': rng.normal(.01, .03, n),
                         'own': own, 'borrow': borrow, 'both': own + borrow,
                         'router_design': 'AB'})


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
    borrowing = synthetic_borrowing(np.random.default_rng(29))
    renders = {
        "supp_clinical_score_age": figures.figure_clinical_score_age(
            risk_patients, outcome_label="Overall survival", ipi_label="IPI", ipi_order=IPI_ORDER, thresholds=RISK_THRESHOLDS,
            event_free=True, mark=mark),
        "supp_embedding_comparison": figures.figure_embedding_comparison(
            synthetic_embedding_comparison(), mark=mark),
        "fig_atlas": figures.figure_atlas(
            synthetic_atlas(cells, rng), synthetic_model_cells(cells, np.random.default_rng(4)),
            mark=mark),
        "supp_atlas_matrix": figures.figure_atlas_matrix(
            synthetic_atlas(cells, np.random.default_rng(9)),
            synthetic_model_cells(cells, np.random.default_rng(4)), mark=mark),
        "fig_pooling": figures.figure_pooling(cells, curve, mark=mark),
        "supp_pooling_cells": figures.figure_pooling_cells(cells, mark=mark),
        "fig_tabular": figures.figure_tabular(cells, curve, mark=mark),
        "fig_external": figures.figure_external(
            estimates, *(lambda m, c: (c, None, m))(*synthetic_efficiency(
                synthetic_model_cells(cells, np.random.default_rng(4)),
                pd.concat([external, cells.loc[cells.contrast == "EA-L:J"]]),
                np.random.default_rng(6))),
            mark=mark),
        "supp_routing": figures.figure_routing(cells, synthetic_gate(), mark=mark),
        "supp_router_usage": figures.figure_router_usage(*synthetic_routes(rng), mark=mark),
        "fig_borrowing_map": figures.figure_borrowing_map(*borrowing[:4], mark=mark),
        "supp_borrowing_detail": figures.figure_borrowing_detail(
            borrowing[0], borrowing[1], borrowing[2], mark=mark),
        "supp_horizons": figures.figure_horizons(cells, mark=mark),
        "fig_hierarchy": figures.figure_hierarchy(synthetic_hierarchy(rng), mark=mark),
        "fig_decision_curve": figure_decision_curve(
            synthetic_decision_curves(np.random.default_rng(17)),
            labels={"xgboost": "XGBoost"}, mark=mark, points=SYNTHETIC_DCA_POINTS),
        "fig_calibration": figures.figure_calibration(*synthetic_calibration(cells, rng),
                                                      mark=mark),
        "fig_benchmarks": figures.figure_benchmarks(
            cells.assign(gain=np.where(cells.contrast == "J-XR", cells.gain, cells.gain)),
            curve, *synthetic_calibration(cells, np.random.default_rng(3)), mark=mark),
        "fig_risk_groups": figures.figure_risk_groups(
            risk_patients, outcome_label="Overall survival",
            ipi_label="IPI", ipi_order=IPI_ORDER, thresholds=RISK_THRESHOLDS, mark=mark, event_free=True),
        "fig_clinical_score": figures.figure_clinical_score(
            risk_patients,
            outcome_label="Overall survival", ipi_label="IPI", ipi_order=IPI_ORDER, thresholds=RISK_THRESHOLDS, mark=mark,
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
    from opera.visualization.attribution_overview import figure_patient_tracks
    patterns = synthetic_attribution_patterns()
    gallery = synthetic_gallery(np.random.default_rng(23))
    renders["supp_attribution_atlas"] = figures.figure_attribution_atlas(
        patterns["variance_sources"], patterns["variance_windows"], patterns["concepts"],
        mark=mark)
    renders["supp_attribution_structure"] = figures.figure_attribution_structure(
        patterns["similarity"], patterns["restrictions"], mark=mark)
    renders["supp_attribution_patient"] = figure_patient_tracks(
        patterns["episodes_logit"], mark=mark)
    renders["fig_attribution"] = figures.figure_attribution_main(
        patterns["variance_sources"], patterns["concepts"], patterns["restrictions"],
        patterns["episodes_risk"], mark=mark)
    # The illustrative patient is synthetic in the paper as well: keep its record
    # so the real main figure (aggregates from the locked run) can reuse it.
    patterns["episodes_risk"].to_csv(REPO / "paper/figures/synthetic_patient_episodes_risk.csv",
                                     index=False)
    patterns["episodes_logit"].to_csv(
        REPO / "paper/figures/synthetic_patient_episodes_logodds.csv", index=False)
    renders["fig_patient_map"] = figures.figure_patient_map(*gallery, mark=mark)
    for page in range(1, 7):
        renders[f"supp_embedding_gallery_{page}"] = figures.figure_embedding_gallery(
            *gallery, page=page, mark=mark)
    who, national = synthetic_subgroups(rng)
    renders["supp_regions"] = figures.figure_regions(
        *synthetic_regions(np.random.default_rng(12)), mark=mark)
    renders["fig_headline"] = figures.figure_headline(
        cells, synthetic_model_cells(cells, np.random.default_rng(4)), who,
        synthetic_headline_scores(np.random.default_rng(5)), mark=mark)
    renders["fig_who_benefits"] = figures.figure_who_benefits(
        who, who.assign(gain=who.gain * 3, lower=who.lower * 3, upper=who.upper * 3), contrast_styles={"J-S": (figures.MODEL_STYLE["J"][0], "Joint vs specialist"),
                              "J-X": (figures.MODEL_STYLE["X"][0], "Joint vs XGBoost")},
        covariate_order={"history": "Recorded history before treatment", "age": "Age",
                         "sex": "Sex", "ipi": "Clinical risk score"},
        mark=mark)
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
