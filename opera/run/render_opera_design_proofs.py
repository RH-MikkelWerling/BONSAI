"""Synthetic design proofs of the new-OPERA figures (no data; watermarked).

Each synthetic input has the columns of the real aggregate it stands in for
and runs through the same builders where one exists (plot-table builders,
task-structure export, arrow aggregation), so the proofs exercise the real
pipeline. Effect sizes follow the tuning picture of 1 November 2026 (J 0.771;
own corrections 0.783; reading + own corrections 0.791; routing ~0; seed
spread ~0.005) but are invented: nothing here is a result.

    python -m opera.run.render_opera_design_proofs --output-dir <dir>
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from opera.run.render_paper_figure_prototypes import taxonomy

MARK = "SYNTHETIC DESIGN PROOF"
REPO = Path(__file__).resolve().parents[2]
DOMAIN_SHIFT = {  # reading gain on top of corrections, by clinical domain
    "Liver injury and coagulation": 0.019, "Infection and antimicrobial use": 0.018,
    "Calcium and magnesium": 0.010, "Metabolic and cell-turnover markers": 0.009,
}
XGB_GAP = {  # joint model minus per-task XGBoost, by clinical domain
    "Liver injury and coagulation": -0.08, "Calcium and magnesium": -0.05,
    "Metabolic and cell-turnover markers": -0.045, "Sodium and potassium": -0.03,
    "Renal injury and support": -0.022, "Cardiovascular and thrombotic events": -0.02,
    "Acute care and other events": -0.015, "Erythroid findings and transfusion": -0.01,
    "Infection and antimicrobial use": -0.012, "Neutrophil, white-cell and platelet findings": -0.004,
    "Lymphocyte and eosinophil counts": 0.0, "Disease course and later malignancy": 0.008,
}


def _cells(rng):
    """Disease x outcome cells pooled over horizons, with one-year support."""
    diseases, outcomes = taxonomy()
    prevalence = dict(zip(outcomes.outcome, rng.uniform(0.01, 0.25, len(outcomes))))
    rows = []
    for disease in diseases.itertuples():
        for outcome in outcomes.itertuples():
            rows.append({"cohort": disease.cohort, "cohort_group": disease.cohort_group,
                         "outcome": outcome.outcome, "endpoint_group": outcome.endpoint_group,
                         "horizon_days": 0,
                         "n_events_train": int(rng.poisson(0.8 * disease.patients * prevalence[outcome.outcome]))})
    return pd.DataFrame(rows)


def _effect_rows(contrast, cells, gains, rng, *, split="tuning", width=0.0025):
    """Saved paired summaries (all cells; by clinical domain) of one contrast."""
    rows = [{"split": split, "seed": 42, "contrast": contrast, "metric": "auc", "horizon_days": 0,
             "stratum": "all_cells", "stratum_value": "all", "gain": float(np.mean(gains)),
             "lower": float(np.mean(gains) - width), "upper": float(np.mean(gains) + width),
             "n_cells": len(gains)}]
    frame = pd.DataFrame({"g": cells.endpoint_group.to_numpy(), "gain": gains})
    for group, part in frame.groupby("g"):
        half = 1.96 * part.gain.std() / np.sqrt(len(part)) if len(part) > 1 else width
        rows.append({**rows[0], "stratum": "endpoint_group", "stratum_value": group,
                     "gain": part.gain.mean(), "lower": part.gain.mean() - half,
                     "upper": part.gain.mean() + half, "n_cells": len(part)})
    return rows


def synthetic_gain_sources(rng):
    """``gain_sources_cells.csv`` through the real builder (ablations flipped)."""
    from opera.run.build_paper_plot_tables import gain_sources_cells

    cells = _cells(rng)
    cells = cells.loc[cells.n_events_train >= 1].reset_index(drop=True)
    log_n = np.log10(cells.n_events_train.clip(lower=1)).to_numpy()
    noise = 0.22 / np.sqrt(cells.n_events_train.to_numpy() + 5) + 0.004
    shift = cells.endpoint_group.map(DOMAIN_SHIFT).fillna(0.004).to_numpy()
    means = {
        "OFF_J": 0.016 * np.tanh(log_n - 1.4) + 0.004,          # own corrections help once data allow
        "OAOWN_OFF": shift + 0.01 / (1 + log_n),                 # reading: rare outcomes, liver, infection
        "AB_OFF": np.zeros(len(cells)),                          # routing adds nothing
        "OAOWN_DIS": np.full(len(cells), 0.004),                 # what each correction adds
        "OAOWN_OUT": np.full(len(cells), 0.006),
    }
    means["OAOWN_J"] = means["OFF_J"] + means["OAOWN_OFF"]
    contrasts, summaries = [], []
    for code, mean in means.items():
        gains = mean + rng.normal(0, noise)
        contrasts.append(cells.assign(split="held_out", seed=42, contrast=code, metric="auc",
                                      gain=gains, lower=gains - 1.96 * noise, upper=gains + 1.96 * noise))
        summaries += _effect_rows(code, cells, gains, rng, split="held_out")
    metadata = cells[["cohort", "outcome", "horizon_days", "cohort_group", "endpoint_group",
                      "n_events_train"]]
    contrasts = pd.concat(contrasts, ignore_index=True).drop(columns=["cohort_group", "endpoint_group",
                                                                      "n_events_train"])
    return gain_sources_cells(contrasts, pd.DataFrame(summaries), metadata)


TUNING_AUC = {
    "N_A_L": 0.7765, "RBH_O_L": 0.7832, "RBH_AB_L": 0.7830, "R_PF_OFF_L": 0.7827, "R_O_L": 0.7833,
    "R_L": 0.7836, "R_ORIG_L": 0.7790, "R_PP_L": 0.7781, "R_PP_FROZEN_L": 0.7794, "R_PT5_L": 0.7799,
    "R_PT3V_L": 0.7805, "R_PF5E_L": 0.7830, "R_PF5_L": 0.7826, "R_PF5B_L": 0.7822, "R_PF10E_L": 0.7828,
    "R_PF5EP_L": 0.7818, "R_PF5EL_L": 0.7829, "R_PF5ED_L": 0.7831, "R_PF5EO_L": 0.7827,
    "R_PF5E_LR10_L": 0.7825, "R_PF5E_W0_L": 0.7830, "R_PF5E_LIVE_L": 0.7821, "R_PF5E_FROZEN_L": 0.7829,
    "R_PH_G_L": 0.7833, "R_PH_GE_L": 0.7835, "R_PH_GE_PF5E_L": 0.7832, "R_PL8_L": 0.7838,
    "R_PF_OFF_R32_L": 0.7826, "R_PF5E_R32_L": 0.7825, "J_OA_OWN_R32_L": 0.7911,
    "J_L": 0.7711, "J_OA_L": 0.7741, "J_OA_LR_L": 0.7752, "J_SP_L": 0.7738, "J_FP_L": 0.7779,
    "J_FPF_L": 0.7783, "J_FPF_PH_L": 0.7800,
    "J_OA_OWN_L": 0.7916, "J_OA_OWN_PH_L": 0.7912, "J_OA_LR_OWN_L": 0.7910, "J_FPF_OWN_L": 0.7904,
    "J_FPF_PH_OWN_PH_L": 0.7908, "J_OA_OWN_NODIS_L": 0.7876, "J_OA_OWN_NOOUT_L": 0.7858,
    "J_OA_OWN_SW20_L": 0.7913, "J_OA_OWN_SW100_L": 0.7909, "J_OA_OWN_X_L": 0.7914,
    "J_OA_OWN_X_SW_L": 0.7911, "J_OA_OWN_PE_L": 0.7915, "J_OA_OWN_PH_X_L": 0.7910,
    "J_OA_OWN_PE_X_SW_L": 0.7912, "J_OA_OWN_MAX_L": 0.7905,
    "J_ELR10_L": 0.7653, "R_AB_ELR10_L": 0.7781, "R_AB_ELR3_L": 0.7812, "J_OA_OWN_ELR10_L": 0.7821,
    "J_FPF_PH_OWN_PH_ELR10_L": 0.7808,
    "J_OA_OWN_s43_L": 0.7890, "J_OA_OWN_s44_L": 0.7905, "J_OA_OWN_s45_L": 0.7940,
    "R_AB_s43_L": 0.7834, "J_s43_L": 0.7708, "R_PF5E_s43_L": 0.7829,
}


def synthetic_design_search(rng):
    """``design_search.csv`` through the real builder from DS_* summaries."""
    from opera.run.analyse_locked_predictions import design_search_contrasts
    from opera.run.build_paper_plot_tables import design_search_table

    rows = []
    for name, expression in design_search_contrasts(set(TUNING_AUC) | {"J_OA_OWN_L", "R_L"}).items():
        model, reference = expression[5:-1].split(",")
        gain = TUNING_AUC[model] - TUNING_AUC[reference] + rng.normal(0, 0.0003)
        half = 0.0018 + 0.0012 * rng.random() + 0.08 * abs(gain)
        rows.append({"split": "tuning", "seed": 42, "contrast": name, "metric": "auc",
                     "horizon_days": 0, "stratum": "all_cells", "stratum_value": "all",
                     "gain": gain, "lower": gain - half, "upper": gain + half, "n_cells": 1340})
    return design_search_table(pd.DataFrame(rows))


def synthetic_task_structure(seed=7, hidden=64, rank=8):
    """Task-structure tables from a synthetic checkpoint state through the real
    export, with domain and lineage structure planted (outcomes read alike
    within a domain, diagnoses are corrected alike within a lineage)."""
    from opera.run.export_opera_task_structure import task_structure
    from opera.visualization.paper_taxonomy import fine_cohort_groups

    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups = yaml.safe_load((REPO / "opera/configs/outcome_display_groups_60_candidate_v1.yaml").read_text())
    family = {o: group for group, members in groups.items() for o in members}
    outcomes = list(family)
    lineage = fine_cohort_groups()
    diseases = list(lineage)
    domains, lineages = list(groups), sorted(set(lineage.values()))
    events = {o: int(rng.lognormal(6.2, 1.3)) + 3 for o in outcomes}
    size = {d: int(rng.lognormal(7.0, 1.2)) + 20 for d in diseases}
    domain_query = torch.randn(len(domains), hidden, generator=g)
    queries = torch.stack([1.1 * domain_query[domains.index(family[o])] + torch.randn(hidden, generator=g)
                           for o in outcomes])
    lineage_down = torch.randn(len(lineages), rank, hidden, generator=g)
    disease_down = torch.stack([0.9 * lineage_down[lineages.index(lineage[d])]
                                + torch.randn(rank, hidden, generator=g) for d in diseases])
    domain_down = torch.randn(len(domains), rank, hidden, generator=g)
    outcome_down = torch.stack([0.6 * domain_down[domains.index(family[o])]
                                + torch.randn(rank, hidden, generator=g) for o in outcomes])
    log_events = torch.tensor([np.log10(events[o]) for o in outcomes], dtype=torch.float32)
    domain_up = torch.randn(len(domains), hidden, rank, generator=g)
    outcome_up = torch.stack([0.7 * domain_up[domains.index(family[o])]
                              + torch.randn(hidden, rank, generator=g) for o in outcomes])
    outcome_up = outcome_up * (0.02 + 0.012 * log_events)[:, None, None]
    log_size = torch.tensor([np.log10(size[d]) for d in diseases], dtype=torch.float32)
    lineage_up = torch.randn(len(lineages), hidden, rank, generator=g)
    disease_up = torch.stack([0.5 * lineage_up[lineages.index(lineage[d])]
                              + torch.randn(hidden, rank, generator=g) for d in diseases])
    disease_up = disease_up * (0.015 + 0.01 * log_size)[:, None, None]
    gates = torch.atanh(torch.clamp(0.025 * log_events + torch.tensor(rng.normal(0, 0.02, len(outcomes)),
                                                                     dtype=torch.float32), 0.003, 0.3))
    state = {"outcome_attention_readout.queries": queries,
             "outcome_attention_readout.attention.in_proj_weight": torch.eye(hidden).repeat(3, 1),
             "outcome_attention_readout.residual_gates": gates,
             "attention_own_corrections.outcome_down": outcome_down,
             "attention_own_corrections.outcome_up": outcome_up,
             "attention_own_corrections.disease_down": disease_down,
             "attention_own_corrections.disease_up": disease_up}
    tables, summary = task_structure(state, outcomes, diseases, outcome_family=family,
                                     disease_lineage=lineage, outcome_events=events,
                                     outcome_cases={o: max(1, events[o] // 6) for o in outcomes},
                                     disease_size=size)
    return tables, summary


def synthetic_correction_arrows(seed=11, n=12000, hidden=32):
    """Arrow aggregates from synthetic patient vectors through the real
    projection and binning (PCA map here; the server uses the paper's UMAP)."""
    from opera.run.build_paper_embedding_tables import fit_projection
    from opera.run.export_opera_correction_arrows import arrow_aggregates, project_parts

    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3.2, (7, 2))
    cluster = rng.integers(0, len(centres), n)
    latent = centres[cluster] + rng.normal(0, 1.0, (n, 2))
    lift = rng.normal(size=(2, hidden))
    shared = latent @ lift + rng.normal(0, 0.35, (n, hidden))
    frame = pd.DataFrame({"subject_id": [str(i) for i in range(n)],
                          "split": np.where(np.arange(n) < n // 2, "tuning", "held_out"),
                          "embedding": list(shared)})
    projection, shown = fit_projection(frame, method="pca")
    held = shown.index.to_numpy()
    s = shared[held]
    names = ["Overall survival", "Infection diagnosis or antimicrobial use", "Neutropenia (G3+)",
             "Elevated ALT (G2+)", "Hospital admission"]
    read, final = [], []
    for k in range(len(names)):
        # Reading: a pull towards one region of the map (outcome-specific), small.
        target = centres[rng.integers(0, len(centres))] @ lift
        pull = 0.10 * (target - s) * (0.4 + 0.6 * rng.random())
        r = s + pull
        # Corrections: a per-diagnosis shift (cluster-specific direction), smaller.
        direction = rng.normal(size=(len(centres), hidden)) * 0.3
        f = r + direction[cluster[held]] + rng.normal(0, 0.02, s.shape)
        read.append(r)
        final.append(f)
    parts = {"shared": s, "read": np.stack(read, axis=1), "final": np.stack(final, axis=1)}
    before, after = project_parts(projection, parts, names)
    arrows, density = arrow_aggregates(before, after)
    meta = {"projection": projection.method, "outcomes": names, "n_patients": int(len(s)),
            "min_count": 20}
    return arrows, density, meta


def synthetic_gap_cells(rng):
    """``paired_cells.csv`` rows for J-X, OPERA-X and OPERA-J with saved summaries."""
    cells = _cells(rng)
    cells = cells.loc[cells.n_events_train >= 5].reset_index(drop=True)
    noise = 0.18 / np.sqrt(cells.n_events_train.to_numpy() + 5) + 0.004
    j_x = cells.endpoint_group.map(XGB_GAP).fillna(-0.02).to_numpy() + rng.normal(0, noise)
    o_j = (0.008 + cells.endpoint_group.map(DOMAIN_SHIFT).fillna(0.0).to_numpy() * 0.6
           + rng.normal(0, noise * 0.8))
    frames = []
    for code, gains in (("J-X", j_x), ("OPERA-J", o_j), ("OPERA-X", j_x + o_j)):
        part = cells.assign(contrast=code, metric="auc", gain=gains, lower=gains - 1.96 * noise,
                            upper=gains + 1.96 * noise)
        summaries = pd.DataFrame(_effect_rows(code, cells, gains, rng))
        group = summaries.loc[summaries.stratum == "endpoint_group"].set_index("stratum_value")
        part["group_mean"] = part.endpoint_group.map(group.gain)
        part["group_lower"] = part.endpoint_group.map(group.lower)
        part["group_upper"] = part.endpoint_group.map(group.upper)
        panel = summaries.iloc[0]
        part = part.assign(panel_mean=panel.gain, panel_lower=panel.lower, panel_upper=panel.upper)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def render(output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from opera.visualization import opera_design_figures as design
    from opera.visualization.layout_audit import audit_layout
    from opera.visualization.style import save_fig

    rng = np.random.default_rng(20261101)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, summary = synthetic_task_structure()
    arrows, density, meta = synthetic_correction_arrows()
    figures = {
        "fig_gain_sources": lambda: design.figure_gain_sources(synthetic_gain_sources(rng), mark=MARK),
        "supp_design_search": lambda: design.figure_design_search(synthetic_design_search(rng), mark=MARK),
        "supp_task_organisation": lambda: design.figure_task_organisation(
            tables["outcome_query_similarity"], tables["disease_correction_similarity"],
            tables["outcome_specialisation"], tables["disease_specialisation"], summary,
            outcome_correction_similarity=tables["outcome_correction_similarity"], mark=MARK),
        "supp_specialisation": lambda: design.figure_specialisation(
            tables["outcome_specialisation"], tables["disease_specialisation"], mark=MARK),
        "supp_correction_arrows": lambda: design.figure_correction_arrows(arrows, density, meta, mark=MARK),
        "supp_xgboost_gap": lambda: design.figure_xgboost_gap(synthetic_gap_cells(rng), mark=MARK),
    }
    report = {}
    for name, build in figures.items():
        fig = build()
        save_fig(fig, str(output_dir / f"{name}.png"))
        report[name] = audit_layout(fig)
        plt.close(fig)
    (output_dir / "layout_report.json").write_text(json.dumps(report, indent=2, default=str))
    write_index(output_dir, list(figures))
    return report


NOTES = {
    "fig_gain_sources": "Main-text candidate. Direct paired contrasts on a shared zero axis, "
                        "with cell-level distributions, headline-style summaries and explicit reference models.",
    "supp_design_search": "Replaces supp_routing. Every tuned design against J_OA_OWN and R_AB "
                          "(tuning data), seed band of the selected design.",
    "supp_task_organisation": "Replaces fig_borrowing_map. Query similarity, outcome-correction and "
                              "diagnosis-correction similarity with within/between permutation tests.",
    "supp_specialisation": "Replaces supp_router_usage. Gate and correction size against training "
                           "events, by domain; diagnosis corrections by disease group.",
    "supp_correction_arrows": "Replaces supp_router_arrows. Binned mean displacement on the shared map "
                              "(synthetic: PCA map; server: the paper's UMAP map).",
    "supp_xgboost_gap": "Figure plan C4. Gap to per-task XGBoost and OPERA's gain over J by domain.",
}


def write_index(output_dir: Path, names) -> None:
    items = "\n".join(
        f'<section><h2>{html.escape(n)}</h2><p>{html.escape(NOTES.get(n, ""))} '
        f'<a href="{n}.pdf">PDF</a></p><img src="{n}.png" alt="{html.escape(n)}"></section>'
        for n in names)
    (output_dir / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>New OPERA figures: synthetic proofs</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 16px;color:#17212B}}
img{{max-width:100%;border:1px solid #CBD2DA}}h2{{font-size:16px;margin-top:2.2em}}
.note{{background:#F3F5FC;padding:.6em 1em;border-left:3px solid #2D3A8C}}</style>
<h1>New OPERA figures: synthetic design proofs</h1>
<p class="note">Every image is a SYNTHETIC DESIGN PROOF: invented inputs with the structure of the
real aggregates, run through the real builders. Nothing here is a result.</p>
{items}
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.output_dir)
    print(json.dumps({name: len(item.get("text_collision_candidates", [])) if isinstance(item, dict) else item
                      for name, item in report.items()}, indent=2))


if __name__ == "__main__":
    main()
