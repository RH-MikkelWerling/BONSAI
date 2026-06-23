# BONSAI / OPERA — Project State Report
*Prepared for supervisor presentation — June 2026*

---

## Prompt for LLM Presentation Generation

> You are creating a 15–20 minute supervisor presentation from the technical report below. The audience is a research supervisor with biomedical background but no need to understand every engineering detail. Structure the presentation as slides with a title, 3–5 bullet points each, and speaker notes. Use the section headings as slide titles. Where there are technical details, convert them to scientific plain language. Emphasize progress, what has been proven to work, and what the current work is setting up.

---

## 1. What Is This Project?

**BONSAI** (the base layer) is a transformer-based EHR foundation model trained on large national registry data. It processes raw electronic health record events — diagnoses, medications, lab values, procedures — encoded as timestamped token sequences and pre-trains a deep transformer (ModernBERT) using masked language modeling on those sequences.

**OPERA** (the hematology-specific layer, built on top of BONSAI) adapts that general EHR model for hematological cancers. The central scientific claim is: *domain-specific adaptation and cross-disease contrastive learning produce reusable patient representations, with the largest benefit observed in small cohort-outcome cells* (rare diseases or rare events).

**Diseases covered**: DLBCL, CLL, myeloma, AML, follicular lymphoma, and ~20 other hematological diagnoses.

**Target outcomes**: 1-year mortality, treatment failure, remission, progression — predicted from the patient's EHR record at the time of diagnosis, before outcome is known.

**Why it matters**: Clinical prediction for rare hematological cancers is difficult precisely because outcome data is scarce. OPERA is designed to show that cross-disease representation learning transfers signal from data-rich to data-poor situations.

---

## 2. Architecture Overview

The system has two namespaces/layers:

| Layer | What it does |
|-------|-------------|
| **BONSAI** | Core pipeline: MEDS data ingestion → tokenization → pre-training → outcome creation → fine-tuning infrastructure |
| **OPERA** | Hematology specialization: domain adaptation → contrastive learning → joint fine-tuning → evaluation → results aggregation |

**Model stack (four training stages)**:

1. **General pre-training** (BONSAI): ModernBERT trained with masked language modeling on all-cause EHR sequences from the Danish national registries.
2. **Domain-adapted pre-training (DAPT)**: Continue pre-training on hematology patients only — the model learns the vocabulary of blood cancer.
3. **Contrastive learning (OPERA)**: A survival-informed SupCon objective trains the model to embed patients with similar outcome trajectories closer together, using Kaplan-Meier cumulative event mass to weight pairs.
4. **Fine-tuning**: Per-task or joint (multi-outcome) classifiers on top of the learned representations.

**Key innovation**: The contrastive stage uses survival time — not just event/no-event — to weight patient pairs. This avoids treating a 6-month survivor the same as a 6-year survivor, which is scientifically important for censored clinical data.

**Frameworks**: PyTorch + PyTorch Lightning (training), Hydra-Core (config composition), Polars/Pandas (data), scikit-learn/scipy (metrics and survival analysis).

---

## 3. Data Pipeline

```
Raw EHR events (MEDS-format shards)
        ↓
  bonsai.run.create_data
        → Tokenized patient sequences (.pt files)
        → Vocabulary mapping
        ↓
  bonsai.run.create_outcome
        → Outcome parquet files (one per disease-outcome pair)
        → Eligibility sidecars (per-patient audit trail)
        → population_full.csv (demographics, IPI scores)
        ↓
  Training stages (4 sequential stages above)
        ↓
  opera.run.evaluate / evaluate_joint
        → metrics.json (AUROC, AUPRC, calibration, C-index)
        → predictions.npz (probabilities, embeddings)
        → plots/ (ROC, calibration, UMAP embeddings)
        → result.jsonl (canonical aggregation input)
        ↓
  opera.run.aggregate_results
        → Cross-disease comparison tables
        → Rarity analysis (synthetic + real)
        → Paper-ready figures
```

**Patient representation** at the core: each patient is a variable-length sequence of tokens (medical codes) with timestamps (hours since epoch), age, and visit segmentation. The model sees the whole longitudinal history up to the index date.

**Prospective split discipline**: the system enforces that no information after the index date leaks into training — eligibility, registry start dates, and competing events are all handled explicitly per outcome.

---

## 4. Evaluation Framework

Evaluation is rigorous and multi-dimensional:

**Discrimination**
- AUROC, AUPRC with bootstrap confidence intervals (stratified by event)
- DeLong test for pairwise comparison between models

**Calibration**
- Brier score, calibration slope (logistic recalibration)
- Decision curve analysis

**Survival metrics**
- Concordance index (C-index)
- IPCW-weighted AUC (accounts for censoring)

**Two evaluation regimes per outcome**:
- *Fixed-horizon*: only patients with full follow-up → clean binary classification
- *Survival*: all patients → survival metrics with censoring

**Subgroup analysis**: metrics broken down by demographic and clinical subgroups (IPI score tiers, age, sex) for fairness and clinical relevance.

**Rarity analysis**: Both synthetic (subsample training labels) and real (natural cohort-size variation) rarity experiments characterize how performance degrades with fewer training events — this is key to the central claim.

---

## 5. Cohort Design

**Training**: The model is trained on a *grouped* cohort architecture — 11 disease groups (e.g., "aggressive B-cell lymphoma") that aggregate multiple fine diagnoses.

**Evaluation**: Evaluation uses *fine-grained* diagnoses — 26 individual disease labels (e.g., DLBCL, primary mediastinal B-cell lymphoma, etc.).

This train-coarse/eval-fine structure tests whether representations learned on grouped data transfer to individual diseases with limited data — directly testing the generalization claim.

**Eligibility**: Each cohort-outcome pair has explicit inclusion/exclusion criteria encoded in eligibility sidecars:
- Registry coverage requirement (data source must predate the index date)
- Minimum baseline period
- Minimum post-index follow-up
- Competing event exclusion (e.g., prior history of same cancer)

---

## 6. What Remains / Next Steps

Based on the current branch state and recent commit history:

1. **Complete the cohorts.py integration** — `evaluate_joint.py` and the sweep orchestrator still use the old inline pattern and need to be updated.
2. **Leukemia sweep** — run the full train-coarse/eval-fine experiment grid across all 26 fine diagnoses × all outcomes × all model variants.
3. **Rarity analysis** — collect real and synthetic rarity results across all cohorts for the central paper claim.
4. **Paper figures** — aggregate results, generate rarity delta plots, embedding projections, and comparison tables.
5. **Final audit** — the recent "massive audit" commit fixed known issues; a final pass through eligibility and registry filtering is planned.

---

## 7. Technical Summary for Slides

| Dimension | Detail |
|-----------|--------|
| **Model** | ModernBERT-based EHR transformer |
| **Pre-training data** | Danish national EHR registries (all-cause) |
| **Adaptation** | DAPT → contrastive → fine-tune (4-stage) |
| **Key innovation** | Survival-informed contrastive learning |
| **Diseases** | ~26 hematological diagnoses |
| **Outcomes** | 1y mortality, treatment failure, remission, progression |
| **Evaluation** | AUROC, AUPRC, C-index, calibration, rarity analysis |
| **Test suite** | 228 passing tests, 47 modules |
| **Current focus** | Consolidate evaluation cohort logic; run leukemia sweep |
| **Branch** | `opera/leukemia` |

---

## 8. Key Figures / Visuals to Include in Presentation

Suggest including these in slides (all generatable from the codebase):
- **Pipeline diagram**: MEDS → tokenization → 4-stage training → evaluation
- **Architecture diagram**: BonsaiEncoder with EhrEmbeddings → contrastive projections → task heads
- **Rarity curve**: Performance vs. training set size (the central paper claim)
- **UMAP embeddings**: Before/after contrastive adaptation, colored by disease and outcome
- **Results table**: AUROC by cohort × model variant (BONSAI vs. DAPT vs. OPERA vs. tabular baseline)

---

*Report generated from repository at `c:\Users\MWER0040\Documents\repositories\bonsai\BONSAI`, branch `opera/leukemia`, commit `1bc1cad`.*
