# BONSAI / OPERA — Project State Report
*Prepared for supervisor presentation — July 2026*

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

1. **General pre-training** (BONSAI): a transformer trained with masked/autoregressive language modeling on all-cause EHR sequences from the Danish national registries.
2. **Domain-adapted pre-training (DAPT)**: Continue pre-training on hematology patients only — the model learns the vocabulary of blood cancer.
3. **Contrastive learning (OPERA)**: A survival-informed SupCon objective trains the model to embed patients with similar outcome trajectories closer together, using Kaplan-Meier cumulative event mass to weight pairs.
4. **Fine-tuning**: Per-task or joint (multi-outcome) classifiers on top of the learned representations.

**Key innovation**: The contrastive stage uses survival time — not just event/no-event — to weight patient pairs. This avoids treating a 6-month survivor the same as a 6-year survivor, which is scientifically important for censored clinical data.

**Architecture rewrite (new since last report)**: The encoder has moved off the third-party ModernBERT/HuggingFace implementation onto a **native, from-scratch BONSAI transformer** (`bonsai-native-rope-v1`) with rotary position embeddings and FlashAttention 2 as the default attention backend (SDPA remains available as a portable CPU/non-CUDA fallback, and the two are tested for numerical equivalence). Tokens are processed packed/variable-length rather than padded, saving compute. This is a hard migration boundary, not a drop-in upgrade: old ModernBERT-era checkpoints are explicitly rejected at load time with a clear "retrain" error rather than silently mismatching — all existing pre-trained checkpoints upstream of this change need to be regenerated under the native architecture before hematology adaptation/evaluation can continue on them.

**Frameworks**: PyTorch + PyTorch Lightning (training), Hydra-Core (config composition), Polars/Pandas (data), scikit-learn/scipy (metrics and survival analysis). FlashAttention 2 is now a first-class (optional) dependency on Linux/CUDA (requires GCC ≥9 to build); Bayesian rarity modeling adds PyMC/ArviZ as an optional dependency group.

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

**Rarity analysis**: The primary analysis now uses natural variation across
cohort-outcome cells. A robust (Student-t) Bayesian hierarchical spline models
paired OPERA-minus-tabular deltas against log training-event count, with
crossed cohort/outcome/outcome-family effects, partial pooling, explicit
patient-bootstrap uncertainty, and a training-seed variance component
(variance components drop out cleanly when only one level is observed).
Convergence (divergences, max R-hat) is a hard gate on publishing results.
The rarity plot itself was reworked: color encodes outcome family, marker
shape encodes disease-course grouping (e.g. aggressive vs. indolent/chronic
vs. plasma-cell), and cell labels are chosen for clinical relevance rather
than statistical extremeness. Synthetic label subsampling remains a
conditional label-efficiency sensitivity analysis, not the main emulation of
true disease rarity.

---

## 5. Cohort Design

**Training**: The model is trained on a *grouped* cohort architecture — 11 disease groups (e.g., "aggressive B-cell lymphoma") that aggregate multiple fine diagnoses.

**Evaluation**: Evaluation uses *fine-grained* diagnoses — 25 analyzed disease labels, with a separate explicit `EXCLUDE_SECONDARY` label kept out of the main evaluation.

This train-coarse/eval-fine structure tests whether representations learned on grouped data transfer to individual diseases with limited data — directly testing the generalization claim.

**Eligibility**: Each cohort-outcome pair has explicit inclusion/exclusion criteria encoded in eligibility sidecars:
- Registry coverage requirement (data source must predate the index date)
- Minimum baseline period
- Separate final fixed-horizon and censor-aware ascertainment eligibility
- Competing event exclusion (e.g., prior history of same cancer)

Fixed-horizon models exclude event-free patients censored before the horizon;
survival and contrastive objectives retain ascertainable patients for their
observed risk time. Event chronology is enforced against the follow-up censor
date, and every main or supplementary workflow receives the same registry,
eligibility, and competing-event inputs.

**Rare-outcome batching**: focused batches use distinct patients only and are
enabled only when an outcome has enough unique events and eligible controls.
Quotas never manufacture sample size through within-batch replacement;
cross-batch reuse is diversity-penalized, and the loss independently masks any
same-subject pair. Rare validation cases and controls are accumulated across
the full epoch.

---

## 6. What Remains / Next Steps

Based on the current branch state and recent commit history:

1. **Land the native-architecture attention-backend follow-up** — a focused, apparently-complete set of changes (currently uncommitted) makes the FlashAttention/SDPA backend independently selectable at checkpoint-load time (so CPU-only evaluation can load a checkpoint trained with FlashAttention), adds a guard against wiring autoregressive pretraining to non-causal attention, and finishes renaming the fine-tuning head for the native model class. All touched tests pass; this is close-out work, not a rewrite.
2. **Retrain/re-adapt checkpoints under the native architecture** — because old ModernBERT-era checkpoints are no longer loadable, any pre-trained/DAPT/contrastive checkpoints produced before the architecture rewrite need to be regenerated before downstream fine-tuning and evaluation can proceed on them.
3. **Complete the cohorts.py integration** — `evaluate_joint.py` and the sweep orchestrator still use the old inline pattern and need to be updated.
4. **Leukemia sweep** — run the full train-coarse/eval-fine experiment grid across all 25 evaluated fine diagnoses × all outcomes × all model variants, now on the native architecture.
5. **Hierarchical natural-rarity analysis** — collect patient-level prediction
artifacts across all evaluable cohort-outcome cells, validate exact comparator
parity, and fit the prespecified Bayesian spline hierarchy (model and plots
already reworked; execution across the full grid is what remains).
6. **Paper figures** — aggregate results, generate rarity delta plots, embedding projections, and comparison tables.
7. **Eligibility audit complete** — primary/competing-event chronology,
   fixed-horizon versus ascertainment masks, and supplementary workflow
   propagation now share one tested contract. The remaining work is execution
   of the locked experiment grid rather than another denominator rewrite.

---

## 7. Technical Summary for Slides

| Dimension | Detail |
|-----------|--------|
| **Model** | Native EHR transformer with RoPE + FlashAttention 2 (`bonsai-native-rope-v1`; replaces prior ModernBERT/HuggingFace encoder) |
| **Pre-training data** | Danish national EHR registries (all-cause) |
| **Adaptation** | DAPT → contrastive → fine-tune (4-stage) |
| **Key innovation** | Survival-informed contrastive learning; native architecture rewrite for speed/control |
| **Diseases** | 25 evaluated hematological diagnoses plus explicit secondary-cancer exclusion label |
| **Outcomes** | 1y mortality, treatment failure, remission, progression |
| **Evaluation** | AUROC, AUPRC, C-index, calibration, rarity analysis |
| **Test suite** | 521 passing tests, 71 modules |
| **Current focus** | Close out native-architecture integration; consolidate evaluation cohort logic; run leukemia sweep |
| **Branch** | `opera/leukemia` |

---

## 8. Key Figures / Visuals to Include in Presentation

Suggest including these in slides (all generatable from the codebase):
- **Pipeline diagram**: MEDS → tokenization → 4-stage training → evaluation
- **Architecture diagram**: Native BONSAI encoder (RoPE + FlashAttention) with EhrEmbeddings → contrastive projections → task heads
- **Hierarchical rarity curve**: raw cohort-outcome OPERA-minus-XGBoost deltas
  against training events, overlaid with posterior mean, credible, and
  new-cell predictive uncertainty bands
- **Patient embedding atlas**: Before/after contrastive adaptation, colored by disease, outcome, and first-line regimen
- **Vocabulary embedding atlas**: learned code-token geography and token movement from pre-training → DAPT → OPERA
- **Results table**: AUROC by cohort × model variant (BONSAI vs. DAPT vs. OPERA vs. tabular baseline)

---

*Report generated from repository at `c:\Users\MWER0040\Documents\repositories\bonsai\BONSAI`, branch `opera/leukemia`, commit `458c083`, with 23 uncommitted files representing a near-complete follow-up to the native-architecture merge.*
