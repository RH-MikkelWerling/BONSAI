# Manuscript gap plan (14 October 2026)

This file lists every missing item in `paper/opera_manuscript.tex`: the result
each one needs, the pipeline stage and output file that produces it, and
whether a script for it already exists. It is written for running jobs on the
secure GPU server (qsub). Launch every stage through
`bash scripts/run_paper_experiments.sh ACTION [--execute]`, written below as
`rpe ACTION`. Paths are relative to `$OPERA_PAPER_ROOT` (shortened to `$ROOT`).
`docs/run_paper_workflow_20261014.md` is the runbook. This plan says which of
its outputs fill which sentence.

Inventory: 31 `\placeholder{}` and 4 `\plotplaceholder{}{}` (35 items), plus
9 promised but unmarked items (U1–U9). The supplement
(`opera_data_definitions_supplement.tex`) has no placeholders.

## 0. Conventions used below

**Analysis runs.** The paper's primary metrics are the IPCW CIF AUC and Brier
score (Methods, "Evaluation and paired comparisons"). The secondary metrics are
the known-status metrics. They come from two separate analysis roots:

| Run | Settings | Fills |
| --- | --- | --- |
| **A-IPCW** (primary) | `OPERA_PAPER_ANALYSIS_ESTIMAND=cif_ipcw`, `OPERA_PAPER_ANALYSIS_ROOT=$ROOT/analysis/main365_cif_ipcw`, `OPERA_PAPER_BAYES_TEMPLATE=<locked template with estimand: cif_ipcw>` | Every headline AUC/Brier number, all figures, the Bayes panel |
| **A-KS** (secondary) | `OPERA_PAPER_ANALYSIS_ESTIMAND=binary_known_status_unweighted`, `OPERA_PAPER_ANALYSIS_ROOT=$ROOT/analysis/main365_known_status`, Bayes template with `estimand: binary_known_status_unweighted` | The "known status at the horizon" metrics (supplement table) and log loss under the training-objective population |
| **A-IPCW-6h** (supplement) | as A-IPCW, plus `OPERA_PAPER_ANALYSIS_HORIZONS=30,60,90,180,365,730` and root `$ROOT/analysis/allh_cif_ipcw` | Six-horizon stability ("Who benefits") and the supplementary atlas |

Set `OPERA_PAPER_ANALYSIS_ROOT` explicitly for every run. The dispatcher
defaults to `$ROOT/analysis/main365` whatever the estimand, and inventory and
analysis directories are immutable, so a second estimand run without a new
root fails. `bayesian_contrasts` rejects a recipe whose `estimand` differs from
`analysis_manifest.json`. The tracked Bayes template says
`binary_known_status_unweighted`, so the primary run needs a copy that says
`cif_ipcw`.

**Output abbreviations.** `PA/` = `<analysis root>/prediction_analysis/`
(the `rpe analyse` outputs). `BY/` = `<analysis root>/bayesian/<family>/<metric>/exchangeable/`
(`posterior_panel.csv`, `posterior_cells.csv`, `posterior_curve.csv`,
`diagnostics.csv`, `predictive_checks.csv`). The `support_spline` and
`support_bins` variants sit beside it. `INV/` = `<analysis root>/analysis_inventory/`
(`coverage.csv`, `analysis.yaml`).

**Contrast IDs** (`opera/configs/manifests/final_contrasts_20261014.yaml`):
pooling P1 D−S, P2 O−S, P3 J−O, P4 J−D, P5 J−S, P6 interaction; routing R0 N−J
(null-equivalence gate), R1 R−J, R2–R7 source banks; tabular T1 R−X, T2 J−X,
T3 S−X, T4 X−LR, T5 R_EA/J_EA−X; external E1 J_E−J_L, E2 R_E−R_L, A1 J_EA−J_E,
A2 R_EA−R_E, E3 J_EA−J_L, E4 R_EA−R_L, interactions I1–I3. Gains are oriented
so that a positive value favours the first model. For Brier and log loss the
gain is the reference loss minus the candidate loss.

**Script status.** YES = an existing command produces the number end to end.
PARTIAL = the command exists, but an input, a column or a step is missing.
NO = a script has to be written.

**Panel numbers: two weightings exist.** `PA/effect_summaries.csv` (`stratum=all_cells`)
weights cells equally. `BY/posterior_panel.csv` weights diseases and endpoint
groups equally (`raw_gain`, `raw_lower_95`, `raw_upper_95` = paired bootstrap
panel; `posterior_*` = partially pooled). Pick one as the headline before
held-out evaluation (Section 4) and quote it everywhere.

## 1. Summary table

Section codes: Abs = abstract, Int = introduction, R-Pop/R-Pool/R-Tab/R-Ext/R-Who =
Results subsections, Disc = discussion, M-* = Methods subsections, Fig = figure.

| # | Item (line) | Section | Needed result | Source stage / output | Script status | Manual input | Depends on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Authors and affiliations (L56) | Title | Author list, affiliations | — | n/a | Author / supervisor | — |
| 2 | Principal results (L75) | Abs | Panel 365-d IPCW AUC gain with 95% CI for P5 (and P1/P2), T2/T1 vs XGBoost, E3/A1 (J and R), plus the count of cells with a negative effect | A-IPCW `PA/effect_summaries.csv` or `BY/posterior_panel.csv` (families pooling, tabular, external) | YES once the runs exist | Choice of headline weighting | Items 12, 16, 17, 19 |
| 3 | One-sentence conclusion (L78) | Abs | Answer to the three questions | Written from 2 | n/a | Author | 2 |
| 4 | "Here we show" (L228) | Int | 2–3 sentences, one answer per question | Same numbers as 2 | n/a | Author | 2 |
| 5 | Median age, sex, largest disease groups (L287) | R-Pop | Median (IQR) age, % female, the 3–5 largest fine subtypes with n (%) | `opera.run.export_table1` → `release_review/table1_pooled_display.csv`, `table1_values_long.csv` | YES (text transcribed by hand) | Supervisor sign-off of Table 1 | 9 |
| 6 | N cells evaluable at one year (L289) | R-Pop | Number of the 24×69 fine-disease × outcome cells at 365 d that pass the estimability rule (held-out ≥ `min_cases` cases and ≥ `min_controls` controls, ≥80% finite draws) | A-IPCW `<analysis root>/bayesian_recipes/rosters/pooling_auc.csv` (and `pooling_auc_exclusions.csv`), or `PA/estimates.csv` `n_cases`/`n_controls` for model J_L | PARTIAL: counts exist, but no script reports this number and the structurally excluded cells | Locked estimability rule (Section 4) | Inventory and analysis (A-IPCW), prepare-bayes |
| 7 | Event-rate range (L290) | R-Pop | Min–max 365-d cumulative event rate across evaluable cells (state the split; training is safest) | `opera.run.audit_cohort_outcome_support --horizon-days 365` → `cohort_outcome_support.csv` `primary_event_rate_train` | PARTIAL: audit exists; no summariser; must use the approved release registry and the paper base config | Which split to quote | 13-NEW-a |
| 8 | N cells with <20 training events (L291) | R-Pop | Count of cells with `n_events_train < 20` at 365 d | Cell metadata (`OPERA_PAPER_CELL_METADATA`) | NO: there is no cell-metadata builder (see 13-NEW-a) | Bin edge 20 must match the locked support bins | 13-NEW-a |
| 9 | Table 1 — DONE 2026-10-20 (approved by the data owner) | R-Pop | `paper/tables/table1.tex` | `export_table1` then `format_table1_release --latex paper/tables/table1.tex` | YES | Supervisor review; disclosure review; reviewer, date and export hash recorded | Frozen supervised population (37,531) |
| 10 | **Fig. atlas** (`\plotplaceholder`, L326) | Fig | Held-out 365-d IPCW AUC of J_L for every outcome, grouped by clinical domain, with 95% CI and event support. Insets: outcome × cohort coverage, calibration. Supplement: six horizons and per fine cohort | A-IPCW `PA/estimates.csv` (model J_L, metric auc, `n_cases`), `PA/calibration.csv`, `INV/coverage.csv`, cell metadata. Six-horizon version from A-IPCW-6h | PARTIAL: `plot_neural_predictability_prototype.py` plots tuning CIF-AUC from archived training logs, not locked held-out estimates. `estimates.csv` has no CI for absolute metrics. Figure code in progress in `opera/visualization/paper_figures.py` | Domain grouping (`outcome_display_groups_*`) | Analysis A-IPCW; 13-NEW-b (absolute CIs) |
| 11 | Atlas summary text (L338) | R-Pop | Outcome families with high and low J_L AUC at 1 y; variation across diseases | As 10 (absolute J_L metrics, not contrasts) | PARTIAL (as 10) | — | 10 |
| 12 | Pooling headline (L345) | R-Pool | Panel 365-d IPCW AUC gains with 95% CI: P1 (D−S) = X, P2 (O−S) = Y, P5 (J−S) = Z, the P6 interaction, and the Brier gains alongside | A-IPCW `BY/pooling/{auc,brier}/exchangeable/posterior_panel.csv` (raw_* and posterior_*), `PA/effect_summaries.csv` (all_cells). Known-status counterpart from A-KS for the supplement | YES: `rpe analyse --execute`, `rpe prepare-bayes --execute`, `rpe bayes --execute` | Headline weighting; practical tolerance for P(benefit/harm/equivalence) | Supervision queue (S, D, O, J) done; evaluate held-out; inventory |
| 13 | Where pooling helped or harmed (L349) | R-Pool | P1–P6 by disease, outcome family, endpoint group, training-event bin; count and list of cells with a negative gain | `PA/effect_summaries.csv` (strata `cohort`, `outcome_family`, `endpoint_group`, `support_bin`), `PA/contrasts.csv`, `BY/pooling/*/support_spline/posterior_curve.csv`, `support_bins` variant | PARTIAL: needs cell metadata with `outcome_family`, `endpoint_group`, `n_events_train` (13-NEW-a) | Locked support-bin edges | 13-NEW-a, 12 |
| 14 | N cells without a specialist; their pooled performance (L351) | R-Pool | Count of cells where S_L has no usable prediction (no training events, single-class training, failed fit), and the D/O/J AUC in those cells, reported separately | `INV/coverage.csv` (S_L `evaluation_missing`), `PA/model_coverage.csv`, `PA/contrast_coverage.csv`, `neural_runs/*/worker_failed.json` | PARTIAL: coverage shows "missing" but not why. No script classifies the reasons or tabulates pooled-model AUC in those cells | — | 13-NEW-c |
| 15 | **Fig. pooling** (`\plotplaceholder`, L356) | Fig | A: D−S and O−S cell maps. B: J−O, J−D and the interaction. C: gain vs training-event support (raw cells, curve, heterogeneity) | A-IPCW `PA/contrasts.csv`, `BY/pooling/*/posterior_cells.csv`, `posterior_curve.csv` | PARTIAL: `bayesian_contrast_plots.plot_contrast_results` draws a diagnostic curve per recipe. Publication figure in progress: `opera/visualization/paper_figures.py` | — | 12, 13 |
| 16 | Routing paragraph (L369) | R-Pool | (a) R0 gate result, (b) R1 (R−J) panel gain with CI, (c) selected banks and the tuning evidence, (d) R1 within the tuning-defined P5-negative stratum (rescue), plus direct R−S | `PA/router_null_equivalence.csv` (R0), `BY/routing/*/posterior_panel.csv` (R1), `BY/source_banks/*` (R2–R7, only if R0 passes), `PA/negative_transfer.csv`, `negative_transfer_summary.csv`, `PA/contrasts.csv` extra contrast `router_vs_specialist`, `$ROOT/tuning_summary_stage*_*/matched_readout_scores.csv`, router lock JSON | YES (all tables exist) | R0 tolerance per metric; router lock JSON | Stage 2 `final_sources` (N, RD, RO, RDO, R) evaluated |
| 17 | Tabular paragraph (L377) | R-Tab | T2 (J−X), T1 (R−X), T3 (S−X) panel AUC and Brier gains; T4 (X−LR); calibration (ECE, signed bias); the cells where X wins | A-IPCW `BY/tabular/*/posterior_panel.csv`, `PA/effect_summaries.csv`, `PA/contrasts.csv` (metrics auc, brier, ece, calibration_bias), `PA/calibration.csv` | YES | — | Tabular queue done; inventory joins X/LR |
| 18 | **Fig. tabular** (`\plotplaceholder`, L383) | Fig | A: absolute and paired 365-d performance (X, S, J, R). B: heterogeneity and coverage. C: Brier, log loss, calibration | As 17, plus `PA/estimates.csv`, `INV/coverage.csv` | PARTIAL: figure in progress in `paper_figures.py`; absolute CIs missing (13-NEW-b). The caption's "Registry/IPI complete-case results" (XR) have no pipeline (13-NEW-g) | Decide whether XR stays | 17 |
| 19 | External paragraph (L399) | R-Ext | E1/E2 (E−L), A1/A2 (EA−E), E3/E4 (EA−L) for J and R; I1–I3; dependence on disease, rarity and vocabulary coverage | `BY/external/*/posterior_panel.csv`, `BY/external_tabular/*` (T5), `PA/effect_summaries.csv` by `cohort`/`support_bin`, `PA/domain_mismatch.csv` | PARTIAL: contrasts YES; "how well the external vocabulary covered our records" needs a per-patient/per-cell coverage covariate, which no script produces (13-NEW-d) | — | export-external, dapt, stage 2 `--include-external`, `external` phase evaluated |
| 20 | **Fig. encoder transfer** (`\plotplaceholder`, L405) | Fig | L/E/EA × J/R matrix with E−L, EA−E, EA−L; inset R−J by origin (I1–I3) | As 19 | PARTIAL: figure in progress in `paper_figures.py` | — | 19 |
| 21 | Who benefits (L417) | R-Who | (a) patient Brier/log-loss gains by pre-index history, age, sex; (b) six-horizon stability; (c) calendar stability; (d) tuning-selected best-of-S/D/O vs J on held-out; (e) computational cost | (a) `PA/patient_subgroups.csv`; (b) A-IPCW-6h `PA/contrasts.csv`, `effect_summaries.csv`; (c) `PA/calendar_heterogeneity.csv`; (d) `PA/strategy_selection.csv`; (e) `PA/cost_performance.csv` | PARTIAL: (a)/(c) need the patient-metadata file (13-NEW-e); (e) needs a compute inventory that nothing produces (13-NEW-f). (b), (d) YES | Locked subgroup bins | 13-NEW-e, 13-NEW-f, A-IPCW-6h |
| 22 | Discussion principal findings (L423) | Disc | (i) pooling + interaction, (ii) vs XGBoost, (iii) external + DAPT, (iv) the strongest negative/null result | Items 12–21 | n/a | Author | 12–21 |
| 23 | Conclusion paragraph (L552) | Disc | Recommendation per question | Items 12–21 | n/a | Author | 22 |
| 24 | EHR inputs "were/were not" (L571) | M-Data | Whether the two-region EHR (e.g. EHR laboratory) data are in the model vocabulary | Sequence data provenance (`OPERA_SEQUENCE_DATA_DIR` vocabulary / data-creation config) | NO script; read off the data-creation config | Data team / author confirms | — |
| 25 | Data extraction date (L585) | M-Data | Date of the DALY-CARE extraction | Release snapshot manifest (`snapshot_manifest_INTERNAL.json`) or the data-management record | n/a | Data manager | — |
| 26 | Pretraining individuals, events, budget, selected epoch (L627) | M-Encoder | N individuals and N events in the pre-2022 SSL train/val split; epochs and optimizer steps, tokens seen, GPU hours; validation-selected epoch | `<pretrain run>/checkpoint_metadata.json` `extra.completed_epochs`, `optimizer_steps`, `training_tokens_seen`, `training_examples_seen`; `best.ckpt["epoch"]` (+1); Lightning `metrics.csv` (`val/loss`) | PARTIAL: budget YES. Corpus N individuals/events: no exporter (13-NEW-h). Selected epoch: not in a sidecar, read `torch.load(best.ckpt)["epoch"]`. GPU hours not recorded | — | Pretraining finished |
| 27 | Selected router configuration (L680) | M-Router | `source_mode`, `rank`, `query_mode`, `query_width`, update policy and epsilon, plus the tuning evidence and tolerance used | `OPERA_PAPER_ROUTER_LOCK` JSON; `$ROOT/tuning_summary_stage1_{source,rank,query}_screen/matched_readout_scores.csv` | YES (`rpe tuning-summary --phase … --execute`) | Router decision (author) recorded in decisions and lock JSON | Source/rank/query screens done |
| 28 | External corpus size (L722) | M-External | N individuals and events in the external BONSAI pretraining corpus (pre-2022 cutoff); its training budget | Collaborators (FGA-DIKU). The export gives only vocabulary sizes: `$ROOT/external_direct/checkpoint_metadata.json` `source_pretraining_transfer` | n/a (external fact) | Collaborators: corpus counts, cutoff date, commit (RoPE layout), segment convention | — |
| 29 | Practical tolerance and comparison family (L754) | M-Selection | Per-metric tolerance (AUC, Brier, log loss) and the multiplicity family for formal claims | Analysis lock; to be copied into `final_contrasts_20261014.yaml` (`practical_tolerance`, `multiplicity_specification`, `router_null_equivalence.tolerance`), analysis template, Bayes template | n/a (decision) | Author + supervisor/statistician, **before held-out evaluation** | — |
| 30 | Priors, tolerances, support bins, sampling diagnostics (L805) | M-Hierarchy | Locked `prior_scale`, `random_effect_scale`, `slope_scale` and `practical_tolerance` per metric; support-bin edges; sampler settings; max R-hat, min ESS, divergences, BFMI | Locked Bayes template (`metric_settings`, `support_bins`, `sampling`); `BY/*/diagnostics.csv`, `predictive_checks.csv`, `__prior_check/prior_panel.csv` | YES for diagnostics (`rpe bayes --prior-only --execute`, then `rpe bayes --execute`). Values are decisions | Locked template | Analysis A-IPCW |
| 31 | Data availability (L818) | Back | Controlled-access statement | — | n/a | DALY-CARE steering / Danish Health Data Authority wording | — |
| 32 | Code availability (L820) | Back | Repository URL, archival DOI, license, environment | Git tag + Zenodo DOI; `pyproject.toml` | n/a | Author creates the release | Final code freeze |
| 33 | Acknowledgements (L822) | Back | Funding, collaborators, compute | — | n/a | Author (grant numbers, server/HPC name) | — |
| 34 | Author contributions (L824) | Back | CRediT statement | — | n/a | Authors | — |
| 35 | Competing interests (L826) | Back | Statement | — | n/a | Authors | — |

### Promised in the text but not marked as placeholders

| # | Promise (location) | Needed result | Source | Status |
| --- | --- | --- | --- | --- |
| U1 | Cohort flow caption: "Draft counts await reconciliation and supervisor sign-off"; 26 patients without supervised membership (Fig. cohort-flow) | Reconciled counts, explanation for the 26 | `opera.run.summarize_cohort_flow`, ascertainment R scripts | YES for counts; reconciliation is manual (data team + supervisor) |
| U2 | M-Data: "We report the number of patients assigned through this reassessment and a sensitivity analysis excluding them" | N assigned in the 90-day pass; paired contrasts excluding them | Ascertainment output (pass flag); A-IPCW `PA/identity_sensitivity_counts.csv`, `PA/late_diagnosis_exclusion_sensitivity.csv` (needs `index_date` and `cohort_identity_evidence_date` in patient metadata) | PARTIAL: depends on 13-NEW-e |
| U3 | M-Supervised: "Equal numbers of epochs do not equate … steps or computational cost; we report both" | Optimizer steps and GPU hours per fit, by scope | Compute inventory | NO (13-NEW-f) |
| U4 | M-Hierarchy: "convergence and posterior predictive checks are reported" | Supplementary table | `BY/*/diagnostics.csv`, `predictive_checks.csv` | YES |
| U5 | M-Evaluation: "We additionally report the corresponding metrics among patients with known status" | Known-status AUC/Brier for every headline contrast | Run A-KS | YES |
| U6 | M-Evaluation: "One year is the main display horizon; all six are reported" | Six-horizon tables and atlas | Run A-IPCW-6h | YES for tables; figure in progress |
| U7 | M-Tabular: "Comparators using registry-specific prognostic variables were evaluated separately on patients with complete data" | XR (registry-enriched XGBoost) complete-case results | none | NO (13-NEW-g): implement or delete the sentence and the Fig. 3 caption clause |
| U8 | Fig. cohort-flow caption: "A sensitivity analysis will restrict routed disease labels to those available at treatment" | As U2 | As U2 | PARTIAL |
| U9 | M-External: E/EA arms from "the identical mapped initialisation"; vocabulary coverage | Matched tokens, local token coverage, new tokens | `$ROOT/external_direct/checkpoint_metadata.json` → `source_pretraining_transfer.{matched_tokens, local_non_special_token_coverage, new_local_tokens}` | YES (type-level coverage only; event-weighted coverage under 13-NEW-d) |

## 2. Missing scripts and work to do (priority order)

Each entry below was checked in the working tree on 14 October.

1. **13-NEW-f: compute inventory. Time-sensitive: do this before the stage-1 queue starts.**
   `analyse_locked_predictions` accepts `compute_inventory` (columns `model`,
   `seed`, optional `cohort`/`outcome`, plus GPU hours, steps, parameters,
   throughput) and writes `cost_performance.csv`. No script produces that
   file. `paper_worker.Claim` records `claimed_at` but deletes the claim on
   release. `training_complete.json` records only the fingerprint and
   checkpoint selection. Pretraining `checkpoint_metadata.json` has steps and
   tokens but no wall time. If the worker does not record timing now, GPU
   hours can only be reconstructed from file mtimes. Work:
   (a) have `paper_worker.py` write `run_timing.json` per fit (start, end,
   host, GPU name, attempts). (b) Write `opera/run/paper_compute_inventory.py`.
   It walks `neural_runs/` and the stage bundles and emits, per canonical
   alias and seed (and per cell for S/D/O), wall-clock GPU hours, optimizer
   steps (`best.ckpt["global_step"]` and the last checkpoint), trainable
   parameter count (from `best.ckpt` state_dict), and inference time from
   `evaluation_complete.json`. Tabular fit time comes from the tabular
   markers. It then sets `compute_inventory` in the analysis template. Fills
   item 21(e) and U3.
2. **13-NEW-a: cell metadata builder** (`OPERA_PAPER_CELL_METADATA`). This
   file is required by `rpe inventory` and the Bayes recipes, and by items 7,
   8 and 13. It needs one row per (fine cohort, outcome, horizon) with
   `n_events_train`, `n_train`, `endpoint_group`, `outcome_family` (and
   `cohort_group` for the nesting sensitivity). `audit_cohort_outcome_support.py`
   produces the counts (`n_primary_events_train`, `n_survival_train`,
   `primary_event_rate_train`), but only one horizon per call, under other
   column names. Endpoint groups and families are in the base config's
   `cross_outcome`. Work: write `opera/run/paper_cell_metadata.py`. It calls
   `audit_support` for all six horizons with `--train-tuning-only`, renames
   the columns, joins the groups and families, and writes the CSV plus a
   `panel_support_summary.json`. That JSON holds item 8 (cells <20 training
   events), item 7 (event-rate range), and the count of structurally excluded
   cells.
3. **13-NEW-e: patient metadata builder** (`OPERA_PAPER_PATIENT_METADATA`).
   `rpe inventory` requires this file. It feeds `patient_subgroups.csv`,
   `calendar_heterogeneity.csv`, the router identity audit and the
   late-diagnosis sensitivity. It needs one row per supervised subject:
   `subject_id`, `cohort_fine`, `age_at_index`, `sex`, `history_event_count`
   (pre-index, background tokens excluded), `index_year`, `index_date`,
   `cohort_identity_evidence_date` (and `cohort_source`). No script writes
   this file. `export_table1.py` already computes the age and MEDS history
   summaries but writes only aggregates. Work: write
   `opera/run/paper_patient_metadata.py`, reusing `export_table1`'s
   `join_metadata` and `summarize_meds_batches`. Fills 21(a)(c), U2 and U8.
4. **13-NEW-b: absolute metric intervals.** `secondary_analysis.paired_analysis`
   computes per-model bootstrap scores (`scores_by_draw`) but writes
   `estimates.csv` without intervals. The atlas (item 10) and Fig. 3A need
   absolute AUC and Brier with 95% CI. Work: add `lower_95`, `upper_95` and
   `n_valid_bootstrap` to the `estimates` rows from the same draws. This is a
   small edit in `opera/evaluation/secondary_analysis.py`.
5. **13-NEW-c: specialist coverage roster and reason codes.** S configs are
   generated for every structurally eligible cell
   (`prepare_supervision_factorial.narrow`), with no support filter. A cell
   with no training events therefore shows up as a failed fit
   (`worker_failed.json`) or as `evaluation_missing` in `INV/coverage.csv`,
   and the reason is lost. Work: write `opera/run/paper_coverage_report.py`.
   It joins `coverage.csv`, `worker_failed.json` errors, cell metadata
   (`n_events_train == 0`, single-class training/tuning) and
   `PA/contrast_coverage.csv`. It emits one reason per S_L cell
   (`no_training_events`, `single_class_training`, `failed_fit`,
   `not_run`, `evaluated`) and the D/O/J AUC in the non-S cells. Fills
   item 14 and item 6 (evaluable cell count).
6. **13-NEW-d: external vocabulary coverage covariate.**
   `export_external_encoder.py` records type-level coverage only
   (`matched_tokens`, `local_non_special_token_coverage`, `new_local_tokens`).
   Item 19 ("how well the external vocabulary covered our records") needs a
   pre-index, per-patient event-weighted coverage (share of a patient's
   pre-index events whose token is in the external vocabulary). Aggregated
   per cell, it goes into `domain_mismatch_covariates`. No script exists.
   Work: write `opera/run/external_vocab_coverage.py`. Inputs are
   `external_direct/vocabulary.pt`, the source vocabulary and the sequence
   data. Output is a column appended to patient metadata (declared
   `train_only`/`pre_index`, bins locked beforehand) and a cell-level
   summary.
7. **13-NEW-h: pretraining corpus summary.** No exporter reports the number of
   individuals and events (tokens) in the pre-2022 SSL train/val split.
   `checkpoint_metadata.json` gives `training_examples_seen` and
   `training_tokens_seen`, which are cumulative over epochs, not corpus size.
   Work: write a short `bonsai/run/summarize_pretraining_corpus.py` over
   `OPERA_SEQUENCE_DATA_DIR` (subjects, events, events per subject,
   truncation share at 8,192) that also reads `best.ckpt["epoch"]` and the
   `metrics.csv` best `val/loss`. Run it on the local corpus and on the DAPT
   run (EA budget). Fills item 26.
8. **13-NEW-g: registry-enriched comparator (XR).** It is listed in the
   manifest notation and in the Methods/Fig. 3 caption. `paper_tabular_queue`
   fits only `xgboost` and `logistic` on the EHR feature matrix.
   `opera/evaluation/rkkp_analysis.py` is an embedding analysis, not this
   comparator. Decide: either remove the XR sentence and caption clause, or
   write a complete-case XR queue (registry/IPI features, complete-case
   restriction applied to every compared model). Recommendation: delete it
   from the main text unless the registry features are already extracted.
9. **Figures (in progress: `opera/visualization/paper_figures.py`).** It must
   cover the atlas, pooling, tabular and encoder-transfer figures from the
   A-IPCW `PA/` and `BY/` tables, and a six-horizon supplementary atlas from
   A-IPCW-6h. `plot_neural_predictability_prototype.py` stays a tuning-log
   prototype and must not be used for the paper atlas. There is no dispatcher
   action for figures or Table 1. Run them as modules.
10. **Panel-number extractor (optional, recommended).** Write
    `opera/run/paper_numbers.py`. It reads the A-IPCW and A-KS `PA/` and `BY/`
    tables and writes `paper/build/numbers.tex` with `\newcommand` macros for
    every quoted number (items 2, 6–8, 12–21, 26). The text then cannot drift
    from the tables, and disclosure review covers one file.
11. **Route weights in the analysis recipe.** `rpe inventory` does not fill
    `route_weights`. Its value comes from the template, which is null by
    default. If routes are reported: run `rpe evaluate --stage 2 --split held_out --export-routes --execute`
    and set `route_weights: <predictions>/<R_L run>/held_out/source_route_weights.parquet`,
    `route_model: R_L`, `route_seed` in the template, before `rpe inventory`.
    Without these steps, `route_weights.csv` is reported as `missing_route_export`.

## 3. Order of operations to fill the paper

1. **Before any queue launch:** implement 13-NEW-f(a) (worker timing).
   Run `rpe check`, then `rpe prepare --execute` (stage 1).
2. **Pretraining completes:** confirm `checkpoint_metadata.json` and
   `training_complete`. Record item 26: budget from the sidecar, selected
   epoch from `best.ckpt`, corpus counts from 13-NEW-h.
3. **Smoke:** `rpe smoke --execute`.
4. **Worker queue:** in every qsub job, `WAIT=1 bash scripts/paper_gpu_worker.sh`.
   Monitor with `rpe queue-status`. This trains J, D, O, the source screen,
   then S.
5. **Tuning decisions:** run `rpe tuning-summary --phase source_screen --execute`,
   then write `{"source_mode": …}` to `OPERA_PAPER_DECISIONS`. Workers pick up
   the rank screen; run `rpe tuning-summary --phase rank_screen --execute`, add
   `rank`. Run the query screen and `rpe tuning-summary --phase query_screen --execute`.
   Write `OPERA_PAPER_ROUTER_LOCK` (source_mode, rank, query_mode,
   query_width, tolerance, evidence). This fills item 27.
6. **Stage 2 (prepare with the external arms from the start):** fill
   `OPERA_PAPER_EXTERNAL_SOURCE_CKPT`, `OPERA_PAPER_EXTERNAL_SOURCE_VOCAB`
   and the RoPE layout (ask the collaborators). Run `rpe export-external --execute`,
   record U9 from its metadata, then `rpe dapt --execute` (record its budget
   for item 26/28). Run `rpe prepare --stage 2 --include-external --execute`.
   Workers then run `final_sources` (N, RD, RO, RDO; R aliased from the lock)
   and `external` (J_E, R_E, J_EA, R_EA).
7. **Tabular:** run `rpe features --execute`, then set
   `OPERA_PAPER_FEATURE_MATRIX` and run
   `rpe tabular --shard-count K --shard-index i --execute` on CPU nodes. This
   can run in parallel with steps 4–6.
8. **Metadata inputs:** run 13-NEW-a (cell metadata), 13-NEW-e (patient
   metadata) and 13-NEW-d (coverage covariate). Set
   `OPERA_PAPER_CELL_METADATA`, `OPERA_PAPER_PATIENT_METADATA` and
   `OPERA_PAPER_DATA_CONTRACT`.
9. **Lock the analysis (Section 4):** commit the locked analysis template
   (`OPERA_PAPER_ANALYSIS_TEMPLATE`), both Bayes templates, and the manifest
   tolerances. Tag the commit. Steps 1–9 must not read any held-out file.
10. **Evaluate:** run `rpe evaluate --stage 2 --split tuning --execute`, then
    `rpe evaluate --stage 2 --split held_out --execute` (add `--export-routes`
    if routes are reported). Run `rpe status --stage 2` to confirm. Then run
    13-NEW-f(b) to build the compute inventory, and set it in the template.
11. **Inventory, both estimands:** with the A-IPCW env settings, run
    `rpe inventory --stage 2 --execute`. With the A-KS settings, run
    `rpe inventory --stage 2 --execute` again. Optionally do the same for
    A-IPCW-6h.
12. **Analyse:** run `rpe analyse --execute` under each setting. Check
    `PA/analysis_status.csv` and `PA/router_null_equivalence.csv` (R0) first.
    Then run 13-NEW-c (coverage reasons; items 6, 14).
13. **Bayes:** for each run, `rpe prepare-bayes --execute`, then
    `rpe bayes --prior-only --execute`, inspect the prior checks, then
    `rpe bayes --execute`. Read `recipe_status.csv` for `not_estimable`
    families and `diagnostics.csv` (item 30, U4). Source-bank results are
    interpreted only if R0 shows `equivalent_within_tolerance`.
14. **Figures:** run `paper_figures.py` (in progress) on the A-IPCW tables
    (items 10, 15, 18, 20), with the six-horizon supplement from A-IPCW-6h.
15. **Table 1:** run `export_table1` (inspect, then targeted disclosure), then
    `format_table1_release --latex paper/tables/table1.tex`. Done: approved
    by the data owner on 2026-10-20 (items 5, 9; record in the file header). This is independent of
    training, so do it as early as possible. Reconcile the cohort flow (U1)
    at the same time.
16. **Text:** run the numbers extractor (recommended item 10 in Section 2).
    Then write items 6–8, 11–14, 16, 17, 19 and 21, then 2–4 and 22–23. Add
    the back matter (items 1, 24, 25, 28, 31–35) and the disclosure review of
    every exported aggregate.

## 4. Decisions that must be locked before held-out evaluation

Write each one into the tracked file named here, commit it, and record the
commit hash in the paper's analysis lock. `final_contrasts_20261014.yaml`
currently has `practical_tolerance: null`,
`router_null_equivalence.tolerance: {auc: null, brier: null, log_loss: null}`
and `multiplicity_specification: pending_analysis_lock`.

| Decision | Where it is set | Note |
| --- | --- | --- |
| Primary metric and estimand | Manuscript Methods; `OPERA_PAPER_ANALYSIS_ESTIMAND=cif_ipcw` for the primary run; both Bayes templates (`estimand`) | The manifest's `secondary_analysis.estimand: binary_known_status_unweighted` describes the secondary run only. Also update the manifest and the stale workflow sentence (see "Inconsistencies") |
| Practical tolerance per metric (AUC, Brier, log loss) | `final_contrasts_20261014.yaml` `practical_tolerance`; Bayes template `metric_settings.<metric>.practical_tolerance` | Feeds item 29 and the P(benefit/harm/equivalence) in `posterior_*` |
| R0 null-equivalence tolerance | Manifest `router_null_equivalence.tolerance` and/or analysis template `router_null_equivalence.tolerance` | If null, R2–R7 are flagged `tolerance_not_locked`. The gate uses the `all_cells` interval in `effect_summaries.csv` |
| Router selection margin | Router lock JSON | Used at tuning time (step 5). Must be fixed before the source screen is read |
| Priors per metric | Bayes template `metric_settings.{auc,brier,log_loss}.{prior_scale,random_effect_scale,slope_scale}` | Check with `--prior-only` before the posterior fit |
| Support-bin edges | Analysis template `cell_groups.support_bin.bins`; Bayes template `support_bins` | **These currently disagree:** `[-1,9,24,99,499,inf]` vs `[0,20,100,500]`, and the Results text uses "<20". Use one set |
| Estimability rule | Bayes template `estimability` (`min_cases`, `min_controls`, `minimum_finite_fraction`); analysis template `rescue.min_cases`/`min_controls` | Defines item 6. Never depends on effect size |
| Headline aggregation weighting | Bayes template `weighting` (`equal_disease_endpoint_group`), with the equal-cell sensitivity in `effect_summaries.csv` | Decide which number goes in the abstract |
| Multiplicity family | Manifest `multiplicity_specification` | e.g. formal claims limited to panel-level P1, P2, P6, T2, E3, A1 at 365 d. All cell-level and subgroup results labelled exploratory |
| Subgroup bins | Analysis template `patient_groups` (age, history_event_count, index_year; coverage covariate bins for 13-NEW-d) | Pre-index or train-only provenance only |
| Seed roster | `OPERA_PAPER_SEEDS`; Bayes `seeds` | Intervals condition on the fitted seeds |
| Negative-transfer descriptive tolerance | Analysis template `rescue.tolerance` (0.0) | Tuning-defined strata only |
| Main horizon | 365 d (manifest) | Already fixed |

## 5. Inconsistencies found while compiling this plan

- `docs/run_paper_workflow_20261014.md` says the unified runner "targets
  unweighted known-status binary effects" and does not do IPCW. The code now
  supports `cif_ipcw` (`secondary_analysis.ESTIMANDS`, `attach_ipcw_weights`,
  `paper_analysis_inventory --estimand`), as documented in
  `locked_prediction_analysis_20261014.md`. The workflow doc and the
  manifest's `secondary_analysis.estimand` are out of date.
- The dispatcher's default analysis root does not depend on the estimand.
  Running both estimands needs explicit `OPERA_PAPER_ANALYSIS_ROOT` values
  (Section 0).
- Support bins disagree between the analysis and Bayes templates (Section 4).
- The tracked Bayes template has `estimand: binary_known_status_unweighted`.
  Prepared recipes inherit it, and `bayesian_contrasts` fails on a `cif_ipcw`
  analysis unless the template is changed.
- XR appears in the Methods and the Fig. 3 caption, but no pipeline produces
  it (13-NEW-g).
- Fit timing is not recorded anywhere, although the Methods promise compute
  reporting (13-NEW-f).

## Resolved since this plan was written (14 Oct 2026)

- **Compute inventory:** every worker attempt now appends wall time, host and
  GPU to `worker_attempts.jsonl` in the fit's run directory;
  `bash scripts/run_paper_experiments.sh compute-inventory` writes
  `compute_per_fit.csv` and `compute_by_scope.csv` (GPU hours by origin and
  supervision scope). Deploy before the stage-1 queue starts.
- **Estimand roots:** the default analysis root is now
  `analysis/main365_ipcw` or `analysis/main365_known_status` depending on
  `OPERA_PAPER_ANALYSIS_ESTIMAND`, so the two runs no longer collide.
- **Bayes estimand:** recipes take the estimand from the analysis manifest
  they read; one template serves both runs.
- **Support bins:** one partition everywhere — 0–19, 20–99, 100–499 and ≥500
  training events (analysis `cell_groups` and Bayesian `support_bins`).
- **IPCW documentation:** the workflow runbook and contrast manifest now name
  `cif_ipcw` as the primary estimand and the known-status run as secondary.
- **Figures:** `opera/visualization/paper_figures.py` renders the atlas,
  pooling, tabular, external, routing, horizon and calibration-map figures from
  the analysis tables (prototypes in `paper/render/prototypes/`). The atlas
  function takes held-out IPCW AUCs; do not use the old tuning-log atlas script.

Still open from the list above: cell- and patient-metadata builders, absolute
metric intervals in `estimates.csv`, specialist coverage reasons, per-patient
external vocabulary coverage, the pretraining corpus summary, and the
registry-enriched XGBoost decision.
