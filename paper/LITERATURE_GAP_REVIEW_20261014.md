# Manuscript and literature gap review — 14 October 2026

This is a focused critical review, not a systematic review or a completed
risk-of-bias assessment. It distinguishes plans, implemented tools and actual
empirical evidence. No secure-server result was inspected here.

## Comparison papers and standards

- [MOTOR, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b36554b97da741b1c48c9de05c73993e-Paper-Conference.pdf): examines transfer across databases, pretraining-objective and scratch controls, temporal robustness and label efficiency. It motivates explicit provenance/controls, not a requirement to replicate its whole study.
- [EHRSHOT, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/d42db1f74df54cb992b3956eb7f15a6f-Paper-Datasets_and_Benchmarks.pdf): pairs pretrained representations with competitive ontology-expanded count baselines, transparent task demographics and few-shot experiments. A weak feature matrix would undermine our tabular comparison.
- [EveryQuery v3](https://arxiv.org/html/2603.07900v3): explicitly defines its task-query scope, compares prediction procedures and discusses expressiveness limitations. OPERA should define its narrower, curated treatment-start panel rather than imply arbitrary-task/zero-shot generality.
- [Earlier DLBCL paper](https://onlinelibrary.wiley.com/doi/10.1002/hem3.70336): ties a specific endpoint/population to risk groups and operating-point interpretation. The new broader panel needs comparable precision about applicability; inherited cutoffs are not universally validated thresholds.
- [TRIPOD+AI official expanded checklist](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf) and [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505): use the former to map reporting and the latter to challenge validity/applicability. Completing a checklist is not a substitute for evidence.

## Submission-critical evidence gaps

| Priority | Gap and why it matters | How to obtain it | Extra fitting? |
| --- | --- | --- | --- |
| P0 | Primary estimand, metric, panel weighting, support criteria and claim family are not fully locked. Thousands of cells make a favourable aggregate easy to obtain accidentally. | Freeze a dated analysis specification before final held-out inspection: primary horizon/metric/censoring estimand, weighting, contrasts, support rules, practical margins and multiplicity approach. Mark exploratory analyses explicitly. | No |
| P0 | Nation-scale training provenance is pending. The central transfer claim depends on actual corpus coverage, event cutoff, objective, architecture, vocabulary and source/destination identity overlap. | Audit the delivered checkpoint and source training records; publish an L/E/EA provenance table and overlap/remapping report. Verify pre-2022 cutoff and identical E initialization for EA. If provenance fails, change the artifact/protocol or narrow the claim; text cannot repair leakage. | Audit first; conditional rerun |
| P0 | Input parity and strength of the tabular baseline are not established by having XGBoost in the paper. | Freeze feature recipe: concepts/modalities, lookback windows, numeric summaries, missingness, pre-index timestamps, train-only preprocessing, exact patient universe and tuning search/budget. Benchmark the existing recipe from the DLBCL study where compatible. Keep enriched registry inputs separate. | Baselines already required; refit only if recipe inadequate |
| P0 | Endpoint code is documented but clinical/source validity is not fully closed. Second-line weak-evidence exclusions and unusual censoring can change the estimand. | Supervisor/clinical review of definitions; record unresolved rules and label provenance. Report endpoint-specific eligible/excluded counts by split, baseline recurrence rules and source coverage. For problematic branches, adjudicate a feasible secure sample or run a prespecified sensitivity; do not claim validated clinical grading from CTCAE-like thresholds alone. | Usually no; conditional label-dependent rerun |
| P0 | Cohort and Table 1 await approval; score availability and follow-up are incompletely described. | Fresh sealed-cohort Table 1 plus flow/risk-set counts; supervisor sign-off; missingness/source coverage; 730-day at-risk and unknown-status counts by held-out index period. Keep SSL/first-line/supervised/outcome-specific denominators separate. | No |
| P0 | Primary real estimates and real figure/table generation are unfinished. Templates and synthetic artwork are not results. | Complete verified fits → tuning/held-out predictions → immutable inventory → paired analysis → real plotting adapters → source-linked tables/PDFs. Smoke-test one completed real comparison end to end before processing everything. General prototype renderer still generates synthetic inputs. | Required fits; renderer engineering |
| P1 | Sparse-event discrimination and probability accuracy need a complete reporting contract. | AUC plus Brier/log loss/calibration with uncertainty and support; consider horizon-specific precision–recall/prevalence summaries for rare outcomes with a declared censoring estimand. Include full absolute and paired tables, not just maps. Do not substitute binary known-status AP for full-risk-set inference without naming the difference. | No |
| P1 | IPCW assumptions and weight stability are not sufficiently reported. | Define censoring estimation population, event/control conventions and positivity; report weight distribution, effective support and follow-up. Code currently floors G at 1e-6 and reuses estimated weights in bootstrap draws: justify stability, report conditioning, and assess whether refitting censoring within resamples or a sensitivity is needed. Do not present extreme-weight long horizons as reliable merely because a number is returned. | No model fitting; analysis work |
| P1 | Replication/selection provenance needs actual completed-run evidence. | Verify all seeds, selected epochs and fixed policy; separate tuning screens from held-out estimates. Report seed spread independently of shared-patient uncertainty. Fixed fitted-seed bootstrap intervals do not represent all training-set uncertainty. | Already planned replicated fits |
| P1 | Model/training/compute specification remains partly pending. | Export from resolved configs/checkpoints/completion markers into reporting tables: dimensions, heads, depth, FFN, temporal representation, parameters, optimizer/batches, actual epochs/steps/exposure, hardware, runtime and memory. Compare scopes fairly without pretending equal epochs equal compute. | No |
| P1 | Clinical-score plots are registered templates rather than a closed all-score study. | Inventory actual scores and applicable disease/treatment populations, index-time evidence, original categories, missingness and support. Lock model cutoffs on tuning using the low-risk/group-size rationale; report achieved held-out risk and specificity. Include per-score summary and supported panels; never silently merge original scores or unsupported groups. | No |
| P1 | Manuscript/model specification still needs reconciliation. | Supplement states a cubic support spline; current Bayesian example defaults to trend:none. Lock one primary model and distinguish support sensitivities. Clinical-score `_known_status` uses event time < horizon while CIF analysis uses <=: reconcile with the label contract and add a boundary test. Do not interpret unmatched operating points. | No |
| P1 | Reporting/reproducibility package incomplete. | Fill TRIPOD+AI with manuscript/supplement locations; record patient/public involvement or its absence, intended use/users, approvals/waivers, extraction dates, code/model access, versions, funding and contributions. Provide aggregate sources and reproducible commands consistent with secure-data restrictions. | No |

## Claim-dependent additions, not automatic prerequisites

- **“Pretraining itself helps”:** local/nation-scale comparisons do not test pretraining versus no pretraining. A small matched J scratch control is useful if making that claim. No need to repeat the whole S/D/O/J matrix just to answer it.
- **“Scale causes improvement”:** different origins/objectives/architectures confound scale. Either narrow the conclusion to delivered packages or add a genuinely matched scale/budget comparison.
- **“Sharing makes rare targets label-efficient”:** natural-support trends are associative. Use a bounded train-only downsampling experiment on prespecified cells if claiming causal label efficiency; otherwise say gains vary with observed support.
- **“The router works through selective borrowing”:** R−J mixes conditioning and extra capacity. For a mechanistic claim, a capacity-matched conditioned/fixed-gate control or source interventions may be needed. Predictive R−J alone does not require a full mechanism paper.
- **“Portable beyond Denmark”:** temporal testing and population-to-clinical transfer within Danish records do not establish geographic external validation. A regional/time sensitivity is useful but not a substitute for independent-site evidence.
- **“Clinically useful decisions”:** one specific action/endpoint with prespecified thresholds and appropriate net-benefit analysis can strengthen the paper. Prospective impact is needed for clinical-benefit claims, not a blanket prerequisite for a prediction-method study.

## Recommended completion order

1. Close analysis locks and checkpoint/input/label provenance while fits run.
2. Verify tabular features and one real end-to-end figure/table path immediately.
3. Finish the agreed matched comparisons and extract cohort/support/compute tables.
4. Produce calibration, follow-up, weight stability and seed/temporal sensitivities.
5. Reconcile manuscript claims with the final evidence and actual configurations.
6. Add only the controls needed for claims we still intend to make.

Do not add an exhaustive confusion-matrix wall, arbitrary attribution plots,
another hierarchy layer or unrelated benchmark models to compensate for these
fundamental gaps. The strongest existing design feature is the S/D/O/J factorial;
the missing work is making its conclusions, and the nation-scale transfer
comparison, auditable and well calibrated.
