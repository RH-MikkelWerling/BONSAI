# Manuscript release to-do (4 October 2026)

The 37,540-patient V4-MM-seed first-line cohort and index are the selected
source of truth. The outcome rerun is ongoing; no outcome or split count from
it is yet locked. Do not merge these denominators with the older 37,467-patient
Table 1 draft or the broader pretraining population.

## While outcome calculations run

- [x] Set paper body text to TeX Gyre Pagella and mathematics to `mathpazo`;
  use Helvetica-like sans figure labels to match the predictability atlas.
- [ ] Visually inspect the compiled PDF at reading and print size before
  accepting the typography and cohort-flow layout.
- [x] Make the cohort figure label encoder pretraining as a separate, broader
  pre-2022 EHR population and the first-line branch as the supervised source.
- [ ] Capture the actual pretraining patient and sequence denominators from the
  frozen MEDS release, and state whether they count distinct patients, shards,
  or sequences. Do not derive them from the 37,540 first-line patients.
- [ ] Reconcile the exact mapped first-line export to the frozen patient-level
  temporal split manifest, then insert training, tuning and held-out counts.
- [ ] Review the figure at two-column print size; simplify labels, balance
  whitespace and check that every arrow represents a real population transition.

## Table 1 decision

- [ ] Re-export from one record per included patient at the selected first-line
  index and the same frozen temporal split used for fine-tuning. Explain any
  difference from 37,540 as an explicit analysis-population restriction.
- [ ] Decide whether the main table is sufficient with demographics, grouped
  disease, performance/risk missingness, index year, observed follow-up and
  pre-index history. Keep fine cohorts, treatment categories and score-system
  details in the supplement if the main table becomes crowded.
- [ ] Audit corrected MEDS history span and coded-record summaries; the current
  table retains placeholders and historical values. Do not carry old values
  forward merely because the row labels are unchanged.
- [ ] Generate exact internal aggregates, then a disclosure-reviewed display
  bundle. Review the combined table and figure for small-cell reconstruction;
  primary-only masking is not publication clearance.

## Once the outcome rerun finishes

- [ ] Check the run log, unique patient/index keys, censoring and death-date
  logic, per-endpoint events, follow-up eligibility and source coverage.
- [ ] Compare the repaired LPR2 procedure outcomes with the old ungated run by
  index year, LPR3 era and time-to-event; keep or revise each coverage gate on
  the resulting ascertainment evidence.
- [ ] Immediately afterward, resolve the flagged grading inconsistencies with
  installed-grader, NPU/unit, boundary and baseline checks. Regrade in a new
  version, label as legacy/proxy, or retire each affected endpoint; audit changed
  patient labels before declaring the outcome release final.
- [ ] Reconcile the two administrative follow-up dates (4 and 19 March 2026)
  with source-snapshot provenance before declaring the outcome panel final.
- [ ] Freeze the outcome release manifest, split/eligibility support, and
  pretraining/first-line provenance separately. Only then populate outcome
  risk-set figures and launch matched supervised fine-tuning.
- [ ] Integrate the locked manuscript into the existing thesis slot at
  `C:/Users/MWER0040/Documents/thesis/thesis/papers/opera/opera.tex`. It is
  currently a placeholder, while `main.tex` includes it as Paper III. Keep
  article and thesis entry points separate: do not paste an article
  `\documentclass`, `\begin{document}`, natbib setup or its `figure*` layout
  verbatim into the memoir/biblatex thesis. Reconcile labels, bibliography,
  figure paths, captions, and the OPERA methods/result summaries in `kappa/`.
- [ ] Compare the final thesis PDF with the standalone article PDF for font,
  diagram, table and cohort-count consistency. Preserve each deliverable's
  separate page geometry and bibliography conventions.
