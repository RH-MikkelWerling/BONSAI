# Provisional 60-outcome panel: figure and source-audit sequence

The 12-group figure map is `opera/configs/outcome_display_groups_60_candidate_v1.yaml`.
Its 60 members now match the uploaded `panel_decisions_INTERNAL.csv` exactly
(no missing or extra outcomes). This is a **display taxonomy only**. The preview uses an existing
87-outcome-trained model; neither model training nor checkpoint selection has
been changed. Source definitions and release status still need validation before
using the figure outside internal review.

The present `endpoint_group_macro` training loss is not these 12 clinical
families: `endpoint_groups_v1` pairs grade variants, then weights each endpoint
group equally. That may overemphasize sparse singleton endpoints. The old
registry families are also very uneven. Freeze the outcome roster and source
definitions first; then compare any revised hierarchical loss against the
same-roster baseline with a common validation metric. Do not compare headline
macro scores across different rosters/group definitions without rescoring.

## Source-audit and correction order

1. **Freeze provenance.** On the server record the exact R script hash,
   `load_dalycare_package.R` and installed DALY-CARE package commit/function
   bodies, raw table snapshots, NPU dictionary, units/reference ranges,
   treatment/index source, and split/cutoff dates. The local script sources
   `/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R`, so its visible
   `AE_*` calls alone cannot establish which thresholds were run.
2. **Audit the 60 retained definitions by priority.** Start with imported
   lab graders (ALT, bilirubin, creatinine, anemia, neutrophils, platelets,
   INR, electrolytes), checking exact threshold boundaries, baseline branch,
   assay/unit and reference interval. Then inspect diagnosis/procedure and
   medication code lists, their calendar coverage and pre-index exclusions.
   Confirm `eosinophilia` tests both >ULN and >baseline, or describe it as a
   lab proxy. Confirm that `serious_infection_composite` denotes its actual
   narrow code set and that `antibiotics` is treatment use, not proven infection.
3. **Resolve held/retired endpoints separately.** Do not silently restore
   ALP, fibrinogen, leukocytosis, lymphopenia, hypokalemia, hyponatremia,
   MACE, or antimicrobial-derived infection endpoints by renaming them.
   Their clinical definitions or labels need explicit decisions. The local
   R script still computes several of these; removing them from the primary
   roster does not remove them from historical exports.
4. **Check dependency edges before pruning exports.** `hepatotoxicity` uses
   AST G3+ even though AST is currently exploratory; `severe_cytopenia` and
   `treatment_failure` also depend on component parquets. Retaining a
   component for composite construction is distinct from retaining it as a
   primary model target. Run the main outcome script, then the LPR3 coverage
   gate, then `build_composites.R` in a new versioned output directory.
5. **Test each change on isolated data.** For each revised endpoint, produce
   an old-versus-new split/horizon count and patient-level transition table
   *on the secure server*, including eligibility, event date and censor date.
   Export aggregate counts only. Add boundary-value unit tests for numerical
   graders and component/censor tests for composites. Re-run chronology,
   duplicate-ID, split and observation-window audits.
6. **Freeze the final data release.** Recompute the panel-decision manifest on
   the final MEDS/outcome release before final held-out evaluation. Update
   Table 1, CONSORT and plots from that single release; keep the historical v4
   results labelled as pilot data.

Immediate source risks to verify rather than patch blindly: hard-coded
`STUDY_CUTOFF <- 2026-03-19` in the R script; exact `cut()` boundaries in local
hyponatremia/hypomagnesemia graders; and an apparent mixed NPU naming contract
(`NPU.MAG` versus imported `NPU.MG2`). The prophylaxis `J01EE01` appears in a
commented example, not an active filter at the visible call site, so its
impact must be checked on the server rather than inferred from that comment.

## Additional experiment: multi-horizon BCE within diseases

After the final roster/data release, compare each endpoint's matched-horizon
performance under multi-horizon/multi-outcome BCE against the strongest
single-horizon and single-outcome controls. Report both all-cohort and
predeclared disease-subcohort estimates on identical eligible patients,
censoring rules, horizons and splits, with event/control counts and paired
bootstrap intervals. Do not infer a disease-specific improvement from an
all-patient macro score. Prioritize subcohorts with adequate tuning and held-out
support; do not select them using held-out AUC. Keep this separate from the
loss-family reweighting ablation so the effects are identifiable.
