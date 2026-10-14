# Data-side closeout sequence (4 October 2026)

Do not call the outcomes final because the R script completed. The selected
37,540-person first-line mapped file is the cohort/index source; outcome risk
sets can be smaller. Keep all audit and TLS outputs in new protected directories
until the checks below pass and their input hashes are recorded.

1. Confirm the completed outcome log has its terminal completion marker and
   review the aggregate first-line follow-up/cutoff messages. Run
   `opera.run.audit_outcome_exports` on the new `opera_outcomes/` directory;
   resolve date, split, ID and inventory failures. Reconcile overall survival
   against the selected mapped export without replacing outcome eligibility by
   the cohort count. The initial completed run reports 37,512 outcome
   candidates versus 37,540 first-line patients (28 excluded before outcome
   construction); reconcile the follow-up and administrative-cutoff messages.
   The LPR2 `v_recnum` overflow was confirmed to change procedure labels;
   the text-key repair needs a fresh ungated run. Compare old/new rates by
   index year and time-to-event on both sides of the LPR3 transition using
   `opera.run.audit_lpr3_procedure_eras`, then decide each coverage gate.
2. Establish the actual source-completeness dates behind 4 March versus
   19 March 2026 administrative ends; the second-line and outcome R scripts
   currently differ. Quantify the impact in the 15-day interval. Freeze one
   documented follow-up policy, with justified endpoint-specific exceptions,
   before the second-line/TTNT and outcome release is immutable.
3. Extract TLS labs with `opera/R/export_candidate_tls_labs_v1.R` and run
   `opera.run.export_candidate_tls_v1` against *this run's* overall-survival
   parquet. Inspect NPU codes, unit distributions, nonnumeric values, same-day
   complete-panel eligibility and day-30 training/tuning support. Keep the
   historical TLS proxy retired; the two new candidates remain exploratory
   unless their narrower definitions and support receive explicit approval.
4. **Next after the join/coverage comparison:** make grading internally
   consistent before freezing labels. For each flagged endpoint, inspect the
   installed DALY-CARE grader, NPU/unit mapping, exact boundary behavior and
   baseline/clinical qualifiers; then choose a versioned regrade, an honest
   legacy/proxy name, or removal from the primary panel. Prioritize the
   leukocytosis G2+ naming mismatch, ALP/lymphopenia non-v6 grades,
   hypokalemia bins, neutropenia thresholds and baseline-dependent ALT/AST,
   bilirubin and creatinine. Differential-audit counts and event dates for
   any regraded labels; do not silently mutate the join-repair audit run.
   Working scientific policy: clinically relevant operational prediction
   targets remain useful even when clinical symptoms or a pre-index baseline
   cannot be observed; describe those as laboratory/registry proxies rather
   than complete CTCAE grades. Run `opera.run.audit_preindex_lab_overlap` on
   the secure candidate lab snapshot and ungated outcomes to quantify 30/90/
   365/730-day baseline availability and pre-index abnormality among events.
   Consider paired baseline-free primary and baseline-qualified sensitivity
   labels for selected endpoints, but do not double-count them as independent
   outcomes. The active prescription ATC literal `JO1EE01` was corrected to
   `J01EE01`, and right-open sodium/magnesium numeric bins were corrected in
   the local generator after the LPR2 repair run started; these require a
   later versioned rerun and label-difference audit, not retroactive claims
   about that in-progress run.
   Keep newly observed/worsened laboratory outcomes as a bounded sensitivity
   study, not additional primary multi-outcome heads at this stage. Use the
   baseline-availability audit and `event_overlap_INTERNAL.csv` from the
   isolated candidate exporter to choose at most one or two illustrative
   endpoints with adequate train/tuning support. Missing pre-index evidence is
   unknown, not event-free; a post-index change is not proof of treatment
   causation. Ask clinicians whether this estimand is preferable for the paper,
   but do not let that consultation or sensitivity rerun block the main-panel
   freeze once source coverage, grading and structural audits pass.
5. Run the versioned outcome-panel decision on the complete verified release.
   Check source grader/version, LPR3/SMR coverage, competing deaths,
   30/365/730-day support and held-out isolation. No panel selection using
   held-out performance.
6. Hash and manifest the selected mapped/expanded first-line artifacts,
   outcome parquets, TLS decision, second-line/TTNT exports, split labels,
   scripts, SQL loader/source snapshot and cutoff policy. Then regenerate
   Table 1/flow and model input manifests. The independent tabular feature
   matrix rebuild and matched supervised fine-tuning comparisons start only
   after those data contracts agree.
