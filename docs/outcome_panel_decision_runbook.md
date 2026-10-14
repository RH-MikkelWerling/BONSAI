# Outcome-panel decision runbook (internal)

This turns the clinical review in `docs/outcome_panel_review_20260930.md`
into a reproducible **provisional** roster for a specified outcome release.
It does not edit parquets, training configurations or historical results.

## Prespecified rule

1. Clinical disposition overrides support. The versioned
   `opera/configs/outcome_panel_decision_v1.yaml` lists outcomes to retire or
   hold for a revised definition. Unlisted outcomes are provisional candidates.
   This is deliberately a short override list, not a claim that all remaining
   source functions are already validated.
2. For a candidate to enter the proposed primary panel, require at least
   **100 distinct training patients with an event by day 365** and at least
   **20 tuning patients with an event by day 365**, in the outcome's own
   eligible population. Otherwise move it to exploratory/low-support. These
   are pragmatic floors, **not** a guarantee of a precise AUC or calibration.
3. Report 30, 60, 90, 180, 365 and 730-day event counts in each split. Short
   horizons may have weaker support even when 365 days passes. A later figure
   should flag or omit unstable outcome–horizon estimates separately.
4. Report held-out counts only as descriptive precision planning. **Held-out
   counts and performance never determine panel inclusion.** After the final
   MEDS release, repeat this run against its versioned outcome directory and
   freeze the resulting decision manifest *before* final held-out comparison.

The event count is first qualifying event by horizon, not the number of
measurements, censoring-adjusted comparable pairs or an IPCW effective sample
size. `n_full_followup_or_event` is descriptive; competing death and IPCW
evaluation require their own support checks. The script uses each exported
outcome's own eligibility mask and fails on missing/duplicate subjects, invalid
dates or split mismatch. Outcome-specific censoring and clinical grading still
need separate adjudication.

## Server command

Sync these files first: `opera/configs/outcome_panel_decision_v1.yaml`,
`opera/run/decide_outcome_panel.py`, this runbook and any changed tests.
From the repository root in the usual OPERA environment:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export BONSAI_OUTCOMES_DIR="$PWD/data/processed_data/daly_care/outcomes"
export PANEL_REVIEW_DIR="$PWD/data/results/outcome_panel_review_$(date -u +%Y%m%dT%H%M%SZ)"
python -m opera.run.decide_outcome_panel \
  --outcomes-dir "$BONSAI_OUTCOMES_DIR" \
  --output-dir "$PANEL_REVIEW_DIR"
```

**Replace `BONSAI_OUTCOMES_DIR` with the exact directory used by the completed
model runs** if it differs from the example. Check it with `ls` first. The
output directory must be new; the script refuses to overwrite a prior review.

Outputs are `summary_INTERNAL.json`, `panel_decisions_INTERNAL.csv`,
`horizon_support_INTERNAL.csv`, `structure_audit_INTERNAL.csv` and
`input_files_sha256_INTERNAL.csv`, plus `threshold_sensitivity_INTERNAL.csv`.
The latter reports how many clinically eligible outcomes would pass a small
50/100/200 training by 10/20/30 tuning event grid; it is a robustness check,
not an invitation to select a floor after seeing held-out performance. These
files contain aggregate counts and file
hashes, not patient IDs, but small counts are **not disclosure-cleared**.
Share through the study's approved internal channel, not as public paper
supplement files. The summary records the exact rules/registry hashes and
whether any extra canonical outcome parquets were ignored.

The next manual sign-off should be limited to: (i) source grader/unit and
calendar-capture checks for primary candidates, (ii) clinical review of the
explicit definition holds, and (iii) whether any rare but essential endpoint
is worth a prespecified exploratory analysis. Do not tune the support floors
after viewing held-out model performance.
