# Agenda — 30 October 2026

Aim: finish the local tuning rehearsal, explain the router's behaviour, and
freeze the analysis before inspecting held-out results. National pretraining
is **still running**, confirmed by the user on 30 October. A combined
local/national test readout depends on completing the planned national
downstream fits and tuning choices. Pretraining completion alone is insufficient.

## 1. Finish recovery after syncing the resume correction

Continue the existing `full_tuning_rehearsal_20261029_234123` workspace using
[the recovery instructions](RESUME_FULL_TUNING_REHEARSAL_20261030.md). Sync the
updated `complete_local_figures.py` before restarting. Its blanket source-code
check is now disabled by default, and code fingerprints are retained as history.
The standalone wrapper reuses completed stages rather than selecting or
training a new router. Scheduler walltime still applies. Avoid changing files
while the process is actively running.

Return these aggregate outputs when available:

- `full_rehearsal.json` and `analysis/completion.json`, plus failed stage logs.
- `router_vectors.json`, cross-family metric reconciliation, and the updated
  review ZIP. The standalone vector/whole-cohort reports are not all inside
  that ZIP.
- Whole-cohort metrics/coverage, clinical inventory and cutoff diagnostics,
  and paper-number outputs.

An exit status alone is not release approval. Identify any `finished_with_gaps`
analyses explicitly. Corrected IG requires the evaluation-gradient fix in the
runtime that actually ran it. Recovery does not install that fix automatically.

## 2. Sign off the intended router and the extreme specialists

The previous server replay already verified the revised R_AB primary fit
(`79d39ad1e7cb51472026`, R_L alias), sampled probabilities and active borrowed
terms. Do not repeat unchanged GPU replay just to obtain a prettier map.

Close the remaining provenance checks: ordered disease/outcome banks, matching
checkpoint/configuration across performance, removal, arrows, clinical figures
and IG, and fresh mean/zero export receipts. Review raw vector agreement before
interpreting similar projected arrows. Review a second seed if claiming
stability across seeds. If that report is unavailable, restrict the claim to
the verified selected fit rather than implying replicated stability.

Use the exact contrast definitions in
[the hierarchy and extreme-AUC review](HIERARCHY_AND_EXTREME_AUC_REVIEW_20261030.md).
Broad allocation and weak disease-pair effects are acceptable findings. They
do not establish that selective gates are necessary. Do not choose another
router because its map is more appealing.

**Additional bounded check:** inspect the existing TCL and TRANSFORMED_FL
specialist checkpoint/selection receipts and training histories. In the old
tuning download, these specialists often rank below chance. Verify cohort and
endpoint identity, event-risk direction and saved-versus-live predictions for
representative extreme cells. The earlier joint/router replay did not replay
these specialists. Use existing fits and tuning patients only. Do not flip
scores, remove difficult cells or retune to improve the comparison. A confirmed
export/identity bug requires a documented correction before the analysis lock.

## 3. Apply the audited reporting corrections after recovery

Once recovery has finished, sync the locally audited fixes from
[the code audit](CODE_RELEASE_AUDIT_20261030.md). Refresh the clinical reports
on CPU using existing predictions:

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
opera_clinical_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_clinical_root="$opera_full_root/clinical_corrected_$opera_clinical_stamp"

.venv-v100/bin/python -m opera.run.refresh_clinical_figures \
  --workspace "$opera_full_root/analysis" \
  --output-dir "$opera_clinical_root"
```

This corrects threshold-boundary handling and the low-risk search. It does not
require retraining or fresh training-prediction export. Use the new
`clinical_scores` directory explicitly when assembling the next review.
Keep incompatible native risk partitions visible as gaps. Binary comparisons
can remain estimable even when a complete ordered partition is not.

Inspect existing tabular tuning metadata for unexplained empty `tune_params`
where model selection was supported. The new fail-loud guard does not establish
that every old tabular fit had successful candidate selection.

## 4. Freeze analyses and sign off the figure roster

Decide these before looking at test results:

1. Primary model/encoder and comparator roster, retained router controls and
   original-router sensitivity. Record actual checkpoint/config/vocabulary
   identities, not only display names.
2. Endpoints, horizons, competing-event handling, IPCW calculation, matched
   populations, case/control eligibility and task/horizon aggregation. Keep
   stricter-support and disease-balanced summaries as declared sensitivities.
   Show absolute specialist AUC alongside gains. Different support cutoffs
   produce different task populations, not learning curves.
3. Native clinical categories, train-derived thresholds, paired evaluation
   populations and binary operating points. Specify uncertainty for extra
   low-risk patients/events captured before making inferential clinical-count
   claims. Existing survival intervals do not supply those paired intervals.
4. Every figure's question, input receipt, estimand, horizon, exclusions and
   uncertainty label. In particular, clarify the hierarchy's separately trained
   controls, correct its all-horizon caption and remove an unsupported precise
   crossover claim. Its current row-bootstrap ribbon is descriptive and does
   not handle shared patients, nested outcomes or repeated horizons.
5. Check clipping, legends, panel headings, clinical labels and missing families
   against the fresh review. Every omitted analysis must be listed rather than
   silently replaced by a previous run's output. Cosmetic improvements can
   continue after the lock, provided they do not change analytical choices.

Keep frozen outcome definitions. The Danish-code qualifications already
identified belong in the supplement. Rebuilding outcomes is not today's task.

## 5. National arm and the test decision

While national pretraining runs, freeze its planned downstream architecture,
initialisation/vocabulary transfer, fine-tuning schedule, tuning criterion,
candidate roster and selection rule. Check completion and transfer receipts
before using its outputs. Complete those tuning choices before a combined
test readout.

**Recommended sequence:** local sign-off and protocol lock today, then one
combined held-out evaluation once the national downstream fits are fixed.
If those fits are not ready today, this is a real dependency rather than a
reason to leave the local work unfinished.

A staged local-first test evaluation is possible only under an explicit prior
lock of the national recipe and reporting rules, without feeding local test
observations back into national development. It sacrifices the simplicity of
one untouched combined readout. It is not the default plan here.

Release only when the dated record contains the resolved checks, actual frozen
roster and thresholds, analysis settings, explicit gaps and held-out commands.
Use a fresh held-out workspace. The full tuning rehearsal remains tuning-only
and must not be converted by changing an environment variable. Neither an
attractive borrowing map nor a national model outperforming XGBoost is a
release condition.
