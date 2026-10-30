# AKI correction: principled recovery and compute plan

30 October 2026. The author confirms that the reversed-window implementation
produced the AKI input used for OPERA. Treat this as a confirmed outcome
construction error. No model training has been launched or cancelled here.
No frozen release has been overwritten. The held-out results remain unopened.

AKI is retained. The implemented comparison, release amendment and selective
runner handoff are described in
[AKI_RELEASE_AND_RUNNER_HANDOFF_20261030.md](AKI_RELEASE_AND_RUNNER_HANDOFF_20261030.md).

The author reports roughly **5–6 hours per full supervised joint/router fit**
and access to **up to eight GPUs**. The estimates below interpret that duration
as hours, assume one GPU per fit, and distinguish GPU time from elapsed time.

## Recommendation

Correct and retain AKI if the co-authors consider the intended phenotype useful.
Create a new label release, measure the development-label changes, then refit
only final models whose training/selection depended on changed inputs. Reuse
independent unaffected fits and all verified label-independent representations.
Freeze the existing architecture rather than replaying every historical search.
Recheck the declared final model/encoder choices on corrected tuning results.

Removing AKI entirely from both supervised training and evaluation is a valid
alternative scientific scope decision. Removing it only from the results does
not make the existing shared models independent of its erroneous supervision.
The correction is warranted by the known error, regardless of whether any
model's performance improves or declines.

## First step: establish what actually changed

Have the source owner regenerate the AKI measurements/episodes in an isolated
location with only the reversed condition corrected. Keep the original
mean/median choice and other branches unchanged for this diagnostic. Compare
complete histories, including pre-index samples, before deriving the same
post-index endpoint and censor dates. Keep held-out performance and selection
out of this audit.

The final AKI export needs comparisons on training and tuning data of:

- Patient eligibility and dates of index, first qualifying post-index episode
  and censoring. The existing lab engine can change censoring when a patient
  changes from event case to event-free.
- Added/removed events and changed event times, plus event status and observation
  status at 30, 60, 90, 180, 365 and 730 days. Report disagreements as a fraction
  of all eligible patients and of original/corrected events separately.
- Measurement-level branch flags, incident episode starts, and how often the
  historical-year branch changes the combined AKI flag. Stage differences alone
  do not alter OPERA's current onset-only endpoint.
- Examples spanning additions, removals and date shifts, reviewed clinically.

If the final training and tuning inputs, masks, event-count priors and selection
quantities are identical, a source-code correction alone does not require
retraining an identical experiment. Equality of event totals or high label
agreement is insufficient to establish this. If inputs do change, use clean
refits as the default for final affected models, without inventing an acceptable
label-error percentage after inspecting prediction results.

Before expensive final refits, agree the operational AKI definition once. The
published source has a mean historical reference and an additional most-recent
sample criterion without a maximum gap. Fixing the inequality does not resolve
those separate departures from the NHS median-based algorithm. A documented
local definition can be legitimate, but must not be called an exact standard
implementation. Do not combine a definition redesign silently with the bug-only
comparison.

## Which existing work can be reused

The paper factorial generator explicitly uses
`same_pretrained_encoder_no_pooled_supervised_weights` for S/D/O/J and rejects
a supervised source checkpoint. Its fits cover all six horizons together.
This is important: the primary specialist portfolio does not inherit a joint
model's AKI supervision merely because the architecture is shared.

| Model/artifact | Default action after changed AKI labels | Reason |
|---|---|---|
| Local/national raw-record pretraining; self-supervised domain adaptation | Reuse | These do not consume the faulty AKI target. Check any separately derived-token vocabulary provenance if relevant. |
| Sequences, raw laboratory values, vocabulary and tabular feature matrices | Reuse where source inputs are unchanged | The bug is in the derived target. |
| S: one disease, one outcome | Refit AKI specialists only | Other targets' independently initialized fits never use AKI labels. |
| D: all diseases, one outcome | Refit the one AKI-pooled model | Other outcomes' D models are independent of AKI. |
| O: one disease, all outcomes | Refit affected disease fits | Each can use AKI supervision and AKI tuning loss. |
| J: all diseases/outcomes | Refit | Shared encoder and checkpoint selection can depend on AKI. |
| Final OPERA, matched own-task and global-only controls | Refit | Shared supervised parameters, source corrections, training event counts and selected epochs can change. |
| Second-seed joint/router models retained in claims | Refit those same final designs/seeds | Preserve the replication actually reported. |
| XGBoost/logistic models fitted separately per task/horizon | Refit AKI jobs only | Reuse other targets with unchanged features, labels and selection inputs. |
| Pooled XGBoost per outcome/horizon | Refit AKI jobs only | It pools diseases, not target outcomes. |
| Supervised models initialized from another supervised model | Follow the parent dependency too | An auxiliary legacy specialist runner supports this pattern; do not assume it is the primary clean-initialization factorial. |
| Historical architecture screens and discarded router variants | Keep as development history | Not automatically part of the final inferential model roster. If their numerical contrasts remain in the paper, refit the affected reported models or omit those contrasts transparently. |

Paths in a newly versioned release may change even for unchanged targets.
Reuse should be explicit in a carry-forward inventory identifying identical
scientific inputs, rather than silently rewriting old run metadata or launching
new copies of every specialist because an overall release hash changed.

## Concrete training scope per retained seed

There are **24 configured fine diseases**. Some small-disease jobs may be
non-estimable and have no trained fit. Determine those from corrected development
support, not from the 15 AKI cells estimable at one year in the historical tuning
performance table.

A minimal final local mechanism/comparison set is:

1. J, the shared joint model.
2. The chosen OPERA architecture, currently the revised AB router.
3. Its matched own-task-only control, RBH_AB.
4. Its matched global-only control, N_A.

These four replace the affected main fitted comparisons. The local sharing
factorial adds up to 24 O fits, up to 24 AKI S fits and one AKI D fit.
The [workload table](AKI_RECOVERY_WORKLOAD_20261030.csv) records these counts.
That is **up to 53 neural fits per seed**, including the four main fits, rather
than the entire S/D portfolio. Each neural fit supplies all six horizons.

If AKI is removed from both training/evaluation, the candidate count is instead
four main fits plus up to 24 O fits, **up to 28**. This saves the AKI S/D work,
but retains the large shared-model refits. It is therefore not a way to avoid
the main correction cost.

These counts exclude optional original bank factorials, random/frozen encoder
controls, sampling sensitivities, data-efficiency fits and extra seeds. Add
only the designs whose claims remain in the final paper. Narrowing these
secondary claims is legitimate when declared before held-out inspection and
reported honestly. It must not depend on which corrected results look best.

For tabular models, the upper candidate count at 24 diseases and six horizons
is 144 disease/horizon jobs, each containing XGBoost and logistic fits, plus
six pooled-XGBoost jobs. Registry-enriched AKI jobs are additional if retained.
These jobs have internal tuning and cannot be costed as single estimator fits.
They run on CPU in the current XGBoost implementation (`n_jobs=4`), so cap total
workers to the allocation's cores/memory and avoid starving the GPU loaders.

## Compute estimate with eight GPUs

| Work | Single-GPU fit assumption | GPU-hours | Elapsed training with eight free GPUs |
|---|---|---:|---:|
| Four main local fits | 5–6 h each | 20–24 | About 5–6 h if independent and launched together |
| Main local four plus a second-seed J/R pair | 5–6 h each | 30–36 total | About 5–6 h for six parallel fits |
| Four national downstream fits, E/EA × J/R | Actual runtime not yet measured | Unknown | Can overlap local work once encoders are ready; do not assume the local fit duration |
| All 53 corrected local candidate fits, treating every fit as full-size | Deliberately crude same-cost scenario | 265–318 | Seven waves, about 35–42 h |
| All 28 local candidates after dropping AKI, same crude scenario | Same-cost scenario | 140–168 | Four waves, about 20–24 h |

**The 53-fit/35–42-hour row is a scheduling scenario, not a measured forecast.**
Disease-specific O/S fits train on smaller cohorts and should not automatically
be assigned the full-cohort joint runtime. Conversely, early stopping, validation,
startup/checkpoint I/O, CPU contention and walltime interruptions affect actual
cost. Counting 53 fits does not mean 53 full-population runs. There are only five
full-population training designs in this minimal local roster (the four core
fits plus D_AKI); the other 48 jobs are partitioned by disease.

A useful rough exposure check is that, with identical epoch counts and roughly
linear per-patient compute, all O disease fits together traverse the full
population once per epoch, as do all AKI S fits together. Combined with five
full-population designs, this is around seven full-cohort epoch workloads,
not 53. This is not a reliable runtime conversion because the models, eligible
populations and early-stopping epochs differ. Use the actual worker records.

National encoder size, sequence length and hardware may give a different runtime
from the local encoder. Do not add four more local-duration fits and present
that as a measured national schedule.

National downstream fits that have not started are already planned work.
Starting them on the corrected release adds no *repeat* training relative to
the plan. Self-supervised national pretraining can continue. A task that has
already been supervised on the old AKI target belongs in the affected inventory.

## Get a measured budget without launching training

The existing compute inventory sums the workers' recorded attempt times and
keeps model aliases, seed, scope and origin. From the repository root:

```bash
opera_compute_stamp=$(date +%Y%m%d_%H%M%S)
.venv-v100/bin/python -m opera.run.paper_compute_inventory \
  --bundle-dir data/paper_256_v3/bundles/stage5 \
  --runs-dir data/paper_256_v3/neural_runs \
  --output-dir "data/paper_256_v3/analysis/aki_compute_${opera_compute_stamp}"
```

Use the actual bundle directory if its name differs; repeat `--bundle-dir`
for other retained bundles. The command reads existing records and launches
no model training or inference. Return `compute_per_fit.csv` and
`compute_by_scope.csv`, both aggregate reports, for an allocation-specific
schedule. Missing attempt records must be labelled unknown, not zero-cost.
A bound from total GPU-hours divided by eight is a capacity lower bound,
not a calendar-time guarantee; long jobs and dependencies set the critical path.

## Analyses and figures after refitting

- New fits require new predictions and router/embedding exports. Checkpoint
  interventions, source means, route vectors, IG and input-removal outputs are
  fitted-model-specific and cannot be transferred to a changed checkpoint.
- Unchanged independent models can retain their predictions/embeddings. Rebuild
  paired comparison tables on compatible labels and shared patient rosters.
- Every frozen threshold estimated from a changed model's predictions must be
  rebuilt on the same declared development population, then fixed for testing.
  The native clinical-index definitions and recorded score values are unchanged.
- Most of the graphical design work remains usable. The new schema still has
  the same outcome count when AKI is corrected and retained. Redraw empirical
  panels with the new outputs; do not redo the design process.
- Keep all old-label artifacts identifiable as development. Do not mix their
  router diagnostics with corrected-checkpoint performance or replace an old
  result filename with a new one without a release identifier.

## What to avoid as a primary repair

Evaluating the old AKI head against new labels measures robustness to a target
revision. It does not produce a model trained for the corrected endpoint.
Likewise, editing cached event columns does not retrain a shared encoder,
remove the old supervised influence or repeat checkpoint selection. Updating
only the AKI head leaves other shared parameters unchanged. These can be
exploratory repair strategies, but do not match the original clean-initialization
factorial unless the study explicitly changes and validates that training regime.

There is no need to repeat all historical hyperparameter searches. Preserve the
existing architecture and optimisation choices, use clean initialization and
the same early-stopping policy, and report how they were originally developed.
Do not describe an old-label search as a corrected-label search that was never
run. Recheck selection among the final J/R/encoder candidates that are actually
refitted on the corrected release.

## Proposed order now

1. Continue raw-record national pretraining. Preserve the running tuning audit
   as development evidence. Avoid launching additional expensive old-checkpoint
   explanations that will need replacing once a corrected final fit is chosen.
2. Regenerate AKI and audit development differences; agree the clinical definition.
   The source owner can do this while existing compute records are collected.
3. Build a new release with all unaffected endpoint files preserved and only AKI
   changed. Record the actual model roster and carry-forward inputs.
4. Launch the four main local fits concurrently from the clean pretrained encoder.
   Launch ready national J/R arms with corrected labels in spare slots. Fill
   remaining capacity with disease O fits and AKI S/D jobs, scheduling large
   cohorts first. Preserve the original selected seed plan.
5. Run mechanistic/interpretation inference on the corrected checkpoint only
   after the required final model identities and tuning choices are settled.
   Clinical thresholds, paired tables and figures follow those predictions.
6. Audit the revised outputs, amend the dated analysis record, then open the test
   set for the fixed model/outcome/analysis roster.

This is a recommendation for a bounded, clean correction, not a mandate from a
reporting checklist. Transparent reporting of the outcome definition, development
history and final evaluation protocol follows
[TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378).
