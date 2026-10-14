# OPERA paper: ordered action and run checklist (7 October 2026)

This is the operational checklist for the paper's central question: do broad
encoder pretraining and supervised borrowing across diseases and outcomes
improve prediction when a disease--outcome cell has few labels? It supersedes
older day plans as a **run order**, not their detailed methods. Keep all
patient-level predictions, routes, checkpoints and unsuppressed tables on the
secure server. Statuses below mean "verified here" only when explicitly said;
the local repository cannot inspect the current server runs.

## What is already established, and what is not

- The approved supervised source has 37,557 first-line patients; 37,531 have
  released survival membership/split (29,105 training, 1,980 tuning, 6,446
  held-out). The panel has 69 approved outcomes. These are **not** outcome-wise
  eligible risk-set counts.
- The first-line flow supplied by the study team starts with 78,668
  diagnosis-linked patients, 38,635 with a first-line index, 37,357 initially
  disease-assigned, 209 additionally assigned after the coding-lag reassessment,
  and nine excluded for IgM MGUS. The team identifies the 78,668 as its
  pretraining population; reconcile distinct-patient and pre-2022 sequence
  counts with the frozen MEDS/pretraining manifest before a final caption.
- The clean router runs have reportedly finished, but their completion
  markers, saved query modes, common-epoch results and paired predictions
  have **not** been reviewed from the server in this local workspace. The
  historical October 6 run labelled interactive is not a valid interaction
  reference; see `router_query_audit_20261007.md`.
- The neural predictability atlas and several tuning figures are drafts, not
  held-out results. The new patient-icon router illustration is parked; the
  original source-anchor schematic remains in the paper.
- PHAIR pretraining is external and pending delivery over the next days. It
  does not block the non-PHAIR work below.

## Run next: finished-run inventory (CPU; read-only inputs)

Run on the secure server from the synced repository. The command fails rather
than silently comparing incomplete runs or overwriting an existing report.
If a marker is absent, inspect that run rather than treating its checkpoint
as proof of completion. Share only the aggregate JSON and status summary.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
ROOT="$PWD/data/models/approved_neural_router_clean_20261007_v1"
OUT="$PWD/data/results/router_clean_validation_next_20261007_v1.json"
for RUN in time2vec_router_additive_e2e time2vec_router_interacting_e2e time2vec_router_interacting_scale02_e2e time2vec_router_interacting_scale03_e2e time2vec_router_disease_only_e2e time2vec_router_outcome_only_e2e time2vec_router_no_sources_e2e; do
  test -s "$ROOT/$RUN/best.ckpt" || { echo "Missing best checkpoint: $RUN" >&2; exit 1; }
  test -s "$ROOT/$RUN/training_complete.json" || { echo "Incomplete run: $RUN" >&2; exit 1; }
done
test ! -e "$OUT" || { echo "Output already exists: $OUT" >&2; exit 1; }
python -m opera.run.compare_approved_neural_validation_v1 \
  --run "additive_01=$ROOT/time2vec_router_additive_e2e" \
  --run "interactive_01=$ROOT/time2vec_router_interacting_e2e" \
  --run "interactive_02=$ROOT/time2vec_router_interacting_scale02_e2e" \
  --run "interactive_03=$ROOT/time2vec_router_interacting_scale03_e2e" \
  --run "disease_sources=$ROOT/time2vec_router_disease_only_e2e" \
  --run "outcome_sources=$ROOT/time2vec_router_outcome_only_e2e" \
  --run "no_sources=$ROOT/time2vec_router_no_sources_e2e" \
  --output "$OUT"
```

Read the output for common epoch, best/last epoch, endpoint-group macro AUC,
loss, residual scale, source mode and saved query mode. Confirm the plain
shared-BCE comparator's checkpoint and release provenance separately; it is
not one of the seven clean router run directories. The next GPU action is
determined by this inventory, not by a rounded validation-loss ranking.

## Ordered research checklist

### 1. Make the pooled-model comparison trustworthy (highest priority)

- [ ] Audit fresh additive and interacting **equal-scale** checkpoints with
  `opera.run.audit_router_query_pair` under the clean run root. Require saved
  interaction tensors, same disease roster, release, encoder initialization,
  seed, residual scale and common evaluation epoch. Do not compare the old
  misleading interaction checkpoint.
- [ ] Export fixed-best-checkpoint **tuning** predictions and routes on the
  same patients, outcomes and horizons for plain shared BCE, the clean
  no-source/disease-only/outcome-only/additive/interacting arms and the 0.2/0.3
  capacity arms. Reuse existing Parquets only when checkpoint hashes and
  keys match. See the command template in `router_query_audit_20261007.md`.
- [ ] Compute patient-paired 365-day AUC, Brier and calibration contrasts,
  per-outcome and family effects, and unsupported-cell inventory. Compare the
  interacting 0.1 versus additive 0.1 query modes separately from the 0.1/
  0.2/0.3 capacity sweep. Use plain shared BCE as the main performance
  reference; also use **endpoint-group-loss-matched grouped shared BCE** to
  isolate the readout/routing effect. The 2-by-2 source-bank factorial is
  mechanistic, not the training-pool factorial. A disease-identity-aware
  non-borrowing control may be needed for the strongest borrowing claim.
- [ ] Quantify routing itself: shared/own/borrowed weights, entropy, full-space
  residual norm, source-removal change in cumulative incidence, and variation
  between patients with the same fine disease and target. Route weights alone
  are not contribution-to-performance estimates.
- [ ] Replicate the selected architecture/scale and key baseline under matched
  seeds before a confirmatory ranking. Freeze a selection rule using tuning
  data only. **Held-out stays closed.**

### 2. Test the core scarcity counterfactual (GPU; start preparation now)

- [ ] Produce a disease-by-outcome support manifest from *training* labels,
  with independent rules for whether a specialist can train and whether a
  held-out AUC can be estimated. Record every supported, unsupported and
  failed cell; report pooled-model coverage of cells with no viable specialist.
- [ ] Build and validate the **primary independent specialist** path: initialize
  from the common pretrained encoder, with no pooled supervised checkpoint;
  fit one fine disease and one outcome using the matched direct-risk loss,
  readout, six horizons, eligibility and checkpoint rule. The existing
  `finetune_all_horizons.py` instead starts from a pooled supervised model:
  retain it only as a separate *post-pooling adaptation* arm. Pilot the new
  path across support strata before launching all trainable cells; a pilot
  cannot support a whole-landscape claim.
- [ ] Compare pooled shared BCE and the selected router against specialists
  on identical patients/risk sets. Make the Bayesian rarity analysis use
  training primary-event count as its scarcity axis and partial-pool by
  disease and outcome (with family variation). Account for correlated
  cell estimates from shared patients and outcomes, using patient-clustered
  paired resampling or an explicit joint uncertainty model. Prespecify mean gain, equal-cell
  gain, lower-tail harm and fraction of cells helped; distinguish performance
  where estimable from coverage where a specialist cannot be fit.
- [ ] If compute allows after the direct specialist test, run the distinct
  one-disease/all-outcomes and all-diseases/one-outcome **training-pool**
  factorial. Do not substitute disease-only/outcome-only *router sources*.

### 3. Finish data validity and the conventional benchmark (parallel CPU/user)

- [ ] On server, reconcile the 26 first-line patients without released
  survival/split against recorded death or end-of-follow-up on/before the
  treatment index. Do not assign all 26 one cause without the audit.
- [ ] Audit post-treatment diagnosis assignment (14-day initial and 90-day
  reassessment windows), including confirmation based on later records.
  Evaluate index-available disease routing with unknown/shared fallback on
  tuning data; retain the coding-lag rationale and report affected counts and
  performance. This is essential for an at-treatment deployment claim.
- [ ] Finish Table 1 from one patient per supervised-cohort member: audited
  patient-table sex/birth date, pre-index MEDS medication/diagnosis/laboratory
  counts and history span, modal pre-index region, unique pre-index specialties,
  disease-specific risk/performance availability, follow-up and splits. Keep
  exact and patient-level files internal; submit the complete combined figure/
  table bundle for disclosure review before release.
- [ ] User-owned separate-repo tabular work: repair the feature matrix and
  XGBoost comparator while GPU work runs. Align index, approved 69 outcomes,
  six horizons, patient eligibility, temporal split and evaluation metrics
  before neural-versus-tabular claims.
- [ ] Verify the pretraining distinct-patient/sequence manifest behind the
  78,668-patient label and the exact pre-2022 cutoff. Update cohort flow,
  Table 1 and manuscript denominators together; review the compiled diagram
  at print size. Do not put internal code names in captions.

### 4. Lock descriptive figures and manuscript while PHAIR is pending

- [ ] Regenerate the full 69-outcome neural predictability atlas and the main
  model comparison from fixed tuning predictions; indicate event support,
  unavailable cells, paired uncertainty and the tuning-only status.
- [ ] Keep the existing source-anchor schematic. Park the extra local icon
  figure. If Figma becomes connected, build an editable original illustration
  with patient/history, fine disease, target outcome, all eligible source banks
  and additive-versus-interacting query; review it before inserting it.
- [ ] Redesign the empirical borrowing display by target outcome and home fine
  disease, with shared/own/borrowed mass and source support. Keep the 69-by-69
  detail in the supplement; do not present a global mean as universal routing.
- [ ] Review forest spacing, movement-arrow density and PCA/UMAP companions;
  report full-space movement and prediction changes alongside projections.
  Keep patient-level plot data server-side and export only reviewed aggregates.
- [ ] Sweep abstract, introduction, methods, results placeholders, limitations
  and captions for one story: scarce disease--outcome labels; value of joint
  supervised learning; what adaptive borrowing adds; what broad pretraining
  adds; clinical calibration and harm, not AUC alone. Clearly separate tuning
  hypotheses from held-out findings.

### 5. When PHAIR arrives (not a prerequisite for steps 1--4)

- [ ] Record checkpoint, architecture, vocabulary, numeric encoding, time
  representation, pretraining dates/population, and licensing/provenance.
  Verify token/weight compatibility and a strict load before any comparison.
- [ ] Under the **same** split, outcomes, optimization and selection rule,
  compare local encoder, PHAIR direct transfer and PHAIR followed by local
  domain-adaptive pretraining. At minimum cross each with plain shared BCE and
  the selected router. Replicate the main contrast and test whether broader
  pretraining substitutes for or complements supervised borrowing.

### 6. Final lock, then one held-out pass

- [ ] Freeze cohort/outcome releases, estimands, eligible-cell thresholds,
  architecture/scale, seeds, checkpoint rule, family definitions, bootstrap/
  Bayesian model and tabular benchmark contract before inspecting held-out
  performance.
- [ ] Run one paired held-out evaluation with discrimination, Brier,
  calibration, family/cell heterogeneity and uncertainty. Treat unsupported
  cells as a coverage finding, not a numeric AUC. Update all figures and the
  thesis manuscript from the same locked result bundle.

## Decision rule for the next server job

1. If the seven-run inventory fails: finish or repair the named run, then
   rerun the aggregate inventory into a **new** report path.
2. If it passes but fresh additive/interacting checkpoints are not verified:
   run their CPU checkpoint audit next.
3. If the audit passes but matched tuning prediction/route Parquets are
   missing: evaluate those fixed checkpoints on a free GPU next.
4. If paired predictions exist: compute the paired contrasts and source
   diagnostics next, while scheduling the specialist pilot.
5. If the pooled comparison is locked: devote GPU capacity to replicated
   finalists and the specialist grid. PHAIR work joins when its checkpoint
   arrives; no held-out job is launched before the final analysis lock.

For the proposed stage-gated, resumable multi-GPU controller and its hard
scientific stop points, see `opera_gpu_experiment_system_20261007.md`.
