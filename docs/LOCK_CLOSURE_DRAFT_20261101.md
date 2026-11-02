# Analysis lock: closing note (DRAFT, 1 November 2026)

Closes the analysis lock of `docs/ANALYSIS_LOCK_DRAFT_20261015.md` before the
test set is opened. Everything below is decided on tuning data only. Items
marked **[fill]** are completed from the paired tuning analysis; nothing else
changes after this note is signed. Background: `docs/BORROWING_EXPERIMENTS_RECORD_20261101.md`.

## 1. Primary OPERA design

**Rule (unchanged, amendments 30 Oct – 1 Nov).** A candidate replaces R_AB if its
pooled tuning AUC (paired, primary estimand) is at least R_AB's minus the seed
noise floor (R_AB seed 42 vs seed 43); among candidates that qualify, the
highest pooled tuning AUC is selected. Routed designs must also pass the two
routing-health criteria. Encoder-rate fits (stages 10–11), scarce-data fits,
controls and seed repeats are never selected.

**Candidates:** R_AB; stage 8 (R_PF5E and variants, R_PH_G, R_PH_GE,
R_PH_GE_PF5E, R_PL8, R_PF5E_R32); stage 9 (J_OA, J_OA_LR, J_SP, J_FP, J_FPF,
J_FPF_PH, J_OA_OWN, J_OA_LR_OWN, J_OA_OWN_PH, J_FPF_OWN, J_FPF_PH_OWN_PH); stage 13
(J_OA_OWN_SW20, _SW100, _X, _X_SW, _PE, _PH_X, _PE_X_SW, _MAX). Stage 12 fits
(second seed, rank 32, ablations, 10% data) are sensitivity analyses, never
selected. Secondary: the selected design averaged over its seeds (amendment of 2 Nov).

**Training-log indication (not the decision):** J_OA_OWN 0.7916, J_OA_OWN_PH
0.7912 and J_OA_LR_OWN 0.7910 at the selected checkpoint against R_AB 0.7836.

**Parsimony under noise (agreed 1 November 2026, before the stage-12/13 fits
finished and before any test data).** The reading designs (stages 9 and 13)
track each other within about ±0.002, and the design's own seed spread may be
larger than R_AB's. Among candidates whose paired tuning AUC is within the
design seed spread of the best candidate, the simplest is selected (J_OA_OWN);
a more complex variant is selected only if it beats J_OA_OWN by more than that
spread. Design seed spread: |paired tuning AUC of J_OA_OWN − J_OA_OWN_s43|.

*Amended 1 November 2026, evening, before s44/s45 were trained (stage 14).*
s43 trailed s42 by about 0.005 at almost every epoch, not only at the selected
checkpoint, while R_AB's two seeds agreed to 0.0002; two seeds cannot say
whether s42 was lucky or s43 unlucky. Stage 14 adds J_OA_OWN s44 and s45 and J
s43. The design seed spread becomes the range (max − min) of tuning AUC over the
four J_OA_OWN seeds. The fit that goes to the test set stays the prespecified
seed 42; the tuning section reports the four-seed mean and range beside it, and
OPERA − J is also reported on the second seed pair (J_OA_OWN_s43 − J_s43).

*Amended 2 November 2026, after s44 and s45 were seen (tuning data only; the
test set is unopened).* Training-log tuning AUC at the selected checkpoint:
J_OA_OWN s42 0.7916, s43 0.7865, s44 0.7859, s45 **[fill]** (0.7829 at epoch 10
while running); R_AB 0.7836 / 0.7834 and R_PF_OFF 0.7827 / 0.7823 on seeds 42 /
43. The reading designs are seed-sensitive, the others are not, and s42 is the
highest of the four seeds. Three consequences, recorded here because they
follow from these results:

1. *Scope of the parsimony rule (clarification).* The design seed spread is now
   about 0.009, as large as the gap between J_OA_OWN and R_AB. Read literally,
   "among candidates within the spread of the best, the simplest" would then
   reach designs without outcome-specific reading. That was not its purpose: it
   was added to choose among near-identical reading variants, and named
   J_OA_OWN. The selection is therefore made in two steps, as the rule of
   section 1 already orders it: (a) the base rule decides between R_AB and the
   candidates (qualify against R_AB's noise floor; highest pooled tuning AUC);
   (b) parsimony then chooses among the qualifying designs of the winning
   family (stages 9 and 13) within the design seed spread of the best. As an
   additional check, the selected design must beat R_AB on every seed on which
   both were trained (seeds 42 and 43). We state in the paper that this
   clarification was made after s44/s45 were seen.
2. *Primary fit unchanged.* The test-set primary stays J_OA_OWN seed 42 as
   prespecified. Switching to another seed or to a seed ensemble on the basis of
   these tuning results would be selection on them, and would compare a
   multi-seed OPERA with single-seed references.
3. *Seed reporting on the test set (prespecified now).* All four J_OA_OWN seeds,
   J_s43, R_AB_s43 and R_PF_OFF_s43 are scored on the test set. Reported beside
   the s42 results: OPERA's test AUC over its four seeds (mean, range); the
   seed-averaged contrasts `OAOWN4_J`, `OAOWN4_OFF`, `OAOWN4_AB`, `OAOWN4_X`
   (each model averaged over its available seeds) and `OAOWN4_S42` (s42 against
   its seed mean); and the Table 2 ladder on seed 43 (`OFF_J_s43`,
   `OAOWN_OFF_s43`, `AB_OFF_s43`, `OAOWN_J_s43`). The expected size of the
   reading step on test is the seed average (about +0.004 over own corrections
   on tuning), not the s42 value (+0.009).

- Design seed spread (range over J_OA_OWN s42–s45, tuning AUC): **[fill]**
- J_OA_OWN four-seed mean tuning AUC (vs s42): **[fill]**
- Noise floor (R_AB s42 − s43; J s42 − s43, paired tuning AUC): **[fill]**
- Pooled tuning AUC, every candidate vs R_AB: **[fill]**
- **Selected primary: [fill]** (alias written to `OPERA_PAPER_ROUTED_MODEL` and
  `OPERA_PAPER_OPERA_MODEL`)

If a stage-9 design is selected: router-only analyses (source removal, borrowing
map, routing health, route concentration) do not apply and are not run (the
launcher refuses them); they are replaced by the analyses in section 5.

## 2. Encoder learning rate (stage 10–11; amendment of 1 Nov)

Primary analyses stay at 0.1x for every arm. The encoder rate is a sensitivity
analysis. **National-arm rule:** if, on paired tuning AUC, J_ELR10 beats J and
R_AB_ELR10 beats R_AB, each by more than 0.002, the national arm is fine-tuned at
1.0x (`OPERA_PAPER_EXTERNAL_ENCODER_LR=1.0`); otherwise at 0.1x.

- J_ELR10 − J: **[fill]**; R_AB_ELR10 − R_AB: **[fill]**; R_AB_ELR3 − R_AB: **[fill]**
- Stage-11 reading designs at 1.0x vs 0.1x: **[fill]**
- **National encoder rate:** 0.1x expected (J_ELR10 0.7653 < J 0.7711, so the
  rule cannot trigger); confirm on paired tuning AUC **[fill]**. If 1.0x: the national-vs-local contrasts
  (E1–E4) use the local fits at 1.0x (J_ELR10 and the selected design at 1.0x,
  added as a stage-12 fit if not already present), so they stay matched.

## 3. Checkpoint-selection sensitivity (added 1 Nov, before any test data)

Checkpoints are selected on the lowest tuning loss (endpoint-group horizon BCE),
which can precede the best tuning AUC by a few epochs as predictions become
overconfident. Every candidate named in sections 1–2 and J are evaluated on the
test set at both `best.ckpt` (primary) and `last.ckpt` (sensitivity), with AUC,
IPA and calibration. No selection is made on the test set.

## 4. National arm (population pretraining)

Protocol: national encoder used directly (E) and after continued pretraining on
the cohort (EA); for each, the joint model (J) and OPERA (R, carrying the
selected design's readout; `task_pooling_config` when a stage-9 design is
selected); data efficiency: J at 10%, 25% and 50% of training patients for the
cohort (L) and adapted national (EA) encoders. Encoder rate per section 2.
Contrasts: E1–E4, N10–N50, T5/T5E. The national fits never use test patients;
they are evaluated on the same test set when ready, with no change to anything
in this note. Expected pattern to report: gains concentrated in sparsely
measured laboratory outcome groups (value resolution) and in rare cells.

## 5. Analyses for a stage-9 primary (descriptive; tuning and test)

- Gain decomposition: J → + reading (J_OA / J_OA_LR) → + own corrections
  (selected), and R_AB, by outcome family and by cell support.
- *Amended 1 Nov, evening (before test data).* Table 2 reports the ladder by
  mechanism, each row a pair of models differing in one element:
  own disease and outcome corrections (R_PF_OFF − J, `OFF_J`); + each outcome's
  own reading (selected − R_PF_OFF, `OAOWN_OFF`); total (selected − J, `O2`);
  learned routing instead, on the same corrections (R_AB − R_PF_OFF, `AB_OFF`).
  R_PF_OFF, R_AB, J_OA, J_OA_OWN_NODIS and J_OA_OWN_NOOUT are therefore scored on
  the test set with the locked models (section 7, step 2). The ablations
  (`OAOWN_DIS`, `OAOWN_OUT`) are reported in the supplement, descriptively.
- Specialisation per outcome: attention gate |tanh(g)| and the size of the
  outcome-specific term relative to the shared summary, against training events.
- Outcome reading map: clustered similarity of outcome queries (tied basis
  coefficients where applicable).
- Disease deviation: size of each diagnosis's own correction against patients.
- What outcomes read: per-head attention (or attribution) by concept group and
  time window, aggregated; never head-averaged.

## 6. Contrasts

As registered in `opera/configs/manifests/final_contrasts_20261014.yaml` and
`opera/run/analyse_locked_predictions.py` (FULL_OWN, TASK_POOLING, ENCODER_LR,
TASK_ROUTED, PARTIAL_POOLING, SAMPLING_SENSITIVITY), each computed when its
models are present; codes listed in `opera/run/build_paper_numbers.py`. The
borrowing ladder, tabular (T1–T10) and clinical-index comparisons are unchanged.

## 7. Opening the test set

1. Fix this note (all **[fill]** completed), commit, record the commit hash.
2. Test predictions for every locked model (`evaluate --split held_out`), at
   `best.ckpt`, and at `last.ckpt` for section 3.
3. Locked analysis on test (`analyse`, primary estimand), paper numbers, figures.
4. National arm evaluated on test when its fits finish, unchanged protocol.

Signed: **[fill]** (date, commit)
