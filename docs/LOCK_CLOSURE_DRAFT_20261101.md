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
selected. Secondary: the average of the selected design's two seeds.

**Training-log indication (not the decision):** J_OA_OWN 0.7916, J_OA_OWN_PH
0.7912 and J_OA_LR_OWN 0.7910 at the selected checkpoint against R_AB 0.7836.

- Noise floor (R_AB s42 − s43, paired tuning AUC): **[fill]**
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
2. Test predictions for every locked model (`evaluate --split test`), at
   `best.ckpt`, and at `last.ckpt` for section 3.
3. Locked analysis on test (`analyse`, primary estimand), paper numbers, figures.
4. National arm evaluated on test when its fits finish, unchanged protocol.

Signed: **[fill]** (date, commit)
