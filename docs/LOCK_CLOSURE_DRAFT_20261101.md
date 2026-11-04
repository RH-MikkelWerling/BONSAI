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
selected. Secondary summary: the selected design averaged over its seeds
(amendment of 2 Nov); not to be confused with the secondary model (R_AB, below).

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
J_OA_OWN s42 0.7916, s43 0.7865, s44 0.7859, s45 0.7829 at epoch 10 while running
(training log, for orientation only; the decision below uses the paired primary
estimand); R_AB 0.7836 / 0.7834 and R_PF_OFF 0.7827 / 0.7823 on seeds 42 /
43. The reading designs are seed-sensitive, the others are not, and s42 is the
highest of the four seeds [on the training log; on the primary estimand s43 is
0.0007 above s42, see the filled values below]. Three consequences, recorded here because they
follow from these results:

1. *Scope of the parsimony rule (clarification).* The design seed spread is now
   about 0.009 on the training log (0.0052 on the primary estimand, which the
   decision uses; see below), as large as the gap between J_OA_OWN and R_AB. Read literally,
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
   `OAOWN_OFF_s43`, `AB_OFF_s43`, `OAOWN_J_s43`).

*Note (2 November, afternoon).* The seed values above are the training-log
validation metric (macro AUC over outcomes), not the primary estimand. On the
primary estimand (paired IPCW tuning AUC, pooled over horizons, every cell
equal) the ladder is similar on seeds 42 and 43: own corrections over J +0.0079
and +0.0086, each outcome's own reading on top +0.0054 and +0.0072, learned
routing instead +0.0029 and +0.0035. The seed spread used by the rule is
computed on the primary estimand (s42 against each other seed, paired), not
from the training log. Cross-seed contrasts require one cell key across seeds;
the analysis now pools seed repeats under the primary seed's key (each model is
one fit under its own name; estimates unchanged).

*Filled 3 November 2026 from the paired tuning analysis (primary estimand,
pooled over horizons, 600 cells; `analysis/tuning_1103`, which holds every fit of
`tuning_1102` plus J_OA_OWN_PH seed 43; `scripts/lock_numbers.py`).*

- Design seed spread: s42 − s43 −0.0007, s42 − s44 +0.0045, s42 − s45 +0.0045,
  so the four seeds lie at 0, +0.0007, −0.0045, −0.0045 relative to s42;
  **range 0.0052**.
- J_OA_OWN four-seed mean vs s42: s42 is +0.0021 (−0.0015 to +0.0056) above it.
- Noise floors: R_AB s43 − s42 −0.0006 (−0.0053 to +0.0041); J s42 − s43
  +0.0019 (−0.0020 to +0.0058).
- Candidates vs R_AB (gain, 95% CI): J_OA_OWN_PH +0.0063 (−0.0010 to +0.0139),
  J_OA_OWN_PH_X +0.0062, J_FPF_PH_OWN_PH +0.0054, J_OA_OWN_MAX +0.0035,
  J_OA_OWN_PE_X_SW +0.0029, J_OA_OWN_PE +0.0028, J_OA_OWN +0.0025 (−0.0046 to
  +0.0101), J_OA_OWN_SW20 +0.0023, J_OA_OWN_X_SW +0.0023, J_OA_OWN_X +0.0019,
  J_OA_OWN_SW100 +0.0017, J_OA_LR_OWN +0.0010, R_PL8 −0.0002; every other
  candidate below R_AB's noise floor (full list: design search, supplement).
- (a) Base rule: every design above qualifies (≥ −0.0006); the highest is
  J_OA_OWN_PH (+0.0063), so the winning family is the reading designs.
- (b) Parsimony: within the spread of the best (≥ +0.0063 − 0.0052 = +0.0011)
  are J_OA_OWN, _PH, _PH_X, J_FPF_PH_OWN_PH, _MAX, _PE, _PE_X_SW, _SW20, _X_SW,
  _X and _SW100; the simplest is J_OA_OWN. J_OA_OWN_PH beats it by +0.0038
  (+0.0002 to +0.0075), less than the spread, so it is not selected. On seed 43
  the same comparison is −0.0006 (−0.0037 to +0.0025).
- Check (2 November): J_OA_OWN beats R_AB on both shared seeds: seed 42
  +0.0025, seed 43 +0.0037 (= OAOWN_OFF_s43 − AB_OFF_s43). Passed.
- **Selected primary design: J_OA_OWN** (`J_OA_OWN_L`, `seed42/L/fix_J_OA_OWN`).
  OPERA is this design; which encoder it uses (cohort or national) is decided
  by section 4b, so if a national fit wins there, OPERA is that fit.
  For the local analysis, `OPERA_PAPER_OPERA_MODEL=J_OA_OWN_L`;
  `OPERA_PAPER_ROUTED_MODEL` stays `R_AB_L` in the analysis environment, so that
  `R_L` keeps meaning learned routing in every `_AB` contrast. It is set to
  `J_OA_OWN_L` only when preparing the national bundle (section 4), where it
  selects the national OPERA readout.

**Secondary model (amended 3 November, before any test data).** Learned routing,
R_AB (`R_AB_L`, seed 42), is a prespecified secondary model: the second kind of
borrowing model. On the test set it is reported beside OPERA in Table 2 and the
ladder (R_AB against J, `R1`; against own corrections, `AB_OFF`; against OPERA,
`OAOWN_AB`; on seed 43, `AB_OFF_s43`) and in its routing summary. OPERA
(the J_OA_OWN design, with the encoder chosen by section 4b) is the primary for
every headline, clinical-index and figure comparison; the clinical-score
configuration names the selected OPERA fit (`J_OA_OWN_L` unless section 4b
selects a national one; it named J_L until 3 November). Both models are reported
in full on the test set. Their roles are fixed now: if learned routing does
better on test, that is reported as such, but it does not become the primary;
if it does worse, it is not dropped.

*Amended 3 November:* learned routing (R_AB) is described in one secondary,
descriptive analysis from the routes exported during its evaluation: per cell,
the own disease's and own outcome's share of the routes against the uniform
share, the normalised entropy over the other sources, and learned against frozen
routing (R_PF5E against R_PF5E_FROZEN, `PF5E_FZ`). The borrowing map and source
removal are run only if the routes carry structure beyond the own source: the
normalised entropy of the routes over the other sources below 0.8 in at least
20% of cells, or learned routing better than frozen (95% interval above 0). On
tuning (28 October export of R_AB) neither holds: the own disease took a median
23% of the disease routes (uniform 4.2%) and the own outcome 8% (uniform 1.4%),
but over the other sources the routes were nearly uniform (median normalised
entropy 0.99 for diseases and outcomes; below 0.8 in 4% and 6% of cells; the
most weighted other source a median 1.6 and 2.2 times its uniform share), and
learned and frozen routing tied (+0.0003 against R_AB). So no borrowing map or
source removal is run: learned routing reduced to the own source plus an even
mix of the others, i.e. partial pooling. The same check is reported on the test
routes; it does not reopen the decision.

## 2. Encoder learning rate (stage 10–11; amendment of 1 Nov)

Primary analyses stay at 0.1x for every arm. The encoder rate is a sensitivity
analysis. **National-arm rule:** if, on paired tuning AUC, J_ELR10 beats J and
R_AB_ELR10 beats R_AB, each by more than 0.002, the national arm is fine-tuned at
1.0x (`OPERA_PAPER_EXTERNAL_ENCODER_LR=1.0`); otherwise at 0.1x.

- J_ELR10 − J: −0.0029 (−0.0108 to +0.0054); R_AB_ELR10 and R_AB_ELR3 were not
  completed (held 1 November once J_ELR10 had lost: the rule needs both).
- Stage-11 reading designs at 1.0x vs 0.1x: J_OA_OWN_ELR10 − J_OA_OWN −0.0062
  (−0.0150 to +0.0023); J_FPF_PH_OWN_PH_ELR10 about −0.0156 against its 0.1x fit (difference of
  the two contrasts against R_AB on the same cells).
- **National encoder rate:** 0.1x expected (J_ELR10 0.7653 < J 0.7711, so the
  rule cannot trigger); confirmed on paired tuning AUC: **0.1x**. If 1.0x: the national-vs-local contrasts
  (E1–E4) use the local fits at 1.0x (J_ELR10 and the selected design at 1.0x,
  added as a stage-12 fit if not already present), so they stay matched.

## 3. Checkpoint selection (amended 3 November, before any test data)

Every fit is evaluated at `best.ckpt`, the checkpoint with the lowest tuning
loss (endpoint-group horizon BCE), as prespecified. The `last.ckpt` sensitivity
analysis added on 1 November is dropped before the test set is opened: it would
double test evaluation for a question the prespecified rule already settles.
No selection is made on the test set.

## 4. National arm (population pretraining)

Protocol: national encoder used directly (E) and after continued pretraining on
the cohort (EA); for each, the joint model (J) and OPERA (R, carrying the
selected design's readout; `task_pooling_config` when a stage-9 design is
selected); data efficiency: J at 10%, 25% and 50% of training patients for the
cohort (L) and adapted national (EA) encoders. Encoder rate per section 2.
Contrasts: E1–E4, N10–N50, T5/T5E.
*Amended 3 November, before any national or test data.* The secondary model,
learned routing (R_AB), is also fitted on both national encoders (`seed42/E/AB`,
`seed42/EA/AB`; models `R_AB_E`, `R_AB_EA`), with its own contrasts against its
local fit and XGBoost (`E2_AB`, `E4_AB`, `A2_AB`, `T5E_AB`, `T5_AB`, `I1_AB`,
`I3_AB`), and OPERA against it on the same national encoder (`OAOWN_AB_E`,
`OAOWN_AB_EA`). The national mechanism ladder: own corrections only (R_PF_OFF)
on both national encoders, and the primary without its disease or its outcome
correction on the adapted one, each the local fit with the national sequences,
vocabulary and encoder (`seed42/{E,EA}/OFF`, `seed42/EA/NODIS`,
`seed42/EA/NOOUT`); contrasts `OFF_J_{E,EA}`, `OAOWN_OFF_{E,EA}`, `AB_OFF_{E,EA}`,
`OAOWN_DIS_EA`, `OAOWN_OUT_EA`, and whether the corrections and the reading
steps grow from the cohort to the adapted national encoder (`OFF_GROWTH_EA`,
`READ_GROWTH_EA`). Descriptive, never selected. National fits in total: J,
OPERA, R_AB and R_PF_OFF on E and EA (8), the two ablations on EA (2), and the
data-efficiency fits of J on EA (3): 13, plus continued pretraining. With a reading-design primary, the manifest's national OPERA contrasts
compare like with like: E2, E4, I1 and I3 take J_OA_OWN_L, not R_L (learned
routing), as the local reference; the encoder-origin panel does the same, with
learned routing in a panel of its own. The national fits never use test patients;
they are evaluated on the same test set when ready, with no change to anything
in this note. Expected pattern to report: gains concentrated in sparsely
measured laboratory outcome groups (value resolution) and in rare cells.

## 4b. Which encoder OPERA uses (amended 3 November, before any national or test data)

The analysis lock (15 October) chooses OPERA among the cohort (L), national (E)
and adapted national (EA) encoders: highest tuning panel AUC, ties within the
noise floor going to the simpler model (cohort before national). Made precise:

- Compared: J_OA_OWN_L against R_E and R_EA (both carrying the J_OA_OWN
  readout), paired tuning AUC, primary estimand, pooled over horizons.
- Noise floor: **the J_OA_OWN four-seed spread, 0.0052** (section 1). A national
  OPERA is selected only if it beats J_OA_OWN_L by more than 0.0052; if both do,
  the higher one, unless they differ by less than 0.0052, in which case E (no
  adaptation) is the simpler.
- Decided on tuning predictions only, before the national fits' test results
  are looked at, and recorded here with the values.
- If a national OPERA is selected, it becomes OPERA (`opera_model`) for O1–O7,
  the headline, benchmark, calibration, risk-group and patient-map figures, and
  the clinical-index comparisons (the clinical-score configuration is set to
  it). The mechanism ladder and ablations, the design search and the seed
  analyses stay local, since their fits exist only on the cohort encoder; the
  national arm reports its own partial ladder (OPERA against J and against R_AB
  on the same encoder). The local results are reported in full either way.

Values (to fill when the national tuning predictions exist): R_E − J_OA_OWN_L
**[fill]**; R_EA − J_OA_OWN_L **[fill]**; OPERA encoder: **[fill]**.

## 5. Analyses for a stage-9 primary (descriptive; tuning and test)

- Gain decomposition: J → + reading (J_OA / J_OA_LR) → + own corrections
  (selected), and R_AB, by outcome family and by cell support. Both orders are
  reported; Table 2 uses the amended order below (corrections first). The
  router's own decomposition is reported beside it: global only (N_A), own
  corrections inside the router (RBH_AB), routing (R_AB): `router_own_AB`,
  `router_borrow_AB`, `router_both_AB`.
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

## 5b. Pretraining controls (recorded 3 November, before any test data)

The two controls of the analysis lock are reported on the test set beside the
joint model: no pretraining (the joint model from a randomly initialised encoder,
J_RI; `P0` = J_L − J_RI) and the readout alone on the frozen cohort encoder
(J_FZ; `P7` = J_L − J_FZ). They appear in Table 2 under pretraining and in the
supplement; neither is selected.

## 6. Contrasts

As registered in `opera/configs/manifests/final_contrasts_20261014.yaml` and
`opera/run/analyse_locked_predictions.py` (FULL_OWN, TASK_POOLING, ENCODER_LR,
TASK_ROUTED, PARTIAL_POOLING, SAMPLING_SENSITIVITY, TASK_ABLATION, TASK_DESIGN,
TASK_SEEDS, NATIONAL_ROUTING, the router decomposition and the design-search
contrasts `DS_<model>__OAOWN` and `DS_<model>__AB`), each computed when its
models are present; codes listed in `opera/run/build_paper_numbers.py`. The
borrowing ladder, tabular (T1–T10) and clinical-index comparisons are unchanged.

## 6b. Headline test family (amended 3 November, before any test data)

The fixed eight-member family is P1, P2, P5, T2, O1, router_borrow_AB,
E1 and O2. Use primary equal-cell pooled-all-horizon AUC, seed 42, two-sided
bootstrap sign-tail tests, Holm alpha 0.05. E1 is the fixed J_E versus J_L
comparison, not whichever national contrast gives the lowest p-value.

The evaluator exports `headline_holm.csv`, retaining raw p-values, adjustment
inputs and adjusted p-values. Inputs have an explicit Monte Carlo resolution
floor of 2/(B+1), where the finite draw count B is available, rather than
interpreting an observed zero tail count as an exactly zero p-value. These
remain approximate bootstrap tests. Bayesian sign tails are not treated as
frequentist p-values. Missing planned members occupy adjustment slots with
p=1; their reported adjusted p-values stay missing and the family is labelled
provisional, not silently reduced. Tuning tables are labelled rehearsal.
Clinical comparisons remain descriptive. Same-encoder reporting contrasts
are not extra family members. Ordinary 95% CI labels remain descriptive and
are distinct from the adjusted hypothesis-test decisions.

## 6c. Clinical indices (clarified 3 November, before any test data)

Clinical-index analyses are estimation-focused, reported for every prespecified
index/population and endpoint with paired uncertainty. They are descriptive and
outside the confirmatory Holm family. We do not require a co-primary AND test
at both risk extremes, or interpret positive point estimates as proof of
non-inferiority or simultaneous superiority across all indices. Any previous
wording including clinical indices in the Holm family is superseded here.

Primary: `cutoff_rule: cohort_size_matched`. On the same evaluable patients,
match the original categories' sizes using baseline index categories and model
risk alone. Report survival/event risk and paired differences at both extremes,
category sizes and uncertainty. Repeat the grouping procedure within paired
bootstrap replicates. This validates a ranking/grouping rule, not a universally
fixed absolute cutoff. No test outcomes determine these group assignments.

Complementary: freeze training-derived low-risk KM and high-specificity anchors
before test evaluation. Report additional patients and detected events alongside
ACHIEVED test survival and specificity; equality on training does not guarantee
equality on test. Nominal intervals are labelled pointwise, not simultaneous.
Counts of favourable estimates are descriptive index/population comparisons,
not independent replications or a statistical demonstration of superiority.

Additional descriptive evaluation-cohort KM matching is now prespecified at
both ends. Within the same index-scored patients, estimate the native lowest
and highest categories' KM survival at the chosen horizon. Search all attainable
model-risk boundaries, preserving whole ties. Select the largest lowest-risk
prefix with survival >= the native Low estimate, and separately the largest
highest-risk prefix with survival <= the native High estimate. Zero tolerance,
no smoothing, and no assumption that empirical survival is monotone in size.
The two binary selections may overlap and do not define a risk-group partition.

Native and model groups each require >=20 patients and >=10 at risk at the
horizon. Missing native groups, insufficient support and absent matches are
reported, not assigned zero gain. For 200 paired patient-bootstrap replicates,
re-estimate the native KM target and repeat the model-boundary search. Export
valid matching counts/fractions and pointwise percentile intervals; model-match
intervals require >=80% valid draws. These intervals can be unstable near
crossings or degenerate in zero-event groups and do not remove selection
optimism. Save the unsmoothed survival/size frontier with all support flags and
mark both clinical and selected model operating points. These are descriptive
evaluation-cohort estimates, not validated future deployment cutoffs. They are
outside Holm and complement, rather than replace, equal-size comparisons and
training-frozen anchors. Configuration: `evaluation_km_matching`.

Use each index's intended disease population, endpoint and assessment time as
closely as the frozen panel permits. Record shorter horizons and explicit proxy
definitions. Unmapped registry names/versions and PFS/EFS proxies require review;
metadata containing an index alone does not establish disease applicability.

## 6d. Secondary weighting and central encoder contrasts (3 November)

The primary equal-cell estimand is unchanged. The secondary `disease_balanced`
summary weights disease, approved training `loss_endpoint`, outcome variant,
then horizon equally at each level on the fixed evaluable roster. It requires
the family map; display groups are not substitutes. Reaggregate existing paired
draws without refitting. Older diagnostic equal-outcome weights remain available
only under the explicit name `disease_outcome_balanced`.

Keep `O2 = final OPERA - J_L` as the total-system comparison. Add
`O2_MATCHED = final OPERA - J_<selected encoder>` as the central same-encoder
comparison. Both are reported even if numerically identical for local OPERA;
they are not two independent tests of the same claim. `O_READ_MATCHED` compares
the selected design with `R_PF_OFF` on its encoder; locally,
`O_CORRECTIONS_MATCHED` compares selected `J_OA_OWN_L` with reading-only `J_OA_L`.
`O2_LOCAL` retains the selected local design against J_L regardless of the
final encoder choice. `O_ROUTING_MATCHED` compares final OPERA with learned
routing on the same encoder. These component contrasts do not by themselves measure optimisation-independent
causal effects. The best design on each encoder is fixed by tuning, never test.
These central reporting additions do not silently expand the confirmatory Holm
family; its exact primary hypotheses must be enumerated before signature.

## 7. Opening the test set

1. Fix this note (all **[fill]** completed except section 4b, which is filled
   from the national tuning predictions before their test results are read),
   commit, record the commit hash.
2. Test predictions for every registered fit (`evaluate --split held_out`), at
   `best.ckpt`. Every design is shown on test descriptively (design search);
   only the prespecified primary and contrasts carry the paper's claims.
3. Locked analysis on test (`analyse`, primary estimand), paper numbers, figures.
4. National arm evaluated on test when its fits finish, unchanged protocol.
5. Attributions: concepts and expected directions are fixed in the attribution
   configuration as of this commit; the review of their code mapping against the
   cohort vocabulary (and the national one, if section 4b selects a national
   OPERA) is a separate, dated commit made before any attribution summary on the
   test set is read. Until then the summaries omit named concepts.

Signed: **[fill]** (date, commit)
