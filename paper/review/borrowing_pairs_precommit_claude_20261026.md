# Borrowing pairs: Claude's pre-committed ratings (26 Oct 2026)

Written before any source-removal result existed: the tuning source-removal
pass had not been run or read, and the test set is closed. Known at the time:
the stage-2 training curves (routing entropy about 0.4 from epoch 1 in both R
and RSI, shared slot near zero), the cohort sizes in Table 1, and the pair list
in `opera/configs/borrowing_hypotheses.yaml`. Commit this file before the
tuning results are read; the commit time is the record.

Rated as the reviewers are asked to rate: how much would what is learned in the
source disease help predict the stated outcomes in the target? The expected
gain is the source-alone AUC gain (own disease plus source vs own disease only),
in AUC points on the test set.

| Pair | Source → target | Outcomes | Rating | Expected gain | Why |
|---|---|---|---|---|---|
| P1 | CLL → SLL | all outcomes | strong | +0.5 to +1.5 | One disease in two presentations; the CLL correction is trained on about 3,200 patients, the SLL one on about 870. The clearest pair in the list. |
| P2 | PTCL → TCL | disease course | moderate | +0.3 to +1.0 | "Unspecified T-cell lymphoma" is mostly PTCL coded without a subtype. TCL's own correction rests on about 150 patients, so borrowing has room to help; 60 test patients make the interval wide. |
| P3 | MM → CLL | infection | weak | 0 to +0.2 | Shared hypogammaglobulinaemia, but first-line drugs and steroid exposure differ, and CLL's own correction is large. |
| P4 | CLL → MM | infection | weak | 0 to +0.1 | As P3, with an even larger target; myeloma infection risk is driven by induction (steroids, IMiDs, transplant). |
| P5 | HCL → DLBCL | disease course | none | 0 (±0.1) | No shared biology or course; DLBCL's own correction dominates. |
| P6 | BL → LBL | neutrophil, white-cell, platelet | weak | 0 to +0.3 | Both get intensive regimens, but severe cytopenia is near-universal in both, which leaves little to rank. |
| P7 | PTCL → AITL | disease course | moderate | +0.2 to +0.8 | Same CHOP-based treatment, similar course, often hard to tell apart; AITL's own correction rests on about 350 patients. |
| P8 | ALCL → CLL | all outcomes | none | 0 (±0.1) | No shared biology, treatment or complications. |
| P9 | MM → HL | all outcomes | none | 0 (±0.15) | No shared biology or treatment. The one control where a large source could add a small generic effect (age, comorbidity), so the widest tolerance. |
| P10 | CLL → LPL | cardiovascular | weak | 0 to +0.2 | The BTK-inhibitor link is real, but first-line Waldenström treatment in Denmark has mostly been chemoimmunotherapy, and atrial fibrillation risk is mostly age and history, which the encoder already sees. |
| P11 | BL → LBL | metabolic and cell turnover | moderate | +0.2 to +0.6 | Tumour lysis markers (urate, phosphate, LDH) follow tumour burden and intensive induction in both. |
| P12 | EMZL → SMZL | disease course | moderate | +0.2 to +1.0 | Shared marginal-zone prognosis (MZL-IPI); SMZL's own correction rests on only about 70 training patients, so it should lean on its relatives. Wide interval (71 test patients). |

## Predictions for the analysis as a whole

1. **The set test.** The pairs rated strong or moderate (P1, P2, P7, P11, P12)
   gain more than the other pairs of their domains. About 70% confident.
2. **The controls.** P5, P8 and P9 stay within ±0.25 points. About 85%
   confident; P9 is the one most likely to drift.
3. **Size.** Most single pairs are small (below 0.25 points), because the
   routing is sharp: entropy about 0.4 means one source carries most of the
   weight. The largest gains go where the target's own correction is weak and
   one relative is nearly the same disease (P1, P2, P12).
4. **Small targets borrow more.** Across all pairs in the borrowing map, the
   source-alone gain falls with the target's training size.
5. **Per-pair labels.** Of the 9 hypotheses, at most 3 come out "better"
   against the 0.25-point margin on the test set (most likely P1, then P2 or
   P12); the rest are "equivalent" or "inconclusive".

Where I am least sure: P11 (I may be overrating how much LDH and urate differ
between BL and LBL patients), and whether sharp routing means each disease
leans on its own correction (then every pair is small, and prediction 1 fails)
or on its nearest relative (then P1, P2 and P12 are large).
