# Router release audit: first server report reviewed, 29 October 2026

**Disposition: first replay/provenance check passed for the reported inputs.**
This establishes the intended revised seed-42 fit, reproducibility of sampled
predictions and numerical consistency of the neural evaluation. It does not
by itself constitute the complete test-release decision.

Received file: `/Users/mikkelwerling/Downloads/report (1).json`.
Preserved byte-for-byte at
`scratch/paper_reading_20261028/server_release_reports_20261029/router_replay_report.json`.
Its checksum and local validation summary are in `router_replay_receipt.json`
in that directory. The report contains aggregate checks and server paths.
No checkpoint, patient records or internal replay ID list were transferred.

## Checks that passed

| Check | Evidence | Interpretation |
|---|---|---|
| Correct primary router | R_L and R_AB_L both identify `seed42/L/fix_R_AB`, scientific run `79d39ad1e7cb51472026` | The alias points to the revised router, not the original collapsed fit. |
| Selected checkpoint | Primary SHA-256 `2d58ab74d124065eb08b719237418dcc0b1507f2ce7b5fafda82b53b87bc1524`; best epoch 12, step 1469 | Evaluation uses the checkpoint selected by endpoint-group-macro tuning BCE. |
| Artifact hashes | Bundle/config, training-selection/checkpoint and prediction-file checks are true for all six artifact entries | Six entries represent five distinct fits, because R_L and R_AB_L are aliases. |
| Fresh replay | Joint and router each replay 46 selected patients, 3,057 patient–outcome rows at every horizon, six horizons | No sampled probability mismatches under `atol=1e-5, rtol=1e-4`. This is an implementation sample, not a performance sample. |
| Joint/router coverage | Replay command requires exact full saved-file patient–cohort–outcome and time/event alignment before replay | A mismatched population or follow-up would have raised an error. |
| Metric reconciliation | Original router, joint, revised router and its alias all use `cif_ipcw`; zero missing saved estimates and zero finite-status mismatches | 18,900 AUC/Brier entries per artifact, including unsupported entries, reconcile to the saved paper estimates. |
| Numerical precision | Native-versus-paper maximum error ≤4.72e−15; recomputed-versus-saved ≤1.11e−16 | No discrepancy in the checked neural AUC/Brier calculations. |
| Replay code | All three reported source-file hashes match the local reviewed files | Replay, native evaluator and component-check code match. The report does not hash every imported module. |
| Source-map identity metadata | Removal manifest identifies R_AB_L / `fix_R_AB`; routes share the primary prediction directory | Exported provenance agrees with the selected router. Directory/manifest identity does not independently certify ordered source labels or every cached removal array. |

Largest replay differences were 1.0431e−6 for the joint model and 7.7486e−7 for
the router. These are small numerical differences, below the declared tolerance.

## Routing health is reproduced for seed 42

The revised router passes every recorded support-weighted health criterion:

- Most frequent top source: DLBCL, 15.07% of rows, below the 50% limit.
- Six distinct modal top borrowed sources across target diseases, above the
  minimum of three.
- Median effective contributing-source count 76.018, above 1.5.

Its unweighted overall effective-source summary is 70.8026. In contrast, the
original router has an effective-source summary of 1.5948 and selects the
overall-survival source as top in 99.82% of rows.

This is evidence that the recorded revised routing is non-collapsed. Its
very broad allocation is not evidence of sharply selective clinical source
choice, and one seed does not establish cross-seed stability.

## Borrowed corrections are active

Fresh replay includes two fixed-checkpoint component diagnostics. At one year:

| Comparison with the full router | Median absolute risk difference | 90th percentile | Maximum |
|---|---:|---:|---:|
| Router checkpoint encoder/head alone | 0.03701 | 0.21683 | 0.57047 |
| Borrowed sources set to zero, keeping own/global corrections and gates | 0.01896 | 0.11745 | 0.38812 |

The second median is about **1.90 percentage points of predicted risk**.
Borrowed terms are therefore not bypassed in the sampled forward pass.
These absolute differences do not say whether those changes improve accuracy.
They are pooled across different outcomes in a small deterministic sample.

Encoder/head alone refers to the router's own trained encoder/head. It is not
the separately trained joint model. Zeroing borrowed terms is also not the
primary tuning-mean removal: the replay removes borrowed disease and outcome
terms, whereas the main source-removal analysis replaces other-disease terms
by tuning means and retains the outcome bank. These are distinct interventions.

## Performance interpretation remains modest

Saved equal-cell tuning AUC comparisons across 3,004 supported cell–horizon
entries are reproduced in the report:

| Contrast | Difference | Saved 95% interval |
|---|---:|---:|
| Revised router − joint | +0.008679 | +0.003134 to +0.014581 |
| Original router − revised router | +0.000820 | −0.002084 to +0.003718 |
| Full revised router − trained own-task control | +0.004529 | +0.000411 to +0.008884 |
| Trained own-task control − global-only control | +0.007386 | +0.001373 to +0.013729 |
| Full revised router − global-only control | +0.011915 | +0.006719 to +0.017594 |

At one year, revised router minus joint is +0.010787 (+0.003386 to +0.018950)
over 600 cells. Full minus own-task is +0.005182 (−0.000091 to +0.010748).
The one-year original-minus-revised difference is −0.000282, with an interval
crossing zero. All comparisons are tuning evidence, conditional on selected
fits. Correlated horizons are not independent replications.

The report supplies a defensible empirical reason to retain the revised
router: it removes the original allocation collapse without a clear predictive
penalty in these paired summaries. It does not show that revised routing is
uniformly better or that the original model is invalid as a predictor.

## Why the pairwise maps can still look nearly empty

The report reproduces tiny shrunken individual-pair estimates:

- Median absolute shrunken effect: 1.60e−6 AUC points.
- Maximum absolute shrunken effect: 6.51e−5 AUC points.
- Median absolute raw estimate: about 0.0578 AUC points.
- Maximum absolute raw estimate: about 3.1111 AUC points.

This diagnostic has 460 off-diagonal rows; the earlier finite-only summaries
have fewer pairs. Do not interchange the denominators or infer changed data
solely from the different row counts.

Together with replay and correct fit identity, these results argue against a
simple wrong-router or inactive-path explanation. They remain compatible with
diffuse contributions, redundancy and imprecise individual source estimates.
They do not establish which explanation is responsible. Ordered bank labels
and removal-cache provenance still need their specific checks.

The maps should communicate weak individual-source evidence honestly. A large
change after removing many borrowed terms does not require a large, identifiable
effect from any one disease source.

## Remaining release checks

1. Ordered disease/outcome source roster versus the checkpoint/configuration
   and support export. Matching counts or paths alone does not prove ordering.
2. Stage-3 decision report and second-seed health if claiming stable routing
   across seeds. The received report concerns seed 42.
3. XGBoost/pooled-XGBoost reconciliation and raw feature timing/profile checks.
   The neural replay does not include X or XP.
4. Matching fit identities for refreshed clinical scores, arrows, attributions
   and recent-record omission outputs. Matching embedding IDs is useful but
   not an independent checkpoint/projection provenance check.
5. Fixed national/local roster, threshold artifacts, endpoints and inference
   rules in the final release record before held-out inspection.

No new fit or repeated GPU replay is justified by this report alone. Continue
with the pending checks and reuse the completed mean/zero removal analyses.

### Small correction to the new cross-family CPU audit

While reviewing this report, the inventory confirmed that `OPERA` is normally
created as an analysis alias rather than a physical artifact. The newly added
explicit-model reconciliation option initially did not resolve that alias.
It now resolves through `opera_model` while validating the source table's
original model declaration. With the earlier code, use
`--metric-models X XP J_L R_L`; after syncing the correction, `OPERA` works too.
This affects the additional CPU audit only. The received neural replay uses
its original default and is unaffected. Five targeted audit/replay tests pass,
including alias resolution and detection of a deliberately changed saved metric.
