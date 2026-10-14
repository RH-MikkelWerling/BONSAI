# Approved 69-outcome absolute-position comparison (internal; 6 October 2026)

Source: user-shared aggregate output from `opera.run.compare_approved_abspos_v2` on the secure server. This note does not inspect patient-level data or held-out outcomes. Run root: `data/models/approved_outcome_abspos_v2_idtype_v1/{fourier,time2vec}`.

| Tuning measure | Fourier | Scaled Time2Vec |
| --- | ---: | ---: |
| Validated epochs | 30 | 30 |
| Best epoch | 24 | 26 |
| Best endpoint-group macro AUC | 0.744856 | 0.741486 |
| Epoch-30 endpoint-group macro AUC | 0.743355 | 0.739703 |
| Supported selection groups / cells | 52 / 376 | 52 / 376 |

The comparator checked that both runs had identical endpoint-selection inventories (414 entries) and the same selected support. Fourier's best-epoch advantage is **0.003370 AUC** (0.337 percentage points); its epoch-30 advantage is 0.003652. Fourier exceeds Time2Vec in each of epochs 21–30, so the best-epoch difference is not a lone spike. Time2Vec generally rises faster in early training. Both curves are close and plateau around 0.74; both are functioning with the approved outcomes and repaired metadata.

Family AUCs in the comparator are recorded at *each arm's own best epoch*, not at a common epoch. The largest descriptive differences (Fourier minus Time2Vec) are cardiovascular/thrombotic +0.0513, subsequent malignancy -0.0381, skeletal complications -0.0176, disease control/survival -0.0108, and treatment trajectory +0.0087. These are not a decomposition of the macro-AUC gap. Check family support and epoch-matched trajectories before interpreting them; small or correlated groups could exaggerate apparent differences.

This is a single-seed, tuning-only comparison. No confidence interval, replicated-seed estimate, held-out performance claim, or PHAIR-task comparison is available from this report. Selecting the best epoch for each arm adds selection noise. The result provisionally favors Fourier for this outcome task and reassures us that its representation runs on the server; it does not establish superiority for PHAIR.

Current training loss and checkpoint selection use endpoint-group macro aggregation, not equal clinical-family weights. The inherited family taxonomy remains a separate question. Before choosing a sharing/weighting branch, review the output of `opera.run.audit_approved_family_weighting_v2` (family members, group counts, current loss shares) and decide any regrouping prospectively. Do not use the held-out split to settle this choice or reinterpret the present comparison after changing the families.

## Family audit supplied after the comparison

The audit accounts for all 69 outcomes in 52 endpoint groups across 11 inherited families. Its `current_loss_share` is the **nominal fraction of equally weighted endpoint groups**, not a measured fraction of gradient magnitude or clinical utility.

| Inherited family | Outcomes | Endpoint groups | Nominal group share |
| --- | ---: | ---: | ---: |
| Disease control & survival | 3 | 3 | 5.8% |
| Treatment trajectory | 1 | 1 | 1.9% |
| Infection | 6 | 6 | 11.5% |
| Hematologic toxicity & support | 16 | 11 | 21.2% |
| Renal toxicity | 5 | 4 | 7.7% |
| Hepatic & coagulation toxicity | 9 | 5 | 9.6% |
| Electrolyte & metabolic toxicity | 19 | 12 | 23.1% |
| Cardiovascular & thrombotic | 4 | 4 | 7.7% |
| Acute care utilization | 3 | 3 | 5.8% |
| Subsequent malignancy | 2 | 2 | 3.8% |
| Skeletal complications | 1 | 1 | 1.9% |

An equal-family counterfactual would give every row 9.1%. It would multiply the nominal weight of the one-group families (treatment trajectory and skeletal complications) by about 4.7 and reduce the 12-group electrolyte/metabolic family to about 0.39 of its current share. This is too large a policy change to adopt merely because the families are uneven. The families also mix different kinds of targets (for example infection diagnoses, antimicrobial proxies and composites; hematologic labs and transfusion support). They are useful reporting labels, but require clinical review before becoming training or selection units.

No weighting or grouping change is approved by this audit. The Fourier/Time2Vec comparison remains a controlled encoder comparison under the existing endpoint-group objective. A later family-aware experiment should specify the taxonomy, weights, and checkpoint metric before it is run, and compare against an unchanged Fourier baseline on tuning data.

## Existing clinical display taxonomy as the starting point

The repository already has `opera/configs/outcome_display_groups_v1.yaml`, and `opera.run.materialize_primary_display_groups_v2` converts it to an exact-coverage, release-specific display taxonomy. This is a better starting point for clinical families than the 11 inherited registry families. It separates infection proxies from serious infections, divides hematologic and metabolic targets into clinically interpretable clusters, and separates hepatic enzymes, bilirubin/injury, and coagulation. It must still be reviewed against the actual 69-outcome release: some resulting groups may have only one or two *endpoint groups* even if they contain several grade-labelled outcomes, and the residual acute-care/fracture combination is not a clean clinical family.

For a training taxonomy, use endpoint-group counts, not raw outcome counts, to spot imbalance. Preserve each grade-linked endpoint group intact. Aim for clinically coherent groups of roughly a few endpoint groups each; allow a justified exception rather than mixing unrelated endpoints solely to hit a numerical target. Freeze an exact-cover mapping and audit it before a new run.

Implementation caveat: changing `outcome_families` alone changes family diagnostics and family-trunk membership but **does not change** the current `endpoint_group_macro` loss or checkpoint metric. The existing `hierarchical_support` loss averages support-weighted *individual outcome heads* within families, not endpoint groups; switching to it would simultaneously change both family weighting and grade-variant weighting. A controlled family-balanced comparison therefore needs an explicit group-within-family aggregation or a separately justified choice of that existing objective. The completed Fourier/Time2Vec comparison should remain untouched.

The versioned proposal in `opera/configs/outcome_training_families_primary_v1.yaml` now maps the 17 display groups to 12 prospective training families. On the reconstructed approved roster it covers all 69 outcomes and 52 endpoint groups exactly once, with **3–6 endpoint groups per family**. It keeps renal and electrolyte clusters, splits the former oversized hematologic family into erythroid, white-cell and platelet clusters, and retains one broad infection family (diagnoses and antimicrobial proxies remain distinct display categories). Hepatic enzymes, bilirubin/injury and coagulation combine into a five-group hepatic/coagulation family. The fracture endpoint stays in the broad acute-events/care-use family; that is a pragmatic grouping, not a claim of shared mechanism. Disease course and later malignancy is the other broad six-group family and merits clinical review before family-trunk use.

On the secure server, after syncing the new files, materialize the proposal into a **new** output directory (the command refuses to overwrite one):

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RELEASE_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
FAMILY_DIR="$PWD/data/results/approved_training_families_v1_$(date -u +%Y%m%dT%H%M%SZ)"
python -m opera.run.materialize_primary_training_families_v1 \
  --release-dir "$RELEASE_DIR" --output-dir "$FAMILY_DIR"
```

The output contains a proposed family map, an otherwise unchanged endpoint-group training config with the new family labels, and an aggregate-only JSON audit with source and output SHA256 hashes. **Do not launch from this config yet:** this step does not implement family-balanced loss, change the checkpoint metric, or verify an architecture-specific training branch. It provides a frozen, inspectable map for the next model-design decision.

The first server attempt failed because `outcome_display_groups_primary_v2.yaml` had not been materialized in the release. The family builder now derives exactly the same map in memory from `outcome_display_groups_v1.yaml` plus the release rename policy when *both* display artifact and audit are absent. It saves a display snapshot and hashes in its own new output directory, leaving the release untouched. If only one of the display artifact/audit exists, or their hashes disagree, it stops. With the old synced code, the immediate workaround is `python -m opera.run.materialize_primary_display_groups_v2 --release-dir "$RELEASE_DIR"` before rerunning the family builder; this creates the missing display file and audit in the release and refuses to overwrite an existing one.

## Next neural experiments (prospective shortlist; not launched)

The already-completed Fourier factorized run remains the endpoint-group-weighted low-rank control. The first new controls are **plain shared-head multi-horizon BCE** from each of the Fourier and scaled-Time2Vec DAPT encoders. Here `plain` means equal per-outcome macro loss; nested grade variants therefore count as separate heads. Those two runs differ only in the DAPT checkpoint. Keep the same 69-outcome files, repaired membership, sequence data, seed, and tuning-only endpoint-group checkpoint metric. This makes the plain-loss comparison separate from the completed Fourier/Time2Vec factorized, endpoint-group-loss comparison.

The next architecture branches use endpoint-group macro BCE: shared linear readout, family trunks using the proposed 12-family map, and additive versus interacting source-anchored adapters/routers trained end-to-end from Fourier DAPT with **no old outcome-head warm start**. Use unique run roots and compare matched tuning support before interpreting differences. The new family map changes family-trunk membership and diagnostic family scores, but does not by itself alter loss or selection.

Outcome attention is a second-wave readout candidate if resources allow. A genuinely family-balanced loss is a distinct objective experiment, not a configuration rename: the current `hierarchical_support` implementation also changes grade-variant weighting, so a group-within-family implementation and test are needed for an isolated comparison. The older router pilot is hypothesis-generating but was not a clean end-to-end comparison on this sealed release. The completed scaled-Time2Vec run establishes server compatibility; PHAIR use does not depend on it winning the outcome-tuning comparison. Keep held-out data closed throughout this shortlist.

The completed Fourier factorized run is already the endpoint-group-weighted **low-rank** control; do not rerun it under a new name. To prepare exact configs after the family audit succeeds, use the exact family output directory printed on the server:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
RELEASE_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
FAMILY_DIR="/exact/approved_training_families_v1_output_dir"
CANDIDATE_DIR="$PWD/data/results/approved_neural_candidates_v1_$(date -u +%Y%m%dT%H%M%SZ)"
python -m opera.run.prepare_approved_neural_runs_v1 \
  --release-dir "$RELEASE_DIR" --family-dir "$FAMILY_DIR" \
  --output-dir "$CANDIDATE_DIR"
```

The ten prepared configs complete the key matched cells and add architecture branches:

| Readout | Encoder | Plain per-outcome BCE | Endpoint-group BCE |
| --- | --- | --- | --- |
| Factorized low-rank | Fourier | `fourier_plain_factorized_bce` | already completed Fourier run |
| Factorized low-rank | Scaled Time2Vec | `time2vec_plain_factorized_bce` | already completed Time2Vec run |
| Shared linear | Fourier | `fourier_plain_shared_bce` | `shared` |
| Shared linear | Scaled Time2Vec | `time2vec_plain_shared_bce` | `time2vec_grouped_shared_bce` |

The remaining candidates are Fourier `family_trunks`, `router_additive_e2e`, `router_interacting_e2e`, and `outcome_attention`, all with endpoint-group BCE. If GPU capacity is finite, prioritize the plain shared pair and `fourier_plain_factorized_bce` to isolate the loss/readout effects, then family trunks and the matched router pair. `time2vec_plain_factorized_bce` and `time2vec_grouped_shared_bce` complete the encoder factorial when capacity allows; outcome attention is exploratory. The routers use the same DAPT-fresh, trainable-encoder setup and differ only in additive versus interacting query mode. Source and config hashes are recorded in `candidate_manifest_INTERNAL.json`. After syncing and preparing into a **new directory**, run one variant in each GPU server's own terminal. Each server's local GPU may be device 0; `OPERA_LOCAL_GPU` can override it if that server exposes multiple devices. Do not treat GPU indices on different servers as a single global numbering scheme.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
export OPERA_RELEASE_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export OPERA_FAMILY_DIR="/exact/approved_training_families_v1_output_dir"
export OPERA_CANDIDATE_DIR="/exact/approved_neural_candidates_v1_output_dir"
export OPERA_NEURAL_RUN_ROOT="$PWD/data/models/approved_neural_v1_YYYYMMDDTHHMMSSZ"
export OPERA_LOCAL_GPU=0
OPERA_LAUNCH=0 bash scripts/run_approved_neural_candidate_v1.sh family_trunks
OPERA_LAUNCH=1 bash scripts/run_approved_neural_candidate_v1.sh family_trunks
```

Replace `family_trunks` with exactly one other candidate name from the table or paragraph above on another server. `time2vec_` variants select the completed scaled-Time2Vec DAPT checkpoint; all others select Fourier. Reuse the same family/candidate directories and run-root label across servers if the filesystem is shared, but each variant writes a unique subdirectory. The launcher verifies the release, source checkpoints, repaired metadata, candidate hashes, and all required paths before requesting the GPU. It refuses an existing variant run directory. Do not use the old v4 router queue or a previous outcome-head checkpoint for these runs.
