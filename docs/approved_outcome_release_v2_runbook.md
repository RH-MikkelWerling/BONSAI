# Primary outcome release v2 (INTERNAL)

This is a new release candidate, not a mutation of the sealed first-line cohort,
the 87-outcome panel, or the historical outcome registry. Use separate R and
Python terminals on the secure server. The R command below only inspects the
installed ALP grader; no outcome-generating R script needs rerunning because
the lab/TLS snapshots and baseline-aware ALT/creatinine candidates already
exist. Patient-level parquets and exact small-cell counts stay on that server.

## Terminal 1: R environment, read-only ALP definition check

The 5 October server check found that the installed grader uses a running
prior-sample mean, not a fixed pre-index baseline. Its tier-2 threshold is
strictly >2.5× and tier-3 strictly >5× either the sample ULN (when the prior
mean is normal) or that prior mean (when abnormal); missing ULN defaults to
105. These are legacy, time-varying definitions, not CTCAE-v6 G2/G3. Retain
only with that description and clinical sign-off. The read-only R command below
can be repeated to verify the installed function version; it prints code, not
patient rows.

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
Rscript -e 'source("/ngc/projects2/dalyca_r/clean_r/load_dalycare_package.R"); x <- getAnywhere("AE_BASP_increased"); if (!length(x$objs)) stop("Installed AE_BASP_increased not found"); print(x)'
```

## Terminal 2: Python environment, build and audit the outcomes

The 5 October first attempt exported potassium successfully, then correctly
stopped because the previous `panel_decision` audit did not match the current
panel/registry/rules. The script now creates a fresh versioned panel-decision
audit against the current `final_panel` before exporting potassium. It checks
the audit's panel path and registry/rules hashes. Only set
`OPERA_PANEL_REVIEW_DIR` if intentionally reusing an already matching audit.
The first retry produced a matching fresh decision audit but the manually
supplied potassium directory lacked its manifest at the typed path. Avoid
transcribing that timestamp: leave `OPERA_POTASSIUM_DIR` unset and let the
script create a new versioned candidate. The previous directories are
preserved, not overwritten.

```bash
set -euo pipefail
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh
unset OPERA_PANEL_REVIEW_DIR OPERA_POTASSIUM_DIR
bash scripts/build_approved_outcomes_v2.sh
```

If potassium or rebuilt hepatotoxicity misses the prespecified 365-day
training/tuning support floor, the release command stops before creating the
output directory. Do not lower the floor or use held-out counts to get it
through; review that endpoint and revise the versioned policy.

Successful output contains matching `outcomes/` and
`experiment_registry_primary_v2.yaml`, a per-file SHA256 manifest, support
report, and semantic release notes. The builder replaces both ALT files,
rebuilds hepatotoxicity, adds separate relative-creatinine and numeric-
potassium targets, renames ALP/SMR/sodium labels, excludes retired/low-
support/held endpoints, and verifies the SMR start gate.

Final sign-off still requires reviewing new support and confirming that the
legacy ALP/time-varying-baseline and infection-proxy labels suit the
manuscript. Historical ALP thresholds are not claimed as CTCAE v6 G2/G3;
the broad infection union is not a confirmed-infection endpoint and inherits
the SMR-era eligibility gate. Lymphopenia and candidate TLS endpoints remain
outside the primary roster.

## After a successful release build: fine-tuning preflight

The historical `scripts/run_opera_overnight_router_and_lr.sh` is a v4 pilot:
it reads shared generated configs and an old outcome-head checkpoint. Do not
use it for this 69-outcome release. The release-local
`direct_cr_bce_lowrank_endpoint_groups_v1.yaml` is generated from the new
registry. It selects checkpoints with
`val/cr_cif_auc_endpoint_group_macro` (minimum 20 cases and 20 controls),
not the old endpoint roster or hand-weighted clinical-family average.

First regenerate cohort membership in the separate R terminal. The sealed
mapped first-line CSV is the authority for IDs, first-line dates and cohorts.
The script loads the server's `IPI` table through DALY-CARE's `load_dataset()`.
It preserves every distinct IPI row as structured JSON in the patient metadata
and a companion long-form table: multiple disease scores are not averaged or
arbitrarily selected. The manifest logs matched rows, patients with multiple
IPI records, and identical source duplicates. The builder checks released
overall-survival IDs, index dates and splits. The 26 sealed patients without
post-index follow-up remain in membership; their outcome-derived `split` is
missing unless the first-line export supplied it.

```bash
REPO="$PWD"
RELEASE_DIR="$REPO/data/results/approved_outcome_release_v2_20261005T200002Z"
FIRSTLINE="$REPO/data/results/firstline_candidate_runs/firstline_candidate_3926f42e3d51e1/firstline_tbl_mapped_CANDIDATE.csv"
METADATA_ROOT="$REPO/data/results/population_metadata_sealed_v2_$(date -u +%Y%m%dT%H%M%SZ)"
Rscript opera/R/build_sealed_population_metadata.R \
  "$FIRSTLINE" \
  "$RELEASE_DIR/outcomes/overall_survival.parquet" \
  "$METADATA_ROOT"
test -s "$METADATA_ROOT/manifest_INTERNAL.json"
```

In the Python terminal, set
`export BONSAI_COHORT_MEMBERSHIP="$METADATA_ROOT/population_metadata.parquet"`
using the exact output directory printed by the R script. Inspect the manifest
and coverage counts before training. The released config currently sets
`ipi_score_col: null`: IPI enriches metadata for later score comparisons, but
does not enter 69-outcome adaptation or checkpoint selection. Clinical-family
labels remain for diagnostics; the approved loss and checkpoint selection are
both endpoint-group macro, not family-weighted.

The build script generates and verifies the new configs inside its fresh
release directory; it does not touch the historical shared configs. Copy the
exact `RELEASE_DIR` printed by that script into the next command. In the
Python environment:

```bash
RELEASE_DIR="/exact/approved_outcome_release_v2_path_printed_by_build_script"
export BONSAI_OUTCOMES_DIR="$RELEASE_DIR/outcomes"
python -m opera.run.materialize_primary_display_groups_v2 \
  --release-dir "$RELEASE_DIR"
python -m opera.run.verify_approved_outcome_training_v2 \
  --release-dir "$RELEASE_DIR" \
  --config "$RELEASE_DIR/generated_configs/direct_cr_bce_lowrank_endpoint_groups_v1.yaml"
python -m opera.run.preflight_direct_competing_risk \
  --config-dir "$RELEASE_DIR/generated_configs" \
  --config-name direct_cr_bce_lowrank_endpoint_groups_v1 \
  --require-existing-paths
```

The display taxonomy is figure-only and covers each approved outcome once;
it does not alter model weighting or checkpoint selection. The release
verifier checks exact registry/config order, clinical families, endpoint-group
coverage, parquet hashes, structural-audit results, and the active outcome
directory. The
model preflight checks outcome/feature/checkpoint paths, but it does not prove
that a checkpoint trained on the old 87-outcome roster can load into the new
head shape. Start the first approved-panel run from the existing DAPT encoder,
not an old outcome-head checkpoint, and use a fresh output directory. Verify
the exact `RELEASE_DIR`, generated outcome names, and preflight output; launch
overnight training only after clinical sign-off.

Once those checks and sign-off pass, use the release-local config directly in
the Python/GPU terminal, with a fresh versioned Hydra run directory:

```bash
python -m opera.run.contrastive_multicohort \
  --config-path "$RELEASE_DIR/generated_configs" \
  --config-name direct_cr_bce_lowrank_endpoint_groups_v1 \
  "hydra.run.dir=$PWD/data/models/approved_outcome_v2_$(date -u +%Y%m%dT%H%M%SZ)"
```

Do not resume from a previous outcome-head checkpoint or use the historical
v4 overnight-pilot launcher.

For the matched Fourier versus scaled-Time2Vec comparison, run one arm on each
of the two **separate one-GPU servers**. Each server's sole GPU is local device
0; both commands therefore use `CUDA_VISIBLE_DEVICES=0`. GPU 1 is never used
for this pair. Use `scripts/run_approved_abspos_pair_v2.sh fourier` on one
server and `scripts/run_approved_abspos_pair_v2.sh time2vec` on the other.
Both pretraining checkpoints live under the additional
`final_b4_20261003/version_0/best.ckpt` nesting shown on the server; the
wrapper uses that path by default (override `OPERA_PRETRAIN_RUN_NAME` only if
the actual nested run name differs). Their resolved run configurations are at
`final_b4_20261003/hydra/config.yaml` (without a leading dot). The wrapper
sets local GPU 0 and
explicitly overrides both the pretraining checkpoint and
the `with_sor_final_v1` sequence-data directory, rather than inheriting the
generated YAML's older `harmonized_v4` defaults. It checks the two completed
pretraining sidecars and resolved configs, the release, and each overridden
fine-tuning config before training. `OPERA_LAUNCH=1` is required to start.

If the data module logs `filter_subject_data: ... -> 0 subjects` with the R
metadata, stop both launches. The R export wrote text patient IDs while the
physical subject tensors use numeric IDs. Do not relabel outcomes or disable
membership filtering. On the secure server, build a new metadata artifact that
verifies conversion against both physical subject partitions and released
survival IDs:

```bash
OLD_META="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z"
NEW_META="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z_idtype_v1"
SEQUENCE="$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1"
RELEASE="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
python -m opera.run.repair_sealed_membership_id_v2 \
  --metadata-root "$OLD_META" \
  --survival "$RELEASE/outcomes/overall_survival.parquet" \
  --sequence-data "$SEQUENCE" \
  --output-root "$NEW_META"
```

Inspect `raw_overlap`, `converted_overlap`, and `target_type`. Only use the
new file if the repair passes. Re-run the `.env` activation helper with
`--metadata "$NEW_META/population_metadata.parquet"`; the paired launcher
defaults to this repaired metadata path and refuses metadata lacking the
verified ID-type repair. Use fresh run directories for both GPU launches.

For a controlled overnight queue, update the server's `.env` with the exact
versioned `BONSAI_COHORT_MEMBERSHIP` and `BONSAI_OUTCOMES_DIR` values, then
export those same values in the GPU shell (an exported value takes precedence
over `.env`). Keep `BONSAI_PROCESSED_DATA` and `BONSAI_MODELS` unchanged unless
their actual paths have changed. Set `OPERA_RELEASE_DIR` to the exact release.
After inspecting the R metadata manifest, the following helper validates both
artifacts, then backs up and updates only those two `.env` keys:

```bash
export OPERA_RELEASE_DIR="$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
export METADATA_ROOT="$PWD/data/results/population_metadata_sealed_v2_20261005T202202Z"
python -m opera.run.activate_approved_outcome_env_v2 \
  --env-file "$PWD/.env" --release-dir "$OPERA_RELEASE_DIR" \
  --metadata "$METADATA_ROOT/population_metadata.parquet"
python -m opera.run.activate_approved_outcome_env_v2 \
  --env-file "$PWD/.env" --release-dir "$OPERA_RELEASE_DIR" \
  --metadata "$METADATA_ROOT/population_metadata.parquet" --apply
```

The first invocation is a dry-run. The second leaves a timestamped backup of
the previous `.env`. In the already-open GPU shell, explicitly export the two
new values; editing `.env` does not replace values already exported there.
The queue preflights by default and does not train until `OPERA_LAUNCH=1`:

```bash
export OPERA_RELEASE_DIR="$RELEASE_DIR"
export BONSAI_COHORT_MEMBERSHIP="$METADATA_ROOT/population_metadata.parquet"
export BONSAI_OUTCOMES_DIR="$RELEASE_DIR/outcomes"
export OPERA_VARIANTS="fourier_factorized time2vec_factorized"
export OPERA_TIME2VEC_CKPT="/exact/completed/scaled_time2vec/best.ckpt"
bash scripts/run_approved_outcome_v2_queue.sh
export OPERA_LAUNCH=1
bash scripts/run_approved_outcome_v2_queue.sh
```

The first comparison is Fourier versus *scaled* Time2Vec with the same
factorized readout. Use only a completed Time2Vec checkpoint with the same
vocabulary and training data; the path above is deliberately a placeholder.
Do not add sharing or family-weighting variants to this initial comparison.
After choosing an encoder on tuning data, `fourier_shared`,
`fourier_family_trunks`, and `fourier_outcome_attention` are possible readout
branches (replace the `fourier_` prefix in the queue only after implementing
and validating a Time2Vec-winner branch). Family trunks use the release
registry's clinical-family map; no manual outcome list is needed. The current
families were inherited from the old registry with renamed/new outcomes
mapped in, not clinically regrouped. Endpoint groups currently combine named
grade variants but do not collapse related diagnoses/proxies/composites. A
family-balanced objective/selection policy therefore deserves a separate,
prespecified experiment after the encoder comparison, not an implicit change
between its two arms.

After both arms finish, compare *tuning* metrics from the exact two run
directories; the comparator requires completion markers, best checkpoints and
identical endpoint-selection inventories:

```bash
RUN_ROOT="$PWD/data/models/approved_outcome_abspos_v2_idtype_v1"
python -m opera.run.compare_approved_abspos_v2 \
  --fourier-run "$RUN_ROOT/fourier" \
  --time2vec-run "$RUN_ROOT/time2vec"
python -m opera.run.audit_approved_family_weighting_v2 \
  --release-dir "$PWD/data/results/approved_outcome_release_v2_20261005T200002Z"
```

The first report compares the selected endpoint-group AUC and per-epoch curves
without reading held-out patients. The second lists inherited family members,
their present endpoint-group loss share, and a clearly labeled *unapproved*
equal-family counterfactual. It is an audit, not a changed training config.
