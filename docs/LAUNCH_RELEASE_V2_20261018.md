# Launch on the new MEDS release (v2), 18 October 2026

Everything trained on `daly_care_t100_joined_bins_with_sor_final_v1` stays as
it is. The v2 release gets its own processed data, encoders and paper root, so
nothing resumes or reads v1 by accident. Run from the repository root with the
server environment active. Scripts are run with `bash`.

```bash
REL=daly_care_t100_joined_bins_with_sor_final_v2
REPO=$(pwd)
NEW_MEDS=<the MEDS output directory of the new ehr2meds run>   # holds data/train, data/tuning
set -a; source .env; set +a
export EHR2MEDS_OUTPUT=$NEW_MEDS      # .env may still hold the old release until step 0 is applied
```

## 0. Update `.env` (once)

Preview, then apply. Only these keys change; everything else in `.env`
(release, outcomes, PHAIR files, feature export, configs) stays.

```bash
SETTINGS=(
  --set "EHR2MEDS_OUTPUT=$NEW_MEDS"
  --set "OPERA_SEQUENCE_DATA_DIR=$BONSAI_PROCESSED_DATA/$REL"
  --set "OPERA_PAPER_LOCAL_VOCAB=$BONSAI_PROCESSED_DATA/$REL/vocabulary.pt"
  --set "OPERA_PAPER_SUBJECT_DATA_DIR=$BONSAI_PROCESSED_DATA/$REL"
  --set "OPERA_PAPER_MEDS_DIR=$NEW_MEDS"
  --set "OPERA_PAPER_ROOT=$REPO/data/paper_256_v2"
  --set "OPERA_PAPER_LOCAL_CKPT=$REPO/data/models/$REL/pretrain_scaled_time2vec_hidden256/version_0/best.ckpt"
  --set "OPERA_PAPER_LOCAL_WIDTH=256"
  --set "OPERA_PAPER_EXTERNAL_DATA_CONFIG=daly_care_joined_bins_data"
  --set "OPERA_PAPER_PATIENT_METADATA=$REPO/data/paper_256_v2/metadata/patient_metadata.csv"
  --set "OPERA_PAPER_CELL_METADATA=$REPO/data/paper_256_v2/metadata/cell_metadata.csv"
  --set "OPERA_PAPER_ANALYSIS_ROOT=$REPO/data/paper_256_v2/analysis"
)
python -m opera.run.configure_paper_env "${SETTINGS[@]}"            # preview
python -m opera.run.configure_paper_env "${SETTINGS[@]}" --apply
set -a; source .env; set +a
echo "$EHR2MEDS_OUTPUT"; ls "$EHR2MEDS_OUTPUT/data"     # the new release: train  tuning
```

`OPERA_PAPER_LOCAL_CKPT` can be set now: `prepare` blocks until that run has
finished (completion sidecar, at least 20 epochs).

## 1. Check the release and its vocabulary (CPU, minutes)

```bash
python -m opera.run.check_meds_release --meds-dir "$EHR2MEDS_OUTPUT" \
  --membership "$BONSAI_COHORT_MEMBERSHIP" --outcomes-dir "$BONSAI_OUTCOMES_DIR" \
  --old-vocab "$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1/vocabulary.pt" \
  --output "$EHR2MEDS_OUTPUT/meds_release_check.json"
python -m opera.run.audit_vocabulary --vocabulary "$OPERA_PAPER_LOCAL_VOCAB" \
  --config opera/configs/attribution_analysis.example.yaml --output-dir "$OPERA_PAPER_ROOT/vocabulary_audit"
bash scripts/run_paper_experiments.sh check
```

Stop on any FAIL. `check` may still report the local encoder as missing; that
is expected until step 2 finishes.

## 2. Pretrain both widths (two GPUs, in parallel)

One GPU allocation each. The launcher refuses a directory that already holds
a run, and reads only `$REL`.

```bash
export BONSAI_DATASET=$REL BONSAI_256_HEADS=4 BONSAI_256_LAYERS=4
bash scripts/run_pretrain_hidden256.sh preflight          # resolved config: 256, 4 heads, 4 layers, $REL
BONSAI_WIDTH=512 bash scripts/run_pretrain_hidden256.sh preflight   # 512, 8 heads

# GPU 1: the paper encoder, 20 epochs (keeps epoch_010.ckpt for the probe)
BONSAI_256_ARCHITECTURE_CONFIRMED=1 bash scripts/run_pretrain_hidden256.sh launch
# GPU 2: the width probe, 10 epochs
BONSAI_WIDTH=512 BONSAI_EPOCHS=10 BONSAI_256_ARCHITECTURE_CONFIRMED=1 \
  bash scripts/run_pretrain_hidden256.sh launch
```

After an interruption: the same command with `resume` instead of `launch`
(plus `BONSAI_WIDTH=512` for the probe).

Runs: `data/models/$REL/pretrain_scaled_time2vec_hidden256/` and
`.../pretrain_scaled_time2vec_hidden512/`.

## 3. CPU work while the encoders train

```bash
bash scripts/run_paper_experiments.sh table1 --inspect-inputs --execute
bash scripts/run_paper_experiments.sh table1 --execute
bash scripts/run_paper_experiments.sh features --execute           # sequence-matched + registry matrices
bash scripts/run_paper_experiments.sh cell-metadata --execute
bash scripts/run_paper_experiments.sh patient-metadata --execute
# National arm, step 1: our patients from the national MEDS, PHAIR vocabulary, joined bins
bash scripts/run_paper_experiments.sh external-data --execute      # audit: coverage and [UNK] share
```

`tabular` needs the stage-1 bundle, so it starts after step 4.

## 4. When the 256 run has finished (20 epochs)

```bash
cat data/models/$REL/pretrain_scaled_time2vec_hidden256/version_0/checkpoint_metadata.json
bash scripts/check_paper_server.sh --gpu
bash scripts/run_paper_experiments.sh prepare --execute            # skip if already built for the width probe (step 5)
bash scripts/run_paper_experiments.sh smoke --execute
bash scripts/run_paper_experiments.sh queue-status
```

Then one worker per free GPU, and the tabular baselines on CPU:

```bash
WAIT=1 bash scripts/paper_gpu_worker.sh
bash scripts/run_paper_experiments.sh tabular --shard-count 4 --shard-index 0 --execute   # 0..3
bash scripts/run_paper_experiments.sh tabular --registry --shard-count 4 --shard-index 0 --execute
```

## 5. Width decision (as soon as the 512 run has its 10 epochs)

The 256 run is still training then, so the probe uses its named epoch-10
milestone (accepted by the width probe only; its epoch is checked against its
name). `prepare` only records the encoder path, so the stage-1 bundle can be
built before either run finishes.

**Now (before the 512 run ends):**

```bash
bash scripts/run_paper_experiments.sh prepare --execute     # stage-1 bundle (records the 256 best.ckpt path)
M=data/models/$REL
ls $M/pretrain_scaled_time2vec_hidden256/version_0/epoch_010.ckpt   # the 256 milestone exists
```

Shortcut: `bash scripts/run_width_probe.sh 256|512 [preview]` and
`bash scripts/run_width_probe.sh compare` fill in every path below from `.env`.

**When the 512 run has finished** (`$M/pretrain_scaled_time2vec_hidden512/version_0/checkpoint_metadata.json`
exists), one GPU each, same protocol for both:

```bash
python -m opera.run.width_probe --bundle-dir "$OPERA_PAPER_ROOT/bundles/stage1" \
  --encoder-checkpoint $M/pretrain_scaled_time2vec_hidden256/version_0/epoch_010.ckpt \
  --width 256 --label w256_e10 --output-dir "$OPERA_PAPER_ROOT/width_probe" --execute
python -m opera.run.width_probe --bundle-dir "$OPERA_PAPER_ROOT/bundles/stage1" \
  --encoder-checkpoint $M/pretrain_scaled_time2vec_hidden512/version_0/last.ckpt \
  --width 512 --label w512_e10 --output-dir "$OPERA_PAPER_ROOT/width_probe" --execute
```

`last.ckpt` of the 512 run is its tenth epoch, matching the 256 milestone.
Fits stop early after 5 epochs without improvement; to bound the time, add the
same `--epochs N` to both. Preview first without `--execute` (it runs the
encoder audit only).

**When both have finished:**

```bash
python -m opera.run.width_probe --compare --output-dir "$OPERA_PAPER_ROOT/width_probe"
```

prints and writes `width_decision.json`: each probe's tuning score (the score
it selected its checkpoint on), the gain of 512 over 256 and the call under the
rule fixed in advance: **keep 256 unless 512 gains at least 0.01.**

- Keep 256: nothing changes; when the 256 run reaches 20 epochs, `smoke` and
  the workers start (step 4).
- Switch to 512: `BONSAI_WIDTH=512 bash scripts/run_pretrain_hidden256.sh resume`
  (continues to 20 epochs), a new paper root with `OPERA_PAPER_LOCAL_WIDTH=512`
  and `OPERA_PAPER_LOCAL_CKPT` at the 512 `best.ckpt`, then `prepare` again.
  The national arms lose their width match with the 256-wide PHAIR encoder.

Report the width comparison either way.

## 6. National encoder (one GPU, once PHAIR files are in `.env`)

```bash
bash scripts/run_paper_experiments.sh export-external --execute    # coverage gate >= 0.9
bash scripts/run_paper_experiments.sh dapt --execute               # 20 epochs on the cohort
```

Then router screens and stage 2 as in `RUNBOOK_20261015.md` §4 onwards:
`tuning-summary` per screen, decisions JSON, router lock,
`prepare --stage 2 --include-external --execute`, workers continue.

## 7. Explainability (prepare now, run on the locked model)

**Now, on CPU (no model needed):**

1. Copy the official code lists into the secure environment (ICD-10, ATC, NPU,
   SKS; sources in `opera/resources/code_systems/README.md`) and list them
   under `code_tables:` in a server copy of
   `opera/configs/attribution_analysis.example.yaml`; point
   `OPERA_PAPER_ATTRIBUTION_CONFIG` at that copy.
2. Rerun the vocabulary audit with that config and read
   `vocabulary_audit/concept_coverage.csv`: every finding and laboratory test
   must say `OK`, with the expected names and a bin range (`NO MATCH` means a
   wrong code). Correct the codes, check the 21 expected directions, then set
   `concept_mapping_reviewed: true`. Without it the summary drops the named
   findings and laboratory curves (Figure 8C and S30C would be empty).

```bash
python -m opera.run.audit_vocabulary --vocabulary "$OPERA_PAPER_LOCAL_VOCAB" \
  --config "$OPERA_PAPER_ATTRIBUTION_CONFIG" --output-dir "$OPERA_PAPER_ROOT/vocabulary_audit"
```

**Molecular findings in pathology (TP53, MYC, IGHV).** Find their codes with
the vocabulary search, in both vocabularies: the national one keeps rare codes
that the cohort's conversion merged into their parents (same 100-event
threshold, far more data). Add the PATOBANK code table under `code_tables:` so
the search also matches names. Put the codes of the vocabulary of the model
that is finally locked into the three `group: Pathology` concepts.

```bash
python -m opera.run.audit_vocabulary --vocabulary "$OPERA_PAPER_LOCAL_VOCAB" \
  --config "$OPERA_PAPER_ATTRIBUTION_CONFIG" --search 'TP53|MYC|IGHV|17p|BCL2' \
  --output-dir "$OPERA_PAPER_ROOT/vocabulary_audit"
python -m opera.run.audit_vocabulary --vocabulary "$OPERA_PAPER_EXTERNAL_SOURCE_VOCAB" \
  --config "$OPERA_PAPER_ATTRIBUTION_CONFIG" --search 'TP53|MYC|IGHV|17p|BCL2' \
  --output-dir "$OPERA_PAPER_ROOT/vocabulary_audit_national"
```

**After the held-out evaluation** (model chosen on tuning): set
`OPERA_PAPER_ATTRIBUTION_CHECKPOINT`, `OPERA_PAPER_ATTRIBUTION_CONFIG_NAME`
(and `_CONFIG_DIR`) to the locked best model, then on one GPU:

```bash
bash scripts/run_paper_experiments.sh attributions --attribution-max-new-patients 50 --execute  # pilot: cost in attribution_run.json
bash scripts/run_paper_experiments.sh attributions --execute          # resumes to every test patient
bash scripts/run_paper_experiments.sh attribution-summary --execute   # the only exportable outputs
bash scripts/run_paper_experiments.sh input-restrictions --execute    # 6 evaluations of the locked model
bash scripts/run_paper_experiments.sh attribution-figures --execute   # Figure 8, S30, S31
bash scripts/run_paper_experiments.sh figure-readiness --execute      # every manuscript figure: ready or what is missing
```

Figure S32 and panel 8D use the synthetic patient by design; real patients'
explanations stay internal (`render_individual_attribution`).
