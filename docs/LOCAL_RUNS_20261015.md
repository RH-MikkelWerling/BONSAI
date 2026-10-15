# Local runs — exact commands, in order (15 October 2026)

Everything below runs on the server, in the repository root, with the server
Python environment activated. Commands without `--execute` only print what
they would do; use that first whenever unsure. `$R` below means
`$OPERA_PAPER_ROOT`.

## 0. Sync and settings (once)

```bash
# On the laptop: code only; never data, .env, checkpoints or run roots.
rsync -av --exclude '.git' --exclude 'data/' --exclude '.env' \
  --exclude 'paper/render/' --exclude '__pycache__' \
  ./ <server>:<repo>/
```

On the server, write a settings file (e.g. `~/paper.settings`) with every
line below filled in, then apply it to `.env` (other `.env` entries are kept):

```bash
OPERA_RELEASE_DIR=<approved release dir>
OPERA_SEQUENCE_DATA_DIR=<tokenised sequence dir: subject_data_*.pt, vocabulary.pt>
BONSAI_COHORT_MEMBERSHIP=<membership file of the release>
BONSAI_OUTCOMES_DIR=<release dir>/outcomes
OPERA_PAPER_ROOT=<fresh paper root, e.g. data/paper_256>
OPERA_PAPER_BASE_CONFIG=<approved 69-outcome training config>
OPERA_PAPER_ROUTER_REFERENCE=<router reference config; must record readout_mode and adapter fields>
OPERA_PAPER_LOCAL_CKPT=<local 256 run>/version_0/best.ckpt
OPERA_PAPER_LOCAL_VOCAB=<OPERA_SEQUENCE_DATA_DIR>/vocabulary.pt
OPERA_PAPER_INDEX_DATE_COLUMN=<membership column with first-line start; default index_date>
OPERA_PAPER_SEEDS=42
OPERA_PAPER_EPOCHS=30
OPERA_PAPER_BATCH_SIZE=1
OPERA_PAPER_MIN_LOCAL_EPOCHS=20
OPERA_PAPER_MIN_DAPT_EPOCHS=20
OPERA_PAPER_DECISIONS=<R>/decisions.json          # written in step 5
OPERA_PAPER_ROUTER_LOCK=<R>/router_lock.json      # written in step 5
OPERA_PAPER_FEATURE_SOURCE=<DALY-CARE feature export (timeseriesflattener, first-line index)>
OPERA_PAPER_FEATURE_MATRIX=<R>/features/<feature source stem>__sequence_matched.parquet
OPERA_PAPER_TABULAR_HORIZONS=30,60,90,180,365,730
OPERA_PAPER_CELL_METADATA=<R>/metadata/cell_metadata.csv
OPERA_PAPER_PATIENT_METADATA=<R>/metadata/patient_metadata.parquet
OPERA_PAPER_CLINICAL_METADATA=<optional: table with subject_id, index, IPI>
OPERA_PAPER_ANALYSIS_ESTIMAND=cif_ipcw
OPERA_PAPER_ANALYSIS_HORIZONS=30,60,90,180,365,730
OPERA_PAPER_DATA_CONTRACT=<release contract ID>
```

```bash
python -m opera.run.configure_paper_env --settings-file ~/paper.settings          # preview
python -m opera.run.configure_paper_env --settings-file ~/paper.settings --apply
```

Check the membership columns the metadata builder needs (subject id, index
date, age or birth date, sex, `cohort_fine`, and `index`/`IPI` unless given
separately):

```bash
python -c "import os,pandas as pd; from dotenv import load_dotenv; load_dotenv(); \
p=os.environ['BONSAI_COHORT_MEMBERSHIP']; \
f=pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p, nrows=5); print(list(f.columns))"
```

### Path map: what each setting must point to

| Setting | Must point to | Checked by | Read by |
| --- | --- | --- | --- |
| `OPERA_RELEASE_DIR` | release folder containing `experiment_registry_primary_v2.yaml` (69 outcomes) and `outcomes/` | `check` | `check`, `prepare`, `cell-metadata` |
| `BONSAI_OUTCOMES_DIR` | exactly `<OPERA_RELEASE_DIR>/outcomes` | `check` refuses anything else | training configs (`${oc.env:...}`) |
| `BONSAI_COHORT_MEMBERSHIP` | the sealed membership file; must equal every cohort's `population_file` in the base config | `check` | training, tabular, metadata builders |
| `OPERA_SEQUENCE_DATA_DIR` | folder with `vocabulary.pt`, `subject_data_train.pt`, `subject_data_tuning.pt` (held-out patients live in these shards) | `check` | training, export, DAPT, patient metadata |
| `OPERA_PAPER_BASE_CONFIG` | approved shared linear horizon-BCE config; its `paths.vocabulary` must be `<OPERA_SEQUENCE_DATA_DIR>/vocabulary.pt`; all 10 disease groups | `check`, `prepare` | `prepare` |
| `OPERA_PAPER_ROUTER_REFERENCE` | router reference config recording `readout_mode` and adapter fields | `prepare` | `prepare` |
| `OPERA_PAPER_LOCAL_CKPT` | `<local 256 run>/version_0/best.ckpt`, next to `checkpoint_metadata.json`; vocabulary size must match the sequence vocabulary | encoder audit (256 wide, 4 layers, 4 heads, scaled Time2Vec, ≥ 20 epochs, strict load) | `prepare` (frozen into the bundle), workers, smoke |
| `OPERA_PAPER_LOCAL_VOCAB` | the same `vocabulary.pt` as above | — | `patient-metadata` |
| `OPERA_PAPER_FEATURE_SOURCE` | the raw DALY-CARE feature export | `features` | `features` |
| `OPERA_PAPER_FEATURE_MATRIX` | the file `features` writes: `<R>/features/<source stem>__sequence_matched.parquet` | `tabular` | `tabular` |
| `OPERA_PAPER_CELL_METADATA`, `OPERA_PAPER_PATIENT_METADATA` | output paths (written by the builders, then read by `inventory`) | — | builders, `inventory`, `plot-tables` |

If the local encoder was trained on a different vocabulary than
`OPERA_SEQUENCE_DATA_DIR/vocabulary.pt` (for example after re-tokenising with
the collaborators' aggregated values), the audit stops at "vocabulary sizes
differ": point the sequence directory and base config at the matching data.

## 1. Confirm pretraining has finished

```bash
RUN=<local 256 run dir>/version_0
ls $RUN                                   # best.ckpt, last.ckpt, epoch_010.ckpt, checkpoint_metadata.json
cat $RUN/checkpoint_metadata.json         # completed_epochs (21 in this run = 20 epochs; known off-by-one)
python -c "import torch; c=torch.load('$RUN/best.ckpt',map_location='cpu',weights_only=False); \
print('best epoch', c['epoch']+1)"
```

Record the best epoch and the validation loss at epoch 10 and at the best epoch
(from the run's `metrics.csv`) for the Supplementary Methods.

## 2. Readiness (no long compute; one GPU allocation)

```bash
bash scripts/check_paper_server.sh --gpu        # imports, one visible GPU, then `check`
python -m pytest -q tests/test_paper_workflow.py tests/test_paper_worker.py \
  tests/test_paper_run_bundle.py tests/test_paper_controls.py tests/test_pretrain_transfer.py
bash scripts/run_paper_experiments.sh prepare --execute     # writes $R/bundles/stage1 (immutable)
bash scripts/run_paper_experiments.sh smoke --execute       # 1 epoch, 2 batches, tuning pass, reload
bash scripts/run_paper_experiments.sh queue-status
```

**Go** when `check` reports no problems, the smoke result exists, and
`queue-status` shows supervision runs as ready (not "encoder" blocked).
If `prepare` refuses because `$R/bundles/stage1` exists from an earlier
attempt, use a fresh `OPERA_PAPER_ROOT` (bundles are immutable by design).

## 3. GPU workers (as many GPUs as you have)

In each GPU allocation (one GPU each):

```bash
WAIT=1 bash scripts/paper_gpu_worker.sh
```

Workers share one to-do list and never duplicate a fit. They take runs in this
order, and wait (with `WAIT=1`) when the next runs are blocked:

| Order | Runs | Count | Blocked until |
| --- | --- | --- | --- |
| 1 | Joint model | 1 | — |
| 2 | Disease-pooled models (one per outcome) | 69 | — |
| 3 | Outcome-pooled models (one per disease) | 24 | — |
| 4 | No-pretraining control (joint, random init) | 1 | — |
| 5 | Router source screen (none, disease, outcome, both; rank 8) | 4 | — |
| 6 | Router rank screen (8 vs 16) | 2 | `source_mode` in decisions |
| 7 | Router query screen | 3 | `source_mode` + `rank` in decisions |
| 8 | Final source banks with the locked router | 4 | stage-2 bundle |
| 9 | National/adapted joint and routed | 4 | stage-2 bundle with external arms |
| 10 | Specialists | 1,644 | — |

Progress from any terminal: `bash scripts/run_paper_experiments.sh queue-status`.

## 4. CPU jobs (start in parallel with step 3)

```bash
bash scripts/run_paper_experiments.sh features --execute          # writes the matched feature matrix
bash scripts/run_paper_experiments.sh cell-metadata --execute     # train/tuning labels only
bash scripts/run_paper_experiments.sh patient-metadata --execute  # history counts, IPI, age, sex
# Tabular: per-cell XGBoost + logistic, and pooled XGBoost per outcome. Shard over N CPU jobs:
bash scripts/run_paper_experiments.sh tabular --shard-count N --shard-index i --execute
```

The tabular step is the largest CPU job (1,644 cells × 6 horizons × 2 models,
plus 69 × 6 pooled fits, with the deeper XGBoost settings); shard it widely.

## 5. Router decisions (tuning data only)

When the four source-screen fits are complete:

```bash
bash scripts/run_paper_experiments.sh tuning-summary --phase source_screen
# read $R/tuning_summary_stage1_source_screen/tuning_scores.csv (lower score is better)
echo '{"source_mode": "<none|disease_only|outcome_only|both>"}' > $OPERA_PAPER_DECISIONS
```

Workers pick up the two rank-screen fits automatically. When complete:

```bash
bash scripts/run_paper_experiments.sh tuning-summary --phase rank_screen
echo '{"source_mode": "<...>", "rank": <8|16>}' > $OPERA_PAPER_DECISIONS
```

When the three query-screen fits are complete:

```bash
bash scripts/run_paper_experiments.sh tuning-summary --phase query_screen
cat > $OPERA_PAPER_ROUTER_LOCK <<'JSON'
{"source_mode": "<...>", "rank": <8|16>, "query_mode": "<additive|interacting>", "query_width": <128|512>}
JSON
```

Prefer the simpler option when scores are within the prespecified margin.
If `none` wins the source screen, routing adds nothing; record that and lock
`none` with rank 8 and the default query (`interacting`, 128).

## 6. Stage 2 (final source banks and the national encoder)

Prepare stage 2 **once**, with the external arms included, as soon as both the
router lock and the national encoder (export + 20-epoch adaptation) exist —
the bundle cannot be extended later. Specialists keep the workers busy in the
meantime, so waiting costs nothing.

```bash
bash scripts/run_paper_experiments.sh export-external --execute   # coverage gate ≥ 0.9
bash scripts/run_paper_experiments.sh dapt --execute              # one GPU, 20 epochs
bash scripts/run_paper_experiments.sh prepare --stage 2 --include-external --execute
```

## 7. Predictions (as fits complete)

```bash
bash scripts/run_paper_experiments.sh evaluate --stage 1 --split tuning --execute
bash scripts/run_paper_experiments.sh evaluate --stage 2 --split tuning --execute
```

Held-out predictions come only after the analysis plan is locked
(`RUNBOOK_20261015.md`, section 5):

```bash
bash scripts/run_paper_experiments.sh evaluate --stage 1 --split held_out --execute
bash scripts/run_paper_experiments.sh evaluate --stage 2 --split held_out --execute
```

Paper models export patient embeddings automatically; the locked routers also
export routing weights.
