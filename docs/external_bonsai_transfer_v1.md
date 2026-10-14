# External BONSAI encoder → local domain adaptation → OPERA

An external pretrained checkpoint must arrive with its **original**
`vocabulary.pt` and, ideally, its resolved training config and
`checkpoint_metadata.json`. Copy these files into a new, immutable input
directory on the secure server. Do not replace the local MEDS vocabulary.

The transfer audit requires a native BONSAI Lightning pretraining checkpoint
containing `state_dict` and `hyper_parameters.model_config`. It checks encoder
tensor names and shapes, causal/time/value architecture, exact token-string
overlap, checkpoint and vocabulary hashes, and a CPU forward pass. Only code
embedding rows are remapped: shared tokens copy by name, while local-only
tokens retain fresh initialization. Source pretraining heads and optimizer
state are **not** transferred. The local domain-adaptation run starts a new
optimizer and writes a normal BONSAI pretraining checkpoint with the local
vocabulary; OPERA then strict-loads its encoder as usual.

On the secure server, after syncing the new code and receiving the source
bundle, replace the two `SOURCE_*` paths below with the actual files:

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source setup_v100.sh

SOURCE_CKPT=/absolute/path/to/external_bonsai/best.ckpt
SOURCE_VOCAB=/absolute/path/to/external_bonsai/vocabulary.pt
LOCAL_SEQUENCE="$BONSAI_PROCESSED_DATA/daly_care_t100_joined_bins_with_sor_final_v1"
AUDIT="$PWD/data/results/external_bonsai_transfer_audit_v1.json"

python -m bonsai.run.audit_pretrain_transfer \
  --checkpoint "$SOURCE_CKPT" \
  --source-vocab "$SOURCE_VOCAB" \
  --target-vocab "$LOCAL_SEQUENCE/vocabulary.pt" \
  --min-token-coverage 0.9 \
  --output-json "$AUDIT"
```

If this fails, inspect the specific incompatibility before requesting GPUs.
Low token coverage, a different native architecture, absent source vocabulary,
or a legacy ModernBERT checkpoint is **not** silently converted. The 90%
non-special-token coverage gate is deliberately conservative and can be
revisited after reviewing the missing tokens and their frequencies.

Once the audit passes, choose a fresh output directory and launch local
domain-adaptive pretraining:

```bash
DAPT_RUN="$PWD/data/models/external_bonsai_local_dapt_v1"
test -s "$AUDIT" || { echo "Transfer audit missing" >&2; exit 1; }
test ! -e "$DAPT_RUN" || { echo "DAPT run directory already exists" >&2; exit 1; }

python -u -m bonsai.run.pretrain \
  --config-name pretrain_external_bonsai_dapt_v1 \
  "paths.initialization_ckpt=$SOURCE_CKPT" \
  "paths.initialization_vocab=$SOURCE_VOCAB" \
  "hydra.run.dir=$DAPT_RUN"
```

The config imports the source encoder architecture explicitly, retaining the
local SDPA execution backend. It reuses the local pretraining data and
vocabulary paths. Review sequence length, pretraining objective, training
budget and GPU memory before launching; a larger external model may require
different batch sizing. The source checkpoint and local data may also differ
in code systems or time representation despite matching tensor shapes, so the
audit report and source metadata need scientific review.

After DAPT, audit the actual saved checkpoint through OPERA's strict encoder
load, local-vocabulary hash, and CPU forward pass:

```bash
python -m bonsai.run.audit_dapt_to_opera \
  --checkpoint "$DAPT_RUN/version_0/best.ckpt" \
  --vocabulary "$LOCAL_SEQUENCE/vocabulary.pt"
```

Then use the approved candidate launcher with a separate run root. The
`OPERA_EXTERNAL_DAPT_CKPT` override invokes the same audit automatically and
does not overwrite the existing local-pretraining comparison:

```bash
export OPERA_EXTERNAL_DAPT_CKPT="$DAPT_RUN/version_0/best.ckpt"
export OPERA_NEURAL_RUN_ROOT="$PWD/data/models/approved_external_bonsai_dapt_v1"
OPERA_LAUNCH=0 bash scripts/run_approved_neural_candidate_v1.sh time2vec_router_interacting_scale03_e2e
OPERA_LAUNCH=1 bash scripts/run_approved_neural_candidate_v1.sh time2vec_router_interacting_scale03_e2e
```

The `time2vec_` candidate name is only scientifically accurate if the source
checkpoint uses scaled Time2Vec. If it uses Fourier timing, select the matched
Fourier candidate instead; otherwise create a clearly named candidate before
reporting results. Keep supervised split, outcomes, readout, seed, budget and
checkpoint metric matched to the local-pretraining arm. The full end-to-end
transfer remains conditional on receiving the actual source checkpoint and
vocabulary and passing these audits on the secure server.

The paper's direct external-pretraining arm (without local DAPT) needs an
additional local-vocabulary checkpoint export. It should not be simulated by
starting a zero-epoch DAPT run; implement and audit that arm separately before
the matched pretraining ladder is declared complete.
