# Collaborator pretraining compatibility — 28 October 2026

Inspected original checkout `/Users/mikkelwerling/Documents/phd_repos/ORIGINAL_BONSAI/BONSAI`, commit `d5eb95c`, and this OPERA/BONSAI fork. The screenshot is a data-creation recipe, not a resolved training recipe. Conclusions about her actual run remain conditional on its training configuration and code revision. The original checkout was not modified and no message was sent to collaborators.

## Verdict

The original code-only causal BONSAI encoder can be imported. However, before a costly training launch, resolve the scalar payload, joined-bin tokenization, SEP target policy, and calendar/vocabulary cutoff. There was a genuine legacy checkpoint import bug in our fork: original checkpoints omit num_attention_heads, since their implementation derives it as hidden_size/64. This is now supported for foreign legacy checkpoints only, without overriding explicit counts or weakening local-native requirements.

A synthetic encoder was created using the actual original-repo classes (width 256, two layers, four inferred heads, SDPA, code-only Time2Vec). Its encoder weights loaded strictly into our implementation. With identical effective float32 time inputs, encoder outputs matched exactly. Converting the same hours using float32 division versus multiplication introduced a maximum output difference of 4.24e-5 in this sample; it is documented in scratch/external_pretraining_audit/forward_equivalence.json. This is a code compatibility check, not a replay of the collaborator's future trained checkpoint or a GPU/FlashAttention equivalence test.

## Numerical values

Original create_data.py forwards cfg.numeric_column to process_split. Setting it null is supported: it omits the scalar column from reading, duplicate keys, output feature tables and subject dictionaries. The ordinary BonsaiPretrain/EhrEmbeddings encoder consumes code, age, absolute time and segment only; only the Value variant consumes scalar values. Therefore the screenshot does not prove duplicate numerical encoding: under the standard code-only model, the extra scalar is unused. Nonetheless it should be null to make the input contract unambiguous and avoid carrying incompatible numeric_value batches into our code-only encoder, which deliberately rejects such a field.

Crucially, the original EHRTokenizer does not bin numbers or append numerical bins to codes. Disabling numeric_column is correct only if ehr2meds already creates joined categorical tokens, e.g. LABTEST//<NPU>//BIN_<k>, and text-valued lab tokens. If the input codes remain bare laboratory test identifiers, dropping the scalar discards the result value. Confirm a few actual input token strings before training. Joined LAB//BIN_k tokens and a separate LAB, BIN_k stream are different representations.

Use in the collaborator's data recipe:

```yaml
numeric_column: null
tokenizer:
  vocabulary: null  # initial vocabulary fit; keep/freeze the resulting mapping
  cutoffs: null
  sep_tokens: true
```

Set input_dir to the actual split-containing directory: this original process_split expects input_dir/train/*.parquet, not the newer auto-resolved MEDS root/data/train layout. Keep subject IDs consistent across releases. Save the exact numeric-bin boundaries and code/text normalization recipe, not just the final vocabulary.

## SEP: input policy and target policy are independent

Original ARPretrainDataset shifts code[1:] into target and masks only token ID zero (padding). SEP (normally ID 2) remains a supervised next-token target. A direct original-code example produced input [1,5,2,6] and target [5,2,6,-100]. CE and validation precision both respect -100. Therefore SEP is in the original default AR loss.

Our production target-policy recipes inherit training.ignore_target_tokens: ['[SEP]']; our AR dataset resolves SEP through the vocabulary and masks its targets to -100 while keeping SEP inputs. Our generic base configs do not all default to excluding SEP: verify the actual resolved production config/checkpoint sidecar, rather than assuming every historical run did. The paper DAPT action uses bonsai.run.pretrain with pretrain_external_bonsai_dapt_v1, which inherits the current SEP-ignore policy. The older alternative opera.run.dapt entrypoint does not forward the new AR target-policy options; do not assume adding the YAML key there changes its objective. The paper workflow does not use that entrypoint.

If she uses MLM instead of AR, original masking_ignore_special_tokens=true excludes SEP from sampled MLM targets; that is a different objective and must be specified, not inferred from tokenizer.sep_tokens.

Recommendation before launch: match our categorical causal objective by retaining SEP in inputs and excluding SEP targets in both train and validation. This is an objective-alignment decision, not a claim that SEP prediction is inherently erroneous or that SEP-inclusive checkpoints cannot transfer. SEP embeddings still learn through predicting subsequent clinical tokens.

The original repo has no AR ignore_target_tokens plumbing. Merely adding that YAML key does nothing. A patch is provided at patches/original_bonsai_sep_target_policy.patch. It wires config -> runner -> datamodule -> AR dataset, validates ignored token names against the vocabulary, masks train and validation targets and adds the config default. Original behavior remains available with an empty list. It supports ValueAR's inherited target mask too, but no Value model is recommended for this joined-bin comparison.

Patch validation: git apply --check passes against the inspected original checkout. The patched original dataset was exercised with a nonstandard SEP ID 17: it retained SEP input, excluded SEP targets and retained original behavior when the option was omitted. Full collaborator distributed training was not run.

From the matching original checkout, after reviewing/copying the patch:

```sh
git apply --check /path/to/original_bonsai_sep_target_policy.patch
git apply /path/to/original_bonsai_sep_target_policy.patch
```

Then use ordinary ARPretrainDataset, BonsaiPretrain, CE and training.ignore_target_tokens: ['[SEP]']. Do not add num_attention_heads or abspos_encoding constructor options to this original implementation blindly: those parameters do not exist there. Record the derived head count and time convention as run metadata instead.

## Architecture and time compatibility

- Original defaults are causal, hidden size 256, four layers and max length 8192; confirm her resolved overrides. The inspected original MHA fixes head dimension 64, so width 256 gives four heads. Our importer now recovers the omitted field and the preflight reports the inference.
- Original features.compute_abspos stores hours since Unix epoch divided by 1000. Our sequences store hours. resolve_source_encoder_config maps foreign original encoders to scaled_time2vec, preserving the original effective clock. Do not initialize those weights into Fourier time encoding or feed thousands-of-hours sequences into a model that scales canonical hours again.
- This original checkout uses adjacent-pair/interleaved RoPE for SDPA and explicitly interleaved=True for FlashAttention. Earlier FlashAttention revisions may differ. Save the actual commit/backend/convention; our converter supports an explicit source RoPE layout override.
- Preserve the vocabulary token-to-ID mapping with the checkpoint. Equal vocabulary size is insufficient. Our transfer path remaps by exact token string where needed; that cannot repair different code meanings, units or bin definitions. The clean direct-transfer path tokenizes our national-format records with the original national vocabulary.
- The source must remain code-only for this comparison: no FiLM scalar pathway, no value-bin embedding branch, no value regression loss. Training on scalar values is not automatically invalid research, but it would require another supported input architecture and would change the comparison.

## Temporal and training settings to confirm before launch

For the manuscript's historical design, pretraining and fitted vocabulary/bin statistics must respect the agreed pre-2022 information boundary. Original training.cutoff_date censors sequences during training but does not retrospectively remove future information from a vocabulary built earlier on unfiltered records. Filter the upstream data before fitting bins/vocabulary as required by the protocol. A physical SSL tuning split is not the same as the downstream calendar-year tuning cohort; document both.

Original pretrain.py always adds EarlyStopping(patience=2). Thus training.epochs:10 is a maximum, not proof of ten completed passes. An early_stopping_patience:null field alone will not disable that callback in the inspected runner. Agree on the intended budget/checkpoint rule, and save actual completed epochs, token counts and the epoch-10 checkpoint if the matched comparison requires it. Optimizer/scheduler, truncation (original tail versus our possible all-windows/mixed-window runs), within-time ordering and patient split differences are provenance/comparability issues rather than necessarily weight-import failures. Record them explicitly. The original runner also does not explicitly seed training; record/set the seed through the actual launch code.

## Handoff required with the pretrained model

Deliver best.ckpt (and any agreed fixed-epoch checkpoint), exact vocabulary.pt, resolved data and training configs, commit/dependency versions, binning and namespace recipes, absolute-time and RoPE conventions, train/validation calendar/subject definitions, completed epochs/token counts and logs, and target policy including SEP. Before the full national run, a small training/save/load pilot can produce these artifacts for our compatibility check.

Our post-delivery preflight:

```sh
.venv-v100/bin/python -m opera.run.check_external_checkpoint \
  --checkpoint /path/to/best.ckpt \
  --vocabulary /path/to/vocabulary.pt \
  --data-config daly_care_joined_bins_data --width 256
```

Use this configuration only after confirming joined-bin semantics match. Then run the existing external-sequence audit and a real checkpoint forward replay before DAPT and downstream fits. Current local regression tests for target policy, transfer and external readiness: 16 passed.
