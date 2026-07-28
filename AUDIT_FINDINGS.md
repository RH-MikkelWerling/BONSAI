# Audit: why pretrained patient embeddings encode calendar era but not disease

Branch `opera/leukemia`, HEAD `3526cd4` ("making server runs and diagnostics"),
`git status` clean at audit start. Recent commits: `3526cd4`, `f269c0e`,
`97ac852`, `e801327`, `7981bfc`.

**Local-checkout limitation, noted per ground rules**: this machine has no
`.env` (`BONSAI_CONFIG_PATH`/`BONSAI_PROCESSED_DATA`/`BONSAI_MODELS` all
unset), no real DALY-CARE checkpoint, no real MEDS/subject-data shards, and no
training logs for the `daly_care_pretrain` run. Only CI/test fixtures
(`.ci/`, `example_data/`, synthetic `correlated_MEDS_data`) exist locally. All
findings below come from static reading of the code and configs actually used
for this run (`opera/configs/daly_care_pretrain.yaml`), not from executing the
pipeline or inspecting real artifacts. Items that require the real checkpoint,
real data, or the cluster's training logs are marked **undetermined (requires
cluster access)**.

---

## A1. What the 64-dim vector is, and what else is reachable

**Tensor trace, `extract_patient_embeddings` → NPZ:**

- `opera/run/extract_patient_embeddings.py:222-232` calls
  `extract_shared_split_embeddings` (defined in
  `opera/run/extract_outcome_transfer_embeddings.py:497-596`), which per
  split builds a `FinetuneDataset` (`bonsai/modules/datasets/FinetuneDataset.py`),
  collates with `dynamic_padding`, and for each batch calls
  `_pool_cls_last(encoder, device_batch)`
  (`extract_outcome_transfer_embeddings.py:482-494`).
- `_pool_cls_last`: `hidden = encoder_hidden_state(encoder(batch))` → shape
  `(batch, seq_len, hidden_size)`. `lengths = attention_mask.sum(dim=1) - 1`;
  returns `hidden[arange(batch), lengths]` → shape `(batch, hidden_size)` =
  `(batch, 64)`. This is the exact tensor written to
  `embeddings=` in the NPZ (`extract_patient_embeddings.py:240-245`).
- `encoder(batch)` → `BonsaiBase.forward` → `BonsaiBase.encode`
  (`bonsai/modules/networks/bonsai_nets.py:107-168`): embeddings `(B, S, 64)` →
  dropout → 4× `TransformerLayer` (`(B, S, 64)` throughout, residual) →
  final `self.layernorm` → returned as `(B, S, 64)`. No pooling, projection,
  or bottleneck head exists anywhere in this path.

**Is 64 the residual-stream width or a projection output?** It is the raw
residual-stream width. `hidden_size=64` is passed straight into
`EhrEmbeddings`, every `TransformerLayer`, and the final `LayerNorm`
(`bonsai_nets.py:49,66-89`); there is no wider backbone with a narrowing head.
Config: `opera/configs/daly_care_pretrain.yaml:46` (`hidden_size: 64`).

**Causal or bidirectional?** Causal.
`opera/configs/daly_care_pretrain.yaml:53` sets `causal: true`, enforced by
`bonsai/functional/model_config.py:128-136`
(`validate_pretraining_attention` raises if `ARPretrainDataset` is paired with
`causal=False`, called from `opera/run/daly_care_pretrain.py:42`). The
attention mask: `bonsai/modules/networks/components/mha.py:13-21`
(`SDPA.forward`) — `F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
is_causal=self.causal and attn_mask is None)`. For this run (`causal=True`,
non-flash `attn_type: sdpa`), `bonsai_nets.py:142-147` sets `attn_mask=None`
whenever `causal` is true, so PyTorch's native lower-triangular `is_causal=True`
mask is used — standard GPT-style causal masking, no explicit padding-aware
mask needed because padding is appended only after the last real token
(see A3).

**Where does the "CLS" position sit, and does it see the patient?** It is
**appended**, not prepended, and it is not a fixed vocabulary position used
during pretraining at all — see the "MOST IMPORTANT" finding below. Given
appended placement + causal attention, if it existed during training it
*could* attend to the entire preceding window; the "prepended-CLS-can't-see-
anything" failure mode does **not** apply here. The real problem is different
and, in this codebase, more concrete.

**MOST IMPORTANT — does the CLS/prediction-token position contribute to any
pretraining loss?**

**No — because it never appears in a pretraining sequence at all.** This is
not merely "unweighted in the loss"; the token is structurally absent from
every pretraining batch:

- The run's dataset class is `bonsai.modules.datasets.PretrainDataset.ARPretrainDataset`
  (`opera/configs/daly_care_pretrain.yaml:39`).
- `ARPretrainDataset._prepare_subject` → `PretrainDataset._prepare_subject`
  (`bonsai/modules/datasets/PretrainDataset.py:35-52`) calls
  `censor_subject(subject, self.cutoff_date, inclusive=False)` — **no
  `predict_token_id` argument** (`PretrainDataset.py:38`).
- `censor_subject` only appends a predict/CLS token when `predict_token_id is
  not None` (`bonsai/functional/censoring.py:8-35`, specifically lines 32-33).
  Since pretraining never passes it, `append_predict_token`
  (`censoring.py:38-86`) is never invoked during pretraining.
- The predict-token append path is exercised only by
  `FinetuneDataset.__getitem__` (`bonsai/modules/datasets/FinetuneDataset.py:27-46`,
  line 34-38 passes `predict_token_id=self.predict_token_id`), which is used
  by finetuning and by the embedding-extraction path
  (`extract_outcome_transfer_embeddings.py:550-556` builds a `FinetuneDataset`
  with `predict_token_id=int(vocabulary["[CLS]"])`).
- The pretraining loss (`bonsai/modules/lightningmodules/PretrainModule.py:28-62`)
  is computed from `ARPretrainDataset`'s `target = code[1:]` (shifted-by-one
  next-token labels, `bonsai/modules/datasets/PretrainDataset.py:197-198`).
  Since `[CLS]` never occurs as an input code or as a target in any
  pretraining example, its embedding row and its role in the residual stream
  are **never exercised as "the summary position"** during training. Its
  vocabulary embedding participates in gradient updates only incidentally,
  via weight tying with `pretrain_head` (`bonsai_nets.py:217`,
  `self.pretrain_head.weight = self.embeddings.code_embedding.weight`) — and
  only if `[CLS]` is ever predicted as a candidate next token for some other
  position, which is essentially never since it doesn't appear in the data.

  Conclusion: the frozen checkpoint's behavior at the appended `[CLS]`
  position, at inference time, is an **extrapolation to a token/role
  configuration the model never saw as a target during training** — not a
  learned patient summary. This is a stronger and more specific version of
  the "no training objective" hypothesis: it isn't that the objective ignores
  CLS's *loss*, it's that CLS as an *input token in this appended role* is
  entirely out-of-distribution relative to pretraining.

**Per-token hidden states / `output_hidden_states`:** `BonsaiBase.encode`
returns only the **final-layer** per-token hidden states as one tensor,
`(B, S, 64)` (`bonsai_nets.py:107-165`); there is no `output_hidden_states`
flag and no mechanism exposing intermediate-layer states. `self.layers` is a
plain `nn.ModuleList` (`bonsai_nets.py:74-88`) that a caller could iterate
manually, but no existing code path does. Part B will need new code to
capture intermediate layers.

**Normalisation/dropout/bottleneck before export:** the final `self.layernorm`
(`bonsai_nets.py:89,156`) is applied to every position, including the CLS
position, before any pooling. `self.drop` (`nn.Dropout(dropout)`,
`bonsai_nets.py:73,149`) sits right after the embedding sum and before layer 1;
at extraction time `encoder.eval()` is called
(`extract_outcome_transfer_embeddings.py:197`), so dropout is inert (identity)
for the reported embeddings — no rank-reducing stochastic noise at export
time. There is no other bottleneck.

**Parameter count / vocab size / architecture for this run:** from
`opera/configs/daly_care_pretrain.yaml:46-49`: `hidden_size=64`,
`num_layers=4`, `num_attention_heads=4` (head_dim=16, `mha.py:33`),
`max_seqlen=3372`. FFN dim = `4*hidden_size=256`
(`bonsai/modules/networks/components/mlp.py:7-8`, `Mlp.fc1`). `bias: false`
(`daly_care_pretrain.yaml:50`). Exact vocabulary size and therefore exact
total parameter count (which is vocab-size-dominated for a 64-dim model,
since `code_embedding` and the weight-tied `pretrain_head` are `vocab_size ×
64`) are **undetermined (requires cluster access)** — the real
`vocabulary.pt` for the DALY-CARE cohort is not present in this checkout.
Structurally: total params = `vocab_size*64` (code embedding, weight-tied to
the LM head so counted once) + segment embedding (`max_seqlen*64` =
`3372*64` ≈ 216K) + 2×Time2Vec (`age`, `abspos`; each has `2 + 2*(63)` ≈ 128
scalar params — negligible) + 4 transformer layers × (`Wqkv`: `64*192`=12,288;
`out_proj`: `64*64`=4,096; `fc1`: `64*256`=16,384; `fc2`: `256*64`=16,384; two
LayerNorms with `bias=False` → weight-only, 64 params each) ≈ 4 × ~49,280 ≈
197K for the transformer stack, plus the final LayerNorm (64 params). The
embedding table (`vocab_size*64`) dominates once vocab size exceeds a few
thousand tokens, which is very likely for a national-registry vocabulary —
**exact split undetermined without the real vocabulary.pt**.

---

## A2. Tokenisation

- **`code`**: tokenised via `EHRTokenizer.tokenize`
  (`bonsai/modules/tokenizer/tokenizer.py:112-114`) — a strict
  string→int dictionary lookup (`codes.replace_strict(self.vocabulary,
  default=self.vocabulary["[UNK]"])`), one vocabulary entry per unique string
  seen during vocabulary construction (`update_vocabulary`, lines 53-63; no
  minimum-frequency threshold is implemented in this tokenizer — every
  distinct code string becomes its own vocabulary id, subject only to the
  optional `cutoffs` prefix-length truncation in `limit_code_length`,
  lines 116-128, e.g. `{"D": 4}` shortens `D123456`→`D1234` before vocab
  assignment). OOV at *tokenize* time (not vocabulary-build time) maps to
  `[UNK]` (line 114).
- **`numeric_value` / lab values**: for this run, `value_bin_vocab_size: 0`
  and `value_embedding_mode: legacy`
  (`opera/configs/daly_care_pretrain.yaml:56-57`). Per
  `DATA_FORMAT.md:425-431`, "ehr2meds emits the joined `LAB_CODE//bin_k` code
  and `configs/daly_care_data.yaml` deliberately sets `numeric_value_mode:
  legacy`. BONSAI therefore consumes the joined code as one ordinary
  categorical token and does not duplicate it with a `[VAL]` position." With
  `value_bin_vocab_size=0`, `EhrEmbeddings.__init__` never constructs
  `value_bin_embedding`/`value_projection` at all
  (`bonsai/modules/networks/components/embeddings.py:25-34`) — there is no
  continuous numeric embedding pathway active in this checkpoint. Every lab
  observation is one categorical vocabulary token (code string with a
  `//bin_k` suffix baked in upstream by ehr2meds); the number of distinct lab
  tokens equals the number of distinct `LAB_CODE//bin_k` strings that
  survived into the vocabulary — **exact count undetermined (requires the
  real vocabulary.pt)**.
- **`numeric_value_normalized` / `numeric_value_bin` / `numeric_value_binned`**:
  only consumed by the *alternative* `combined_binning` mode
  (`bonsai/functional/create_data.py:25-90`, `DATA_FORMAT.md:433-457`), which
  this run does not use (`value_embedding_mode: legacy`). Under
  `combined_binning` these would drive a separate `[VAL]` token position with
  a projected scalar; **not applicable to the checkpoint under audit**.
- **`text_value`**: no reference to a `text_value` column anywhere in
  `bonsai/` or `opera/` (checked via grep across both packages). Undetermined
  whether ehr2meds ever emits it upstream — from this repo's perspective it
  is simply not a recognized/consumed MEDS column.
- **`source_block`, `row_idx`**: `row_idx`/`row_id` are consumed only as
  ordering keys during `create_data.py` processing (`OPTIONAL_TOKEN_COLUMNS`,
  `bonsai/functional/create_data.py:7-17`; kept in the parquet schema at
  lines 171-174) — not part of `code`, `age`, `abspos`, or `segment`, so they
  are **not model input features**. There is no `source_block` column
  anywhere in `bonsai/functional/create_data.py`'s selected output columns
  (`create_data.py:53-59,164-181`) or in `EhrEmbeddings.forward`'s consumed
  fields (`embeddings.py:36-79`: `code, age, abspos, segment,
  value_bin, value_normalized, value_present` only). **However**, provenance
  is not absent from the model — it is folded into the `code` string itself
  upstream of this repo, as a `"{source}//{concept}"` namespace prefix. This
  is confirmed by `opera/functional/source_analysis.py:21-27,74-79`, which
  documents and parses exactly this convention (`"LPR3//D501"` → source
  `"LPR3"`, `"BACKGROUND//sex_male"` → ignored, etc.) purely from the token
  *string*, i.e. this analysis module treats the vocabulary itself as the
  carrier of source identity. Since `EHRTokenizer.tokenize` is a verbatim
  string→id map with no normalization of the prefix
  (`tokenizer.py:112-114`), **any source/table distinction ehr2meds bakes
  into the code string is directly present in the vocabulary and directly
  visible to the model as ordinary token identity** — there is no mechanism
  in this repo that strips or obfuscates it. The specific literal LPR2 table
  names (`SDS_t_adm`, `SDS_t_diag`, `SDS_t_sksopr`, `SDS_t_sksube`) and LPR3
  table names (`SDS_kontakter`, `SDS_diagnoser`, `SDS_procedurer_kirurgi`,
  `SDS_procedurer_andre`) that the user's own source-analysis found predictive
  of era do not literally appear in this repo (they live in ehr2meds, an
  external upstream pipeline not vendored here) — **undetermined at that
  granularity from this repo alone**, but the mechanism that would carry
  such a signal (arbitrary provenance text folded into the tokenized `code`
  string, with no filtering) is confirmed to exist and is exactly what
  `source_analysis.py` was built to detect.
- **`code_components` struct fields** (gender, diagnosekode, diagnosetype,
  c_diag, c_diagtype, c_tildiag, diagnoseart, aktionsdiagnose, kontaktarsag,
  prioritet, kontakttype, hovedspeciale_ans, region_ans, c_adiag, c_kontaars,
  c_pattype, priority, procedurekode, proceduretype, c_spec, c_sgh, c_opr,
  c_oprart, atc, c_atc, c_patienttype, analysiscode): none of these field
  names appear anywhere in `bonsai/` or `opera/` (grepped). BONSAI's ingestion
  (`create_data.py`, `create_features` in `bonsai/functional/features.py`)
  only ever reads `subject_id, code, time` (+ optional
  `row_idx/row_id/value_normalized/value_bin/value_present`) from the input
  parquet (`features.py:9-15,53-59`). If `code_components` exists, it is
  consumed and flattened into the single `code` string by ehr2meds before the
  parquet reaches this repo — **undetermined from this repo** whether/how
  each named subfield is folded in; it is out of scope of the BONSAI/OPERA
  codebase.

- **Absolute calendar time**: **yes, tokenised/embedded directly**, and this
  is the second major, concrete root cause alongside A1's "CLS never trained"
  finding. `abspos` is defined as literal **hours since the Unix epoch**
  (`bonsai/functional/features.py:118-129`, `compute_abspos`; confirmed again
  in the code comment at `embeddings.py:132`: "Absolute position is expressed
  in hours since the Unix epoch and is therefore around 450,000 for
  contemporary records"). `EhrEmbeddings.forward` feeds `abspos` through
  `self.abspos_embedding = Time2Vec(hidden_size, clip_range=100)`
  (`embeddings.py:22`) and **adds it directly into every token's embedding**:
  `embeddings += self.abspos_embedding(abspos)` (`embeddings.py:49`).
  `Time2Vec`'s first output channel is a **linear** function of the raw input,
  clipped to `[-100, 100]` (`embeddings.py:141,144-149`:
  `linear_1 = w0*tau + phi0`, then clamped); the remaining channels are
  periodic (`cos`) functions of `w*tau + phi` (`embeddings.py:151`). This is
  an explicit, continuous, monotonic-ish (until clipping saturates) encoding
  of raw calendar time added straight into the residual stream — **for every
  position, including the appended CLS/predict token**, whose own `abspos` is
  set to `censor_date_abspos` (the exact prediction/index date) by
  `append_predict_token` (`censoring.py:52-57`). So even setting aside A1's
  "CLS never trained" finding, the CLS token's embedding contains a directly
  injected, largely linear feature of the patient's real-world index date
  before any attention happens — attention is not even required for era to
  reach the CLS residual stream, since it's summed into that token's own
  starting vector. There are no relative-only temporal features in this
  architecture: `age` uses the same `Time2Vec(clip_range=100)` mechanism
  (`embeddings.py:21,48`) and is relative (years since birth), but `abspos`
  is absolute and calendar-anchored, and both are added to *every* token,
  not just used for a positional/attention bias.
  There is no year-token / date-bin vocabulary anywhere in `bonsai/` or
  `opera/` (grepped) — the leakage mechanism is this continuous embedding,
  not a discrete "year token."
- **Birth date and sex**: birth date drives per-token `age` via
  `create_background`/`compute_age` (`bonsai/functional/features.py:62-116`)
  and is not itself stored as a token value in the model input tensors;
  DOB and other zero-timestamp demographic rows are relabeled
  `"BACKGROUND//{code}"` and time-filled to the birth timestamp
  (`features.py:84-99`), then included in the sequence as ordinary tokens at
  segment 0 (`DATA_FORMAT.md:39,47`, e.g. `"BACKGROUND//sex_male"`,
  example vocabulary at `DATA_FORMAT.md:66-67`). Because these background
  tokens still carry a real `abspos` (birth's absolute time) through the same
  `Time2Vec` mechanism, **birth cohort/era is a second, independent calendar
  leakage channel**, though it reaches the CLS token only via attention
  (indirect) rather than via direct summation (as the CLS token's own abspos
  does).
- **Vocabulary construction**: `EHRTokenizer.update_vocabulary`
  (`tokenizer.py:53-63`) adds every new unique code string with no frequency
  filter. `cutoffs` (constructor arg, used via `bonsai/run/create_data.py`,
  not fully re-read here) truncates code-string length by prefix before
  vocabulary assignment, which is a code-shortening policy, not a frequency
  threshold. `vocabulary_cutoff_abspos` (`tokenizer.py:11,26,39-43`;
  `bonsai/run/create_data.py:45-48`, populated from a
  `vocabulary_cutoff_date` config key) restricts which codes are eligible to
  **enter the vocabulary at all** to events before that date — a train/vocab
  leakage guard, not a frequency threshold. Default special tokens:
  `[PAD]=0, [CLS]=1, [SEP]=2, [UNK]=3, [MASK]=4` (`tokenizer.py:15-21`).
  Whether the DALY-CARE run's actual `create_data` invocation set a
  `vocabulary_cutoff_date` or a `cutoffs` dict is a run-config question —
  **undetermined (requires the cluster's actual `create_data` config/run
  logs for this cohort)**; `opera/configs/daly_care_pretrain.yaml` itself
  only points at pre-built artifact paths (`paths.vocab`, etc.), it does not
  re-run `create_data`.

---

## A3. Windowing and collation

- **`max_seq_len` for this run**: `3372`
  (`opera/configs/daly_care_pretrain.yaml:49`, `model.max_seqlen`; also
  `training.max_len: ${model.max_seqlen}` at line 66).
- **Right-aligned, most-recent-events window, confirmed**:
  `truncate_subject` (`bonsai/functional/truncation.py:55-101`) with
  `strategy="tail"` (`training.truncation_strategy: tail`,
  `daly_care_pretrain.yaml:67`, and `validation_truncation_strategy: tail`,
  line 68) calls `_clinical_window_start` (`truncation.py:27-52`), which for
  `strategy == "tail"` returns `max(last_start, 0)` where
  `last_start = clinical_length - kept_clinical_tokens`
  (line 34) — i.e., it keeps the window ending at the very last available
  clinical event (the most recent pre-cutoff events), not a random slice.
  `tail_window_probability: 1.0` (line 69) is consistent with — though
  redundant given `strategy: tail` already always takes the tail deterministically.
- **Post-index events during pretraining?** No. Truncation/censoring in
  `PretrainDataset._prepare_subject` (`bonsai/modules/datasets/PretrainDataset.py:35-52`)
  applies the *dataset-wide calendar cutoff* (2022-01-01, see below) via
  `censor_subject(..., inclusive=False)`, which drops every event at or after
  the cutoff (`censoring.py:20-26`: `bisect_left` when `inclusive=False`,
  then slices `[:idx]`, `censoring.py:29-30`). Since no `predict_token_id` is
  passed, no per-patient index-date censoring/appending occurs at all during
  pretraining — pretraining only knows about the single global 2022-01-01
  boundary, not any per-patient treatment/index date. (Per-patient index-date
  censoring is a *finetuning/extraction-time* concept, via
  `FinetuneDataset`/`attach_prediction_censor_abspos`, not a pretraining one.)
- **Index-anchored windows vs. chunked full records**: **one window per
  patient per `__getitem__` call**, not chunked. `PretrainDataset.__getitem__`
  (`PretrainDataset.py:54-56`) and `ARPretrainDataset.__getitem__`
  (`PretrainDataset.py:195-233`) each call `_prepare_subject` once and return
  one truncated window; there is no chunk-sampling logic, no multi-window
  iteration, and no windows-per-epoch parameter anywhere in
  `PretrainDataset.py` or `PretrainDataModule.py`. Each training epoch sees
  exactly one (tail-truncated, or `mixed_window`/`random_window` if configured
  — this run uses `tail`) slice per patient.
- **2022 cutoff, exact location**: `opera/configs/daly_care_pretrain.yaml:60-65`:
  ```yaml
  training:
    cutoff_date:
      year: 2022
      month: 1
      day: 1
  ```
  passed to `PretrainDataModule(..., cutoff_date=cfg.training.get("cutoff_date"), ...)`
  (`opera/run/daly_care_pretrain.py:52`), stored on the data module
  (`bonsai/modules/datamodules/PretrainDataModule.py:41`), and passed
  identically to **both** the train dataset and the val dataset
  (`PretrainDataModule.py:79,92,107,117` all pass `cutoff_date=self.cutoff_date`).
  Inside the dataset, `PretrainDataset.__init__` converts it once to an
  absolute position (`compute_abspos(datetime(**cutoff_date))`,
  `PretrainDataset.py:31-33`) and every `_prepare_subject` call censors with
  `inclusive=False` (line 38) — i.e. this is a **data-loader-level (Dataset
  constructor) filter**, applied per-`__getitem__`, identically for train and
  validation, confirming the config comment
  (`daly_care_pretrain.yaml:60-61`: "neither SSL training nor validation sees
  events at or after 2022-01-01").
- **Padding / attention mask at extraction time**: `PretrainDataset._prepare_subject`
  sets `attention_mask = torch.ones(len(code), dtype=torch.bool)` per
  unpadded sample (`PretrainDataset.py:48-50`); `FinetuneDataset.__getitem__`
  does the same after its own truncation (`FinetuneDataset.py:44`).
  `dynamic_padding` (`bonsai/functional/collate.py:13-36`) then right-pads
  every per-token field (including `attention_mask`) to the batch's max
  length using `torch.nn.utils.rnn.pad_sequence` (`collate.py:26-30`), with
  `attention_mask` padded with `0`/`False` (default branch of
  `_padding_value`, `collate.py:5-10`). So yes — a correct, real attention
  mask (`1` for real tokens including the appended CLS/predict token, `0` for
  padding) is available at extraction time and is exactly what
  `_pool_cls_last` uses to find the true last-token index
  (`extract_outcome_transfer_embeddings.py:486-494`). Padding is on the
  **right**, after the real tokens (including any appended predict token),
  which is safe under causal, `is_causal=True` attention because no real
  token ever attends to a later, padded position.

---

## A4. Objective and training state

- **Loss function**: causal (autoregressive) next-token cross-entropy.
  `compute_pretrain_loss` (`bonsai/modules/lightningmodules/PretrainModule.py:28-62`):
  `loss = code_loss_fn(logits.view(-1, vocab), labels.view(-1))` where
  `code_loss_fn` is `nn.CrossEntropyLoss` (or the FlashAttention CE variant
  for `attn_type="flash"`; this run uses `attn_type: sdpa` →
  plain `nn.CrossEntropyLoss`, `PretrainModule.py:13,88`).
- **Is loss computed at every position?** Only at positions where
  `labels != -100`. In `BonsaiPretrain.forward`
  (`bonsai/modules/networks/bonsai_nets.py:219-235`): `mask = labels != -100;
  code_hidden_state = last_hidden_state[mask]`. For `ARPretrainDataset`,
  `target = code[1:]` with `target.masked_fill(target == 0, -100)`
  (`PretrainDataset.py:197-198`) — i.e. every real next-token position is a
  loss target; only positions whose *next* token is `[PAD]` (id 0, meaning
  end-of-sequence-before-padding) are excluded. There is **no per-token-type
  weighting, masking, or exclusion** beyond that pad rule — lab tokens (which
  are ordinary categorical tokens under `numeric_value_mode: legacy`, see A2)
  are included in the CE loss on the same footing as any other code.
- **Auxiliary/patient-level losses, regularisation on pooled
  representations**: none. `compute_pretrain_loss` only ever adds two other
  optional terms — `value_bin` CE and `value_regression` MSE — and both are
  gated on `value_bin_head`/`self.value_bin_vocab_size>0`
  (`bonsai_nets.py:208-214,229-261`) and on `value_regression_loss_weight`
  (`daly_care_pretrain.yaml:76`: `value_regression_loss_weight: 0.0`) and
  `value_bin_vocab_size: 0` (line 56). Both are therefore **inactive for this
  run** — the loss is pure per-token next-code CE, nothing else. Nothing in
  `PretrainModule` or `BonsaiPretrain` ever reads a pooled/CLS-style
  representation during training (confirms A1: pooling literally does not
  exist inside the pretraining loop).
- **Training logs for this checkpoint** (final train/val loss, plateau,
  steps/epochs, LR schedule completion, smoke-test vs. full run):
  **undetermined (requires cluster access)**. No `pretrain.log`, CSV-logger
  `training_runs/version_*/metrics.csv`, or wandb directory for a DALY-CARE
  run exists in this checkout; the only `pretrain.log`/`metrics.csv` present
  belong to the tiny synthetic CI fixture under `.ci/models/correlated_MEDS_data/...`
  and `.tmp/e2e-models/...`, which are unrelated smoke-test artifacts, not
  the real run. From the config alone: `training.epochs: 10`,
  `learning_rate: 3e-4`, `scheduler_warmup_epochs: 0.1`,
  `batch_size: 128`, `limit_val_batches: 1.0`, `limit_train_batches: 1.0`
  (`daly_care_pretrain.yaml:70-78`) — i.e. the config is not itself a
  reduced/smoke-test config (no `limit_train_batches < 1`, no `fast_dev_run`),
  but whether the actual run reached all 10 epochs, and what the loss curve
  looked like, cannot be determined without the cluster's logs. **This is
  flagged prominently per the ground rules: if the real run under-trained or
  was truncated, that alone would produce a collapsed low-rank
  representation independent of the CLS/abspos findings above, and I cannot
  rule that out from this checkout.**

---

## A5. Splits

- Pretraining reads exactly two physical shard files, configured as
  `paths.train_split: ${paths.dir}/subject_data_train.pt` and
  `paths.val_split: ${paths.dir}/subject_data_tuning.pt`
  (`opera/configs/daly_care_pretrain.yaml:36-37`), loaded in
  `PretrainDataModule.setup_fit` (`bonsai/modules/datamodules/PretrainDataModule.py:58-64`).
- **Confirmed: the prospective train/tuning/held-out split used for
  downstream evaluation is a *different, later* partition than these two SSL
  shards, and does not exclude anyone from them.** Quoting
  `DATA_FORMAT.md:49-53` directly:
  > "**Physical splits**: ehr2meds' random 90/10 preprocessing partitions are
  > written to `subject_data_train.pt` and `subject_data_tuning.pt`. OPERA
  > pools both files for supervised tasks. Prospective train/tuning/held-out
  > membership is defined by the outcome/index-date manifest, not by a third
  > physical subject-data file."
  This is corroborated by the extraction code itself:
  `extract_shared_split_embeddings`
  (`opera/run/extract_outcome_transfer_embeddings.py:497-527`) loads a single
  pooled `subject_pool` from exactly those same two physical files
  (`_subject_paths`/`ssl_train`+`ssl_validation`,
  `extract_outcome_transfer_embeddings.py:131-142` and the analogous
  `_subject_paths` at line 599 for the outcome-transfer CLI), then partitions
  that **same pool** into `train`/`tuning`/`held_out` **only at extraction
  time**, by intersecting with `subject_id`s from the outcome/index-date
  manifest (`extract_shared_split_embeddings`,
  lines 532-549: `for split in ("train","tuning","held_out"): ... selected =
  [s for s in subject_pool if int(s["subject_id"]) in records]`).
  `population_full.csv` — used to filter `subject_data_train.pt`/
  `subject_data_tuning.pt` before pretraining
  (`PretrainDataModule.setup_fit`, lines 62-64) — is documented to "contain
  **all patients** across all splits" (`DATA_FORMAT.md:237`).
  **Conclusion: the 6,393 downstream held-out patients are not excluded from
  pretraining; they are physically present in `subject_data_train.pt`/
  `subject_data_tuning.pt` and are only labeled "held_out" by the separate,
  later outcome-manifest split used for prospective evaluation.** This
  matches the suspicion stated in the task and is corroborated by the docs
  and by the pooling logic in the extraction script.

---

## A6. End-to-end trace of two patients (2010 and 2024 first-line treatment)

**Undetermined — cannot be completed from this checkout.** This requires:
real DALY-CARE MEDS/subject-data shards, the real vocabulary (to invert
token ids back to human-readable strings), and a real index-date manifest to
pick one 2010 patient and one 2024 patient. None of these exist locally —
only the small synthetic `correlated_MEDS_data` CI fixture is present, which
has no real diagnosis codes, no realistic 2010–2024 date spread, and no
correspondence to the DALY-CARE cohort. Running this trace requires cluster
access to the real `$BONSAI_PROCESSED_DATA/daly_care/` artifacts.

If/when cluster access is available, the trace is mechanical given the code
above: load `subject_data_{train,tuning}.pt`, filter to the two subject_ids,
invert `vocabulary.pt` to get human-readable code strings
(`opera/functional/source_analysis.py:83-85`, `invert_vocabulary`, is a ready-
made helper), and run each subject through `EHRTokenizer`/the same
`FinetuneDataset` + `censor_subject(..., predict_token_id=vocab["[CLS]"])`
path used at extraction to see the exact emitted sequence, its length, first/
last 20 tokens, and per-source breakdown (reusing
`opera/functional/source_analysis.py:36-80`, `infer_patient_sources`, for the
source/token-type breakdown request). I did not fabricate numbers for this
section.

---

## Summary of root-cause findings from Part A

Two independent, code-confirmed mechanisms are sufficient to explain the
observed pathology without invoking undertraining:

1. **The CLS/prediction-token position is entirely absent from pretraining.**
   It is appended only by `FinetuneDataset`/extraction-time code
   (`censoring.py:38-86`, invoked from `FinetuneDataset.py:34-38` and
   `extract_outcome_transfer_embeddings.py:550-556`), never by
   `ARPretrainDataset` (the actual pretraining dataset,
   `daly_care_pretrain.yaml:39`). The frozen checkpoint was never optimized
   to produce a patient summary at this position — reading it out is reading
   an out-of-distribution extrapolation, not a trained representation.
2. **Absolute calendar time is injected directly into every token's
   embedding, including the CLS token's own embedding, via a linear+periodic
   `Time2Vec(abspos)` term** (`embeddings.py:22,49,131-154`), and the CLS
   token's `abspos` is set to the exact index/treatment date
   (`censoring.py:52-57`). This reaches the CLS residual stream by direct
   summation, with no dependence on attention at all — a much shorter, more
   reliable path to strong era-predictability than any inferred data-source
   leakage, and fully consistent with kNN purity lift being high for exact
   treatment year specifically.

Both mechanisms point the same direction as the task's leading hypothesis
(pooled token hidden states over intermediate layers should do far better
than the CLS token), and Part B is designed to test that directly. A4's
missing training logs mean I cannot rule out undertraining as a *contributing*
third factor; this should be checked against the real logs before treating
Part B/C/D results as definitive.

---

# PART B: multi-variant pooled extraction — implementation

**Status: code written and unit-tested against a tiny synthetic encoder on
this machine (no GPU, no real checkpoint). Not yet run against the real
DALY-CARE checkpoint/data — that requires cluster access. See "Running on
the cluster" below.**

## What was added (no existing extraction code was modified)

1. `bonsai/modules/networks/bonsai_nets.py`, `BonsaiBase.encode`: added an
   **additive, default-off** `output_hidden_states: bool = False` parameter.
   When `False` (the default, used by every existing call site —
   `BonsaiBase.forward` and `BonsaiFinetune.get_pooled_representation`, the
   only two callers of `.encode(` on a model instance in this repo, grepped
   to confirm), behavior and return type are byte-for-byte identical to
   before. When `True`, it additionally returns a list of per-layer
   residual-stream outputs (one per transformer layer, unpacked to
   `(batch, seq, hidden)`, **pre**-final-layernorm), so a single forward
   pass can serve every requested depth. This is the only change to
   pretraining-adjacent code, and it does not change training or config —
   it is an inference-time capability addition with a verified-identical
   default path (see `tests/test_pooled_extraction.py::test_encode_default_behavior_is_unchanged`).
2. `opera/functional/pooled_extraction.py` (new): `resolve_target_layers`
   (maps `final`/`d75`/`d50`/`d25` to concrete 1-indexed layer numbers —
   `{final: 4, d75: 3, d50: 2, d25: 1}` for this run's `num_layers=4`),
   `predict_token_mask`, and `pool_variants` (the five pooling
   implementations, mask-aware).
3. `opera/run/extract_pooled_patient_embeddings.py` (new): a new CLI, built
   on the same `read_index_table`/`prepare_prediction_origins`/
   `load_frozen_encoder`/`FinetuneDataset`/`dynamic_padding` primitives as
   the existing `extract_patient_embeddings.py`, that runs one forward pass
   per batch with `output_hidden_states=True` and writes one NPZ per
   `(layer, pooling)` combination with the schema `subject_ids, embeddings,
   splits` — identical to the existing cls_last NPZ. Filenames:
   `{output-prefix}__L{layer}__{pooling}.npz` (e.g.
   `daly_care_pretrain__L4__mean.npz`).

## Design decisions worth flagging

- **`attn_cls` is not implemented.** This architecture's attention
  (`bonsai/modules/networks/components/mha.py:13-21`) calls
  `torch.nn.functional.scaled_dot_product_attention`, a fused kernel that
  does not return attention probabilities (no `need_weights`-style option
  exists for it, unlike `nn.MultiheadAttention`). Retrieving final-layer
  attention weights into the CLS position would require reimplementing the
  attention math (manual `q @ k^T`, causal mask, softmax) rather than
  reading an existing value — more than "cheaply available" per the task
  instructions, so it is skipped, and the script records this reason in its
  metadata sidecar (`attn_cls_skipped_reason`) rather than silently omitting
  it.
- **`mean`, `last`, `mean_last_128`, and `max` all exclude the appended
  prediction/CLS token from their pooling window**; only `cls` reads it.
  This is a deliberate interpretation of the task's `last` definition ("final
  real token, i.e. **the event immediately preceding index**" — explicitly
  not the appended token), generalized consistently to the other three
  poolings. Rationale: Part A established that the appended token never
  occurs in any pretraining sequence (A1) and that it carries a direct,
  almost-linear calendar-time signal via its own `abspos` (A2). Including it
  in an average/max alongside genuine clinical tokens would let one
  out-of-distribution, era-encoding position contaminate every non-`cls`
  variant, undermining the entire point of the ablation. This is
  implemented as a pooling-level mask (`content_mask = attention_mask &
  ~predict_mask`), not a change to how the batch is built — every variant
  still comes from the exact same input window (censoring + appended
  predict token) as the existing cls_last pipeline, so the "identical
  patient set, identical ordering" requirement holds by construction.
- **Only the `final` depth applies the model's final `LayerNorm`.**
  Intermediate depths (`d75`, `d50`, `d25`) are read directly from the
  transformer stack's residual stream, pre-layernorm, since that final
  `LayerNorm` was fit specifically to condition inputs to the LM head after
  all 4 layers, not to normalize an intermediate layer's output.
- **Correctness check built into the test suite, not just asserted**:
  `tests/test_pooled_extraction.py::test_extract_pooled_variants_final_cls_matches_existing_cls_last_convention`
  independently rebuilds the same 5 synthetic patients through
  `BonsaiFinetune.get_pooled_representation` (the existing cls_last
  convention) and checks exact numerical agreement with this script's
  `("final", "cls")` variant. This is the sanity check that the new
  extraction path is a strict superset of the old one, not a
  reimplementation that could silently diverge.
- **Ordering/identity guarantee**: `extract_pooled_variants` walks
  `subject_pool`/splits in exactly the same order and with the same
  duplicate/coverage checks as
  `extract_outcome_transfer_embeddings.extract_shared_split_embeddings`
  (reused via direct import: `_reference_records`, `_resolve_device`,
  `load_frozen_encoder`), so subject ordering matches the existing cls_last
  NPZ for the same checkpoint/index-table inputs. The CLI additionally
  accepts `--reference-cls-npz`: if given, it asserts
  `np.array_equal(subject_ids, reference["subject_ids"])` before writing
  anything, and refuses to write on mismatch; if omitted, it prints an
  explicit warning that the cross-check was skipped rather than silently
  assuming agreement.

## Testing performed on this machine

`tests/test_pooled_extraction.py` (11 tests, all passing) covers: the
`encode()` additive contract and its right-padding invariance under causal
attention; `resolve_target_layers`/`predict_token_mask`/`pool_variants`
correctness on hand-computed synthetic tensors; and an end-to-end run of
`extract_pooled_variants` against a tiny synthetic `BonsaiBase` encoder (not
a real checkpoint) and a 5-patient synthetic subject pool spread across two
physical shards and three prospective splits, checking output schema,
subject-id ordering, and the final-cls-agreement property above. The full
existing test suite was re-run after the `bonsai_nets.py` change to confirm
no regression (see run log; all passing, ruff clean).

**Not done on this machine, and cannot be**: running this script against the
real DALY-CARE checkpoint, vocabulary, subject-data shards, or index table
(none exist locally, no `.env`/`BONSAI_PROCESSED_DATA`; see the top-of-file
limitation note). Peak-GPU-memory and wall-time numbers in this file's test
run are for the tiny CPU-only synthetic fixture and are not representative
of the real 37,494-patient, `max_seqlen=3372` run — the script reports both
per real invocation via `torch.cuda.max_memory_allocated`/wall-clock timing
in its metadata sidecar, but real numbers require the cluster.

## Running on the cluster

```bash
python -m opera.run.extract_pooled_patient_embeddings \
  --checkpoint $BONSAI_MODELS/daly_care/daly_care_pretrain/best.ckpt \
  --vocabulary $BONSAI_PROCESSED_DATA/daly_care/vocabulary.pt \
  --subject-data-dir $BONSAI_PROCESSED_DATA/daly_care \
  --index-table <path-to-the-first-line-treatment index table used for the \
    original CLS extraction> \
  --reference-cls-npz <path-to-the-existing-cls_last-NPZ> \
  --batch-size 16 --num-workers 6 \
  --output-prefix daly_care_pretrain \
  --output-dir <output-dir>
```

Use the *same* `--index-table`/`--subject-col`/`--index-date-col`/
`--split-col` arguments that produced the existing CLS NPZ (whatever those
were originally — not directly visible from this checkout since the exact
invocation wasn't in a committed script I could find), and pass
`--reference-cls-npz` pointing at that existing NPZ so the script asserts
identical subject ordering before writing anything, per the task's
"Critical" requirement. Batch size may need to be smaller than 16 at
`max_seqlen=3372` depending on GPU memory; the script will report the peak
it actually used.
