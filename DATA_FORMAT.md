# OPERA Data Format Reference

Everything the pipeline reads or writes, and what each field must contain.

---

## 1. Directory layout

```
$BONSAI_PROCESSED_DATA/
├── {cohort}/                          # one directory per disease cohort
│   ├── subject_data_train.pt          # list[dict] — tokenised EHR, train split
│   ├── subject_data_tuning.pt         # list[dict] — tokenised EHR, val split
│   ├── subject_data_held_out.pt       # list[dict] — tokenised EHR, test split
│   ├── vocabulary.pt                  # dict[str, int] — token → id
│   ├── population_full.csv            # subject-level metadata + IPI scores
│   └── outcomes/
│       ├── mortality_1y.parquet
│       ├── mortality_2y.parquet
│       ├── treatment_failure.parquet
│       ├── aki_30d.parquet
│       ├── severe_infection_90d.parquet
│       └── {any_other_outcome}.parquet
```

All cohorts **share the same vocabulary** (token string → integer id).  
The vocabulary file from any cohort can be passed to any stage.

---

## 2. Subject data files (`subject_data_*.pt`)

A Python `list` of dicts, one dict per patient. Load with `torch.load(path)`.

| Key | Type | Shape | Description |
|---|---|---|---|
| `subject_id` | `int` | scalar | Unique patient identifier. Must match outcome parquets and population CSV. |
| `code` | `torch.LongTensor` | `(L,)` | Token ids. `L` varies per patient. |
| `abspos` | `torch.FloatTensor` | `(L,)` | Absolute position in hours since Unix epoch (from `compute_abspos`). |
| `segment` | `torch.LongTensor` | `(L,)` | Segment/visit index. Background tokens are segment 0. |
| `age` | `torch.FloatTensor` | `(L,)` | Patient age in years at each token. |

**Special tokens** (ids defined in vocabulary):
- `[PAD]` = 0
- `[CLS]` = 1  
- `[SEP]` = 2

**Background tokens**: each patient's sequence begins with static background tokens (sex, birth year, etc.) in segment 0. The number of background tokens is the same for all patients in a cohort and is inferred at runtime as `(subjects[0]["segment"] == 0).sum()`.

**Splits**: train → `subject_data_train.pt`, val → `subject_data_tuning.pt`, test → `subject_data_held_out.pt`. The `split` field in outcome parquets must use the string keys `"train"`, `"tuning"`, `"held_out"`.

---

## 3. Vocabulary file (`vocabulary.pt`)

A Python `dict` mapping token strings to integer ids. Load with `torch.load(path)`.

```python
{
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "BACKGROUND//sex_male": 3,
    "BACKGROUND//sex_female": 4,
    "LPR3//D501": 5,
    ...
}
```

All cohorts must use the same vocabulary file (or a superset of it). If DAPT adds new tokens, the expanded vocabulary is saved alongside the checkpoint and must be used for all downstream stages.

---

## 4. Outcome parquets (`outcomes/{name}.parquet`)

One parquet file per outcome per cohort. Each row is one patient.

**Required columns:**

| Column | dtype | Description |
|---|---|---|
| `subject_id` | int64 | Must match subject_data files and population CSV. |
| `split` | str | One of `"train"`, `"tuning"`, `"held_out"`. |
| `index_date` | datetime64[ns] | The prediction origin — the date from which the follow-up window is measured. Typically first-line treatment start. All outcomes for a patient should share the same index date so sequence truncation is consistent. |
| `outcome_date` | datetime64[ns] or NaT | Date the outcome occurred. `NaT` for patients who never experienced the event (censored). |
| `censor_date` | datetime64[ns] | The date beyond which the patient is no longer observed (last known alive date, or administrative censoring date). Must be ≥ `index_date`. |

**No `label`, `time_days`, or `event` columns are needed** — `binarize_outcomes()` computes these at runtime from the above columns using the `n_hours_start_include` / `n_hours_end_include` parameters in the config. Do not pre-binarize.

**How `index_date` is set for first-line treatment outcomes:**

```yaml
# In your create_outcome config:
index:
  type: relative
  hour_shift: 0        # index_date = date of the matching event (e.g. first chemo)

censor:
  type: absolute
  date:
    year: 2024
    month: 12
    day: 31            # follow-up/admin censoring date for non-event patients
```

OPERA truncates EHR sequences at `index_date`, not at `censor_date`.
`censor_date` is retained for follow-up and survival/IPCW calculations. If the
index-defining treatment token itself must be excluded, construct `index_date`
slightly before that event or remove that defining token during data creation.

**Minimum viable example** (5 patients):

```
subject_id  split       index_date           outcome_date         censor_date
1001        train       2018-03-15 00:00:00  2019-01-10 00:00:00  2019-01-10 00:00:00
1002        train       2017-07-22 00:00:00  NaT                  2020-12-31 00:00:00
1003        tuning      2019-11-05 00:00:00  2020-02-14 00:00:00  2020-02-14 00:00:00
1004        held_out    2016-04-01 00:00:00  NaT                  2018-04-01 00:00:00
1005        held_out    2020-01-10 00:00:00  2020-08-30 00:00:00  2020-08-30 00:00:00
```

`censor_date` is the last date on which follow-up is known. It may be later
than `outcome_date`; event time is always derived from `outcome_date`. For
patients without the event (`outcome_date = NaT`), `censor_date` is the last
known follow-up date.

---

## 5. Outcome eligibility sidecars

When outcome ascertainment differs by patient or outcome, write one eligibility
sidecar per cohort-outcome cell. Do not encode an unascertainable outcome as a
negative label.

The sidecar may be CSV or parquet and must contain exactly one row per patient:

| Column | Required | Description |
|---|---|---|
| `subject_id` | yes | Stable patient identifier. Duplicate rows are invalid. |
| `split` | yes | `train`, `tuning`, or `held_out`. |
| `eligible` | yes | Whether this outcome is ascertainable under the locked outcome-specific rule. |
| `eligibility_reason` | yes | `eligible` or a specific primary exclusion reason. Ineligible rows may not have an empty reason. |
| `ascertainment_eligible` | no | Whether the patient can contribute observed risk time to a censor-aware survival objective. Set this explicitly when incomplete fixed-horizon follow-up should still be retained as right-censored. |
| `source_covered` | no | Whether the relevant registry/source could observe the outcome at the prediction date. |
| `baseline_adequate` | no | Whether required pre-index measurements exist, for definitions such as creatinine-change AKI. |
| `post_index_adequate` | no | Whether the source remains observable over the endpoint's risk window. |
| `followup_adequate` | no | Whether follow-up is sufficient for the final fixed-horizon endpoint definition. |
| `outcome_observed` | no | Whether an outcome event was actually observed. This must not define source coverage. |

The eligibility rule must be fixed before model fitting. In particular:

- `eligible` means the event could have been ascertained, not that a clinician
  ordered the outcome-defining test.
- Absence of a laboratory result is not automatically a negative outcome.
- Presence of the outcome-defining result must not be used to prove coverage;
  that makes eligibility depend on the label.
- Death before a laboratory endpoint is handled according to the endpoint's
  declared censoring or competing-event rule. It is not automatically a
  full-window negative.
- AKI-like definitions should expose baseline adequacy separately from source
  coverage and post-index follow-up.

Fixed-horizon BCE and binary evaluation use `eligible`. Survival and
survival-contrastive objectives use `ascertainment_eligible` when present. For
older sidecars, a row with `followup_adequate=false` and no structural gate
failure is treated as ascertainable but right-censored. Complete-horizon
eligibility is then derived from `censor_date`; it must not be encoded as a
negative label.

An observed primary or competing event must be on or before `censor_date`. If
both are present, the earlier event determines the cause (ties favor the
primary event).

Configure a sidecar and, where appropriate, an outcome-specific coverage date:

```yaml
outcomes:
  mortality_1y:
    outcome_file: mortality.parquet
    registry_start_date: null
    eligibility_file: mortality__audit.parquet
  aki_30d:
    outcome_file: aki_first_documented.parquet
    registry_start_date: null
    eligibility_file: aki_first_documented__audit.parquet
    competing_outcome_file: mortality.parquet
```

An explicitly configured `registry_start_date: null` disables a cohort-level
fallback for that outcome. If the key is omitted, the cohort-level date is
used. Configured sidecars are enforced by contrastive, joint, per-task, hybrid,
survival, and evaluation loaders before labels are constructed; they are not
audit-only metadata.

Validate configured sidecars during readiness checks and create a tidy
cohort-flow artifact:

```bash
python -m opera.run.check_readiness \
  --config opera/configs/sweep_example.yaml \
  --require_existing_paths \
  --fail_on_issue

python -m opera.run.summarize_cohort_flow \
  --config opera/configs/sweep_example.yaml \
  --output ./results/cohort_flow.csv
```

The cohort-flow command is intentionally strict: every configured
cohort-outcome cell must name an existing, valid sidecar.

---

## 6. Population CSV (`population_full.csv`)

One row per patient. Used to filter subject data to a defined study population and to supply clinical score baselines (IPI etc.).

**Required columns:**

| Column | dtype | Description |
|---|---|---|
| `subject_id` | int64 | Must match subject_data and outcome parquets. |

**Optional columns** (used by sweep IPI baseline):

| Column | dtype | Description |
|---|---|---|
| `{ipi_score_col}` | float | Pre-computed clinical risk score (e.g. `nccn_ipi`, `cll_ipi`, `flipi2`). Higher = higher risk. Any scale — normalised to [0,1] internally. `NaN` for patients without a score. |

The population CSV should contain **all patients** across all splits. It is used purely for filtering (only patients in this CSV are kept) and for IPI lookup.

---

## 7. Outcome config in `contrastive_multicohort.yaml` / `joint_finetune.yaml`

```yaml
outcomes:
  {outcome_name}:
    outcome_file: {outcome_name}.parquet
    eligibility_file: {outcome_name}__audit.parquet
    n_hours_start_include: 1            # events must occur >= 1h after index_date
    n_hours_end_include: 8760           # events must occur <= 8760h (1 year) after index_date
                                        # null = open-ended (no upper bound)
```

Both hour bounds are inclusive. With date-only timestamps normalized to
midnight, `n_hours_start_include: 1` excludes the entire index calendar date;
use `0` only when same-time or same-day events belong in the estimand.

Survival-aware pair similarity is not scaled in raw days. Training builds a
pooled Kaplan-Meier primary-event mass for each outcome, maps each event or
censoring time onto that cumulative-mass grid with `torch.searchsorted`, and
applies the fixed `km_time_scale` in CDF space (default `0.25`). The task horizon
defines label eligibility but does not automatically set this kernel scale.

**Standard outcome definitions:**

| Outcome name | `n_hours_end_include` | Window |
|---|---|---|
| `mortality_1y` | 8760 | 1 year |
| `mortality_2y` | 17520 | 2 years |
| `treatment_failure` | null | open-ended |
| `aki_30d` | 720 | 30 days |
| `severe_infection_90d` | 2160 | 90 days |
| `pfs_1y` (progression-free survival) | 8760 | 1 year |
| `crp_response_90d` | 2160 | 90 days |

For fixed-horizon tasks, event-free follow-up is capped at the task horizon.
Training and binary evaluation exclude administratively censored patients who
do not reach that horizon. When `competing_outcome_file` is configured, death
inside the risk window is encoded as `event=2`; death after the horizon does
not become a competing event for that task. Competing deaths retain binary
label `0` for the cause-specific endpoint and remain explicit in survival-aware
contrastive weighting and survival summaries.

---

## 8. Sweep config (`sweep_example.yaml`)

```yaml
output_dir: ${BONSAI_RESULTS_ROOT}/opera_sweep
finetune_base_config: opera/configs/finetune.yaml

cohorts:
  {cohort_name}:
    data_dir: ${BONSAI_PROCESSED_DATA}/{cohort_name}
    ipi_score_col: {column_name}    # or null if no IPI
    population_file: null           # optional override, defaults to data_dir/population_full.csv

outcomes:
  {outcome_name}:
    n_hours_start_include: 1
    n_hours_end_include: {int or null}

model_variants:
  base_pretrain:
    encoder_ckpt: ${BONSAI_CHECKPOINT_ROOT}/pretrain/best.ckpt
    encoder_source: pretrain
  dapt:
    encoder_ckpt: ${BONSAI_CHECKPOINT_ROOT}/dapt/best.ckpt
    encoder_source: dapt
  opera:
    encoder_ckpt: ${BONSAI_CHECKPOINT_ROOT}/contrastive/best.ckpt
    encoder_source: contrastive
  opera_joint:
    encoder_ckpt: ${BONSAI_CHECKPOINT_ROOT}/joint_finetune/best.ckpt
    encoder_source: joint
  tabular_xgb:
    results_file: ${BONSAI_RESULTS_ROOT}/tabular/xgb_{cohort}_{outcome}.json
```

**Tabular baseline JSON format** (the file pointed to by `results_file`):

```json
{
  "discrimination": {
    "auroc": 0.742,
    "auprc": 0.381,
    "sensitivity": 0.651,
    "specificity": 0.778,
    "n_total": 412,
    "n_positive": 87,
    "prevalence": 0.211
  },
  "calibration": {
    "brier_score": 0.142,
    "ece": 0.031
  },
  "bootstrap_ci": {
    "auroc": {"mean": 0.742, "lower": 0.698, "upper": 0.783},
    "auprc": {"mean": 0.381, "lower": 0.321, "upper": 0.445}
  },
  "survival": {
    "concordance_index": 0.731,
    "n_total": 498,
    "n_events": 87,
    "per_horizon": {
      "365d": {"ipcw_auc": 0.738, "ipcw_brier": 0.138, "n_cases": 52, "n_controls": 289, "n_excluded": 157},
      "730d": {"ipcw_auc": 0.721, "ipcw_brier": 0.161, "n_cases": 87, "n_controls": 201, "n_excluded": 210}
    }
  }
}
```

The `survival` block is optional — if absent, only binary metrics appear in the results table. If you save `predictions.npz` alongside (see below), pairwise significance tests against OPERA also become available.

**Tabular `predictions.npz` format** (optional, enables DeLong/bootstrap significance tests):

```python
np.savez(
    "predictions.npz",
    subject_ids   = np.array([1001, 1002, ...]),   # int64, all test patients
    labels        = np.array([1, 0, ...]),           # int, full-follow-up patients only (binary_mask applied)
    probabilities = np.array([0.72, 0.31, ...]),     # float, same subset as labels
    logits        = np.array([0.94, -0.77, ...]),    # float (optional but recommended)
    times         = np.array([365.0, 820.0, ...]),   # float, days, ALL test patients
    events        = np.array([1, 0, ...]),            # int, ALL test patients
    binary_mask   = np.array([1, 0, ...], dtype=np.uint8),  # 1 = full follow-up
)
```

The `subject_ids` in `predictions.npz` must match across all model variants for a given cohort × outcome cell — significance testing aligns models on shared subject IDs.

---

## 8. Adding a new cohort — checklist

1. Process EHR data into MEDS format
2. Run `bonsai.run.create_data` → produces `subject_data_{split}.pt` and `vocabulary.pt`
3. For each outcome, run `bonsai.run.create_outcome` with an appropriate config → produces `outcomes/{name}.parquet`
4. Create `population_full.csv` with at minimum `subject_id` (+ IPI columns if available)
5. Add cohort to `contrastive_multicohort.yaml` under `cohorts:`
6. Add cohort to `joint_finetune.yaml` under `cohorts:`
7. Add cohort to `sweep_example.yaml` (or your sweep config) under `cohorts:`
8. Run contrastive training → new cohort's patients participate in cross-disease contrastive pairs automatically
9. Run readiness with `--require_existing_paths --fail_on_issue`
10. Run the sweep; configured missing cells fail readiness and strict training
    instead of disappearing from the experiment

**No code changes are required to add a cohort.** The production configs use
`require_all_configured_cells: true`: every listed cohort-outcome file and
sidecar must exist. If an outcome is intentionally unavailable for a cohort,
use a separate reduced experiment config or omit that cell from the canonical
contract rather than relying on a silent skip.

---

## 9. Adding a new outcome — checklist

1. Define the outcome event logic in a `create_outcome` config YAML
2. Run `bonsai.run.create_outcome` for each cohort → produces `outcomes/{name}.parquet` in each cohort directory
3. Add the outcome to `contrastive_multicohort.yaml` under `outcomes:` with
   `outcome_file`, `eligibility_file`, `n_hours_start_include`, and
   `n_hours_end_include`
4. Add the same block to `joint_finetune.yaml`
5. Add to your sweep config under `outcomes:`
6. Run contrastive training (or continue from a checkpoint with `save_last=True`) → new outcome automatically gets a `log_sigma` parameter and appears in all evaluation reports

**Choosing `n_hours_end_include`:**
- For acute outcomes (infections, AKI): use the clinical definition window (30d = 720h, 90d = 2160h)
- For mortality/progression: typically 1y = 8760h or 2y = 17520h
- For open-ended outcomes (treatment failure, any relapse ever): use `null`
- The contrastive kernel acts on Kaplan-Meier cumulative event mass. Keep the
  locked default `km_time_scale=0.25` for the primary analysis and treat any
  alternative as a named sensitivity analysis.

**Choosing `index_date`:**  
All outcomes for a patient must share the same `index_date` (typically
first-line treatment start). `ContrastiveDataset` validates the derived
prediction positions and fails before training when they disagree.
# Numeric combined binning

BONSAI's paper-aligned numeric path is enabled with
`numeric_value_mode: combined_binning` during data creation. An ehr2meds event
with `numeric_value_present=true` is expanded into two adjacent positions:

```text
LAB/CODE, [VAL]
```

The clinical position has no numeric payload. The `[VAL]` position carries
`numeric_value_bin` as `value_bin` and preferentially carries
`numeric_value_binned` as the model scalar (`value_normalized` is the fallback).
The normalized bin representative keeps inputs and MSE targets in `[0, 1]`
across concepts with different bin counts.

With causal pretraining, the hidden state at `LAB/CODE` predicts the scalar at
the following `[VAL]` position. `[VAL]` is excluded from categorical CE, so the
combined objective is ordinary next-code CE plus numeric MSE. At the `[VAL]`
input position its projected scalar replaces the code embedding; time, age, and
segment embeddings are still added normally. Rows without numeric values are
unchanged.
