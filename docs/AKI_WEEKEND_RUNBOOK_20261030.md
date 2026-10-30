# AKI correction: weekend runbook

30 October 2026. Goal: corrected AKI labels, affected refits running over the
weekend, and the national downstream fits trained on the same corrected labels,
so that the national comparison is ready on Monday. Background and tooling:
[impact audit](AKI_BASELINE_BUG_IMPACT_20261030.md),
[recovery plan](AKI_CORRECTION_RECOVERY_PLAN_20261030.md),
[release and runner handoff](AKI_RELEASE_AND_RUNNER_HANDOFF_20261030.md).
Held-out results stay unopened throughout.

Decided before any corrected label is inspected: **every model whose training
or selection used the AKI target is refitted, however small the label change.**

## Critical path

| When | Step | Blocks |
|---|---|---|
| Now | 0. Sync files. 1. XGBoost reliance diagnostic | Nothing (record only) |
| Now | 2. Regenerate the AKI table: original must reproduce, then corrected | Everything below |
| As soon as 2 is done | Rebuild the tabular feature matrices with the corrected table (tabular repo), then refit every tabular model on CPU | Tabular comparisons |
| Fri night / Sat | 3. Audit, 4. corrected release, 5. refit bundle | Training |
| Sat | 6. Refits on all free GPUs; start with J, OPERA (R_AB) and both controls | E1--E4, headline |
| As soon as the national encoder (and DAPT) is ready | National arms from the same repair bundle (steps 5--7) | National results |
| Sat--Sun | Wave 2: outcome-pooled (O), AKI specialists/D, retained sensitivities. 8. AKI tabular refits on CPU | Full factorial |
| Sun--Mon | 9. Tuning predictions, inventory, analysis | Monday review |

Approximate GPU load (5--6 h per full-cohort fit, one GPU each): about ten
full-cohort fits (J, R_AB, RBH_AB, N_A, J_IPW, R_AB_IPW, four national) and
three data-efficiency fractions, plus up to 24 O and 25 AKI S/D fits on single
diseases. With eight GPUs, wave 1 takes about 6 h, and everything fits in the
weekend if wave 1 starts by Saturday morning. Disease-level fits are much
shorter than full-cohort fits.

## 0. Sync to the server (no git there)

New: `opera/R/regenerate_aki_table.R`, `opera/run/audit_aki_correction.py`, `opera/run/build_aki_corrected_release.py`,
`opera/run/prepare_aki_repair_bundle.py`, `opera/run/prepare_aki_tabular_overlay.py`.
Also updated today: `opera/configs/manifests/final_contrasts_20261014.yaml` and
`opera/run/build_paper_numbers.py` (national vs XGBoost contrasts).

## 1. AKI as a predictor: confirmed, so every tabular model is refitted

The tabular feature matrices read the same AKI table as their own data source
(`data_source = "AKI"`, `variable_code = "creatinine_increase"`), so the
`adverse_event` name filter did not remove it. Every tabular model (X, XP, LR,
XR) used it, for every outcome. With the faulty window, a pre-treatment onset
can depend on post-treatment creatinine, so this is **leakage into a
predictor**, not only label error. Decision: rebuild the feature matrices with
the corrected table and refit every tabular model for every outcome. The
sequence models are unaffected as predictors (raw creatinine only, no derived
AKI tokens in the MEDS conversion).

### Diagnostic: how much did XGBoost rely on it? (aggregates only, no refit)

This does not decide anything: the refit happens regardless. It records how
much the old XGBoost results leaned on the faulty feature, for the development
history and for interpreting old-versus-corrected tabular results.

```bash
.venv-v100/bin/python - "$PWD/data/paper_256_v3/tabular_runs" <<'PY'
import re, sys, pathlib
import pandas as pd
root = pathlib.Path(sys.argv[1])
pattern = re.compile(r"creatinine_increase|(^|[^a-z])aki([^a-z]|$)", re.I)
rows = []
for path in root.rglob("*xgboost*_feature_importance.csv"):
    table = pd.read_csv(path)
    if "gain_importance" not in table or table.gain_importance.sum() <= 0:
        continue
    table = table.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    hit = table.feature.astype(str).str.contains(pattern)
    parts = path.relative_to(root).parts
    rows.append(dict(fit="/".join(parts[:-1]), outcome=parts[-3] if len(parts) >= 3 else "",
                     aki_share=table.gain_importance[hit].sum() / table.gain_importance.sum(),
                     best_rank=int(hit.idxmax()) + 1 if hit.any() else None))
fits = pd.DataFrame(rows)
print(f"{len(fits)} XGBoost fits; AKI feature present in {fits.best_rank.notna().sum()}")
print(f"in top 10 features: {(fits.best_rank <= 10).sum()}, top 3: {(fits.best_rank <= 3).sum()}")
by_outcome = fits.groupby("outcome").agg(fits=("fit", "size"), median_share=("aki_share", "median"),
                                         max_share=("aki_share", "max"),
                                         top10=("best_rank", lambda r: (r <= 10).sum()))
print(by_outcome.sort_values("median_share", ascending=False).head(15).round(4).to_string())
print(fits.sort_values("aki_share", ascending=False).head(10).round(4).to_string(index=False))
PY
```

Adjust the root if the tabular runs live elsewhere (and add the
registry-enriched root). The output holds gain shares and ranks only.

## 2. Regenerate the AKI table ourselves

[`opera/R/regenerate_aki_table.R`](../opera/R/regenerate_aki_table.R) runs the
DALY-CARE core writer (`core_scripts/AE_AKI.R`, the `AKI_function_toNGC_v2.R`
lineage) as published or with only the annual-baseline window corrected. It
refuses to run if the published window line is not found exactly once.

First reproduce the existing table with the **original** writer. Only if the
script reports `identical TRUE` is the corrected table trusted. A mismatch
means a different laboratory extract, patient set or writer version: stop and
explain it before going further.

```bash
opera_aki_writer=/path/to/dalycare_package_v2/core_scripts/AE_AKI.R
opera_aki_reference="$OPERA_AKI_TABLE"     # the table used so far
opera_aki_regen="$PWD/data/results/aki_regeneration_$(date -u +%Y%m%dT%H%M%SZ)"

Rscript opera/R/regenerate_aki_table.R --writer "$opera_aki_writer" --version original \
  --output "$opera_aki_regen/original/AKI_table_all_dalycare_simplified.csv" \
  --reference "$opera_aki_reference"

Rscript opera/R/regenerate_aki_table.R --writer "$opera_aki_writer" --version corrected \
  --output "$opera_aki_regen/corrected/AKI_table_all_dalycare_simplified.csv" \
  --reference "$opera_aki_reference"
```

Add `--setup FILE.R` if the dalycare package is loaded by a script rather than
`library(dalycare)`. The two runs are independent and can run at the same time.
The corrected run's comparison prints how many onsets changed (aggregate only).
The corrected table then feeds both the outcome audit (`--corrected-episodes`)
and the tabular feature rebuild in the tabular repository.

## 3--4. Audit and corrected release

Run steps 1 and 2 of the [handoff](AKI_RELEASE_AND_RUNNER_HANDOFF_20261030.md)
exactly as written (`audit_aki_correction`, then `build_aki_corrected_release`).
The audit must first reproduce the original endpoint from the original episodes.
If it does not, stop. Keep `*_INTERNAL` outputs on the server.

## 5. Refit bundle (local and national together)

First see which frozen bundles hold which fits, including whether the national
arms (origins E and EA) were already prepared with `--include-external`.
Bundle metadata only:

```bash
.venv-v100/bin/python - <<'PY'
import json, collections, pathlib
for manifest in sorted(pathlib.Path("data/paper_256_v3/bundles").glob("*/bundle.json")):
    runs = json.loads(manifest.read_text())["runs"]
    origins = collections.Counter(r.get("origin") for r in runs)
    names = sorted({r.get("name", "") for r in runs if r.get("origin") in ("E", "EA", "external_direct", "external_plus_local_dapt")})
    print(manifest.parent.name, len(runs), dict(origins), names[:12])
PY
```

The `--models` roster is every reported fit that used AKI, local and national.
Repeat `--bundle-dir` for every frozen bundle that holds one of them. The tool
errors on a missing alias, which doubles as a roster check.

```bash
"$opera_python" -m opera.run.prepare_aki_repair_bundle \
  --bundle-dir "$PWD/data/paper_256_v3/bundles/stage5" \
  --release-dir "$opera_aki_release" \
  --output-dir "$opera_aki_bundle" \
  --models S_L D_L O_L J_L R_AB_L RBH_AB_L N_A_L J_IPW_L R_AB_IPW_L \
           J_L_F10 J_L_F25 J_L_F50 \
           J_E R_E J_EA R_EA J_EA_F10 J_EA_F25 J_EA_F50
```

Drop `J_IPW_L R_AB_IPW_L` or the data-efficiency fractions only if those
analyses leave the paper, and record that before training. If the national rows
are not in any frozen bundle, stop and prepare them deliberately (see step 7);
do not drop them from the roster silently.

## 6. Refits

One worker per GPU allocation, each started on its own:

```bash
"$opera_python" -m opera.run.paper_worker next \
  --bundle-dir "$opera_aki_bundle/refit_bundle" \
  --runs-dir "$PWD/data/paper_256_v3/neural_runs" \
  --wait
```

National fits wait for their encoder checkpoints (`external_direct/encoder.ckpt`,
`external_dapt/version_0/best.ckpt`). Check that the worker parks them rather
than failing while the encoders finish.

## 7. National arms: never on the original release

Do **not** run `bash scripts/run_paper_experiments.sh prepare` for this. It
writes to `bundles/stage{N}` (overwriting a frozen bundle), and values in `.env`
override exported variables, so exporting `OPERA_RELEASE_DIR` has no effect.
If the national rows are missing from every frozen bundle, prepare them with
`opera.run.prepare_paper_run_bundle` directly, pointing `--release-registry` at
`$opera_aki_release/experiment_registry_primary_v2.yaml` and `--output-dir` at a
fresh directory. National and local fits must share the corrected AKI labels,
or E1--E4 and the national-versus-XGBoost contrasts compare different targets.

## 8. Tabular

Every tabular model, every outcome: rebuild the feature matrices with the
corrected AKI table in the tabular repository (into a new directory, keeping
the old matrices), then prepare them into a **fresh** directory by calling the
module directly. Do not use `run_paper_experiments.sh features`: it writes to
the existing `features/` directory, and preparation reuses matrices already
there, which could silently keep the leaky ones.

```bash
"$opera_python" -m opera.run.prepare_tabular_features \
  --features /path/to/rebuilt_matrices/*.parquet \
  --population "$BONSAI_COHORT_MEMBERSHIP" \
  --output_dir "$PWD/data/paper_256_v3/features_aki_corrected" \
  --feature_profile sequence_matched registry_augmented
```

(add `--demographics` as in the original preparation if it was used). Then run `paper_tabular_queue` for all outcomes with `--features-dir` set to the
fresh directory, into a fresh output root (no `--outcome` filter), for both the `sequence_matched` and the
`registry_augmented` profiles. Point the analysis inventory at the new roots.
CPU only. Cap workers so GPU data loaders are not starved. Start as early as
possible: this is likely the longest single job.

## 9. Corrected tuning analysis

Step 5 of the handoff, with `--split tuning` / `--splits tuning` explicitly.
Source removal, router audits, IG and clinical thresholds run on the corrected
checkpoints only. Old-label outputs stay as development history, never mixed
with corrected-checkpoint results.
