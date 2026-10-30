# Partial-pooling router: training, tuning analysis and audits

30 October 2026. Design, health criteria and selection rule: the dated R_PP
amendment in `ANALYSIS_LOCK_DRAFT_20261015.md`. Tuning data only. Nothing here
opens test predictions.

## Files to sync

New: `opera/modules/networks/partial_pooling_readout.py`,
`opera/diagnostics/router_positive_control.py`.
Changed: `opera/modules/networks/opera_nets.py`,
`opera/functional/direct_cr_contract.py`, `opera/run/contrastive_multicohort.py`,
`opera/diagnostics/router_mechanism.py`, `opera/run/audit_frozen_router_mechanism.py`,
`opera/run/prepare_paper_run_bundle.py`, `opera/run/paper_experiments.py`,
`opera/run/paper_prediction_queue.py`, `opera/run/analyse_locked_predictions.py`,
`opera/run/build_paper_numbers.py`, `opera/run/source_removal.py`.

## 1. Train (GPU, overnight)

```bash
cd /ngc/people/mikwer_r/projects/BONSAI-opera-leukemia_novelty/BONSAI-opera-leukemia
source .venv-v100/bin/activate
bash scripts/run_paper_experiments.sh prepare --stage 6 --execute
bash scripts/run_paper_experiments.sh queue-status --execute
```

Expect exactly three ready fits: `fix_R_PP`, `fix_R_PP_s43`, `fix_R_PP_FROZEN`.
Then one worker per GPU allocation:

```bash
mkdir -p logs
nohup bash scripts/run_paper_experiments.sh worker --worker-name gpu1 --max-runs 1 --execute \
  > logs/worker_gpu1_rpp.log 2>&1 &
```

(`gpu2`, `gpu3` on the other allocations.) Each training run writes
`cell_support.csv` (training events per disease x outcome) next to its source
support table.

## 2. Tuning predictions (GPU, about 20 minutes per fit)

```bash
bash scripts/run_paper_experiments.sh evaluate --stage 6 --split tuning \
  --fit-name fix_R_PP --fit-name fix_R_PP_s43 --fit-name fix_R_PP_FROZEN --execute
```

Writes predictions, patient vectors and routes for the three fits.

## 3. Tuning analysis in a fresh workspace (CPU, about 1.5 hours)

Values in `.env` override exported variables, so use a copy of the env file
pointing at a new analysis root. The existing rehearsal workspace is untouched.

```bash
opera_rpp_root="$PWD/data/paper_256_v3/analysis/rpp_tuning_$(date +%Y%m%d_%H%M%S)"
grep -v -E '^(OPERA_PAPER_ANALYSIS_ROOT|OPERA_PAPER_INVENTORY_SPLITS|OPERA_PAPER_ROUTED_MODEL)=' .env > .env.rpp
cat >> .env.rpp <<EOF
OPERA_PAPER_ANALYSIS_ROOT=$opera_rpp_root
OPERA_PAPER_INVENTORY_SPLITS=tuning
OPERA_PAPER_ROUTED_MODEL=R_AB_L
EOF
bash scripts/run_paper_experiments.sh inventory --stage 6 --env-file .env.rpp --execute
nohup env OPERA_ANALYSIS_WORKERS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash scripts/run_paper_experiments.sh analyse --env-file .env.rpp --execute \
  > logs/rpp_analysis.log 2>&1 &
```

R_AB stays the routed model (R_L) here, so the contrasts below compare against it:

| Code | Contrast | Question |
|---|---|---|
| PP1 | R_PP vs R_L (R_AB) | Selection rule |
| PP2 | R_PP vs R_PP_FROZEN | Gain from learned routing |
| PP3 | R_PP_FROZEN vs R_L (R_AB) | Gain from the structural changes |
| PP4 | R_PP vs J_L | Against the joint model |
| PP0 | R_PP vs R_PP_s43 | Seed noise floor |

The analysis needs a completion record for the mechanism audit only when it
audits the declared OPERA model. With `--model` it does not (step 4).

## 4. Mechanism audits (CPU, about 20 minutes each)

The same audit as for R_AB, now with the two-sided health criteria, and for
R_PP with tau, temperature, the own-weight curve and the R_PP ladder
(J -> R_AB -> R_PP_FROZEN -> R_PP). R_AB is rerun for the new health criteria.

```bash
for model in R_PP_L R_AB_L; do
  nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv-v100/bin/python -u -m opera.run.audit_frozen_router_mechanism \
    --analysis-root "$opera_rpp_root" --model $model \
    --output-dir "$opera_rpp_root/../rpp_mechanism_${model}" \
    --bootstrap 400 --threads 4 > "logs/rpp_mechanism_${model}.log" 2>&1 &
done
nohup env OMP_NUM_THREADS=1 .venv-v100/bin/python -u -m opera.run.audit_frozen_router_mechanism \
  --analysis-root "$opera_rpp_root" --model R_PP_s43_L --seed 43 \
  --output-dir "$opera_rpp_root/../rpp_mechanism_R_PP_s43_L" \
  --bootstrap 400 --threads 4 > logs/rpp_mechanism_R_PP_s43_L.log 2>&1 &
```

The checkpoint is found from the standard `predictions/` to `neural_runs/` layout
and confirmed against the saved vectors' own record. Pass `--checkpoint` if a
fit lives elsewhere.

Each audit now also writes `borrowed_weights_by_task.csv` (mean gate of every
borrowed source for each disease x outcome task, and its share of the borrowed
weight) and `borrowed_weights_by_disease.csv`, and reports the top three
sources per disease and the two-sided entropy/health summary in `report.json`.

## 4b. Borrowing map for R_PP (what the predictions depend on)

The gate tables say where the weight goes. Source removal says what the
predictions depend on: each other disease's correction replaced by its tuning
mean (one GPU pass, then CPU). Same procedure as for R_AB, with R_PP as the
routed model, in its own env copy:

```bash
grep -v -E '^(OPERA_PAPER_ANALYSIS_ROOT|OPERA_PAPER_ROUTED_MODEL)=' .env > .env.rpp_map
cat >> .env.rpp_map <<EOF
OPERA_PAPER_ANALYSIS_ROOT=$opera_rpp_root
OPERA_PAPER_ROUTED_MODEL=R_PP_L
EOF
bash scripts/run_paper_experiments.sh source-removal --split tuning --env-file .env.rpp_map --execute   # GPU
nohup bash scripts/run_paper_experiments.sh borrowing-map --split tuning --env-file .env.rpp_map --execute \
  > logs/rpp_borrowing_map.log 2>&1 &                                                                # CPU
```

The export uses the stage-6 bundle holding R_PP. R_PP has no own-only fit, so
the map reports dependence and per-source in/out effects without a
"trained borrowing" comparison; learned routing is PP2. The prespecified
borrowing pairs (SLL from CLL, CLL and myeloma for infection, the negative
controls) are evaluated as before, in `borrowing_hypotheses.csv`.

## 5. What to bring back (aggregates only)

```bash
.venv-v100/bin/python - "$opera_rpp_root/prediction_analysis" <<'PY'
import sys, pathlib, pandas as pd
root = pathlib.Path(sys.argv[1])
for name in ("effect_summaries.csv", "effect_summaries_disease_balanced.csv"):
    table = pd.read_csv(root / name)
    rows = table.loc[table.contrast.astype(str).str.startswith("PP")]
    print(f"\n== {name}\n" + rows.to_string(index=False))
PY
```

Plus, from each `rpp_mechanism_*` directory: `router_mechanism_review.zip`
(report, ladder, borrowed-weight tables, figures). From
`$opera_rpp_root/source_removal/tuning/`: `borrowing_map.csv`,
`borrowing_hypotheses.csv`, `borrowing_levels.csv` and the map figure.

## Reading the result (from the lock)

R_PP replaces R_AB only if its borrowed mixture passes both health criteria
(`borrowed_mixture_health`: `not_collapsed` and `not_uniform`) and PP1 is at
least minus the noise floor |PP0|. Whatever the selection, report tau, the
own-weight curve, PP2 (routing) and PP3 (structure), and the task-mean-gates
intervention. R_PP is the last router design before the test set.
