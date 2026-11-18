#!/usr/bin/env bash
# User-launched lanes; never trains, commits, pushes or crosses allocations.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python_bin="${OPERA_PYTHON:-$PWD/.venv-v100/bin/python}"
lane="${1:-plan}"
root=data/paper_256_v3_national_20261111
comparison_root="${OPERA_COMPARISON_ROOT:-data/figure_closure_20261117}"
figure_root="${OPERA_FIGURE_ROOT:-$comparison_root}"
mkdir -p logs
case "$lane" in
  plan)
    cat <<'PLAN'
Run named lanes on the appropriate allocations (this plan starts no jobs):
  inventory       CPU: read-only source inventory
  cpu             CPU: recover, maps, story clinical/embeddings/no-index; >=24 cores
  interpretation  Free GPU: resume corrected IG, restrictions and figures
  arrows          Free GPU: correction arrows + completed local ablation evaluations
  comparisons     Dedicated >=96 GiB FREE RAM: core, ablations, pretraining controls
  diagnostics     Dedicated free GPU + RAM: correction replay then tuning diagnostics
  pooling         CPU: analysis 5, after diagnostics prepared snapshot exists
  fixed_budget    CPU: exploratory highest/lowest deciles from completed core snapshot
  table1          CPU/R: filled registry template then score/PS refresh
  mechanism       CPU: explicit matching tuning analysis/checkpoint/vectors required
  vocabulary      CPU: explicit original cohort records and national vocabulary required
  harvest         CPU: refresh recovered exports AFTER producers complete
  status          CPU: display receipts, not cross-allocation process monitoring
Before interpretation: preserve the failed workspace ONCE as described in the runbook.
Training and the 20-epoch arm's missing fits are not launched by any lane.
PLAN
    ;;
  inventory)
    "$python_bin" scripts/audit_results_request.py > logs/results_inventory_20261117.json
    "$python_bin" scripts/inventory_paper_controls.py \
      --output "logs/pretraining_controls_inventory_$(date +%Y%m%d_%H%M%S).json"
    ;;
  cpu)
    # Continue independent lanes after a failure, but retain a nonzero exit status.
    failed=0
    "$python_bin" -m opera.run.finish_figure_inputs --lane recover --execute --output-root "$figure_root" || failed=1
    "$python_bin" -m opera.run.finish_figure_inputs --lane maps --execute --output-root "$figure_root" || failed=1
    "$python_bin" -m opera.run.finish_story_inputs --clinical-workspace "$root/clinical_20261115/R_EA" \
      --flipi2-endpoint overall_survival --execute --redo embeddings clinical || failed=1
    exit "$failed"
    ;;
  interpretation)
    "$python_bin" -u -m opera.run.run_final_interpretation --execute
    ;;
  arrows)
    "$python_bin" -m opera.run.finish_figure_inputs --lane arrows --execute --output-root "$figure_root"
    ;;
  comparisons)
    failed=0
    for kind in core ablations pretraining_controls; do
      "$python_bin" -u -m opera.run.finish_figure_comparisons --kind "$kind" \
        --output-root "$comparison_root" --execute || failed=1
    done
    exit "$failed"
    ;;
  diagnostics)
    bash scripts/run_representation_diagnostics.sh --all-terms
    ;;
  pooling)
    bash scripts/run_pooling_absorption.sh \
      data/diagnostics/representation_absorption_20261112/tuning_all_terms.prepared_INTERNAL
    ;;
  fixed_budget)
    "$python_bin" scripts/check_results_resources.py --minimum-free-gib 16 --require-no-coordinator
    "$python_bin" -u -m opera.analysis.fixed_budget_risk \
      --checkpoint-dir "$comparison_root/comparisons_core/checkpoints" \
      --output-dir "${OPERA_FIXED_BUDGET_OUTPUT:-data/fixed_budget_risk_20261118/held_out}"
    ;;
  table1)
    # Edit the R source blocks first; these steps require fresh output directories.
    Rscript opera/R/prepare_table1_quality_registry_inputs_20261116.R
    "$python_bin" - "${OPERA_TABLE1_QUALITY_METADATA:-data/clinical_registry_refresh/20261111/quality_table1_20261116/quality_ps_INTERNAL.parquet}" <<'PY'
import json
import sys
from pathlib import Path
quality = Path(sys.argv[1])
manifest = quality.parent / 'quality_registry_manifest.json'
if not quality.is_file() or not manifest.is_file():
    raise SystemExit('Missing prepared PS input/manifest; match OPERA_TABLE1_QUALITY_METADATA to the R output directory')
if json.loads(manifest.read_text()).get('all_required_sources_loaded') is not True:
    raise SystemExit('Table 1 PS sources are not declared complete; finish the R source mappings before final refresh')
PY
    bash scripts/refresh_table1_today.sh "${OPERA_TABLE1_OUTPUT:-data/clinical_registry_refresh/20261111/table1_quality_refresh_20261117}"
    ;;
  mechanism)
    : "${OPERA_MECHANISM_ANALYSIS:?Set the matching tuning analysis root}"
    : "${OPERA_MECHANISM_MODEL:?Set the actual model alias in that inventory}"
    : "${OPERA_MECHANISM_CHECKPOINT:?Set the matching checkpoint}"
    : "${OPERA_MECHANISM_VECTORS:?Set matching tuning pooled vectors}"
    : "${OPERA_MECHANISM_OUTPUT:?Set a fresh audit output directory}"
    "$python_bin" -u -m opera.run.audit_frozen_router_mechanism \
      --analysis-root "$OPERA_MECHANISM_ANALYSIS" --model "$OPERA_MECHANISM_MODEL" \
      --checkpoint "$OPERA_MECHANISM_CHECKPOINT" --vectors "$OPERA_MECHANISM_VECTORS" \
      --output-dir "$OPERA_MECHANISM_OUTPUT" --bootstrap 400 --threads 8
    ;;
  vocabulary)
    : "${OPERA_COHORT_PRETRAIN_RECORDS:?Set original cohort MEDS data directory}"
    : "${OPERA_NATIONAL_VOCABULARY:?Set frozen national vocabulary}"
    : "${OPERA_VOCAB_PREFLIGHT_OUTPUT:?Set a fresh preflight output directory}"
    : "${OPERA_VOCAB_TIMING_PLAN:?Set measured-log-based pretraining/joint/OPERA timing JSON}"
    # For a required adaptive mapping/static-event policy use the documented module directly.
    "$python_bin" -u -m opera.diagnostics.vocabulary_population_preflight \
      --records "$OPERA_COHORT_PRETRAIN_RECORDS" --national-vocabulary "$OPERA_NATIONAL_VOCABULARY" \
      --timing-plan "$OPERA_VOCAB_TIMING_PLAN" --output-dir "$OPERA_VOCAB_PREFLIGHT_OUTPUT"
    ;;
  harvest)
    "$python_bin" -m opera.run.finish_figure_inputs --lane recover --execute --refresh-harvest \
      --output-root "$comparison_root"
    ;;
  status)
    "$python_bin" -m opera.run.finish_figure_inputs --status
    "$python_bin" -m opera.run.finish_figure_inputs --status --output-root "$figure_root"
    "$python_bin" - "$comparison_root" <<'PY'
import json
import sys
from pathlib import Path
paths = [Path('data/story_closure_20261116/story_status.json'),
         Path('data/paper_256_v3_national_20261111/final_interpretation_20261116/interpretation_manifest.json')]
paths += list(Path(sys.argv[1]).glob('comparisons_*/preparation_report.json'))
paths += list(Path(sys.argv[1]).glob('comparisons_*/blocked.json'))
for path in paths:
    print('\nRECEIPT', path)
    print(json.dumps(json.loads(path.read_text()), indent=2) if path.is_file() else 'MISSING')
PY
    ;;
  *) echo "Unknown lane: $lane; run with plan" >&2; exit 2 ;;
esac
