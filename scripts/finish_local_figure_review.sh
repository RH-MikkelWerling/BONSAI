#!/usr/bin/env bash
# Run within a compute allocation. Reuse completed mean/zero source analyses.
set -euo pipefail
figure_root="${1:-data/paper_256_v3/analysis/local_figures_20261028_143127}"
python_bin="${OPERA_REVIEW_PYTHON:-.venv-v100/bin/python}"
for required in completion.json overnight.env source_removal/tuning/borrowing_map.csv source_removal/tuning/zero_ablation/borrowing_map.csv; do
  if [[ ! -f "$figure_root/$required" ]]; then
    printf 'Missing required completed input: %s/%s\n' "$figure_root" "$required" >&2
    exit 1
  fi
done
review_stamp="$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)"
review_parent="$(dirname "$figure_root")"
clinical_root="$review_parent/clinical_training_groups_$review_stamp"
ig_root="$review_parent/ig_refresh_$review_stamp"
review_root="$review_parent/paper_figure_review_$review_stamp"
mkdir -p logs
status_file="logs/finish_figure_review_${review_stamp}.tsv"
printf 'stage\texit_code\n' > "$status_file"
printf 'Clinical workspace: %s\nIG workspace: %s\nReview directory: %s\nStatus file: %s\n' "$clinical_root" "$ig_root" "$review_root" "$status_file"
# Save recovery commands before starting expensive inference.
"$python_bin" - "$figure_root" "$clinical_root" "$ig_root" "$review_root" "$status_file.paths.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[5]).write_text(json.dumps(dict(zip(
    ['analysis_root','clinical_root','ig_root','review_root'],sys.argv[1:5])),indent=2))
PY
run_stage() {
  local name="$1"
  shift
  local result=0
  printf 'Starting %s\n' "$name"
  "$@" > >(tee "logs/${name}_${review_stamp}.log") 2>&1 || result=$?
  printf '%s\t%s\n' "$name" "$result" >> "$status_file"
  printf 'Finished %s (exit %s)\n' "$name" "$result"
}
run_stage clinical "$python_bin" -m opera.run.complete_local_figures \
  --analysis-root "$figure_root" --output-dir "$clinical_root" --clinical-only
run_stage integrated_gradients "$python_bin" -m opera.run.prepare_attribution_refresh \
  --analysis-root "$figure_root" --output-dir "$ig_root" --execute
# Even an interrupted/failed IG summary can leave usable completed patient exports.
# Individual stage failures remain recorded and do not hide independent outputs.
if [[ -f "$ig_root/overnight.env" ]]; then
  run_stage input_restrictions "$python_bin" -m opera.run.paper_experiments input-restrictions \
    --env-file "$ig_root/overnight.env" --execute
fi
run_stage assemble "$python_bin" -m opera.run.assemble_figure_review \
  --analysis-root "$figure_root" --clinical-dir "$clinical_root/clinical_scores" \
  --attribution-dir "$ig_root/attributions" --restriction-dir "$ig_root/input_restrictions" \
  --output-dir "$review_root"
printf 'Review directory: %s\nStage results: %s\n' "$review_root" "$status_file"
printf 'Inspect review_manifest.json and score_inventory.csv before transferring figures.\n'
