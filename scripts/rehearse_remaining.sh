#!/usr/bin/env bash
# Rehearse on TUNING data the steps the first rehearsal did not run, so the
# held-out run meets no surprises: clinical scores, attributions (small
# sample), attribution summary, input restrictions, attribution figures,
# embedding tables, source removal and the borrowing map; then plot tables and
# figures again. Nothing here reads held-out data.
#
# Stand-ins for what is only fixed later: the attributed model is the current
# primary routed fit (R_L, stage 3); embeddings use J_L's tuning export only
# (fit on half the tuning patients, shown on the other half).
#
# On a GPU node, from the repo root:
#   nohup bash scripts/rehearse_remaining.sh > logs/rehearsal/remaining.log 2>&1 &
# Each step logs to logs/rehearsal/<step>.log; a failed step is reported and
# the next one still runs. Summary at the end of remaining.log.
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/paper_256_v3
R=$D/analysis/tuning_rehearsal
L=logs/rehearsal
mkdir -p "$L" "$R/rehearsal_configs"
ENV=$R/rehearsal_configs/env.remaining

echo "== resolve fits and write rehearsal configs $(date +%H:%M)"
python - "$D" "$R" .env.rehearsal "$ENV" <<'EOF' || exit 1
import shutil, sys
from pathlib import Path

import yaml
from dotenv import dotenv_values

from opera.run.paper_prediction_queue import bundle_runs, canonical_models

root, rehearsal, base_env, env_out = map(Path, sys.argv[1:])
bundle, rows = bundle_runs(root / "bundles" / "stage3")
fits = {}
for row, path, config in rows:
    for name in canonical_models(row, config, bundle):
        fits.setdefault(name, (row, path))
for name in ("R_L", "J_L"):
    if name not in fits:
        sys.exit(f"No {name} fit in the stage-3 bundle")
r_row, r_path = fits["R_L"]
j_row, _ = fits["J_L"]
predictions = (root / "predictions").resolve()
j_tuning = predictions / j_row["run_directory"] / "tuning"
for needed in (j_tuning / "patient_embeddings.npz", j_tuning / "direct_cr_predictions.parquet",
               root / "neural_runs" / r_row["run_directory"] / "best.ckpt"):
    if not needed.is_file():
        sys.exit(f"Missing {needed}")

configs = rehearsal / "rehearsal_configs"
attribution = yaml.safe_load(Path("opera/configs/attribution_analysis.example.yaml").read_text())
attribution["run"].update({"split": "tuning", "max_patients": 200, "per_subtype": 5,
                           "faithfulness_patients": 40, "sanity_patients": 20})
# Exercise the concept and lab-curve code paths too (rehearsal only; the real
# run stays unreviewed until the mapping has been checked).
attribution["concept_mapping_reviewed"] = True
(configs / "attribution.yaml").write_text(yaml.safe_dump(attribution, sort_keys=False))

embedding = yaml.safe_load(Path("opera/configs/paper_embedding_tables.example.yaml").read_text())
embedding["stages"] = {"Local fine-tuned": [str(j_tuning / "patient_embeddings.npz")]}
embedding["calibration_maps"]["stage"] = "Local fine-tuned"
embedding["calibration_maps"]["predictions"] = str(j_tuning / "direct_cr_predictions.parquet")
embedding.pop("gallery", None)
(configs / "embedding.yaml").write_text(yaml.safe_dump(embedding, sort_keys=False))

# The rehearsal env plus the stand-ins; later lines win.
shutil.copy(base_env, env_out)
with open(env_out, "a") as out:
    out.write("\n# rehearse_remaining.sh stand-ins\n")
    out.write(f"OPERA_PAPER_ATTRIBUTION_CHECKPOINT={(root / 'neural_runs' / r_row['run_directory'] / 'best.ckpt').resolve()}\n")
    out.write(f"OPERA_PAPER_ATTRIBUTION_CONFIG_DIR={r_path.parent}\n")
    out.write(f"OPERA_PAPER_ATTRIBUTION_CONFIG_NAME={r_path.stem}\n")
    out.write(f"OPERA_PAPER_ATTRIBUTION_CONFIG={(configs / 'attribution.yaml').resolve()}\n")
    out.write(f"OPERA_PAPER_EMBEDDING_CONFIG={(configs / 'embedding.yaml').resolve()}\n")
    out.write("OPERA_PAPER_PLOT_SPLIT=tuning\n")
    out.write("OPERA_PAPER_REMOVAL_REPLICATES=200\n")
    out.write("OPERA_PAPER_FIGURES_AVAILABLE_ONLY=1\n")
    if not dotenv_values(base_env).get("OPERA_PAPER_CLINICAL_SCORE_CONFIG"):
        example = Path("opera/configs/clinical_score_panels.example.yaml").resolve()
        out.write(f"OPERA_PAPER_CLINICAL_SCORE_CONFIG={example}\n")
print(f"  attributed: R_L = {r_row['name']}\n  embeddings: J_L = {j_row['name']}")
EOF

failed=()
step() {  # step <log name> <action and flags...>
  local name=$1; shift
  echo "== $name $(date +%H:%M)"
  if bash scripts/run_paper_experiments.sh "$@" --env-file "$ENV" > "$L/$name.log" 2>&1; then
    echo "   ok"
  else
    echo "   FAILED (exit $?):"; grep -an "Error\|error:" "$L/$name.log" | tail -3 | sed 's/^/     /'
    failed+=("$name")
  fi
}

step clinical-scores clinical-scores --execute
step attributions attributions --execute
step attribution-summary attribution-summary --execute
step input-restrictions input-restrictions --execute
step attribution-figures attribution-figures --execute
step embedding-tables embedding-tables --execute
step source-removal source-removal --split tuning --execute
step borrowing-map borrowing-map --split tuning --execute
step plot-tables plot-tables --execute
step figure-readiness figure-readiness   # lists what is still missing; failure expected
[[ -d $R/figures ]] && mv "$R/figures" "$R/figures_$(date +%m%d_%H%M)"
step figures figures --execute

echo "== done $(date +%H:%M); failed: ${failed[*]:-none}"
