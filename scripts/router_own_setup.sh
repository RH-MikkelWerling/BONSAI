#!/usr/bin/env bash
# Run once (CPU), after router_fix_setup.sh: prepare the stage-4 bundle (stage 3
# plus the three own-outside router fits) and stop unless exactly those three
# are new. Then start GPU workers as usual: bash scripts/paper_gpu_worker.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/paper_256_v3
if [[ -d $D/bundles/stage4 ]]; then echo "stage4 exists; keeping it"
else bash scripts/run_paper_experiments.sh prepare --stage 4 --execute; fi
python - "$D" <<'PY'
import json, sys
root = sys.argv[1]
old = {r["run_directory"] for r in json.load(open(f"{root}/bundles/stage3/bundle.json"))["runs"]}
runs = json.load(open(f"{root}/bundles/stage4/bundle.json"))["runs"]
new = [r["name"] for r in runs if r["run_directory"] not in old]
print(len(new), "new:", *new, sep="\n  ")
if sorted(n.rsplit("fix_", 1)[-1] for n in new) != ["RBH_O", "R_O", "R_O_s43"]:
    sys.exit("Stage 4 is not stage 3 plus the three own-outside fits: do not start workers.")
PY
echo "Stage 4 ready: start GPU workers (bash scripts/paper_gpu_worker.sh)."
