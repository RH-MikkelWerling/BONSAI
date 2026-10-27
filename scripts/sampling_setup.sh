#!/usr/bin/env bash
# Run once (CPU), after router_own_setup.sh: prepare the stage-5 bundle (stage 4
# plus J and R_AB with importance-weighted event sampling; lock addendum
# 2026-10-27) and stop unless exactly those two fits are new. It also reports the
# DAPT anchor weight of the existing fits, which the IPW fits set to zero.
# Then start fresh GPU workers (workers only see bundles that existed when they
# started): bash scripts/paper_gpu_worker.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/paper_256_v3
if [[ -d $D/bundles/stage5 ]]; then echo "stage5 exists; keeping it"
else bash scripts/run_paper_experiments.sh prepare --stage 5 --execute; fi
python - "$D" <<'PY'
import json, sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
old = {r["run_directory"] for r in json.load(open(root / "bundles/stage4/bundle.json"))["runs"]}
runs = json.load(open(root / "bundles/stage5/bundle.json"))["runs"]
new = [r for r in runs if r["run_directory"] not in old]
print(len(new), "new:", *[r["name"] for r in new], sep="\n  ")
if sorted(r["name"].rsplit("fix_", 1)[-1] for r in new) != ["J_IPW", "R_AB_IPW"]:
    sys.exit("Stage 5 is not stage 4 plus the two IPW sampling fits: do not start workers.")
reference = next(r for r in runs if r["name"].endswith("fix_R_AB"))
config = yaml.safe_load((root / "bundles/stage5" / reference["config_path"]).read_text())
anchor = config.get("model", {}).get("dapt_anchor_weight", 0.0)
print(f"DAPT anchor weight of R_AB: {anchor} (the IPW fits use 0.0)")
if float(anchor or 0) != 0:
    print("NOTE: the IPW fits then differ from R_AB and J in the anchor weight too; "
          "record this in the lock addendum.")
PY
echo "Stage 5 ready: start fresh GPU workers (bash scripts/paper_gpu_worker.sh)."
