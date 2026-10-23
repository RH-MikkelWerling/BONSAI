"""Check that each router screen fit trained what its name says (CPU, read-only).

For every source, rank and query screen fit in the bundle, compares the bundle
config with the config the training process actually ran (Hydra's copy in the
run directory) and with the selected checkpoint's parameters:

* source mode, rank (adapter bottleneck = second dimension of ``down``) and
  query mode/width (``query_interaction`` present, first layer 3h -> width);
* how far the interaction branch moved from its zero start (it begins as
  exactly the additive router: last layer initialised to zero);
* shared-slot logit, selected epoch, and whether two names share one run.

    python -m opera.run.audit_router_fits --bundle-dir <root>/bundles/stage1 \
        --runs-dir <root>/neural_runs
"""

import argparse
import json
from pathlib import Path

import yaml

from opera.run.paper_prediction_queue import bundle_runs

SCREENS = ("source_screen", "rank_screen", "query_screen")
FIELDS = ("adapter_source_mode", "adapter_bottleneck", "adapter_query_mode",
          "adapter_query_hidden_dim", "adapter_shared_logit_init", "adapter_home_floor",
          "adapter_residual_scale")


def readout_state(checkpoint):
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    prefix = next(k.split("source_keys")[0] for k in state if k.endswith("source_keys"))
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def audit(row, config, runs_dir):
    run = Path(runs_dir) / row["run_directory"]
    risk = config["competing_risk"]
    out = {"name": row["name"], "phase": row["phase"], "run_directory": row["run_directory"],
           "bundle": {f: risk.get(f) for f in FIELDS}}
    ran = run / ".hydra" / "config.yaml"
    if ran.is_file():
        used = yaml.safe_load(ran.read_text()).get("competing_risk", {})
        out["ran"] = {f: used.get(f) for f in FIELDS}
        out["config_matches"] = out["ran"] == out["bundle"]
    marker = run / "training_complete.json"
    if not marker.is_file():
        out["status"] = "no training_complete.json"
        return out
    selection = json.loads(marker.read_text()).get("selection", {})
    out["selected"] = {k: selection.get(k) for k in ("best_epoch", "best_score", "monitor")
                       if k in selection}
    state = readout_state(run / selection.get("best_checkpoint", "best.ckpt"))
    width = state["query_interaction.0.weight"].shape[0] if "query_interaction.0.weight" in state else None
    out["checkpoint"] = {
        "rank": int(state["down"].shape[1]),
        "query_mode": "interacting" if width else "additive",
        "query_width": width,
        "shared_logit": float(state["shared_logit"]),
    }
    if width:
        last = state["query_interaction.2.weight"]
        out["checkpoint"]["interaction_last_layer_norm"] = float(last.norm())
        out["checkpoint"]["interaction_last_bias_norm"] = float(state["query_interaction.2.bias"].norm())
        out["checkpoint"]["patient_query_norm"] = float(state["patient_query.weight"].norm())
    expected = (risk.get("adapter_bottleneck"), risk.get("adapter_query_mode"),
                risk.get("adapter_query_hidden_dim") if risk.get("adapter_query_mode") == "interacting" else None)
    found = (out["checkpoint"]["rank"], out["checkpoint"]["query_mode"], width)
    out["checkpoint_matches_name"] = expected == found
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, rows = bundle_runs(args.bundle_dir)
    results = [audit(row, config, args.runs_dir) for row, _, config in rows
               if row["phase"] in SCREENS and row["origin"] == "L"]
    shared = {}
    for r in results:
        shared.setdefault(r["run_directory"], []).append(r["name"])
    for r in results:
        r["same_run_as"] = [n for n in shared[r["run_directory"]] if n != r["name"]]
        c = r.get("checkpoint", {})
        print(f"{r['phase']:<14} {r['name']:<45} rank={c.get('rank')} "
              f"{c.get('query_mode')}{c.get('query_width') or ''} "
              f"match={r.get('checkpoint_matches_name')} cfg={r.get('config_matches')} "
              f"epoch={r.get('selected', {}).get('best_epoch')} "
              f"|interaction|={c.get('interaction_last_layer_norm', '-')}"
              + (f" same run as {r['same_run_as']}" if r["same_run_as"] else ""))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
