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


def optimizer_check(checkpoint):
    """Learning rate and epsilon each router tensor trained with, against its gradient scale.

    Adam's step is m / (sqrt(v) + eps): where sqrt(v) (the running gradient
    size) is far below eps, eps damps the step. The optimizer state stores no
    names, so the router tensors are found by registration order and shape:
    source_keys (d + o, h), shared_logit (), patient_query (h, h), fine_query
    (d, h), target_query (o, h), then the interaction layers if present.
    """
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not saved.get("optimizer_states"):
        return {"note": "checkpoint has no optimizer state"}
    optimizer = saved["optimizer_states"][0]
    state = saved.get("state_dict", saved)
    d, h = next(v.shape for k, v in state.items() if k.endswith("fine_query.weight"))
    o = next(v.shape[0] for k, v in state.items() if k.endswith("target_query.weight"))
    width = next((v.shape[0] for k, v in state.items()
                  if k.endswith("query_interaction.0.weight")), None)
    names = ["source_keys", "shared_logit", "patient_query", "disease_query", "outcome_query"]
    if width:
        names += ["interaction.0.weight", "interaction.0.bias",
                  "interaction.2.weight", "interaction.2.bias"]
    out = {}
    for group in optimizer["param_groups"]:
        ids = group["params"]
        shapes = [tuple(optimizer["state"][i]["exp_avg_sq"].shape) if i in optimizer["state"]
                  else None for i in ids]
        start = next((j for j in range(len(shapes) - 4)
                      if shapes[j:j + 5] == [(d + o, h), (), (h, h), (d, h), (o, h)]), None)
        if start is None:
            continue
        for k, name in enumerate(names):
            entry = optimizer["state"][ids[start + k]]
            step = float(entry.get("step", 1))
            v_hat = entry["exp_avg_sq"] / (1 - 0.999 ** max(step, 1))
            root = v_hat.sqrt().flatten()
            out[name] = {"lr": group["lr"], "eps": group["eps"],
                         "weight_decay": group.get("weight_decay"),
                         "median_sqrt_v": float(root.median()),
                         "p10_sqrt_v": float(root.quantile(0.1)) if root.numel() > 1 else float(root[0]),
                         "share_below_eps": float((root < group["eps"]).float().mean()),
                         # Candidate epsilons: the share of entries each would damp.
                         **{f"share_below_{c:.0e}": float((root < c).float().mean())
                            for c in (1e-8, 1e-10, 1e-12)}}
        break
    return out or {"note": "router tensors not found in the optimizer state"}


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
        # Routing sharpness: scores are query . key / sqrt(hidden); keys start at std 0.02
        # and the disease/outcome queries at zero, so small norms mean near-uniform routing.
        "source_keys_norm": float(state["source_keys"].norm()),
        "disease_query_norm": float(state["fine_query.weight"].norm()),
        "outcome_query_norm": float(state["target_query.weight"].norm()),
    }
    if width:
        last = state["query_interaction.2.weight"]
        out["checkpoint"]["interaction_last_layer_norm"] = float(last.norm())
        out["checkpoint"]["interaction_last_bias_norm"] = float(state["query_interaction.2.bias"].norm())
        out["checkpoint"]["patient_query_norm"] = float(state["patient_query.weight"].norm())
    out["optimizer"] = optimizer_check(run / selection.get("best_checkpoint", "best.ckpt"))
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
              f"|interaction|={c.get('interaction_last_layer_norm', '-')} "
              f"|keys|={c.get('source_keys_norm', float('nan')):.2f} "
              f"|disease q|={c.get('disease_query_norm', float('nan')):.2f} "
              f"|outcome q|={c.get('outcome_query_norm', float('nan')):.2f}"
              + (f" same run as {r['same_run_as']}" if r["same_run_as"] else ""))
        for name, o in r.get("optimizer", {}).items():
            if isinstance(o, dict):
                print(f"{'':<16}{name:<22} lr={o['lr']:.2e} eps={o['eps']:.0e} "
                      f"median sqrt(v)={o['median_sqrt_v']:.1e} p10={o['p10_sqrt_v']:.1e} "
                      f"below: eps {o['share_below_eps']:.0%}, 1e-8 {o['share_below_1e-08']:.0%}, "
                      f"1e-10 {o['share_below_1e-10']:.0%}, 1e-12 {o['share_below_1e-12']:.0%}")
            else:
                print(f"{'':<16}{o}")
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
