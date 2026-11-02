"""One table of every relevant fit: status, validation macro AUC per epoch
(x1000, * = the checkpoint selected by the locked loss rule), AUC at that
checkpoint and its difference from R_AB and from J_OA_OWN. Training logs
only; aggregates only.

    .venv-v100/bin/python scripts/monitor_router_runs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GROUPS = {
    "Reference": ["J_all_diseases_all_outcomes", "R_AB", "R_AB_s43", "R_PF_OFF"],
    "Stage 9: reading + own corrections": ["J_OA_OWN", "J_OA_OWN_PH", "J_OA_LR_OWN", "J_FPF_PH_OWN_PH", "J_FPF_OWN"],
    "Stage 12: what the design is made of": ["J_OA_OWN_s43", "J_OA_OWN_R32", "J_OA_OWN_NODIS", "J_OA_OWN_NOOUT",
                                             "J_OA_OWN_F10", "R_PF_OFF_F10", "R_PH_GE_F10"],
    "Stage 13: design changes": ["J_OA_OWN_SW20", "J_OA_OWN_SW100", "J_OA_OWN_X", "J_OA_OWN_X_SW", "J_OA_OWN_PE",
                                 "J_OA_OWN_PH_X", "J_OA_OWN_PE_X_SW", "J_OA_OWN_MAX"],
    "Stage 14-15: seed spread": ["J_OA_OWN_s44", "J_OA_OWN_s45", "J_s43", "R_PF_OFF_s43",
                                 "J_OA_OWN_PH_s43", "J_OA_OWN_PH_s44"],
    "Encoder rate (sensitivity)": ["J_ELR10", "R_AB_ELR10", "R_AB_ELR3", "J_OA_OWN_ELR10", "J_FPF_PH_OWN_PH_ELR10"],
}
SELECTION = "val/cr_horizon_bce_endpoint_group_macro"
AUC = "val/cr_cif_auc_macro"


def latest_runs(root: Path) -> dict:
    bundles = sorted((p for p in (root / "bundles").glob("stage*") if (p / "bundle.json").is_file()),
                     key=lambda p: int(p.name[5:]))
    runs = {}
    for bundle in bundles:
        for row in json.loads((bundle / "bundle.json").read_text())["runs"]:
            if "/L/" in row["name"]:
                runs[row["name"].split("/")[-1].removeprefix("fix_")] = row
    return runs


def summarise(root: Path, row: dict) -> dict:
    run = root / "neural_runs" / row["run_directory"]
    claim = run / ".worker_claim.json"
    if (run / "training_complete.json").is_file():
        status = "done"
    elif (run / "worker_failed.json").is_file() and not claim.is_file():
        status = "FAILED"
    elif claim.is_file():
        host = str(json.loads(claim.read_text()).get("host", "?")).split(".")[0]
        status = f"run {host[-5:]}"
    else:
        status = "wait"
    out = {"status": status, "per_epoch": {}, "selected": None, "auc": None, "best": None}
    logs = sorted(run.rglob("metrics.csv"))
    if not logs:
        return out
    metrics = pd.concat([pd.read_csv(path) for path in logs], ignore_index=True)
    if "epoch" not in metrics:
        return out
    # Logs read the epoch column as floats when some rows lack it.
    metrics = metrics.dropna(subset=["epoch"]).astype({"epoch": int})
    if AUC in metrics:
        out["per_epoch"] = metrics.dropna(subset=[AUC]).groupby("epoch")[AUC].last().to_dict()
        out["best"] = max(out["per_epoch"].values()) if out["per_epoch"] else None
    if SELECTION in metrics and metrics[SELECTION].notna().any():
        loss = metrics.dropna(subset=[SELECTION]).groupby("epoch")[SELECTION].last()
        out["selected"] = int(loss.idxmin())
        out["auc"] = out["per_epoch"].get(out["selected"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/paper_256_v3"))
    args = parser.parse_args()
    runs = latest_runs(args.root)
    rows = {name: summarise(args.root, runs[name]) for group in GROUPS.values() for name in group if name in runs}
    ref = rows.get("R_AB", {}).get("auc")
    base = rows.get("J_OA_OWN", {}).get("auc")
    epochs = max((max(r["per_epoch"]) + 1 for r in rows.values() if r["per_epoch"]), default=0)
    header = f"{'model':24s} {'status':11s} " + "".join(f"{e:>4d} " for e in range(epochs))
    print("validation macro AUC x1000 per epoch, * = selected checkpoint (lowest tuning loss)\n")
    print(header + "  AUC@sel   vs R_AB  vs OA_OWN    best")
    for group, names in GROUPS.items():
        present = [n for n in names if n in rows]
        if not present:
            continue
        print(f"-- {group}")
        for name in present:
            r = rows[name]
            cells = "".join(
                (f"{r['per_epoch'][e] * 1000:4.0f}" + ("*" if e == r["selected"] else " ")) if e in r["per_epoch"]
                else "   . " for e in range(epochs))
            tail = ""
            if r["auc"] is not None:
                tail = f"  {r['auc']:.4f}"
                tail += f"  {r['auc'] - ref:+.4f}" if ref is not None else "         "
                tail += f"  {r['auc'] - base:+.4f}" if base is not None and name != "J_OA_OWN" else "          "
                tail += f"  {r['best']:.4f}" if r["best"] is not None else ""
            label = name.replace("J_all_diseases_all_outcomes", "J")
            print(f"{label:24s} {r['status']:11s} {cells}{tail}")


if __name__ == "__main__":
    main()
