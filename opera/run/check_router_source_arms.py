"""Read-only audit of resolved source-arm configs and early training metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


ARMS = {
    "disease_only": "time2vec_router_disease_only_e2e_guarded_scale07_e2e",
    "outcome_only": "time2vec_router_outcome_only_e2e_guarded_scale07_e2e",
    "none": "time2vec_router_no_sources_e2e_guarded_scale07_e2e",
}
KEYS = (
    "val/loss", "val/cr/route_shared_weight", "val/cr/effective_residual_magnitude",
    "val/endpoint_group_auc", "val/cr_cif_auc_endpoint_group_macro",
    "train/loss", "train/cr/route_shared_weight",
    "train/cr/effective_residual_magnitude",
)


def latest_metrics(path: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in KEYS:
                for actual in (key, key + "_epoch", key + "_step"):
                    if row.get(actual) not in (None, ""):
                        result[actual] = (row.get("epoch", "?"), row[actual])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    for arm, name in ARMS.items():
        run = args.run_root / name
        config_path = run / "hydra" / "config.yaml"
        print(f"\n{arm}: {run}")
        if not config_path.is_file():
            print("  MISSING resolved config")
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        risk = config["competing_risk"]
        for key in ("adapter_source_mode", "readout_mode", "adapter_residual_scale",
                    "adapter_bottleneck", "adapter_home_floor", "adapter_event_tau"):
            print(f"  {key}: {risk.get(key, '<default>')}")
        print(f"  dapt_ckpt: {config.get('dapt_ckpt')}")
        files = sorted(run.glob("contrastive_multicohort_runs/version_*/metrics.csv"))
        if not files:
            print("  MISSING metrics.csv")
            continue
        path = files[-1]
        print(f"  metrics_file: {path}")
        for key, (epoch, value) in latest_metrics(path).items():
            print(f"  {key} [epoch {epoch}]: {value}")


if __name__ == "__main__":
    main()
