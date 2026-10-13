"""Inspect saved adapter norms and Adam moments without loading patient data."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import yaml

from opera.run.check_router_source_arms import ARMS


def adam_adaptive_update(state: dict, group: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive update implied by saved moments/LR, excluding weight decay."""
    step = float(state["step"])
    if step <= 0:
        raise ValueError("Optimizer state has no completed step")
    beta1, beta2 = group["betas"]
    mean = state["exp_avg"].float() / (1 - beta1 ** step)
    variance = state.get("max_exp_avg_sq", state["exp_avg_sq"]).float() / (1 - beta2 ** step)
    rms = variance.sqrt()
    update = float(group["lr"]) * mean / (rms + float(group["eps"]))
    epsilon_factor = rms / (rms + float(group["eps"]))
    return update, epsilon_factor


def inspect_run(run: Path, checkpoint_name: str) -> dict:
    config = yaml.safe_load((run / "hydra" / "config.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / f"{checkpoint_name}.ckpt", map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    risk = config["competing_risk"]
    fine_names = sorted(risk["fine_names"])
    outcomes = sorted(config["outcomes"])
    n_disease = len(fine_names)
    mode = risk.get("adapter_source_mode", "both")
    result = {"run": str(run), "checkpoint_epoch": checkpoint.get("epoch"),
              "source_mode": mode,
              "update_note": "Adaptive update implied by saved Adam moments and saved learning rate; excludes weight decay"}
    optimizers = checkpoint.get("optimizer_states", [])
    optimizer = optimizers[0] if optimizers else None
    for bank, start, end, enabled in (
        ("disease", 0, n_disease, mode in {"both", "disease_only"}),
        ("outcome", n_disease, n_disease + len(outcomes), mode in {"both", "outcome_only"}),
    ):
        report = {"enabled": enabled, "n_sources": end - start}
        for name in ("down", "up"):
            matches = [value for key, value in state.items()
                       if key.endswith(f"source_adapter_readout.{name}")]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely locate source {name} parameter")
            parameter = matches[0].float()
            if parameter.shape[0] != n_disease + len(outcomes):
                raise ValueError("Saved source parameter shape disagrees with config inventory")
            values = parameter[start:end]
            report[f"{name}_parameter_rms"] = float(values.square().mean().sqrt())
            if optimizer is None:
                report[f"{name}_optimizer_state"] = "absent"
                continue
            matches = [(index, saved) for index, saved in optimizer["state"].items()
                       if "exp_avg" in saved and tuple(saved["exp_avg"].shape) == tuple(parameter.shape)]
            if len(matches) > 1:
                raise ValueError("Ambiguous optimizer-state shape: cannot assign to source parameter")
            if not matches:
                report[f"{name}_optimizer_state"] = "absent"
                continue
            index, saved = matches[0]
            groups = [group for group in optimizer["param_groups"] if index in group["params"]]
            if len(groups) != 1:
                raise ValueError("Cannot resolve source optimizer parameter group")
            update, epsilon_factor = adam_adaptive_update(saved, groups[0])
            report[f"{name}_optimizer_epsilon"] = float(groups[0]['eps'])
            report[f"{name}_implied_adaptive_update_rms"] = float(update[start:end].square().mean().sqrt())
            report[f"{name}_mean_adam_epsilon_factor"] = float(epsilon_factor[start:end].mean())
            report[f"{name}_optimizer_step"] = float(saved["step"])
        result[bank] = report
    keys = [value for key, value in state.items() if key.endswith("competing_risk_head.weight")]
    if len(keys) == 1:
        heads = keys[0].float().reshape(len(outcomes), 2, -1, keys[0].shape[-1])[:, 0].flatten(1)
        related = []
        for group, members in config["cross_outcome"].get("endpoint_groups", {}).items():
            for left, right in itertools.combinations(members, 2):
                if left not in outcomes or right not in outcomes:
                    continue
                a, b = heads[outcomes.index(left)], heads[outcomes.index(right)]
                related.append({"endpoint_group": group, "left": left, "right": right,
                                "primary_head_weight_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0))})
        result["related_grade_head_alignment"] = related
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    reports = {arm: inspect_run(args.run_root / name, args.checkpoint) for arm, name in ARMS.items()}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "optimizer_audit.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    compact = {arm: {key: value for key, value in report.items() if key != "related_grade_head_alignment"}
               for arm, report in reports.items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
