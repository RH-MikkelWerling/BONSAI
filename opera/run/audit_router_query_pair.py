"""Audit matched additive/interacting OPERA router runs without retraining.

Reads trusted local checkpoints, resolved configs, Lightning logs, and optional
INTERNAL tuning predictions/routes. Writes aggregate diagnostics only. This is
not a held-out performance comparison or a proof of causal route contribution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
import yaml

from opera.run.compare_approved_abspos_v2 import MONITOR, _epochs, _metrics_paths
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout


QUERY_PATH = "competing_risk.adapter_query_mode"
ROUTE_PREFIX = "source_adapter_readout."
METRICS = (
    MONITOR, "val/loss", "val/cr/route_shared_weight",
    "val/cr/route_entropy", "val/cr/effective_residual_magnitude",
)


def _config(run: Path) -> dict:
    for candidate in (run / "hydra" / "config.yaml", run / ".hydra" / "config.yaml"):
        if candidate.is_file():
            config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError(f"Invalid resolved config: {candidate}")
            return config
    raise FileNotFoundError(f"No resolved Hydra config under {run}")


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(child, name))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_router_code() -> dict:
    """Identify the implementation available to this audit process, not historic training code."""
    path = Path(inspect.getfile(SourceAnchoredReadout)).resolve()
    root = Path(__file__).resolve().parents[2]
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True,
                check=False, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        relative = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        relative = str(path)
    return {
        "loaded_class_file": str(path),
        "loaded_class_file_sha256": _sha256(path),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_for_file": git("status", "--short", "--", relative),
        "has_interacting_branch": "query_interaction" in inspect.getsource(SourceAnchoredReadout),
        "scope": "Current server process only; historical training source is unverified unless separately archived.",
    }


def _tensor_summary(value: torch.Tensor) -> dict:
    data = value.detach().float().cpu()
    return {
        "shape": list(value.shape), "all_finite": bool(torch.isfinite(data).all()),
        "l2": float(torch.linalg.vector_norm(data)),
        "max_abs": float(data.abs().max()) if data.numel() else 0.,
        "std": float(data.std(unbiased=False)) if data.numel() else 0.,
    }


def _checkpoint(path: Path, expected_mode: str) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    hparams = checkpoint.get("hyper_parameters", {})
    init = hparams.get("model_init_config", {})
    saved_mode = init.get("competing_risk_config", {}).get("adapter_query_mode")
    if saved_mode != expected_mode:
        raise ValueError(f"{path}: checkpoint mode {saved_mode!r}, expected {expected_mode!r}")
    saved_cr = init.get("competing_risk_config", {})
    fine_names = saved_cr.get("fine_names", [])
    if saved_cr.get("fine_column", "cohort_fine") != "cohort_fine":
        raise ValueError(f"{path}: router is not configured for cohort_fine")
    if not fine_names or len(set(fine_names)) != len(fine_names):
        raise ValueError(f"{path}: missing or duplicated fine-disease roster")
    router = {
        key[key.index(ROUTE_PREFIX) + len(ROUTE_PREFIX):]: value
        for key, value in state.items() if ROUTE_PREFIX in key
    }
    interaction = {key: value for key, value in router.items()
                   if key.startswith("query_interaction.")}
    if bool(interaction) != (expected_mode == "interacting"):
        raise ValueError(f"{path}: query-interaction tensors disagree with mode")
    required = {"patient_query.weight", "fine_query.weight", "target_query.weight",
                "source_keys", "shared_logit", "down", "up", "source_shrinkage"}
    if required - set(router):
        raise ValueError(f"{path}: missing router tensors {sorted(required - set(router))}")
    if any(not torch.isfinite(value).all() for value in router.values()):
        raise ValueError(f"{path}: nonfinite router tensor")
    info = {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "completed_epochs": int(checkpoint["epoch"]) + 1,
        "global_step": int(checkpoint["global_step"]),
        "saved_query_mode": saved_mode,
        "saved_fine_column": saved_cr.get("fine_column", "cohort_fine"),
        "saved_fine_names": list(fine_names),
        "router_tensor_count": len(router),
        "interaction_tensor_count": len(interaction),
        "interaction_parameters": int(sum(value.numel() for value in interaction.values())),
        "router_tensors": {key: _tensor_summary(value) for key, value in router.items()},
        "optimizer_state_count": len(checkpoint.get("optimizer_states", [])),
    }
    head = {key: value for key, value in state.items()
            if key.endswith("competing_risk_head.weight")
            or key.endswith("competing_risk_head.bias")}
    if not head:
        raise ValueError(f"{path}: no direct-risk head in checkpoint")
    if any(not torch.isfinite(value).all() for value in head.values()):
        raise ValueError(f"{path}: nonfinite risk-head tensor")
    info["risk_head_tensors"] = {key: _tensor_summary(value)
                                 for key, value in head.items()}
    return info, router


def _tensor_differences(additive: dict, interacting: dict) -> pd.DataFrame:
    rows = []
    for name in sorted(set(additive) & set(interacting)):
        left, right = additive[name].detach().float(), interacting[name].detach().float()
        if left.shape != right.shape:
            raise ValueError(f"Router tensor shape mismatch: {name}")
        difference = (right - left).abs()
        rows.append({"tensor": name, "n_parameters": int(left.numel()),
                     "exactly_equal": bool(torch.equal(left, right)),
                     "max_abs_difference": float(difference.max()),
                     "mean_abs_difference": float(difference.mean())})
    return pd.DataFrame(rows)


def _epoch_comparison(additive: Path, interacting: Path) -> pd.DataFrame:
    first = _epochs(_metrics_paths(additive))
    second = _epochs(_metrics_paths(interacting))
    rows = []
    for epoch in sorted(set(first) & set(second)):
        for metric in METRICS:
            a, b = first[epoch].get(metric), second[epoch].get(metric)
            if a is None or b is None:
                continue
            rows.append({"epoch": epoch, "metric": metric, "additive": a,
                         "interacting": b, "interacting_minus_additive": b - a,
                         "exactly_equal": a == b})
    if not rows:
        raise ValueError("No common completed validation metrics across the pair")
    return pd.DataFrame(rows)


def _paired_predictions(first_path: Path, second_path: Path) -> pd.DataFrame:
    key = ["subject_id", "cohort", "outcome", "split"]
    first, second = pd.read_parquet(first_path), pd.read_parquet(second_path)
    if not first["split"].eq("tuning").all() or not second["split"].eq("tuning").all():
        raise ValueError("Prediction audit is tuning-only")
    if first.duplicated(key).any() or second.duplicated(key).any():
        raise ValueError("Duplicate patient-target prediction rows")
    columns = sorted(set(first) & set(second))
    cif_columns = [name for name in columns if name.startswith("cif_") and name.endswith("d")]
    if not cif_columns:
        raise ValueError("No common CIF horizon columns")
    merged = first[key + ["time_days", "event"] + cif_columns].merge(
        second[key + ["time_days", "event"] + cif_columns], on=key,
        how="outer", validate="one_to_one", indicator=True, suffixes=("_a", "_i"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("The pair has different patient/target keys")
    if not (merged["time_days_a"].eq(merged["time_days_i"]).all()
            and merged["event_a"].eq(merged["event_i"]).all()):
        raise ValueError("Tuning labels or follow-up differ between runs")
    rows = []
    for column in cif_columns:
        left = merged[f"{column}_a"].to_numpy(dtype=float)
        right = merged[f"{column}_i"].to_numpy(dtype=float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Nonfinite predictions: {column}")
        delta = np.abs(left - right)
        rows.append({"horizon": column, "n_patient_target_rows": len(delta),
                     "fraction_exactly_equal": float(np.mean(left == right)),
                     "mean_abs_difference": float(delta.mean()),
                     "median_abs_difference": float(np.median(delta)),
                     "p95_abs_difference": float(np.quantile(delta, .95)),
                     "max_abs_difference": float(delta.max())})
    return pd.DataFrame(rows)


def _paired_routes(first_path: Path, second_path: Path) -> pd.DataFrame:
    key = ["subject_id", "cohort", "outcome", "split"]
    first, second = pd.read_parquet(first_path), pd.read_parquet(second_path)
    if not first["split"].eq("tuning").all() or not second["split"].eq("tuning").all():
        raise ValueError("Route audit is tuning-only")
    if first.duplicated(key).any() or second.duplicated(key).any():
        raise ValueError("Duplicate patient-target route rows")
    columns = key + ["fine_index", "shared_weight", "disease_weights", "outcome_weights"]
    merged = first[columns].merge(second[columns], on=key, how="outer",
                                  validate="one_to_one", indicator=True,
                                  suffixes=("_a", "_i"))
    if merged.empty:
        raise ValueError("No paired route rows")
    if not merged["_merge"].eq("both").all():
        raise ValueError("The pair has different patient/target route keys")
    if not merged["fine_index_a"].eq(merged["fine_index_i"]).all():
        raise ValueError("Fine-disease identities differ across the pair")
    rows = []
    for bank in ("shared", "disease", "outcome"):
        if bank == "shared":
            a = merged["shared_weight_a"].to_numpy(dtype=float)[:, None]
            b = merged["shared_weight_i"].to_numpy(dtype=float)[:, None]
        else:
            names = sorted(json.loads(merged.iloc[0][f"{bank}_weights_a"]))
            def unpack(column: str) -> np.ndarray:
                parsed = [json.loads(value) for value in merged[column]]
                if any(set(item) != set(names) for item in parsed):
                    raise ValueError(f"{bank} source inventory differs across rows")
                return np.asarray([[item[name] for name in names] for item in parsed], dtype=float)
            a = unpack(f"{bank}_weights_a")
            b = unpack(f"{bank}_weights_i")
        difference = np.abs(a - b)
        rows.append({"bank": bank, "n_patient_target_rows": len(merged),
                     "n_sources": a.shape[1],
                     "fraction_rows_exactly_equal": float(np.mean(np.all(a == b, axis=1))),
                     "mean_abs_weight_difference": float(difference.mean()),
                     "p95_abs_weight_difference": float(np.quantile(difference, .95)),
                     "max_abs_weight_difference": float(difference.max())})
    return pd.DataFrame(rows)


def _route_target_interaction_probe(path: Path, *,
                                    targets: tuple[str, str] = (
                                        "overall_survival", "sepsis"),
                                    source_pair: tuple[str, str] | None = None
                                    ) -> tuple[dict, tuple[str, str]]:
    """Actual-route test of patient-varying target contrasts in source log odds.

    Excluding both home-disease sources removes the forced home floor. In the
    additive model, target contrast of log odds for two remaining sources is
    constant across patients; the interacting model need not obey that.
    """
    frame = pd.read_parquet(path)
    key = ["subject_id", "cohort", "split"]
    if not frame["split"].eq("tuning").all():
        raise ValueError("Target-interaction probe is tuning-only")
    paired = frame.loc[frame["outcome"].eq(targets[0])].merge(
        frame.loc[frame["outcome"].eq(targets[1])],
        on=key, how="inner", validate="one_to_one", suffixes=("_0", "_1"),
    )
    if len(paired) < 20:
        raise ValueError(f"Fewer than 20 patients have both probe targets {targets}")
    if not paired["fine_index_0"].eq(paired["fine_index_1"]).all():
        raise ValueError("Fine disease changed between target rows")
    names = sorted(json.loads(paired.iloc[0]["disease_weights_0"]))
    if len(names) < 2:
        raise ValueError("Need at least two disease sources")
    def matrix(column: str) -> np.ndarray:
        records = [json.loads(value) for value in paired[column]]
        if any(set(record) != set(names) for record in records):
            raise ValueError("Disease source roster changed between patient-target rows")
        return np.asarray([[record[name] for name in names] for record in records], dtype=float)
    first = matrix("disease_weights_0")
    second = matrix("disease_weights_1")
    home = paired["fine_index_0"].to_numpy(dtype=int)
    if source_pair is None:
        candidates = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                valid = ((home != i) & (home != j)
                         & (first[:, i] > 0) & (first[:, j] > 0)
                         & (second[:, i] > 0) & (second[:, j] > 0))
                candidates.append((int(valid.sum()), i, j))
        count, i, j = max(candidates)
        if count < 20:
            raise ValueError("No supported non-home disease-source pair for the probe")
        source_pair = (names[i], names[j])
    else:
        if any(name not in names for name in source_pair):
            raise ValueError("Requested source pair absent from route export")
        i, j = names.index(source_pair[0]), names.index(source_pair[1])
    valid = ((home != i) & (home != j)
             & (first[:, i] > 0) & (first[:, j] > 0)
             & (second[:, i] > 0) & (second[:, j] > 0))
    if int(valid.sum()) < 20:
        raise ValueError("Too few paired non-home positive source weights")
    contrast = (np.log(first[valid, i] / first[valid, j])
                - np.log(second[valid, i] / second[valid, j]))
    centered = contrast - contrast.mean()
    return {
        "target_0": targets[0], "target_1": targets[1],
        "source_0": source_pair[0], "source_1": source_pair[1],
        "n_matched_nonhome_patients": int(valid.sum()),
        "mean_target_log_odds_contrast": float(contrast.mean()),
        "sd_across_patients": float(contrast.std()),
        "p95_abs_deviation_from_mean": float(np.quantile(np.abs(centered), .95)),
        "max_abs_deviation_from_mean": float(np.abs(centered).max()),
    }, source_pair


def audit(additive_run: Path, interacting_run: Path, *,
          additive_eval: Path | None = None,
          interacting_eval: Path | None = None) -> tuple[dict, dict[str, pd.DataFrame]]:
    if (additive_eval is None) != (interacting_eval is None):
        raise ValueError("Provide both evaluation directories or neither")
    configs = [_config(additive_run), _config(interacting_run)]
    flat = [_flatten(config) for config in configs]
    differences = {
        key: {"additive": flat[0].get(key), "interacting": flat[1].get(key)}
        for key in sorted(set(flat[0]) | set(flat[1]))
        if flat[0].get(key) != flat[1].get(key)
    }
    if (flat[0].get(QUERY_PATH), flat[1].get(QUERY_PATH)) != ("additive", "interacting"):
        raise ValueError("Resolved configs do not contain the expected query modes")
    for label, config in zip(("additive", "interacting"), configs):
        cr = config.get("competing_risk", {})
        if cr.get("fine_column", "cohort_fine") != "cohort_fine":
            raise ValueError(f"{label}: resolved router does not use cohort_fine")
        if not cr.get("fine_names"):
            raise ValueError(f"{label}: resolved router has no fine-disease roster")
    checkpoints, states = {}, {}
    last_checkpoints, run_completion, selected_log_epochs = {}, {}, {}
    current_code = _current_router_code()
    training_code = {}
    for label, run in (("additive", additive_run), ("interacting", interacting_run)):
        checkpoints[label], states[label] = _checkpoint(run / "best.ckpt", label)
        provenance_path = run / "router_code_provenance_INTERNAL.json"
        training_code[label] = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file() else None
        )
        if training_code[label] is not None and (
            training_code[label].get("loaded_class_file_sha256")
            != current_code["loaded_class_file_sha256"]
        ):
            raise ValueError(f"{label}: training router source hash differs from current implementation")
        if sorted(checkpoints[label]["saved_fine_names"]) != sorted(
            configs[0 if label == "additive" else 1]["competing_risk"]["fine_names"]
        ):
            raise ValueError(f"{label}: checkpoint and resolved config fine rosters disagree")
        if (run / "last.ckpt").is_file():
            last_checkpoints[label], _ = _checkpoint(run / "last.ckpt", label)
        completion_path = run / "training_complete.json"
        run_completion[label] = (
            json.loads(completion_path.read_text(encoding="utf-8"))
            if completion_path.is_file() else None
        )
        logged = _epochs(_metrics_paths(run))
        selected_log_epochs[label] = max(logged, key=lambda epoch: logged[epoch][MONITOR])
    tables = {
        "common_router_tensors": _tensor_differences(states["additive"], states["interacting"]),
        "common_epoch_metrics": _epoch_comparison(additive_run, interacting_run),
    }
    if additive_eval is not None:
        tables["paired_predictions"] = _paired_predictions(
            additive_eval / "direct_cr_predictions.parquet",
            interacting_eval / "direct_cr_predictions.parquet",
        )
        tables["paired_routes"] = _paired_routes(
            additive_eval / "source_route_weights.parquet",
            interacting_eval / "source_route_weights.parquet",
        )
        additive_probe, source_pair = _route_target_interaction_probe(
            additive_eval / "source_route_weights.parquet")
        interactive_probe, _ = _route_target_interaction_probe(
            interacting_eval / "source_route_weights.parquet",
            source_pair=source_pair,
        )
        tables["target_interaction_probe"] = pd.DataFrame([
            {"model": "additive", **additive_probe},
            {"model": "interacting", **interactive_probe},
        ])
    warnings = [
        "Interactive mode initializes target/fine queries differently and constructs an extra MLP before the risk head; this is not an initialization-matched interaction-only ablation.",
        "Aggregate AUC equality does not imply identical predictions or routes.",
    ]
    for label in ("additive", "interacting"):
        if training_code[label] is None:
            warnings.append(f"{label}: no training-time router source hash was recorded")
    if set(differences) != {QUERY_PATH}:
        warnings.append("Resolved configs differ beyond adapter_query_mode; inspect config_differences.")
    if checkpoints["additive"]["sha256"] == checkpoints["interacting"]["sha256"]:
        warnings.append("The best checkpoint files are byte-identical; investigate run paths.")
    for label in ("additive", "interacting"):
        if checkpoints[label]["completed_epochs"] != selected_log_epochs[label]:
            warnings.append(f"{label}: best checkpoint epoch differs from logged best metric epoch; inspect resumed versions and checkpoint selection.")
        if run_completion[label] is None or run_completion[label].get("status") != "completed":
            warnings.append(f"{label}: training_complete.json is absent or not completed.")
        if label in last_checkpoints and last_checkpoints[label]["completed_epochs"] < checkpoints[label]["completed_epochs"]:
            warnings.append(f"{label}: last checkpoint precedes best checkpoint; inspect resume/checkpoint files.")
    if "paired_predictions" not in tables:
        warnings.append("Patient-level prediction and route comparison remains pending; evaluate both best checkpoints on identical tuning data.")
    else:
        if tables["paired_predictions"]["fraction_exactly_equal"].eq(1).all():
            warnings.append("Every audited CIF prediction is exactly identical; investigate checkpoint loading or inactive router path.")
        if tables["paired_routes"]["fraction_rows_exactly_equal"].eq(1).all():
            warnings.append("Every audited route is exactly identical; investigate query application or route export.")
        probe = tables["target_interaction_probe"].set_index("model")
        if probe.loc["interacting", "sd_across_patients"] <= 1e-6:
            warnings.append("Interacting route log-odds probe is nearly patient-invariant; interaction may be practically inactive for this target/source pair.")
        if probe.loc["additive", "sd_across_patients"] > 1e-5:
            warnings.append("Additive route log-odds probe violates expected separability; inspect home floor, source ordering, or export.")
    report = {
        "status": "paired_tuning_audit" if "paired_predictions" in tables else "checkpoint_and_log_audit_pending_paired_tuning_evaluation",
        "additive_run": str(additive_run.resolve()),
        "interacting_run": str(interacting_run.resolve()),
        "config_differences": differences,
        "current_router_code": current_code,
        "training_router_code": training_code,
        "configs_differ_only_in_query_mode": set(differences) == {QUERY_PATH},
        "checkpoints": checkpoints,
        "last_checkpoints": last_checkpoints,
        "training_completion": run_completion,
        "logged_best_epochs": selected_log_epochs,
        "warnings": warnings,
        "interpretation": "This audit checks implementation and numerical divergence. It does not establish a generalization benefit from interaction.",
    }
    return report, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additive-run", type=Path, required=True)
    parser.add_argument("--interacting-run", type=Path, required=True)
    parser.add_argument("--additive-eval", type=Path)
    parser.add_argument("--interacting-eval", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, tables = audit(
        args.additive_run, args.interacting_run,
        additive_eval=args.additive_eval, interacting_eval=args.interacting_eval,
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "router_query_audit_INTERNAL.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir),
                      "configs_differ_only_in_query_mode": report["configs_differ_only_in_query_mode"],
                      "warnings": report["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
