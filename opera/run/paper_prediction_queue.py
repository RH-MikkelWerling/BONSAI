"""Evaluate completed paper fits, recording missing fits and immutable provenance."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from bonsai.functional.checkpointing import verified_training_selection
from opera.run.run_paper_queue import file_hash


def bundle_runs(bundle_dir):
    directory = Path(bundle_dir)
    bundle = json.loads((directory / "bundle.json").read_text())
    priority = {
        "final_sources": 0,
        "external": 1,
        "supervision": 2,
        "source_screen": 3,
        "rank_screen": 4,
        "query_screen": 5,
        "pretraining_control": 6,
        "router_controls": 7,
        "router_fix": 8,
    }
    seen = set()
    rows = []
    for row in sorted(bundle["runs"], key=lambda r: (priority[r["phase"]], r["name"])):
        if row["scientific_hash"] in seen:
            continue
        path = directory / row["config_path"]
        if file_hash(path) != row["config_sha256"]:
            raise ValueError(f"Bundle config was modified: {path}")
        config = yaml.safe_load(path.read_text())
        if (
            hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode()
            ).hexdigest()
            != row["scientific_hash"]
        ):
            raise ValueError("Scientific config hash differs from bundle")
        seen.add(row["scientific_hash"])
        rows.append((row, path.resolve(), config))
    return bundle, rows


def canonical_models(row, config, bundle):
    origin = row["origin"]
    if row["phase"] == "supervision":
        return [f"{row['regime']}_{origin}"]
    if row["phase"] == "router_controls":
        return [row["name"].rsplit("control_", 1)[-1] + f"_{origin}"]  # JDE_L, RDH_L, ROH_L
    if row["phase"] == "router_fix":
        return [row["name"].rsplit("fix_", 1)[-1] + f"_{origin}"]  # R_AB_L, N_A_L, R_AB_s43_L
    if row["phase"] == "data_efficiency":
        fraction = round(config["training"]["train_patient_fraction"] * 100)
        return [f"J_{origin}_F{fraction}"]  # J_L_F10, J_EA_F25, ...
    if row["phase"] == "pretraining_control":
        return [f"J_{origin}"]  # joint model without pretraining: J_RI
    if row["phase"] == "external":
        return [
            f"{'J' if config['competing_risk']['readout_mode'] == 'shared' else 'R'}_{origin}"
        ]
    if row["phase"] == "final_sources":
        bank = config["competing_risk"]["adapter_source_mode"]
        aliases = [
            {
                "none": "N_L",
                "disease_only": "RD_L",
                "outcome_only": "RO_L",
                "both": "RDO_L",
            }[bank]
        ]
        if bank == bundle["selected_router"]["source_mode"]:
            aliases.append("R_L")
        return aliases
    return []  # Tuning candidates are never relabelled as the final model.


ROUTED_MODELS = {"R_L", "R_E", "R_EA",
                 # router fix (2026-10-26): every fit with a routing choice
                 "R_AB_L", "RBH_AB_L", "R_A_L", "R_ABT_L", "R_AB_s43_L",
                 "R_O_L", "RBH_O_L", "R_O_s43_L"}


def export_flags(aliases, args):
    """Embeddings for every locked paper model; routes for the locked routers."""
    return {
        "export_embeddings": bool(aliases),
        "export_routes": bool(args.export_routes or ROUTED_MODELS & set(aliases)),
    }


EXPORT_KEYS = ("export_embeddings", "export_routes")


def adds_exports_only(saved, signature):
    """True when ``signature`` differs from ``saved`` only by switching export
    flags on (never off, never any scientific input)."""
    same_inputs = ({k: v for k, v in saved.items() if k not in EXPORT_KEYS}
                   == {k: v for k, v in signature.items() if k not in EXPORT_KEYS})
    gained = [k for k in EXPORT_KEYS if signature.get(k) and not saved.get(k)]
    lost = [k for k in EXPORT_KEYS if saved.get(k) and not signature.get(k)]
    return same_inputs and bool(gained) and not lost


def covers(saved, signature):
    """True when ``saved`` is ``signature`` or the same evaluation with more
    exports switched on (a later paper-model evaluation serves earlier asks)."""
    return saved == signature or adds_exports_only(signature, saved)


def evaluation_command(path, checkpoint, output, args, flags=None):
    command = [
        sys.executable,
        "-m",
        "opera.run.evaluate_direct_competing_risk",
        "--checkpoint",
        str(checkpoint),
        "--config-dir",
        str(path.parent),
        "--config-name",
        path.stem,
        "--split",
        args.split,
        "--horizons",
        args.horizons,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--output-dir",
        str(output),
    ]
    flags = flags or {"export_routes": args.export_routes}
    if flags.get("export_routes"):
        command.append("--export-routes")
    if flags.get("export_embeddings"):
        command.append("--export-embeddings")
    if flags.get("borrowing_design"):
        command.append("--borrowing-design")
    return command


def removal_directory(output_dir, run_directory, split):
    """The borrowing-design evaluation, beside (never inside) the fit's ordinary
    evaluation: <run>/<split>_source_removal."""
    return Path(output_dir) / run_directory / f"{split}_source_removal"


def evaluate_row(row, path, config, args, aliases=(), borrowing_design=False):
    """Evaluate one fit; ``borrowing_design`` also writes the CIF of every
    source-removal variant (one pass, its own directory)."""
    run = Path(args.runs_dir) / row["run_directory"]
    output = Path(args.output_dir) / row["run_directory"] / args.split
    if borrowing_design:
        output = removal_directory(args.output_dir, row["run_directory"], args.split)
    checkpoint = run / "best.ckpt"
    result = {
        "name": row["name"],
        "scientific_hash": row["scientific_hash"],
        "run_directory": str(run),
        "prediction_directory": str(output),
        "split": args.split,
        "paper_models": list(aliases),
    }
    if borrowing_design:
        result["borrowing_design"] = True
    if args.split == "held_out" and not aliases:
        # Held-out labels are only opened for locked paper models; tuning-only
        # screening candidates are never scored on held-out patients.
        return result | {"status": "held_out_closed_for_screening_candidate"}
    if (run / "not_estimable.json").is_file():
        return result | {"status": "not_estimable"}
    try:
        selection = verified_training_selection(run)
    except ValueError as error:
        return result | {"status": str(error)}
    if selection["monitor"] != config["training"]["eval_monitor_metric"]:
        return result | {"status": "selection_monitor_differs_from_config"}
    marker = output / "evaluation_complete.json"
    if not args.execute:
        return result | {
            "status": "evaluation_recorded" if marker.exists() else "ready_to_evaluate"
        }
    signature = {
        "schema": 1,
        "checkpoint_sha256": selection["best_checkpoint_sha256"],
        "config_sha256": row["config_sha256"],
        "split": args.split,
        "horizons": args.horizons,
        **(export_flags(aliases, args) if not borrowing_design
           else {"borrowing_design": "drop_source_term_v1"}),
    }
    output.mkdir(parents=True, exist_ok=True)
    intent = output / "evaluation_intent.json"
    if intent.exists() and adds_exports_only(json.loads(intent.read_text()), signature):
        # A screening candidate that became a locked paper model: same
        # checkpoint, config, split and horizons, but it now also needs its
        # routes/embeddings. Keep the old evaluation beside it and redo it.
        output.rename(output.with_name(f"{output.name}.superseded_without_exports"))
        output.mkdir(parents=True)
        result["superseded"] = True
    with (output / ".evaluation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if intent.exists() and not covers(json.loads(intent.read_text()), signature):
            raise ValueError(
                f"Evaluation inputs changed; use a fresh prediction root: {output}"
            )
        if marker.exists():
            saved = json.loads(marker.read_text())
            if not covers(saved["inputs"], signature) or saved["prediction_sha256"] != file_hash(
                output / "direct_cr_predictions.parquet"
            ):
                raise ValueError("Recorded predictions/provenance changed")
            return result | {"status": "reused_verified_evaluation"}
        if not intent.exists() and any(
            p.name != ".evaluation.lock" for p in output.iterdir()
        ):
            raise FileExistsError(
                f"Untracked evaluation directory already exists: {output}"
            )
        intent.write_text(json.dumps(signature, indent=2) + "\n")
        flags = (export_flags(aliases, args) if not borrowing_design
                 else {"borrowing_design": True})
        command = evaluation_command(path, checkpoint, output, args, flags)
        with (output / "evaluation.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        predictions = output / "direct_cr_predictions.parquet"
        if not predictions.is_file():
            raise RuntimeError("Evaluation did not produce predictions")
        if borrowing_design and not (output / "source_removal" / "design.json").is_file():
            raise RuntimeError("Evaluation did not produce the borrowing design")
        marker.write_text(
            json.dumps(
                {
                    "inputs": signature,
                    "command": command,
                    "prediction_sha256": file_hash(predictions),
                },
                indent=2,
            )
            + "\n"
        )
    return result | {"status": "evaluated"}


def removal_report(rows, bundle, args):
    fits = [(row, path, config, aliases) for row, path, config in rows
            for aliases in [canonical_models(row, config, bundle)]
            if args.removal_model in aliases]
    if len(fits) != 1:
        raise ValueError(f"Expected one fit for {args.removal_model}, found {len(fits)}; "
                         "source removal needs the stage-2 bundle")
    row, path, config, aliases = fits[0]
    if config["competing_risk"].get("adapter_source_mode") not in ("disease_only", "both"):
        raise ValueError(f"{args.removal_model} has no disease sources to remove")
    return [evaluate_row(row, path, config, args, aliases, borrowing_design=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("tuning", "held_out"), default="tuning")
    parser.add_argument("--horizons", default="30,60,90,180,365,730")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--export-routes", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--only", action="append", default=[],
                        help="Evaluate only these phases, e.g. source_screen or "
                             "supervision:J (repeatable)")
    parser.add_argument("--source-removal", action="store_true",
                        help="Borrowing design of the locked router: every disease "
                             "source dropped, kept alone and left out, in one pass")
    parser.add_argument("--removal-model", default="R_L",
                        help="--source-removal: the locked routed model")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (
        args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
        or (args.max_runs is not None and args.max_runs < 1)
    ):
        parser.error("Invalid shard or max-runs")
    if args.status:
        args.execute = False
    bundle, rows = bundle_runs(args.bundle_dir)
    if args.only:
        rows = [item for item in rows
                if item[0]["phase"] in args.only
                or f"{item[0]['phase']}:{item[0].get('regime')}" in args.only]
    if args.split == "held_out":
        rows = [item for item in rows if canonical_models(item[0], item[2], bundle)]
    if not args.source_removal:
        rows = rows[args.shard_index :: args.shard_count]
        if args.max_runs:
            rows = rows[: args.max_runs]
    if args.source_removal:
        report = removal_report(rows, bundle, args)
    else:
        report = [
            evaluate_row(row, path, config, args, canonical_models(row, config, bundle))
            for row, path, config in rows
        ]
    if args.status:
        from collections import Counter

        print(
            json.dumps(
                {
                    "runs": len(report),
                    "status_counts": dict(Counter(row["status"] for row in report)),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2))
    if args.execute:
        destination = (
            Path(args.output_dir)
            / f"queue_{'removal_' if args.source_removal else ''}{args.split}"
              f"_shard{args.shard_index}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
