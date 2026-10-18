"""Prepare a staged paper run inventory from approved server inputs.

Configs may be prepared before the checkpoint exists. Execution audits the
actual checkpoint and requires the tuning decisions appropriate to each phase.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from opera.functional.cohort_groups import ALL_EVALUATED_FINE, fine_to_grouped
from opera.functional.direct_cr_contract import validate_direct_cr_config
from opera.run.prepare_supervision_factorial import factorial_configs

ADAPTER_FIELDS = (
    "readout_mode",
    "adapter_residual_scale",
    "adapter_home_floor",
    "adapter_event_tau",
    "adapter_shared_logit_init",
)


def router_config(shared, reference, *, source_mode, rank, query_mode, query_width):
    """Change readout only; preserve the matched shared training protocol."""
    policy = reference["competing_risk"]
    if policy.get("readout_mode") not in {
        "source_anchored_adapters",
        "unanchored_adapters",
    }:
        raise ValueError(
            "Supply an explicitly chosen anchored/borrowed router reference"
        )
    missing = set(ADAPTER_FIELDS) - set(policy)
    if missing:
        raise ValueError(f"Router reference must explicitly record: {sorted(missing)}")
    result = copy.deepcopy(shared)
    risk = result["competing_risk"]
    risk.update({key: policy[key] for key in ADAPTER_FIELDS})
    risk.update(
        adapter_source_mode=source_mode,
        adapter_bottleneck=rank,
        adapter_query_mode=query_mode,
        adapter_query_hidden_dim=query_width,
        fine_column="cohort_fine",
        fine_names=[
            fine
            for fine in ALL_EVALUATED_FINE
            if fine_to_grouped(fine) in shared["cohorts"]
        ],
    )
    result["adapter_initialization"] = "dapt_fresh"
    result["source_opera_ckpt"] = None
    epsilon = reference.get("training", {}).get("adapter_optimizer_epsilon")
    if epsilon is not None:
        result["training"]["adapter_optimizer_epsilon"] = epsilon
    else:
        result["training"].pop("adapter_optimizer_epsilon", None)
    validate_direct_cr_config(result)
    return result


ROUTER_CONTROLS = {
    # alias: (source bank, borrowing, disease embedding)
    "JDE": ("none", True, True),           # joint + disease identity only
    "RDH": ("disease_only", False, False),  # own-disease adapter, no borrowing
    "ROH": ("outcome_only", False, False),  # own-outcome adapter, no borrowing
}


def external_sequences(config, sequence_dir):
    """Point a national arm at our patients tokenised in the national format.

    Outcomes, membership and splits stay the paper's (each cohort's
    ``data_dir``); only the sequences and vocabulary are the national ones.
    """
    if sequence_dir is None:
        return config
    sequence_dir = Path(sequence_dir).resolve()
    config["paths"]["vocabulary"] = str(sequence_dir / "vocabulary.pt")
    files = [str(sequence_dir / f"subject_data_{split}.pt") for split in ("train", "tuning")]
    for cohort in config["cohorts"].values():
        cohort["subject_data_paths"] = files
    return config


# Data efficiency: the joint model trained on nested shares of the training
# patients, from the local and from the adapted national encoder.
DATA_EFFICIENCY_FRACTIONS = (0.1, 0.25, 0.5)


def build_bundle(
    base, reference, checkpoints, *, seeds=(42,), epochs=30, selected_router=None,
    router_controls=True,
    external_sequence_dir=None,
    data_efficiency=True,
):
    """All supervision configs and conditional bounded router candidates."""
    if not checkpoints.get("L") or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Need a local checkpoint path and unique nonempty seeds")
    if bool(checkpoints.get("E")) != bool(checkpoints.get("EA")):
        raise ValueError("Matched external matrix needs both E and EA checkpoint paths")
    if selected_router is not None:
        required = {"source_mode", "rank", "query_mode", "query_width"}
        if required - set(selected_router):
            raise ValueError(
                "Final lock needs source_mode, rank, query_mode and query_width"
            )
        if selected_router["source_mode"] not in {
            "none",
            "disease_only",
            "outcome_only",
            "both",
        }:
            raise ValueError("Unknown locked source bank")
        if selected_router["rank"] not in (8, 16) or (
            selected_router["query_mode"],
            selected_router["query_width"],
        ) not in {("additive", 128), ("interacting", 128), ("interacting", 512)}:
            raise ValueError("Router lock is outside the prespecified bounded screen")
    entries = []
    for seed in seeds:
        factorial = factorial_configs(
            base, encoder_checkpoint=checkpoints["L"], epochs=epochs, seed=seed
        )
        shared = factorial["J_all_diseases_all_outcomes"]
        for name, config in factorial.items():
            entries.append(
                {
                    "name": f"seed{seed}/L/{name}",
                    "phase": "supervision",
                    "origin": "L",
                    "seed": seed,
                    "condition": {},
                    "config": config,
                }
            )
        # Joint model from a randomly initialised encoder of the same
        # architecture: does pretraining help at all? One fit (first seed).
        if seed == seeds[0]:
            random_init = copy.deepcopy(shared)
            random_init["encoder_initialization"] = "random"
            entries.append(
                {
                    "name": f"seed{seed}/RI/J_all_diseases_all_outcomes",
                    "phase": "pretraining_control",
                    "origin": "RI",
                    "seed": seed,
                    "condition": {},
                    "config": random_init,
                }
            )
            # Joint model on the frozen local encoder: only the readout is
            # trained, separating what fine-tuning adds from what pretraining
            # already represents.
            frozen = copy.deepcopy(shared)
            frozen.setdefault("model", {})["freeze_encoder"] = True
            entries.append(
                {
                    "name": f"seed{seed}/FZ/J_all_diseases_all_outcomes",
                    "phase": "pretraining_control",
                    "origin": "FZ",
                    "seed": seed,
                    "condition": {},
                    "config": frozen,
                }
            )
        # Only the first seed is used for bounded method selection. All final
        # source controls are replicated once an explicit router lock is given.
        if seed == seeds[0]:
            for source in ("none", "disease_only", "outcome_only", "both"):
                config = router_config(
                    shared,
                    reference,
                    source_mode=source,
                    rank=8,
                    query_mode="interacting",
                    query_width=128,
                )
                entries.append(
                    {
                        "name": f"seed{seed}/L/source_{source}_r8_q128",
                        "phase": "source_screen",
                        "origin": "L",
                        "seed": seed,
                        "condition": {},
                        "config": config,
                    }
                )
            for source in ("disease_only", "outcome_only", "both"):
                for rank in (8, 16):
                    config = router_config(
                        shared,
                        reference,
                        source_mode=source,
                        rank=rank,
                        query_mode="interacting",
                        query_width=128,
                    )
                    entries.append(
                        {
                            "name": f"seed{seed}/L/rank_{source}_r{rank}_q128",
                            "phase": "rank_screen",
                            "origin": "L",
                            "seed": seed,
                            "condition": {"source_mode": source},
                            "config": config,
                        }
                    )
                    for mode, width in (
                        ("additive", 128),
                        ("interacting", 128),
                        ("interacting", 512),
                    ):
                        config = router_config(
                            shared,
                            reference,
                            source_mode=source,
                            rank=rank,
                            query_mode=mode,
                            query_width=width,
                        )
                        entries.append(
                            {
                                "name": f"seed{seed}/L/query_{source}_r{rank}_{mode}_q{width}",
                                "phase": "query_screen",
                                "origin": "L",
                                "seed": seed,
                                "condition": {"source_mode": source, "rank": rank},
                                "config": config,
                            }
                        )
        if selected_router is not None and router_controls and seed == seeds[0]:
            # Decompose routing: identity, own-adapter capacity, then borrowing.
            for alias, (source, borrowing, embedding) in ROUTER_CONTROLS.items():
                config = router_config(
                    shared, reference, source_mode=source,
                    rank=int(selected_router["rank"]),
                    query_mode=selected_router["query_mode"],
                    query_width=int(selected_router["query_width"]),
                )
                config["competing_risk"].update(
                    adapter_borrowing=borrowing, adapter_disease_embedding=embedding)
                entries.append({
                    "name": f"seed{seed}/L/control_{alias}",
                    "phase": "router_controls", "origin": "L", "seed": seed,
                    "condition": {}, "config": config,
                })
        if selected_router is not None:
            for source in ("none", "disease_only", "outcome_only", "both"):
                config = router_config(
                    shared,
                    reference,
                    source_mode=source,
                    rank=int(selected_router["rank"]),
                    query_mode=selected_router["query_mode"],
                    query_width=int(selected_router["query_width"]),
                )
                entries.append(
                    {
                        "name": f"seed{seed}/L/final_source_{source}",
                        "phase": "final_sources",
                        "origin": "L",
                        "seed": seed,
                        "condition": {},
                        "config": config,
                    }
                )
            for origin in ("E", "EA"):
                if not checkpoints.get(origin):
                    continue
                external_shared = external_sequences(
                    copy.deepcopy(shared), external_sequence_dir)
                external_shared["dapt_ckpt"] = checkpoints[origin]
                for mode, config in (
                    ("J", external_shared),
                    (
                        "R",
                        router_config(
                            external_shared,
                            reference,
                            source_mode=selected_router["source_mode"],
                            rank=int(selected_router["rank"]),
                            query_mode=selected_router["query_mode"],
                            query_width=int(selected_router["query_width"]),
                        ),
                    ),
                ):
                    entries.append(
                        {
                            "name": f"seed{seed}/{origin}/{mode}",
                            "phase": "external",
                            "origin": origin,
                            "seed": seed,
                            "condition": {},
                            "config": config,
                        }
                    )
    if data_efficiency and checkpoints.get("EA") and selected_router is not None:
        seed = seeds[0]
        local = next(e for e in entries if e["name"] == f"seed{seed}/L/J_all_diseases_all_outcomes")
        national = next(e for e in entries if e["name"] == f"seed{seed}/EA/J")
        for origin, source in (("L", local), ("EA", national)):
            for fraction in DATA_EFFICIENCY_FRACTIONS:
                config = copy.deepcopy(source["config"])
                config.pop("paper_run", None)
                config["training"]["train_patient_fraction"] = fraction
                config["training"]["train_fraction_seed"] = seed
                entries.append({
                    "name": f"seed{seed}/{origin}/J_frac{round(fraction * 100)}",
                    "phase": "data_efficiency",
                    "origin": origin,
                    "seed": seed,
                    "condition": {},
                    "config": config,
                })
    for entry in entries:
        entry["config"]["paper_run"] = {key: entry[key] for key in ("origin", "seed")}
        if entry["origin"] == "EA":
            entry["config"]["paper_run"]["direct_external_checkpoint"] = checkpoints[
                "E"
            ]
    return entries


def write_bundle(entries, output, *, provenance, selected_router=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Use a fresh bundle directory; existing configs are immutable"
        )
    output.mkdir(parents=True)
    manifest = []
    for entry in entries:
        path = output / "configs" / (entry["name"] + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# @package _global_\n" + yaml.safe_dump(entry["config"], sort_keys=False)
        )
        scientific_hash = hashlib.sha256(
            json.dumps(entry["config"], sort_keys=True, default=str).encode()
        ).hexdigest()
        manifest.append(
            {key: value for key, value in entry.items() if key != "config"}
            | {
                "regime": entry["config"]["supervision_factorial"]["regime"],
                "config_path": str(path.relative_to(output)),
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scientific_hash": scientific_hash,
                "run_directory": f"{entry['origin']}/seed{entry['seed']}/{scientific_hash[:20]}",
            }
        )
    report = {
        "status": "prepared_not_checkpoint_audited_or_launched",
        "provenance": provenance,
        "selected_router": selected_router,
        "runs": manifest,
        "phase_counts": {
            phase: sum(row["phase"] == phase for row in manifest)
            for phase in sorted({row["phase"] for row in manifest})
        },
    }
    (output / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_resolved_config(path, sequence_data_dir=None):
    """Freeze scientific env-dependent paths before generating immutable configs.

    ``sequence_data_dir`` replaces the config's ``paths.sequence_data_dir``; the
    vocabulary and every cohort's ``data_dir`` interpolate from it, so all point
    at the sequences the encoder was pretrained on.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    path = Path(path).resolve()
    OmegaConf.register_new_resolver("version", lambda: 0, replace=True)
    with initialize_config_dir(config_dir=str(path.parent), version_base="1.2"):
        overrides = ([f"paths.sequence_data_dir={Path(sequence_data_dir).resolve()}"]
                     if sequence_data_dir else [])
        config = compose(config_name=path.stem, overrides=overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    # Relative paths are relative to the invocation directory, as in the
    # existing runners. Save their absolute identities in the paper bundle.
    for key in ("vocabulary", "sequence_data_dir"):
        if resolved.get("paths", {}).get(key):
            resolved["paths"][key] = str(Path(resolved["paths"][key]).resolve())
    for cohort in resolved.get("cohorts", {}).values():
        for key in ("data_dir", "population_file"):
            if cohort.get(key):
                cohort[key] = str(Path(cohort[key]).resolve())
    directories = {
        cohort["data_dir"]
        for cohort in resolved.get("cohorts", {}).values()
        if cohort.get("data_dir")
    }
    if len(directories) == 1:
        outcome_directory = Path(next(iter(directories))) / "outcomes"
        for outcome in resolved.get("outcomes", {}).values():
            for key in (
                "outcome_path",
                "outcome_file",
                "filename",
                "competing_outcome_path",
                "competing_outcome_file",
                "eligibility_path",
                "eligibility_file",
            ):
                if outcome.get(key):
                    value = Path(outcome[key])
                    outcome[key] = str(
                        value.resolve()
                        if value.is_absolute()
                        else (outcome_directory / value).resolve()
                    )
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--release-registry", type=Path, required=True)
    parser.add_argument("--router-reference-config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--external-direct-checkpoint")
    parser.add_argument("--external-dapt-checkpoint")
    parser.add_argument("--router-lock", type=Path)
    parser.add_argument("--sequence-data-dir", type=Path,
                        help="Sequences the local encoder was pretrained on (overrides the base config).")
    parser.add_argument("--external-sequence-dir", type=Path,
                        help="Our patients tokenised in the national encoder's own format "
                        "(prepare_external_sequences); used by the national arms.")
    parser.add_argument("--no-data-efficiency", action="store_true",
                        help="Omit the training-fraction runs (local and national joint).")
    parser.add_argument("--no-router-controls", action="store_true",
                        help="Omit the routing decomposition controls from stage 2.")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override physical batch size consistently in every arm",
    )
    parser.add_argument("--num-workers", type=int,
                        help="Override data-loading workers consistently in every arm")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    base = load_resolved_config(args.base_config, args.sequence_data_dir)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        base["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("num-workers must be nonnegative")
        base.setdefault("hardware", {})["num_workers"] = args.num_workers
    registry = yaml.safe_load(args.release_registry.read_text())
    if (
        set(base["outcomes"]) != set(registry["outcomes"])
        or len(base["outcomes"]) != 69
    ):
        raise ValueError("Paper bundle requires the approved 69-outcome release roster")
    reference = load_resolved_config(args.router_reference_config)
    selected = json.loads(args.router_lock.read_text()) if args.router_lock else None
    checkpoints = {
        "L": args.encoder_checkpoint,
        "E": args.external_direct_checkpoint,
        "EA": args.external_dapt_checkpoint,
    }
    seeds = tuple(int(value) for value in args.seeds.split(","))
    entries = build_bundle(
        base,
        reference,
        checkpoints,
        seeds=seeds,
        epochs=args.epochs,
        selected_router=selected,
        router_controls=not args.no_router_controls,
        external_sequence_dir=args.external_sequence_dir,
        data_efficiency=not args.no_data_efficiency,
    )
    paths = [args.base_config, args.release_registry, args.router_reference_config]
    if args.router_lock:
        paths.append(args.router_lock)
    report = write_bundle(
        entries,
        args.output_dir,
        selected_router=selected,
        provenance={
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    )
    print(
        json.dumps(
            {"output": str(args.output_dir), "phase_counts": report["phase_counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
