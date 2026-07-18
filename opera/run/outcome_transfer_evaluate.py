"""Run the focused OPERA outcome-transfer frozen-probe evaluation.

This command deliberately consumes *pre-extracted frozen embeddings*.  It
does not load an encoder checkpoint or change encoder weights, which makes the
boundary between contrastive adaptation and target-specific downstream probes
auditable.  Extract one embedding table for every required condition/seed,
then pass it using ``--embedding CONDITION:SEED=PATH``.  For example:

.. code-block:: bash

   python -m opera.run.outcome_transfer_evaluate \
     --embedding dapt:42=/results/dapt/seed_42/embeddings.parquet \
     --embedding opera_full:42=/results/opera_full/seed_42/embeddings.parquet \
     --embedding opera_no_g3:42=/results/opera_no_g3/seed_42/embeddings.parquet \
     ... \
     --output-dir /results/outcome_transfer/probes

The production invocation supplies all seven OPERA conditions plus DAPT for
each canonical seed.  CSV/Parquet inputs require ``subject_id`` and numeric
``embedding_*`` columns; NPZ inputs require ``subject_ids`` and ``embeddings``.
Optional checkpoint-metadata files use the same key syntax and are validated
against the resolved transfer plan where they expose provenance fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opera.evaluation.outcome_transfer_evaluation import (
    DEFAULT_C_GRID,
    evaluate_frozen_transfer_probes,
    load_embedding_artifact,
    resolve_registry,
    write_frozen_probe_outputs,
)
from opera.functional.outcome_transfer import DEFAULT_MANIFEST, resolve_transfer_manifest


def _artifact_spec(value: str) -> tuple[tuple[str, int], str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Embedding inputs must use CONDITION:SEED=PATH."
        )
    raw_key, raw_path = value.split("=", 1)
    if ":" not in raw_key or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "Embedding inputs must use CONDITION:SEED=PATH."
        )
    representation, raw_seed = raw_key.rsplit(":", 1)
    if not representation.strip():
        raise argparse.ArgumentTypeError("Embedding condition must be non-empty.")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Embedding seed must be an integer.") from exc
    return (representation.strip(), seed), raw_path.strip()


def _c_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--c-grid must be comma-separated numbers.") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--c-grid must contain positive values.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen OPERA outcome-transfer representations with pan-hematology linear probes."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--base-config", default=None)
    parser.add_argument(
        "--embedding",
        action="append",
        type=_artifact_spec,
        required=True,
        metavar="CONDITION:SEED=PATH",
        help="Frozen embedding artifact; repeat for every required condition/seed.",
    )
    parser.add_argument(
        "--checkpoint-metadata",
        action="append",
        type=_artifact_spec,
        default=[],
        metavar="CONDITION:SEED=PATH",
        help="Optional checkpoint metadata sidecar using the matching embedding key.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--c-grid",
        type=_c_grid,
        default=DEFAULT_C_GRID,
        help="Comma-separated probe C values; selected only on pan-hematology tuning AUROC.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embedding_specs = dict(args.embedding)
    if len(embedding_specs) != len(args.embedding):
        raise ValueError("Embedding condition/seed keys must be unique.")
    metadata_specs = dict(args.checkpoint_metadata)
    if len(metadata_specs) != len(args.checkpoint_metadata):
        raise ValueError("Checkpoint metadata condition/seed keys must be unique.")
    unknown_metadata = sorted(set(metadata_specs) - set(embedding_specs))
    if unknown_metadata:
        raise ValueError(
            "Checkpoint metadata was supplied without a matching embedding artifact: "
            f"{unknown_metadata}."
        )

    plan = resolve_transfer_manifest(
        args.manifest,
        registry_path=args.registry,
        base_config_path=args.base_config,
    )
    registry = resolve_registry(args.registry or plan["registry"])
    artifacts = {
        key: load_embedding_artifact(
            key[0], key[1], path, metadata_path=metadata_specs.get(key)
        )
        for key, path in embedding_specs.items()
    }
    results, predictions, status, failures = evaluate_frozen_transfer_probes(
        plan,
        registry=registry,
        artifacts=artifacts,
        c_grid=args.c_grid,
    )
    written = write_frozen_probe_outputs(
        args.output_dir,
        results=results,
        predictions=predictions,
        probe_status=status,
        failures=failures,
    )
    metadata_path = Path(args.output_dir) / "transfer_evaluation_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "name": "outcome_transfer_frozen_probe_evaluation",
                "manifest": plan["manifest"],
                "manifest_hash": plan["manifest_hash"],
                "registry": plan["registry"],
                "registry_hash": plan["registry_hash"],
                "split_contract": plan["split_contract"],
                "split_contract_hash": plan["split_contract_hash"],
                "base_contrastive_config_hash": plan[
                    "base_contrastive_config_hash"
                ],
                "seeds": plan["seeds"],
                "c_grid": list(args.c_grid),
                "probe": "standardized_logistic_regression",
                "encoder_frozen": True,
                "embedding_inputs": {
                    f"{name}:{seed}": str(artifact.path)
                    for (name, seed), artifact in artifacts.items()
                },
                "checkpoint_metadata_inputs": {
                    f"{name}:{seed}": path
                    for (name, seed), path in metadata_specs.items()
                },
                "output_files": {name: str(path) for name, path in written.items()},
                "n_result_rows": int(len(results)),
                "n_prediction_rows": int(len(predictions)),
                "n_failures": int(len(failures)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(results):,} frozen-probe metric rows and {len(predictions):,} "
        f"held-out predictions to {Path(args.output_dir)}."
    )
    if not failures.empty:
        print(
            f"{len(failures):,} targets/representations lacked support; see "
            f"{written['transfer_failures']}."
        )


if __name__ == "__main__":
    main()
