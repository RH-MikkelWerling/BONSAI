"""Versioned repair for text IDs in sealed metadata versus numeric EHR IDs.

Reads actual sequence subjects and the released survival parquet before any
write. Refuses to guess if canonicalized IDs do not overlap sufficiently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
from pathlib import Path

import pandas as pd
import torch


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert(ids: pd.Series, sample_id: object) -> pd.Series:
    text = ids.astype("string")
    if isinstance(sample_id, numbers.Integral):
        if text.isna().any() or not text.str.fullmatch(r"[0-9]+").all():
            raise ValueError("Metadata IDs are not all nonmissing decimal integers")
        numbers_out = pd.to_numeric(text, errors="raise").astype("int64")
        if numbers_out.astype("string").str.lstrip("0").replace("", "0").tolist() != text.str.lstrip("0").replace("", "0").tolist():
            raise ValueError("Numeric ID conversion changed a patient identifier")
        return numbers_out
    if isinstance(sample_id, str):
        return text.astype(str)
    raise ValueError(f"Unsupported sequence subject ID type: {type(sample_id).__name__}")


def repair(source: Path, survival: Path, sequence: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Versioned metadata output already exists: {output}")
    metadata_path = source / "population_metadata.parquet"
    long_path = source / "ipi_records_long_INTERNAL.parquet"
    manifest_path = source / "manifest_INTERNAL.json"
    for path in (metadata_path, long_path, manifest_path, survival,
                 sequence / "subject_data_train.pt", sequence / "subject_data_tuning.pt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256", {}).get("metadata") != _sha(metadata_path):
        raise ValueError("Source metadata hash differs from its manifest")
    metadata = pd.read_parquet(metadata_path)
    outcome = pd.read_parquet(survival, columns=["subject_id"])
    if len(metadata) != 37557 or metadata.subject_id.isna().any() or metadata.subject_id.duplicated().any():
        raise ValueError("Source metadata is not the unique sealed cohort")
    # Load the same physical subject partitions the training data module uses.
    sequence_ids = []
    for name in ("subject_data_train.pt", "subject_data_tuning.pt"):
        subjects = torch.load(sequence / name, map_location="cpu", weights_only=False)
        sequence_ids.extend(item["subject_id"] for item in subjects)
        del subjects
    if not sequence_ids or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("Physical sequence subjects are missing or duplicated")
    sample_id = sequence_ids[0]
    if any(type(item) is not type(sample_id) for item in sequence_ids):
        raise ValueError("Physical sequence subject ID types are mixed")
    raw_overlap = len(set(metadata.subject_id) & set(sequence_ids))
    converted = _convert(metadata.subject_id, sample_id)
    if converted.duplicated().any():
        raise ValueError("ID conversion collapses distinct metadata patients")
    converted_overlap = len(set(converted) & set(sequence_ids))
    if converted_overlap < 10000 or converted_overlap <= raw_overlap:
        raise ValueError(f"ID type conversion is not a demonstrated repair: raw={raw_overlap}, converted={converted_overlap}")
    if not set(outcome.subject_id) <= set(converted):
        raise ValueError("Converted metadata IDs do not cover released survival patients")
    metadata["subject_id"] = converted
    long = pd.read_parquet(long_path)
    long["subject_id"] = _convert(long.subject_id, sample_id)
    if not set(long.subject_id) <= set(metadata.subject_id):
        raise ValueError("IPI companion IDs differ after conversion")
    output.mkdir(parents=True, exist_ok=False)
    new_metadata = output / "population_metadata.parquet"
    new_long = output / "ipi_records_long_INTERNAL.parquet"
    metadata.to_parquet(new_metadata, index=False)
    long.to_parquet(new_long, index=False)
    result = {**manifest, "status": "internal_metadata_candidate",
              "id_type_repair": {"source_metadata": str(metadata_path.resolve()),
                                 "source_metadata_sha256": _sha(metadata_path),
                                 "sequence_data": str(sequence.resolve()),
                                 "physical_subjects": len(sequence_ids),
                                 "raw_overlap": raw_overlap,
                                 "converted_overlap": converted_overlap,
                                 "target_type": type(sample_id).__name__}}
    result["sha256"] = {**manifest.get("sha256", {}),
                        "metadata": _sha(new_metadata),
                        "ipi_records_long": _sha(new_long)}
    (output / "manifest_INTERNAL.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"output_dir": str(output.resolve()), **result["id_type_repair"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--survival", type=Path, required=True)
    parser.add_argument("--sequence-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.metadata_root, args.survival, args.sequence_data, args.output_root), indent=2))


if __name__ == "__main__":
    main()
