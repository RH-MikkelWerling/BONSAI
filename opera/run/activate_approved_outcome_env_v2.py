"""Safely point an existing server .env at the sealed metadata and v2 outcomes.

Dry-run by default; --apply backs up the original and atomically updates only
BONSAI_COHORT_MEMBERSHIP and BONSAI_OUTCOMES_DIR. Never prints other .env values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile


TARGETS = ("BONSAI_COHORT_MEMBERSHIP", "BONSAI_OUTCOMES_DIR")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activate(env_file: Path, release: Path, metadata: Path, *, apply: bool) -> dict:
    if any(not path.is_absolute() for path in (env_file, release, metadata)):
        raise ValueError("Use absolute paths for .env, release and metadata")
    if env_file.is_symlink() or not env_file.is_file():
        raise ValueError("Expected an existing, non-symlink .env file")
    if release.is_symlink() or metadata.is_symlink():
        raise ValueError("Release and metadata must not be symlinks")
    manifest_file = metadata.parent / "manifest_INTERNAL.json"
    release_manifest_file = release / "release_manifest_INTERNAL.json"
    if not all(path.is_file() for path in (metadata, manifest_file, release_manifest_file)):
        raise ValueError("Metadata or release manifest is missing")
    meta_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    release_manifest = json.loads(release_manifest_file.read_text(encoding="utf-8"))
    if (metadata.name != "population_metadata.parquet"
            or meta_manifest.get("status") != "internal_metadata_candidate"
            or meta_manifest.get("n_firstline") != 37557
            or meta_manifest.get("n_patients_with_ipi") != 27356
            or meta_manifest.get("n_patients_with_multiple_ipi_rows") != 207
            or meta_manifest.get("sha256", {}).get("metadata") != sha256(metadata)):
        raise ValueError("Metadata does not match the reported sealed cohort/IPI manifest")
    outcomes = release / "outcomes"
    if (release_manifest.get("version") != 2
            or len(release_manifest.get("outcomes", [])) != 69
            or not outcomes.is_dir()
            or len(list(outcomes.glob("*.parquet"))) != 69):
        raise ValueError("Release is not the expected 69-outcome v2 candidate")
    values = {
        "BONSAI_COHORT_MEMBERSHIP": str(metadata),
        "BONSAI_OUTCOMES_DIR": str(outcomes),
    }
    original = env_file.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines(keepends=True)
    seen = set()
    updated = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key not in TARGETS:
            updated.append(line)
            continue
        if key in seen:
            raise ValueError(f"Duplicate {key} in .env; resolve manually")
        seen.add(key)
        updated.append(f"{key}={values[key]}{newline}")
    if updated and not updated[-1].endswith(("\n", "\r")):
        updated[-1] += newline
    for key in TARGETS:
        if key not in seen:
            updated.append(f"{key}={values[key]}{newline}")
    replacement = "".join(updated)
    report = {"status": "validated_dry_run", "env_file": str(env_file),
              "metadata": str(metadata), "outcomes": str(outcomes),
              "n_firstline": meta_manifest["n_firstline"],
              "n_patients_with_ipi": meta_manifest["n_patients_with_ipi"],
              "n_outcomes": len(release_manifest["outcomes"])}
    if not apply:
        return report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = env_file.with_name(f"{env_file.name}.before_approved_outcomes_v2_{stamp}")
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    shutil.copy2(env_file, backup)
    mode = env_file.stat().st_mode & 0o777
    fd, temporary = tempfile.mkstemp(prefix=".env.approved_v2.", dir=env_file.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(replacement)
        os.chmod(temporary, mode)
        os.replace(temporary, env_file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report.update(status="applied", backup=str(backup))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(activate(args.env_file, args.release_dir, args.metadata, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
