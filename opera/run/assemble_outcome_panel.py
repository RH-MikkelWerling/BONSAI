"""Copy canonical outcomes from three versioned stages into one frozen panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(base: Path, secondline: Path, composites: Path,
             output: Path, registry: Path) -> dict:
    sources = (base.resolve(), secondline.resolve(), composites.resolve())
    for source in sources:
        if not source.is_dir():
            raise ValueError(f"Missing source directory: {source}")
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    if not (base / "coverage_gated_outcomes.csv").is_file():
        raise ValueError("Base outcomes are not the reviewed, coverage-gated copy")
    expected = yaml.safe_load(registry.read_text(encoding="utf-8"))["outcomes"]
    if len(expected) != len(set(expected)):
        raise ValueError("Registry contains duplicate outcomes")
    selected: dict[str, Path] = {}
    for source in sources:
        for path in sorted(source.glob("*.parquet")):
            if "__" in path.stem:
                continue
            if path.stem in selected:
                raise ValueError(f"Duplicate outcome {path.stem}: {selected[path.stem]}, {path}")
            selected[path.stem] = path
    missing = sorted(set(expected) - set(selected))
    extra = sorted(set(selected) - set(expected))
    if missing or extra:
        raise ValueError(f"Outcome inventory mismatch; missing={missing}, extra={extra}")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in expected:
        source = selected[name]
        target = output / source.name
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise IOError(f"Copy verification failed: {source}")
        rows.append({"outcome": name, "source": str(source),
                     "sha256": sha256(target), "size_bytes": target.stat().st_size})
    for name in ("coverage_gated_outcomes.csv", "coverage_gate_provenance.txt"):
        shutil.copy2(base / name, output / name)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "registry": str(registry.resolve()),
                "registry_sha256": sha256(registry),
                "outcomes": rows}
    (output / "assembly_manifest_INTERNAL.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output.resolve()), "outcomes": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-outcomes-dir", type=Path, required=True)
    parser.add_argument("--secondline-dir", type=Path, required=True)
    parser.add_argument("--composites-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path,
                        default=Path("opera/configs/experiment_registry.yaml"))
    args = parser.parse_args()
    print(json.dumps(assemble(args.base_outcomes_dir, args.secondline_dir,
                              args.composites_dir, args.output_dir,
                              args.registry)))


if __name__ == "__main__":
    main()
