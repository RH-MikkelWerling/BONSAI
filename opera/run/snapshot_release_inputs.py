"""Copy explicit upstream inputs into a new, checksummed INTERNAL release snapshot.

The spec is a JSON object with ``items`` entries containing ``name`` and
absolute ``path``. Names become top-level directories under the snapshot.
Dry-run is the default. This tool must run on the secure data server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _files(source: Path) -> list[Path]:
    if source.is_symlink():
        raise ValueError(f"Symlink source requires explicit resolution: {source}")
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink inside source requires explicit resolution: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError(f"Empty source directory: {source}")
    return sorted(files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(spec_path: Path, output_dir: Path) -> tuple[list[dict], int]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("version") != 1 or not isinstance(spec.get("items"), list) or not spec["items"]:
        raise ValueError("Expected version 1 and a nonempty items list")
    names = set()
    rows = []
    total = 0
    for item in spec["items"]:
        name = item.get("name")
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Invalid snapshot item name: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate snapshot item name: {name}")
        names.add(name)
        source = Path(item["path"])
        if not source.is_absolute():
            raise ValueError(f"Use an absolute source path: {source}")
        if output_dir.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Snapshot and source must not be nested: {source}")
        for file in _files(source):
            relative = file.relative_to(source) if source.is_dir() else Path(file.name)
            size = file.stat().st_size
            rows.append({"item": name, "source": str(file.resolve()),
                         "relative": str(Path(name) / relative), "size_bytes": size})
            total += size
    return rows, total


def run(spec_path: Path, output_dir: Path, *, execute: bool = False) -> dict:
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite snapshot: {output_dir}")
    rows, total = inventory(spec_path, output_dir)
    report = {"status": "dry_run" if not execute else "copied_and_verified",
              "output_dir": str(output_dir.resolve()), "n_files": len(rows),
              "total_bytes": total, "files": rows}
    if not execute:
        return report
    free = shutil.disk_usage(output_dir.parent).free
    if free < total * 1.1:
        raise OSError(f"Insufficient free space: need ~{int(total * 1.1)} bytes; have {free}")
    output_dir.mkdir(parents=False, exist_ok=False)
    for row in rows:
        source = Path(row["source"])
        target = output_dir / row["relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        row["source_sha256"] = _sha256(source)
        shutil.copy2(source, target)
        row["copy_sha256"] = _sha256(target)
        if row["copy_sha256"] != row["source_sha256"]:
            raise IOError(f"Snapshot copy did not verify: {target}")
    (output_dir / "snapshot_manifest_INTERNAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually copy and hash; dry-run is default")
    args = parser.parse_args()
    report = run(args.spec, args.output_dir, execute=args.execute)
    print(json.dumps({k: report[k] for k in ("status", "output_dir", "n_files", "total_bytes")}))


if __name__ == "__main__":
    main()
