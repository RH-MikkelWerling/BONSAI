"""Hash every explicitly resolved first-line input; list loader inputs still unknown.

Run on the secure server before a new cohort release. An older manifest or
archived input copy is required to prove that any file changed since v4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DIRECT_INPUTS = {
    "cll_seed": "../../projects2/dalyca_r/andkat_r/medicine data/data/output tables/CLL_wide_V4_100lines.parquet",
    "mm_seed_loaded_then_overwritten": "../../projects2/dalyca_r/andkat_r/medicine data/data/output tables/MM_wide_V4_100lines.parquet",
    "mm_seed_effective": "/ngc/projects2/dalyca_r/mikwer_r/MM_V4_5lines.parquet",
    "treatment_table": "../../projects2/dalyca_r/andkat_r/data_for_treatment/table_treatment_clean_2026_04_05.parquet",
}
LOADER_INPUTS = ("patient", "t_dalycare_diagnoses", "icd10_codebook")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(path: Path) -> dict:
    if path.is_symlink() or not path.exists():
        raise ValueError(f"Missing or symlink input: {path}")
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    if not files or any(file.is_symlink() for file in files):
        raise ValueError(f"Empty or symlink-containing input: {path}")
    files = [file for file in files if file.is_file()]
    if not files:
        raise ValueError(f"No files in input: {path}")
    return {
        "path": str(path.resolve()), "n_files": len(files),
        "total_bytes": sum(file.stat().st_size for file in files),
        "files": [{"relative": str(file.relative_to(path) if path.is_dir() else Path(file.name)),
                   "size_bytes": file.stat().st_size,
                   "mtime_ns": file.stat().st_mtime_ns, "sha256": _sha256(file)} for file in files],
    }


def audit(source_workdir: Path, script: Path, loader: Path, known_tables: dict[str, Path],
          sql_tables: dict[str, str] | None = None, sql_snapshot_id: str | None = None) -> dict:
    source_workdir = source_workdir.resolve(strict=True)
    sql_tables = sql_tables or {}
    if set(known_tables).intersection(sql_tables):
        raise ValueError("A loader input cannot be both a file and SQL table")
    inputs = {"firstline_script": script, "dalycare_loader": loader}
    inputs.update({name: source_workdir / relative for name, relative in DIRECT_INPUTS.items()})
    inputs.update(known_tables)
    unresolved = sorted(set(LOADER_INPUTS) - set(known_tables) - set(sql_tables))
    return {
        "status": "loader_sources_unresolved" if unresolved else (
            "sql_live_unfrozen" if sql_tables and not sql_snapshot_id
            else "sources_declared_not_runtime_verified"),
        "source_workdir": str(source_workdir),
        "unresolved_loader_inputs": unresolved,
        "sql_tables": sql_tables,
        "sql_snapshot_id": sql_snapshot_id,
        "note": "SQL identifiers and snapshot IDs are declarations, not proof of the executed query or historical table contents. Verify against loader/query logs. RKKP tables loaded after the mapped-export marker are excluded.",
        "inputs": {name: _describe(path) for name, path in inputs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workdir", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--known-table", action="append", default=[], metavar="NAME=ABSOLUTE_PATH")
    parser.add_argument("--sql-table", action="append", default=[], metavar="NAME=QUALIFIED_TABLE")
    parser.add_argument("--sql-snapshot-id", help="Database snapshot/backup ID, not merely today's date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")
    tables = {}
    for item in args.known_table:
        name, sep, path = item.partition("=")
        if not sep or name not in LOADER_INPUTS or name in tables or not Path(path).is_absolute():
            raise ValueError(f"Invalid known table {item!r}")
        tables[name] = Path(path)
    sql_tables = {}
    for item in args.sql_table:
        name, sep, identifier = item.partition("=")
        if not sep or name not in LOADER_INPUTS or name in sql_tables or not identifier.strip():
            raise ValueError(f"Invalid SQL table {item!r}")
        sql_tables[name] = identifier.strip()
    report = audit(args.source_workdir, args.script, args.loader, tables,
                   sql_tables, args.sql_snapshot_id)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "unresolved_loader_inputs": report["unresolved_loader_inputs"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
