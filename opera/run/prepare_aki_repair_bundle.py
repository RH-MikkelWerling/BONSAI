"""Amend frozen paper bundles for AKI, preserving independent fits' identities.

Creates a combined bundle for evaluation, a refit-only bundle for existing GPU
workers, and an explicit dependency inventory. Nothing is trained or evaluated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from opera.run.audit_aki_correction import AKI
from opera.run.build_aki_corrected_release import MANIFEST, validate_release
from opera.run.decide_outcome_panel import sha256_file

DEFAULT_MODELS = ("S_L", "D_L", "O_L", "J_L", "R_AB_L", "RBH_AB_L", "N_A_L")
TARGET_KEYS = ("outcome_path", "outcome_file", "filename")


def scientific_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def canonical_models(row: dict, config: dict, bundle: dict) -> list[str]:
    # Imported lazily: the release comparison tools do not need Torch/Lightning.
    from opera.run.paper_prediction_queue import canonical_models as existing
    return existing(row, config, bundle)


def prepare(bundles: list[Path], release: Path, output: Path, models: list[str]) -> dict:
    release, output = release.resolve(), output.resolve()
    if output.exists() or not models or len(set(models)) != len(models):
        raise ValueError("Require a fresh bundle and unique, explicit retained model aliases")
    amendment, registry = validate_release(release)
    if amendment.get("changed_outcomes") != [AKI] or not amendment.get("parent_release"):
        raise ValueError("Require an AKI-only amended release")
    parent = Path(amendment["parent_release"])
    if sha256_file(parent / MANIFEST) != amendment["parent_manifest_sha256"]:
        raise ValueError("Parent manifest changed")
    old_manifest, _ = validate_release(parent)
    original = {entry["outcome"]: entry["sha256"] for entry in old_manifest["outcomes"]}
    if (amendment["registry_sha256"] != old_manifest["registry_sha256"]
            or any(entry["sha256"] != original[entry["outcome"]]
                   for entry in amendment["outcomes"] if entry["outcome"] != AKI)):
        raise ValueError("AKI amendment changed another endpoint or the registry")
    allowed = set(registry["outcomes"])
    prepared, inventory, sources = [], [], {}
    seen, found_models = set(), set()
    selected_router = None
    for directory in bundles:
        directory = directory.resolve()
        manifest_path = directory / "bundle.json"
        bundle = json.loads(manifest_path.read_text())
        sources[str(manifest_path)] = sha256_file(manifest_path)
        if bundle.get("selected_router"):
            if selected_router and selected_router != bundle["selected_router"]:
                raise ValueError("Source bundles have different locked router choices")
            selected_router = bundle["selected_router"]
        for row in bundle["runs"]:
            path = directory / row["config_path"]
            if sha256_file(path) != row["config_sha256"]:
                raise ValueError(f"Bundle config identity mismatch: {path}")
            config = yaml.safe_load(path.read_text())
            if scientific_hash(config) != row["scientific_hash"]:
                raise ValueError(f"Scientific config hash mismatch: {path}")
            aliases = canonical_models(row, config, bundle)
            retained = set(aliases) & set(models)
            if not retained:
                continue
            found_models.update(retained)
            if row["scientific_hash"] in seen:
                continue
            seen.add(row["scientific_hash"])
            if "${" in json.dumps(config):
                raise ValueError("Use the resolved/frozen paper bundle, not an environment-dependent template")
            outcomes = config.get("outcomes", {})
            if not outcomes or not set(outcomes).issubset(allowed):
                raise ValueError("Retained config uses a different outcome roster")
            if (config.get("source_opera_ckpt") or config.get("resume_from_checkpoint")
                    or config.get("paths", {}).get("resume_ckpt")):
                raise ValueError("Fit inherits supervised weights: audit its parent dependency before repair/reuse")
            # Validate each used target against its parent digest, not merely its
            # filename. Unchanged targets retain their original absolute paths.
            for name, spec in outcomes.items():
                paths = [Path(spec[key]) for key in TARGET_KEYS if spec.get(key)]
                if not paths or any(not p.is_absolute() or sha256_file(p) != original[name] for p in paths):
                    raise ValueError(f"Config target differs from parent release: {row['name']} / {name}")
            affected = AKI in outcomes
            updated, new_row = copy.deepcopy(config), copy.deepcopy(row)
            if affected:
                for key in TARGET_KEYS:
                    if updated["outcomes"][AKI].get(key):
                        updated["outcomes"][AKI][key] = str(release / "outcomes" / f"{AKI}.parquet")
                digest = scientific_hash(updated)
                if digest == row["scientific_hash"]:
                    raise ValueError("Affected fit did not acquire a new identity")
                new_row.update(scientific_hash=digest,
                               run_directory=f"{row['origin']}/seed{row['seed']}/{digest[:20]}")
                content = ("# @package _global_\n" + yaml.safe_dump(updated, sort_keys=False)).encode()
            else:
                content = path.read_bytes()
            # Content-based filenames avoid name collisions across multiple bundles.
            new_row.update(config_path=f"configs/{new_row['scientific_hash']}.yaml",
                           config_sha256=hashlib.sha256(content).hexdigest())
            inventory.append({"name": row["name"], "models": ",".join(aliases), "seed": row["seed"],
                              "action": "refit" if affected else "carry_forward",
                              "old_scientific_hash": row["scientific_hash"],
                              "scientific_hash": new_row["scientific_hash"],
                              "old_run_directory": row["run_directory"],
                              "run_directory": new_row["run_directory"],
                              "source_bundle": str(directory),
                              "reason": "AKI in supervised/selection target set" if affected
                                        else "Independent fit; no AKI target; original config unchanged"})
            prepared.append((new_row, content, affected))
    if set(models) - found_models:
        raise ValueError(f"Retained models not found in source bundles: {sorted(set(models) - found_models)}")
    if not prepared:
        raise ValueError("No retained fits")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = dict(Counter(entry["action"] for entry in inventory))
    report = {"status": "prepared_not_checkpoint_audited_or_launched",
              "selected_router": selected_router, "provenance": sources,
              "outcome_amendment": {"release": str(release), "manifest_sha256": sha256_file(release / MANIFEST),
                                    "changed_outcomes": [AKI], "retained_models": models,
                                    "unaffected_target_paths_preserved": True},
              "dependency_counts": counts}
    with tempfile.TemporaryDirectory(prefix=".aki_bundle_", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for subdir, refits_only in ((staged, False), (staged / "refit_bundle", True)):
            (subdir / "configs").mkdir(parents=True)
            rows = []
            for row, content, affected in prepared:
                if refits_only and not affected:
                    continue
                (subdir / row["config_path"]).write_bytes(content)
                rows.append(row)
            payload = report | {"runs": rows, "phase_counts": dict(Counter(row["phase"] for row in rows))}
            (subdir / "bundle.json").write_text(json.dumps(payload, indent=2) + "\n")
        pd.DataFrame(inventory).to_csv(staged / "dependency_inventory.csv", index=False)
        staged.rename(output)
    return {"bundle_dir": str(output), "worker_bundle": str(output / "refit_bundle"),
            "dependency_counts": counts, "retained_models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", action="append", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="Final reported aliases, including any additional retained seeds/controls")
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle_dir, args.release_dir, args.output_dir, args.models), indent=2))


if __name__ == "__main__":
    main()
