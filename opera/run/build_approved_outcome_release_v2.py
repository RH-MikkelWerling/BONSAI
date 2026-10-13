"""Build a new, support-gated primary outcome release without mutating sealed inputs.

Applies the explicit outcome_release_primary_v2 policy to an audited 87-outcome
panel, candidate ALT/creatinine and potassium exports. Held-out events are
never used for selection. The output contains a matching primary registry.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from opera.run.decide_outcome_panel import sha256_file, support_for_outcome
from opera.run.export_candidate_outcomes_v1 import SCHEMA, _read_outcome
from opera.run.generate_sweep_configs import validate_registry


SPLITS = {"train", "tuning", "held_out"}
SMR_START = pd.Timestamp("2018-05-15")


def composite_earliest(paths: dict[str, Path], components: list[str]) -> pd.DataFrame:
    """Match the R composite's intersection eligibility/censor and event union."""
    if len(components) < 2 or len(set(components)) != len(components):
        raise ValueError("Composite requires distinct components")
    frames = {name: _read_outcome(paths[name]).set_index("subject_id") for name in components}
    shared = set.intersection(*(set(frame.index) for frame in frames.values()))
    if not shared:
        raise ValueError("Composite has no shared eligible patients")
    ordered = sorted(shared)
    aligned = {name: frame.loc[ordered] for name, frame in frames.items()}
    first = aligned[components[0]]
    for name in components[1:]:
        if (not first["index_date"].equals(aligned[name]["index_date"])
                or not first["split"].astype(str).equals(aligned[name]["split"].astype(str))):
            raise ValueError(f"Composite component {name} has different index/split")
    censor = pd.concat([aligned[name]["censor_date"] for name in components], axis=1).min(axis=1)
    event = pd.concat([aligned[name]["outcome_date"] for name in components], axis=1).min(axis=1)
    event = event.where(event.le(censor))
    result = pd.DataFrame({"subject_id": ordered, "split": first["split"].to_numpy(),
                           "outcome_date": event.to_numpy(), "index_date": first["index_date"].to_numpy(),
                           "censor_date": censor.to_numpy()})[SCHEMA]
    support_for_outcome(result, "rebuilt_composite", [365])
    return result


def _file_table(path: Path, columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not columns.issubset(frame):
        raise ValueError(f"{path} lacks {sorted(columns - set(frame))}")
    return frame


def _check(path: Path, digest: str, name: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{name}: source missing or SHA256 differs: {path}")


def _registry(source: dict, selected: set[str], rename: dict[str, str],
              additions: list[str]) -> dict:
    result = dict(source)
    original = source["outcomes"]
    outcomes = [rename.get(name, name) for name in original if name in selected] + additions
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("Approved roster contains duplicate names")
    result["outcomes"] = outcomes
    result["outcomes_containing_death"] = [rename.get(name, name) for name in
        source["outcomes_containing_death"] if name in selected]
    families = {}
    for family, members in source["outcome_families"].items():
        retained = [rename.get(name, name) for name in members if name in selected]
        if family == "Renal toxicity":
            retained += [name for name in additions if "creatinine" in name]
        if family == "Electrolyte & metabolic toxicity":
            retained += [name for name in additions if "potassium" in name]
        if retained:
            families[family] = retained
    result["outcome_families"] = families
    result["availability_rules"] = {
        key: {**rule, "outcomes": [rename.get(name, name) for name in rule["outcomes"]
                                    if name in selected]}
        for key, rule in source.get("availability_rules", {}).items()
        if any(name in selected for name in rule["outcomes"])
    }
    validate_registry(result)
    return result


def run(panel: Path, base: Path, decision_dir: Path, lab_dir: Path,
        potassium_dir: Path, registry_path: Path, rules_path: Path, policy_path: Path,
        output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if policy.get("version") != 2:
        raise ValueError("Expected primary-release policy version 2")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    names = registry["outcomes"]
    summary = json.loads((decision_dir / "summary_INTERNAL.json").read_text(encoding="utf-8"))
    if (Path(summary["outcomes_dir"]).resolve() != panel.resolve()
            or summary["registry_sha256"] != sha256_file(registry_path)
            or summary["rules_sha256"] != sha256_file(rules_path)):
        raise ValueError("Decision audit belongs to a different panel, registry or rules version")
    decisions = _file_table(decision_dir / "panel_decisions_INTERNAL.csv",
                            {"outcome", "disposition", "train_events", "tuning_events"})
    if set(decisions["outcome"]) != set(names) or decisions["outcome"].duplicated().any():
        raise ValueError("Decision report inventory differs from registry")
    held = set(decisions.loc[decisions["disposition"].eq("definition_hold"), "outcome"])
    include_held = set(policy["include_held"])
    exclude_held = set(policy["exclude_held"])
    if (include_held & exclude_held) or ((include_held | exclude_held) != held):
        raise ValueError(f"Policy must resolve every hold exactly once; unresolved={held - include_held - exclude_held}")
    floors = policy["minimum_events_by_day_365"]
    selected = set(decisions.loc[decisions["disposition"].eq("primary_candidate"), "outcome"]) | include_held
    decision_index = decisions.set_index("outcome")
    for name in selected:
        row = decision_index.loc[name]
        if row["train_events"] < floors["train"] or row["tuning_events"] < floors["tuning"]:
            raise ValueError(f"Selected held outcome is below primary support floor: {name}")
    source_hashes = _file_table(decision_dir / "input_files_sha256_INTERNAL.csv", {"outcome", "sha256"})
    if set(source_hashes["outcome"]) != set(names):
        raise ValueError("Audited source hash inventory differs from registry")
    source_digests = dict(zip(source_hashes["outcome"], source_hashes["sha256"]))
    source_paths = {name: panel / f"{name}.parquet" for name in names}
    for name in selected | set(sum(policy["rebuild_composites"].values(), [])):
        if name in source_paths:
            _check(source_paths[name], source_digests[name], name)
    lab_manifest = json.loads((lab_dir / "manifest_INTERNAL.json").read_text(encoding="utf-8"))
    potassium_manifest = json.loads((potassium_dir / "manifest_INTERNAL.json").read_text(encoding="utf-8"))
    if Path(lab_manifest["old_outcomes_dir"]).resolve() != base.resolve():
        raise ValueError("Lab candidates come from a different base release")
    if str((base / "overall_survival.parquet").resolve()) not in potassium_manifest["input_sha256"]:
        raise ValueError("Potassium candidate comes from a different base release")
    for manifest in (lab_manifest, potassium_manifest):
        for raw, digest in manifest["input_sha256"].items():
            _check(Path(raw), digest, "candidate input")
    replacements = policy["replace_from_lab_candidates"]
    additions = policy["add_from_lab_candidates"] + policy["add_from_potassium_candidates"]
    if set(replacements) != {"ALT_increased_g2plus", "ALT_increased_g3plus"}:
        raise ValueError("ALT must be replaced as a paired G2/G3 definition")
    if len(additions) != len(set(additions)):
        raise ValueError("Duplicate additions")
    files = {name: source_paths[name] for name in selected}
    for name in replacements + policy["add_from_lab_candidates"]:
        path = lab_dir / "outcomes" / f"{name}.parquet"
        _check(path, lab_manifest["output_sha256"][name], name)
        files[name] = path
    for name in policy["add_from_potassium_candidates"]:
        path = potassium_dir / "outcomes" / f"{name}.parquet"
        if potassium_manifest.get("outcome") != name:
            raise ValueError("Potassium candidate manifest does not name expected outcome")
        _check(path, potassium_manifest["output_sha256"], name)
        files[name] = path
    if set(policy["rebuild_composites"]) != {"hepatotoxicity"}:
        raise ValueError("ALT replacement requires explicit hepatotoxicity rebuild")
    composite_parts = {**source_paths, **files}
    rebuilt = {name: composite_earliest(composite_parts, components)
               for name, components in policy["rebuild_composites"].items()}
    alias = policy["rename"]
    if not set(alias).issubset(selected) or set(alias.values()) & set(files):
        raise ValueError("Invalid outcome rename mapping")
    approved_registry = _registry(registry, selected, alias, additions)
    if not set(policy["release_notes"]).issubset(approved_registry["outcomes"]):
        raise ValueError("Release notes refer to outcomes absent from the approved registry")
    # The broad union has an intentionally post-SMR-start risk set, not an
    # era-changing definition on the full historical cohort.
    for name in ("infections_IV_based", "infections_all_definitions"):
        frame = _read_outcome(source_paths[name])
        if frame["index_date"].min() < SMR_START:
            raise ValueError(f"{name} includes pre-SMR-start index dates")
    # Audit every selected parquet and recomputed composite before creating
    # output. A support failure must not leave a partial release directory.
    support_rows = []
    reverse_alias = {new: old for old, new in alias.items()}
    for name in approved_registry["outcomes"]:
        source_name = reverse_alias.get(name, name)
        if source_name in rebuilt:
            frame = rebuilt[source_name]
        else:
            frame = _read_outcome(files[source_name])
        support = pd.DataFrame(support_for_outcome(frame, name, [30, 60, 90, 180, 365, 730]))
        by_365 = support.loc[support["horizon_days"].eq(365)].set_index("split")
        if (by_365.loc["train", "n_events"] < floors["train"]
                or by_365.loc["tuning", "n_events"] < floors["tuning"]):
            raise ValueError(f"New primary outcome below support floor: {name}")
        support_rows.extend(support.to_dict("records"))
    output.mkdir(parents=True, exist_ok=False)
    out_files = output / "outcomes"
    out_files.mkdir()
    provenance = []
    for name in approved_registry["outcomes"]:
        source_name = reverse_alias.get(name, name)
        target = out_files / f"{name}.parquet"
        if source_name in rebuilt:
            rebuilt[source_name].to_parquet(target, index=False)
            origin = "rebuilt_composite"
        else:
            source = files[source_name]
            shutil.copy2(source, target)
            if sha256_file(source) != sha256_file(target):
                raise IOError(f"Failed to verify copied outcome: {name}")
            origin = str(source.resolve())
        provenance.append({"outcome": name, "source": origin, "sha256": sha256_file(target)})
    approved_registry_path = output / "experiment_registry_primary_v2.yaml"
    approved_registry_path.write_text(yaml.safe_dump(approved_registry, sort_keys=False), encoding="utf-8")
    (output / "release_notes_INTERNAL.yaml").write_text(
        yaml.safe_dump(policy["release_notes"], sort_keys=False), encoding="utf-8")
    pd.DataFrame(support_rows).to_csv(output / "horizon_support_INTERNAL.csv", index=False)
    manifest = {"version": 2, "status": "primary_release_candidate_pending_final_signoff",
                "outcomes": provenance, "registry_sha256": sha256_file(approved_registry_path),
                "policy_sha256": sha256_file(policy_path),
                "decision_report_sha256": sha256_file(decision_dir / "panel_decisions_INTERNAL.csv"),
                "held_out_used_for_selection": False,
                "lymphopenia_status": "historical parquets preserved; not in primary registry pending installed-threshold and clinical review",
                "tls_status": "candidate TLS outcomes remain separate exploratory files, not primary roster"}
    (output / "release_manifest_INTERNAL.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output.resolve()), "primary_outcomes": len(provenance),
            "status": manifest["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--panel-review-dir", type=Path, required=True)
    parser.add_argument("--lab-candidates-dir", type=Path, required=True)
    parser.add_argument("--potassium-candidates-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("opera/configs/experiment_registry.yaml"))
    parser.add_argument("--rules", type=Path, default=Path("opera/configs/outcome_panel_decision_v1.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("opera/configs/outcome_release_primary_v2.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.panel_dir, args.base_dir, args.panel_review_dir,
                         args.lab_candidates_dir, args.potassium_candidates_dir,
                         args.registry, args.rules, args.policy, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
