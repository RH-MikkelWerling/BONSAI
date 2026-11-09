"""Set the analysis environment the lock prescribes before the test set is opened.

Edits ``.env`` in place (a dated backup is kept) so that:
  OPERA_PAPER_ANALYSIS_ESTIMAND=cif_ipcw          (primary estimand)
  OPERA_PAPER_OPERA_MODEL=J_OA_OWN_L              (lock, section 1)
  OPERA_PAPER_ROUTED_MODEL=R_AB_L                 (R_L stays learned routing)
  OPERA_PAPER_ATTRIBUTION_CHECKPOINT / _CONFIG_DIR / _CONFIG_NAME
  OPERA_PAPER_TASK_STRUCTURE_CHECKPOINT           (all: J_OA_OWN seed 42, best.ckpt)
and, in the clinical-score configuration, points the compared model at
the --model alias and the cutoff rule at cohort_size_matched (backup kept). Prints every change; run again and it changes nothing.

    .venv-v100/bin/python scripts/set_test_env.py [--stage 16]
"""
from __future__ import annotations

import argparse
import datetime
import re
import shutil
from pathlib import Path

from opera.run.paper_prediction_queue import bundle_runs, canonical_models

ROOT = Path("data/paper_256_v3")



def selected_fit(bundle, rows, model):
    """Resolve the explicitly selected alias at seed 42 on any encoder."""
    found = [(row, path) for row, path, config in rows
             if model in canonical_models(row, config, bundle) and row.get("seed") == 42]
    if len(found) != 1:
        raise ValueError(f"Expected one seed-42 fit for {model}, found {len(found)}")
    return found[0]


def set_keys(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text().splitlines()
    seen = set()
    for index, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in values and not line.lstrip().startswith("#"):
            new = f"{key}={values[key]}"
            if line != new:
                print(f"  {path}: {line!r} -> {new!r}")
                lines[index] = new
            seen.add(key)
    missing = [k for k in values if k not in seen]
    if missing:
        lines.append(f"# set by scripts/set_test_env.py ({datetime.date.today()})")
        for key in missing:
            print(f"  {path}: + {key}={values[key]}")
            lines.append(f"{key}={values[key]}")
    path.write_text("\n".join(lines) + "\n")


def env_value(path: Path, key: str) -> str | None:
    for line in path.read_text().splitlines():
        if line.split("=", 1)[0].strip() == key and not line.lstrip().startswith("#"):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


def clinical_models(config: dict) -> dict[str, str]:
    """The model compared with the indices: all_indices.template.model and each score's model."""
    found = {}
    template = (config.get("all_indices") or {}).get("template") or {}
    if "model" in template:
        found["all_indices/template"] = template["model"]
    for score in config.get("scores") or []:
        if "model" in score:
            found[f"scores/{score.get('id')}"] = score["model"]
    return found


def cutoff_rules(config: dict) -> dict[str, str]:
    found = {}
    template = (config.get("all_indices") or {}).get("template") or {}
    if "cutoff_rule" in template:
        found["all_indices/template"] = template["cutoff_rule"]
    for score in config.get("scores") or []:
        if "cutoff_rule" in score:
            found[f"scores/{score.get('id')}"] = score["cutoff_rule"]
    return found


def set_clinical_model(path: Path, stamp: str, target: str = "J_OA_OWN_L") -> None:
    """Point exactly those keys at OPERA; refuse if anything else would change."""
    import yaml

    text = path.read_text()
    before = yaml.safe_load(text)
    # The model keys sit at four spaces (inside all_indices and the score list);
    # comparators and every other key are left alone.
    new = re.sub(r"(?m)^( +model:[ \t]*)[A-Za-z0-9_]+\b", rf"\g<1>{target}", text)
    # The primary grouped comparison is size-matched on the same patients (lock).
    new = re.sub(r"(?m)^(    cutoff_rule:\s*)risk_matched_low\b", r"\g<1>cohort_size_matched", new)
    after = yaml.safe_load(new)
    rules = cutoff_rules(after)
    if any(rule != "cohort_size_matched" for rule in rules.values()):
        raise SystemExit(f"Clinical-score cutoff rules not size-matched: {rules}")
    expected = {k: target for k in clinical_models(before)}
    if clinical_models(after) != expected:
        raise SystemExit(f"Unexpected clinical-score model keys: {clinical_models(after)}")
    rest_before = {k: v for k, v in before.items() if k not in ("all_indices", "scores")}
    rest_after = {k: v for k, v in after.items() if k not in ("all_indices", "scores")}
    if rest_before != rest_after:
        raise SystemExit("The edit would change keys other than the compared model; not written")
    for key, value in clinical_models(before).items():
        print(f"  {path}: {key}.model {value} -> {expected[key]}")
    for key, value in cutoff_rules(before).items():
        print(f"  {path}: {key}.cutoff_rule {value} -> {rules[key]}")
    if new != text:
        shutil.copy(path, path.with_name(f"{path.name}.bak_{stamp}"))
        path.write_text(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, default=16)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="J_OA_OWN_L", help="Tuning-selected primary alias, e.g. R_E or R_EA")
    parser.add_argument("--clinical-config", type=Path,
                        help="Explicit clinical release recipe to wire into the environment")
    parser.add_argument("--clinical-source-analysis", type=Path,
                        help="Completed clinical workspace analysis.yaml, with prepared metadata and training threshold artifacts")
    parser.add_argument("--split", choices=["tuning", "held_out"], default="tuning",
                        help="Configure tuning rehearsal by default; held_out only after lock")
    args = parser.parse_args()
    bundle, rows = bundle_runs(ROOT / "bundles" / f"stage{args.stage}")
    row, config_path = selected_fit(bundle, rows, args.model)
    checkpoint = (ROOT / "neural_runs" / row["run_directory"] / "best.ckpt").resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Missing {checkpoint}")
    clinical_settings = {}
    if args.clinical_config:
        if not args.clinical_config.is_file():
            raise SystemExit(f"Missing clinical recipe: {args.clinical_config}")
        clinical_settings['OPERA_PAPER_CLINICAL_SCORE_CONFIG'] = str(args.clinical_config.resolve())
    if args.clinical_source_analysis:
        import yaml
        source = yaml.safe_load(args.clinical_source_analysis.read_text())
        metadata = Path(source['patient_metadata'])
        thresholds = source.get('clinical_threshold_artifacts') or []
        if not metadata.is_file() or not any(
                a.get('model') == args.model and a.get('split') == 'train'
                and int(a.get('seed', 42)) == 42 and Path(a['path']).is_file()
                for a in thresholds):
            raise SystemExit('Clinical source needs existing prepared metadata and registered training threshold artifacts')
        clinical_settings.update(OPERA_PAPER_PATIENT_METADATA=str(metadata.resolve()),
            OPERA_PAPER_CLINICAL_SOURCE_ANALYSIS=str(args.clinical_source_analysis.resolve()))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(args.env_file, args.env_file.with_name(f"{args.env_file.name}.bak_{stamp}"))
    print(f"{args.model} seed 42: {row['name']} -> {checkpoint}")
    set_keys(args.env_file, {
        **clinical_settings,
        "OPERA_PAPER_ANALYSIS_ESTIMAND": "cif_ipcw",
        "OPERA_PAPER_OPERA_MODEL": args.model,
        "OPERA_PAPER_ANALYSIS_HORIZONS": "30,60,90,180,365,730",
        "OPERA_PAPER_TABULAR_HORIZONS": "30,60,90,180,365,730",
        "OPERA_PAPER_INVENTORY_SPLITS": "tuning,held_out" if args.split == "held_out" else "tuning",
        "OPERA_PAPER_PLOT_SPLIT": args.split,
        "OPERA_PAPER_ROUTED_MODEL": "R_AB_L",
        "OPERA_PAPER_ATTRIBUTION_CHECKPOINT": str(checkpoint),
        "OPERA_PAPER_ATTRIBUTION_CONFIG_DIR": str(config_path.parent.resolve()),
        "OPERA_PAPER_ATTRIBUTION_CONFIG_NAME": config_path.stem,
        "OPERA_PAPER_TASK_STRUCTURE_CHECKPOINT": str(checkpoint),
    })
    clinical = env_value(args.env_file, "OPERA_PAPER_CLINICAL_SCORE_CONFIG")
    if clinical and Path(clinical).is_file():
        set_clinical_model(Path(clinical), stamp, args.model)
    else:
        print(f"  clinical-score configuration not found: {clinical!r}")
    print("done; backups end in .bak_" + stamp)


if __name__ == "__main__":
    main()
