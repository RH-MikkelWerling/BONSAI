"""Read-only, INTERNAL outcome support audit and provisional panel decision.

The clinical override file is reviewed separately. Held-out events are reported
but never used to select the primary panel. This script does not edit outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from opera.run.audit_outcome_exports import audit_outcome


REQUIRED = ("subject_id", "split", "index_date", "outcome_date", "censor_date")
SPLITS = ("train", "tuning", "held_out")
BAD_AUDIT_KEYS = (
    "duplicate_subjects", "missing_subjects", "missing_index", "missing_censor",
    "censor_before_index", "event_on_or_before_index", "event_after_censor",
    "invalid_split", "split_index_mismatch",
)
ALLOWED_ACTIONS = {"candidate", "definition_hold", "retire"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rules(path: Path, registry_outcomes: list[str]) -> dict:
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    if rules.get("version") != 1:
        raise ValueError("Decision rubric must have version: 1")
    horizons = rules.get("report_horizons_days")
    primary = rules.get("primary_horizon_days")
    if (not isinstance(horizons, list) or not horizons
            or any(not isinstance(h, int) or h <= 0 for h in horizons)
            or len(set(horizons)) != len(horizons) or primary not in horizons):
        raise ValueError("Require unique positive report horizons including primary horizon")
    for key in ("min_train_events", "min_tuning_events"):
        if not isinstance(rules.get(key), int) or rules[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if rules.get("default_action") not in ALLOWED_ACTIONS:
        raise ValueError("Invalid default_action")
    actions = rules.get("clinical_actions", {})
    if not isinstance(actions, dict):
        raise ValueError("clinical_actions must be a mapping")
    unknown = set(actions) - set(registry_outcomes)
    if unknown:
        raise ValueError(f"Clinical actions not in registry: {sorted(unknown)}")
    for name, spec in actions.items():
        if not isinstance(spec, dict) or spec.get("action") not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid clinical action for {name}")
        if not isinstance(spec.get("reason"), str) or not spec["reason"].strip():
            raise ValueError(f"Clinical action for {name} needs a reason")
    return rules


def support_for_outcome(frame: pd.DataFrame, name: str, horizons: list[int]) -> list[dict]:
    """Count distinct patient-events within horizon, not censoring-adjusted AUC pairs."""
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
        raise ValueError(f"{name}: missing or duplicate subject_id")
    split = frame["split"].astype("string")
    if not split.isin(SPLITS).all():
        raise ValueError(f"{name}: invalid split")
    index = pd.to_datetime(frame["index_date"], errors="coerce")
    censor = pd.to_datetime(frame["censor_date"], errors="coerce")
    event = pd.to_datetime(frame["outcome_date"], errors="coerce")
    if (index.isna().any() or censor.isna().any() or (censor < index).any()
            or (event.notna() & ((event <= index) | (event > censor))).any()):
        raise ValueError(f"{name}: invalid index/event/censor dates")
    rows = []
    for horizon in horizons:
        end = index + pd.to_timedelta(horizon, unit="D")
        event_by_horizon = event.notna() & (event <= end)
        full_followup = censor >= end
        for label in SPLITS:
            in_split = split.eq(label).fillna(False)
            rows.append({
                "outcome": name,
                "split": label,
                "horizon_days": horizon,
                "n_eligible": int(in_split.sum()),
                "n_events": int((in_split & event_by_horizon).sum()),
                "n_full_followup_or_event": int(
                    (in_split & (full_followup | event_by_horizon)).sum()
                ),
                "n_ever_events": int((in_split & event.notna()).sum()),
            })
    return rows


def decide_outcome(name: str, support_rows: pd.DataFrame, rules: dict) -> dict:
    """Use only training/tuning support at the prespecified primary horizon."""
    h = rules["primary_horizon_days"]
    primary = support_rows.loc[support_rows["horizon_days"].eq(h)]
    if set(primary["split"]) != set(SPLITS) or len(primary) != len(SPLITS):
        raise ValueError(f"{name}: incomplete primary-horizon support")
    counts = dict(zip(primary["split"], primary["n_events"]))
    spec = rules["clinical_actions"].get(name, {
        "action": rules["default_action"], "reason": "No prespecified clinical override"
    })
    enough_train = counts["train"] >= rules["min_train_events"]
    enough_tuning = counts["tuning"] >= rules["min_tuning_events"]
    if spec["action"] == "retire":
        disposition = "retire"
        reason = spec["reason"]
    elif spec["action"] == "definition_hold":
        disposition = "definition_hold"
        reason = spec["reason"]
    elif enough_train and enough_tuning:
        disposition = "primary_candidate"
        reason = "Meets prespecified training/tuning event floor; source review still required"
    else:
        disposition = "exploratory_low_support"
        reason = (
            f"Below {rules['min_train_events']} training and/or "
            f"{rules['min_tuning_events']} tuning events by day {h}"
        )
    return {
        "outcome": name,
        "disposition": disposition,
        "reason": reason,
        "clinical_action": spec["action"],
        "clinical_reason": spec["reason"],
        "primary_horizon_days": h,
        "train_events": counts["train"],
        "tuning_events": counts["tuning"],
        "held_out_events_descriptive_only": counts["held_out"],
        "meets_train_floor": enough_train,
        "meets_tuning_floor": enough_tuning,
    }


def threshold_sensitivity(decisions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive robustness grid; clinical holds/retirements stay excluded."""
    candidates = decisions.loc[decisions["clinical_action"].eq("candidate")]
    rows = []
    for train_floor in (50, 100, 200):
        for tuning_floor in (10, 20, 30):
            eligible = (candidates["train_events"] >= train_floor) & (
                candidates["tuning_events"] >= tuning_floor
            )
            rows.append({
                "min_train_events": train_floor,
                "min_tuning_events": tuning_floor,
                "n_primary_candidates": int(eligible.sum()),
                "n_clinically_retired": int(decisions["clinical_action"].eq("retire").sum()),
                "n_definition_holds": int(decisions["clinical_action"].eq("definition_hold").sum()),
            })
    return pd.DataFrame(rows)


def run(outcomes_dir: Path, registry_path: Path, rules_path: Path,
        output_dir: Path) -> dict:
    if not outcomes_dir.is_dir():
        raise ValueError(f"Outcome directory not found: {outcomes_dir}")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; use a fresh path: {output_dir}")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    outcomes = registry["outcomes"]
    if len(outcomes) != len(set(outcomes)):
        raise ValueError("Registry has duplicate outcomes")
    rules = load_rules(rules_path, outcomes)
    missing = [name for name in outcomes if not (outcomes_dir / f"{name}.parquet").is_file()]
    if missing:
        raise ValueError(f"Missing canonical outcome parquets: {missing}")
    extra = sorted(path.stem for path in outcomes_dir.glob("*.parquet")
                   if "__" not in path.stem and path.stem not in outcomes)
    support_rows = []
    audit_rows = []
    file_rows = []
    for name in outcomes:
        path = outcomes_dir / f"{name}.parquet"
        audit = audit_outcome(path)
        if "error" in audit or any(audit.get(key, 0) for key in BAD_AUDIT_KEYS):
            raise ValueError(f"{name}: structural audit failed: {audit}")
        audit_rows.append(audit)
        file_rows.append({"outcome": name, "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
        frame = pd.read_parquet(path, columns=list(REQUIRED))
        support_rows.extend(support_for_outcome(
            frame, name, rules["report_horizons_days"]
        ))
    support = pd.DataFrame(support_rows)
    decisions = pd.DataFrame([
        decide_outcome(name, support.loc[support["outcome"].eq(name)], rules)
        for name in outcomes
    ])
    output_dir.mkdir(parents=True, exist_ok=False)
    support.to_csv(output_dir / "horizon_support_INTERNAL.csv", index=False)
    decisions.to_csv(output_dir / "panel_decisions_INTERNAL.csv", index=False)
    threshold_sensitivity(decisions).to_csv(
        output_dir / "threshold_sensitivity_INTERNAL.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(output_dir / "structure_audit_INTERNAL.csv", index=False)
    pd.DataFrame(file_rows).to_csv(output_dir / "input_files_sha256_INTERNAL.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outcomes_dir": str(outcomes_dir.resolve()),
        "registry": str(registry_path.resolve()),
        "rules": str(rules_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "rules_sha256": sha256_file(rules_path),
        "primary_horizon_days": rules["primary_horizon_days"],
        "min_train_events": rules["min_train_events"],
        "min_tuning_events": rules["min_tuning_events"],
        "n_outcomes": len(outcomes),
        "dispositions": decisions["disposition"].value_counts().to_dict(),
        "extra_canonical_parquets_ignored": extra,
        "held_out_used_for_decisions": False,
        "files_are_internal_not_disclosure_cleared": True,
    }
    (output_dir / "summary_INTERNAL.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path,
                        default=Path("opera/configs/experiment_registry.yaml"))
    parser.add_argument("--rules", type=Path,
                        default=Path("opera/configs/outcome_panel_decision_v1.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.outcomes_dir, args.registry, args.rules, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
