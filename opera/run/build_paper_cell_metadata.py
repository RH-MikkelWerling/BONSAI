"""Train-only metadata for every disease–outcome cell in the paper panel.

The paired analysis joins this table on (cohort, outcome, horizon_days) to
stratify effects by training support and clinical domain; the figures use the
same columns. Only training and tuning labels are read, so the strata cannot
depend on held-out outcomes.

    python -m opera.run.build_paper_cell_metadata \
        --outcomes-config <approved training YAML> --output <cell_metadata.csv>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from opera.visualization.paper_taxonomy import DISPLAY_GROUPS, outcome_endpoint_groups

HORIZONS = (30, 60, 90, 180, 365, 730)


def build_cell_metadata(registry, training_config, *, horizons=HORIZONS,
                        groups_path=DISPLAY_GROUPS, audit=None):
    """One row per fine cohort × outcome × horizon in the training panel."""
    if audit is None:
        from opera.run.audit_cohort_outcome_support import audit_support as audit
    endpoint_groups = outcome_endpoint_groups(groups_path)
    tables = [
        audit(registry, outcomes=list(training_config["outcomes"]), horizon_days=horizon,
              include_held_out=False, training_config=training_config)
        for horizon in horizons
    ]
    cells = pd.concat(tables, ignore_index=True)
    missing = sorted(set(cells.outcome) - set(endpoint_groups))
    if missing:
        raise ValueError(f"Outcomes without a display group: {missing}")
    cells = cells.rename(columns={"cohort_grouped": "cohort_group"})
    cells["n_events_train"] = cells["n_primary_events_train"].astype(int)
    if "n_survival_train" in cells:
        # Training patients of the disease (head count), for ordering and labels.
        cells["n_patients_train"] = cells["n_survival_train"].astype(int)
    cells["endpoint_group"] = cells.outcome.map(endpoint_groups)
    # The training loss's endpoints: grade variants of one finding share one
    # endpoint (69 outcomes, 52 endpoints); used for the primary panel weights.
    loss_endpoints = {}
    for endpoint, members in (training_config.get("cross_outcome", {})
                              .get("endpoint_groups") or {}).items():
        for outcome in members:
            loss_endpoints.setdefault(outcome, endpoint)
    cells["loss_endpoint"] = cells.outcome.map(lambda o: loss_endpoints.get(o, o))
    if cells.duplicated(["cohort", "outcome", "horizon_days"]).any():
        raise ValueError("Duplicate cohort/outcome/horizon cells")
    leading = ["cohort", "cohort_group", "outcome", "endpoint_group", "loss_endpoint",
               "horizon_days",
               "n_events_train"] + (["n_patients_train"] if "n_patients_train" in cells else [])
    return cells[leading + [c for c in cells.columns if c not in leading]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="opera/configs/experiment_registry.yaml")
    parser.add_argument("--outcomes-config", type=Path, required=True,
                        help="Approved training YAML defining the outcome panel.")
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from opera.run.generate_sweep_configs import load_registry

    registry = load_registry(args.registry)
    training_config = yaml.safe_load(args.outcomes_config.read_text(encoding="utf-8"))
    horizons = tuple(int(h) for h in args.horizons.split(","))
    cells = build_cell_metadata(registry, training_config, horizons=horizons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output, index=False)
    print(json.dumps({"output": str(args.output), "cells": len(cells),
                      "horizons": list(horizons), "labels": "train_and_tuning_only"}))


if __name__ == "__main__":
    main()
