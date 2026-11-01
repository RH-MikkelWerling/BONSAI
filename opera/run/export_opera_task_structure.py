"""How the new OPERA organises its tasks (aggregates from one checkpoint; CPU).

Reads the selected reading design's checkpoint (outcome attention + own
corrections) and writes, without any patient data:

* ``outcome_query_similarity.csv``: cosine similarity between outcome queries as
  used inside the attention (projected by W_q; heads concatenated): which
  outcomes read the record alike.
* ``outcome_correction_similarity.csv`` / ``disease_correction_similarity.csv``:
  cosine similarity between the effective low-rank maps up @ down of the own
  outcome and own diagnosis corrections: which tasks are adjusted alike.
* ``outcome_specialisation.csv``: per outcome, the attention gate |tanh g|, the
  size of its own correction (Frobenius norm of up @ down), its validation cases
  and endpoint family; ``disease_specialisation.csv`` likewise per diagnosis.
* ``summary.json``: for each similarity, mean similarity within vs between
  endpoint families (outcomes) or lineages (diagnoses), with a permutation p.

    python -m opera.run.export_opera_task_structure --run-dir <neural run dir> \\
        --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _cosine(matrix: torch.Tensor) -> np.ndarray:
    unit = torch.nn.functional.normalize(matrix.double().flatten(1), dim=1)
    return (unit @ unit.T).numpy()


def within_between(similarity: np.ndarray, labels, *, permutations=2000, seed=0) -> dict:
    """Mean off-diagonal similarity within vs between label groups, and the share
    of label permutations with a difference at least as large (one-sided)."""
    labels = np.asarray(labels)
    off = ~np.eye(len(labels), dtype=bool)

    def gap(lab):
        same = (lab[:, None] == lab[None, :]) & off
        diff = (lab[:, None] != lab[None, :])
        return float(similarity[same].mean() - similarity[diff].mean())

    observed = gap(labels)
    rng = np.random.default_rng(seed)
    exceed = sum(gap(rng.permutation(labels)) >= observed for _ in range(permutations))
    same = (labels[:, None] == labels[None, :]) & off
    return {"within": float(similarity[same].mean()),
            "between": float(similarity[labels[:, None] != labels[None, :]].mean()),
            "difference": observed, "p_permutation": (exceed + 1) / (permutations + 1),
            "n": int(len(labels))}


def task_structure(state: dict, outcomes, diseases, *, outcome_family=None, disease_lineage=None,
                   outcome_cases=None, disease_size=None):
    """All tables from a model state dict (keys without the ``model.`` prefix)."""
    readout = {k.split("outcome_attention_readout.", 1)[1]: v.float() for k, v in state.items()
               if "outcome_attention_readout." in k}
    own = {k.split("attention_own_corrections.", 1)[1]: v.float() for k, v in state.items()
           if "attention_own_corrections." in k}
    if "queries" not in readout:
        raise ValueError("Not an outcome-attention checkpoint (no outcome queries)")
    queries = readout["queries"]
    if "query_basis" in readout:
        queries = readout["query_coefficients"] @ readout["query_basis"] + queries
    hidden = queries.shape[1]
    projected = queries @ readout["attention.in_proj_weight"][:hidden].T
    if "attention.in_proj_bias" in readout:
        projected = projected + readout["attention.in_proj_bias"][:hidden]
    tables = {"outcome_query_similarity": pd.DataFrame(_cosine(projected), outcomes, outcomes)}
    gates = torch.tanh(readout["residual_gates"]).abs().numpy()

    def maps(prefix):
        down, up = own[f"{prefix}_down"], own[f"{prefix}_up"]          # (n, r, h), (n, h, r)
        return torch.einsum("nhr,nrk->nhk", up, down)                  # (n, h, h)

    outcome_maps, disease_maps = maps("outcome"), maps("disease")
    tables["outcome_correction_similarity"] = pd.DataFrame(_cosine(outcome_maps), outcomes, outcomes)
    tables["disease_correction_similarity"] = pd.DataFrame(_cosine(disease_maps), diseases, diseases)
    family = pd.Series(outcome_family or {}, dtype=object)
    lineage = pd.Series(disease_lineage or {}, dtype=object)
    tables["outcome_specialisation"] = pd.DataFrame({
        "outcome": outcomes, "gate": gates,
        "correction_size": outcome_maps.flatten(1).norm(dim=1).numpy(),
        "validation_cases": [None if outcome_cases is None else outcome_cases.get(o) for o in outcomes],
        "endpoint_family": [family.get(o) for o in outcomes]})
    tables["disease_specialisation"] = pd.DataFrame({
        "disease": diseases, "correction_size": disease_maps.flatten(1).norm(dim=1).numpy(),
        "size": [None if disease_size is None else disease_size.get(d) for d in diseases],
        "lineage": [lineage.get(d) for d in diseases]})
    summary = {}
    for key, labels, names in (("outcome_query_similarity", family, outcomes),
                               ("outcome_correction_similarity", family, outcomes),
                               ("disease_correction_similarity", lineage, diseases)):
        known = [i for i, n in enumerate(names) if labels.get(n) is not None]
        if len(set(labels.get(names[i]) for i in known)) > 1:
            matrix = tables[key].to_numpy()[np.ix_(known, known)]
            summary[key] = within_between(matrix, [labels[names[i]] for i in known])
    return tables, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.ckpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from opera.visualization.paper_taxonomy import fine_cohort_groups, outcome_endpoint_groups

    checkpoint = torch.load(args.run_dir / args.checkpoint, map_location="cpu", weights_only=False)
    hp = checkpoint["hyper_parameters"]
    outcomes = list(hp["outcome_names"])
    diseases = list(hp["model_init_config"]["competing_risk_config"]["fine_names"])
    state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
    cases = None
    inventory = args.run_dir / "endpoint_selection_inventory.json"
    if inventory.is_file():
        frame = pd.DataFrame(json.loads(inventory.read_text())).query("horizon_days == 365")
        cases = frame.set_index("outcome")["n_cases"].to_dict()
    size = None
    support = args.run_dir / "attention_own_cell_support.csv"
    if support.is_file():
        size = pd.read_csv(support).groupby("disease")["training_events"].sum().to_dict()
    tables, summary = task_structure(state, outcomes, diseases, outcome_family=outcome_endpoint_groups(),
                                     disease_lineage=fine_cohort_groups(), outcome_cases=cases, disease_size=size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=name.endswith("similarity"))
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"checkpoint": str(args.run_dir / args.checkpoint), "epoch": checkpoint.get("epoch"), **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
