"""What did a collapsed router collapse onto? (aggregates only; CPU)

For one routed fit, rebuild every source adapter's correction for a sample of
tuning patients from the checkpoint (correction_s(x) = up_s gelu(down_s x),
scaled by the source's support shrinkage), using the patient representations
the readout saw (``patient_embeddings.npz`` from the fit's tuning evaluation).

Reports:
- each source's mean correction size (after shrinkage), and its routed
  contribution (mean route weight x correction size);
- the hub (the source with most routed contribution) and the sources whose
  corrections point the same way as the hub's (mean cosine over patients);
- the distribution of pairwise cosines between all sources, to tell "every
  adapter learned roughly the same general direction" from "the hub is special".

    python -m opera.run.audit_router_hub --run-dir <neural_runs/...> \\
        --prediction-dir <predictions/.../tuning> --output hub_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def readout_tensors(checkpoint: Path) -> dict:
    """down, up and source_shrinkage of the source readout, whatever the prefix."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    found = {}
    for key, value in state.items():
        for name in ("down", "up", "source_shrinkage"):
            if key.endswith(f"source_adapter_readout.{name}"):
                found[name] = value.float()
    missing = {"down", "up", "source_shrinkage"} - set(found)
    if missing:
        raise ValueError(f"{checkpoint} has no source readout ({sorted(missing)} missing)")
    return found


def source_names(routes: pd.DataFrame) -> list[str]:
    first = routes.iloc[0]
    return [*json.loads(first.disease_weights), *json.loads(first.outcome_weights)]


def mean_route_weights(routes: pd.DataFrame, names: list[str]) -> np.ndarray:
    total = np.zeros(len(names))
    for row in routes.itertuples(index=False):
        weights = {**json.loads(row.disease_weights), **json.loads(row.outcome_weights)}
        total += np.array([weights[name] for name in names])
    return total / len(routes)


def source_corrections(run_dir: Path, prediction_dir: Path, *, max_patients: int = 2000):
    """(names, route rows, readout tensors, corrections) for a sample of tuning
    patients; corrections is (patients, sources, hidden), scaled by shrinkage."""
    tensors = readout_tensors(Path(run_dir) / "best.ckpt")
    routes = pd.read_parquet(Path(prediction_dir) / "source_route_weights.parquet")
    names = source_names(routes)
    if len(names) != tensors["down"].shape[0]:
        raise ValueError("Route export and checkpoint disagree on the number of sources")
    embeddings = np.load(Path(prediction_dir) / "patient_embeddings.npz")["embeddings"]
    rng = np.random.default_rng(0)
    keep = rng.choice(len(embeddings), min(max_patients, len(embeddings)), replace=False)
    patients = torch.as_tensor(embeddings[np.sort(keep)], dtype=torch.float32)
    with torch.no_grad():
        hidden = F.gelu(torch.einsum("bh,srh->bsr", patients, tensors["down"]))
        corrections = torch.einsum("bsr,shr->bsh", hidden, tensors["up"])
        corrections = corrections * tensors["source_shrinkage"][None, :, None]
    return names, routes, tensors, corrections


def correction_sizes(run_dir: Path, prediction_dir: Path, **kwargs) -> dict[str, float]:
    """Each source's mean correction size (after shrinkage) over tuning patients."""
    names, _, _, corrections = source_corrections(run_dir, prediction_dir, **kwargs)
    return dict(zip(names, corrections.norm(dim=-1).mean(0).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="the fit's run directory")
    parser.add_argument("--prediction-dir", type=Path, required=True,
                        help="the fit's tuning evaluation (embeddings and route weights)")
    parser.add_argument("--max-patients", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names, routes, tensors, corrections = source_corrections(
        args.run_dir, args.prediction_dir, max_patients=args.max_patients)
    n_patients = len(corrections)
    size = corrections.norm(dim=-1).mean(0).numpy()  # per source
    weights = mean_route_weights(routes, names)
    contribution = weights * size
    hub = int(contribution.argmax())

    unit = F.normalize(corrections, dim=-1)
    to_hub = (unit * unit[:, hub : hub + 1]).sum(-1).mean(0).numpy()  # per source
    pairwise = torch.einsum("bsh,bth->st", unit, unit).numpy() / n_patients
    upper = pairwise[np.triu_indices(len(names), k=1)]

    # Who borrows from whom: per target disease, the contribution (route weight x
    # the source's mean correction size) of every source other than the patient's
    # own disease and own outcome, as shares of the borrowed total.
    profiles = {}
    for row in routes.itertuples(index=False):
        weights_row = {**json.loads(row.disease_weights), **json.loads(row.outcome_weights)}
        own = {list(json.loads(row.disease_weights))[row.fine_index]}
        own |= {n for n in weights_row if n.split(":")[-1] == str(row.outcome)}
        target = list(json.loads(row.disease_weights))[row.fine_index].split(":")[-1]
        vector = np.array([0.0 if n in own else weights_row[n] for n in names]) * size
        profiles[target] = profiles.get(target, 0) + vector
    shares = {t: v / v.sum() for t, v in profiles.items() if v.sum() > 0}
    targets = sorted(shares)
    matrix = np.array([shares[t] for t in targets])
    similarity = np.corrcoef(matrix) if len(targets) > 1 else np.ones((1, 1))
    off_diagonal = similarity[np.triu_indices(len(targets), k=1)]
    borrowing = {
        "profile_similarity_between_diseases": {
            "median": round(float(np.median(off_diagonal)), 3) if len(off_diagonal) else None,
            "min": round(float(off_diagonal.min()), 3) if len(off_diagonal) else None},
        "top_borrowed_by_target": {
            t: [(names[i], round(float(shares[t][i]), 3)) for i in np.argsort(-shares[t])[:3]]
            for t in targets},
    }

    order = np.argsort(-contribution)
    similar = [i for i in np.argsort(-to_hub) if i != hub][:10]
    report = {
        "run_dir": str(args.run_dir),
        "patients": int(n_patients),
        "hub": {"source": names[hub], "mean_weight": round(float(weights[hub]), 4),
                "correction_size": round(float(size[hub]), 4),
                "share_of_contribution": round(float(contribution[hub] / contribution.sum()), 4)},
        "top_contributors": [
            {"source": names[i], "mean_weight": round(float(weights[i]), 4),
             "correction_size": round(float(size[i]), 4),
             "shrinkage": round(float(tensors["source_shrinkage"][i]), 4)}
            for i in order[:10]],
        "most_like_hub": [{"source": names[i], "mean_cosine": round(float(to_hub[i]), 3)}
                          for i in similar],
        "pairwise_cosine": {q: round(float(np.quantile(upper, p)), 3)
                            for q, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))},
        "correction_size_quantiles": {q: round(float(np.quantile(size, p)), 4)
                                      for q, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))},
        "borrowing": borrowing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
