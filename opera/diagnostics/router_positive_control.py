"""Positive control for the partial-pooling router (lock amendment 2026-10-30).

Can the router learn to borrow selectively when selective borrowing is what
helps? Synthetic tasks: one outcome, four diseases. Disease A is data-poor
(40 patients) and shares its true risk direction with B; C and D have unrelated
directions. A router that works should, for A's patients, put most borrowed
weight on B and predict A well. The same fit with the R_AB-style readout (own
sources inside the mix at a 0.05 floor) is the comparison.

Run: python -m opera.diagnostics.router_positive_control
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from opera.modules.networks.partial_pooling_readout import PartialPoolingReadout
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout

HIDDEN, TRAIN_SIZES, TEST_SIZE, STEPS, WARMUP = 16, (40, 1500, 1500, 1500), 2000, 3000, 600
SEEDS = (0, 1, 2)


def synthetic(seed, sizes, *, sample_seed=None):
    """Risk directions depend on the problem seed; samples on sample_seed."""
    directions = F.normalize(
        torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(seed)), dim=-1)
    directions[0] = directions[1]  # A behaves like B
    generator = torch.Generator().manual_seed(seed if sample_seed is None else sample_seed)
    xs, diseases, labels = [], [], []
    for disease, n in enumerate(sizes):
        x = torch.randn(n, HIDDEN, generator=generator)
        labels.append(torch.bernoulli(torch.sigmoid(3 * (x @ directions[disease])),
                                      generator=generator))
        xs.append(x)
        diseases.append(torch.full((n,), disease))
    return torch.cat(xs), torch.cat(diseases), torch.cat(labels)


def _auc(labels, scores):
    """Rank AUC without ties handling beyond the average rank."""
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    positive = labels.bool()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit(design, seed):
    """Train one readout and linear head; return A's AUC, A's borrowed weights
    over diseases A..D, and the learned tau (partial pooling only)."""
    torch.manual_seed(seed)
    options = dict(bottleneck=4, global_adapter=True, query_mode="interacting",
                   query_hidden_dim=32, residual_scale=0.7)
    if design == "R_PP":
        readout = PartialPoolingReadout(HIDDEN, ["A", "B", "C", "D"], ["y"], tau_prior=10.0, **options)
    else:
        readout = SourceAnchoredReadout(HIDDEN, ["A", "B", "C", "D"], ["y"], home_floor=0.05, **options)
    x, disease, label = synthetic(seed, TRAIN_SIZES)
    events = [int(label[disease == d].sum()) for d in range(4)]
    readout.set_training_support([*TRAIN_SIZES, sum(TRAIN_SIZES)], [*events, sum(events)],
                                 [1] * 5, event_tau=10.0)
    if design == "R_PP":
        readout.set_cell_support([[e] for e in events])
    head = nn.Linear(HIDDEN, 1)
    optimizer = torch.optim.AdamW([*readout.parameters(), *head.parameters()], lr=3e-3)
    readout.router_warmup_epochs = 1
    readout.train()
    for step in range(STEPS):
        readout.router_frozen = step < WARMUP
        batch = torch.randint(len(x), (256,))
        logit = head(readout(x[batch], disease[batch])[:, 0]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, label[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    readout.eval()
    xt, dt, yt = synthetic(seed, (TEST_SIZE,) * 4, sample_seed=seed + 100)
    with torch.no_grad():
        output, weights = readout(xt, dt, return_weights=True)
        score = head(output[:, 0]).squeeze(-1)
    a = dt == 0
    tau = float(readout.tau.detach()) if design == "R_PP" else None
    return _auc(yt[a], score[a]), weights[a][:, 0, 1:5].mean(0), tau


def run(seeds=SEEDS):
    rows = []
    for design in ("R_PP", "R_AB-style"):
        for seed in seeds:
            auc, borrowed, tau = fit(design, seed)
            rows.append({"design": design, "seed": seed, "auc_A": auc,
                         "weight_A_from": dict(zip("ABCD", borrowed.tolist())), "tau": tau})
    return rows


if __name__ == "__main__":
    for row in run():
        weights = " ".join(f"{k}={v:.2f}" for k, v in row["weight_A_from"].items())
        tau = "" if row["tau"] is None else f"  tau={row['tau']:.1f}"
        print(f"{row['design']:10s} seed {row['seed']}: AUC(A)={row['auc_A']:.3f}  "
              f"borrowed weight for A's patients {weights}{tau}")
