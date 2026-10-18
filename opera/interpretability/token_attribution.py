"""Integrated gradients for the direct competing-risk models, per event.

What is attributed
    The *content* of each event (which diagnosis, drug, laboratory result and
    value bin it was). Its time, the patient's age at it, its segment and the
    order of events are held fixed. The baseline replaces every event's content
    by an average event (the mean content embedding of training events), so

        f(input) - f(baseline) = sum of event attributions        (completeness)

    where ``f(baseline)`` is the risk the model gives a patient with the same
    timeline of care but reference event embeddings. The encoder was pretrained on
    next-event prediction, so ``[MASK]`` was never trained and is not used as a
    baseline.

What is explained
    The routed prediction: the target runs through the encoder, the router
    (whose weights depend on the patient) and the hazard head to the cumulative
    incidence at one horizon. The fine diagnosis that selects the home adapters
    is a conditioning input, not a token, and is not attributed.

Everything here works on padded batches of independent patients (dropout off,
per-token normalisation, causal attention with right padding), so summing the
targets of several patients gives each patient's own gradient.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager

import torch

NON_ATTRIBUTABLE = ("[PAD]", "[CLS]", "[SEP]", "[MASK]")


def non_attributable_ids(vocabulary: Mapping[str, int]) -> list[int]:
    """Structural tokens whose content stays fixed (padding, separators, prediction)."""
    return sorted(
        int(vocabulary[name]) for name in NON_ATTRIBUTABLE if name in vocabulary
    )


def attributable_mask(
    batch: Mapping[str, torch.Tensor], fixed_ids: Sequence[int]
) -> torch.Tensor:
    """(B, L) True for real events (including background tokens and [UNK])."""
    code = batch["code"]
    fixed = torch.isin(
        code, torch.as_tensor(list(fixed_ids), dtype=code.dtype, device=code.device)
    )
    return batch["attention_mask"].bool() & ~fixed


def _embedding_inputs(batch):
    return dict(
        code=batch["code"],
        age=batch["age"],
        abspos=batch["abspos"],
        segment=batch["segment"],
        value_bin=batch.get("value_bin"),
        value_normalized=batch.get("value_normalized"),
        value_present=batch.get("value_present"),
        numeric_value=batch.get("numeric_value"),
    )


def content_embeddings(embeddings, batch) -> torch.Tensor:
    """The part of each token's input embedding that says what the event was.

    Code embedding, with its value folded in (FiLM or legacy value bins) when
    the model reads values that way. Time, age and segment are excluded.
    """
    if getattr(embeddings, "value_embedding_mode", "legacy") == "combined_binning":
        raise ValueError(
            "combined_binning replaces whole token embeddings; not supported"
        )
    content = embeddings.code_embedding(batch["code"])
    numeric = batch.get("numeric_value")
    if numeric is not None and embeddings.continuous_value_embedding is not None:
        content = embeddings.continuous_value_embedding(numeric, content)
    value_bin, value_normalized = batch.get("value_bin"), batch.get("value_normalized")
    if (
        value_bin is not None or value_normalized is not None
    ) and embeddings.value_bin_embedding is not None:
        if value_bin is None:
            value_bin = torch.zeros_like(batch["code"])
        if value_normalized is None:
            value_normalized = torch.zeros_like(batch["age"])
        present = batch.get("value_present")
        present = (value_bin != 0) if present is None else present
        value = embeddings.value_bin_embedding(value_bin.long())
        value = value + embeddings.value_projection(
            torch.nan_to_num(value_normalized.float()).unsqueeze(-1)
        )
        content = content + value * present.bool().unsqueeze(-1).to(content.dtype)
    return content


def split_embeddings(encoder, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """(full, content) input embeddings; ``full - content`` is the fixed context."""
    with torch.no_grad():
        full = encoder.embeddings(**_embedding_inputs(batch)).float()
        content = content_embeddings(encoder.embeddings, batch).float()
    return full, content


@torch.no_grad()
def mean_content(encoder, batches, fixed_ids, *, groups: Callable | None = None):
    """Mean content embedding over the events of ``batches`` (an average event).

    With ``groups`` (a function from a code tensor to integer group ids), also
    return one mean per group, for a per-source baseline.
    """
    total, count, by_group = None, 0, {}
    for batch in batches:
        mask = attributable_mask(batch, fixed_ids)
        _, content = split_embeddings(encoder, batch)
        rows = content[mask]
        total = rows.sum(0) if total is None else total + rows.sum(0)
        count += len(rows)
        if groups is not None:
            ids = groups(batch["code"])[mask]
            for group in ids.unique().tolist():
                part = rows[ids == group]
                old = by_group.get(group, (0.0, 0))
                by_group[group] = (old[0] + part.sum(0), old[1] + len(part))
    if not count:
        raise ValueError("No events to average for the baseline")
    mean = total / count
    if groups is None:
        return mean
    return mean, {group: value / n for group, (value, n) in by_group.items()}


def baseline_tensor(content, mask, baseline, *, codes=None, groups=None):
    """Per-token baseline content: a vector, a per-group mapping, or zeros."""
    if isinstance(baseline, str):
        if baseline != "zero":
            raise ValueError("A string baseline must be 'zero'")
        values = torch.zeros_like(content)
    elif isinstance(baseline, Mapping):
        if codes is None or groups is None:
            raise ValueError("A per-group baseline needs codes and a group function")
        ids = groups(codes)
        fallback = baseline.get(-1)
        values = torch.zeros_like(content)
        for group in ids.unique().tolist():
            vector = baseline.get(group, fallback)
            if vector is None:
                raise ValueError(f"No baseline for event group {group}")
            values[ids == group] = vector.to(content)
    else:
        values = baseline.to(content).expand_as(content).clone()
    # Fixed tokens keep their own content: they are never interpolated.
    return torch.where(mask.unsqueeze(-1), values, content)


@torch.no_grad()
def reference_embeddings(encoder, batch, reference, fixed_ids):
    """Input embeddings with every event's content replaced by ``reference``.

    The same timeline (times, age, order, segments, subtype), with average
    events: the integrated-gradients baseline, and the input of the
    timeline-only evaluation.
    """
    mask = attributable_mask(batch, fixed_ids)
    full, content = split_embeddings(encoder, batch)
    return full - (content - reference.to(content)) * mask.unsqueeze(-1)


def _repeat(batch, times):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            out[key] = value.repeat(times, *([1] * (value.ndim - 1)))
        else:
            out[key] = value
    return out


@contextmanager
def gradients_through_encoder(model):
    """Frozen-encoder models run the encoder without autograd; lift that here."""
    previous = getattr(model, "freeze_encoder", False)
    model.freeze_encoder = False
    try:
        yield
    finally:
        model.freeze_encoder = previous


def _nodes(steps, rule, *, dtype, device):
    """Integration nodes and weights on [0, 1]."""
    if rule == "midpoint":
        nodes = (torch.arange(steps, dtype=torch.float64) + 0.5) / steps
        weights = torch.full((steps,), 1.0 / steps, dtype=torch.float64)
    elif rule == "gauss":
        import numpy as np

        x, w = np.polynomial.legendre.leggauss(steps)
        nodes, weights = torch.as_tensor((x + 1) / 2), torch.as_tensor(w / 2)
    else:
        raise ValueError("rule must be 'gauss' or 'midpoint'")
    return nodes.to(device=device, dtype=dtype), weights.to(device=device, dtype=dtype)


def _target_gradients(values, points, *, batched, group):
    """Gradient of every target column w.r.t. ``points``: ``(T, N, L, H)``.

    With ``batched``, the columns share one vectorised backward pass (same
    arithmetic as one pass per column, far better use of the GPU for a small
    model), in groups of ``group`` columns to bound memory. Falls back to one
    pass per column if the backend cannot vectorise the backward.
    """
    count = values.shape[1]
    if batched:
        try:
            parts = []
            for begin in range(0, count, group):
                stop = min(begin + group, count)
                eye = torch.zeros(stop - begin, *values.shape, dtype=values.dtype,
                                  device=values.device)
                for i in range(stop - begin):
                    eye[i, :, begin + i] = 1
                (grad,) = torch.autograd.grad(values, points, grad_outputs=eye,
                                              is_grads_batched=True,
                                              retain_graph=stop < count)
                parts.append(grad)
            return torch.cat(parts)
        except RuntimeError:
            pass  # fall back to one backward pass per column
    return torch.stack([torch.autograd.grad(values[:, column].sum(), points,
                                            retain_graph=column < count - 1)[0]
                        for column in range(count)])


def integrated_gradients(
    target,
    batch,
    full,
    content,
    baseline,
    mask,
    *,
    steps: int = 8,
    chunk: int | None = None,
    rule: str = "gauss",
    batched: bool = True,
    group: int = 24,
):
    """Event-content attributions for every patient and target column.

    ``target(batch_with_token_embeddings)`` returns ``(N, T)``. The path runs
    from ``context + baseline`` to ``full``. The path integral is computed by
    Gauss-Legendre quadrature (``rule="gauss"``, 8 nodes by default), which on
    the smooth paths of this model matches the accuracy of many more midpoint
    steps; ``rule="midpoint"`` keeps the classical Riemann sum. The caller
    checks completeness and refines the nodes where it fails. All target columns
    share each forward pass and, with ``batched``, one vectorised backward.
    Returns ``attributions (B, T, L)``, ``f_input (B, T)`` and ``f_baseline
    (B, T)``; fixed and padding tokens get exactly zero.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    size = full.shape[0]
    delta = (content - baseline) * mask.unsqueeze(-1)
    start = full - delta
    alphas, weights = _nodes(steps, rule, dtype=full.dtype, device=full.device)
    chunk = steps if chunk is None else max(1, int(chunk))
    total = None
    for begin in range(0, steps, chunk):
        part = alphas[begin : begin + chunk]
        weight = weights[begin : begin + chunk]
        points = start.unsqueeze(0) + part[:, None, None, None] * delta.unsqueeze(0)
        points = points.reshape(-1, *full.shape[1:]).detach().requires_grad_(True)
        local = _repeat(batch, len(part))
        local["token_embeddings"] = points
        values = target(local)
        grads = _target_gradients(values, points, batched=batched, group=group)
        grads = grads.reshape(values.shape[1], len(part), size, *full.shape[1:])
        summed = (grads * weight[None, :, None, None, None]).sum(1).transpose(0, 1)
        total = summed if total is None else total + summed
    attributions = (total * delta.unsqueeze(1)).sum(-1)
    with torch.no_grad():
        f_input = target({**batch, "token_embeddings": full})
        f_baseline = target({**batch, "token_embeddings": start})
    return attributions.detach(), f_input.detach(), f_baseline.detach()


@torch.no_grad()
def replaced_target(target, batch, full, content, baseline, replace):
    """Target after replacing selected events' content by the baseline.

    ``replace`` is ``(V, B, L)`` boolean: V variants of the same batch, run as
    one stacked forward pass. Returns ``(V, B, T)``.
    """
    variants = replace.shape[0]
    delta = (content - baseline).unsqueeze(0) * replace.unsqueeze(-1)
    points = (full.unsqueeze(0) - delta).reshape(-1, *full.shape[1:])
    local = _repeat(batch, variants)
    local["token_embeddings"] = points
    values = target(local)
    return values.reshape(variants, full.shape[0], -1)


def event_ids(batch, mask) -> torch.Tensor:
    """(B, L) event index per token: attributable tokens sharing a time; -1 elsewhere."""
    abspos = batch["abspos"]
    ids = torch.full_like(batch["code"], -1)
    for row in range(len(mask)):
        positions = torch.nonzero(mask[row]).flatten()
        times = abspos[row, positions]
        if not len(times):
            continue
        changes = torch.ones_like(times, dtype=torch.long)
        changes[1:] = (times[1:] != times[:-1]).long()
        ids[row, positions] = changes.cumsum(0) - 1
    return ids


def risk_target(
    model,
    outcomes: Sequence[str],
    horizons_days: Sequence[float] | float,
    *,
    removed_sources: Sequence[str] = (),
):
    """Predicted cumulative incidence: ``(N, outcomes x horizons)``, outcome-major."""
    from opera.modules.networks.competing_risk import piecewise_exponential_cif

    objective = model.competing_risk_loss
    indices = [objective.outcome_index[name] for name in outcomes]
    horizons = [
        float(h)
        for h in (
            horizons_days if isinstance(horizons_days, Sequence) else [horizons_days]
        )
    ]
    no_competing = torch.tensor(
        [name in objective.no_competing_outcomes for name in outcomes]
    )

    def target(batch):
        representations = model.get_objective_representations(batch)
        hazards = model.competing_risk_log_hazards(
            representations,
            batch.get("cohort_fine_index"),
            removed_sources=removed_sources,
        )
        cif = piecewise_exponential_cif(
            hazards[:, indices].float(),
            objective.boundaries * objective.time_scale_days,
            horizons,
            time_scale_days=objective.time_scale_days,
            no_competing_outcome_mask=no_competing.to(hazards.device),
        )
        return cif.reshape(len(cif), -1)

    return target


def borrowed_weight_target(model, outcomes: Sequence[str]):
    """Router weight a prediction puts on borrowed sources (not shared, not home)."""
    readout = model.source_adapter_readout
    if readout is None:
        raise ValueError("Borrowed-weight attribution needs a router checkpoint")
    objective = model.competing_risk_loss
    indices = [objective.outcome_index[name] for name in outcomes]

    def target(batch):
        representations = batch.get("_representations")
        if representations is None:
            representations = model.get_objective_representations(batch)
        fine = batch["cohort_fine_index"].long().reshape(-1)
        weights = readout.route_weights(representations, fine)[
            :, indices
        ]  # (N, T, 1+S)
        rows = torch.arange(len(fine), device=fine.device)[:, None]
        cols = torch.arange(len(indices), device=fine.device)[None, :]
        home_disease = weights[rows, cols, (1 + fine.clamp_min(0))[:, None]]
        home_disease = home_disease * (fine >= 0)[:, None]
        own = 1 + readout.n_disease + torch.tensor(indices, device=fine.device)
        home_outcome = weights[:, torch.arange(len(indices), device=fine.device), own]
        return 1 - weights[..., 0] - home_disease - home_outcome

    return target


def combined_target(model, columns: Sequence[tuple[str, str, float | None]]):
    """One target over named columns ``(kind, outcome, horizon)``, sharing one forward pass.

    ``kind`` is ``risk`` (cumulative incidence at ``horizon`` days, so credit
    is in absolute risk), ``logit`` (its log-odds, so credit is a change in
    log-odds and comparable across rare and common outcomes and across low- and
    high-risk patients) or ``borrowed`` (router weight on borrowed sources, no
    horizon). Every column
    costs one backward pass in ``integrated_gradients``; the encoder forward
    along the path is shared by all of them.
    """
    unknown = {kind for kind, _, _ in columns} - {"risk", "logit", "borrowed"}
    if unknown:
        raise ValueError(f"Unknown target kinds: {sorted(unknown)}")
    risk = list(dict.fromkeys((o, float(h)) for kind, o, h in columns
                              if kind in {"risk", "logit"}))
    borrowed = [o for kind, o, _ in columns if kind == "borrowed"]
    outcomes = list(dict.fromkeys(o for o, _ in risk))
    horizons = sorted({float(h) for _, h in risk})
    order = {
        (o, h): i * len(horizons) + j
        for i, o in enumerate(outcomes)
        for j, h in enumerate(horizons)
    }
    objective = model.competing_risk_loss
    no_competing = torch.tensor(
        [name in objective.no_competing_outcomes for name in outcomes]
    )
    risk_index = [objective.outcome_index[name] for name in outcomes]
    borrowed_fn = borrowed_weight_target(model, borrowed) if borrowed else None
    # Risk columns first, then their log-odds, then router weights.
    picks = []
    for kind, o, h in columns:
        if kind == "risk":
            picks.append(order[(o, float(h))])
        elif kind == "logit":
            picks.append(len(order) + order[(o, float(h))])
        else:
            picks.append(2 * len(order) + borrowed.index(o))

    def target(batch):
        from opera.modules.networks.competing_risk import piecewise_exponential_cif

        representations = model.get_objective_representations(batch)
        parts = []
        if risk:
            hazards = model.competing_risk_log_hazards(
                representations, batch.get("cohort_fine_index")
            )
            cif = piecewise_exponential_cif(
                hazards[:, risk_index].float(),
                objective.boundaries * objective.time_scale_days,
                horizons,
                time_scale_days=objective.time_scale_days,
                no_competing_outcome_mask=no_competing.to(hazards.device),
            )
            flat = cif.reshape(len(cif), -1)
            parts += [flat, torch.logit(flat.clamp(1e-7, 1 - 1e-7))]
        if borrowed_fn is not None:
            readout_batch = dict(batch)
            parts.append(_borrowed_from(borrowed_fn, representations, readout_batch))
        values = torch.cat(parts, dim=1)
        return values[:, picks]

    return target


def _borrowed_from(borrowed_fn, representations, batch):
    # Reuse the representation computed for the risk columns.
    return borrowed_fn({**batch, "_representations": representations})


__all__ = [
    "attributable_mask",
    "baseline_tensor",
    "borrowed_weight_target",
    "combined_target",
    "content_embeddings",
    "event_ids",
    "gradients_through_encoder",
    "integrated_gradients",
    "mean_content",
    "non_attributable_ids",
    "reference_embeddings",
    "replaced_target",
    "risk_target",
    "split_embeddings",
]
