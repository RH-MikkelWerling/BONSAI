"""Frozen readout interventions, with unchanged source functions and risk head.

Head terms add before hazard clipping/integration. Risk and metric differences
do not. Gate substitutions test this fit's reliance, not a retrained comparator.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from opera.modules.networks.competing_risk import piecewise_exponential_cif
from opera.modules.networks.partial_pooling_readout import PartialPoolingReadout
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout


COMPONENTS = ('global', 'own_disease', 'own_outcome', 'borrowed_disease', 'borrowed_outcome')
REMOVALS = {
    'GLOBAL': ('global',),
    'OWN': ('own_disease', 'own_outcome'),
    'BORROWED_DISEASE': ('borrowed_disease',),
    'BORROWED_OUTCOME': ('borrowed_outcome',),
    'ALL_BORROWED': ('borrowed_disease', 'borrowed_outcome'),
}
VARIANTS = {
    **{f'{mode}_{name}': f'{mode.lower()} replacement of {name.lower()} at unchanged gates'
       for name in REMOVALS for mode in ('ZERO', 'MEAN')},
    'ENCODER_HEAD_ONLY': 'Router checkpoint encoder/head without any corrections, not J_L',
    'GLOBAL_ONLY': 'Router checkpoint shared and global paths only, not trained N_A_L',
    'TASK_MEAN_GATES': 'Within-disease/target mean gates, with patient-varying adapter outputs',
    'TASK_MEAN_BORROWED_COMPOSITION': 'Mean relative borrowed mixture, retaining patient borrowing mass and own/null weights',
}


def _substate(state, name):
    """Extract one named module without constructing the encoder."""
    prefixes = {key[:key.index(name + '.') + len(name) + 1]
                for key in state if name + '.' in key}
    if len(prefixes) != 1:
        raise ValueError(f'Need one checkpoint module: {name}')
    prefix, = prefixes
    return {key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)}


class FrozenRouterHead(nn.Module):
    def __init__(self, readout, weight, bias, boundaries, *, time_scale_days=365.25,
                 no_competing_outcomes=()):
        super().__init__()
        pooling = isinstance(readout, PartialPoolingReadout)
        if (readout.own_outside and not pooling) or readout.disease_embedding is not None:
            raise ValueError('This audit requires own sources inside the mixture (or the '
                             'partial-pooling router) and no identity embedding')
        self.readout = readout.eval()
        self.outcomes = tuple(readout.outcome_names)
        self.boundaries = list(boundaries)
        self.time_scale_days = float(time_scale_days)
        width = 2 * (len(self.boundaries) + 1)
        if weight.shape != (len(self.outcomes) * width, readout.hidden_size):
            raise ValueError('Checkpoint head dimensions do not match outcomes/intervals')
        if bias.shape != (len(self.outcomes) * width,):
            raise ValueError('Checkpoint head bias dimensions do not match')
        self.register_buffer('weight', weight.detach().float().reshape(len(self.outcomes), width, -1))
        self.register_buffer('bias', bias.detach().float().reshape(len(self.outcomes), width))
        self.register_buffer('no_competing', torch.tensor(
            [o in no_competing_outcomes for o in self.outcomes], dtype=torch.bool))
        self.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path: Path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        hp = checkpoint['hyper_parameters']
        init = hp['model_init_config']
        risk = init['competing_risk_config']
        outcomes = hp['outcome_names']
        if risk.get('readout_mode') != 'source_anchored_adapters' or risk.get('head_mode', 'linear') != 'linear':
            raise ValueError('Need the anchored router and its linear competing-risk head')
        state = checkpoint['state_dict']
        tensors = _substate(state, 'source_adapter_readout')
        pooling = bool(risk.get('adapter_partial_pooling', False))
        readout_class = PartialPoolingReadout if pooling else SourceAnchoredReadout
        extra = {'tau_prior': float(risk.get('adapter_event_tau', 20.0))} if pooling else {}
        readout = readout_class(
            tensors['down'].shape[-1], risk['fine_names'], outcomes,
            bottleneck=tensors['down'].shape[1],
            residual_scale=float(risk.get('adapter_residual_scale', .1)),
            home_floor=float(risk.get('adapter_home_floor', .1)),
            anchored=True, query_mode=risk.get('adapter_query_mode', 'additive'),
            query_hidden_dim=int(risk.get('adapter_query_hidden_dim', 128)),
            source_mode=risk.get('adapter_source_mode', 'both'),
            borrowing=bool(risk.get('adapter_borrowing', True)),
            disease_embedding=bool(risk.get('adapter_disease_embedding', False)),
            global_adapter=bool(risk.get('adapter_global', False)),
            router_warmup_epochs=int(risk.get('adapter_router_warmup_epochs', 0)),
            logit_scale=float(risk.get('adapter_logit_scale', 0)),
            own_outside=bool(risk.get('adapter_own_outside', False)), **extra)
        readout.load_state_dict(tensors, strict=True)
        head = _substate(state, 'competing_risk_head')
        if set(head) != {'weight', 'bias'}:
            raise ValueError('Need a single linear risk head')
        return cls(readout, head['weight'], head['bias'], risk['interval_boundaries_days'],
                   time_scale_days=risk.get('time_scale_days', 365.25),
                   no_competing_outcomes=risk.get('no_competing_outcomes', ['overall_survival']))

    def borrowed_mask(self, fine, indices):
        r = self.readout
        source = torch.arange(r.n_sources, device=fine.device)
        own_disease = source[None, None] == fine[:, None, None]
        own_outcome = source[None, None] == (r.n_disease + indices)[None, :, None]
        return ~(own_disease | own_outcome)

    def terms(self, patient, fine, indices, *, weights=None):
        """Exact scaled component contributions in unclamped head space."""
        r = self.readout
        if fine.numel() != len(patient) or not bool(((fine >= 0) & (fine < r.n_disease)).all()):
            raise ValueError('All audit patients need known fine-disease identity')
        if weights is None:
            weights = r.route_weights(patient, fine)[:, indices]
        if (weights.shape != (len(patient), len(indices), r.n_sources + 1)
                or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
                or not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-5)):
            raise ValueError('Invalid intervention gates')
        weight = self.weight[indices]
        if isinstance(r, PartialPoolingReadout):
            # Same arithmetic as the readout's forward pass (component_vectors).
            vectors = r.component_vectors(patient, fine, weights, indices)
            terms = {'base': torch.einsum('bh,okh->bok', patient, weight) + self.bias[indices][None]}
            for name in COMPONENTS:
                terms[name] = r.residual_scale * torch.einsum('boh,okh->bok', vectors[name], weight)
            return terms, weights
        values = F.gelu(torch.einsum('bh,srh->bsr', patient, r.down))
        values = torch.einsum('bsr,shr->bsh', values, r.up)
        values = values * r.source_shrinkage[None, :, None]
        source = torch.arange(r.n_sources, device=patient.device)
        disease = source < r.n_disease
        home_d = source[None, None] == fine[:, None, None]
        home_o = source[None, None] == (r.n_disease + indices)[None, :, None]
        masks = {'own_disease': home_d, 'own_outcome': home_o,
                 'borrowed_disease': disease[None, None] & ~home_d,
                 'borrowed_outcome': ~disease[None, None] & ~home_o}
        weight = self.weight[indices]
        terms = {'base': torch.einsum('bh,okh->bok', patient, weight) + self.bias[indices][None]}
        for name, mask in masks.items():
            vector = torch.einsum('bos,bsh->boh', weights[..., 1:] * mask, values)
            terms[name] = r.residual_scale * torch.einsum('boh,okh->bok', vector, weight)
        general = torch.zeros_like(patient) if r.global_down is None else F.linear(
            F.gelu(F.linear(patient, r.global_down)), r.global_up)
        terms['global'] = r.residual_scale * torch.einsum('bh,okh->bok', general, weight)
        return terms, weights

    def cif(self, flat, indices, horizons):
        return piecewise_exponential_cif(
            flat.reshape(len(flat), len(indices), 2, len(self.boundaries) + 1),
            self.boundaries, horizons, time_scale_days=self.time_scale_days,
            no_competing_outcome_mask=self.no_competing[indices])


class TaskReferences:
    """Unlabelled tuning means conditional on disease and target eligibility."""
    def __init__(self, head, n_outcomes):
        r = head.readout
        shape = (r.n_disease, n_outcomes)
        self.count = torch.zeros(shape, dtype=torch.float64)
        self.borrow_count = torch.zeros(shape, dtype=torch.float64)
        self.gates = torch.zeros(*shape, r.n_sources + 1, dtype=torch.float64)
        self.composition = torch.zeros(*shape, r.n_sources, dtype=torch.float64)
        self.terms = {name: torch.zeros(*shape, head.weight.shape[1], dtype=torch.float64)
                      for name in COMPONENTS}
        self.absolute_terms = {name: torch.zeros_like(value) for name, value in self.terms.items()}
        self.finalized = False

    def add(self, fine, eligible, terms, gates, borrowed_mask):
        if self.finalized:
            raise ValueError('References already finalized')
        e = eligible.double()
        self.count.index_add_(0, fine, e)
        self.gates.index_add_(0, fine, gates.double() * e[..., None])
        borrowed = gates[..., 1:] * borrowed_mask
        mass = borrowed.sum(-1, keepdim=True)
        positive = (mass[..., 0] > 0).double() * e
        relative = borrowed / mass.clamp_min(torch.finfo(borrowed.dtype).tiny)
        self.borrow_count.index_add_(0, fine, positive)
        self.composition.index_add_(0, fine, relative.double() * positive[..., None])
        for name in COMPONENTS:
            self.terms[name].index_add_(0, fine, terms[name].double() * e[..., None])
            self.absolute_terms[name].index_add_(0, fine, terms[name].double().abs() * e[..., None])

    def finalize(self):
        if not self.finalized:
            denom = self.count.clamp_min(1)[..., None]
            self.gates /= denom
            self.composition /= self.borrow_count.clamp_min(1)[..., None]
            for name in COMPONENTS:
                self.terms[name] /= denom
                self.absolute_terms[name] /= denom
            self.finalized = True
        return self

    def frozen_composition(self, fine, gates, borrowed_mask):
        if not self.finalized:
            raise ValueError('Finalize references before gate substitution')
        borrowed = gates[..., 1:] * borrowed_mask
        mass = borrowed.sum(-1, keepdim=True)
        relative = self.composition[fine].to(gates.dtype)
        # No reference composition is needed when this patient borrows nothing.
        changed = gates.clone()
        changed[..., 1:] = torch.where(borrowed_mask, mass * relative, gates[..., 1:])
        return changed


def variant_terms(head, patient, fine, indices, reference):
    """Interventions expressed in head space, before clipping/CIF integration."""
    terms, gates = head.terms(patient, fine, indices)
    full = sum(terms.values())
    yield 'FULL', full
    for name, members in REMOVALS.items():
        removed = sum(terms[member] for member in members)
        mean = sum(reference.terms[member][fine].to(full.dtype) for member in members)
        yield f'ZERO_{name}', full - removed
        yield f'MEAN_{name}', full - removed + mean
    yield 'ENCODER_HEAD_ONLY', terms['base']
    yield 'GLOBAL_ONLY', terms['base'] + terms['global']
    # Unobserved disease/target cells are never evaluated; use original gates
    # there to keep intermediate tensors valid, without inventing predictions.
    observed = reference.count[fine] > 0
    average = reference.gates[fine].to(gates.dtype)
    average = torch.where(observed[..., None], average, gates)
    mean_terms, _ = head.terms(patient, fine, indices, weights=average)
    yield 'TASK_MEAN_GATES', sum(mean_terms.values())
    borrowed = head.borrowed_mask(fine, indices)
    fixed = reference.frozen_composition(fine, gates, borrowed)
    # Unobserved target slots have no reference composition either.
    fixed = torch.where(observed[..., None], fixed, gates)
    fixed_terms, _ = head.terms(patient, fine, indices, weights=fixed)
    yield 'TASK_MEAN_BORROWED_COMPOSITION', sum(fixed_terms.values())


# Two-sided health of the borrowed mixture (lock amendment 2026-10-30): not
# collapsed and not uniform, on eligible tuning predictions.
HEALTH_TOP_SHARE, HEALTH_ROW_FRACTION, HEALTH_DISTINCT, HEALTH_ENTROPY_RATIO = 0.5, 0.5, 3, 0.9


@torch.no_grad()
def borrowed_mixture_health(head, patient, fine, eligible, indices, *, batch_size=256):
    """Entropy of the relative borrowed mixture against the uniform maximum over
    the sources available to that prediction, top-source shares, and distinct
    top borrowed sources across diseases."""
    r = head.readout
    ratios, top_shares = [], []
    by_disease = torch.zeros(r.n_disease, r.n_sources, dtype=torch.float64)
    for start in range(0, len(patient), batch_size):
        p, f = patient[start:start + batch_size], fine[start:start + batch_size]
        e = eligible[start:start + batch_size].bool()
        weights = r.route_weights(p, f)[:, indices, 1:].double()
        borrowed = weights * head.borrowed_mask(f, indices)
        mass = borrowed.sum(-1, keepdim=True)
        keep = e & (mass[..., 0] > 0)
        relative = (borrowed / mass.clamp_min(1e-300))[keep]
        available = (relative > 1e-12).sum(-1).double()
        entropy = -(relative * relative.clamp_min(1e-300).log()).sum(-1)
        valid = available > 1
        ratios.append(entropy[valid] / available[valid].log())
        top_shares.append(relative.max(-1).values)
        rows = f[:, None].expand_as(keep)[keep]
        by_disease.index_add_(0, rows, relative)
    ratio, top = torch.cat(ratios), torch.cat(top_shares)
    present = by_disease.sum(-1) > 0
    distinct = int(by_disease[present].argmax(-1).unique().numel())
    median_ratio = float(ratio.median()) if len(ratio) else float('nan')
    collapsed_rows = float((top > HEALTH_TOP_SHARE).double().mean()) if len(top) else float('nan')
    return {
        'n_predictions': int(len(top)),
        'median_entropy_ratio': median_ratio,
        'mean_top_share': float(top.mean()) if len(top) else float('nan'),
        'fraction_rows_top_share_above_half': collapsed_rows,
        'distinct_top_borrowed_sources_across_diseases': distinct,
        'not_collapsed': bool(collapsed_rows <= HEALTH_ROW_FRACTION and distinct >= HEALTH_DISTINCT),
        'not_uniform': bool(median_ratio <= HEALTH_ENTROPY_RATIO),
        'criteria': {'top_share': HEALTH_TOP_SHARE, 'row_fraction': HEALTH_ROW_FRACTION,
                     'distinct_top_sources': HEALTH_DISTINCT, 'entropy_ratio': HEALTH_ENTROPY_RATIO},
    }
