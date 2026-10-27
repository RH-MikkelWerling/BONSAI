"""Small deterministic checks of routed inference, separate from model selection."""
import numpy as np
import torch
import torch.nn.functional as F

from opera.modules.networks.competing_risk import piecewise_exponential_cif


def audit_subject_subset(dataset, subject_ids):
    """Select before collation/encoding; preserve the production sample builder."""
    indices = [i for i, sample in enumerate(dataset.subjects)
               if str(sample['subject_id']) in subject_ids]
    return torch.utils.data.Subset(dataset, indices)


def component_cifs(model, patient, fine, horizons):
    """Zero only borrowed source terms at unchanged gates; keep own/global terms.

    Diagnostic zero ablation, not a retrained comparator or the primary
    tuning-mean source-removal analysis. No gates are renormalized.
    """
    readout = model.source_adapter_readout
    fine = fine.long().reshape(-1).to(patient.device)
    adapted, weights = readout(patient, fine, return_weights=True)
    values = torch.einsum('bh,srh->bsr', patient, readout.down)
    values = torch.einsum('bsr,shr->bsh', F.gelu(values), readout.up)
    values = values * readout.source_shrinkage[None, :, None]
    borrowed = torch.ones_like(weights[..., 1:], dtype=torch.bool)
    valid = (fine >= 0) & (fine < readout.n_disease)
    rows = torch.nonzero(valid, as_tuple=True)[0]
    borrowed[rows, :, fine[rows]] = False
    outcomes = torch.arange(readout.n_outcome, device=patient.device)
    borrowed[:, outcomes, readout.n_disease + outcomes] = False
    contribution = torch.einsum('bos,bsh->boh', weights[..., 1:] * borrowed, values).float()
    without_borrowing = adapted - readout.residual_scale * contribution.to(adapted.dtype)
    head = model.competing_risk_head
    width = head.weight.shape[0] // readout.n_outcome
    own_flat = torch.einsum('boh,okh->bok', without_borrowing,
                            head.weight.reshape(readout.n_outcome, width, -1))
    own_flat = own_flat + head.bias.reshape(readout.n_outcome, width)[None]
    objective = model.competing_risk_loss
    no_competing = torch.tensor([o in objective.no_competing_outcomes
                                 for o in model.contrastive_loss.outcome_names], device=patient.device)

    def cif(flat):
        return piecewise_exponential_cif(
            flat.reshape(len(patient), readout.n_outcome, 2, objective.n_intervals),
            objective.boundaries * objective.time_scale_days, horizons,
            time_scale_days=objective.time_scale_days,
            no_competing_outcome_mask=no_competing)

    return {'encoder_only': cif(head(patient)), 'no_borrowing_zero': cif(own_flat)}


def compare_saved_predictions(saved, replay, *, atol=1e-5, rtol=1e-4):
    """Full key/follow-up parity and tolerant CIF equality for the replayed rows."""
    keys = ['subject_id', 'cohort', 'outcome']
    saved, replay = saved.copy(), replay.copy()
    for frame in (saved, replay):
        frame['subject_id'] = frame.subject_id.astype(str)
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate patient/cohort/outcome prediction rows')
    selected = saved.loc[saved.subject_id.isin(replay.subject_id)]
    matched = selected.merge(replay, on=keys, how='outer', suffixes=('_saved', '_replay'),
                             validate='one_to_one', indicator=True)
    if not matched['_merge'].eq('both').all():
        raise ValueError('Replay and saved predictions have different patient/outcome coverage')
    for field in ['time_days', 'event']:
        if not np.array_equal(matched[field + '_saved'], matched[field + '_replay']):
            raise ValueError('Replay follow-up differs: ' + field)
    columns = sorted(c for c in saved if c.startswith('cif_') and c.endswith('d'))
    if not columns or any(c not in replay for c in columns):
        raise ValueError('Replay must cover all saved CIF horizons')
    rows = []
    for column in columns:
        before = matched[column + '_saved'].to_numpy(float)
        after = matched[column + '_replay'].to_numpy(float)
        close = np.isclose(before, after, atol=atol, rtol=rtol)
        rows.append({'horizon': column, 'rows': len(matched),
                     'max_abs_difference': float(np.max(np.abs(before - after))),
                     'mismatched_rows': int((~close).sum()), 'passed': bool(close.all())})
    return {'passed': all(r['passed'] for r in rows), 'atol': atol, 'rtol': rtol,
            'patients': int(replay.subject_id.nunique()), 'horizons': rows}
