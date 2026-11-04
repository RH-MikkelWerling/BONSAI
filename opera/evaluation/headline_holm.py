"""Fixed eight-member headline family; clinical comparisons stay descriptive."""
from __future__ import annotations

import numpy as np
import pandas as pd

HEADLINE_FAMILY = ('P1', 'P2', 'P5', 'T2', 'O1', 'router_borrow_AB', 'E1', 'O2')


def headline_holm(summaries, *, splits=('held_out', 'tuning'), bootstrap_method='multiplier'):
    """Adjust primary equal-cell, all-horizon AUC, seed 42 only.

    Missing planned hypotheses occupy their slots with p=1 for adjustment;
    their own adjusted values remain missing. Available results are labelled
    provisional until all eight exist. Tuning is always a rehearsal, not
    confirmatory evidence. Bayesian sign tails are not frequentist p-values.
    """
    rows = []
    for split in splits:
        eligible = summaries.loc[
            summaries.split.eq(split) & summaries.seed.eq(42)
            & summaries.metric.eq('auc') & summaries.horizon_days.eq(0)
            & summaries.stratum.eq('all_cells')
            & summaries.aggregation.eq('equal_cell_not_population_or_endpoint_group_macro')
            & summaries.contrast.isin(HEADLINE_FAMILY)]
        if eligible.contrast.duplicated().any():
            raise ValueError('Duplicate primary headline summary')
        by_code = eligible.set_index('contrast').to_dict('index')
        block = []
        for code in HEADLINE_FAMILY:
            source = by_code.get(code, {})
            raw = source.get('p_value', np.nan)
            finite = np.isfinite(raw)
            if finite and not 0 <= raw <= 1:
                raise ValueError('Headline p-value outside [0, 1]')
            status = 'available' if finite else 'missing_or_not_estimable'
            if bootstrap_method == 'bayesian':
                status = 'not_frequentist_p_value'
            # Finite simulation cannot establish an exactly zero tail probability.
            # Retain the raw value, explicitly report the resolution floor.
            n = source.get('n_valid_bootstrap', source.get('n_bootstrap_replicates', np.nan))
            floor = 2 / (n + 1) if np.isfinite(n) and n > 0 else 0.
            p = max(raw, floor) if finite else np.nan
            if status != 'available':
                p = np.nan
            block.append(dict(split=split, seed=42, contrast=code, metric='auc', horizon_days=0,
                aggregation='equal_cell', gain=source.get('gain', np.nan),
                lower=source.get('lower', np.nan), upper=source.get('upper', np.nan),
                p_value_raw=raw, p_value_for_holm=p, monte_carlo_floor=floor,
                status=status, family_size=len(HEADLINE_FAMILY), alpha=.05,
                bootstrap_method=bootstrap_method, inference='approximate bootstrap sign-tail test',
                analysis_role='confirmatory' if split == 'held_out' else 'tuning_rehearsal'))
        valid = np.array([r['status'] == 'available' for r in block])
        p = np.array([r['p_value_for_holm'] if ok else 1. for r, ok in zip(block, valid)])
        order = np.argsort(p, kind='stable')
        adjusted = np.empty(len(p))
        adjusted[order] = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
        for i, row in enumerate(block):
            row.update(p_value_holm=float(adjusted[i]) if valid[i] else np.nan,
                reject_holm=bool(valid[i] and adjusted[i] <= .05),
                family_complete=bool(valid.all()),
                family_status='complete' if valid.all() else 'provisional_missing_planned_members')
        rows.extend(block)
    return pd.DataFrame(rows)


def main():
    """Add the table to an existing analysis without prediction scoring."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effect-summaries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-method', choices=['multiplier', 'multinomial', 'bayesian'], required=True)
    args = parser.parse_args()
    result = headline_holm(pd.read_csv(args.effect_summaries), bootstrap_method=args.bootstrap_method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Wrote {args.output}; no prediction scoring or bootstrap resampling performed.')


if __name__ == '__main__':
    main()
