"""Trace router domain effects back to per-cell gains and saved source summaries.

Only aggregate cell information is written; patient-level data are not loaded.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def diagnose(root, seed=42):
    root = Path(root)
    cells = pd.read_csv(root / 'figure_inputs/plot_tables/paired_cells.csv')
    cells = cells.loc[(cells.metric == 'auc') & cells.contrast.isin(['R1', 'OPERA-J'])].copy()
    # OPERA-J and R1 can alias the same model pair; never count both.
    aliases_match = None
    if {'R1', 'OPERA-J'} <= set(cells.contrast):
        keys = ['cohort', 'outcome', 'horizon_days']
        joined = cells.loc[cells.contrast == 'R1', keys + ['gain']].merge(
            cells.loc[cells.contrast == 'OPERA-J', keys + ['gain']], on=keys,
            how='outer', suffixes=('_router', '_opera'), indicator=True, validate='one_to_one')
        aliases_match = bool(joined._merge.eq('both').all() and np.allclose(
            joined.gain_router, joined.gain_opera, equal_nan=True, atol=1e-12, rtol=0))
    contrast = 'R1' if (cells.contrast == 'R1').any() else 'OPERA-J'
    cells = cells.loc[cells.contrast == contrast]
    cells['horizon_days'] = pd.to_numeric(cells.horizon_days)
    if cells.duplicated(['cohort', 'outcome', 'horizon_days']).any():
        raise ValueError('Duplicate router cell identities; select exactly one seed and split')
    path = root / 'prediction_analysis/effect_summaries.csv'
    source = pd.read_csv(path)
    source = source.loc[(source.metric == 'auc') & source.contrast.eq('R1')
                        & pd.to_numeric(source.seed).eq(seed) & source.stratum.eq('endpoint_group')]
    rows = []
    for (domain, horizon), part in cells.groupby(['endpoint_group', 'horizon_days']):
        finite = part.loc[np.isfinite(part.gain)]
        if finite.empty:
            continue
        row = dict(domain=domain, horizon_days=int(horizon), n_cells=len(finite),
            n_diseases=finite.cohort.nunique(), mean=float(finite.gain.mean()),
            median=float(finite.gain.median()),
            disease_balanced_mean=float(finite.groupby('cohort').gain.mean().mean()),
            minimum=float(finite.gain.min()), maximum=float(finite.gain.max()),
            fraction_improved=float((finite.gain > 0).mean()))
        if {'n_cases', 'n_controls'} <= set(finite):
            row['minimum_cases'] = float(finite.n_cases.min())
            row['minimum_controls'] = float(finite.n_controls.min())
        expected = source.loc[(source.stratum_value == domain) & (source.horizon_days == horizon)]
        if len(expected) == 1:
            row['source_mean'] = float(expected.gain.iloc[0])
            # At h=0 the plot cells are pooled disease-outcome estimates, whereas
            # the source summary weights disease-outcome-horizon cells. Distinct estimands.
            if horizon > 0:
                row['source_matches_cell_mean'] = bool(np.isclose(row['mean'], row['source_mean'], atol=1e-10))
        if 'group_mean' in part:
            exported = part.group_mean.dropna().unique()
            row['unique_exported_means'] = exported.tolist()
            if horizon > 0 and len(exported) == 1:
                row['export_matches_cell_mean'] = bool(np.isclose(row['mean'], exported[0], atol=1e-10))
        keep = [c for c in ['cohort', 'outcome', 'gain', 'n_cases', 'n_controls', 'n_events_train'] if c in finite]
        row['lowest_cells'] = finite.nsmallest(5, 'gain')[keep].to_dict('records')
        row['highest_cells'] = finite.nlargest(5, 'gain')[keep].to_dict('records')
        rows.append(row)
    return dict(seed=seed, plot_contrast=contrast, opera_and_router_aliases_match=aliases_match, domains=rows,
        note='These are tuning/evaluation cell effects, not proof of prediction parity. '
             'h=0 plot cells and all-horizon source summaries use different weighting units.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.analysis_root, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=lambda x: x.item()))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
