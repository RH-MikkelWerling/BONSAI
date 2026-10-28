"""Audit render freshness, aggregation and coverage using existing server tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def audit(root):
    root = Path(root)
    repo = Path(__file__).resolve().parents[2]
    report = {}
    path = root / 'figures/figure_manifest.json'
    if path.is_file():
        manifest = json.loads(path.read_text())
        report['render_code_matches_checkout'] = {
            name: manifest.get('code_sha256', {}).get(name) == hashlib.sha256((repo / source).read_bytes()).hexdigest()
            for name, source in [('figures', 'opera/visualization/paper_figures.py'),
                                 ('renderer', 'opera/run/render_paper_results.py'),
                                 ('style', 'opera/visualization/style.py')]}
        report['render_skips'] = manifest.get('skipped', {})
    tables = root / 'figure_inputs/plot_tables'
    path = tables / 'paired_cells.csv'
    if path.is_file():
        cells = pd.read_csv(path)
        result = []
        for key, group in cells.groupby(['contrast', 'metric', 'horizon_days']):
            value = pd.to_numeric(group.gain, errors='coerce')
            finite = value[np.isfinite(value)]
            result.append(dict(zip(['contrast', 'metric', 'horizon_days'], key),
                               n_rows=len(group), n_finite=len(finite),
                               mean=float(finite.mean()) if len(finite) else None,
                               median=float(finite.median()) if len(finite) else None,
                               fraction_positive=float((finite > 0).mean()) if len(finite) else None))
        report['plot_cell_summaries'] = result
    for name in ['patient_subgroups_auc', 'coverage', 'route_summaries']:
        path = tables / f'{name}.csv'
        if not path.is_file():
            report[name] = {'missing': True}
            continue
        frame = pd.read_csv(path)
        report[name] = dict(rows=len(frame), columns=list(frame))
        if 'gain' in frame:
            report[name]['finite_gains'] = int(np.isfinite(pd.to_numeric(frame.gain, errors="coerce")).sum())
        if 'contrast' in frame:
            report[name]['contrasts'] = frame.contrast.value_counts().to_dict()
        if 'status' in frame:
            report[name]['statuses'] = frame.status.value_counts().to_dict()
    path = root / 'prediction_analysis/analysis_status.csv'
    if path.is_file():
        report['analysis_status'] = pd.read_csv(path).fillna('').to_dict('records')
    report['stratified_auc_sources'] = {}
    for path in (root / 'prediction_analysis').glob('stratified_*_effects.csv'):
        frame = pd.read_csv(path)
        auc = frame.loc[frame.metric == 'auc']
        report['stratified_auc_sources'][path.name] = dict(
            auc_rows=len(auc), finite_gains=int(np.isfinite(pd.to_numeric(auc.gain, errors="coerce")).sum()),
            contrasts=auc.contrast.value_counts().to_dict())
    path = root / 'clinical_scores/headline_scores.csv'
    if path.is_file():
        frame = pd.read_csv(path)
        report['clinical_headline_labels'] = frame.label.tolist()
    report['borrowing_provenance_files'] = [str(p.relative_to(root))
        for p in (root / 'source_removal').rglob('*provenance*.json')]
    report['interpretation'] = (
        'Plot cell means/medians are descriptive. Compare the saved paired-mean tables for '
        'the declared estimand. Render freshness mismatch requires rerendering, not retraining. '
        'Empty borrowing provenance list does not establish that caches are valid.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.analysis_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
