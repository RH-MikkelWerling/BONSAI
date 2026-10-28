"""Audit render freshness, aggregation and coverage using existing server tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read_csv_audit(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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
        report['rendered_module'] = manifest.get('figure_module')
        report['render_input_mismatches'] = []
        report['render_output_mismatches'] = []
        for plate in manifest.get('figures', []):
            for source, digest in plate.get('input_hashes', {}).items():
                path = Path(source)
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    report['render_input_mismatches'].append({'figure': plate['id'], 'input': source})
            output = root / 'figures' / (plate['id'] + '.pdf')
            if not output.is_file() or hashlib.sha256(output.read_bytes()).hexdigest() != plate.get('output_pdf_sha'):
                report['render_output_mismatches'].append(plate['id'])
    tables = root / 'figure_inputs/plot_tables'
    path = tables / 'paired_cells.csv'
    if path.is_file():
        cells = read_csv_audit(path)
        report['paired_cell_summary_columns'] = [c for c in cells if c.startswith(('panel_', 'group_'))]
        result = []
        required = {'contrast', 'metric', 'horizon_days', 'gain'}
        if not required <= set(cells):
            report['plot_cells_error'] = 'Missing columns: ' + str(sorted(required - set(cells)))
            cells = pd.DataFrame(columns=sorted(required))
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
        frame = read_csv_audit(path)
        report[name] = dict(rows=len(frame), columns=list(frame))
        if 'gain' in frame:
            report[name]['finite_gains'] = int(np.isfinite(pd.to_numeric(frame.gain, errors="coerce")).sum())
        if 'contrast' in frame:
            report[name]['contrasts'] = frame.contrast.value_counts().to_dict()
        if 'status' in frame:
            report[name]['statuses'] = frame.status.value_counts().to_dict()
    path = root / 'prediction_analysis/analysis_status.csv'
    if path.is_file():
        report['analysis_status'] = read_csv_audit(path).fillna('').to_dict('records')
    report['stratified_auc_sources'] = {}
    for path in (root / 'prediction_analysis').glob('stratified_*_effects.csv'):
        frame = read_csv_audit(path)
        required = {'metric', 'gain', 'seed', 'split', 'horizon_days', 'covariate', 'contrast'}
        if not required <= set(frame):
            report['stratified_auc_sources'][path.name] = {'error': 'empty or invalid schema'}
            continue
        auc = frame.loc[frame.metric == 'auc']
        report['stratified_auc_sources'][path.name] = dict(
            auc_rows=len(auc), finite_gains=int(np.isfinite(pd.to_numeric(auc.gain, errors="coerce")).sum()),
            seeds=frame.seed.astype(str).unique().tolist(),
            splits=frame.split.astype(str).unique().tolist(),
            horizons=frame.horizon_days.astype(str).unique().tolist(),
            covariates=frame.covariate.astype(str).unique().tolist(),
            contrasts=auc.contrast.value_counts().to_dict())
    path = root / 'clinical_scores/headline_scores.csv'
    if path.is_file():
        frame = read_csv_audit(path)
        report['clinical_headline_labels'] = frame.label.tolist() if 'label' in frame else []
    env_file = root / 'overnight.env'
    if env_file.is_file():
        from opera.run.paper_experiments import load_environment
        from opera.run.analyse_locked_predictions import read_table
        env = load_environment(env_file)
        source = env.get('OPERA_PAPER_PATIENT_METADATA')
        if source and Path(source).is_file():
            columns = set(read_table(source).columns)
            report['computed_clinical_score_columns'] = {
                label: {'available': {score, group} <= columns,
                        'missing_columns': sorted({score, group} - columns)}
                for label, score, group in [
                    ('IPI', 'ipi_computed_score', 'ipi_computed_group'),
                    ('NCCN-IPI', 'nccn_ipi_score', 'nccn_ipi_group')]}
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
