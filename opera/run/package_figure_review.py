"""Package available paper figures and explicitly selected aggregate tables for local review.

No inference or rendering. Unknown files, patient arrays and raw predictions are
excluded. Missing families are recorded rather than blocking a partial bundle.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

PLOT_TABLES = {
    'atlas.csv', 'model_cells.csv', 'paired_cells.csv', 'support_curve.csv',
    'panel_effects.csv', 'external_panel.csv', 'origin_estimates.csv',
    'hierarchy_cells.csv', 'strategy_cells.csv', 'patient_subgroups.csv',
    'gate_effects.csv', 'route_summaries.csv', 'calibration_summary.csv',
    'reliability.csv', 'coverage.csv', 'training_support.csv',
    'region_models.csv', 'region_coverage.csv', 'region_effects.csv', 'route_health.csv',
    'rescue_cells.csv', 'paired_domain_summary.csv', 'outcome_support_summary.csv',
    'patient_subgroups_auc.csv', 'national_subgroups.csv', 'selected_strategy.csv',
}
REPORT_TABLES = {
    'score_inventory.csv', 'headline_scores.csv', 'classification_at_thresholds.csv',
    'classification_at_anchors.csv', 'cutoff_diagnostics.csv', 'score_population_flow.csv',
    'stratification_diagnostics.csv', 'borrowing_map.csv', 'borrowing_levels.csv',
    'borrowing_hypotheses.csv', 'borrowing_hypothesis_test.csv',
    'input_restriction_summary.csv', 'attribution_quality.csv',
    'attribution_variance_sources.csv', 'attribution_variance_windows.csv',
    'attribution_outcome_similarity.csv', 'attribution_concepts.csv',
    'attribution_concept_coverage.csv', 'attribution_lab_response.csv',
    'attribution_top_concepts.csv', 'attribution_faithfulness.csv', 'attribution_sanity.csv',
    'attribution_numbers.json', 'summary_manifest.json', 'layout_report.json',
}
ANALYSIS_TABLES = {
    'effect_summaries.csv', 'effect_summaries_disease_balanced.csv',
    'contrasts.csv', 'contrasts_all_horizons.csv', 'estimates.csv',
}
PATIENT_COLUMNS = {'patient_id', 'subject_id', 'person_id', 'pid', 'cpr', 'pnr'}


def package(review: Path, output: Path, analysis_root: Path | None = None):
    review = review.resolve()
    if not (review / 'review_manifest.json').is_file():
        raise ValueError('Expected an assembled review folder containing review_manifest.json')
    clinical = review / 'clinical_scores'
    if clinical.is_dir() and any(p.suffix.lower() in {'.pdf', '.png'} for p in clinical.rglob('*')):
        from opera.run.clinical_figure_provenance import verify_clinical_render
        check = verify_clinical_render(clinical)
        if not check['valid']:
            raise ValueError('Cannot package stale clinical figures: ' + '; '.join(check['errors'])
                             + '. Refresh clinical reports and assemble a fresh review directory.')
    if output.exists():
        raise FileExistsError(output)
    selected = {}
    excluded = []

    def add(path, member):
        if path.is_symlink() or 'INTERNAL' in path.name.upper():
            excluded.append(str(member))
            return
        if path.suffix == '.csv':
            with path.open(newline='') as handle:
                header = next(csv.reader(handle), [])
            if PATIENT_COLUMNS.intersection(x.strip().lower() for x in header):
                excluded.append(str(member))
                return
        selected[str(member)] = path

    for path in sorted(review.rglob('*')):
        if not path.is_file() or any(part.startswith('.') for part in path.relative_to(review).parts):
            continue
        rel = path.relative_to(review)
        if path.suffix.lower() in {'.pdf', '.png'} or path.name in {
            'review_manifest.json', 'figure_manifest.json', 'report_manifest.json',
            'analysis_provenance.json', 'figure_results_audit.json',
        } or (rel.parts[0] == 'reports' and path.name in REPORT_TABLES):
            add(path, Path('review') / rel)
    if analysis_root:
        for folder, names in [('figure_inputs/plot_tables', PLOT_TABLES),
                              ('prediction_analysis', ANALYSIS_TABLES)]:
            for name in sorted(names):
                path = analysis_root / folder / name
                if path.is_file():
                    add(path, Path('aggregates') / folder / name)
    if analysis_root:
        for name in ('disease_groups.json','outcome_groups.json','calibration_examples.json','patient_covariate_order.json','patient_contrast_styles.json'):
            source = analysis_root / 'figure_inputs/plot_tables' / name
            if source.is_file():
                add(source, Path('aggregates/figure_inputs/plot_tables') / name)
        for name in ('figure_results_audit.json', 'router_domain_diagnostic.json',
                     'router_cell_distribution.json'):
            source = analysis_root / name
            if source.is_file():
                add(source, Path('aggregates') / name)
        for branch in ('source_removal/tuning', 'source_removal/tuning/zero_ablation'):
            for name in ('borrowing_map.csv', 'borrowing_levels.csv',
                         'borrowing_hypotheses.csv', 'analysis_provenance.json'):
                source = analysis_root / branch / name
                if source.is_file():
                    add(source, Path('aggregates') / branch / name)
    report = json.loads((review / 'review_manifest.json').read_text())
    manifest = dict(status=report.get('status', 'unknown'), missing=report.get('missing', []),
                    source_review=str(review), excluded=excluded,
                    files={name: dict(bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for name, p in selected.items()},
                    notes=['Partial bundles are useful for review; missing analyses were not rerun.',
                           'Counts and figures retain their source disclosure-review requirements.',
                           'Only named aggregate tables are included; this is not a full analysis backup.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', compression=ZIP_DEFLATED) as archive:
        for name, path in selected.items():
            archive.write(path, name)
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    return dict(output=str(output), files=len(selected), status=manifest['status'], missing=manifest['missing'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--analysis-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.review_dir, args.output, args.analysis_root), indent=2))


if __name__ == '__main__':
    main()
