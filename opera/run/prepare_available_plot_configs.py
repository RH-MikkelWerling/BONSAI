"""Prepare server-specific plotting recipes from the audited analysis inventory."""
import argparse
import json
from pathlib import Path
import yaml
from opera.run.analyse_locked_predictions import read_table


GROUP_MAP = {
    'low': 'Low', 'low risk': 'Low', 'low-risk': 'Low',
    'medium': 'Intermediate', 'medium risk': 'Intermediate',
    'intermediate': 'Intermediate', 'intermediate risk': 'Intermediate',
    'intermediate-risk': 'Intermediate', 'moderate': 'Intermediate',
    'low intermediate': 'Intermediate', 'low-intermediate': 'Intermediate',
    'low intermediate risk': 'Intermediate', 'low-intermediate risk': 'Intermediate',
    'high intermediate': 'Intermediate', 'high-intermediate': 'Intermediate',
    'high intermediate risk': 'Intermediate', 'high-intermediate risk': 'Intermediate',
    'high': 'High', 'high risk': 'High', 'high-risk': 'High',
    'very high': 'High', 'very high risk': 'High', 'very-high': 'High',
}
# Clinical score strata must not use the embedding's coarsened IPI palette.
NATIVE_ORDER = ['Low', 'Low-intermediate', 'Intermediate', 'High-intermediate', 'High', 'Very high']
NATIVE_GROUP_MAP = dict(GROUP_MAP)
for label in NATIVE_ORDER:
    for spelling in (label.lower(), label.lower().replace('-', ' ')):
        for suffix in ('', ' risk'):
            NATIVE_GROUP_MAP[spelling + suffix] = label
FOUR_GROUP_ORDER = ['Low', 'Low-intermediate', 'High-intermediate', 'High']


def recorded_group_settings(values):
    raw = values.dropna().astype(str).str.lower().str.strip()
    unknown = sorted(set(raw) - set(NATIVE_GROUP_MAP))
    if unknown:
        raise ValueError(f'Unmapped clinical score categories: {unknown}')
    observed = set(raw.map(NATIVE_GROUP_MAP))
    return dict(group_map=NATIVE_GROUP_MAP,
                group_order=[v for v in NATIVE_ORDER if v in observed])


OUTCOMES = ['infection_diagnosis_or_antimicrobial_use_composite',
            'overall_survival', 'neutropenia_g3plus', 'hospitalisation',
            'creatinine_increased_g2plus']
# Release-v2 rename of the same infection endpoint; retain older exports.
OUTCOME_ALIASES = {'infection_diagnosis_or_antimicrobial_use_composite':
                   'infections_all_definitions'}


def build(analysis, output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = read_table(analysis['patient_metadata']).rename(
        columns=analysis.get('patient_metadata_column_map', {}))
    model = analysis.get('opera_model', 'R_L')
    artifacts = [a for a in analysis['artifacts'] if a['model'] == model
                 and a.get('split') == 'tuning' and int(a.get('seed', 42)) == 42]
    if len(artifacts) != 1:
        raise ValueError(f'Need one selected {model} seed-42 tuning artifact, found {len(artifacts)}')
    predictions = Path(artifacts[0]['path']).resolve()
    embeddings = predictions.parent / 'patient_embeddings.npz'
    if not embeddings.is_file():
        raise FileNotFoundError(embeddings)
    data = read_table(predictions)
    available = set(data.outcome)
    map_candidates = list(OUTCOMES)
    if 'neutropenia_g3plus' not in available and 'anemia_g2plus' in available:
        # Compatibility for older exports, not an alias between endpoints.
        map_candidates[map_candidates.index('neutropenia_g3plus')] = 'anemia_g2plus'
    outcomes = [name if name in available else OUTCOME_ALIASES[name]
                for name in map_candidates
                if name in available or OUTCOME_ALIASES.get(name) in available]
    if not outcomes:
        raise ValueError('No proposed map outcomes present in selected predictions')
    if 'IPI' not in metadata:
        raise ValueError('Patient metadata has no IPI category column')
    raw = sorted(metadata.IPI.dropna().astype(str).unique())
    unknown = [v for v in raw if v.strip().lower() not in GROUP_MAP]
    if unknown:
        raise ValueError(f'Unmapped IPI categories: {unknown}; add explicit mappings before proceeding')
    stages = {'OPERA router': [str(embeddings)]}
    # Include the joint comparator only when its actual export is available.
    joint = [a for a in analysis['artifacts'] if a['model'] == 'J_L'
             and a.get('split') == 'tuning' and int(a.get('seed', 42)) == 42]
    if len(joint) == 1:
        path = Path(joint[0]['path']).resolve().parent / 'patient_embeddings.npz'
        if path.is_file():
            stages = {'Joint model': [str(path)], **stages}
    embedding = dict(method='umap', ipi_group_map=GROUP_MAP, stages=stages,
                     calibration_maps=dict(stage='OPERA router', predictions=str(predictions),
                                           horizon_days=365, outcomes=outcomes))
    template = dict(model=model, seed=42, outcome='overall_survival',
                    outcome_label='Overall survival', horizon_days=365,
                    death_in_endpoint=True, group_column='IPI', group_map=NATIVE_GROUP_MAP,
                    group_order=['Low', 'Intermediate', 'High'], score_high_group='High',
                    age_column='age_at_index', score_date_column=None,
                    score_timing='Recorded at diagnosis in the registry; timing requires verification before final publication',
                    cutoff_rule='cohort_size_matched', comparison_bootstrap=1000,
                    threshold_split='train', minimum_patients=50,
                    lock_provenance='Baseline-only category-size matching in the evaluation cohort. Independent clinical anchors use training predictions when available.',
                    applicability_review='Descriptive comparison restricted to patients with this recorded index; not a claim of validated use in other diseases')
    clinical = dict(availability=dict(cohort_column='cohort_fine', index_column='index', group_column='IPI'),
                    all_indices=dict(index_column='index', cohort_column='cohort_fine',
                                     minimum_patients=1, pooled=True, per_cohort=False,
                                     manuscript=dict(index='IPI'), template=template),
                    scores=[], index_endpoint='shared')
    clinical['all_indices']['overrides'] = {str(index): recorded_group_settings(part.IPI)
        for index, part in metadata.loc[metadata['index'].notna()].groupby('index')}
    # Keep the complete scale even when this sample lacks one native stratum.
    for index, settings in clinical['all_indices']['overrides'].items():
        if index.upper().replace(' ', '') in {'IPI', 'NCCN-IPI', 'NCCNIPI'} and 'Intermediate' not in settings['group_order']:
            settings['group_order'] = FOUR_GROUP_ORDER
        elif index.upper().replace(' ', '') in {'CLL-IPI', 'CLLIPI'}:
            settings['group_order'] = ['Low', 'Intermediate', 'High', 'Very high']
            settings['score_high_group'] = 'Very high'
        elif len(settings['group_order']) < 2:
            settings['group_order'] = ['Low', 'Intermediate', 'High']
    recorded_category_audit = {}
    for index, settings in clinical['all_indices']['overrides'].items():
        values = metadata.loc[metadata['index'].astype(str).eq(index), 'IPI'].dropna().astype(str)
        observed = sorted(set(values.str.strip().str.lower().map(NATIVE_GROUP_MAP)))
        key = index.upper().replace(' ', '')
        coarsened = (key in {'IPI', 'NCCN-IPI', 'NCCNIPI'} and 'Intermediate' in observed)
        incomplete_cll = key in {'CLL-IPI', 'CLLIPI'} and 'Very high' not in observed
        recorded_category_audit[index] = dict(observed_categories=observed,
            status='source_categories_coarsened' if coarsened else
                   'native_category_absent_or_coarsened' if incomplete_cll else 'recorded_categories_preserved',
            native_definition_verified=False)
    score_availability = {}
    # Include available computed scores only in populations where they were recorded.
    for slug, label, score, group in [
        ('ipi_computed', 'IPI (computed)', 'ipi_computed_score', 'ipi_computed_group'),
        ('nccn_ipi', 'NCCN-IPI', 'nccn_ipi_score', 'nccn_ipi_group')]:
        missing = sorted({score, group} - set(metadata))
        score_availability[label] = dict(missing_columns=missing,
            status='missing_columns' if missing else 'available',
            n_complete=int(metadata[[score, group]].notna().all(axis=1).sum()) if not missing else 0)
        if {score, group} <= set(metadata):
            values = metadata[group].dropna().astype(str)
            bad = sorted(set(values) - {v for v in values if v.strip().lower() in NATIVE_GROUP_MAP})
            if bad:
                raise ValueError(f'Unmapped {group} categories: {bad}')
            clinical['scores'].append(dict(template, id=slug, score_label=label,
                cohorts=['DLBCL'] if slug == 'nccn_ipi' else 'all', score_column=score, group_column=group,
                group_map=NATIVE_GROUP_MAP, group_order=FOUR_GROUP_ORDER,
                score_high_threshold=4 if slug == 'ipi_computed' else 6,
                manuscript_figures=False))
            if slug == 'nccn_ipi' and 'treatment_failure' in available and 'cif_730d' in data:
                clinical['scores'].append(dict(clinical['scores'][-1],
                    id='nccn_ipi_treatment_failure_2y', outcome='treatment_failure',
                    outcome_label='Second-line treatment or death', horizon_days=730,
                    include_in_headline=False, manuscript_figures=False,
                    endpoint_definition_note='Frozen composite differs from the HemaSphere relapse-inclusive endpoint'))
    if not {'index', 'cohort_fine', 'age_at_index'} <= set(metadata):
        raise ValueError('Clinical recipe needs index, cohort_fine and age_at_index metadata columns')
    from opera.run.render_clinical_score_panels import expand_index_scores
    specs = expand_index_scores(clinical, metadata)
    if not specs:
        raise ValueError('No recorded clinical scores available')
    # Root clinical plates need one explicitly selected illustration. The old
    # hardcoded recorded index 'IPI' produced none when only R-IPI was recorded.
    evaluated = metadata.copy()
    if 'subject_id' in data:
        ids = set(data.loc[data.outcome == 'overall_survival', 'subject_id'].astype(str))
        evaluated = evaluated.loc[evaluated.subject_id.astype(str).isin(ids)]
    eligible_specs = []
    for spec in specs:
        if not spec.get('include_in_headline', True):
            continue
        patients = evaluated.loc[evaluated.cohort_fine.isin(spec['cohorts'])]
        if spec.get('index_values'):
            patients = patients.loc[patients[spec.get('index_column', 'index')].isin(spec['index_values'])]
        required = [spec['group_column']] + ([spec['score_column']] if spec.get('score_column') else [])
        n = int(patients[required].notna().all(axis=1).sum())
        if n >= spec.get('minimum_patients', 50):
            priority = {'nccn_ipi': 0, 'ipi_computed': 1}.get(spec['id'],
                2 if spec['score_label'] == 'IPI' else 3 if spec['score_label'] == 'R-IPI' else 4)
            eligible_specs.append((priority, -n, spec['id'], spec))
    primary = min(eligible_specs, key=lambda row: row[:3])[3] if eligible_specs else None
    clinical['all_indices']['manuscript'] = {}
    for spec in clinical['scores']:
        spec['manuscript_figures'] = bool(primary and spec['id'] == primary['id'])
    if primary and primary.get('index_values'):
        clinical['all_indices']['manuscript'] = {'index': primary['score_label']}
    for name, recipe in [('embedding_tables.tuning.yaml', embedding), ('clinical_score_panels.tuning.yaml', clinical)]:
        (output / name).write_text(yaml.safe_dump(recipe, sort_keys=False))
    (output / 'config_audit.json').write_text(json.dumps(dict(model=model, categories=raw,
        selected_map_outcomes=outcomes, unavailable_map_outcomes=[o for o in OUTCOMES if o not in available and OUTCOME_ALIASES.get(o) not in available],
        computed_score_availability=score_availability,
        recorded_category_audit=recorded_category_audit,
        clinical_primary=primary['id'] if primary else None,
        clinical_populations=[s['id'] for s in specs],
        note='Shared one-year overall-survival comparison; numeric IPI and NCCN-IPI use four native strata. Recorded scores preserve available categories; already-coarsened source categories cannot be reconstructed. Tuning figures are exploratory.'), indent=2))
    return embedding, clinical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    build(yaml.safe_load(args.analysis_config.read_text()), args.output_dir)
    print(args.output_dir.resolve())


if __name__ == '__main__':
    main()
