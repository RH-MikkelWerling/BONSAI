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
OUTCOMES = ['infection_diagnosis_or_antimicrobial_use_composite',
            'overall_survival', 'anemia_g2plus', 'hospitalisation',
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
    outcomes = [name if name in available else OUTCOME_ALIASES[name]
                for name in OUTCOMES
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
                    death_in_endpoint=True, group_column='IPI', group_map=GROUP_MAP,
                    group_order=['Low', 'Intermediate', 'High'], score_high_group='High',
                    age_column='age_at_index', score_date_column=None,
                    score_timing='Recorded at diagnosis in the registry; timing requires verification before final publication',
                    cutoff_rule='risk_matched_low', minimum_patients=50,
                    lock_provenance='Tuning rehearsal recipe; model cutoffs fitted on tuning patients',
                    applicability_review='Descriptive comparison restricted to patients with this recorded index; not a claim of validated use in other diseases')
    clinical = dict(availability=dict(cohort_column='cohort_fine', index_column='index', group_column='IPI'),
                    all_indices=dict(index_column='index', cohort_column='cohort_fine',
                                     minimum_patients=1, pooled=True, per_cohort=False,
                                     manuscript=dict(index='IPI'), template=template),
                    scores=[], index_endpoint='shared')
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
            bad = sorted(set(values) - {v for v in values if v.strip().lower() in GROUP_MAP})
            if bad:
                raise ValueError(f'Unmapped {group} categories: {bad}')
            clinical['scores'].append(dict(template, id=slug, score_label=label,
                cohorts='all', score_column=score, group_column=group,
                score_high_threshold=4 if slug == 'ipi_computed' else 6,
                manuscript_figures=False))
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
        patients = evaluated.loc[evaluated.cohort_fine.isin(spec['cohorts'])]
        if spec.get('index_values'):
            patients = patients.loc[patients[spec.get('index_column', 'index')].isin(spec['index_values'])]
        required = [spec['group_column']] + ([spec['score_column']] if spec.get('score_column') else [])
        n = int(patients[required].notna().all(axis=1).sum())
        if n >= spec.get('minimum_patients', 50):
            priority = {'ipi_computed': 0, 'nccn_ipi': 1}.get(spec['id'],
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
        clinical_primary=primary['id'] if primary else None,
        clinical_populations=[s['id'] for s in specs],
        note='Shared one-year overall-survival comparison; harmonized categories are a coarsened score. Tuning figures are exploratory.'), indent=2))
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
