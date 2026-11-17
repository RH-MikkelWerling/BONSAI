"""Recover native empirical figure inputs from the verified downloaded aggregates.

Reads large per-cell exports in chunks. No patient data, model fitting, or new
uncertainty estimates. Missing patient subgroups and pooled-patient atlas AUCs
remain explicitly unavailable.
"""
import argparse
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tables = output / 'plot_tables'
    tables.mkdir()
    from opera.run.build_paper_plot_tables import model_cells, origin_estimates, coverage_table
    export = Path('scratch/final_figure_inputs_20261116')
    base = Path('scratch/registered_figures_20261115_v2/plot_tables')
    local = pd.read_csv(base / 'paired_cells.csv', low_memory=False)
    # Retain the original verified local-model comparisons, and use the new
    # external export only for the named origin comparisons.
    external_codes = {'E-L:J', 'E-L:R', 'EA-E:J', 'EA-E:R', 'EA-L:J', 'EA-L:R'}
    parts = []
    for chunk in pd.read_csv(export / 'extended/plot_tables/paired_cells.csv', chunksize=50000):
        part = chunk.loc[chunk.metric.eq('auc')]
        if len(part):
            parts.append(part)
    external = pd.concat(parts, ignore_index=True)
    keys = ['cohort', 'outcome', 'horizon_days', 'contrast', 'metric']
    overlap = local.merge(external, on=keys, suffixes=('_old', '_new'))
    if ((overlap.gain_old - overlap.gain_new).abs().dropna() > 1e-10).any():
        raise ValueError('Local and extended downloads disagree on a common paired cell')
    combined = pd.concat([local.loc[~local.contrast.isin(set(external.contrast))], external], ignore_index=True)
    combined.to_csv(tables / 'paired_cells.csv', index=False)
    with ZipFile('scratch/paper_results_export_20261116_v3/support_coverage/support_coverage_inputs.zip') as z:
        metadata = pd.read_csv(z.open('cell_metadata.csv'))
        coverage_raw = pd.read_csv(z.open('aggregates/contrast_coverage.csv'), low_memory=False)
        models_raw = pd.read_csv(z.open('aggregates/model_coverage.csv'), low_memory=False)
    # Coverage here describes the registered joint-versus-specialist comparison.
    # Apply the canonical builder's legacy early-censoring count correction,
    # using the saved evaluated counts from the completed fraction export.
    models_raw = models_raw.loc[models_raw.model.isin(['J_L', 'S_L'])].copy()
    evaluated = pd.read_csv('scratch/paper_results_export_20261116_v3/label_efficiency/estimates.csv')
    evaluated = evaluated.loc[evaluated.metric.eq('auc') & evaluated.model.isin(['J_L', 'S_L'])]
    keys = ['split', 'seed', 'cohort', 'outcome', 'horizon_days', 'model']
    counts = evaluated[keys + ['n_cases', 'n_controls']].drop_duplicates()
    models_raw = models_raw.merge(counts, on=keys, how='left', suffixes=('', '_evaluated'), validate='one_to_one')
    for column in ['n_cases', 'n_controls']:
        models_raw[column] = models_raw[column + '_evaluated'].fillna(models_raw[column])
    present = models_raw.n_cases + models_raw.n_controls
    models_raw['prediction_status'] = models_raw.prediction_status.mask(present > 0, 'two_class').mask(
        (present > 0) & ((models_raw.n_cases == 0) | (models_raw.n_controls == 0)), 'single_class')
    coverage_table(metadata, coverage_raw, models_raw, seed=42).to_csv(tables / 'coverage.csv', index=False)
    shutil.copy2(export / 'extended/plot_tables/training_support.csv', tables / 'training_support.csv')
    # The old bundle already contains all local absolute cell estimates.
    estimates = []
    for chunk in pd.read_csv('scratch/download_bundle_20261112/tables/estimates.csv.gz', chunksize=50000):
        part = chunk.loc[chunk['split'].eq('held_out') & chunk.seed.eq(42) & chunk.metric.eq('auc')
                         & chunk.model.isin(['S_L', 'D_L', 'O_L', 'J_L', 'OPERA', 'R_L', 'X', 'XP', 'XR', 'LR'])]
        if len(part):
            estimates.append(part)
    local_estimates = pd.concat(estimates, ignore_index=True)
    models = model_cells(local_estimates, metadata, seed=42)
    fraction_estimates = pd.read_csv('scratch/paper_results_export_20261116_v3/label_efficiency/estimates.csv')
    fractions = model_cells(fraction_estimates, metadata, seed=42)
    keys = ['cohort', 'outcome', 'horizon_days']
    overlapping = [c for c in fractions if c in models and c not in keys]
    check = models.merge(fractions, on=keys, suffixes=('_local', '_fraction'))
    for column in overlapping:
        a, b = check[column + '_local'], check[column + '_fraction']
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            conflict = ((a - b).abs().dropna() > 1e-10).any()
        else:
            conflict = (a.notna() & b.notna() & a.ne(b)).any()
        if conflict:
            raise ValueError(f'Conflicting downloaded cell values: {column}')
    models = models.merge(fractions[keys + [c for c in fractions if c not in models]],
                          on=keys, how='outer', validate='one_to_one')
    models.to_csv(tables / 'model_cells.csv', index=False)
    panels = pd.read_csv(export / 'extended/aggregates/panel_estimates.csv')
    origins = origin_estimates(panels, seed=42)
    if origins.duplicated(['origin', 'readout', 'metric', 'horizon_days']).any():
        raise ValueError('Ambiguous encoder-origin panel')
    origins.to_csv(tables / 'origin_estimates.csv', index=False)
    shutil.copy2(export / 'extended/plot_tables/external_panel.csv', tables / 'external_panel.csv')
    for name in ['calibration_summary.csv', 'reliability.csv', 'calibration_examples.json']:
        shutil.copy2(base / name, tables / name)
    shutil.copy2('scratch/descriptive_support_20261116/plot_tables/support_curve.csv', tables / 'support_curve.csv')
    with ZipFile('scratch/paper_results_export_20261116_v3/clinical/J_OA_OWN_L.zip') as z:
        (tables / 'headline_scores.csv').write_bytes(z.read('headline_scores.csv'))
    source = yaml.safe_load(Path('opera/configs/paper_results_figures.full.example.yaml').read_text())
    ids = ['fig_external', 'fig_headline', 'supp_coverage', 'supp_support', 'supp_horizons', 'fig_benchmarks']
    plates = []
    for plate in source['figures']:
        if plate['id'] not in ids:
            continue
        for section in ['tables', 'json_inputs']:
            for key, path in list(plate.get(section, {}).items()):
                candidate = tables / Path(path).name
                if candidate.exists():
                    plate[section][key] = str(candidate)
                else:
                    del plate[section][key]
        if plate['id'] == 'fig_headline':
            plate['options'].update(clinical_model='J_OA_OWN_L', model_overview=True)
        plates.append(plate)
    (output / 'figures.yaml').write_text(yaml.safe_dump({'figures': plates, 'required_figures': ids}, sort_keys=False))
    (output / 'recovery_manifest.json').write_text(json.dumps({
        'scope': 'Verified held-out downloads; original local and external model identities retained',
        'external_local_OPERA_alias': 'J_OA_OWN_L -> origin L, readout R',
        'unavailable': ['Headline patient-history subgroup panel', 'Pooled-patient atlas metrics',
                        'Local fraction fits', 'Task/correction geometry', 'IG/embeddings pending exports'],
        'external_fraction_display': 'Available adapted fractions and local 100% reference only',
        'row_counts': {'model_cells': len(models), 'external_cells': len(external), 'local_estimates': len(local_estimates)},
        'paired_intervals_recomputed': False}, indent=2))
    print(output / 'figures.yaml')


if __name__ == '__main__':
    main()
