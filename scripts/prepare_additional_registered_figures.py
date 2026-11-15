"""Prepare further existing paper plates from the verified aggregate download."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from opera.run.build_paper_plot_tables import calibration_tables, calibration_examples


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--registered-inputs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    manifest=json.loads((a.bundle/'bundle_manifest.json').read_text())
    parent=json.loads((a.registered_inputs/'input_manifest.json').read_text())
    for name,h in parent['source_hashes'].items():
        if h!=manifest['files']['tables/'+name]:
            raise ValueError('Existing registered inputs come from another download')
    sources={}
    for name in ['calibration_summary.csv.gz','calibration.csv.gz']:
        path=a.bundle/'tables'/name
        sources[str(path)]=digest(path)
        if sources[str(path)]!=manifest['files']['tables/'+name]:
            raise ValueError(f'Checksum mismatch: {name}')
    paired_path=a.registered_inputs/'plot_tables/paired_cells.csv'
    meta_path=a.registered_inputs/'plot_tables/cell_metadata.csv'
    sources[str(paired_path)]=digest(paired_path);sources[str(meta_path)]=digest(meta_path)
    cells=pd.read_csv(paired_path,low_memory=False)
    metadata=pd.read_csv(meta_path)
    def year_table(name):
        parts=[]
        for d in pd.read_csv(a.bundle/'tables'/name,chunksize=100_000,low_memory=False):
            d=d[(d.split=='held_out')&(d.seed==42)&(d.horizon_days==365)]
            parts.append(d)
        return pd.concat(parts,ignore_index=True)
    calibration,reliability=calibration_tables(year_table('calibration_summary.csv.gz'),
        year_table('calibration.csv.gz'),metadata,seed=42)
    choices=Path('opera/configs/paper_figure_choices.yaml')
    sources[str(choices)]=digest(choices)
    examples=calibration_examples(yaml.safe_load(choices.read_text()),metadata)
    if not examples:
        raise ValueError('No examples satisfy the registered training-only rule')
    a.output_dir.mkdir(parents=True)
    tables=a.output_dir/'plot_tables';tables.mkdir()
    cells.to_csv(tables/'paired_cells.csv',index=False)
    calibration.to_csv(tables/'calibration_summary.csv',index=False)
    reliability.to_csv(tables/'reliability.csv',index=False)
    (tables/'calibration_examples.json').write_text(json.dumps(examples,indent=2))
    recipe=dict(required_figures=['supp_pooling_cells','supp_xgboost_gap','fig_calibration'],figures=[
        dict(id='supp_pooling_cells',function='figure_pooling_cells',
             tables=dict(cells='plot_tables/paired_cells.csv')),
        dict(id='supp_xgboost_gap',function='figure_xgboost_gap',
             tables=dict(cells='plot_tables/paired_cells.csv')),
        dict(id='fig_calibration',function='figure_calibration',options=dict(horizon=365),
             tables=dict(summary='plot_tables/calibration_summary.csv',
                         reliability='plot_tables/reliability.csv'),
             json_inputs=dict(examples='plot_tables/calibration_examples.json'))])
    # Use the already registered one-map-per-comparison pages, unchanged.
    full_recipe=Path('opera/configs/paper_results_figures.full.example.yaml')
    sources[str(full_recipe)]=digest(full_recipe)
    for plate in yaml.safe_load(full_recipe.read_text())['figures']:
        if plate['id'] in {f'supp_pooling_cells_{i:02d}' for i in range(1,6)}:
            recipe['figures'].append(dict(id=plate['id'],function=plate['function'],
                tables=dict(cells='plot_tables/paired_cells.csv'),options=plate['options']))
            recipe['required_figures'].append(plate['id'])
    (a.output_dir/'figures.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    (a.output_dir/'input_manifest.json').write_text(json.dumps(dict(
        source_hashes=sources,split='held_out',seed=42,
        implementation_sha256=digest(__file__),
        layout_source='Existing registered figure functions and canonical plot-table converters',
        calibration_horizon=365,
        calibration_horizon_reason='Matches the existing manuscript caption specifying one-year calibration',
        calibration_example_rule='Existing paper_figure_choices.yaml rule uses training events and event share only',
        model_identity='OPERA alias in this download is cohort-encoder J_OA_OWN_L; no adapted model is substituted',
        new_analysis_or_fits=False),indent=2))
    print(a.output_dir)


if __name__=='__main__':
    main()
