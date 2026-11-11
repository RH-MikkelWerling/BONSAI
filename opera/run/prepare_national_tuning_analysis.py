"""Prepare isolated paired tuning comparisons for completed direct national fits."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml

MODELS = {'J':'J_E','R':'R_E','AB':'R_AB_E','OFF':'R_PF_OFF_E'}
BASELINES = {'J_L','J_OA_OWN_L','R_AB_L','R_PF_OFF_L','X','S_L','D_L','O_L'}
CONTRASTS = {
    'NATIONAL_OPERA_LOCAL':'gain(R_E,J_OA_OWN_L)',
    'NATIONAL_JOINT_LOCAL':'gain(J_E,J_L)',
    'NATIONAL_ROUTING_LOCAL':'gain(R_AB_E,R_AB_L)',
    'NATIONAL_CORRECTIONS_LOCAL':'gain(R_PF_OFF_E,R_PF_OFF_L)',
    'NATIONAL_OPERA_XGB':'gain(R_E,X)',
    'LOCAL_OPERA_XGB':'gain(J_OA_OWN_L,X)',
    'NATIONAL_JOINT_XGB':'gain(J_E,X)',
    'NATIONAL_ROUTING_XGB':'gain(R_AB_E,X)',
    'NATIONAL_OPERA_JOINT':'gain(R_E,J_E)',
    'NATIONAL_READING':'gain(R_E,R_PF_OFF_E)',
    'NATIONAL_ROUTING':'gain(R_AB_E,R_PF_OFF_E)',
    'NATIONAL_OPERA_ROUTER':'gain(R_E,R_AB_E)',
}


def prepare(source_path,national_root,output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config=yaml.safe_load(Path(source_path).read_text())
    artifacts=[a for a in config['artifacts'] if a.get('split')=='tuning'
               and a.get('model') in BASELINES and int(a.get('seed',42))==42]
    required={'J_L','J_OA_OWN_L','R_AB_L','R_PF_OFF_L','X'}
    missing=required-{a['model'] for a in artifacts}
    if missing:
        raise ValueError(f'Source inventory lacks tuning baselines: {sorted(missing)}')
    contracts={a['data_contract'] for a in artifacts}
    if len(contracts)!=1:
        raise ValueError('Expected one common data contract')
    root=Path(national_root).resolve()
    bundle=json.loads((root/'direct_fits/bundle.json').read_text())
    added=[]
    for row in bundle['runs']:
        mode=row['name'].rsplit('/',1)[-1]
        if mode not in MODELS or row['origin']!='E' or row['seed']!=42:
            raise ValueError('Unexpected direct national matrix')
        directory=root/'predictions'/row['run_directory']/'tuning'
        if not (directory/'evaluation_complete.json').is_file():
            raise ValueError(f'Incomplete tuning evaluation: {row["name"]}')
        path=directory/'direct_cr_predictions.parquet'
        if not path.is_file():
            raise FileNotFoundError(path)
        added.append(dict(path=str(path),format='direct_cif',model=MODELS[mode],seed=42,
            split='tuning',data_contract=next(iter(contracts)),cohort_from_metadata=True,
            include_horizons_days=config.get('analysis_horizons_days',[30,60,90,180,365,730])))
    if len(added)!=4 or len({a['model'] for a in added})!=4:
        raise ValueError('Expected all four direct fits')
    config['artifacts']=artifacts+added
    config['extra_contrasts']=CONTRASTS
    config['patient_gain_contrasts']=list(CONTRASTS)
    config['opera_model']='J_OA_OWN_L'  # Do not select a new primary encoder here.
    config['design_search']=False
    config['route_weights']=None
    config['clinical_threshold_artifacts']=[]
    # This is a focused comparison, not a repeat of clinical or subgroup work.
    config['identity_date_sensitivity']=False
    config['stratified_reanalysis']={}
    config['patient_groups']={}
    config['operating_points']={}
    output.mkdir(parents=True)
    manifest={'essential_contrasts':{'direct_national_tuning':CONTRASTS}}
    (output/'contrasts.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
    config['contrast_manifest']=str(output/'contrasts.yaml')
    (output/'analysis.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    report=dict(scope='Tuning only; descriptive comparisons, no encoder selection or held-out evaluation',
        baseline_artifacts=len(artifacts),national_artifacts=len(added),
        source_analysis=str(Path(source_path).resolve()),
        source_sha256=hashlib.sha256(Path(source_path).read_bytes()).hexdigest())
    (output/'preparation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['source-analysis','national-root','output-dir']:
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    prepare(a.source_analysis,a.national_root,a.output_dir)


if __name__=='__main__':
    main()
