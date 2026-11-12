"""Record the existing encoder rule from completed paired tuning summaries only."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from bonsai.functional.checkpointing import verified_training_selection
from opera.run.paper_prediction_queue import bundle_runs,canonical_models


def choose_encoder(national_gain,adapted_gain,adaptation_gain,margin=.0052):
    if abs((adapted_gain-national_gain)-adaptation_gain)>1e-5:
        raise ValueError('Tuning contrasts are not algebraically consistent')
    eligible={m:g for m,g in [('R_E',national_gain),('R_EA',adapted_gain)] if g>margin}
    if not eligible:
        return 'J_OA_OWN_L'
    if len(eligible)==1:
        return next(iter(eligible))
    if abs(adaptation_gain)<margin:
        return 'R_E'
    return 'R_EA' if adaptation_gain>0 else 'R_E'


def record(results,root,output):
    results,root,output=map(Path,(results,root,output))
    if output.exists():
        raise FileExistsError(output)
    if not (results/'analysis_manifest.json').is_file():
        raise ValueError('Wait for complete tuning analysis')
    path=results/'effect_summaries.csv'
    frame=pd.read_csv(path,low_memory=False)
    codes=['EXT_R_E_VS_J_OA_OWN_L','EXT_R_EA_VS_J_OA_OWN_L','EXT_ADAPTATION']
    rows=frame.loc[frame.contrast.isin(codes)&frame.split.eq('tuning')&frame.seed.eq(42)
                   &frame.metric.eq('auc')&frame.stratum.eq('all_cells')&frame.horizon_days.eq(0)]
    if len(rows)!=3 or set(rows.contrast)!=set(codes) or rows.n_cells.nunique()!=1:
        raise ValueError('Expected three unique tuning contrasts on equal supported-cell counts')
    gains=rows.set_index('contrast').gain
    selected=choose_encoder(*(float(gains[c]) for c in codes))
    identities={}
    for bundle_name in ['direct_fits','adapted_fits']:
        bundle,entries=bundle_runs(root/bundle_name)
        for row,_,config in entries:
            if row['seed']!=42 or row['phase']!='external':
                continue
            for model in canonical_models(row,config,bundle):
                if model not in ['R_E','R_EA','J_EA']:
                    continue
                directory=root/'neural_runs'/row['run_directory']
                selection=verified_training_selection(directory)
                identities[model]=dict(run_id=row['name'],checkpoint=str(directory/'best.ckpt'),
                    checkpoint_sha256=selection['best_checkpoint_sha256'],scientific_hash=row['scientific_hash'])
    if selected not in identities:
        raise ValueError('Selected local model needs its separate local checkpoint record; refusing an incomplete record')
    value=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),decision_basis='paired tuning only',
        margin=.0052,selected_opera=selected,model_identity=identities[selected],
        illustrative_model='J_EA',illustrative_identity=identities.get('J_EA'),
        illustrative_rationale='Simplest adapted architecture; comparable tuning performance; report every fixed architecture arm',
        tuning_contrasts=rows.to_dict(orient='records'),
        tuning_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        analysis_manifest_sha256=hashlib.sha256((results/'analysis_manifest.json').read_bytes()).hexdigest(),
        scope='Records the prior encoder rule and agreed presentation rationale; does not rewrite active analysis configs')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump(value,handle,indent=2)
    print(json.dumps(value,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--national-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();record(a.results_dir,a.national_root,a.output)


if __name__=='__main__':
    main()
