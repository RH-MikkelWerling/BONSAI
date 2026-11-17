import json
import numpy as np
import pandas as pd
from opera.evaluation.distributed_analysis import save_prepared
from opera.analysis.correction_reliance import run


def test_replay_effects_reuse_main_weights(tmp_path):
    ids=list(range(40))
    frame=pd.DataFrame(dict(subject_id=ids,model='J_OA_OWN_L',split='tuning',seed=42,
        cohort='d',outcome='o',horizon_days=30,label=np.arange(40)%2,
        probability=np.linspace(.1,.9,40),ipcw_weight=1.))
    prepared=tmp_path/'prepared'
    context=dict(metadata=pd.DataFrame(dict(subject_id=ids,cohort_fine='d')),
                 draws={('tuning',42,'d'):(dict(zip(ids,ids)),np.ones((10,40)))},
                 performance_support={'min_cases':1,'min_controls':1})
    save_prepared(prepared,frame,context,chunk_rows=20)
    root=tmp_path/'replays'
    for mode in ['full','all']:
        directory=root/'L_seed42'/mode
        directory.mkdir(parents=True)
        values=frame.probability.to_numpy()
        native=pd.DataFrame(dict(subject_id=ids,cohort='unused',outcome='o',
             split='tuning',event=np.arange(40)%2,time_days=np.where(np.arange(40)%2,10,100),
             cif_30d=values if mode=='full' else .1+.8*values))
        native.to_parquet(directory/'direct_cr_predictions.parquet')
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'jobs':[dict(encoder='L',seed=42,run_id='fixture',modes=['full','all'])]}))
    output=tmp_path/'output'
    run(prepared,plan,root,output)
    effects=pd.read_csv(output/'inference_ablation_cells.csv')
    assert effects.gain.abs().max()<1e-12
    assert effects.mean_abs_delta_cif.min()>0
    assert effects.n.iloc[0]==40
    assert pd.read_csv(output/'inference_ablation_summaries.csv').n_cells.eq(1).all()
