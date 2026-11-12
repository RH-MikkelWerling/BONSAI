"""Read both weighting schemes from final CSVs or completed summary queues."""
import argparse
import json
from pathlib import Path
import pickle

import pandas as pd

CORE = {'EXT_ADAPTATION','EXT_ARCH_L','EXT_ARCH_E','EXT_ARCH_EA',
        'I1','I2','I3','OFF_J','OFF_J_E','OFF_J_EA','EXT_READING_EA','EXT_ROUTING_EA'}
for model in ['J_EA','R_PF_OFF_EA','R_EA','R_AB_EA']:
    CORE |= {f'EXT_{model}_VS_{reference}' for reference in ['XP','XR','J_L','J_OA_OWN_L']}
CORE |= {f'EXT_{m}_VS_{r}' for m in ['J_E','R_E'] for r in ['XP','XR']}
CORE |= {'EXT_R_E_VS_J_OA_OWN_L','EXT_J_E_VS_J_L','EXT_J_OA_OWN_L_VS_XP'}


def select(frame,all_extended=False):
    result=frame
    for column,value in [('split','held_out'),('seed',42),('metric','auc'),
                          ('stratum','all_cells'),('horizon_days',0)]:
        if column in result:
            result=result.loc[result[column].eq(value)]
    keep=result.contrast.isin(CORE)
    if all_extended:
        keep |= result.contrast.str.startswith('EXT_',na=False)
    return result.loc[keep]


def read_summary(project,aggregation,all_extended=False):
    project=Path(project)
    name='effect_summaries' if aggregation=='equal_cell' else 'effect_summaries_disease_balanced'
    for path in [project/'results'/f'{name}.csv',project/'results.progress'/f'{name}.csv']:
        if path.is_file():
            return select(pd.read_csv(path,low_memory=False),all_extended),str(path)
    base=project/'checkpoints'
    generation=json.loads((base/'active.json').read_text())['generation']
    if Path(generation).name!=generation:
        raise ValueError('Invalid checkpoint generation')
    queue=base/generation/'tasks'/('equal_cell' if aggregation=='equal_cell' else 'disease_balanced')
    descriptor=json.loads((queue/'queue.json').read_text())
    if descriptor['options'].get('kind')!='effect_summary' or descriptor['options'].get('aggregation')!=aggregation:
        raise ValueError('Unexpected summary queue kind or weighting')
    paths=[queue/f'batch_{i:05d}.result_INTERNAL.pkl' for i in range(descriptor['total'])]
    if not all(p.is_file() for p in paths):
        raise RuntimeError(f'Summary queue is not complete: {queue}')
    parts=[]
    for path in paths:
        # Only trusted internal scalar summary results, never the large stage tuple.
        with path.open('rb') as handle:
            result=pickle.load(handle)
        if not isinstance(result,pd.DataFrame):
            raise TypeError('Expected scalar effect-summary DataFrame')
        parts.append(select(result,all_extended))
    return pd.concat(parts,ignore_index=True),str(queue)+' (completed, pre-final annotations)'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir',type=Path,default=Path('data/paper_256_v3_national_20261111/extended_test_comparison'))
    p.add_argument('--all-extended',action='store_true')
    args=p.parse_args()
    for aggregation in ['equal_cell','disease_balanced']:
        rows,source=read_summary(args.project_dir,aggregation,args.all_extended)
        columns=[c for c in ['contrast','gain','lower','upper','p_value','n_cells','n_diseases'] if c in rows]
        print('\n'+aggregation.upper()+'\nSource: '+source,flush=True)
        print(rows[columns].sort_values('contrast').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
