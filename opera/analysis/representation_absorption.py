"""Exploratory representation/correction diagnostics on prepared main-analysis data.

Read-only with respect to the main analysis. Bounded parquet bucketing avoids
loading the complete prediction inventory. Uses its fixed IPCW and exact shared
patient-bootstrap weights. No training or primary-model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from opera.evaluation.distributed_analysis import load_pickle
from opera.evaluation.secondary_analysis import CELL, metric_values, bootstrap_metric_values

ARMS = {'L': ('J_L', 'J_OA_OWN_L'), 'E': ('J_E', 'R_E'), 'EA': ('J_EA', 'R_EA')}
SCARCE_ARMS = {10: {'L_100': ARMS['L'], 'EA_100': ARMS['EA'],
                    'L_10': ('J_L_F10','J_OA_OWN_F10_L'),
                    'EA_10': ('J_EA_F10','R_EA_F10')},
               25: {'L_100': ARMS['L'], 'EA_100': ARMS['EA'],
                    'L_25': ('J_L_F25','J_OA_OWN_F25_L'),
                    'EA_25': ('J_EA_F25','R_EA_F25')}}


def verify_scarcity_contracts(path):
    """Require matched fraction/subset and training settings before using fraction arms."""
    import yaml
    if path is None:
        return set(), {}
    manifest=json.loads(Path(path).read_text())
    hashes={}
    shares=set()
    for share,origins in manifest.items():
        share=int(share)
        if share not in SCARCE_ARMS or set(origins)!={'L','EA'}:
            raise ValueError('Scarcity contracts require L and EA at a supported share')
        for origin,paths in origins.items():
            configs=[]
            for role in ['joint_config','opera_config']:
                source=Path(paths[role])
                configs.append(yaml.safe_load(source.read_text()))
                hashes[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
            a,b=configs
            if a['training'] != b['training']:
                raise ValueError('Scarcity arms differ in training or patient-subset settings')
            if not np.isclose(a['training'].get('train_patient_fraction',1),share/100):
                raise ValueError('Scarcity fraction does not match contract')
            if 'train_fraction_seed' not in a['training']:
                raise ValueError('Scarcity contract must specify the nested-subset seed')
            for field in ['seed','dataset','paths','dapt_ckpt']:
                if a.get(field)!=b.get(field):
                    raise ValueError(f'Scarcity arms differ in {field}')
        shares.add(share)
    return shares,hashes


def patient_half(subject, seed=2026):
    return hashlib.sha256(f'{seed}:{subject}'.encode()).digest()[0] % 2


def rank_correlation(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.unique(np.asarray(a)[valid]).size < 2 or np.unique(np.asarray(b)[valid]).size < 2:
        return np.nan
    return float(spearmanr(np.asarray(a)[valid], np.asarray(b)[valid]).statistic)


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return (np.quantile(values, [.025, .975]) if len(values) else [np.nan, np.nan])


def bootstrap_auc(labels, probabilities, weights, ipcw=None):
    """AUC-only form of the main bootstrap metric, including identical tie handling."""
    y,p=np.asarray(labels,float),np.asarray(probabilities,float)
    row_weight=np.ones(len(y)) if ipcw is None else np.asarray(ipcw,float)
    order=np.argsort(p,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    result=[]
    for start in range(0,len(weights),128):
        w=weights[start:start+128].astype(float)*row_weight
        positive=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
        negative=np.add.reduceat(w[:,order]*(1-y[order]),starts,axis=1)
        denominator=positive.sum(axis=1)*negative.sum(axis=1)
        numerator=(positive*(negative.cumsum(axis=1)-.5*negative)).sum(axis=1)
        result.append(np.divide(numerator,denominator,out=np.full(len(w),np.nan),where=denominator>0))
    return np.concatenate(result) if result else np.array([])


def cell_values(frame, draws, min_cases=5, min_controls=5, arms=None, include_halves=True):
    """One common patient roster across six arms; full and disjoint-half AUCs."""
    models = list(dict.fromkeys(m for pair in (ARMS if arms is None else arms).values() for m in pair))
    groups = {m: g.set_index('subject_id').sort_index() for m, g in frame.groupby('model') if m in models}
    if set(groups) != set(models):
        return None, 'missing_arm'
    reference = groups[models[0]]
    if any(g.index.has_duplicates for g in groups.values()):
        raise ValueError('Duplicate patient predictions in a cell')
    for g in groups.values():
        if not g.index.equals(reference.index):
            return None, 'different_patient_roster'
        for column in ['label', 'ipcw_weight']:
            if column in reference and (column not in g or not np.array_equal(g[column], reference[column], equal_nan=True)):
                raise ValueError(f'Arms disagree on {column}')
    y = reference.label.to_numpy(float)
    ipcw = reference.ipcw_weight.to_numpy(float) if 'ipcw_weight' in reference else None
    known = np.ones(len(y), bool) if ipcw is None else ipcw > 0
    def supported(mask):
        return ((known & mask & (y == 1)).sum() >= min_cases
                and (known & mask & (y == 0)).sum() >= min_controls)
    if not supported(np.ones(len(y), bool)):
        return None, 'insufficient_full_cell_support'
    mapping, counts = draws
    weights = counts[:, [mapping[s] for s in reference.index]]
    halves = np.array([patient_half(s) for s in reference.index])
    result = {}
    scopes=[('full', np.ones(len(y), bool))]
    if include_halves:
        scopes += [('half0', halves == 0), ('half1', halves == 1)]
    for scope, mask in scopes:
        if not supported(mask):
            continue
        result[scope] = {}
        for model, g in groups.items():
            p = g.probability.to_numpy(float)[mask]
            w = None if ipcw is None else ipcw[mask]
            result[scope][model] = (
                metric_values(y[mask], p, ipcw=w)['auc'],
                bootstrap_auc(y[mask], p, weights[:, mask], ipcw=w))
    return result, 'supported'


def quantities(values, origin):
    j, opera = ARMS[origin]
    local_j, local_o = ARMS['L']
    def difference(a, b):
        return values[a][0] - values[b][0], values[a][1] - values[b][1]
    gap_l = difference(local_o, local_j)
    gap = difference(opera, j)
    return dict(gap_local=gap_l, gap=gap,
                closure=(gap_l[0]-gap[0], gap_l[1]-gap[1]),
                repgain=difference(j, local_j), level=values[j])


def bucket_predictions(prepared, output, split, buckets=64):
    import pyarrow.parquet as pq
    manifest = json.loads((prepared/'prepared.json').read_text())
    wanted = {m for pair in ARMS.values() for m in pair} | {'XP'}
    wanted |= {m for arms in SCARCE_ARMS.values() for pair in arms.values() for m in pair}
    directory = output/'buckets_INTERNAL'
    directory.mkdir()
    columns = CELL + ['subject_id', 'model', 'label', 'probability', 'ipcw_weight']
    for number, name in enumerate(manifest['parts']):
        file = pq.ParquetFile(prepared/name)
        available = set(file.schema.names)
        for batch_number, batch in enumerate(file.iter_batches(batch_size=100_000, columns=[c for c in columns if c in available])):
            frame = batch.to_pandas()
            frame = frame.loc[frame.model.isin(wanted) & frame.split.eq(split)]
            if frame.empty:
                continue
            codes = pd.util.hash_pandas_object(frame[CELL], index=False).to_numpy() % buckets
            for bucket in np.unique(codes):
                frame.loc[codes == bucket].to_parquet(directory/f'{int(bucket):03d}_{number:05d}_{batch_number:04d}.parquet', index=False)
        print(f'Diagnostic bucketing: {number+1}/{len(manifest["parts"])} parts', flush=True)
    return directory


def run(prepared, output, split='tuning', cell_metadata=None, held_out_ready=False, scarcity_contracts=None):
    prepared, output = Path(prepared), Path(output)
    if split == 'held_out' and not held_out_ready:
        raise ValueError('Repeat held-out only after its main evaluation completes; pass --held-out-ready')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    verified_shares, scarcity_hashes=verify_scarcity_contracts(scarcity_contracts)
    context = load_pickle(prepared/'context_INTERNAL.pkl')
    config_support = context['performance_support']
    buckets = bucket_predictions(prepared, output, split)
    metadata = (pd.read_parquet(cell_metadata) if Path(cell_metadata).suffix=='.parquet'
                else pd.read_csv(cell_metadata)) if cell_metadata else None
    rows, dropped, agreement = [], [], []
    vectors, half_vectors = {}, {}
    scarcity_rows, scarcity_draws, scarcity_missing = [], {}, []
    for bucket in sorted({p.name.split('_')[0] for p in buckets.glob('*.parquet')}):
        frame = pd.concat([pd.read_parquet(p) for p in sorted(buckets.glob(bucket+'_*.parquet'))], ignore_index=True)
        for key, cell in frame.groupby(CELL, sort=True):
            identity = dict(zip(CELL, key))
            for share, arms in SCARCE_ARMS.items():
                if share not in verified_shares:
                    scarcity_missing.append({**identity,'share':share,'missing_models':'matched_training_contracts_not_supplied'})
                    continue
                available=set(cell.model)
                required={m for pair in arms.values() for m in pair}
                absent=required-available
                if absent:
                    scarcity_missing.append({**identity,'share':share,'missing_models':','.join(sorted(absent))})
                    continue
                scarce, scarce_status=cell_values(cell,context['draws'][key[:3]],arms=arms,include_halves=False,**config_support)
                if scarce is None:
                    scarcity_missing.append({**identity,'share':share,'missing_models':scarce_status})
                    continue
                full=scarce['full']
                for origin in ['L','EA']:
                    def gap(label):
                        joint,opera=arms[label]
                        return full[opera][0]-full[joint][0],full[opera][1]-full[joint][1]
                    small,base=gap(f'{origin}_{share}'),gap(f'{origin}_100')
                    for label,(point,draw) in [('gap_fraction',small),('gap_full',base),
                        ('scarcity_interaction',(small[0]-base[0],small[1]-base[1]))]:
                        index=len(scarcity_rows)
                        scarcity_rows.append({**identity,'encoder':origin,'share':share,'quantity':label,'gain':point})
                        scarcity_draws[index]=draw
            values, status = cell_values(cell, context['draws'][key[:3]], **config_support)
            if values is None:
                dropped.append({**identity, 'reason':status})
                continue
            if not {'half0', 'half1'} <= values.keys():
                dropped.append({**identity, 'reason':'insufficient_disjoint_half_support_for_correlation_only'})
            for origin in ['E', 'EA']:
                q = quantities(values['full'], origin)
                index = len(rows)
                rows.append({**identity, 'encoder':origin, **{k:v[0] for k,v in q.items()}})
                vectors[index] = q
                if {'half0','half1'} <= values.keys():
                    half_vectors[index] = (quantities(values['half0'],origin), quantities(values['half1'],origin))
            for a,b in [('R_EA','J_EA'),('R_EA','XP'),('J_EA','XP')]:
                subset = cell.loc[cell.model.isin([a,b])]
                if set(subset.model) != {a,b}:
                    continue
                pivot = subset.pivot(index='subject_id',columns='model',values='probability').dropna()
                agreement.append({**identity,'model_a':a,'model_b':b,'rho':rank_correlation(pivot[a],pivot[b]),'n':len(pivot)})
        print(f'Diagnostic cells: {len(rows)//2} supported', flush=True)
    cells = pd.DataFrame(rows)
    if cells.empty:
        raise ValueError('No common supported six-arm cells; inspect inputs')
    if metadata is not None:
        keys = [k for k in ['cohort','outcome','horizon_days'] if k in metadata]
        extra = [c for c in ['n_events_train','outcome_family'] if c in metadata]
        if metadata.duplicated(keys).any():
            raise ValueError('Cell metadata keys are not unique')
        cells = cells.merge(metadata[keys+extra], on=keys, how='left', validate='many_to_one', sort=False)
    # Merge preserves the original row order; keep explicit index for bootstrap vectors.
    cells['draw_index'] = np.arange(len(cells))
    summaries, correlations = [], []
    for (seed, origin), group in cells.groupby(['seed','encoder']):
        for column in ['level', 'n_events_train']:
            if column in group:
                cells.loc[group.index,column+'_quartile'] = pd.qcut(group[column].rank(method='average'),4,labels=False,duplicates='drop')
        strata = [('overall', 'all', group)]
        for column in ['cohort','outcome_family','level_quartile','n_events_train_quartile']:
            if column in cells:
                for value, g in cells.loc[group.index].groupby(column):
                    strata.append((column,str(value),g))
        for stratum,value,g in strata:
            ids=list(g.draw_index.astype(int))
            for quantity in ['closure','repgain','gap','gap_local']:
                draws=np.stack([vectors[i][quantity][1] for i in ids])
                # Strict finite roster: a cell must support all shared replicates.
                finite=np.isfinite(draws).all(axis=1)
                mean=draws[finite].mean(axis=0) if finite.any() else np.array([])
                lo,hi=interval(mean)
                points=[vectors[i][quantity][0] for i,ok in zip(ids,finite) if ok]
                summaries.append(dict(seed=seed,encoder=origin,stratum=stratum,value=value,quantity=quantity,
                                      gain=np.mean(points) if points else np.nan,lower=lo,upper=hi,n_cells=int(finite.sum()),
                                      dropped_nonfinite_draw_cells=int((~finite).sum())))
            half_eligible=[i for i in ids if i in half_vectors]
            for direction in [0,1]:
                eligible=[i for i in half_eligible
                          if np.isfinite(half_vectors[i][direction]['gap_local'][1]).all()
                          and np.isfinite(half_vectors[i][1-direction]['repgain'][1]).all()]
                x=np.array([half_vectors[i][direction]['gap_local'][0] for i in eligible])
                y=np.array([half_vectors[i][1-direction]['repgain'][0] for i in eligible])
                replicates=[]
                if eligible:
                    for r in range(len(half_vectors[eligible[0]][0]['gap_local'][1])):
                        replicates.append(rank_correlation(
                            np.array([half_vectors[i][direction]['gap_local'][1][r] for i in eligible]),
                            np.array([half_vectors[i][1-direction]['repgain'][1][r] for i in eligible])))
                lo,hi=interval(replicates)
                correlations.append(dict(seed=seed,encoder=origin,stratum=stratum,value=value,
                    gap_half=direction,repgain_half=1-direction,rho=rank_correlation(x,y),lower=lo,upper=hi,n_cells=len(eligible)))
    scarce_frame=pd.DataFrame(scarcity_rows)
    pd.DataFrame(scarcity_missing).to_csv(output/'scarcity_missing_arms.csv',index=False)
    scarce_frame.to_csv(output/'scarcity_cells.csv',index=False)
    scarce_summaries=[]
    if not scarce_frame.empty:
        for identity,g in scarce_frame.groupby(['seed','encoder','share','quantity']):
            ids=[i for i in g.index if np.isfinite(scarcity_draws[i]).all()]
            lo,hi=interval(np.mean([scarcity_draws[i] for i in ids],axis=0)) if ids else [np.nan,np.nan]
            scarce_summaries.append(dict(zip(['seed','encoder','share','quantity'],identity)) | dict(
                gain=scarce_frame.loc[ids,'gain'].mean(),lower=lo,upper=hi,n_cells=len(ids)))
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for (encoder,share),g in scarce_frame.groupby(['encoder','share']):
            ax.plot([share,100],[g.loc[g.quantity.eq('gap_fraction'),'gain'].mean(),
                                g.loc[g.quantity.eq('gap_full'),'gain'].mean()],
                    marker='o',label=f'{encoder}, matched {share}% roster')
        ax.set(xlabel='Training patients (%)',ylabel='OPERA − joint AUC');ax.legend()
        fig.savefig(output/'scarcity_gaps.pdf',bbox_inches='tight');plt.close(fig)
    pd.DataFrame(scarce_summaries).to_csv(output/'scarcity_summaries.csv',index=False)
    cells.to_csv(output/'cell_effects.csv',index=False)
    pd.DataFrame(dropped).to_csv(output/'roster_exclusions.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'closure_summaries.csv',index=False)
    pd.DataFrame(correlations).to_csv(output/'absorption_correlations.csv',index=False)
    agreement_frame=pd.DataFrame(agreement)
    agreement_frame.to_csv(output/'risk_agreement.csv',index=False)
    if not agreement_frame.empty:
        agreement_frame.groupby(['seed','model_a','model_b']).rho.agg(['mean','median','count']).reset_index().to_csv(
            output/'risk_agreement_summary.csv',index=False)
    seed_spread=cells.groupby(['encoder','seed'])[['closure','repgain','gap']].mean().reset_index()
    seed_spread.to_csv(output/'seed_results.csv',index=False)
    seed_spread.groupby('encoder')[['closure','repgain','gap']].agg(['min','max','std','count']).to_csv(
        output/'seed_spread.csv')
    manifest=dict(exploratory=True,split=split,prepared=str(prepared.resolve()),
                  input_hashes=context.get('hashes',{}),n_cells=len(cells)//2,seeds=sorted(map(int,cells.seed.unique())),
                  implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ['representation_absorption.py','correction_reliance.py']},
                  main_bootstrap_reused=True,disjoint_halves=True,scarcity_contract_hashes=scarcity_hashes,
                  limitations=['Association is not causal identification of absorption or a ceiling.',
                               'AUC level is not an estimate of irreducible uncertainty.',
                               'Event quartiles and level quartiles are descriptive, fixed from point estimates.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    figures(output,cells,half_vectors)
    blockers=['Analysis 1 requires checkpoint replay via correction_replay (separate GPU work).',
              'Analysis 3: see scarcity_missing_arms.csv for missing arms/contracts; no training launched.']
    if len(cells.seed.unique()) < 2:
        blockers.append('Only one common six-arm seed is available; training-seed spread cannot be estimated.')
    if 'n_events_train' not in cells:
        blockers.append('Training-event strata unavailable: supply the existing cell metadata with n_events_train.')
    (output/'report.md').write_text('# Exploratory representation diagnostics\n\n'
        'Verdict: no automatic mechanistic verdict; inspect paired closure intervals and disjoint-half associations. '
        'Neither analysis uniquely distinguishes absorption from a ceiling.\n\n'
        +'```text\n'+seed_spread.to_string(index=False)+'\n```\n\n'
        +'\n'.join('- '+b for b in blockers)+'\n\nSee manifest, roster exclusions and correlation tables.\n')
    return manifest


def figures(output,cells,halves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for origin in ['E','EA']:
        g=cells.loc[cells.encoder.eq(origin)]
        fig,ax=plt.subplots()
        ax.scatter(g.level,g.closure,s=5,alpha=.25)
        ax.axhline(0,color='grey'); ax.set(xlabel='Adapted joint AUC' if origin=='EA' else 'National joint AUC',ylabel='Local gap − target gap')
        fig.savefig(output/f'closure_vs_level_{origin}.pdf',bbox_inches='tight');plt.close(fig)
        summary=pd.read_csv(output/'closure_summaries.csv')
        for stratum in ['level_quartile','n_events_train_quartile']:
            selected=summary.loc[summary.encoder.eq(origin)&summary.quantity.eq('closure')&summary.stratum.eq(stratum)]
            if selected.empty:
                continue
            fig,ax=plt.subplots()
            for seed,part in selected.groupby('seed'):
                x=pd.to_numeric(part.value)
                points=ax.plot(x,part.gain,'o',label=f'seed {seed}')
                ax.vlines(x,part.lower,part.upper,color=points[0].get_color())
            ax.axhline(0,color='grey');ax.set(xlabel=stratum,ylabel='Mean closure, paired 95% interval');ax.legend()
            fig.savefig(output/f'closure_{stratum}_{origin}.pdf',bbox_inches='tight');plt.close(fig)
        ids=[int(i) for i in g.draw_index if int(i) in halves]
        fig,ax=plt.subplots()
        ax.scatter([halves[i][0]['gap_local'][0] for i in ids], [halves[i][1]['repgain'][0] for i in ids],s=5,alpha=.25)
        ax.set(xlabel='Local OPERA − joint (patient half A)',ylabel='Target joint − local joint (patient half B)')
        fig.savefig(output/f'absorption_{origin}.pdf',bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prepared-dir',type=Path)
    source.add_argument('--analysis-config',type=Path,
                        help='Build a separate prepared snapshot; never runs the reporting pipeline')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cell-metadata',type=Path)
    parser.add_argument('--split',choices=['tuning','held_out'],default='tuning')
    parser.add_argument('--held-out-ready',action='store_true')
    parser.add_argument('--scarcity-contracts',type=Path)
    parser.add_argument('--replay-plan',type=Path)
    parser.add_argument('--replay-root',type=Path,
                        help='Completed correction_replay outputs; no GPU jobs launched here')
    args=parser.parse_args()
    if args.split == 'held_out' and not args.held_out_ready:
        parser.error('Wait until main held-out evaluation completes')
    if bool(args.replay_plan) != bool(args.replay_root):
        parser.error('--replay-plan and --replay-root must be supplied together')
    if args.analysis_config:
        import yaml
        from opera.run.analyse_locked_predictions import prepare_analysis_predictions
        from opera.evaluation.distributed_analysis import save_prepared
        config=yaml.safe_load(args.analysis_config.read_text())
        config['artifacts']=[a for a in config['artifacts'] if a.get('split')==args.split]
        if not config['artifacts']:
            parser.error('Source analysis has no artifacts for this split')
        available={a.get('model') for a in config['artifacts']}
        config.setdefault('extra_contrasts',{})
        for share,arms in SCARCE_ARMS.items():
            for name,(joint,opera) in arms.items():
                if {joint,opera} <= available:
                    config['extra_contrasts'][f'REP_{share}_{name}']=f'gain({opera},{joint})'
        prepared=args.output_dir.with_name(args.output_dir.name+'.prepared_INTERNAL')
        if prepared.exists():
            raise FileExistsError(prepared)
        frame,context=prepare_analysis_predictions(config,config['estimand'])
        save_prepared(prepared,frame,context)
        (prepared/'source_analysis.yaml').write_text(args.analysis_config.read_text())
        del frame,context
        args.prepared_dir=prepared
        if args.cell_metadata is None:
            args.cell_metadata=Path(config['cell_metadata'])
    run(args.prepared_dir,args.output_dir,args.split,args.cell_metadata,args.held_out_ready,args.scarcity_contracts)
    if args.replay_plan:
        from opera.analysis.correction_reliance import run as reliance
        reliance(args.prepared_dir,args.replay_plan,args.replay_root,
                 args.output_dir/'correction_reliance',args.split,
                 common_roster=args.output_dir/'cell_effects.csv')
        report=args.output_dir/'report.md'
        text=report.read_text().replace('Analysis 1 requires checkpoint replay via correction_replay (separate GPU work).',
                                      'Analysis 1 replay processed; see correction_reliance outputs and blocked.csv.')
        effects=pd.read_csv(args.output_dir/'correction_reliance/inference_ablation_summaries.csv')
        if not effects.empty:
            text+='\n## Inference reliance (exploratory)\n\n```text\n'+effects.loc[effects.stratum.eq('overall')].to_string(index=False)+'\n```\n'
        report.write_text(text)


if __name__=='__main__':
    main()
