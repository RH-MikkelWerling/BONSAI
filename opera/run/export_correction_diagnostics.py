"""Separate correction sources and audit fitted-versus-transformed map shifts.

Fresh exploratory output; existing arrows and IG workspaces are never changed.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile,ZIP_DEFLATED
from unittest.mock import patch
import numpy as np


def anchored_displacements(projection, parts, outcomes, anchors):
    """Use transform-to-transform displacements, anchored on fitted coordinates.

    A transformed copy of unchanged shared vectors provides the null displacement.
    Isolated adapter effects are evaluated on the SAME post-reading input; their
    projected arrows need not sum because the projection is nonlinear.
    """
    transformed=projection.transform(parts['shared']);after={}
    for j,outcome in enumerate(outcomes):
        read=projection.transform(parts['read'][:,j]);final=projection.transform(parts['final'][:,j])
        read[np.all(parts['read'][:,j]==parts['shared'],axis=1)]=transformed[
            np.all(parts['read'][:,j]==parts['shared'],axis=1)]
        final[np.all(parts['final'][:,j]==parts['read'][:,j],axis=1)]=read[
            np.all(parts['final'][:,j]==parts['read'][:,j],axis=1)]
        r=anchors+read-transformed;f=anchors+final-transformed
        after[outcome,'total']=(anchors,f)
        after[outcome,'reading']=(anchors,r)
        after[outcome,'corrections']=(r,f)
        after[outcome,'projection_null']=(anchors,transformed)
        for key in parts:
            if not key.startswith('isolated_'):continue
            candidate=projection.transform(parts[key][:,j])
            same=np.all(parts[key][:,j]==parts['read'][:,j],axis=1);candidate[same]=read[same]
            after[outcome,key]=(r,r+candidate-read)
    return after


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='auto')
    a=p.parse_args()
    import pandas as pd
    import torch
    import yaml
    from opera.run import export_opera_correction_arrows as old
    from opera.run.build_paper_embedding_tables import fit_projection,matched_stage_frames
    from opera.run.interpretation_worker import claim,digest,write_json
    root=a.root.resolve();out=(a.output_dir or root/'exports/correction_diagnostics_v1').resolve()
    out.mkdir(parents=True,exist_ok=True)
    with claim(out/'.lock') as owned:
        if not owned:raise RuntimeError('Another correction diagnostic is running')
        fit=json.loads((root/'final_interpretation/interpretation_manifest.json').read_text())['fit']
        config=root/'exports/model_exports_held_out_descriptive_v1/maps_SERVER_INTERNAL/embedding_config.yaml'
        identity={'fit':fit,'embedding_config_sha256':digest(config),
            'implementation_sha256':{str(f):digest(f) for f in [Path(__file__),Path(old.__file__),
                Path(__file__).parents[1]/'modules/networks/attention_own_corrections.py',
                Path(__file__).with_name('build_paper_embedding_tables.py')]}}
        if digest(Path(fit['checkpoint']))!=fit['checkpoint_sha256'] or digest(Path(fit['config']))!=fit['config_sha256']:
            raise ValueError('Final checkpoint/config identity differs')
        intent=out/'intent_INTERNAL.json'
        if intent.exists() and json.loads(intent.read_text())!=identity:raise ValueError('Changed identity: use a fresh output directory')
        write_json(intent,identity)
        recipe=yaml.safe_load(config.read_text());maps=recipe['calibration_maps'];outcomes=maps['outcomes']
        frame=matched_stage_frames(recipe['stages'],rehearse=False)[maps['stage']]
        vectors=out/'vectors_INTERNAL.npz'
        if vectors.exists():
            with np.load(vectors,allow_pickle=False) as saved:parts={k:saved[k] for k in saved.files}
            ids=parts.pop('subject_ids')
        else:
            captured={};diagnoses=[]
            processed=0
            keep=set(frame.subject_id.astype(str))
            def decomposed(model,batch,indices):
                nonlocal processed
                hidden=model._encode_hidden(batch);shared=model._pool_hidden(hidden,batch)
                read=model.outcome_attention_readout(hidden,shared,batch['attention_mask'])
                own=model.attention_own_corrections
                terms=own.components(read,batch['cohort_fine_index'])
                final=own(read,batch['cohort_fine_index'])
                summed=read+own.residual_scale*sum(terms.values()).to(read.dtype)
                if not torch.allclose(final,summed,atol=1e-5,rtol=1e-4):raise ValueError('Correction decomposition does not reconstruct forward')
                result={'shared':shared.float(),'read':read[:,indices].float(),'final':final[:,indices].float()}
                wanted=np.isin(batch['subject_id'].cpu().numpy().astype(str),list(keep))
                fine=batch['cohort_fine_index'].cpu().numpy().reshape(-1)
                diagnoses.extend([own.fine_names[int(i)] if 0<=int(i)<len(own.fine_names) else 'Unassigned'
                                  for i in fine[wanted]])
                for key,value in terms.items():
                    isolated=(read+own.residual_scale*value.to(read.dtype))[:,indices].float()
                    captured.setdefault('isolated_'+key,[]).append(isolated[torch.as_tensor(wanted,device=isolated.device)].cpu().numpy())
                processed+=int(wanted.sum())
                if processed%50==0:print(f'Correction decomposition: {processed} patients',flush=True)
                return result
            args=SimpleNamespace(config_dir=Path(fit['config']).parent,config_name=Path(fit['config']).stem,
                checkpoint=Path(fit['checkpoint']),override=[],batch_size=1,num_workers=0,device=a.device,data_split='held_out')
            with patch.object(old,'representation_parts',decomposed):ids,parts=old._collect(args,outcomes,keep)
            parts.update({k:np.concatenate(v) for k,v in captured.items()})
            parts['diagnosis_INTERNAL']=np.asarray(diagnoses,dtype=str)
            with vectors.with_suffix('.writing').open('wb') as f:np.savez_compressed(f,subject_ids=ids,**parts)
            vectors.with_suffix('.writing').replace(vectors)
            if torch.cuda.is_available():torch.cuda.empty_cache()
        diagnoses=parts.pop('diagnosis_INTERNAL')
        projection,shown=fit_projection(frame,method=recipe.get('method','umap'),rehearse=False)
        if set(ids)!=set(shown.subject_id.astype(str)) or len(ids)!=6446:raise ValueError('Unexpected displayed roster')
        reference=shown.set_index(shown.subject_id.astype(str)).loc[ids]
        if not np.allclose(parts['shared'],np.stack(reference.embedding),atol=1e-4,rtol=1e-3):raise ValueError('Shared representations differ from map stage')
        anchors=projection.fitted_coordinates(ids)
        after=anchored_displacements(projection,parts,outcomes,anchors)
        arrows,density=old.arrow_aggregates(anchors,after,min_count=20)
        arrows.to_csv(out/'correction_arrows.csv',index=False);density.to_csv(out/'correction_arrow_density.csv',index=False)
        rows=[]
        for j,outcome in enumerate(outcomes):
            pairs={'reading':(parts['shared'],parts['read'][:,j]),
                   'corrections':(parts['read'][:,j],parts['final'][:,j]),
                   'total':(parts['shared'],parts['final'][:,j])}
            pairs.update({k:(parts['read'][:,j],v[:,j]) for k,v in parts.items() if k.startswith('isolated_')})
            for key,(before,end) in pairs.items():
                ratio=np.linalg.norm(end-before,axis=1)/np.maximum(np.linalg.norm(before,axis=1),1e-12)
                cosine=(before*end).sum(1)/np.maximum(np.linalg.norm(before,axis=1)*np.linalg.norm(end,axis=1),1e-12)
                strata=[('all',np.ones(len(ids),dtype=bool))]
                strata.extend((str(d),diagnoses==d) for d in sorted(set(diagnoses)))
                for disease,mask in strata:
                    if int(mask.sum())<20:continue
                    rows.append(dict(outcome=outcome,component=key,diagnosis=disease,n_patients=int(mask.sum()),
                        median_relative_norm=float(np.median(ratio[mask])),p90_relative_norm=float(np.quantile(ratio[mask],.9)),
                        median_before_after_cosine=float(np.median(cosine[mask]))))
        pd.DataFrame(rows).to_csv(out/'representation_shift_summary.csv',index=False)
        write_json(out/'manifest.json',{'status':'complete','patients':len(ids),'checkpoint_sha256':fit['checkpoint_sha256'],
            'outcomes':outcomes,'components':sorted({c for _,c in after}),
            'displacement':'Transform-to-transform differences anchored on fitted coordinates; zero-input transform displacement exported as projection_null',
            'decomposition':'Each isolated correction acts on the same post-reading representation; projected components are not additive',
            'scope':'Exploratory projection/representation diagnostic, not risk change or performance',
            'projection_version':'held_out_descriptive_v1'})
        files=['correction_arrows.csv','correction_arrow_density.csv','representation_shift_summary.csv','manifest.json']
        target=out/'download_correction_diagnostics.zip'
        with ZipFile(target.with_suffix('.writing'),'w',ZIP_DEFLATED) as z:
            for name in files:z.write(out/name,name)
        target.with_suffix('.writing').replace(target)
        print('DOWNLOAD:',target,flush=True)


if __name__=='__main__':main()
