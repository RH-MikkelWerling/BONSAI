"""Resume forward-only attention comparisons; export task aggregates, never token weights."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='auto')
    p.add_argument('--expected-patients', type=int, default=6446)
    p.add_argument('--max-new-patients', type=int, default=0, help='0: entire held-out split; otherwise resumable pilot')
    a = p.parse_args()
    if a.max_new_patients < 0 or a.expected_patients < 20:
        raise ValueError('Invalid patient limits')
    import numpy as np
    import pandas as pd
    import torch
    from dotenv import load_dotenv
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from bonsai.functional.checkpointing import get_saved_encoder_config
    from bonsai.functional.input_contract import resolve_numeric_value_control
    from opera.diagnostics.representation_gradient_conflict import _load_model
    from opera.functional.direct_cr_contract import needs_fine_index
    from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
    from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
        MultiCohortContrastiveDataModule, compute_pooled_event_time_probability_grids)
    from opera.interpretability.attention_patterns import extract_statistics
    from opera.run.interpretation_worker import claim, digest, write_json
    from opera.run.export_opera_task_structure import within_between
    from opera.visualization.paper_taxonomy import outcome_endpoint_groups

    root = a.recovery_root.resolve()
    out = (a.output_dir or root/'exports/attention_patterns_v1').resolve()
    out.mkdir(parents=True, exist_ok=True)
    with claim(out/'.lock') as owned:
        if not owned:
            raise RuntimeError('Another attention exporter owns this output')
        fit = json.loads((root/'final_interpretation/interpretation_manifest.json').read_text())['fit']
        if fit['model'] != 'R_EA' or fit['split'] != 'held_out':
            raise ValueError('Expected final adapted OPERA held-out identity')
        config, checkpoint_path = Path(fit['config']), Path(fit['checkpoint'])
        if digest(config) != fit['config_sha256'] or digest(checkpoint_path) != fit['checkpoint_sha256']:
            raise ValueError('Final-model identity changed')
        load_dotenv()
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config.parent.resolve()), version_base='1.2'):
            cfg = compose(config_name=config.stem)
        identity = {'checkpoint_sha256':fit['checkpoint_sha256'], 'config_sha256':fit['config_sha256'],
                    'expected_patients':a.expected_patients,
                    'resolved_config':OmegaConf.to_container(cfg, resolve=True),
                    'implementation':{str(f):digest(f) for f in [Path(__file__),
                        Path(__file__).parents[1]/'interpretability/attention_patterns.py',
                        Path(__file__).parents[1]/'modules/networks/opera_nets.py']}}
        def input_stamps(value):
            result = {}
            if isinstance(value, dict):
                for item in value.values():result.update(input_stamps(item))
            elif isinstance(value, list):
                for item in value:result.update(input_stamps(item))
            elif isinstance(value, str) and len(value) < 4096:
                file = Path(value)
                if file.is_file():
                    stat = file.stat()
                    result[str(file.resolve())] = [stat.st_size, stat.st_mtime_ns]
            return result
        identity['input_file_stat_stamps'] = input_stamps(identity['resolved_config'])
        intent = out/'intent_INTERNAL.json'
        if intent.exists() and json.loads(intent.read_text()) != identity:
            raise ValueError('Inputs/code changed; use a fresh output directory')
        write_json(intent, identity)
        cohorts = OmegaConf.to_container(cfg.cohorts, resolve=True)
        outcomes = OmegaConf.to_container(cfg.outcomes, resolve=True)
        grids, probabilities = compute_pooled_event_time_probability_grids(cohorts, outcomes,
            split='train', require_all_configured_cells=bool(cfg.training.get('require_all_configured_cells', True)))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        hp = checkpoint['hyper_parameters']
        vocabulary = torch.load(cfg.paths.vocabulary, map_location='cpu', weights_only=False)
        competing = OmegaConf.to_container(cfg.competing_risk, resolve=True)
        dm = MultiCohortContrastiveDataModule(cohort_configs=cohorts, outcome_configs=outcomes,
            predict_token_id=int(vocabulary['[CLS]']), batch_size=1, num_workers=0,
            require_all_configured_cells=bool(cfg.training.get('require_all_configured_cells', True)),
            require_min_followup_val=False, max_len=int(get_saved_encoder_config(hp)['max_seqlen']),
            numeric_value_control=resolve_numeric_value_control(cfg.training.get('numeric_value_control','inherit'), hp),
            fine_names=cfg.competing_risk.get('fine_names') if needs_fine_index(competing) else None,
            fine_column=str(cfg.competing_risk.get('fine_column','cohort_fine')))
        device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
        model = _load_model(checkpoint_path, cfg, sorted(outcomes), grids, probabilities, device)
        model.eval()
        names = list(model.outcome_attention_readout.outcome_names)
        del checkpoint, vocabulary
        cache = out/'patients_INTERNAL';cache.mkdir(exist_ok=True)
        seen, paths, new = set(), [], 0
        complete = True
        stop = False
        inference_seconds = 0.
        for cohort, definition in cohorts.items():
            ds = dm._build_dataset_for_cohort(cohort, definition, 'held_out')
            if ds is None:continue
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=contrastive_collate)
            with torch.inference_mode():
                for raw in loader:
                    subject = str(raw['subject_id'].item())
                    key = hashlib.sha256(subject.encode()).hexdigest()
                    if key in seen:raise ValueError('Duplicate patient across cohort datasets')
                    seen.add(key)
                    path = cache/(key+'.npz')
                    if not path.exists():
                        if a.max_new_patients and new >= a.max_new_patients:
                            complete = False;stop = True;break
                        batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
                        if device.type == 'cuda':torch.cuda.synchronize()
                        started = time.monotonic()
                        values = extract_statistics(model, batch)
                        if device.type == 'cuda':torch.cuda.synchronize()
                        inference_seconds += time.monotonic() - started
                        with path.with_suffix('.writing').open('wb') as handle:
                            np.savez_compressed(handle, **{k:v[0].cpu().numpy().astype('float32') for k,v in values.items()})
                        path.with_suffix('.writing').replace(path)
                        new += 1
                    paths.append(path)
                    if len(paths) % 50 == 0:
                        print(f'Attention patterns: {len(paths)} patients; {new} newly computed', flush=True)
            if stop:break
        if not paths:raise ValueError('No held-out patients')
        status = {'status':'complete' if complete else 'pilot_complete', 'patients':len(paths),
                  'new_patients':new, 'scope':'Exploratory final-model held-out attention; no fitting or IG',
                  'inference_seconds_this_attempt':inference_seconds,
                  'mean_inference_seconds_per_new_patient':inference_seconds/new if new else None,
                  'checkpoint_sha256':fit['checkpoint_sha256'], 'outcomes':names,
                  'aggregation':'Corresponding-head cosine within patient; equal-head then equal-patient mean',
                  'centred':'Subtract uniform attention on the valid reading tokens; undefined heads omitted',
                  'uncertainty':'Descriptive patient-level standard errors, not paired model-comparison intervals',
                  'note':'Input stat stamps detect file replacement but are not cryptographic hashes of sequence contents'}
        if not complete:
            write_json(out/'progress.json', status)
            print('Pilot saved; rerun without --max-new-patients to finish', flush=True)
            return
        if len(paths) < 20:raise ValueError('Too few patients for aggregate export')
        if len(paths) != a.expected_patients:
            raise ValueError(f'Held-out roster count differs: expected {a.expected_patients}, found {len(paths)}')
        sums, squares, counts = {}, {}, {}
        for path in paths:
            with np.load(path, allow_pickle=False) as record:
                for key in record.files:
                    x = record[key].astype('float64');valid = np.isfinite(x);x = np.nan_to_num(x)
                    if key not in sums:
                        sums[key]=np.zeros_like(x);squares[key]=np.zeros_like(x);counts[key]=np.zeros_like(x,dtype=int)
                    sums[key]+=x;squares[key]+=x*x;counts[key]+=valid
        files, statistics = [], {}
        family = outcome_endpoint_groups()
        for key in ('attention_cosine','centred_attention_cosine'):
            n = counts[key]
            mean = np.divide(sums[key], n, out=np.full_like(sums[key],np.nan), where=n>=20)
            variance = np.divide(squares[key]-np.divide(sums[key]**2,n,out=np.zeros_like(sums[key]),where=n>0),
                                 n-1,out=np.full_like(mean,np.nan),where=n>=20)
            se = np.sqrt(np.maximum(variance,0)/np.maximum(n,1))
            for suffix,data in [('',mean),('_se',se),('_patients',np.where(n>=20,n,np.nan))]:
                name=key+suffix+'.csv';pd.DataFrame(data,index=names,columns=names).to_csv(out/name);files.append(name)
            known = [i for i,name in enumerate(names) if name in family]
            sub = mean[np.ix_(known,known)]
            if np.isfinite(sub).all():statistics[key]=within_between(sub,[family[names[i]] for i in known])
        n = counts['normalised_entropy']
        entropy = np.divide(sums['normalised_entropy'],n,out=np.full(len(names),np.nan),where=n>=20)
        pd.DataFrame({'outcome':names,'endpoint_family':[family.get(x) for x in names],
            'normalised_attention_entropy':entropy,'entropy_patients':np.where(n>=20,n,np.nan),
            'reading_gate':model.outcome_attention_readout.residual_gates.detach().tanh().abs().cpu().numpy()}).to_csv(out/'outcome_attention_summary.csv',index=False)
        files.append('outcome_attention_summary.csv')
        status['clinical_group_statistics']=statistics
        write_json(out/'manifest.json',status);files.append('manifest.json')
        archive=out/'download_attention_patterns.zip'
        with ZipFile(archive.with_suffix('.writing'),'w',ZIP_DEFLATED) as z:
            for name in files:z.write(out/name,name)
        archive.with_suffix('.writing').replace(archive)
        print('DOWNLOAD:',archive,flush=True)


if __name__ == '__main__':main()
