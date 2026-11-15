"""Read-only analysis 6 feasibility audit. Never tokenizes, trains or evaluates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def coverage(counts, vocabulary, mapping=None):
    """Event- and type-weighted exact coverage; never truncate or infer a mapping."""
    mapping={} if mapping is None else mapping
    rows=[]
    for code,n in sorted(counts.items()):
        target=mapping.get(code,code)
        rows.append(dict(code=code,mapped_code=target,events=n,
                         direct_known=code in vocabulary and code!='[UNK]',
                         mapped_known=target in vocabulary and target!='[UNK]'))
    frame=pd.DataFrame(rows,columns=['code','mapped_code','events','direct_known','mapped_known'])
    total=sum(counts.values())
    summary=dict(events=total,code_types=len(counts),
        direct_known_events=sum(r['events'] for r in rows if r['direct_known']),
        mapped_known_events=sum(r['events'] for r in rows if r['mapped_known']))
    for mode in ['direct','mapped']:
        summary[mode+'_known_event_share']=summary[mode+'_known_events']/total if total else None
        summary[mode+'_known_type_share']=sum(r[mode+'_known'] for r in rows)/len(rows) if rows else None
    return frame,summary


def timing_estimates(plan):
    """Conditional same-GPU step scaling; vocabulary sensitivity is not a measured bound."""
    rows=[]
    for stage in ['pretraining','joint','opera']:
        reference=plan.get(stage)
        if not reference:
            rows.append(dict(stage=stage,status='blocked_missing_timing_reference'));continue
        path=Path(reference['measurement_file'])
        if not path.is_file():
            raise FileNotFoundError(path)
        for key in ['reference_seconds','reference_optimizer_steps','target_optimizer_steps',
                    'reference_vocab_size','target_vocab_size']:
            if reference[key]<=0:
                raise ValueError(f'{stage}: {key} must be positive')
        if not reference.get('same_architecture_batch_sequence_and_gpu'):
            rows.append(dict(stage=stage,status='blocked_unmatched_reference'));continue
        base=reference['reference_seconds']*reference['target_optimizer_steps']/reference['reference_optimizer_steps']/3600
        sensitivity=base*max(1,reference['target_vocab_size']/reference['reference_vocab_size']) if stage=='pretraining' else base
        rows.append(dict(stage=stage,status='conditional_estimate',gpu=reference['gpu'],
            step_scaled_gpu_hours=base,vocab_linear_sensitivity_gpu_hours=sensitivity,
            reference_sha256=digest(path),reference=str(path),
            assumption='Same GPU, batch, sequence length and architecture; excludes preprocessing/queue time. '
                       'Vocab-linear sensitivity is a scenario, not a measured upper bound.'))
    return pd.DataFrame(rows)


def run(records, vocabulary_path, output, mapping_path=None, timing_plan=None, cutoff='2022-01-01',
        include_static=False, code_column='code', time_column='time'):
    import pyarrow.parquet as pq
    from opera.evaluation.vocabulary_embeddings import load_vocabulary
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    files=sorted({p for item in records for p in ([Path(item)] if Path(item).is_file() else Path(item).rglob('*.parquet'))})
    if not files:
        raise ValueError('No cohort-record parquet files supplied')
    vocabulary=load_vocabulary(vocabulary_path)
    if '[UNK]' not in vocabulary:
        raise ValueError('National vocabulary lacks [UNK]')
    mapping={}
    if mapping_path:
        path=Path(mapping_path)
        frame=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
        if not {'code','adaptive/mapped_code'}<=set(frame) or frame.code.duplicated().any():
            raise ValueError('Require unique explicit code -> adaptive/mapped_code mapping')
        if frame[['code','adaptive/mapped_code']].isna().any().any():
            raise ValueError('Null mapping entries')
        mapping=dict(zip(frame.code,frame['adaptive/mapped_code']))
    counts=Counter();static_rows=excluded_future=missing_codes=0
    shards=[]
    boundary=pd.Timestamp(cutoff,tz='UTC')
    for path in files:
        parquet=pq.ParquetFile(path)
        if not {code_column,time_column}<=set(parquet.schema_arrow.names):
            raise ValueError(f'{path}: expected finalized cohort MEDS records with {code_column}, {time_column}')
        for batch in parquet.iter_batches(batch_size=65536,columns=[code_column,time_column]):
            d=batch.to_pandas()
            times=pd.to_datetime(d[time_column],utc=True,errors='raise')
            static_rows+=int(times.isna().sum());excluded_future+=int((times>=boundary).sum())
            mask=times.lt(boundary)|(times.isna() if include_static else False)
            codes=d.loc[mask,code_column];missing_codes+=int(codes.isna().sum())
            counts.update(codes.dropna().astype(str).value_counts().to_dict())
        shards.append(dict(path=str(path.resolve()),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
        print(f'Audited {path}',flush=True)
    frame,summary=coverage(counts,vocabulary,mapping)
    summary.update(cutoff=cutoff,include_static=include_static,static_rows=static_rows,
                   excluded_post_cutoff_rows=excluded_future,missing_codes=missing_codes,
                   scope='Exact code lookup on supplied cohort records, not end-to-end tokenization coverage. '
                         'Inserted special tokens excluded; static-event policy explicit. '
                         'No national numeric transforms or missing records reconstructed.')
    timings=timing_estimates(json.loads(Path(timing_plan).read_text()) if timing_plan else {})
    output.mkdir(parents=True)
    frame.to_csv(output/'code_coverage.csv',index=False)
    source=frame.assign(source=frame.code.str.split('//').str[0])
    grouped=[]
    for family,g in source.groupby('source'):
        _,s=coverage(dict(zip(g.code,g.events)),vocabulary,mapping);grouped.append(dict(source=family,**s))
    pd.DataFrame(grouped).to_csv(output/'coverage_by_source.csv',index=False)
    timings.to_csv(output/'gpu_time_estimates.csv',index=False)
    manifest=dict(exploratory=True,analysis=6,training_launched=False,approval_required=True,
                  vocabulary_sha256=digest(vocabulary_path),vocabulary_size=len(vocabulary),
                  mapping_sha256=digest(mapping_path) if mapping_path else None,
                  timing_plan_sha256=digest(timing_plan) if timing_plan else None,
                  implementation_sha256=digest(__file__),record_shards=shards,coverage=summary,
                  blockers=['Verify exact cohort patients/splits, local architecture and seed policy against frozen pretraining config.',
                            'Verify national code/numeric preprocessing and tokenizer configuration; exact membership alone is insufficient.',
                            'Obtain author confirmation before any pretraining or fine-tuning.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'report.md').write_text('# Vocabulary-controlled pretraining: preflight only\n\n'
        'No training launched. Coverage describes supplied records, not inferred records or tokenizer equivalence.\n\n'
        +'```json\n'+json.dumps(summary,indent=2)+'\n```\n\n```text\n'+timings.to_string(index=False)+'\n```\n\n'
        +'\n'.join('- '+s for s in manifest['blockers']))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,action='append',required=True,help='Final cohort MEDS data shard or data/ directory; supply cohort pretraining patients, not national records')
    p.add_argument('--national-vocabulary',type=Path,required=True)
    p.add_argument('--national-code-mapping',type=Path)
    p.add_argument('--timing-plan',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--include-static',action='store_true',help='Include null-time rows only if frozen local pretraining did so')
    p.add_argument('--time-column',default='time')
    p.add_argument('--code-column',default='code')
    a=p.parse_args()
    run(a.records,a.national_vocabulary,a.output_dir,a.national_code_mapping,a.timing_plan,
        include_static=a.include_static,code_column=a.code_column,time_column=a.time_column)


if __name__=='__main__':
    main()
