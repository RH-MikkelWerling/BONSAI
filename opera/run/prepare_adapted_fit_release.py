"""Audit completed DAPT and release the pending adapted matrix with correct phases."""
import argparse
import json
from pathlib import Path
from bonsai.run.audit_dapt_to_opera import audit_dapt_to_opera
from opera.run.run_paper_queue import audit_encoder, load_run_config
from opera.run.prepare_paper_run_bundle import write_bundle


def prepare(root):
    root=Path(root).resolve()
    source=root/'adapted_fits_pending'
    destination=root/'adapted_fits'
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; reuse its immutable bundle')
    bundle=json.loads((source/'bundle.json').read_text())
    if len(bundle['runs'])!=9:
        raise ValueError('Expected nine adapted fits')
    entries=[]
    audit=None
    for row in bundle['runs']:
        _,config,_=load_run_config(row,source)
        if row['origin']!='EA':
            raise ValueError('Unexpected non-adapted fit')
        if audit is None:
            audit=audit_encoder(config,'EA',expected_width=256,minimum_local_epochs=10,minimum_dapt_epochs=10)
            audit_dapt_to_opera(Path(config['dapt_ckpt']),Path(config['paths']['vocabulary']))
        elif config['dapt_ckpt']!=entries[0]['config']['dapt_ckpt']:
            raise ValueError('Adapted fits must share one checkpoint')
        entry={k:v for k,v in row.items() if k in ('name','phase','origin','seed','condition')}
        # Fraction fits must retain their fraction-specific prediction aliases.
        entry['phase']='data_efficiency' if '/J_frac' in row['name'] else 'external'
        entry['config']=config
        entries.append(entry)
    report=write_bundle(entries,destination,provenance={
        'pending_bundle':str(source),'encoder_audit':audit,
        'change':'Fraction jobs assigned data_efficiency phase; scientific configs unchanged'})
    print(json.dumps({'bundle':str(destination),'phase_counts':report['phase_counts']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--national-root',type=Path,required=True)
    prepare(p.parse_args().national_root)


if __name__=='__main__':
    main()
