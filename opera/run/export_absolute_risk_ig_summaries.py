"""Summarise the additive absolute-risk IG pass with the existing aggregate recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/cutoff_recovery_20261121'))
    a=p.parse_args();root=a.root.resolve()
    from opera.run.interpretation_worker import claim, digest, write_json
    from scripts.audit_paper_download_archives import audit_archive
    import yaml
    from opera.run.summarise_attributions import run
    source=root/'absolute_risk_ig_extension';out=source/'aggregate_export'
    config_path=root/'final_interpretation/attribution.yaml'
    original=json.loads((source/'attribution_run.json').read_text())
    if not original.get('complete'):raise ValueError('Collect the completed extension before summarising')
    out.mkdir(parents=True,exist_ok=True)
    with claim(out/'.lock') as owned:
        if not owned:raise RuntimeError('Another extension summariser is active')
        identity={'source_sha256':digest(source/'attribution_run.json'),
                  'config_sha256':digest(config_path)}
        receipt=out/'complete_INTERNAL.json';summary=out/'summary'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['identity']!=identity:raise ValueError('Extension summary inputs changed')
            for name,value in saved['files'].items():
                if digest(summary/name)!=value:raise ValueError('Aggregate extension summary changed')
        else:
            # Only incomplete derived summaries are rebuilt. Gradient caches stay intact.
            staging=out/'summary.writing_INTERNAL'
            if staging.exists():shutil.rmtree(staging)
            run(source,yaml.safe_load(config_path.read_text()),staging,horizon_days=365)
            if summary.exists():raise ValueError('Unverified summary exists; inspect before replacing')
            staging.replace(summary)
            files=[*summary.glob('*.csv'),summary/'attribution_numbers.json',summary/'summary_manifest.json']
            write_json(receipt,{'identity':identity,'files':{f.name:digest(f) for f in files}})
        files=[*summary.glob('*.csv'),summary/'attribution_numbers.json',summary/'summary_manifest.json']
        manifest={'status':'complete','scope':'Additional absolute-risk IG targets only; combine with original four risk targets from main interpretation',
                  'horizon_days':365,'source_identity':identity,
                  'scale_note':'Risk-only extension atlas is on absolute-risk scale; do not relabel as log-odds',
                  'targets':original.get('columns',[])}
        archive=source/'download_absolute_risk_ig_extension.zip'
        with ZipFile(archive.with_suffix('.writing'),'w',ZIP_DEFLATED) as z:
            for file in files:z.write(file,file.name)
            z.writestr('manifest.json',json.dumps(manifest,indent=2))
        audit=audit_archive(archive.with_suffix('.writing'))
        write_json(out/'archive_audit.json',audit)
        if not audit['automated_check_passed']:raise ValueError('Extension archive requires disclosure review')
        archive.with_suffix('.writing').replace(archive)
        print('DOWNLOAD:',archive)


if __name__=='__main__':main()
