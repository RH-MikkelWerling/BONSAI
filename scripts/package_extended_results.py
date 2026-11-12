"""Package completed aggregate test outputs for local plotting; no patient exports."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def package(project,output,local_root=None):
    project,output=Path(project),Path(output)
    results=project/'results'
    manifest_path=results/'analysis_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Main analysis has not published final outputs yet')
    source_hash=file_hash(manifest_path)
    if output.exists():
        marker=output/'bundle_manifest.json'
        if not marker.is_file() or json.loads(marker.read_text())['source_manifest_sha256']!=source_hash:
            raise FileExistsError('Existing bundle is incomplete or has another source; use a fresh output path')
        archive=output.with_suffix('.tar.gz')
        if not archive.is_file():
            make_archive(output,archive)
        return archive
    temporary=output.with_name('.'+output.name+'.writing_'+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        tables=temporary/'tables';tables.mkdir()
        excluded=[]
        for path in sorted(results.glob('*.csv')):
            if '_INTERNAL' in path.name or path.name=='paired_bootstrap_index.csv':
                excluded.append(dict(file=path.name,reason='Internal bootstrap or patient-level output'))
                continue
            with path.open(newline='') as handle:
                columns=next(csv.reader(handle),[])
            if any(c.lower() in {'subject_id','subject_ids','patient_id','patientid'} for c in columns):
                excluded.append(dict(file=path.name,reason='Patient identifier column'))
                continue
            with path.open('rb') as source,gzip.open(tables/(path.name+'.gz'),'wb',compresslevel=1) as target:
                shutil.copyfileobj(source,target,length=1024*1024)
            print(f'Bundled aggregate table: {path.name}',flush=True)
        shutil.copy2(manifest_path,temporary/'analysis_manifest.json')
        if (project/'analysis.yaml').is_file():
            shutil.copy2(project/'analysis.yaml',temporary/'analysis.yaml')
        shutil.copy2('scripts/export_encoder_architecture_figure.py',temporary/'render_comparison.py')
        subprocess.run([sys.executable,'scripts/export_encoder_architecture_figure.py',
            '--effects',str(results/'effect_summaries.csv'),'--output-dir',str(temporary/'figures'),
            '--label','Held-out comparisons'],check=True)
        local_complete=False
        if local_root:
            local_root=Path(local_root)
            local_complete=(local_root/'pipeline_actions_complete').is_file()
            if local_complete and (local_root/'figures').is_dir():
                for source in (local_root/'figures').rglob('*'):
                    if source.is_file() and source.suffix.lower() in {'.pdf','.png','.svg'}:
                        target=temporary/'local_model_figures'/source.relative_to(local_root/'figures')
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        source_manifest=json.loads(manifest_path.read_text())
        config=source_manifest.get('config',{})
        remaining=dict(
            registry_refresh='Separate workspace; existing XR does not certify refreshed registry inputs',
            national_clinical_comparisons='Not included by the focused extended comparison recipe',
            national_identity_sensitivity='Disabled in the focused extended comparison recipe',
            national_patient_subgroups='Disabled in the focused extended comparison recipe',
            national_route_and_attribution_exports='Separate inference/figure work; not supplied by aggregate comparisons',
            label_efficiency='Fraction-arm predictions and their analysis are separate from external full-data fits',
            representation_diagnostics='Separate exploratory output directory',
            xpr='Not implemented in the current comparator matrix')
        metadata=dict(scope='Completed extended test aggregate results and available figures; not the entire paper release',
            source_manifest_sha256=source_hash,excluded=excluded,remaining=remaining,
            local_figure_pipeline_complete=local_complete,
            primary_alias_in_source=config.get('opera_model'),
            model_note='Fixed-arm EXT_* comparisons identify architectures explicitly; do not infer selected model from an old OPERA alias',
            files={str(p.relative_to(temporary)):file_hash(p) for p in temporary.rglob('*') if p.is_file()})
        (temporary/'bundle_manifest.json').write_text(json.dumps(metadata,indent=2))
        (temporary/'README.md').write_text(
            '# Extended test results for local iteration\n\n'
            'CSV files are gzip-compressed aggregate outputs; pandas reads them directly. '
            'Patient predictions, patient gains and bootstrap arrays are excluded.\n\n'
            'The encoder/architecture figure is in figures/. Rebuild locally with:\n\n'
            '```bash\npython render_comparison.py --effects tables/effect_summaries.csv.gz '
            '--output-dir figures_revised --label "Held-out comparisons"\n```\n\n'
            'Requires pandas and matplotlib. See bundle_manifest.json for checksums, '
            'the source model alias and all remaining analyses. This package is not a declaration '
            'that clinical refresh, fraction-arm analyses or national interpretation figures are complete.\n')
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    archive=output.with_suffix('.tar.gz')
    make_archive(output,archive)
    print(f'Download bundle ready: {archive}',flush=True)
    return archive


def make_archive(output,archive):
    temporary=archive.with_name('.'+archive.name+'.writing_'+uuid.uuid4().hex)
    try:
        with tarfile.open(temporary,'w:gz',compresslevel=1) as handle:
            handle.add(output,arcname=output.name)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir',type=Path,default=Path('data/paper_256_v3_national_20261111/extended_test_comparison'))
    p.add_argument('--output-dir',type=Path,default=Path('data/paper_256_v3_national_20261111/download_bundle_20261112'))
    p.add_argument('--local-root',type=Path,default=Path('data/paper_256_v3/analysis/local_test_20261109_four_allocations'))
    p.add_argument('--wait-hours',type=float,default=0)
    a=p.parse_args();deadline=time.monotonic()+a.wait_hours*3600
    while not (a.project_dir/'results/analysis_manifest.json').is_file():
        if time.monotonic()>=deadline:
            p.error('Final results are not ready; use --wait-hours to wait without restarting analysis')
        time.sleep(30)
    package(a.project_dir,a.output_dir,a.local_root)


if __name__=='__main__':
    main()
