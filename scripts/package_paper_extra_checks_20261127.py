"""Package completed extra aggregate exports with an explicit missing-output inventory."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from scripts.audit_paper_download_archives import audit_archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/cutoff_recovery_20261121'))
    a=p.parse_args();root=a.root.resolve();exports=root/'exports'
    required={
        'encoder_roc':('test_roc_encoders_75_v1','download_test_roc_points.zip'),
        'tabular_roc':('test_roc_tabular_75_v1','download_test_roc_points.zip'),
        'risk_support_all_outcomes':('risk_support_all_outcomes_20261127','download_risk_support.zip'),
        'functional_attention':('attention_patterns_v1','download_attention_patterns.zip'),
        'correction_diagnostics':('correction_diagnostics_v1','download_correction_diagnostics.zip')}
    included=[];missing=[];audit=[]
    for label,(folder,name) in required.items():
        directory=exports/folder;archive=directory/name
        try:
            manifest=json.loads((directory/'manifest.json').read_text())
            if manifest['status']!='complete':raise ValueError('Incomplete manifest')
            report=audit_archive(archive)
            if not report['automated_check_passed']:raise ValueError('Archive requires disclosure review')
            audit.append(report);included.append(archive)
        except (OSError,ValueError,KeyError) as error:
            missing.append({'family':label,'reason':str(error)})
    for folder in ['task_similarity_sensitivity_20261127','attention_definition_sensitivity_20261127']:
        directory=exports/folder
        if not (directory/'manifest.json').exists():
            missing.append({'family':folder,'reason':'No sensitivity manifest'})
            continue
        files=[*directory.glob('*.csv'),directory/'manifest.json']
        included.extend(files)
    extension=root/'absolute_risk_ig_extension/download_absolute_risk_ig_extension.zip'
    if extension.exists():
        report=audit_archive(extension)
        if not report['automated_check_passed']:raise ValueError('Extension archive requires review')
        included.append(extension);audit.append(report)
    registered={}
    for i in range(9):
        name=f'paper_registered_{i:02d}'
        registered[name]=(root/'analyses'/name/'results/analysis_manifest.json').is_file()
    inventory={'extra_exports_status':'complete' if not missing else 'complete_with_gaps',
        'missing_extra_exports':missing,'registered_analysis_final_manifests':registered,
        'main_ig_manifest':json.loads((root/'final_interpretation/interpretation_manifest.json').read_text())
            if (root/'final_interpretation/interpretation_manifest.json').exists() else None,
        'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in included},
        'absolute_risk_extension_included':extension in included,
        'note':'Main results remain in download_cutoff_recovery.zip; these additions do not replace prior analyses. Manifest presence is not proof of a live process.'}
    # Do not export the IG manifest: it may grow internal paths/rosters in future.
    interpretation=inventory.pop('main_ig_manifest')
    inventory['main_ig_status']=interpretation.get('status') if interpretation else 'absent'
    inventory['status']=('complete' if not missing and all(registered.values())
                         and inventory['main_ig_status']=='complete' else 'complete_with_gaps')
    audit_path=exports/'extra_checks_archive_audit_20261127.json'
    audit_path.write_text(json.dumps(audit,indent=2))
    target=root/'download_extra_checks_20261127.zip'
    with ZipFile(target.with_suffix('.writing'),'w',ZIP_DEFLATED) as z:
        for file in included:z.write(file,file.relative_to(root))
        z.writestr('completion_inventory.json',json.dumps(inventory,indent=2))
    final_audit=audit_archive(target.with_suffix('.writing'))
    if not final_audit['automated_check_passed']:
        raise ValueError('Extra-check archive requires disclosure review; temporary file retained on server')
    target.with_suffix('.writing').replace(target)
    print(json.dumps(inventory,indent=2));print('DOWNLOAD:',target)


if __name__=='__main__':main()
