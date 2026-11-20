"""Package only aggregate matched-control tables, with checkpoint compute facts."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package(root, kind):
    from opera.run.prepare_epoch20_joint_finetune import checkpoint_record
    root=Path(root);project=root/f'comparisons_{kind}'
    results=project/'results'
    if not (results/'analysis_manifest.json').is_file():raise ValueError('Matched analysis not complete')
    national=Path('data/paper_256_v3_national_20261111')
    cohort=json.loads(Path('data/epoch20_joint_control_20261119/preparation.json').read_text())
    sources={'cohort10':cohort['encoder10']['path'],'cohort20':cohort['encoder20']['path']}
    import yaml
    for label,bundlepath,name in [('national',national/'direct_fits','seed42/E/J'),('adapted10',national/'adapted_fits','seed42/EA/J')]:
        bundle=json.loads((bundlepath/'bundle.json').read_text());row,=[r for r in bundle['runs'] if r['name']==name]
        sources[label]=yaml.safe_load((bundlepath/row['config_path']).read_text())['dapt_ckpt']
    if kind=='adapted20':sources['adapted20']='data/adapted20_control_20261120/pretraining/version_0/last.ckpt'
    facts={label:checkpoint_record(Path(path)) for label,path in sources.items()}
    facts['note']='Checkpoint step/token counters are local to the recorded training stage; original national totals may be unavailable. No equal-compute claim.'
    compute=project/'pretraining_compute.json';compute.write_text(json.dumps(facts,indent=2))
    target=root/('download_adapted20_matched.zip' if kind=='adapted20' else 'download_epoch20_matched.zip')
    names=['effect_summaries.csv','effect_summaries_disease_balanced.csv','contrasts.csv',
           'panel_estimates.csv','contrast_coverage.csv','analysis_status.csv']
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name in names:
            path=results/name
            if not path.is_file():continue
            with path.open(newline='') as f:
                if {'subject_id','patient_id','cpr','pnr'} & {c.lower() for c in next(csv.reader(f))}:
                    raise ValueError('Patient column in aggregate table')
            z.write(path,name)
        z.write(project/'preparation_report.json','preparation_report.json')
        z.write(compute,'pretraining_compute.json')
    print(f'Download: {target}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/epoch20_matched_20261120'))
    p.add_argument('--kind',choices=['epoch20','adapted20'],default='epoch20')
    a=p.parse_args();package(a.root,a.kind)
