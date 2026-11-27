"""Read-only inventory of configured subgroup calculations and exportable tables."""
import argparse
import csv
import json
from pathlib import Path
import yaml
from opera.evaluation.aggregate_export_files import aggregate_export_paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/cutoff_recovery_20261121'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();reports=[]
    sources=root/'registered_analyses.yaml'
    if not sources.exists():
        plan=root/'queue_plan.yaml'
        if plan.exists():sources=Path(yaml.safe_load(plan.read_text())['analyses'])
    declarations=yaml.safe_load(sources.read_text()) if sources.exists() else []
    for i in range(9):
        name=f'paper_registered_{i:02d}';directory=root/'analyses'/name
        path=directory/'analysis.yaml';prepared=path.exists()
        if not prepared and i<len(declarations):path=Path(declarations[i]['source'])
        config=yaml.safe_load(path.read_text()) if path.exists() else {}
        results=directory/'results';completed=(results/'analysis_manifest.json').exists()
        statuses=[]
        if (results/'analysis_status.csv').exists():
            with (results/'analysis_status.csv').open() as f:
                statuses=[dict(analysis=r.get('analysis'),status=r.get('status')) for r in csv.DictReader(f)
                          if r.get('analysis','').startswith(('stratified_','patient_','identity_'))]
        configured=config.get('stratified_reanalysis') or {}
        tables=aggregate_export_paths(results,set())
        reports.append(dict(analysis=name,prepared=prepared,final_manifest=completed,
            stratified_reanalysis={key:{'models':value.get('models',['J_L','X']),
                                      'contrasts':value.get('contrasts','all configured contrasts')}
                                   for key,value in configured.items()},
            patient_groups=list(config.get('patient_groups') or {}),
            statuses=statuses,subgroup_aggregate_files=[x.name for x in tables],
            expected_stratum_tables_present={key:all((results/f'stratified_{key}_{kind}.csv').exists()
                                                     for kind in ['effects','models']) for key in configured}))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'scope':'Aggregate receipts/configuration only; no live-process claim',
                                   'registered_analyses':reports},indent=2))
    for r in reports:
        print(r['analysis'],'final=',r['final_manifest'],'strata=',list(r['stratified_reanalysis']),
              'tables=',r['expected_stratum_tables_present'],'statuses=',r['statuses'])
    print('REPORT:',a.output)


if __name__=='__main__':main()
