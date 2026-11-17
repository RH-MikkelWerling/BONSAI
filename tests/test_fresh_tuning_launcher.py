import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from opera.run.paper_experiments import load_environment


def test_launcher_builds_fresh_tuning_only_inputs_without_old_completion(tmp_path):
    scripts=tmp_path/'scripts'
    scripts.mkdir()
    shutil.copy('scripts/run_fresh_tuning_rehearsal.sh', scripts/'run_fresh_tuning_rehearsal.sh')
    paper=tmp_path/'paper'
    (paper/'bundles/stage16').mkdir(parents=True)
    (paper/'bundles/stage16/bundle.json').write_text('{}')
    (tmp_path/'.env').write_text(f'OPERA_PAPER_ROOT={paper}\nOPERA_PAPER_INVENTORY_SPLITS=held_out\n')
    fake=tmp_path/'python'
    calls=tmp_path/'calls.jsonl'
    fake.write_text(f'''#!{sys.executable}
import json, os, sys
if sys.argv[1] == '-':
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
with open({str(calls)!r}, 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
''')
    fake.chmod(0o755)
    job=tmp_path/'fresh'
    env={**os.environ,'PYTHONPATH':str(Path.cwd()),'OPERA_PYTHON':str(fake)}
    result=subprocess.run(['bash',str(scripts/'run_fresh_tuning_rehearsal.sh'),str(job)],
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0, result.stderr
    prepared=load_environment(job/'inputs/overnight.env')
    assert prepared['OPERA_PAPER_INVENTORY_SPLITS']=='tuning'
    assert prepared['OPERA_PAPER_ANALYSIS_ESTIMAND']=='cif_ipcw'
    assert prepared['OPERA_PAPER_OPERA_MODEL']=='J_OA_OWN_L'
    assert prepared['OPERA_PAPER_TABULAR_INVENTORY_CACHE']==str(job/'inputs/tabular_inventory_cache')
    recorded=[json.loads(line) for line in calls.read_text().splitlines()]
    start=recorded[0].index('-m') + 1
    assert recorded[0][start:start+2]==['opera.run.paper_experiments','inventory']
    assert recorded[1][recorded[1].index('-m') + 1]=='opera.run.rerun_full_tuning_rehearsal'
    assert '--defer-national' in recorded[1]
    assert not (job/'inputs/completion.json').exists()
    again=subprocess.run(['bash',str(scripts/'run_fresh_tuning_rehearsal.sh'),str(job)],
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert again.returncode!=0
