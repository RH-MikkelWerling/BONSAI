"""Export provisional/final presentation figures as aggregate results appear."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def export_once(project,output_root):
    project,output_root=Path(project),Path(output_root)
    final=project/'results/analysis_manifest.json'
    complete=final.is_file()
    effects=project/('results' if complete else 'results.progress')/'effect_summaries.csv'
    if not effects.is_file():
        return False
    digest=hashlib.sha256(effects.read_bytes()).hexdigest()
    label='final' if complete else 'provisional'
    output=output_root/f'{label}_{digest[:12]}'
    marker=output/'export_complete.json'
    if marker.is_file():
        saved=json.loads(marker.read_text())
        if saved['source_sha256']!=digest:
            raise ValueError('Figure export marker does not match source')
        return complete
    if output.exists():
        raise FileExistsError(f'Incomplete figure export retained: {output}')
    subprocess.run([sys.executable,'scripts/export_encoder_architecture_figure.py',
        '--effects',str(effects),'--output-dir',str(output),'--label',
        'Held-out comparisons' if complete else 'Held-out comparisons — provisional reporting'],check=True)
    marker.write_text(json.dumps(dict(source=str(effects),source_sha256=digest,final=complete)))
    print(f'Figures ready: {output}',flush=True)
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir',type=Path,default=Path('data/paper_256_v3_national_20261111/extended_test_comparison'))
    p.add_argument('--output-dir',type=Path,default=Path('data/paper_256_v3_national_20261111/presentation_20261112'))
    p.add_argument('--wait-hours',type=float,default=36)
    a=p.parse_args()
    if a.wait_hours<=0:
        p.error('Wait duration must be positive')
    deadline=time.monotonic()+a.wait_hours*3600
    while time.monotonic()<deadline:
        if export_once(a.project_dir,a.output_dir):
            return
        time.sleep(30)
    raise TimeoutError('Figure watcher expired; saved figures and analyses are unchanged')


if __name__=='__main__':
    main()
