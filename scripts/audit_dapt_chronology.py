"""Read-only chronology and printed-cutoff evidence for original external DAPT."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess


def policy_lines(text):
    lines=text.splitlines();selected=[]
    for i,line in enumerate(lines):
        if re.search(r'cutoff_date\s*[:=]|Version:|Run dir:|Resuming training from|Auto-resuming interrupted|Initialized local pretraining encoder',line):
            selected.append(dict(line=i+1,text=line[:600]))
            if 'cutoff_date:' in line and not re.search(r'cutoff_date:\s*\S',line):
                selected.extend(dict(line=j+1,text=lines[j][:200]) for j in range(i+1,min(i+5,len(lines))))
    return selected


def info(path):
    stat=path.stat()
    row=dict(path=str(path),bytes=stat.st_size,
        modified_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),
        inode_changed_utc=datetime.fromtimestamp(stat.st_ctime,timezone.utc).isoformat())
    # GNU stat birth time is optional; ctime is explicitly not creation time.
    result=subprocess.run(['stat','-c','%w',str(path)],capture_output=True,text=True)
    if result.returncode==0 and result.stdout.strip()!='-':row['filesystem_birth_time']=result.stdout.strip()
    return row


def audit(root,logs):
    from zoneinfo import ZoneInfo
    report=dict(scope='Original DAPT chronology; file modification/ctime are not definitive training start dates',
        cutoff_commit='b2ef297d016213856d52dd78fb564e140215dcc6',
        cutoff_commit_copenhagen='2026-11-11T15:26:59+01:00',
        artifacts=[],logs=[])
    for directory in sorted(root.parent.glob('external_dapt*')):
        if not directory.is_dir():continue
        for name in ['hydra/config.yaml','hydra/overrides.yaml','hydra/hydra.yaml',
                     'version_0/hparams.yaml','version_0/metrics.csv','version_0/checkpoint_metadata.json',
                     'version_0/training_complete.json','version_0/last.ckpt','version_0/best.ckpt']:
            path=directory/name
            if not path.is_file():continue
            row=info(path)
            row['modified_copenhagen']=datetime.fromtimestamp(path.stat().st_mtime,ZoneInfo('Europe/Copenhagen')).isoformat()
            if path.suffix in ['.yaml','.log']:
                with path.open(errors='replace') as f:text=f.read(500000)
                row['cutoff_or_launch_lines']=policy_lines(text)
            report['artifacts'].append(row)
        for path in directory.rglob('events.out.tfevents.*'):
            row=info(path);match=re.search(r'tfevents\.(\d+)',path.name)
            if match:row['event_writer_filename_utc']=datetime.fromtimestamp(int(match[1]),timezone.utc).isoformat()
            report['artifacts'].append(row)
    for path in sorted(logs.glob('*dapt*')):
        if not path.is_file() or path.suffix=='.json':continue
        row=info(path)
        with path.open(errors='replace') as f:text=f.read(500000)
        row['header_bytes_scanned']=len(text.encode(errors='replace'))
        row['cutoff_or_launch_lines']=policy_lines(text)
        report['logs'].append(row)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/paper_256_v3_national_20261111/external_dapt'))
    p.add_argument('--logs',type=Path,default=Path('logs'))
    p.add_argument('--output',type=Path,default=Path('logs/dapt_chronology_20261120.json'))
    a=p.parse_args();report=audit(a.root,a.logs)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print('Download:',a.output)
    for row in report['logs']:
        print(row['path'],row['modified_utc'])
        for line in row['cutoff_or_launch_lines']:print(line)

if __name__=='__main__':main()
