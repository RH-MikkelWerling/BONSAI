"""Render a portable OPERA plotting ZIP locally, with no server or model access."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def extract_offline(archive, destination):
    """Extract only the portable payload; ignore large analysis backups."""
    with ZipFile(archive) as reader:
        for entry in reader.infolist():
            if not entry.filename.startswith('offline/'):
                continue
            path = destination/entry.filename
            if not path.resolve().is_relative_to(destination.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe path in offline ZIP')
            if not entry.is_dir():
                path.parent.mkdir(parents=True,exist_ok=True)
                with reader.open(entry) as src,path.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)


def render(bundle, output, *, figures=None, require_complete=False, list_only=False):
    from opera.run.prepare_offline_figure_bundle import bundle_files
    offline=bundle/'offline' if (bundle/'offline').is_dir() else bundle
    bundle_files(offline)  # structural/data boundary validation; no source freshness scan
    manifest=json.loads((offline/'manifest.json').read_text())
    entries=manifest['entries']
    if figures:
        unknown=set(figures)-{e['id'] for e in entries}
        if unknown:raise ValueError(f'Unknown figure IDs: {sorted(unknown)}')
        entries=[e for e in entries if e['id'] in figures]
    if list_only:
        print(json.dumps(dict(status=manifest['status'],figures=entries),indent=2));return
    if require_complete and any(e['mode']=='missing' for e in entries):
        raise ValueError('Requested figures include missing analyses; inspect --list')
    if output is None:raise ValueError('Provide --output-dir for rendering')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','MPLBACKEND':'Agg'}
    environment['OPERA_PAPER_FIGURE_MARK']=('TUNING SET — not test results'
        if manifest['evaluation_split']=='tuning' else '')
    environment.pop('OPERA_PAPER_EXPORT_PLOT_STATE',None)
    results=[]
    for entry in entries:
        identity=Path(entry['id'])
        if identity.is_absolute() or '..' in identity.parts:
            raise ValueError('Unsafe offline figure ID')
        target=output/identity
        if entry['mode']=='missing':
            results.append(entry);continue
        print(f"Offline render: {entry['id']} ({entry['mode']})",flush=True)
        if entry['mode']=='native':
            command=[sys.executable,'-m','opera.run.render_paper_results','--config',
                     str(offline/entry['recipe']),'--output-dir',str(target),'--execute']
            target.parent.mkdir(parents=True,exist_ok=True)
            log=target.with_suffix('.log')
            with log.open('w') as handle:
                result=subprocess.run(command,env=environment,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Local figure failed: {entry["id"]}; inspect {log}')
        else:
            from opera.visualization.plot_state import render_plot_state
            from opera.visualization.style import save_fig
            import matplotlib.pyplot as plt
            fig=render_plot_state(offline/entry['source'])
            target.parent.mkdir(parents=True,exist_ok=True)
            old=os.environ.pop('OPERA_PAPER_EXPORT_PLOT_STATE',None)
            try:save_fig(fig,str(target.with_suffix('.png')))
            finally:
                plt.close(fig)
                if old is not None:os.environ['OPERA_PAPER_EXPORT_PLOT_STATE']=old
        results.append(dict(id=entry['id'],mode=entry['mode'],status='rendered'))
    (output/'offline_render.json').write_text(json.dumps(dict(
        evaluation_split=manifest['evaluation_split'],figures=results),indent=2))
    print(f'Local figure iteration: {output}',flush=True)


def main():
    os.environ.setdefault('MPLBACKEND','Agg')
    parser=argparse.ArgumentParser(description=__doc__)
    sources=parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--bundle',type=Path,help='Extracted ZIP root or its offline directory')
    sources.add_argument('--zip',type=Path,help='Read only the offline payload from a review ZIP')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--figures',nargs='+',help='IDs from --list, e.g. main/fig_pooling')
    parser.add_argument('--list',action='store_true',dest='list_only')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    if args.zip:
        with tempfile.TemporaryDirectory(prefix='opera_offline_') as directory:
            root=Path(directory);extract_offline(args.zip,root)
            render(root,args.output_dir,figures=args.figures,require_complete=args.require_complete,list_only=args.list_only)
    else:
        render(args.bundle.resolve(),args.output_dir,figures=args.figures,
               require_complete=args.require_complete,list_only=args.list_only)


if __name__=='__main__':main()
