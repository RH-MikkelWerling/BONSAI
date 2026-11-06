"""Prepare selected-checkpoint tuning IG and summary commands in a fresh workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from opera.run.paper_experiments import load_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    import fcntl
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ('.' + output.name + '.refresh.lock')
    with lock.open('a') as handle:
        print(f'Waiting for exclusive attribution workspace: {output}', flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        _refresh(args)


def _refresh(args):
    root, output = args.analysis_root.resolve(), args.output_dir.resolve()
    fit = json.loads((root / 'completion.json').read_text())['inputs']['fit']
    for key in ('checkpoint','config'):
        digest = hashlib.sha256(Path(fit[key]).read_bytes()).hexdigest()
        if digest != fit[key + '_sha256']:
            raise ValueError(f'Selected {key} changed since completion')
    env = load_environment(root / 'overnight.env')
    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                             'configs/attribution_analysis.example.yaml').read_text())
    import pandas as pd
    available = set(pd.read_parquet(fit['predictions'], columns=['outcome']).outcome.unique())
    config['run'].update(split='tuning', outcomes='all', horizons=[365],
        check_outcomes=[o for o in ['overall_survival',
            'infection_diagnosis_or_antimicrobial_use_composite',
            'infections_all_definitions','neutropenia_g3plus'] if o in available])
    manifest_path = output / 'refresh_manifest.json'
    recipe = output / 'attribution.yaml'
    if output.exists():
        if not manifest_path.is_file() or not recipe.is_file():
            raise ValueError(f'Existing attribution workspace has no matching receipt: {output}')
        previous = json.loads(manifest_path.read_text())
        if (previous.get('source') != str(root) or previous.get('fit') != fit
                or yaml.safe_load(recipe.read_text()) != config):
            raise ValueError('Existing attribution workspace belongs to different inputs or recipe')
        if previous.get('status') == 'complete':
            if not (output / 'attributions').is_dir():
                raise ValueError('Completed attribution workspace is missing its outputs')
            print(f'Attributions already completed for this rehearsal: {output}', flush=True)
            return
        if previous.get('status') not in (None, 'prepared'):
            raise ValueError('Previous attribution attempt did not complete; inspect its log before retrying')
    else:
        output.mkdir(parents=True, exist_ok=False)
    recipe.write_text(yaml.safe_dump(config, sort_keys=False))
    env.update(OPERA_PAPER_ANALYSIS_ROOT=str(output),
        OPERA_PAPER_ATTRIBUTION_CONFIG=str(recipe),
        OPERA_PAPER_ATTRIBUTION_CHECKPOINT=fit['checkpoint'],
        OPERA_PAPER_ATTRIBUTION_CONFIG_NAME=Path(fit['config']).stem,
        OPERA_PAPER_ATTRIBUTION_CONFIG_DIR=str(Path(fit['config']).parent),
        OPERA_PAPER_PLOT_SPLIT='tuning', OPERA_PAPER_FIGURE_MARK='TUNING SET — not test results')
    path = output / 'overnight.env'
    path.write_text(''.join(f'{k}={json.dumps(str(v))}\n' for k,v in env.items()
        if k.startswith(('OPERA_','BONSAI_'))))
    path.chmod(0o600)
    commands = [[sys.executable,'-m','opera.run.paper_experiments', action,
                 '--env-file',str(path),'--execute'] for action in ('attributions','attribution-summary')]
    receipt = dict(source=str(root), fit=fit, commands=commands, status='prepared')
    def save():
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(manifest_path)
    save()
    print(json.dumps(dict(commands=commands, attribution_dir=str(output / 'attributions')),indent=2))
    if args.execute:
        receipt['status'] = 'running'
        save()
        try:
            for command in commands:
                subprocess.run(command,check=True)
        except BaseException:
            receipt['status'] = 'failed'
            save()
            raise
        receipt['status'] = 'complete'
        save()


if __name__ == '__main__':
    main()
