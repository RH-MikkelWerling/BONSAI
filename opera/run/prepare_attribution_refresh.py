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
    output.mkdir(parents=True, exist_ok=False)
    recipe = output / 'attribution.yaml'
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
    (output / 'refresh_manifest.json').write_text(json.dumps(dict(source=str(root),fit=fit,commands=commands),indent=2))
    print(json.dumps(dict(commands=commands, attribution_dir=str(output / 'attributions')),indent=2))
    if args.execute:
        for command in commands:
            subprocess.run(command,check=True)


if __name__ == '__main__':
    main()
