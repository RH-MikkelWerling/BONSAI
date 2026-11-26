"""Late GPU export of corrected OPERA arrows; no fitting and no patient exports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from opera.run.finish_figure_inputs import final_fit, map_config, immutable


def run(root):
    root = root.resolve()
    fit = final_fit(root)
    from opera.run.build_paper_embedding_tables import PROJECTION_VERSION
    directory = root/'exports'/f'model_exports_{PROJECTION_VERSION}'
    directory.mkdir(parents=True, exist_ok=True)
    # Use the identical cross-encoder roster as the descriptive OPERA plate.
    stages = None
    if (root/'queue_plan.yaml').is_file():
        from opera.run.cutoff_recovery_worker import read
        from opera.run.cutoff_recovery_results import rebound_artifacts
        from opera.run.cutoff_recovery_exports import stages_config
        plan = read(root/'queue_plan.yaml')
        wanted = {'J_L', 'J_OA_OWN_L', 'J_E', 'R_E', 'J_EA', 'R_EA'}
        declarations = [d for source in read(plan['analyses']) for d in source['artifacts']
                        if d['artifact']['model'] in wanted and int(d['artifact'].get('seed', 42)) == 42]
        artifacts, _, waiting, _ = rebound_artifacts(root, plan, declarations)
        if waiting:
            raise ValueError('Main embedding exports must finish before correction-arrow projection')
        stages_path = stages_config(root, artifacts)
        stages = read(stages_path)['opera']
    private, metadata, binding, _ = map_config(root, fit, directory, stages)
    immutable(directory/'intent.yaml', binding)
    arrows = directory/'correction_arrows'
    if not (arrows/'correction_arrows.json').is_file():
        subprocess.run([sys.executable, '-u', '-m', 'opera.run.export_opera_correction_arrows',
            '--checkpoint', str(fit['checkpoint']), '--config-dir', str(fit['config'].parent),
            '--config-name', fit['config'].stem, '--embedding-config', str(private/'embedding_config.yaml'),
            '--output-dir', str(arrows), '--num-workers', '0'], check=True)
    recipe = directory/'arrows.yaml'
    immutable(recipe, dict(figures=[dict(id='supp_correction_arrows', function='figure_correction_arrows',
        tables=dict(arrows=str(arrows/'correction_arrows.csv'), density=str(arrows/'correction_arrow_density.csv')),
        json_inputs=dict(meta=str(arrows/'correction_arrows.json')))]))
    figures = directory/'figures'
    if not (figures/'figure_manifest.json').is_file():
        subprocess.run([sys.executable, '-u', '-m', 'opera.run.render_paper_results',
            '--config', str(recipe), '--output-dir', str(figures), '--execute'], check=True)
    from opera.run.package_figure_review import PATIENT_COLUMNS
    import csv
    archive = directory/'download_model_exports.zip'
    temporary = archive.with_suffix('.writing')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as target:
        for name in ('correction_arrows.csv', 'correction_arrow_density.csv', 'correction_arrows.json'):
            path = arrows/name
            if path.suffix == '.csv':
                with path.open(newline='') as handle:
                    if {c.lower() for c in next(csv.reader(handle), [])} & PATIENT_COLUMNS:
                        raise ValueError('Patient columns in arrow aggregate')
            target.write(path, name)
        for path in figures.glob('supp_correction_arrows.*'):
            if path.suffix in {'.pdf', '.png'}: target.write(path, path.name)
    temporary.replace(archive)
    (directory/'status.json').write_text(json.dumps(dict(status='complete', archive=str(archive),
        checkpoint_sha256=fit['checkpoint_sha256'], projection_version=PROJECTION_VERSION), indent=2))
    print('DOWNLOAD:', archive, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/cutoff_recovery_20261121'))
    run(p.parse_args().root)


if __name__ == '__main__': main()
