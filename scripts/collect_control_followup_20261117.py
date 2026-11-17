"""Package small, explicit control provenance and aggregate reports; no inference.

No checkpoints, prediction tables, patient vectors or prepared snapshots are read.
Missing reports stay explicit. Run on CPU; running allocations are untouched.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import yaml

IDENTIFIERS = {'subject_id', 'subject_ids', 'patient_id', 'patient_ids', 'patientid',
               'person_id', 'cpr', 'pnr', 'pid', 'training_subject_ids', 'held_out_subject_ids'}


def check_keys(value):
    if isinstance(value, dict):
        if {str(k).strip().lower() for k in value} & IDENTIFIERS:
            raise ValueError('Identifier fields in supposed metadata/report')
        for item in value.values():
            check_keys(item)
    elif isinstance(value, list):
        for item in value:
            check_keys(item)


def collect(repo, output):
    report = {'scope': 'Existing provenance and aggregate reports only; no runs launched',
              'included': [], 'missing': [], 'excluded': []}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', ZIP_DEFLATED) as archive:
        def add(relative):
            path = repo / relative
            if not path.is_file():
                report['missing'].append(relative)
                return
            if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
                report['excluded'].append({'path': relative, 'reason': 'symlink or >16 MiB'})
                return
            raw = path.read_bytes()
            try:
                text = raw.decode()
                if path.suffix == '.csv':
                    header = next(csv.reader(io.StringIO(text)), [])
                    if {v.strip().lower() for v in header} & IDENTIFIERS:
                        raise ValueError('Patient-level CSV columns')
                elif path.suffix in {'.json', '.yaml'}:
                    check_keys(json.loads(text) if path.suffix == '.json' else yaml.safe_load(text))
                else:
                    raise ValueError('Unregistered report type')
            except (ValueError, UnicodeError, yaml.YAMLError) as error:
                report['excluded'].append({'path': relative, 'reason': str(error)})
                return
            archive.writestr(relative, raw)
            report['included'].append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest()})

        pretrain = 'data/models/daly_care_t100_joined_bins_with_sor_final_v2/pretrain_scaled_time2vec_hidden256'
        for name in ['hydra/config.yaml', 'version_0/hparams.yaml',
                     'version_0/checkpoint_metadata.json', 'version_0/metrics.csv']:
            add(pretrain + '/' + name)
        probe = 'data/paper_256_v2/width_probe/w256_e10'
        for name in ['checkpoint_metadata.json', 'contrastive_multicohort_runs/version_0/hparams.yaml',
                     'contrastive_multicohort_runs/version_0/metrics.csv']:
            add(probe + '/' + name)
        for origin, run_hash in [('RI', 'e8f6e99bdf8a9796336d'), ('FZ', '1c9f0a99b86889a2fe72')]:
            run = f'data/paper_256_v3/neural_runs/{origin}/seed42/{run_hash}'
            for name in ['checkpoint_metadata.json', 'contrastive_multicohort_runs/version_0/hparams.yaml',
                         'contrastive_multicohort_runs/version_0/metrics.csv']:
                add(run + '/' + name)
        memo = 'data/memorization/hidden256_epoch010'
        for name in ['run.json', 'memorization_summary.csv', 'memorization_excess.csv', 'loss_check.csv']:
            add(memo + '/' + name)
        diagnostic = 'data/diagnostics/representation_absorption_20261112/tuning_core'
        for name in ['manifest.json', 'pipeline_complete.json', 'closure_summaries.csv',
                     'absorption_correlations.csv', 'risk_agreement_summary.csv',
                     'scarcity_summaries.csv', 'scarcity_missing_arms.csv', 'seed_results.csv', 'seed_spread.csv']:
            add(diagnostic + '/' + name)
        for name in ['manifest.json', 'inference_ablation_summaries.csv', 'correction_norm_overall.csv',
                     'correction_norm_intervals.csv', 'blocked.csv']:
            add(diagnostic + '/correction_reliance/' + name)
        archive.writestr('control_followup_manifest.json', json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('data/control_followup_20261117.zip'))
    args = parser.parse_args()
    result = collect(args.repo_root, args.output)
    print(json.dumps({'download': str(args.output), **{key: len(result[key]) for key in
                      ['included', 'missing', 'excluded']}}, indent=2))
