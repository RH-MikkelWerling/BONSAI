"""Small, separate figure-comparison coordinator with its own local CPU worker.

Uses saved predictions and the existing estimand, support and bootstrap rules.
Never changes fits, selects a primary encoder, trains or evaluates a model.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from opera.run.finish_figure_inputs import immutable, read, sha


COMPARISONS = {
    'P5': 'gain(J_L,S_L)', 'T2': 'gain(J_L,X)', 'O1': 'gain(OPERA,X)', 'O3': 'gain(OPERA,S_L)',
    'E1': 'gain(J_E,J_L)', 'E2': 'gain(R_E,OPERA)', 'A1': 'gain(J_EA,J_E)', 'A2': 'gain(R_EA,R_E)',
    'E3': 'gain(J_EA,J_L)', 'E4': 'gain(R_EA,OPERA)',
    'FINAL_R_EA_VS_S_L': 'gain(R_EA,S_L)',
    'FINAL_R_EA_VS_XP': 'gain(R_EA,XP)',
    # Adapted OPERA against XGBoost for each task, for the subgroup and region figures
    # (19 Nov): the per-model averages gave only a point estimate, without an interval.
    'FINAL_R_EA_VS_X': 'gain(R_EA,X)',
    # Adapted OPERA against ridge logistic regression per task (19 Nov): the Methods name LR as a
    # comparator; only the cohort-encoder O7 and XGBoost T4 contrasts with LR existed.
    'FINAL_R_EA_VS_LR': 'gain(R_EA,LR)',
    'FINAL_J_EA_VS_XP': 'gain(J_EA,XP)',
}
ABLATIONS = {
    'OAOWN_DIS': 'gain(OPERA,J_OA_OWN_NODIS_L)',
    'OAOWN_OUT': 'gain(OPERA,J_OA_OWN_NOOUT_L)',
    'OFF_J': 'gain(R_PF_OFF_L,J_L)', 'OAOWN_OFF': 'gain(OPERA,R_PF_OFF_L)',
    'AB_OFF': 'gain(R_AB_L,R_PF_OFF_L)', 'OAOWN_J': 'gain(OPERA,J_L)',
}
PRETRAINING_CONTROLS = {
    'P0': 'gain(J_L,J_RI)',
    'P7': 'gain(J_L,J_FZ)',
}
LOGISTIC_COMPARISON = {'FINAL_R_EA_VS_LR': 'gain(R_EA,LR)'}


def prepare(repo, output, kind, patient_metadata=None):
    from opera.run.analyse_locked_predictions import models_for, DESIGN_SEARCH_MODELS, design_search_contrasts
    national = repo / 'data/paper_256_v3_national_20261111'
    original = repo / 'data/paper_256_v3/analysis/local_test_20261109_four_allocations/analysis_inventory/analysis.yaml'
    source_paths = [original, national / 'extended_test_comparison/analysis.yaml',
                    national / 'extended_tuning_comparison/analysis.yaml']
    sources = [(path, read(path)) for path in source_paths if path.is_file()]
    if not sources: raise FileNotFoundError('No registered source analysis configs')
    config = copy.deepcopy(next((spec for path, spec in sources if path == original), sources[0][1]))
    designs = set(DESIGN_SEARCH_MODELS) | {'J_L_F10', 'R_PF_OFF_F10_L', 'R_PF5E_F10_L', 'R_PH_GE_F10_L',
                                         'R_PF5E_LIVE_F10_L', 'R_PL8_F10_L'}
    design_contrasts = design_search_contrasts(designs)
    for tag, reference in [('J', 'J_L'), ('OFF_F10', 'R_PF_OFF_F10_L')]:
        for model in designs - {reference}:
            if tag == 'OFF_F10' and '_F10' not in model:
                continue
            design_contrasts[f'DS_{model}__{tag}'] = f'gain({model},{reference})'
    desired = {'core': COMPARISONS, 'logistic': LOGISTIC_COMPARISON,
               'vocabulary': COMPARISONS, 'design_search': design_contrasts,
               'ablations': ABLATIONS, 'pretraining_controls': PRETRAINING_CONTROLS}[kind]
    wanted = models_for(desired)
    wanted.add('J_OA_OWN_L')
    if kind in {'core', 'vocabulary'}: wanted |= {'D_L', 'O_L', 'XP', 'XR'}
    artifacts = {}
    for _, spec in sources:
        for item in spec.get('artifacts', []):
            if int(item.get('seed', -1)) != 42 or item.get('split') not in ['held_out', 'tuning']:
                continue
            if kind == 'design_search' and item.get('split') != 'tuning':
                continue
            item = copy.deepcopy(item)
            if item['model'] not in wanted: continue
            path = Path(item['path'])
            if not path.is_absolute(): path = repo / path
            if not path.is_file(): continue
            item['path'] = str(path)
            key = (item['model'], item['split'], str(path))
            if key in artifacts and artifacts[key] != item:
                raise ValueError(f'Conflicting source artifact declarations: {key}')
            artifacts[key] = item
    artifacts = list(artifacts.values())
    recovered_ablations = []
    if kind == 'ablations' and artifacts:
        # GPU lane can finish these existing fits' evaluations without editing
        # the original inventory; consume only verified completed exports.
        from opera.run.finish_figure_inputs import local_ablations
        from bonsai.functional.checkpointing import verified_training_selection
        try:
            candidates = local_ablations(repo, national)
        except (FileNotFoundError, ValueError, KeyError):
            candidates = []
        reference = next((a for a in artifacts if a['model'] in {'J_OA_OWN_L', 'OPERA'}
                          and a['split'] == 'held_out'), None)
        for item in candidates:
            model, = item['models']
            if any(a['model'] == model and a['split'] == 'held_out' for a in artifacts): continue
            marker_path = item['predictions'] / 'evaluation_complete.json'
            prediction = item['predictions'] / 'direct_cr_predictions.parquet'
            if reference is None or not marker_path.is_file() or not prediction.is_file(): continue
            marker = read(marker_path)
            selected = verified_training_selection(item['run'])
            if (marker['inputs']['config_sha256'] != item['row']['config_sha256']
                    or marker['inputs']['checkpoint_sha256'] != selected['best_checkpoint_sha256']
                    or marker['inputs']['split'] != 'held_out' or marker['prediction_sha256'] != sha(prediction)):
                raise ValueError(f'Local ablation evaluation changed: {model}')
            artifacts.append({**copy.deepcopy(reference), 'model': model, 'path': str(prediction),
                              'format': 'direct_cif', 'cohort_from_metadata': True})
            recovered_ablations.append(model)
    alias_source = 'J_OA_OWN_L' if any(a['model'] == 'J_OA_OWN_L' for a in artifacts) else None
    if alias_source:
        # Let the established alias mechanism add OPERA after native loading;
        # never rewrite a declared model inside a long-format prediction table.
        artifacts = [a for a in artifacts if a['model'] != 'OPERA']
    contracts = {a['data_contract'] for a in artifacts}
    if len(contracts) != 1:
        raise ValueError('Figure comparisons need one shared frozen downstream data contract')
    evaluation_split = 'tuning' if kind == 'design_search' else 'held_out'
    present = {a['model'] for a in artifacts if a['split'] == evaluation_split}
    if alias_source in present: present.add('OPERA')
    required = ({'J_L', 'S_L', 'X', 'OPERA', 'R_EA'} if kind in {'core', 'vocabulary'}
                else {'J_L'} if kind in {'pretraining_controls', 'design_search'}
                else {'R_EA', 'LR'} if kind == 'logistic' else {'OPERA'})
    if not required <= present:
        raise ValueError(f'Missing required held-out models: {sorted(required - present)}')
    contrasts = {name: expression for name, expression in desired.items()
                 if models_for({name: expression}) <= present}
    missing = {name: sorted(models_for({name: expression}) - present)
               for name, expression in desired.items() if name not in contrasts}
    if kind == 'pretraining_controls' and not contrasts:
        raise ValueError('No existing held-out random-initialisation/frozen-encoder '
                         'control predictions in registered inventories; no fits launched. '
                         + json.dumps(missing))
    if kind == 'ablations' and not ({'OAOWN_DIS', 'OAOWN_OUT'} & set(contrasts)):
        raise ValueError('Local NODIS/NOOUT predictions absent; no ablation job launched. '
                         'Adapted arms are not substituted. ' + json.dumps(missing))
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(config['contrast_manifest'])
    manifest['essential_contrasts'] = {'figure_completion': contrasts}
    contrast_path = output / 'contrasts.yaml'
    immutable(contrast_path, manifest)
    config.update(artifacts=artifacts, contrast_manifest=str(contrast_path), extra_contrasts={},
                  opera_model=alias_source, clinical_threshold_artifacts=[], route_weights=None,
                  identity_date_sensitivity=False, coverage_roster=None,
                  operating_points={'rules': [], 'pairs': []},
                  panel_estimate_groups={'figure_models': sorted(present)},
                  patient_gain_contrasts=([c for c in ['P5', 'T2', 'O1', 'O3', 'E1', 'A1', 'A2', 'E3', 'E4', 'FINAL_R_EA_VS_S_L']
                                           if c in contrasts] if kind == 'core' else []))
    if kind in {'design_search', 'vocabulary'}:
        config['design_search'] = kind == 'design_search'
    if kind in {'core', 'vocabulary'}:
        # Retain the source's pre-index groups/bins. No held-out cutpoint choice.
        registered = read(repo / 'opera/configs/locked_prediction_analysis.example.yaml')
        config['patient_groups'] = config.get('patient_groups') or registered['patient_groups']
        columns = ['history_event_count', 'age_at_index', 'sex', 'IPI', 'predominant_region', 'index_year']
        group_contrasts = [c for c in ['P5', 'T2', 'O1', 'O3', 'E3', 'E4', 'FINAL_R_EA_VS_S_L',
                                      'FINAL_R_EA_VS_XP', 'FINAL_J_EA_VS_XP', 'FINAL_R_EA_VS_X']
                           if c in contrasts]
        config['stratified_reanalysis'] = {column: {'contrasts': group_contrasts,
            'models': [m for m in ['J_L', 'OPERA', 'R_EA', 'J_EA', 'XP', 'X'] if m in present]} for column in columns}
        if kind == 'vocabulary':
            if patient_metadata is None:
                raise ValueError('Supply --patient-metadata from the national coverage producer')
            import pyarrow.parquet as pq
            if 'national_vocabulary_coverage' not in pq.ParquetFile(patient_metadata).schema.names:
                raise ValueError('National coverage column missing from supplied metadata')
            config['patient_metadata'] = str(Path(patient_metadata).resolve())
            config['patient_groups'] = {'national_vocabulary_coverage': registered['patient_groups']['national_vocabulary_coverage']}
            config['stratified_reanalysis'] = {'national_vocabulary_coverage': {
                'contrasts': group_contrasts, 'models': [m for m in ['J_L', 'OPERA', 'R_EA', 'J_EA', 'XP', 'X'] if m in present]}}
    else:
        config.update(patient_groups={}, stratified_reanalysis={})
    immutable(output / 'analysis.yaml', config)
    report = {'scope': 'Additional figure inputs from the same frozen predictions; no fitting/selection',
              'kind': kind, 'models': sorted(present), 'contrasts': contrasts, 'missing_contrasts': missing,
              'source_configs': {str(p): sha(p) for p, _ in sources},
              'recovered_completed_local_ablations': recovered_ablations,
              'bootstrap_replicates': config.get('bootstrap_replicates', 1000),
              'operating_point_jobs': 0,
              'tuning_models': sorted({a['model'] for a in artifacts if a['split'] == 'tuning'}),
              'strategy_note': 'The source strategy rule is retained; missing tuning candidates remain explicit.'}
    (output / 'preparation_report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['core', 'ablations', 'pretraining_controls', 'design_search', 'vocabulary', 'logistic'], default='core')
    parser.add_argument('--patient-metadata', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('data/figure_closure_20261116'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    repo, root = Path.cwd().resolve(), args.output_root.resolve()
    output = root / f'comparisons_{args.kind}'
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'results/analysis_manifest.json').is_file():
            print(f'COMPARISONS {args.kind}: already complete: {output / "results"}'); return
        try:
            report = prepare(repo, output, args.kind, args.patient_metadata)
        except (ValueError, FileNotFoundError, KeyError) as error:
            (output / 'blocked.json').write_text(json.dumps({'status': 'blocked', 'reason': str(error)}, indent=2))
            raise SystemExit(f'COMPARISONS {args.kind}: BLOCKED: {error}')
        print(json.dumps(report, indent=2), flush=True)
        if not args.execute: return
        subprocess.run([sys.executable, 'scripts/check_results_resources.py', '--minimum-free-gib', '96',
                        '--require-no-coordinator'], check=True)
        checkpoints = output / 'checkpoints'
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OPERA_ANALYSIS_CHECKPOINT_DIR': str(checkpoints),
               'OPERA_ANALYSIS_WORKERS': '1', 'OPERA_SUMMARY_WORKERS': '1',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
        logs = output / 'logs'; logs.mkdir(exist_ok=True)
        with (logs / 'worker.log').open('a') as worker_log, (logs / 'coordinator.log').open('a') as coordinator_log:
            worker = subprocess.Popen([sys.executable, '-u', '-m', 'opera.run.results_worker',
                '--checkpoint-dir', str(checkpoints), '--wait-hours', '48'], env=env,
                stdout=worker_log, stderr=subprocess.STDOUT)
            coordinator = None
            try:
                coordinator = subprocess.Popen([sys.executable, '-u', '-m', 'opera.run.analyse_locked_predictions',
                    '--config', str(output / 'analysis.yaml'), '--output-dir', str(output / 'results')],
                    env=env, stdout=coordinator_log, stderr=subprocess.STDOUT)
                while coordinator.poll() is None:
                    if worker.poll() is not None:
                        raise RuntimeError(f'Local worker stopped; inspect {logs / "worker.log"}')
                    time.sleep(5)
                if coordinator.returncode:
                    raise RuntimeError(f'Coordinator failed; completed checkpoints retained: {logs / "coordinator.log"}')
            finally:
                if coordinator is not None and coordinator.poll() is None:
                    coordinator.terminate()
                worker.terminate()
                try: worker.wait(timeout=10)
                except subprocess.TimeoutExpired: worker.kill(); worker.wait()
        print(f'COMPARISONS {args.kind}: COMPLETE: {output / "results"}', flush=True)


if __name__ == '__main__':
    main()
