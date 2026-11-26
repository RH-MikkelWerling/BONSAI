"""Read-only receipt/artifact inventory; does not load predictions or inspect other allocations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def collect(root):
    root = Path(root)
    patterns = [
        'data/flipi2_os_repair_20261120/download_flipi2_os_repair.zip',
        'data/flipi2_os_repair_20261120/clinical_results/report_manifest.json',
        'data/vocabulary_control_20261120/preparation.json',
        'data/vocabulary_control_20261120/manifest.json',
        'data/vocabulary_control_20261120/download_vocab_control.zip',
        'data/vocabulary_control_20261120/matched/comparisons_vocab_control/results/analysis_manifest.json',
        'data/decision_curves_20261120/manifest.json',
        'data/decision_curves_20261120/download_decision_curves.zip',
        'data/epoch20_matched_20261120/download_*.zip',
        'data/adapted20_matched_20261120/download_*.zip',
        'data/*20_matched_20261120/comparisons_*/results/analysis_manifest.json',
        'data/*20_matched_20261120/comparisons_*/preparation_report.json',
        'data/adapted20_control_20261120/preparation.json',
        'data/adapted20_control_20261120/review_summary.json',
        'data/adapted20_control_20261120/download_adapted20.zip',
        'data/outcome_rates_by_year_20261120*/manifest.json',
        'data/outcome_rates_by_year_20261120*/download_*.zip',
        'data/story_closure_20261116/story_status.json',
        'data/story_closure_20261116/download_story.zip',
        'data/story_closure_20261116/ready/tables/embedding_grids*',
        'data/figure_closure_2026111*/download_*.zip',
        'data/figure_closure_2026111*/*status.json',
        'data/figure_closure_2026111*/comparisons_*/preparation_report.json',
        'data/figure_closure_2026111*/comparisons_*/blocked.json',
        'data/figure_closure_2026111*/comparisons_*/results/analysis_manifest.json',
        'data/figure_closure_2026111*/comparisons_*/results/effect_summaries*.csv',
        'data/figure_closure_2026111*/comparisons_*/checkpoints/active.json',
        'data/paper_256_v3_national_20261111/final_interpretation_20261116/*manifest.json',
        'data/paper_256_v3_national_20261111/final_interpretation_20261116/*status.json',
        'data/clinical_registry_refresh/20261111/*/table1_review.zip',
        'data/control_followup_20261117.zip',
        'data/outcome_embedding_grids_joint_20261119*/manifest.json',
        'data/outcome_embedding_grids_joint_20261119*/download_outcome_grids_joint.zip',
        'data/epoch20_joint_control_20261119/preparation.json',
        'data/epoch20_joint_control_20261119/review_summary.json',
        'data/epoch20_joint_control_20261119/epoch20_joint_review.zip',
        'data/epoch20_joint_control_20261119/**/checkpoint_metadata.json',
        'data/national_coverage_20261119/*manifest.json',
        'data/*disease*fine*/*manifest.json',
        'data/*disease*fine*/download_*.zip',
        'data/*pretrained*grid*/*manifest.json',
        'data/*pretrained*grid*/download_*.zip',
        'logs/remaining_paper_cpu_20261119.log',
        'logs/paired_logistic_20261119.log',
        'logs/epoch20_joint_20261119.log',
        'data/outcome_embedding_grids_20261119*/manifest.json',
        'data/outcome_embedding_grids_20261119*/download_outcome_grids.zip',
        'data/task_structure_comparison_20261119/task_structure_comparison.zip',
        'data/paper_256_v3_national_20261111/final_recent_record_gaps_20261119/recent_record_review.zip',
        'data/diagnostics/representation_absorption_20261112/*/manifest.json',
        'data/diagnostics/representation_absorption_20261112/*/pipeline_complete.json',
        'data/fixed_budget_risk_20261118/*/manifest.json',
        'data/fixed_budget_risk_20261118/*/fixed_budget_review.zip',
        'logs/reproducibility_facts_2026111*.json',
        'logs/tabular_inference_recovery_2026111*.json',
        'logs/pretraining_downstream_search_2026111*.json',
        'logs/epoch20_*_20261119*.json',
    ]
    result = {'collected_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Files in this checkout only; presence is not proof of complete coverage or a live process',
              'artifacts': [], 'unmatched_patterns': []}
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches:
            result['unmatched_patterns'].append(pattern)
        for path in matches:
            if not path.is_file():
                continue
            stat = path.stat()
            row = {'path': str(path.relative_to(root)), 'bytes': stat.st_size,
                   'modified_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
            # Only aggregate status fields: never load patient tables, vectors or checkpoint pickles.
            if path.suffix == '.json' and stat.st_size < 2_000_000:
                try:
                    data = json.loads(path.read_text())
                    if isinstance(data, dict):
                        row['receipt'] = {k: data[k] for k in
                            ['status', 'complete', 'embeddings', 'clinical', 'no_index',
                             'flipi2_endpoint', 'generation', 'exploratory'] if k in data}
                        if isinstance(data.get('steps'), dict):
                            row['step_statuses'] = {
                                k: {field: value[field] for field in ['status', 'returncode'] if field in value}
                                for k, value in data['steps'].items() if isinstance(value, dict)}
                except (ValueError, OSError) as error:
                    row['receipt_error'] = str(error)
            result['artifacts'].append(row)
    recipe = root / 'data/paper_256_v3_national_20261111/final_interpretation_20261116/attribution.yaml'
    if recipe.is_file():
        try:
            import yaml
            data = yaml.safe_load(recipe.read_text())
            ids = [r['id'] for r in data.get('restrictions', [])]
            required = {'gap_7d', 'gap_14d', 'gap_21d', 'gap_28d', 'last_year',
                        'no_labs', 'no_medication', 'no_diagnoses', 'timeline', 'no_region_specialty'}
            result['saved_interpretation_recipe'] = {'path': str(recipe.relative_to(root)),
                'split': data.get('run', {}).get('split'), 'restriction_ids': ids,
                'missing_requested_restrictions': sorted(required - set(ids))}
        except (ImportError, ValueError, OSError, KeyError, TypeError) as error:
            result['saved_interpretation_recipe'] = {'error': str(error)}
    return result


def ig_cache_progress(directory, *, planned_patients=None):
    """Aggregate cached timing only; no patient identifiers leave this function."""
    import statistics
    directory = Path(directory)
    markers = sorted((directory/'patient_cache_INTERNAL').glob('patient-*/complete.json'))
    times, unreadable = [], 0
    for marker in markers:
        try:
            value = json.loads(marker.read_text()).get('wall_seconds')
            if value is not None and float(value) >= 0:
                times.append(float(value))
        except (OSError, ValueError, TypeError):
            unreadable += 1
    run_path = directory/'attribution_run.json'
    if run_path.is_file():
        try:
            run = json.loads(run_path.read_text())
            planned_patients = run.get('planned_patients', planned_patients)
        except (OSError, ValueError):
            pass
    result = dict(cached_patient_receipts=len(markers), unreadable_receipts=unreadable,
                  patients_timed=len(times), planned_patients=planned_patients,
                  checkpoint_identity_verified=False,
                  note='Timing estimate only; normal resume verifies checkpoint, roster and cache hashes. Initial patients include additional checks; future records may differ in length.')
    if times:
        recent = times[-50:]
        mean = statistics.mean(times)
        result.update(mean_seconds_per_patient=mean,
                      median_seconds_per_patient=statistics.median(times),
                      latest_50_median_seconds_per_patient=statistics.median(recent))
        if planned_patients is not None:
            remaining = max(0, int(planned_patients)-len(markers))
            result.update(remaining_patients=remaining,
                remaining_hours_at_observed_mean=remaining*mean/3600,
                remaining_hours_at_latest_50_median=remaining*statistics.median(recent)/3600)
    return result


def collect_recovery(root, *, projection_version="held_out_descriptive_v1", ig_planned_patients=None):
    """Read aggregate receipts and ZIP directories only; never load patient data."""
    from zipfile import ZipFile, BadZipFile
    root = Path(root)

    def receipt(path):
        if not path.is_file():
            return {"status": "absent"}
        try:
            if path.stat().st_size > 2_000_000:
                return {"status": "present_not_parsed", "bytes": path.stat().st_size}
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                return {"status": "unexpected_receipt_type"}
            return value
        except (OSError, ValueError) as error:
            return {"status": "unreadable", "reason": str(error)}

    def archive(path):
        item = {"path": str(path), "exists": path.is_file(), "directory_readable": False}
        if not path.is_file():
            return item
        try:
            item["bytes"] = path.stat().st_size
            with ZipFile(path) as z:
                names = z.namelist()
                item.update(directory_readable=True, members=len(names),
                            contains_test_fitted_maps=any(f"story_{projection_version}/" in n for n in names))
        except (OSError, BadZipFile, EOFError) as error:
            item["error"] = str(error)
        return item

    pipeline = receipt(root/'pipeline_status.json')
    stages = {name: {key: value[key] for key in ('status', 'workload_profile', 'deferred_jobs') if key in value}
              for name, value in pipeline.items() if isinstance(value, dict)}
    analyses = {}
    for directory in sorted((root/'analyses').glob('*')):
        if not directory.is_dir():
            continue
        final = directory/'results'
        progress = directory/'results.progress'
        final_manifest = final/'analysis_manifest.json'
        files = {}
        for name in ('effect_summaries.csv', 'effect_summaries_disease_balanced.csv',
                     'calibration_summary.csv', 'operating_points.csv', 'analysis_status.csv'):
            path = final/name if (final/name).is_file() else progress/name
            files[name] = {"present": path.is_file(), "final": path.parent == final,
                           "bytes": path.stat().st_size if path.is_file() else 0}
        analyses[directory.name] = {"final_manifest_present": final_manifest.is_file(), "tables": files}

    story = root/'exports'/f'story_{projection_version}'
    story_receipt = receipt(story/'story_status.json').get('embeddings', {})
    maps = {"status": story_receipt.get('status', 'absent'),
            "projection_version": story_receipt.get('projection_version'), "families": {}}
    for family in ('joint', 'opera'):
        info = story_receipt.get(family, {})
        required = [story/'ready/tables'/f'embedding_grids_{family}.csv.gz',
                    story/'ready/tables'/f'embedding_grids_{family}_meta.csv']
        maps['families'][family] = {"patients": info.get('patients'),
            "projection_fit_split": info.get('projection_fit_split'),
            "grid_files_present": all(p.is_file() for p in required)}
    maps['ready'] = (maps['status'] == 'complete' and maps['projection_version'] == projection_version
                     and all(v['projection_fit_split'] == 'held_out' and v['grid_files_present']
                             for v in maps['families'].values()))

    interpretation = receipt(root/'final_interpretation/interpretation_manifest.json')
    interpretation = {"status": interpretation.get('status', 'unknown'),
        "steps": {name: value.get('status') for name, value in interpretation.get('steps', {}).items()
                  if isinstance(value, dict)}}
    arrows = receipt(root/'exports'/f'model_exports_{projection_version}'/'status.json')
    interpretation['cache_progress'] = ig_cache_progress(root/'final_interpretation/attributions',
                                                         planned_patients=ig_planned_patients)
    interpretation['test_fitted_arrows'] = {"status": arrows.get('status', 'unknown'),
                                            "projection_version": arrows.get('projection_version')}
    paths = [root/'download_cutoff_recovery.zip']
    patterns = ['clinical/*/download_clinical_*.zip', f'story_{projection_version}/download_story.zip',
        f'outcome_grids_*_{projection_version}/download_outcome_grids*.zip',
        f'model_exports_{projection_version}/download_model_exports.zip',
        'deciles/fixed_budget_review.zip', 'decision_curves/download_decision_curves.zip',
        'task_structure/download_task_structure.zip']
    for pattern in patterns:
        paths.extend(sorted((root/'exports').glob(pattern)))
    downloads = [archive(path) for path in paths]
    return {"collected_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "scope": "Read-only recovery receipts. Not live process status, full file-hash verification or ZIP CRC validation.",
        "pipeline": stages, "analyses": analyses, "test_fitted_maps": maps,
        "interpretation": interpretation, "downloads": downloads,
        "missing_export_families": [pattern for pattern in patterns if not list((root/'exports').glob(pattern))],
        "transfer_note": "A newly delivered national epoch-20 encoder is separate; this collector does not certify its transfer or downstream fits."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery-root', type=Path, help='Inspect the cutoff-recovery release instead of legacy paths')
    parser.add_argument('--ig-planned-patients', type=int, help='Planned IG count if no attribution receipt exists yet (timing estimates only)')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        if args.recovery_root:
            path = args.recovery_root if args.recovery_root.is_absolute() else args.root/args.recovery_root
            report = collect_recovery(path, ig_planned_patients=args.ig_planned_patients)
        else:
            report = collect(args.root)
        json.dump(report, handle, indent=2)
    print(f'Status inventory: {args.output}')
    if args.recovery_root:
        print('PIPELINE:', json.dumps(report['pipeline']))
        print('FINAL ANALYSES:', ', '.join(k for k, v in report['analyses'].items() if v['final_manifest_present']) or 'none')
        print('TEST-FITTED MAPS READY:', report['test_fitted_maps']['ready'])
        print('INTERPRETATION:', json.dumps(report['interpretation']))
        for item in report['downloads']:
            if item['directory_readable']:
                print('AVAILABLE ZIP:', item['path'], f"({item['bytes']:,} bytes)")
        print('MISSING EXPORT FAMILIES:', json.dumps(report['missing_export_families']))


if __name__ == '__main__':
    main()
