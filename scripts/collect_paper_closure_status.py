"""Read-only receipt/artifact inventory; does not load predictions or inspect other allocations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def collect(root):
    root = Path(root)
    patterns = [
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
        'data/diagnostics/representation_absorption_20261112/*/manifest.json',
        'data/diagnostics/representation_absorption_20261112/*/pipeline_complete.json',
        'data/fixed_budget_risk_20261118/*/manifest.json',
        'data/fixed_budget_risk_20261118/*/fixed_budget_review.zip',
        'logs/reproducibility_facts_2026111*.json',
        'logs/tabular_inference_recovery_2026111*.json',
        'logs/pretraining_downstream_search_2026111*.json',
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
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(collect(args.root), handle, indent=2)
    print(f'Status inventory: {args.output}')


if __name__ == '__main__':
    main()
