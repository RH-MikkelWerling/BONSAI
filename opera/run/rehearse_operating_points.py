"""Compute apparent tuning threshold metrics only, without repeating paired analysis."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from opera.run.analyse_locked_predictions import (
    read_table, read_artifacts, collapse_seed_repeats, with_opera_alias,
)
from opera.evaluation.secondary_analysis import normalize_predictions
from opera.evaluation.operating_points import DEFAULT_RULES, operating_points, suppress_small_counts
from opera.run.build_paper_plot_tables import operating_point_summary
from opera.run.rerun_full_tuning_rehearsal import tuning_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['J_L', 'X', 'XP', 'R_L'])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a fresh operating-point recovery directory')
    config = tuning_inventory(yaml.safe_load(args.config.read_text()))
    # Reuse the rehearsal's declaration and table-row split validation.
    artifacts = config['artifacts']
    if any(a.get('split') != 'tuning' for a in artifacts):
        raise ValueError('This command accepts explicitly tuning-only inventories')
    selected = set(args.models) | {config.get('opera_model')}
    artifacts = [a for a in artifacts if a.get('model') in selected]
    if not artifacts:
        raise ValueError('No requested models in inventory')
    metadata = read_table(config['patient_metadata']) if config.get('patient_metadata') else None
    if metadata is not None:
        metadata = metadata.rename(columns=config.get('patient_metadata_column_map', {}))
        metadata['subject_id'] = metadata.subject_id.astype(str)
    print(f'Reading tuning predictions from {len(artifacts)} artifacts', flush=True)
    inputs = [data for data, _, _ in read_artifacts(artifacts, metadata, config['estimand'])]
    predictions, _ = normalize_predictions(pd.concat(inputs, ignore_index=True), config['estimand'])
    if set(predictions.split) != {'tuning'}:
        raise ValueError('Prediction rows must all be tuning')
    predictions = collapse_seed_repeats(predictions, config)
    predictions, _ = with_opera_alias(predictions, config.get('opera_model'))
    if config.get('analysis_horizons_days'):
        predictions = predictions.loc[predictions.horizon_days.isin(config['analysis_horizons_days'])]
    # Apparent performance only; no inferential intervals for thresholds selected
    # on the same patients. Full held-out evaluation retains its paired bootstrap.
    settings = config.get('operating_points', {})
    parts = []
    for rule in settings.get('rules', DEFAULT_RULES):
        print(f'Operating points: {rule}', flush=True)
        points, _ = operating_points(predictions, {}, rule=rule, evaluation_split='tuning',
            minimum_events=int(settings.get('minimum_tuning_events', 5)))
        parts.append(points)
    points = pd.concat(parts, ignore_index=True)
    if points.empty:
        raise ValueError('No evaluable operating points')
    args.output_dir.mkdir(parents=True)
    suppress_small_counts(points, int(settings.get('minimum_cell_count', 5))).to_csv(
        args.output_dir / 'operating_points.csv', index=False)
    operating_point_summary(points).to_csv(args.output_dir / 'operating_point_summary.csv', index=False)
    (args.output_dir / 'rehearsal.json').write_text(json.dumps(dict(
        evaluation_split='tuning', evaluation_role='apparent_tuning_rehearsal',
        intervals='Not estimated: thresholds selected on the same tuning patients',
        rows=len(points), config=str(args.config.resolve())), indent=2))


if __name__ == '__main__':
    main()
