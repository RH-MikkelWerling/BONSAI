"""Render downloaded common-roster panels with the existing label-efficiency plot."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir / 'panel_estimates.csv'
    panels = pd.read_csv(source)
    panels = panels.loc[(panels['split'] == 'held_out') & (panels.seed == 42)
                        & (panels.metric == 'auc')
                        & (panels['group'] == 'label_efficiency_common_roster')]
    expected = {'J_EA', 'J_EA_F10', 'J_EA_F25', 'J_EA_F50', 'J_L'}
    if panels.empty or set(panels.model) != expected:
        raise ValueError('Expected the four adapted fractions and full-data local joint model')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from opera.visualization.comparison_plots import plot_label_efficiency
    import matplotlib.pyplot as plt
    files = []
    for horizon, rows in panels.groupby('horizon_days'):
        if len(rows) != 5 or rows.model.duplicated().any() or rows.n_cells.nunique() != 1:
            raise ValueError('Panel must contain one common-roster estimate per model')
        curves = {'J_EA': {}, 'J_L': {}}
        for row in rows.itertuples():
            origin = 'J_L' if row.model == 'J_L' else 'J_EA'
            fraction = float(row.model.split('_F')[-1]) / 100 if '_F' in row.model else 1.0
            curves[origin][fraction] = dict(mean=row.estimate, lower=row.lower, upper=row.upper)
        fig = plot_label_efficiency(curves, metric='auc')
        ax = fig.axes[0]
        # Only full-data local joint is available: show its actual point and
        # a labelled reference line, never synthesize missing fraction arms.
        baseline = curves['J_L'][1.0]['mean']
        ax.lines[1].set_color('#667085')
        ax.lines[1].set_marker('D')
        ax.axhline(baseline, color='#667085', linestyle='--', linewidth=.8)
        ax.set_title(f'Label efficiency: {int(horizon)} days')
        ax.set_ylabel('Equal-cell mean AUC (95% CI)')
        ax.set_xticks([10, 25, 50, 100])
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ['Adapted national joint', 'Local joint: 100% only'], fontsize=8)
        fig.tight_layout()
        for extension in ('pdf', 'png'):
            target = args.output_dir / f'label_efficiency_{int(horizon)}d.{extension}'
            fig.savefig(target, dpi=200, bbox_inches='tight')
            files.append(target.name)
        plt.close(fig)
    panels.to_csv(args.output_dir / 'plotted_panel_estimates.csv', index=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(dict(
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        function='opera.visualization.comparison_plots.plot_label_efficiency',
        split='held_out', seed=42, files=files,
        estimand='Equal-cell common-roster mean at each horizon; rosters can differ across horizons',
        limitations='Local fraction arms absent. Bands are supplied panel intervals, not median/IQR. '
                     'This standalone figure does not replace the registered external plate.'), indent=2))


if __name__ == '__main__':
    main()
