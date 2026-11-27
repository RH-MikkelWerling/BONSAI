"""Replay existing paper recipes against the corrected aggregate download.

No estimation, model fitting, patient data or outcome-gallery rendering.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd

from opera.run import paper_story_figures as paper
from opera.visualization import opera_design_figures as design
from scripts import render_clinical_splits as splits
from scripts import render_recovered_supplement as recovered
from opera.analysis.fixed_budget_risk import plot_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, default=Path('scratch/corrected_results_20261126'))
    p.add_argument('--output', type=Path, default=Path('scratch/corrected_results_20261126/additional_figures'))
    a = p.parse_args()
    exports = a.review / 'bundle/exports'
    a.output.mkdir(parents=True, exist_ok=True)
    paper.thesis_style()
    from matplotlib.figure import Figure
    import inspect
    from matplotlib.legend import Legend
    # Older local Matplotlib lacks the public legend-alignment argument.
    # Apply its equivalent box alignment without changing the plotting recipe.
    legend = Figure.legend
    def compatible_legend(fig, *args, **kwargs):
        alignment = kwargs.pop('alignment', None)
        result = legend(fig, *args, **kwargs)
        if alignment is not None:
            result._legend_box.align = alignment
        return result
    structure = exports / 'task_structure/download_task_structure_unpacked'
    if 'alignment' in inspect.signature(Legend).parameters:
        recovered.task_structure(structure, a.output)
    else:
        with patch.object(Figure, 'legend', compatible_legend):
            recovered.task_structure(structure, a.output)
    recovered.disease_map(structure, a.output)
    effects = pd.read_csv(a.review / 'bundle/core_damage/contrasts_all_horizons.csv')
    effects = effects.loc[effects['split'].eq('held_out') & effects.seed.eq(42)
                          & effects.metric.eq('auc') & effects.contrast.eq('ARCHITECTURE_ADAPTED')]
    if effects.empty:
        raise ValueError('Corrected architecture cell effects are missing')
    effects = effects.assign(outcome=effects.outcome.map(design._outcome_label), added=effects.gain)
    outcome_effects, disease_effects = recovered.correction_value(structure, a.output,
                                                                cell_effects=effects)
    from scipy.stats import spearmanr
    (a.output / 'correction_value_statistics.json').write_text(json.dumps({
        'outcome_size_gain_rho':spearmanr(outcome_effects.correction_size, outcome_effects.added,
                                         nan_policy='omit')[0],
        'disease_size_gain_rho':spearmanr(disease_effects.correction_size, disease_effects.added,
                                         nan_policy='omit')[0]}, indent=2))
    save = paper._save
    labels = {'R_EA':'Adapted OPERA', 'J_EA':'Adapted joint',
              'R_AB_EA':'Adapted router', 'J_OA_OWN_L':'Cohort OPERA'}
    for model, label in labels.items():
        source = exports / 'clinical' / model / f'download_clinical_{model}_unpacked'
        out = a.output / 'clinical' / model
        def labelled_save(fig, stem):
            if model != 'R_EA':
                def replace(text):
                    return text.replace('Adapted OPERA', '__MODEL__').replace('OPERA', '__MODEL__').replace('__MODEL__', label)
                for ax in fig.axes:
                    ys = [t.get_text() for t in ax.get_yticklabels()]
                    if any('OPERA' in text for text in ys):
                        ax.set_yticks(ax.get_yticks(), [replace(text) for text in ys])
                for text in fig.findobj(Text):
                    # Labels already set on ticks are left untouched, avoiding
                    # re-expanding the word OPERA inside "Cohort OPERA".
                    if text not in [t for ax in fig.axes for t in ax.get_yticklabels()]:
                        text.set_text(replace(text.get_text()))
            save(fig, stem)
        with patch.object(paper, '_save', labelled_save), patch.object(splits, '_save', labelled_save):
            paper.fig6_clinical(a.review / 'ready_inputs' / f'clinical_{model}', out, figures=source)
            paper.supp_clinical_groups(source, a.review / 'ready_inputs' / f'clinical_{model}', out)
            summary = pd.read_csv(source / 'km_split_summary.csv')
            curves = pd.read_csv(source / 'km_split_curves.csv')
            comparisons = splits._comparisons(summary)
            splits.overview(summary, comparisons, out)
            splits.km_page(summary, curves, comparisons[:6], out, 'supp_index_splits_km_1')
            splits.km_page(summary, curves, comparisons[6:], out, 'supp_index_splits_km_2')
    arrows = exports / 'model_exports_held_out_descriptive_v1/download_model_exports_unpacked'
    save(design.figure_correction_arrows(pd.read_csv(arrows / 'correction_arrows.csv'),
        pd.read_csv(arrows / 'correction_arrow_density.csv'),
        json.loads((arrows / 'correction_arrows.json').read_text())), a.output / 'supp_correction_arrows')
    deciles = exports / 'deciles/fixed_budget_review_unpacked'
    # Retain the decile recipe's layout, marks, colours and statistics; use paper typography.
    with patch('opera.visualization.style.setup_style', paper.thesis_style):
        out = a.output / 'deciles'
        out.mkdir(exist_ok=True)
        plot_summary(pd.read_csv(deciles / 'summary.csv'), out, .1)
    sources = list(exports.glob('clinical/*/*_unpacked/*.csv'))
    sources += list(structure.rglob('*.csv')) + list(structure.rglob('*.json'))
    sources += [a.review / 'bundle/core_damage/contrasts_all_horizons.csv']
    sources += list(arrows.glob('*.csv')) + [arrows / 'correction_arrows.json', deciles / 'summary.csv']
    digest = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
    (a.output / 'refresh_manifest.json').write_text(json.dumps({
        'scope':'Existing designs; corrected saved aggregates only; no galleries',
        'inputs':{str(x):digest(x) for x in sources},
        'figures':{str(x):digest(x) for x in a.output.rglob('*.pdf')}}, indent=2))


if __name__ == '__main__':
    main()
