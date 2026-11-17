"""Assemble available empirical results without substituting rehearsal figures."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('scratch/paper_assembly_20261116'))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    entries = []

    def collect(source, section, scope, patterns):
        source = Path(source)
        for pattern in patterns:
            for path in sorted(source.glob(pattern)):
                if not path.is_file():
                    continue
                relative = Path(section) / path.name
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f'Duplicate assembly destination: {relative}')
                shutil.copy2(path, target)
                entries.append(dict(file=str(relative), source=str(path), scope=scope,
                                    sha256=hashlib.sha256(target.read_bytes()).hexdigest()))

    collect('scratch/descriptive_support_20261116/rendered', '01_pooling',
            'Held-out local encoder; native pooling plate; descriptive cell-bootstrap support bands',
            ['fig_pooling.pdf', 'fig_pooling.png', 'figure_manifest.json'])
    collect('scratch/registered_figures_20261115_v2/rendered', '02_calibration_and_pooling_supplements',
            'Held-out local encoder; registered native calibration and pooling supplement figures',
            ['*.pdf', '*.png', 'figure_manifest.json'])
    collect('scratch/fraction_figures_20261116_ready', '03_label_efficiency',
            'Held-out seed42; common-roster equal-cell AUC at each horizon; local 100% reference only',
            ['*.pdf', '*.png', '*.csv', 'manifest.json'])
    collect('paper/figures/results', '04_clinical_OPERA',
            'Adapted OPERA held-out clinical risk-group plate',
            ['fig_risk_groups.pdf', 'fig_risk_groups.png', 'clinical_R_EA_20261116.json'])
    collect('scratch/R_EA_clinical_review_20261115', '04_clinical_OPERA',
            'Adapted OPERA held-out clinical summaries; observed-risk matching is descriptive',
            ['*.csv', '*.md', 'matched_risk_sources.json'])
    collect('scratch/paper_results_export_20261116_v3/clinical', '04_clinical_OPERA/full_native_exports',
            'Complete exported native clinical artwork and aggregates; model named in archive', ['*.zip'])
    collect('scratch/paper_results_export_20261116_v3/label_efficiency', '05_fraction_control_tables',
            'Held-out fractions and pretraining controls, both weighting schemes',
            ['effect_summaries*.csv', 'panel_estimates.csv', 'preparation_report.json'])
    collect('paper/tables/generated', '06_manuscript_tables',
            'Current generated manuscript numerical tables/macros; Table 1 refresh remains pending',
            ['*.tex', 'paper_numbers.json'])
    collect('scratch/recovered_final_figures_20261116_v4/rendered', '06b_recovered_native_plates',
            'Verified held-out aggregates; headline history panel explicitly unavailable; local fraction curve absent',
            ['*.pdf', '*.png', 'figure_manifest.json'])
    collect('scratch/paper_results_export_20261116_v3/exploratory_representation', '07_exploratory_TUNING',
            'Exploratory tuning core; not held-out mechanistic confirmation', ['*.pdf', '*.csv', 'report.md'])
    collect('scratch/paper_results_export_20261116_v3/exploratory_representation/correction_reliance',
            '07_exploratory_TUNING/correction_reliance',
            'Exploratory tuning inference ablations; missing component replays listed in blocked.csv', ['*.csv'])
    collect('scratch/bundle_review_20261116_v3', '08_review', 'Verified bundle review', ['REVIEW.md'])
    # Keep the current incomplete plate accessible, but outside ready artwork.
    collect('paper/figures/results', '09_partial_figures',
            'PARTIAL: gain-source ablation components are missing', ['fig_gain_sources.pdf', 'fig_gain_sources.png'])
    gaps = [
        'Headline plate restored; patient-history panel and national per-cell absolute estimates for its complete overview remain absent. Direct R_EA versus S_L paired CI is absent.',
        'Registered four-panel external plate restored; local fraction arms remain absent, so the cohort reference is 100% only.',
        'Gain-source plate: disease/outcome-specific correction ablation exports remain missing.',
        'Clinical J_EA and R_AB_EA: final exports absent; server jobs need completion.',
        'Embeddings: final embedding_figures.zip absent.',
        'Final adapted OPERA IG/restrictions: running at export; completion and figures absent.',
        'Table 1: registry mappings/refresh pending.',
        'Pooled XGBoost clinical comparisons: model recovery/export still needed.',
        'Local fraction arms absent; no matched-fraction local-versus-adapted curve.',
        'Additional seeds/subgroups: final coverage unresolved; exploratory common six-arm analysis has one seed.',
    ]
    manifest = dict(files=entries, missing=gaps,
                    policy='Empirical sources only. No synthetic or tuning-rehearsal replacement for held-out figures.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    lines = ['# Local paper and presentation assembly', '',
             'Available empirical figures and tables, with provenance in manifest.json. '
             'Exploratory tuning analyses and partial figures have separate directories.', '',
             '## Remaining gaps', '', *['- ' + gap for gap in gaps], '',
             'The main manuscript retains its current figure slots; standalone label-efficiency plots '
             'have not silently replaced the registered external plate.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    page = ['<!doctype html><meta charset="utf-8"><title>Local results review</title>',
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}'
            'img{max-width:100%;max-height:800px}section{border-top:1px solid #ccc;margin-top:32px}'
            'small{color:#555}li{margin:8px 0}</style>',
            '<h1>Local results review</h1><p>Empirical figures and tables; scopes shown below. '
            'Open PDFs for full resolution.</p><p><a href="README.md">Remaining gaps</a> · '
            '<a href="manifest.json">Provenance manifest</a></p>']
    for section in sorted({Path(e['file']).parts[0] for e in entries}):
        page.append('<section><h2>' + html.escape(section.replace('_', ' ')) + '</h2><ul>')
        for entry in [e for e in entries if Path(e['file']).parts[0] == section]:
            file = entry['file']
            page.append(f'<li><a href="{html.escape(file)}">{html.escape(file)}</a>'
                        f'<br><small>{html.escape(entry["scope"])}</small></li>')
            if file.endswith('.png'):
                page.append(f'<img loading="lazy" src="{html.escape(file)}">')
        page.append('</ul></section>')
    (output / 'index.html').write_text('\n'.join(page))
    archive = output.with_suffix('.zip')
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(output))
    print(json.dumps(dict(output=str(output), files=len(entries), archive=str(archive)), indent=2))


if __name__ == '__main__':
    main()
