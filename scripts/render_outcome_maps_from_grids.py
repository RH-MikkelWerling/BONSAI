"""Render all-outcome risk maps locally using the encoder-map renderer and style."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import textwrap
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd


def render(tables, output, per_page=4):
    from opera.run.paper_story_figures import thesis_style, _save
    from opera.visualization.encoder_embedding_comparison import figure_encoder_maps_v2
    import matplotlib.pyplot as plt
    tables, output = Path(tables), Path(output)
    manifest = json.loads((tables / 'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('model') not in {'R_EA', 'J_EA'}:
        raise ValueError('Need complete adapted joint/OPERA outcome grid export')
    meta = pd.read_csv(tables / 'outcome_grid_meta.csv')
    meta = meta.sort_values(['domain', 'label'])
    if per_page < 1:
        raise ValueError('per_page must be positive')
    output.mkdir(parents=True, exist_ok=True)
    thesis_style()
    written = []
    for page, start in enumerate(range(0, len(meta), per_page), 1):
        parts, entries, labels = [], [], {}
        for _, row in meta.iloc[start:start + per_page].iterrows():
            name = row['file']
            if Path(name).name != name:
                raise ValueError('Unsafe grid filename')
            path = tables / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.grid_sha256:
                raise ValueError(f'Grid hash mismatch: {name}')
            cells = pd.read_csv(path)
            cells['model'] = row.outcome
            parts.append(cells)
            entry = row.to_dict(); entry['model'] = row.outcome
            entries.append(entry)
            labels[row.outcome] = textwrap.fill(row.label, 27)
        fig = figure_encoder_maps_v2(pd.concat(parts, ignore_index=True), pd.DataFrame(entries),
            model_order=[r['model'] for r in entries], model_labels=labels,
            rows=('predicted', 'mortality'), pixels=800, panel_notes=True,
            row_titles={'predicted': 'Predicted one-year risk', 'mortality': 'Observed one-year risk'},
            risk_label='One-year event risk relative to the outcome-specific observed average (both rows)')
        # Unsupported surfaces are visible in the manifest and explicitly marked.
        missing = [r['label'] for r in entries if r['status'] != 'complete']
        if missing:
            fig.text(.01, .005, 'Insufficient supported risk surface: ' + '; '.join(missing), fontsize=6)
        stem = output / f'supp_outcome_maps_{page:02d}'
        _save(fig, stem)
        plt.close(fig)
        written.append({'page': page, 'outcomes': [r['outcome'] for r in entries], 'stem': stem.name})
    (output / 'outcome_maps_render_manifest.json').write_text(json.dumps(
        {'source': str(tables.resolve()), 'pages': written,
         'style': 'figure_encoder_maps_v2 and thesis_style; fixed shared frame; no patient data'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--per-page', type=int, default=4)
    a = p.parse_args()
    render(a.tables, a.output_dir, a.per_page)


if __name__ == '__main__':
    main()
