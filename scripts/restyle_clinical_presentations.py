"""Apply the paper's typography to validated portable clinical figures."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.input_dir.rglob('*.plot.json'))
    if not sources:
        raise ValueError('No editable plot states; PNG layers alone cannot reproduce vector figures')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib.pyplot as plt
    from opera.visualization.style import setup_style
    from opera.visualization.plot_state import render_plot_state
    from opera.visualization.paper_figures import _finish_clinical
    setup_style()
    rows = []
    for source in sources:
        relative = source.relative_to(args.input_dir)
        target = args.output_dir / relative.parent / relative.name.removesuffix('.plot.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        fig = render_plot_state(source)
        for ax in fig.axes:
            for getter, setter in [(ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)]:
                label = getter()
                if 'second-line treatment or death' in label.lower():
                    # Endpoint definition stays explicit in the manuscript caption.
                    suffix = ' (95% CI)' if '95%' in label else ''
                    setter('Treatment-failure-free survival' + suffix)
        _finish_clinical(fig)
        for suffix in ['.pdf', '.png']:
            fig.savefig(target.with_suffix(suffix), dpi=300, bbox_inches='tight')
        plt.close(fig)
        rows.append({'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'target': str(target)})
    (args.output_dir / 'style_manifest.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
