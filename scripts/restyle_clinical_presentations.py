"""Apply the paper's typography to validated portable clinical figures."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure-width', type=float, default=7.2,
                        help='Paper-width canvas in inches, preserving aspect and point-size type')
    args = parser.parse_args()
    sources = sorted(args.input_dir.rglob('*.plot.json'))
    if not sources:
        raise ValueError('No editable plot states; PNG layers alone cannot reproduce vector figures')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib.pyplot as plt
    from opera.visualization.style import setup_style
    from opera.run.paper_story_figures import thesis_style
    from opera.visualization.plot_state import render_plot_state
    from opera.visualization.paper_figures import _finish_clinical
    setup_style()
    thesis_style()
    rows = []
    for source in sources:
        relative = source.relative_to(args.input_dir)
        target = args.output_dir / relative.parent / relative.name.removesuffix('.plot.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        fig = render_plot_state(source)
        from matplotlib.text import Text
        for text in fig.findobj(Text):
            text.set_fontfamily('TeX Gyre Pagella')
        width, height = fig.get_size_inches()
        fig.set_size_inches(args.figure_width, height * args.figure_width / width)
        for ax in fig.axes:
            for getter, setter in [(ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)]:
                label = getter()
                if 'second-line treatment or death' in label.lower():
                    # Endpoint definition stays explicit in the manuscript caption.
                    suffix = ' (95% CI)' if '95%' in label else ''
                    setter('Treatment-failure-free survival' + suffix)
            labels = [t.get_text() for t in ax.get_xticklabels()]
            if any('intermediate' in label.lower() for label in labels):
                compact = [label.replace('Low-\nintermediate', 'Low–\nint.')
                                .replace('High-\nintermediate', 'High–\nint.')
                                .replace('Low-intermediate', 'Low–int.')
                                .replace('High-intermediate', 'High–int.') for label in labels]
                ax.set_xticks(ax.get_xticks(), compact)
        if source.name == 'risk_groups.plot.json':
            # Presentation legends store their old figure coordinates. Anchor
            # them to the resized curve axes so they cannot spill over ticks.
            for legend, ax in zip(fig.legends[:2], fig.axes[:2]):
                box = ax.get_position()
                legend.set_loc('lower left')
                legend.set_bbox_to_anchor((box.x0 + .01 * box.width,
                                          box.y0 + .01 * box.height), transform=fig.transFigure)
                legend.set_title('')
                for text in legend.get_texts():
                    text.set_fontsize(7)
            for ax in fig.axes:
                if 'Treatment-failure-free survival' in ax.get_xlabel():
                    ax.set_xlabel('Treatment-failure-free survival\n(95% CI)')
        _finish_clinical(fig)
        for suffix in ['.pdf', '.png']:
            fig.savefig(target.with_suffix(suffix), dpi=300, bbox_inches='tight')
        plt.close(fig)
        rows.append({'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'target': str(target)})
    (args.output_dir / 'style_manifest.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
