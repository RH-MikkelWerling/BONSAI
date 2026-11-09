"""Portable presentation state: aggregate curves plus image-only dense layers.

This is a plotting fallback for clinical and embedding figures whose native
inputs must stay on the server. No scatter offsets, scalar arrays, patient
tables, pickles, arbitrary objects, or callbacks are serialized. Dense content
is retained as a PNG layer, like the already exportable publication figure.
Native aggregate recipes remain the preferred way to change statistical plots.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

SCHEMA = 'opera.presentation.v1'


def _colour(value):
    from matplotlib.colors import to_rgba
    return list(to_rgba(value))


def export_plot_state(fig, destination):
    """Export editable axes/text/aggregate lines and pixel-only dense layers."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    layers = destination.with_suffix('')
    layers.mkdir(exist_ok=True)
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    report = dict(schema=SCHEMA, inches=list(fig.get_size_inches()), axes=[], texts=[],
                  dense_layers='Publication-image pixels only; no patient point arrays',
                  scope='Presentation replay; new grouping or statistical calculations require native inputs')

    def text_record(text, transform):
        x, y = transform.inverted().transform(text.get_transform().transform(text.get_position()))
        return dict(text=text.get_text(), x=float(x), y=float(y),
                    fontsize=float(text.get_fontsize()), color=_colour(text.get_color()),
                    weight=text.get_fontweight(), style=text.get_fontstyle(),
                    family=text.get_fontfamily(), rotation=text.get_rotation(),
                    ha=text.get_ha(), va=text.get_va())

    visibility = [(artist, artist.get_visible()) for artist in fig.findobj()
                  if hasattr(artist, 'get_visible') and hasattr(artist, 'set_visible')]
    axes, ancestors = [], {}
    def collect(ax, parents=()):
        if ax in axes:return
        axes.append(ax);ancestors[ax]=parents
        for child in ax.child_axes:collect(child,parents+(ax,))
    for ax in fig.axes:collect(ax)
    try:
        for i, ax in enumerate(axes):
            record = dict(position=list(ax.get_position().bounds), xlim=list(ax.get_xlim()),
                          ylim=list(ax.get_ylim()), xscale=ax.get_xscale(), yscale=ax.get_yscale(),
                          xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), title=ax.get_title(),
                          xticks=list(ax.get_xticks()), yticks=list(ax.get_yticks()),
                          xticklabels=[t.get_text() for t in ax.get_xticklabels()],
                          yticklabels=[t.get_text() for t in ax.get_yticklabels()],
                          xrotation=[t.get_rotation() for t in ax.get_xticklabels()],
                          ticksize=float(ax.xaxis.get_ticklabels()[0].get_fontsize())
                                   if ax.xaxis.get_ticklabels() else 7,
                          labelsize=float(ax.xaxis.label.get_fontsize()),
                          axis_on=ax.axison, texts=[], lines=[], tables=[],
                          spines={k: dict(visible=v.get_visible(), color=_colour(v.get_edgecolor()),
                                         width=v.get_linewidth()) for k, v in ax.spines.items()})
            record['texts'] = [text_record(t, ax.transAxes) for t in
                [*ax.texts, ax.title, ax._left_title, ax._right_title]
                if t.get_visible() and t.get_text()]
            record['title']=''  # all three title locations are editable text above
            record['xgrid']=any(t.get_visible() for t in ax.get_xgridlines())
            record['ygrid']=any(t.get_visible() for t in ax.get_ygridlines())
            # Only curve coordinates, never PathCollection offsets or scalar arrays.
            for line in ax.lines if getattr(fig, 'opera_export_aggregate_curves', False) else []:
                if not line.get_visible():
                    continue
                xy = np.asarray(line.get_xydata(), dtype=float)
                if xy.ndim != 2 or xy.shape[1] != 2:
                    continue
                points = ax.transAxes.inverted().transform(line.get_transform().transform(xy))
                record['lines'].append(dict(points=np.where(np.isfinite(points), points, np.nan).tolist(),
                    color=_colour(line.get_color()), linewidth=line.get_linewidth(),
                    linestyle=line.get_linestyle(), marker=line.get_marker(),
                    markersize=line.get_markersize(), alpha=line.get_alpha(),
                    drawstyle=line.get_drawstyle(), zorder=line.get_zorder()))
            for table in ax.tables:
                cells = []
                for (row, column), cell in table.get_celld().items():
                    cells.append(dict(row=row, column=column, bounds=list(cell.get_bbox().bounds),
                        text=cell.get_text().get_text(), fontsize=cell.get_text().get_fontsize(),
                        weight=cell.get_text().get_fontweight(),
                        color=_colour(cell.get_text().get_color()), facecolor=list(cell.get_facecolor()),
                        edgecolor=list(cell.get_edgecolor())))
                record['tables'].append(cells)
            legend = ax.get_legend()
            if legend is not None and legend.get_visible():
                # Keep its approved appearance as an image; move/scale it offline.
                record['legend_bounds'] = list(legend.get_window_extent().bounds)
                from matplotlib.lines import Line2D
                handles=getattr(legend,'legend_handles',[])
                if handles and all(isinstance(h,Line2D) for h in handles):
                    record['legend']=dict(title=legend.get_title().get_text(),
                        fontsize=legend.get_texts()[0].get_fontsize(),
                        title_fontsize=legend.get_title().get_fontsize(),
                        ncols=getattr(legend,'_ncols',1),frameon=legend.get_frame_on(),
                        handles=[dict(label=t.get_text(),color=_colour(h.get_color()),
                            linestyle=h.get_linestyle(),linewidth=h.get_linewidth(),
                            marker=h.get_marker(),markersize=h.get_markersize())
                            for h,t in zip(handles,legend.get_texts())])
            report['axes'].append(record)

        report['texts'] = [text_record(t, fig.transFigure) for t in fig.texts if t.get_visible()]
        # Curves/text/tables are separate editable layers. Other graphical content
        # is raster only, so actual patient coordinates cannot enter the JSON.
        for artist, _ in visibility:
            from matplotlib.text import Text
            from matplotlib.lines import Line2D
            from matplotlib.table import Table
            if isinstance(artist, (Text, Table)) or (
                isinstance(artist, Line2D) and getattr(fig, 'opera_export_aggregate_curves', False)
            ):
                artist.set_visible(False)
        for ax in axes:
            ax.set_axis_off()
            if ax.get_legend() is not None:
                ax.get_legend().set_visible(False)
        for i, ax in enumerate(axes):
            for other in axes:
                other.set_visible(other is ax or other in ancestors[ax])
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
            box = ax.bbox
            left, right = max(0, int(np.floor(box.x0))), min(width, int(np.ceil(box.x1)))
            top, bottom = max(0, height-int(np.ceil(box.y1))), min(height, height-int(np.floor(box.y0)))
            if right > left and bottom > top:
                path = layers / f'axis-{i:03d}.png'
                Image.fromarray(rgba[top:bottom, left:right]).save(path)
                report['axes'][i]['image'] = str(path.relative_to(destination.parent))
        # Legend layers contain only the already-public legend graphic.
        for artist, visible in visibility:
            artist.set_visible(visible)
        for ax, settings in zip(axes, report['axes']):
            ax.set_axis_on() if settings['axis_on'] else ax.set_axis_off()
        for i, ax in enumerate(axes):
            legend = ax.get_legend()
            if legend is None or not legend.get_visible():
                continue
            for other in axes:
                other.set_visible(other is ax or other in ancestors[ax])
            fig.canvas.draw()
            box = legend.get_window_extent()
            left, right = max(0, int(box.x0)), min(width, int(np.ceil(box.x1)))
            top, bottom = max(0, height-int(np.ceil(box.y1))), min(height, height-int(box.y0))
            if right > left and bottom > top:
                path = layers / f'legend-{i:03d}.png'
                rgba = np.asarray(fig.canvas.buffer_rgba())
                Image.fromarray(rgba[top:bottom, left:right]).save(path)
                report['axes'][i]['legend_image'] = str(path.relative_to(destination.parent))
                report['axes'][i]['legend_position'] = [left/width, 1-bottom/height,
                                                       (right-left)/width, (bottom-top)/height]
    finally:
        for artist, visible in visibility:
            artist.set_visible(visible)
        for ax, settings in zip(axes, report['axes']):
            ax.set_axis_on() if settings['axis_on'] else ax.set_axis_off()
        fig.canvas.draw()
    # JSON null represents gaps in a curve, not arbitrary Python NaN literals.
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
    destination.write_text(json.dumps(clean(report), allow_nan=False))


def render_plot_state(source):
    """Replay portable presentation data with editable axes, labels and curves."""
    import matplotlib.pyplot as plt
    from matplotlib.table import Table
    source = Path(source)
    state = json.loads(source.read_text())
    if state.get('schema') != SCHEMA:
        raise ValueError('Unknown presentation schema')
    fig = plt.figure(figsize=state['inches'])

    def asset(name):
        path = (source.parent / name).resolve()
        if not path.is_relative_to(source.parent.resolve()):
            raise ValueError('Plot layer escapes its presentation directory')
        return plt.imread(path)

    for record in state['axes']:
        ax = fig.add_axes(record['position'])
        if record.get('image'):
            ax.imshow(asset(record['image']), extent=(0, 1, 0, 1), transform=ax.transAxes,
                      aspect='auto', zorder=0)
        ax.set_xscale(record['xscale']); ax.set_yscale(record['yscale'])
        ax.set_xlim(record['xlim']); ax.set_ylim(record['ylim'])
        ax.set_xticks(record['xticks'], record['xticklabels'])
        ax.set_yticks(record['yticks'], record['yticklabels'])
        ax.set_xlim(record['xlim']); ax.set_ylim(record['ylim'])
        for text, rotation in zip(ax.get_xticklabels(), record['xrotation']):
            text.set_rotation(rotation)
        ax.tick_params(labelsize=record['ticksize'])
        ax.set_xlabel(record['xlabel'], fontsize=record['labelsize'])
        ax.set_ylabel(record['ylabel'], fontsize=record['labelsize'])
        ax.set_title(record['title'], fontsize=record['labelsize'])
        ax.grid(False)
        if record.get('xgrid'):ax.grid(True,axis='x')
        if record.get('ygrid'):ax.grid(True,axis='y')
        for name, settings in record['spines'].items():
            ax.spines[name].set_visible(settings['visible'])
            ax.spines[name].set_color(settings['color'])
            ax.spines[name].set_linewidth(settings['width'])
        for settings in record['lines']:
            settings = dict(settings)
            points = np.asarray(settings.pop('points'), float)
            ax.plot(points[:, 0], points[:, 1], transform=ax.transAxes, **settings)
        for settings in record['texts']:
            settings = dict(settings)
            ax.text(settings.pop('x'), settings.pop('y'), settings.pop('text'),
                    transform=ax.transAxes, **settings)
        for cells in record['tables']:
            from matplotlib.patches import Rectangle
            for cell in cells:
                x, y, width, height = cell['bounds']
                ax.add_patch(Rectangle((x,y),width,height,transform=ax.transAxes,
                    facecolor=cell['facecolor'],edgecolor=cell['edgecolor'],clip_on=False))
                ax.text(x+width/2,y+height/2,cell['text'],transform=ax.transAxes,
                    ha='center',va='center',fontsize=cell['fontsize'],
                    color=cell['color'],fontweight=cell.get('weight','normal'),clip_on=False)
        if not record['axis_on']:
            ax.set_axis_off()
        if record.get('legend') and record.get('legend_position'):
            from matplotlib.lines import Line2D
            settings=record['legend'];x,y,width,height=record['legend_position']
            fig.legend(handles=[Line2D([],[],**h) for h in settings['handles']],
                title=settings['title'],fontsize=settings['fontsize'],
                title_fontsize=settings['title_fontsize'],ncol=settings['ncols'],
                frameon=settings['frameon'],loc='upper left',bbox_to_anchor=(x,y+height),
                bbox_transform=fig.transFigure,borderaxespad=0)
        elif record.get('legend_image'):
            overlay = fig.add_axes(record['legend_position'], zorder=100)
            overlay.imshow(asset(record['legend_image'])); overlay.set_axis_off()
    for settings in state['texts']:
        settings = dict(settings)
        fig.text(settings.pop('x'), settings.pop('y'), settings.pop('text'), **settings)
    return fig
