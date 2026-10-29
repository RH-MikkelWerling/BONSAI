"""Geometry checks for rendered artwork. Findings require visual adjudication.

Unlike a successful canvas draw, this catches overlapping labels and legends
that escape the intended page. It never changes data or silently shrinks text.
"""
from itertools import combinations
import math

from matplotlib.text import Text
from matplotlib.transforms import Bbox


def audit_layout(fig, *, minimum_overlap_px=2.0):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden_ticks = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not lo - 1e-10 <= tick.get_loc() <= hi + 1e-10:
                    hidden_ticks.update([id(tick.label1), id(tick.label2)])
    texts = []
    for artist in fig.findobj(Text):
        if id(artist) in hidden_ticks or not artist.get_visible() or not artist.get_text().strip():
            continue
        # Annotation extents include their connector arrow; audit only text.
        box = Text.get_window_extent(artist, renderer=renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        # Tick objects outside the visible view are not part of the export.
        if artist.get_clip_on() and artist.axes is not None:
            if not box.overlaps(artist.axes.get_window_extent(renderer)):
                continue
        texts.append((artist, box))
    collisions = []
    for (a, x), (b, y) in combinations(texts, 2):
        # Axis-aligned envelopes of parallel diagonal labels overlap even when
        # the lettering has a clear gap. Measure separation across baselines.
        angle = a.get_rotation()
        if angle and angle == b.get_rotation() and '\n' not in a.get_text() + b.get_text():
            theta = math.radians(angle)
            dx, dy = (x.x0+x.x1-y.x0-y.x1)/2, (x.y0+x.y1-y.y0-y.y1)/2
            separation = abs(-math.sin(theta)*dx + math.cos(theta)*dy)
            line_height = max(a.get_fontsize(), b.get_fontsize()) * fig.dpi / 72 * 1.2
            if separation >= line_height:
                continue
        overlap = Bbox.intersection(x, y)
        if overlap is None or min(overlap.width, overlap.height) < minimum_overlap_px:
            continue
        collisions.append(dict(first=a.get_text(), second=b.get_text(),
                               width_px=round(overlap.width, 2),
                               height_px=round(overlap.height, 2)))
    legends = [*fig.legends, *[ax.get_legend() for ax in fig.axes if ax.get_legend()]]
    bounds = fig.bbox
    overflow = []
    for legend in legends:
        box = legend.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 2 or box.x1 > bounds.x1 + 2:
            overflow.append(dict(kind='legend_width', labels=[t.get_text() for t in legend.get_texts()]))
    return dict(text_collision_candidates=collisions, legend_overflow=overflow,
                text_count=len(texts), figure_inches=list(fig.get_size_inches()),
                note='Candidate overlaps require visual review; not a guarantee for unseen data.')
