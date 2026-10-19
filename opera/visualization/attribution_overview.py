"""Figures for what the model relies on, in the OPERA figure style.

``figure_attribution_atlas`` (scientific)
    every outcome by clinical domain: what makes patients differ in predicted
    risk, by source and by time before treatment, and which findings raise or
    lower which risks, with the cells where an effect was expected boxed.
``figure_attribution_structure`` (scientific)
    which outcomes rest on the same parts of the history, and what each part
    of the record adds to discrimination, by clinical domain.
``figure_patient_tracks`` (clinical)
    one patient's history read for several complications: one track per
    outcome, with the episodes of care that move each risk at their time.
"""

from __future__ import annotations

import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

from opera.visualization.style import PALETTE


def outcome_name(outcome, *, short=False):
    from opera.visualization.outcome_names import outcome_display_name

    if outcome == "overall_survival":
        return "Death"
    name = outcome_display_name(outcome)
    if short:
        name = (name.replace(" composite", "").replace("Serious infection", "Infection")
                .replace(" (G3+)", " G3+"))
    return name


def _title(ax, letter, text, **kwargs):
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=8.6, **kwargs)


WINDOW_COLORS = {"Last 30 days": "#1A237E", "31 days to 1 year": "#5C7AEA",
                 "Over 1 year": "#AEB8DA", "No time": "#737D8C"}
REFERENCE_COLOR = "#ECEEF1"  # the timing of care alone: neutral, not a source


def _domain_rows(outcomes, domains):
    """Row layout with a bold header row before each clinical domain."""
    import yaml
    from pathlib import Path

    from opera.visualization.paper_figures import SHORT_GROUP
    from opera.visualization.paper_taxonomy import DISPLAY_GROUPS

    order = list(yaml.safe_load(Path(DISPLAY_GROUPS).read_text()))
    rank = {d: i for i, d in enumerate(order)}
    present = sorted({domains.get(o, "Other") for o in outcomes},
                     key=lambda d: rank.get(d, 99))
    rows, labels, headers = [], [], []
    for domain in present:
        headers.append(len(rows))
        rows.append(None)
        labels.append(SHORT_GROUP.get(domain, domain))
        for outcome in sorted((o for o in outcomes if domains.get(o, "Other") == domain),
                              key=outcome_name):
            rows.append(outcome)
            labels.append(outcome_name(outcome))
    return rows, labels, headers


def _row_axis(ax, rows, labels, headers, *, show):
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), labels if show else [""] * len(rows))
    ax.tick_params(axis="y", length=0, labelsize=5.4)
    if show:
        for index in headers:
            tick = ax.get_yticklabels()[index]
            tick.set_fontweight("bold")
            tick.set_fontsize(6.2)
            tick.set_color(PALETTE["ink"])
    for index in headers[1:]:
        ax.axhline(index - 0.5, color=PALETTE["grid"], lw=0.8)


def _stacked_shares(ax, table, rows, order, colors):
    values = table.pivot_table(index="outcome", columns="group", values="share")
    right = np.zeros(len(rows))
    left = np.zeros(len(rows))
    present = [r for r in rows if r is not None and r in values.index]
    for name in order:
        if name not in values:
            continue
        column = values[name]
        heights = np.array([100 * column.get(r, 0.0) if r is not None else 0.0 for r in rows])
        heights = np.nan_to_num(heights)
        pos, neg = heights.clip(min=0), heights.clip(max=0)
        ax.barh(range(len(rows)), pos, left=right, color=colors[name], height=0.78, lw=0.5, edgecolor="white")
        ax.barh(range(len(rows)), neg, left=left, color=colors[name], height=0.78, lw=0.5, edgecolor="white")
        right, left = right + pos, left + neg
    ax.axvline(0, color=PALETTE["zero_line"], lw=0.6)
    ax.set_xlim(min(0, left.min()) - 2, 100)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=5.8, length=2)
    ax.grid(False)
    for spine in ("right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    return present


def figure_attribution_atlas(variance_sources, variance_windows, concepts, *, domains=None,
                             mark=None):
    """What the model relies on, for every outcome.

    Rows are outcomes grouped by clinical domain. A: the differences in
    predicted risk between patients (variance of the predicted log-odds)
    divided into the part carried by the timing of care, age and subtype alone
    (grey) and the parts carried by what was recorded, by source. B: the same
    division by time before treatment. C: how much each finding changes the
    predicted log-odds among the patients who have it; boxed cells, with an
    arrow for the expected direction, are the prespecified expectations, and
    dots mark 95% intervals that exclude zero.
    """
    from opera.interpretability.attribution_patterns import REFERENCE
    from opera.visualization.paper_figures import RISK_CMAP, SOURCE_COLORS, watermark
    from opera.visualization.paper_taxonomy import outcome_endpoint_groups

    domains = domains or outcome_endpoint_groups()
    outcomes = list(dict.fromkeys(variance_sources.outcome))
    rows, labels, headers = _domain_rows(outcomes, domains)
    findings = list(dict.fromkeys(concepts.label)) if len(concepts) else []
    height = 0.105 * len(rows) + 2.6
    fig = plt.figure(figsize=(7.2, height))
    grid = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.3, max(2.0, 0.2 * len(findings))],
                            wspace=0.06, left=0.25, right=0.985,
                            top=1 - 1.35 / height, bottom=0.85 / height)
    title_y = 1 - 0.1 / height
    fig.text(0.25, title_y, "What makes patients differ in predicted risk", fontsize=8.6,
             va="top", fontweight="bold")
    panel_y = 1 - 0.38 / height

    ax = fig.add_subplot(grid[0])
    source_colors = {REFERENCE: REFERENCE_COLOR,
                     **{k: v for k, v in SOURCE_COLORS.items() if k != "Other"}}
    _stacked_shares(ax, variance_sources, rows, list(source_colors), source_colors)
    _row_axis(ax, rows, labels, headers, show=True)
    fig.text(ax.get_position().x0, panel_y, "A  By source", fontsize=8.2, va="top")
    ax.legend(handles=[Patch(color=c, label=n if n != REFERENCE else "Timing of care alone")
                       for n, c in source_colors.items()],
              loc="upper left", bbox_to_anchor=(-0.02, -0.12 / height * 3), ncol=2,
              frameon=False, fontsize=5.6, handlelength=1.0, columnspacing=0.8)
    ax.set_xlabel("Share of the differences in predicted risk (%)", fontsize=6)
    ax.xaxis.set_label_position("top")

    bx = fig.add_subplot(grid[1])
    window_colors = {REFERENCE: REFERENCE_COLOR, **WINDOW_COLORS}
    _stacked_shares(bx, variance_windows, rows, list(window_colors), window_colors)
    _row_axis(bx, rows, labels, headers, show=False)
    fig.text(bx.get_position().x0, panel_y, "B  By time", fontsize=8.2,
             va="top")
    bx.legend(handles=[Patch(color=c, label=n.replace("No time", "Background")
                             .replace("31 days to 1 year", "1 month to 1 year"))
                       for n, c in WINDOW_COLORS.items()],
              loc="upper left", bbox_to_anchor=(-0.02, -0.12 / height * 3), ncol=1,
              frameon=False, fontsize=5.6, handlelength=1.0)
    bx.set_xlabel("Share (%)", fontsize=6)
    bx.xaxis.set_label_position("top")

    if findings:
        cx = fig.add_subplot(grid[2])
        values = concepts.pivot_table(index="outcome", columns="label", values="mean_pp")
        matrix = np.array([[values.at[r, f] if r is not None and r in values.index
                            and f in values else np.nan for f in findings] for r in rows])
        limit = max(0.05, float(np.nanquantile(np.abs(matrix), 0.98)))
        image = cx.imshow(np.ma.masked_invalid(matrix), cmap=RISK_CMAP,
                          norm=TwoSlopeNorm(0, -limit, limit), aspect="auto",
                          interpolation="nearest")
        cx.grid(False)
        for row in concepts.itertuples():
            if row.outcome not in rows or row.label not in findings:
                continue
            i, j = rows.index(row.outcome), findings.index(row.label)
            if np.isfinite(row.lower_pp) and (row.lower_pp > 0 or row.upper_pp < 0):
                cx.scatter(j, i, s=3, color="white", edgecolors=PALETTE["ink"],
                           linewidths=0.3, zorder=3)
            if row.expected in ("+", "-"):
                cx.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                       ec=PALETTE["ink"], lw=0.8, zorder=4))
        for index in headers:
            cx.axhspan(index - 0.5, index + 0.5, color="white", zorder=2, lw=0)
        cx.set_xticks(range(len(findings)), findings, fontsize=5.6, rotation=90)
        cx.xaxis.tick_top()
        _row_axis(cx, rows, labels, headers, show=False)
        cx.tick_params(axis="x", length=0)
        for spine in cx.spines.values():
            spine.set_visible(False)
        tested = concepts.loc[concepts.verdict.isin(["supported", "contradicted",
                                                     "inconclusive"])]
        supported = int((tested.verdict == "supported").sum())
        fig.text(cx.get_position().x0, title_y, "Which findings raise which risks",
                 fontsize=8.6, va="top", fontweight="bold")
        fig.text(cx.get_position().x0, panel_y,
                 f"C  {supported} of {len(tested)} boxed expectations seen", fontsize=8.2,
                 va="top")
        position = cx.get_position()
        cax = fig.add_axes([position.x0 + 0.1 * position.width, 0.42 / height,
                            0.8 * position.width, 0.08 / height])
        bar = fig.colorbar(image, cax=cax, orientation="horizontal")
        bar.set_label("Odds ratio among patients with the finding (model)", fontsize=5.6)
        bar.ax.tick_params(labelsize=5.4)
        bar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"×{np.exp(v):.2f}"))
        bar.outline.set_visible(False)
    watermark(fig, mark)
    return fig


def figure_attribution_structure(similarity, restrictions=None, *, domains=None, mark=None):
    """Does the model read the same history differently for different outcomes?

    A: for every pair of outcomes, how much they agree on which episodes of a
    patient's history raise or lower risk (Spearman correlation of episode
    contributions within patients, averaged over patients), outcomes ordered
    by clinical domain with each domain outlined. B: change in held-out AUC
    (x100) when the fitted model reads only part of the record, averaged over
    the outcomes of each domain, in the same order.
    """
    from opera.visualization.paper_figures import RISK_CMAP, SHORT_GROUP, watermark
    from opera.visualization.paper_taxonomy import outcome_endpoint_groups

    domains = domains or outcome_endpoint_groups()
    rows, labels, headers = _domain_rows(list(dict.fromkeys(similarity.outcome_a)), domains)
    outcomes = [r for r in rows if r is not None]
    matrix = (similarity.pivot_table(index="outcome_a", columns="outcome_b", values="spearman")
              .reindex(index=outcomes, columns=outcomes))
    groups = [domains.get(o, "Other") for o in outcomes]
    starts = [i for i in range(len(outcomes)) if i == 0 or groups[i] != groups[i - 1]]
    ends = [*starts[1:], len(outcomes)]
    pairs = similarity.loc[(similarity.outcome_a < similarity.outcome_b)
                           & similarity.spearman.notna()]
    same = pairs.outcome_a.map(domains) == pairs.outcome_b.map(domains)
    within = pairs.loc[same].spearman.median() if same.any() else np.nan
    between = pairs.loc[~same].spearman.median() if (~same).any() else np.nan

    has_b = restrictions is not None and len(restrictions)
    fig = plt.figure(figsize=(7.2, 4.4))
    grid = fig.add_gridspec(1, 2 if has_b else 1, width_ratios=[1.35, 1] if has_b else [1],
                            left=0.13, right=0.99, top=0.84, bottom=0.17, wspace=0.42)
    ax = fig.add_subplot(grid[0])
    image = ax.imshow(matrix.to_numpy(float), cmap=RISK_CMAP.reversed(),
                      norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest", aspect="auto")
    ax.grid(False)
    for start, end in zip(starts, ends):
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), end - start, end - start,
                               fill=False, ec=PALETTE["ink"], lw=0.8))
    centres = [(s0 + e0 - 1) / 2 for s0, e0 in zip(starts, ends)]
    names = [SHORT_GROUP.get(groups[s0], groups[s0]) for s0 in starts]
    ax.set_yticks(centres, names, fontsize=6.0)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    bar.set_label("Agreement on which episodes raise or lower risk", fontsize=6.2)
    bar.ax.tick_params(labelsize=5.8)
    bar.ax.yaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
    bar.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"×{np.exp(v):.2f}"))
    bar.outline.set_visible(False)
    _title(ax, "A", "Outcomes of one domain rest on the same history", pad=16)
    ax.text(0, 1.02, f"Median agreement within domains {within:.2f}, between domains "
            f"{between:.2f}", transform=ax.transAxes, fontsize=6.2,
            color=PALETTE["ink_secondary"])
    if has_b:
        bx = fig.add_subplot(grid[1])
        restriction_names = list(dict.fromkeys(restrictions.restriction))
        order = ["All outcomes", *[groups[s0] for s0 in starts]]
        table = (restrictions.pivot_table(index="group", columns="restriction",
                                          values="mean_delta")
                 .reindex(index=order, columns=restriction_names)) * 100
        limit = max(0.5, float(np.nanmax(np.abs(table.to_numpy()))))
        bx.imshow(table.to_numpy(float), cmap=RISK_CMAP.reversed(),
                  norm=TwoSlopeNorm(0, -limit, limit), aspect="auto", interpolation="nearest")
        bx.grid(False)
        for (i, j), value in np.ndenumerate(table.to_numpy(float)):
            if np.isfinite(value):
                bx.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=5.8,
                        color="white" if abs(value) > 0.6 * limit else PALETTE["ink"],
                        fontweight="bold" if i == 0 else "normal")
        bx.set_yticks(range(len(order)), [SHORT_GROUP.get(g, g) for g in order], fontsize=6.0)
        bx.get_yticklabels()[0].set_fontweight("bold")
        bx.axhline(0.5, color="white", lw=2)
        bx.set_xticks(range(len(restriction_names)),
                      [textwrap.fill(n, 12) for n in restriction_names], fontsize=5.8)
        bx.xaxis.tick_top()
        bx.tick_params(length=0)
        for spine in bx.spines.values():
            spine.set_visible(False)
        _title(bx, "B", "AUC change (x100) on part of the record", pad=26)
    watermark(fig, mark)
    return fig


def _sigmoid(value):
    return 1 / (1 + np.exp(-value))


def _tracks(ax, episodes, outcomes, *, scale="risk", label_top=5, size=1.0, levels=3,
            key=False):
    """One patient's history read for several outcomes: one track per outcome.

    x is time before treatment (log scale, treatment on the right). Every
    episode of care is a faint dot on every track; episodes that move an
    outcome's risk are circles sized by how much and coloured by direction
    (sienna raises, indigo lowers). The episodes that matter most for any
    outcome are labelled once, above the tracks (up to two findings, each in
    full, and the timing), with a guide line through all tracks. ``key`` adds a
    circle-size key below the tracks.
    """
    from opera.visualization.paper_figures import RISK_CMAP

    x = lambda d: np.asarray(d, float) + 1.0  # noqa: E731  (treatment at 1)
    part = episodes.loc[episodes.outcome.isin(outcomes)]
    largest = max(float(part.contribution.abs().max()), 1e-9)
    strength = part.groupby("episode").contribution.apply(lambda v: v.abs().max())
    labelled = strength.sort_values(ascending=False).index[:label_top]
    info = part.drop_duplicates("episode").set_index("episode")
    chosen = info.loc[labelled].sort_values("start_days", ascending=False)
    step = 0.62
    for k, (episode, row) in enumerate(chosen.iterrows()):
        centre = np.sqrt(x(row.start_days) * x(row.end_days))
        level = -0.75 - step * (k % levels)
        ax.plot([centre, centre], [level + 0.12, len(outcomes) - 0.5], color=PALETTE["grid"],
                lw=0.8, zorder=0)
        findings = [textwrap.shorten(f, 34, placeholder="…")
                    for f in str(row.description).split("; ")[:2]]
        ax.text(centre, level - 0.02, "\n".join(findings), ha="center", va="bottom",
                fontsize=6.0, color=PALETTE["ink"], linespacing=1.15,
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.92), zorder=5)
        ax.text(centre, level + 0.02, row.when, ha="center", va="top", fontsize=5.6,
                color=PALETTE["ink_muted"], zorder=5)
    for y, outcome in enumerate(outcomes):
        track = part.loc[part.outcome == outcome]
        ax.axhline(y, color=PALETTE["grid"], lw=0.8, zorder=1)
        ax.scatter(x(track.start_days), np.full(len(track), y), s=4, color="#C9CED6",
                   zorder=2, linewidths=0)
        strong = track.loc[track.contribution.abs() >= 0.08 * largest]
        areas = 20 + 520 * size * (strong.contribution.abs() / largest)
        colors = [RISK_CMAP(0.88) if c > 0 else RISK_CMAP(0.12) for c in strong.contribution]
        ax.scatter(x(strong.start_days), np.full(len(strong), y), s=areas, c=colors,
                   alpha=0.85, zorder=3, linewidths=0.6, edgecolors="white")
        for row in strong.itertuples():
            if abs(row.contribution) >= 0.35 * largest:
                text = (f"{row.contribution:+.1f}" if scale == "risk"
                        else f"{row.contribution:+.2f}")
                ax.text(x(row.start_days), y, text, ha="center", va="center", fontsize=5.4,
                        color="white", fontweight="bold", zorder=4)
        if len(track):
            prediction, reference = track.prediction.iloc[0], track.reference.iloc[0]
            if scale == "logit":
                prediction, reference = _sigmoid(prediction), _sigmoid(reference)
            ax.annotate(f"{100 * prediction:.0f}%", xy=(1.0, y),
                        xycoords=("axes fraction", "data"), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        fontweight="bold", color=PALETTE["ink"])
            ax.annotate(f"from {100 * reference:.0f}%", xy=(1.0, y),
                        xycoords=("axes fraction", "data"), xytext=(30, 0),
                        textcoords="offset points", va="center", fontsize=5.6,
                        color=PALETTE["ink_muted"])
    ax.set_xscale("log")
    ticks = [1, 8, 31, 91, 366, 1826]
    ax.set_xticks(ticks, ["Treatment", "1 wk", "1 mo", "3 mo", "1 y", "5 y"], fontsize=6.2)
    ax.minorticks_off()
    ax.set_xlim(max(float(part.start_days.max()) * 1.6 + 1, 400), 0.8)
    ax.set_yticks(range(len(outcomes)), [outcome_name(o, short=True) for o in outcomes],
                  fontsize=6.8)
    ax.set_ylim(len(outcomes) - 0.5, -0.75 - step * (min(levels, len(chosen)) - 1) - 0.55)
    ax.annotate("Predicted", xy=(1.0, -0.55), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=5.6,
                color=PALETTE["ink_muted"])
    if key:
        values = [v for v in (largest / 4, largest / 2, largest) if v > 0]
        unit = " pp" if scale == "risk" else ""
        inset = ax.inset_axes([0.5, -0.3, 0.5, 0.12])
        inset.set_axis_off()
        inset.set_xlim(-0.2, len(values) + 0.1)
        inset.set_ylim(-1, 1)
        inset.text(-0.2, 0, "Change in the\npredicted risk", ha="right", va="center",
                   fontsize=6, color=PALETTE["ink_muted"], linespacing=1.1)
        for i, v in enumerate(values):
            inset.scatter([i + 0.25], [0], s=20 + 520 * size * v / largest, color="#B9BFC8",
                          linewidths=0, clip_on=False)
            inset.text(i + 0.62, 0, f"{v:.1f}{unit}", va="center", fontsize=6,
                       color=PALETTE["ink"])
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def figure_patient_tracks(episodes, *, outcomes=None, scale="logit", mark=None):
    """Supplement: one synthetic patient's history read for several complications.

    ``episodes``: :func:`opera.interpretability.attribution_patterns.episode_contributions`.
    Each track is one outcome; circles are the episodes of care that move its
    predicted risk, sized by how much and coloured by direction, at their time
    before treatment. The predicted risk is given at the right, with the risk
    for the same timing of care and average findings.
    """
    from opera.visualization.paper_figures import watermark

    outcomes = outcomes or list(dict.fromkeys(episodes.outcome))
    fig = plt.figure(figsize=(7.2, 0.55 * len(outcomes) + 1.9))
    ax = fig.add_axes([0.2, 0.12, 0.66, 0.84])
    _tracks(ax, episodes, outcomes, scale=scale, label_top=6)
    ax.set_xlabel("Time before the start of treatment", fontsize=7)
    watermark(fig, mark)
    return fig


def _by_domain(frame, domains, value, keys):
    frame = frame.assign(domain=frame.outcome.map(domains))
    return frame.groupby(["domain", *keys])[value].mean().reset_index()


def _cell_grid(ax, shape):
    """White borders between heatmap cells, as in the other paper figures."""
    rows, cols = shape
    for y in range(1, rows):
        ax.axhline(y - 0.5, color="white", lw=1.2)
    for x in range(1, cols):
        ax.axvline(x - 0.5, color="white", lw=1.2)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_attribution_main(variance_sources, concepts, restrictions, episodes, *,
                            domains=None, mark=None):
    """Main text: the model relies on sensible findings, read per complication.

    Rows of A-C are clinical domains. A: the differences in predicted risk
    between patients carried by the timing of care alone (grey) and by each
    source of records. B: change in held-out AUC when the fitted model reads
    only part of the record (column headers: all outcomes). C: change in
    predicted log-odds among the patients with each finding; boxed cells are
    prespecified expected effects, solid where seen. D: one synthetic patient's
    history read for three complications. Sienna and indigo mark effects
    throughout; the other colours identify sources.
    """
    import yaml
    from pathlib import Path

    from opera.interpretability.attribution_patterns import REFERENCE
    from opera.visualization.paper_figures import (
        GAIN_CMAP, RISK_CMAP, SHORT_GROUP, SOURCE_COLORS, watermark)
    from opera.visualization.paper_taxonomy import DISPLAY_GROUPS, outcome_endpoint_groups

    domains = domains or outcome_endpoint_groups()
    order = [d for d in yaml.safe_load(Path(DISPLAY_GROUPS).read_text())
             if d in set(variance_sources.outcome.map(domains))]
    names = [SHORT_GROUP.get(d, d) for d in order]
    rows = np.arange(len(order))
    fig = plt.figure(figsize=(7.2, 9.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.3, left=0.135,
                             right=0.885, top=0.815, bottom=0.085)
    top = outer[0].subgridspec(1, 3, width_ratios=[0.85, 1.2, 2.3], wspace=0.06)
    title_y = 0.99

    # A: sources of the differences between patients.
    ax = fig.add_subplot(top[0])
    shares = _by_domain(variance_sources, domains, "share", ["group"]).pivot_table(
        index="domain", columns="group", values="share").reindex(order) * 100
    colors = {REFERENCE: REFERENCE_COLOR,
              **{k: v for k, v in SOURCE_COLORS.items() if k != "Other"}}
    left = np.zeros(len(order))
    for name, color in colors.items():
        if name in shares:
            values = shares[name].fillna(0).clip(lower=0).to_numpy()
            ax.barh(rows, values, left=left, color=color, height=0.8, lw=0.5, edgecolor="white")
            left += values
    ax.set_yticks(rows, names, fontsize=7)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of the differences\nbetween patients (%)", fontsize=7)
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # B: discrimination without each part of the record.
    bx = fig.add_subplot(top[1])
    names_b = list(dict.fromkeys(restrictions.restriction))
    grid_b = (restrictions.loc[restrictions.group != "All outcomes"]
              .pivot_table(index="group", columns="restriction", values="mean_delta")
              .reindex(index=order, columns=names_b)) * 100
    limit_b = max(0.5, float(np.nanmax(np.abs(grid_b.to_numpy()))))
    bx.imshow(grid_b.to_numpy(float), cmap=GAIN_CMAP, norm=TwoSlopeNorm(0, -limit_b, limit_b),
              aspect="auto", interpolation="nearest")
    for (i, j), value in np.ndenumerate(grid_b.to_numpy(float)):
        if np.isfinite(value):
            bx.text(j, i, f"{value:.1f}".replace("-", "−"), ha="center", va="center",
                    fontsize=5.8, color="white" if abs(value) > 0.6 * limit_b else PALETTE["ink"])
    short = {"Only the timing of care": "Timing\nonly", "Without laboratory results": "No\nlabs",
             "Without medication": "No\nmeds", "Without diagnoses": "No\ndiagn.",
             "Without region and specialty": "No\nregion",
             "Only the last year": "Last\nyear"}
    overall = (restrictions.loc[restrictions.group == "All outcomes"]
               .set_index("restriction").mean_delta * 100)
    heads = [f"{short.get(n, textwrap.fill(n, 8))}\n{overall.get(n, np.nan):.1f}"
             .replace("-", "−") for n in names_b]
    bx.set_xticks(range(len(names_b)), heads, fontsize=5.6)
    bx.xaxis.tick_top()
    bx.set_yticks([])
    _cell_grid(bx, grid_b.shape)
    bx.set_xlabel("Change in AUC (×100)\nheader: all outcomes", fontsize=7)

    # C: findings and the risks they raise, columns grouped by source.
    cx = fig.add_subplot(top[2])
    group_order = ["Laboratory", "Pathology", "Diagnoses", "Medication"]
    labels = (concepts.drop_duplicates("label")
              .assign(rank=lambda d: d.group.map({g: i for i, g in enumerate(group_order)}))
              .sort_values("rank", kind="stable"))
    findings = list(labels.label)
    group_of = dict(zip(labels.label, labels.group))
    grid_c = _by_domain(concepts, domains, "mean_pp", ["label"]).pivot_table(
        index="domain", columns="label", values="mean_pp").reindex(index=order, columns=findings)
    limit_c = max(0.05, float(np.nanquantile(np.abs(grid_c.to_numpy()), 0.97)))
    image = cx.imshow(np.ma.masked_invalid(grid_c.to_numpy(float)), cmap=RISK_CMAP,
                      norm=TwoSlopeNorm(0, -limit_c, limit_c), aspect="auto",
                      interpolation="nearest")
    _cell_grid(cx, grid_c.shape)
    groups_c = [group_of.get(f) for f in findings]
    edges = [0] + [j for j in range(1, len(findings)) if groups_c[j] != groups_c[j - 1]] + [
        len(findings)]
    for j in edges[1:-1]:
        cx.axvline(j - 0.5, color="white", lw=4)
    expected = concepts.loc[concepts.expected.isin(["+", "-"])].assign(
        domain=lambda d: d.outcome.map(domains))
    for (domain, label), part in expected.groupby(["domain", "label"]):
        if domain in order and label in findings:
            seen = (part.verdict == "supported").mean() >= 0.5
            cx.add_patch(Rectangle((findings.index(label) - 0.42, order.index(domain) - 0.42),
                                   0.84, 0.84, fill=False, lw=1.0 if seen else 0.8,
                                   ec=PALETTE["ink"], ls="-" if seen else (0, (2, 1.5))))
    cx.set_xticks(range(len(findings)), [f.replace(", last", ",\nlast") for f in findings],
                  fontsize=6, rotation=90)
    cx.xaxis.tick_top()
    cx.set_yticks([])
    # Source headers just above the finding labels, in the source colour.
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    label_top = max(inv.transform((0, t.get_window_extent(renderer).y1))[1]
                    for t in cx.get_xticklabels())
    for start, stop in zip(edges[:-1], edges[1:]):
        group = groups_c[start]
        color = SOURCE_COLORS.get(group, PALETTE["ink"])
        fx0 = inv.transform(cx.transData.transform((start - 0.45, 0)))[0]
        fx1 = inv.transform(cx.transData.transform((stop - 0.55, 0)))[0]
        fig.add_artist(plt.Line2D([fx0, fx1], [label_top + 0.008] * 2, color=color, lw=2,
                                  solid_capstyle="butt", transform=fig.transFigure))
        fig.text((fx0 + fx1) / 2, label_top + 0.014, group, ha="center", va="bottom",
                 fontsize=6.8, color=color, fontweight="bold")
    tested = concepts.loc[concepts.verdict.isin(["supported", "contradicted", "inconclusive"])]
    supported = int((tested.verdict == "supported").sum())
    position = cx.get_position()
    cax = fig.add_axes([position.x1 + 0.012, position.y0 + 0.15 * position.height,
                        0.012, 0.7 * position.height])
    bar = fig.colorbar(image, cax=cax)
    # Values are mean changes in log-odds (comparable across outcomes with
    # different base rates); labelled as the odds ratios they correspond to.
    bar.set_label("Odds ratio with the finding (model)", fontsize=6.4)
    bar.ax.tick_params(labelsize=5.8)
    bar.ax.yaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
    bar.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"×{np.exp(v):.2f}"))
    bar.outline.set_visible(False)
    cx.set_xlabel(f"Boxes: prespecified expected effects, solid where seen "
                  f"({supported} of {len(tested)})", fontsize=6.6)

    for x0, text in ((0.012, "A  Sources of the differences"),
                     (bx.get_position().x0, "B  Inputs removed"),
                     (cx.get_position().x0, "C  Findings and the risks they raise")):
        fig.text(x0, title_y, text, fontsize=10.5, va="top")
    fig.legend(handles=[Patch(color=c, label="Timing of care alone" if n == REFERENCE else n)
                        for n, c in colors.items()],
               loc="upper left", bbox_to_anchor=(0.012, title_y - 0.03), ncol=1,
               frameon=False, fontsize=6.2, handlelength=1.0, labelspacing=0.35)

    # D: one patient's history read for three complications, full width.
    dx = fig.add_subplot(outer[1])
    outcomes_d = [o for o in ("overall_survival", "serious_infection_composite",
                              "neutropenia_g3plus") if o in set(episodes.outcome)]
    _tracks(dx, episodes, outcomes_d, scale="risk", label_top=6, size=1.2, levels=2,
            key=True)
    dx.set_xlabel("Time before the start of treatment", fontsize=7, loc="left")
    fig.text(0.012, dx.get_position().y1 + 0.012, "D  One patient's history, read for three "
             "risks", fontsize=10.5, va="bottom")
    watermark(fig, mark)
    return fig


__all__ = ["figure_attribution_atlas", "figure_attribution_main", "figure_attribution_structure",
           "figure_patient_tracks",
           "outcome_name"]
