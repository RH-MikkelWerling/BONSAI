"""Decision curves: net benefit of acting on a risk prediction, with censoring.

Net benefit at threshold probability p is the share of patients correctly
flagged minus the share wrongly flagged, weighted by the odds p / (1 - p) at
which a clinician would trade one for the other (Vickers & Elkin, 2006). The
status at the horizon is unknown for patients censored before it, so true and
false positives are weighted by the inverse probability of remaining
uncensored, as in the paper's primary metrics. A clinical index has
categories, not probabilities; it enters as the observed risk of each category
among tuning patients, so its curve is a step function fixed before testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from opera.evaluation.secondary_analysis import _censoring_survival_before


def ipcw_status(time_days, event, horizon_days):
    """Case indicator and censoring weight at the horizon for each patient.

    Cases (event 1 by the horizon) and competing deaths by the horizon get
    1 / G(t-), patients followed beyond the horizon 1 / G(h-), and patients
    censored before it weight 0, so weighted shares estimate the full cohort.
    """
    t = np.asarray(time_days, float)
    e = np.asarray(event, int)
    before = _censoring_survival_before(t, e)
    case = (t <= horizon_days) & (e == 1)
    competing = (t <= horizon_days) & (e == 2)
    beyond = (t > horizon_days) | ((t >= horizon_days) & (e == 0))
    weight = np.zeros(len(t))
    weight[case | competing] = 1.0 / before(t[case | competing])
    weight[beyond & ~competing] = 1.0 / before(np.full((beyond & ~competing).sum(), horizon_days))
    return case, weight


def net_benefit(risk, case, weight, thresholds):
    """Net benefit of acting when ``risk`` is at or above each threshold.

    ``risk`` may be 2-D (resamples x patients) with ``case`` and ``weight``
    indexed the same way, which the bootstrap uses.
    """
    risk, case, weight = (np.atleast_2d(np.asarray(a, float)) for a in (risk, case, weight))
    n = case.shape[1]
    rows = []
    for p in thresholds:
        act = risk >= p
        true_pos = (weight * case * act).sum(axis=1) / n
        false_pos = (weight * (1 - case) * act).sum(axis=1) / n
        rows.append(true_pos - false_pos * p / (1 - p))
    return np.array(rows).T  # resamples x thresholds


def index_risk(tuning_groups, tuning_case, tuning_weight, held_groups):
    """Each held-out patient's index category mapped to its tuning-observed risk."""
    frame = pd.DataFrame({"group": tuning_groups, "case": tuning_case, "w": tuning_weight})
    frame = frame[frame.w > 0]
    risk = (frame.case * frame.w).groupby(frame.group).sum() / frame.w.groupby(frame.group).sum()
    mapped = pd.Series(held_groups).map(risk)
    if mapped.isna().any():
        raise ValueError("An index category has no tuning patients with known status")
    return mapped.to_numpy(float)


def decision_curves(held, tuning, horizon_days, thresholds, comparators=(),
                    reference="index", n_boot=1000, seed=0):
    """Net benefit of each strategy, and of the model minus each other, with 95% CIs.

    ``held`` and ``tuning`` hold ``model_risk``, ``ipi_group``, ``time_days`` and
    ``event`` per patient; ``comparators`` names extra risk columns in ``held``
    (for example tabular models). Weights are held fixed across resamples.
    """
    case, weight = ipcw_status(held.time_days, held.event, horizon_days)
    t_case, t_weight = ipcw_status(tuning.time_days, tuning.event, horizon_days)
    risks = {"model": held.model_risk.to_numpy(float),
             "index": index_risk(tuning.ipi_group.to_numpy(), t_case, t_weight,
                                 held.ipi_group.to_numpy()),
             **{name: held[name].to_numpy(float) for name in comparators},
             "treat_all": np.ones(len(held))}
    thresholds = np.asarray(thresholds, float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(held), size=(n_boot, len(held)))
    point = {name: net_benefit(r, case, weight, thresholds)[0] for name, r in risks.items()}
    boot = {name: net_benefit(r[draws], case[draws], weight[draws], thresholds)
            for name, r in risks.items()}
    rows = []
    for name in risks:
        low, high = np.percentile(boot[name], [2.5, 97.5], axis=0)
        rows += [{"strategy": name, "threshold": p, "net_benefit": v, "lower": lo, "upper": hi}
                 for p, v, lo, hi in zip(thresholds, point[name], low, high)]
    rows += [{"strategy": "treat_none", "threshold": p, "net_benefit": 0.0, "lower": 0.0, "upper": 0.0}
             for p in thresholds]
    for other in [s for s in risks if s != "model"]:
        diff = boot["model"] - boot[other]
        low, high = np.percentile(diff, [2.5, 97.5], axis=0)
        rows += [{"strategy": f"model_minus_{other}", "threshold": p,
                      "net_benefit": point["model"][i] - point[other][i], "lower": low[i], "upper": high[i]}
                 for i, p in enumerate(thresholds)]
    return pd.DataFrame(rows)


STRATEGY_STYLE = {
    # The paper's model colours (paper_figures.MODEL_STYLE): OPERA as the joint
    # model's navy, tabular models as XGBoost's teal; the index in brick.
    "model": ("#2D3A8C", "-", "Model"),
    "index": ("#B5451B", "-", "Index"),
    "treat_all": ("#9B9B9B", "--", "Act for everyone"),
    "treat_none": ("#5F6670", ":", "Act for no one"),
}
COMPARATOR_COLOURS = ["#408579", "#7189BC", "#8974AC"]


def operating_point_rows(table, points):
    """Net benefit of every strategy at named thresholds (``points``: label ->
    threshold, e.g. the model's low- and high-risk cutoffs fixed on tuning)."""
    rows = []
    for label, threshold in points.items():
        at = table.loc[(table.threshold - threshold).abs() < 1e-9]
        if at.empty:
            raise ValueError(f"Threshold {threshold} for {label!r} is not in the curve")
        rows.append(at.assign(operating_point=label))
    return pd.concat(rows, ignore_index=True) if rows else table.iloc[:0]


def figure_decision_curve(table, *, index_label="IPI", model_label="OPERA",
                          labels=None, outcome_label="death", horizon_days=730,
                          mark=None, points=None):
    """A: net benefit of each strategy. B: the model minus the index and each comparator.

    Both panels in net true positives per 1,000 patients; type, panel titles
    and keys as in the other paper figures (``paper_figures``). ``points``
    (label -> threshold) are marked as dotted vertical lines, e.g. the model's
    risk-group cutoffs fixed on tuning data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from opera.visualization.paper_figures import _key_legend, _panel_title, watermark
    from opera.visualization.style import PALETTE, despine, setup_style

    setup_style()
    labels = {"model": model_label, "index": index_label, **(labels or {})}
    per = 1000
    fig = plt.figure(figsize=(7.2, 3.5))
    a = fig.add_axes([0.085, 0.3, 0.38, 0.6])
    b = fig.add_axes([0.6, 0.3, 0.38, 0.6])
    strategies = [s for s in table.strategy.unique() if not s.startswith("model_minus_")]
    extra = iter(COMPARATOR_COLOURS)
    styles = {}
    for name in strategies:
        colour, line, label = (STRATEGY_STYLE[name] if name in STRATEGY_STYLE
                               else (next(extra), "-", name))
        styles[name] = (colour, line, labels.get(name, label))
    handles = []
    # References first, so the strategies are drawn on top of them.
    order = ([s for s in ("treat_none", "treat_all") if s in strategies]
             + [s for s in strategies if s not in ("treat_none", "treat_all", "model")]
             + (["model"] if "model" in strategies else []))
    for name in order:
        colour, line, label = styles[name]
        part = table[table.strategy == name]
        a.plot(part.threshold * 100, part.net_benefit * per, line, color=colour,
               lw=1.8 if name == "model" else (1.0 if name.startswith("treat") else 1.4),
               zorder=3 if name == "model" else 2)
    model = table[table.strategy == "model"]
    a.fill_between(model.threshold * 100, model.lower * per, model.upper * per,
                   color=styles["model"][0], alpha=0.12, lw=0, zorder=1)
    top = table[table.strategy.isin(strategies)].net_benefit.max() * per
    a.set_ylim(-0.08 * top, 1.08 * top)
    a.set_xlim(table.threshold.min() * 100, table.threshold.max() * 100)
    a.set_xlabel("Threshold probability (%)")
    a.set_ylabel("Net benefit\n(true positives per 1,000)")
    a.yaxis.set_major_locator(MaxNLocator(5))
    despine(a, "y")
    _panel_title(a, "A", "Net benefit")
    for name in ["model", *[s for s in order if s != "model"][::-1]]:
        colour, line, label = styles[name]
        handles.append(Line2D([], [], color=colour, ls=line, lw=1.6, label=label))
    _key_legend(a, handles, y=-0.22, ncol=3)

    keys = []
    for name in [s for s in table.strategy.unique() if s.startswith("model_minus_")]:
        other = name.removeprefix("model_minus_")
        if other == "treat_all" or other not in styles:
            continue
        colour, _, label = styles[other]
        part = table[table.strategy == name]
        b.fill_between(part.threshold * 100, part.lower * per, part.upper * per,
                       color=colour, alpha=0.1, lw=0)
        b.plot(part.threshold * 100, part.net_benefit * per, color=colour, lw=1.6)
        keys.append(Line2D([], [], color=colour, lw=1.6, label=f"{model_label} vs {label}"))
    b.axhline(0, color=PALETTE["zero_line"], lw=0.8)
    b.set_xlim(table.threshold.min() * 100, table.threshold.max() * 100)
    b.set_xlabel("Threshold probability (%)")
    b.set_ylabel("Difference in net benefit\n(per 1,000; 95% CI)")
    b.yaxis.set_major_locator(MaxNLocator(5))
    despine(b, "y")
    _panel_title(b, "B", f"{model_label} against each")
    for label, threshold in (points or {}).items():
        x = threshold * 100
        if not table.threshold.min() * 100 <= x <= table.threshold.max() * 100:
            continue
        for ax in (a, b):
            ax.axvline(x, color=PALETTE["ink_muted"], lw=0.8, ls=(0, (1, 2)), zorder=0)
        a.text(x, 0.03, label + " ", transform=a.get_xaxis_transform(), rotation=90,
               va="bottom", ha="right", fontsize=6.3, color=PALETTE["ink_secondary"])
    _key_legend(b, keys, y=-0.22, ncol=2)
    watermark(fig, mark)
    return fig
