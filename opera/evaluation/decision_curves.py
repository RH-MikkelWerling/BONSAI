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
    "model": ("#2D3A8C", "-", "Model"),
    "index": ("#B5451B", "-", "Index"),
    "treat_all": ("#9B9B9B", "--", "Act for everyone"),
    "treat_none": ("#2B2B2B", ":", "Act for no one"),
}
COMPARATOR_COLOURS = ["#408579", "#8974AC", "#7189BC"]


def figure_decision_curve(table, *, index_label="IPI", model_label="OPERA",
                          labels=None, outcome_label="death", horizon_days=730):
    """A: net benefit of each strategy. B: the model minus the index and each comparator."""
    import matplotlib.pyplot as plt

    from opera.visualization.style import add_panel_label, despine, setup_style

    setup_style()
    labels = {"model": model_label, "index": index_label, **(labels or {})}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.09, 3.2))
    strategies = [s for s in table.strategy.unique() if not s.startswith("model_minus_")]
    extra = iter(COMPARATOR_COLOURS)
    colours = {}
    for name in strategies:
        colour, style, label = (STRATEGY_STYLE[name] if name in STRATEGY_STYLE
                                else (next(extra), "-", name))
        colours[name] = colour
        part = table[table.strategy == name]
        a.plot(part.threshold * 100, part.net_benefit, style, color=colour, lw=1.5,
               label=labels.get(name, label))
    model = table[table.strategy == "model"]
    a.fill_between(model.threshold * 100, model.lower, model.upper,
                   color=colours["model"], alpha=0.13, lw=0)
    top = table[table.strategy.isin(strategies)].net_benefit.max()
    a.set_ylim(-0.1 * top, 1.1 * top)
    a.set(xlabel="Threshold probability (%)", ylabel="Net benefit")
    a.legend(frameon=False, fontsize=7.5)
    for name in [s for s in table.strategy.unique() if s.startswith("model_minus_")]:
        other = name.removeprefix("model_minus_")
        if other == "treat_all":
            continue
        part = table[table.strategy == name]
        b.plot(part.threshold * 100, part.net_benefit * 1000, color=colours[other], lw=1.5,
               label=f"vs {labels.get(other, STRATEGY_STYLE.get(other, ('', '', other))[2])}")
        b.fill_between(part.threshold * 100, part.lower * 1000, part.upper * 1000,
                       color=colours[other], alpha=0.13, lw=0)
    b.axhline(0, color="#2B2B2B", lw=0.8)
    b.set(xlabel="Threshold probability (%)",
          ylabel="Difference in net benefit\n(net true positives per 1,000)")
    b.legend(frameon=False, fontsize=7.5)
    for ax, letter in ((a, "A"), (b, "B")):
        despine(ax)
        add_panel_label(ax, letter)
    years = horizon_days // 365
    fig.suptitle(f"Acting on predicted {outcome_label} within {years} years", fontsize=10)
    fig.tight_layout()
    return fig
