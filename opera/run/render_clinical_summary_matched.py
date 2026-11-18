"""Supplementary clinical-index summary on the held-out patients, without frozen cutoffs.

Replaces panels C--D of ``figure_clinical_index_summary`` (training-frozen equal-risk
cutoffs, which did not keep observed risk matched on held-out patients) with the
evaluation-cohort matching of ``MATCHED_RISK_REVIEW.md``. Reads existing outputs only:

  A--B  size_matched_survival_comparisons.csv   (equal group sizes; lowest and
        highest group, difference in KM survival at the horizon)
  C--D  matched_risk_and_auc.csv                (largest model low-/high-risk group at
        observed risk at least as favourable / unfavourable as the index's
        extreme category; additional patients, % of patients)

    python -m opera.run.render_clinical_summary_matched \
        --figures scratch/R_EA_clinical_figures \
        --review scratch/R_EA_clinical_review_20261115 \
        --out paper/figures/results/supp_clinical_index_summary.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from opera.visualization.style import LABEL_SIZE, PALETTE, TICK_SIZE, setup_style  # noqa: E402

INDIGO, SIENNA = "#2D3A8C", "#B5451B"
MUTED, GRID, ZERO = "#657180", "#E8ECF1", "#737D8C"

# Display order (disease-specific first, pooled populations last) and row labels.
ROWS = [
    ("nccn_ipi_native", "NCCN-IPI", "DLBCL", "OS, 1 y"),
    ("nccn_ipi_treatment_failure_2y_native", "NCCN-IPI", "DLBCL", "TF, 2 y"),
    ("cll_ipi_all_subtypes_native", "CLL-IPI", "recorded", "OS, 1 y"),
    ("r_iss_all_subtypes_native", "R-ISS", "recorded", "OS, 1 y"),
    ("mipi_all_subtypes_native", "MIPI", "recorded", "OS, 1 y"),
    ("ipsswm_all_subtypes_native", "IPSS-WM", "recorded", "OS, 1 y"),
    ("ips_all_subtypes_native", "IPS", "recorded", "TF, 1 y"),
    ("r_ipi_all_subtypes_native", "R-IPI", "recorded", "OS, 1 y"),
    ("flipi2_all_subtypes_native", "FLIPI2", "recorded", "TF, 1 y"),
    ("malt_ipi_all_subtypes_native", "MALT-IPI", "recorded", "OS, 1 y"),
    ("ipi_computed_native", "IPI", "all", "OS, 1 y"),
]
PANELS = [
    ("A", "Lowest group", "OPERA − index\n(percentage points)", "Equal group sizes"),
    ("B", "Highest group", "Index − OPERA\n(percentage points)", "Equal group sizes"),
    ("C", "Low-risk group", "Additional patients\n(% of patients)", "Matched observed risk"),
    ("D", "High-risk group", "Additional patients\n(% of patients)", "Matched observed risk"),
]


def endpoint_text(row):
    """'OS, 1 y' / 'TF, 2 y' from the results, so a changed endpoint is labelled correctly."""
    return f"{'OS' if row.outcome == 'overall_survival' else 'TF'}, {int(round(row.horizon_days / 365))} y"


def panel_values(figures: Path, review: Path) -> dict[str, pd.DataFrame]:
    sm = pd.read_csv(figures / "size_matched_survival_comparisons.csv")
    sm = sm[sm.estimand.eq("km_survival")]
    matched = pd.read_csv(review / "matched_risk_and_auc.csv").set_index("score_id")
    from opera.run.paper_story_figures import _population_survival, degenerate
    population = _population_survival(figures)
    out = {k: [] for k in "ABCD"}
    for score, *_ in ROWS:
        g = sm[sm.score_id.eq(score)]
        # groups are listed lowest to highest risk
        lo, hi = (g.iloc[0], g.iloc[-1]) if len(g) else (None, None)
        # KM survival differences, in percentage points
        out["A"].append(tuple(100 * x for x in (lo.difference, lo.lower, lo.upper)) if lo is not None
                        else (np.nan,) * 3)
        out["B"].append(tuple(100 * x for x in (-hi.difference, -hi.upper, -hi.lower)) if hi is not None
                        else (np.nan,) * 3)
        m = matched.loc[score]
        n = m.n_patients
        for key, side in (("C", "low"), ("D", "high")):
            if m[f"{side}_status"] != "estimable":
                out[key].append((np.nan,) * 3)
                continue
            if degenerate(m, side, population):
                out[key].append((np.inf, np.nan, np.nan))  # drawn as "index finds no such group"
                continue
            out[key].append(tuple(100 * m[f"{side}_additional_n{s}"] / n for s in ("", "_lower", "_upper")))
    frames = {k: pd.DataFrame(v, columns=["point", "lower", "upper"]) for k, v in out.items()}
    frames["n"] = pd.Series([int(matched.loc[s].n_patients) for s, *_ in ROWS])
    return frames


def render(figures: Path, review: Path, out: Path) -> None:
    setup_style()
    from opera.run.paper_story_figures import thesis_style
    thesis_style()
    v = panel_values(figures, review)
    n = len(ROWS)
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 0.30 * n + 2.0), sharey=True,
                             gridspec_kw=dict(wspace=0.22, left=0.25, right=0.985, top=0.78, bottom=0.17))
    for ax, (letter, title, xlabel, group) in zip(axes, PANELS):
        f = v[letter]
        finite = f.to_numpy(float)[np.isfinite(f.to_numpy(float))]
        lo, hi = min(0.0, finite.min()), max(0.0, finite.max())
        span = hi - lo
        ax.set_xlim(lo - 0.08 * span, hi + 0.08 * span)
        ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
        for y, row in f.iterrows():
            if not np.isfinite(row.point):
                ax.text(0.5, y, "Index finds no such group" if row.point == np.inf else "Not estimable",
                        transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=7, color=MUTED)
                continue
            has_ci = np.isfinite(row.lower) and np.isfinite(row.upper)
            colour = SIENNA if has_ci and row.upper < 0 else INDIGO
            if has_ci:
                ax.plot([row.lower, row.upper], [y, y], color=colour, lw=1.3, alpha=0.45, zorder=2,
                        solid_capstyle="round")
            filled = has_ci and (row.lower > 0 or row.upper < 0)
            ax.scatter(row.point, y, marker="D", s=24, zorder=3, linewidths=1, edgecolors=colour,
                       facecolors=colour if filled else "white")
        ax.axhline(1.5, color=GRID, lw=0.6, zorder=0)  # disease-specific rows above
        ax.axhline(n - 1.5, color=GRID, lw=0.6, zorder=0)  # pooled IPI below
        ax.set_ylim(n - 0.5, -0.7)
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=9.6, fontweight="bold", pad=16)
        ax.annotate("OPERA better →", xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    matched = pd.read_csv(review / "matched_risk_and_auc.csv").set_index("score_id")
    labels = [f"{name}, {endpoint_text(matched.loc[sid])} (n={v['n'][i]:,})" for i, (sid, name, pop, end) in enumerate(ROWS)]
    axes[0].set_yticks(range(n), labels, fontsize=8)
    for (x0, x1), text in (((axes[0], axes[1]), "Equal group sizes"), ((axes[2], axes[3]), "Matched observed risk")):
        left, right = x0.get_position().x0, x1.get_position().x1
        fig.text((left + right) / 2, 0.925, text, ha="center", va="center", fontsize=10, color=PALETTE["ink"]
                 if "ink" in PALETTE else "#17212B")
        fig.add_artist(plt.Line2D([left, right], [0.90, 0.90], transform=fig.transFigure, color="#CBD2DA", lw=0.8))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _num(x, digits=1, signed=True):
    if not np.isfinite(x):
        return "n.e."
    text = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return text.replace("-", "$-$")


def _ci(point, lo, hi, digits=1):
    if not np.isfinite(point):
        return "n.e."
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{_num(point, digits)} (no interval)"
    text = f"{_num(point, digits)} ({_num(lo, digits, False)} to {_num(hi, digits, False)})"
    return rf"\textbf{{{text}}}" if lo > 0 else text


def _auc(point, lo, hi):
    """AUC with its own 95% interval, on two lines to keep the column narrow."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{point:.3f}"
    return rf"\shortstack{{{point:.3f}\\[-1pt]{{\scriptsize ({lo:.3f}--{hi:.3f})}}}}"


def _wrap_table(label, caption, colspec, header, rows, note=""):
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
{note}\end{{table}}
"""


def write_table(figures: Path, review: Path, out: Path) -> None:
    """The summary tables (discrimination, equal group sizes, matched observed risk),
    from the same inputs as the figure."""
    v = panel_values(figures, review)
    m = pd.read_csv(review / "matched_risk_and_auc.csv").set_index("score_id")
    from opera.run.paper_story_figures import _population_survival, degenerate
    population = _population_survival(figures)
    disc, sizes, matched = [], [], []
    for i, (score, name, pop, end) in enumerate(ROWS):
        r = m.loc[score]
        end = endpoint_text(r)
        lead = f"{name} & {end}"
        if i == len(ROWS) - 1:
            disc.append(r"\addlinespace")
            sizes.append(r"\addlinespace")
            matched.append(r"\addlinespace")
        disc.append(f"{name} & {end} & {int(r.n_patients):,} & {int(r.n_cases):,} "
                    f"& {_auc(r.index_auc, r.index_lower, r.index_upper)} & {_auc(r.model_auc, r.model_lower, r.model_upper)} "
                    f"& {_ci(r.gain, r.lower, r.upper, 3)} \\\\")
        sizes.append(f"{lead} & {_ci(*v['A'].iloc[i])} & {_ci(*v['B'].iloc[i])} \\\\")
        cells = []
        for side in ("low", "high"):
            if r[f"{side}_status"] != "estimable":
                cells += [r"\multicolumn{3}{c}{n.e.}"]
                continue
            if degenerate(r, side, population):
                cells += [r"\multicolumn{3}{c}{index finds no %s-risk group}" % side]
                continue
            add, lo, hi = r[f"{side}_additional_n"], r[f"{side}_additional_n_lower"], r[f"{side}_additional_n_upper"]
            ci = (f"+{add:,.0f} ({lo:,.0f} to {hi:,.0f})".replace("-", "$-$")
                  if np.isfinite(lo) and np.isfinite(hi) else f"+{add:,.0f}")
            if np.isfinite(lo) and lo > 0:
                ci = rf"\textbf{{{ci}}}"
            cells += [f"{r[f'{side}_index_n']:,.0f}", f"{r[f'{side}_model_n']:,.0f}", ci]
        matched.append(f"{lead} & " + " & ".join(cells) + r" \\")

    shared = (r"OS, overall survival; TF, treatment failure (second-line treatment or death). "
              r"Intervals are paired patient-bootstrap 95\% intervals; bold intervals exclude zero. "
              r"Each index is evaluated in the held-out patients with it recorded (the IPI: "
              r"with its components).")
    t1 = _wrap_table(
        "tab:clinical-index-summary",
        r"\textbf{Discrimination of adapted \opera{} and each clinical prognostic index on the "
        r"held-out patients.} AUC of the index (its ordinal categories, or its points for the "
        r"IPI and NCCN-IPI) and of \opera{} on the same patients, at the endpoint and "
        r"horizon given, each with its 95\% interval, and the paired difference. Separate AUC intervals can "
        r"overlap even where the paired difference excludes zero; the paired difference is the comparison. " + shared,
        "@{}llrrccl@{}",
        r"Index & Endpoint & Patients & Events & \multicolumn{2}{c}{AUC} & Difference (95\% CI) \\"
        "\n" r"\cmidrule(lr){5-6}" "\n"
        r" & & & & Index & \opera{} & \\",
        disc)
    t2 = _wrap_table(
        "tab:clinical-index-sizes",
        r"\textbf{Survival in the extreme groups at equal group sizes.} \opera{}'s groups take "
        r"the sizes of the index's categories. Lowest group: Kaplan--Meier survival at the horizon "
        r"of \opera{}'s lowest group minus that of the index's lowest category. Highest group: the "
        r"index's highest category minus \opera{}'s highest group. Both in percentage points, so "
        r"positive values favour \opera{}. " + shared,
        "@{}llll@{}",
        r"Index & Endpoint & Lowest group (pp) & Highest group (pp) \\",
        sizes)
    t3 = _wrap_table(
        "tab:clinical-index-matched",
        r"\textbf{Patients identified at matched observed risk.} For the index's lowest (highest) "
        r"category, the largest \opera{} low-risk (high-risk) group with observed survival at least "
        r"as high (no higher): patients in the index category, patients in the \opera{} group, and "
        r"the additional patients. The low- and high-risk selections are separate and can overlap. "
        r"These operating points use held-out outcomes and describe achievable ranking, not "
        r"validated thresholds. Index finds no low- (high-) risk group: the index's category was no safer "
        r"(riskier) than all patients with the index, so any group, up to all patients, would match it. "
        r"n.e., not estimable (insufficient follow-up in the index's categories, or no supported "
        r"match). " + shared,
        "@{}llrrlrrl@{}",
        r"& & \multicolumn{3}{c}{Low risk} & \multicolumn{3}{c}{High risk} \\"
        "\n" r"\cmidrule(lr){3-5}\cmidrule(l){6-8}" "\n"
        r"Index & Endpoint & Index & \opera{} & Additional & Index & \opera{} & Additional \\",
        matched)
    t4 = tails_table(figures, m, shared.replace(r"Intervals are paired patient-bootstrap 95\% intervals; bold intervals exclude zero. ", ""))
    out.write_text("% Written by opera.run.render_clinical_summary_matched from the held-out clinical outputs\n"
                   "% (scratch/R_EA_clinical_figures, scratch/R_EA_clinical_review_20261115); do not edit by hand.\n"
                   + t1 + "\n" + t2 + "\n" + t3 + "\n" + t4)


def tails_table(figures: Path, matched: pd.DataFrame, shared: str) -> str:
    """Observed risk in the index's extreme categories, in OPERA's groups of the same sizes, and in
    OPERA's top 5% and 10%: where the model's ranking shows beyond the index's coarse categories."""
    d = pd.read_csv(figures / "stratification_diagnostics.csv")
    pct = lambda x: "--" if not np.isfinite(x) else f"{100 * x:.0f}"
    rows = []
    for i, (score, name, pop, end) in enumerate(ROWS):
        g = d[d.score_id.eq(score)]
        if g.empty:
            continue
        idx, mod = g[g.system.eq("index")], g[g.system.eq("model")]
        named = mod[~mod.group.str.startswith("top")]
        lo_i, hi_i = idx.iloc[0], idx.iloc[-1]
        lo_m, hi_m = named.iloc[0], named.iloc[-1]
        top = {k: mod.loc[mod.group.eq(k), "observed_risk"] for k in ("top 10%", "top 5%")}
        top = {k: (v.iloc[0] if len(v) else np.nan) for k, v in top.items()}
        if i == len(ROWS) - 1:
            rows.append(r"\addlinespace")
        rows.append(f"{name} & {endpoint_text(matched.loc[score])} & {pct(lo_i.share)} & {pct(lo_i.observed_risk)} & "
                    f"{pct(lo_m.observed_risk)} & {pct(hi_i.share)} & {pct(hi_i.observed_risk)} & "
                    f"{pct(hi_m.observed_risk)} & {pct(top['top 10%'])} & {pct(top['top 5%'])} \\\\")
    return _wrap_table(
        "tab:clinical-index-tails",
        r"\textbf{Observed risk at the extremes.} For each index, the share of patients in its lowest and "
        r"highest categories and their observed risk at the horizon, the observed risk in adapted \opera{}'s "
        r"groups of the same sizes, and in \opera{}'s 10\% and 5\% of patients at highest predicted risk. "
        r"Observed risk is the proportion with the event among patients with known status at the horizon, "
        r"in per cent (not Kaplan--Meier, so values can differ from \\cref{tab:clinical-index-sizes}). Equal-size groups inherit the index's categories, which are often wide, so they show "
        r"only part of the model's ranking; the top 10\% and 5\% show where the rest of it lies. " + shared,
        "@{}llrrrrrrrr@{}",
        r"& & \multicolumn{3}{c}{Lowest category} & \multicolumn{3}{c}{Highest category} & \multicolumn{2}{c}{\opera{} top} \\"
        "\n" r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}" "\n"
        r"Index & Endpoint & \% pts & Index & \opera{} & \% pts & Index & \opera{} & 10\% & 5\% \\",
        rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--table", type=Path, help="also write the summary table here")
    a = p.parse_args()
    render(a.figures, a.review, a.out)
    print(f"wrote {a.out}")
    if a.table:
        write_table(a.figures, a.review, a.table)
        print(f"wrote {a.table}")


if __name__ == "__main__":
    main()
