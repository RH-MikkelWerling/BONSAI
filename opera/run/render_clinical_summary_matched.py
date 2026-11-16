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
    ("ipi_computed_native", "IPI (computed)", "all", "OS, 1 y"),
]
PANELS = [
    ("A", "Lowest group", "OPERA − index\n(percentage points)", "Equal group sizes"),
    ("B", "Highest group", "Index − OPERA\n(percentage points)", "Equal group sizes"),
    ("C", "Low-risk group", "Additional patients\n(% of patients)", "Matched observed risk"),
    ("D", "High-risk group", "Additional patients\n(% of patients)", "Matched observed risk"),
]


def panel_values(figures: Path, review: Path) -> dict[str, pd.DataFrame]:
    sm = pd.read_csv(figures / "size_matched_survival_comparisons.csv")
    sm = sm[sm.estimand.eq("km_survival")]
    matched = pd.read_csv(review / "matched_risk_and_auc.csv").set_index("score_id")
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
            out[key].append(tuple(100 * m[f"{side}_additional_n{s}"] / n for s in ("", "_lower", "_upper")))
    frames = {k: pd.DataFrame(v, columns=["point", "lower", "upper"]) for k, v in out.items()}
    frames["n"] = pd.Series([int(matched.loc[s].n_patients) for s, *_ in ROWS])
    return frames


def render(figures: Path, review: Path, out: Path) -> None:
    setup_style()
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
                ax.text(0.5, y, "Not estimable", transform=ax.get_yaxis_transform(), ha="center",
                        va="center", fontsize=7, color=MUTED)
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
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=10, pad=16)
        ax.annotate("OPERA better →", xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    labels = [f"{name}, {pop}, {end} (n={v['n'][i]:,})" for i, (_, name, pop, end) in enumerate(ROWS)]
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


def write_table(figures: Path, review: Path, out: Path) -> None:
    """The summary table, from the same inputs as the figure."""
    v = panel_values(figures, review)
    m = pd.read_csv(review / "matched_risk_and_auc.csv").set_index("score_id")
    rows = []
    for i, (score, name, pop, end) in enumerate(ROWS):
        r = m.loc[score]
        auc = f"{r.index_auc:.3f} / {r.model_auc:.3f}"
        gain = _ci(r.gain, r.lower, r.upper, 3)
        a = _ci(*v["A"].iloc[i])
        b = _ci(*v["B"].iloc[i])
        sides = []
        for side in ("low", "high"):
            if r[f"{side}_status"] != "estimable":
                sides.append("n.e.")
                continue
            add, lo, hi = r[f"{side}_additional_n"], r[f"{side}_additional_n_lower"], r[f"{side}_additional_n_upper"]
            ci = f" ({lo:,.0f} to {hi:,.0f})".replace("-", "$-$") if np.isfinite(lo) and np.isfinite(hi) else ""
            sides.append(f"{r[f'{side}_index_n']:,.0f} $\\to$ {r[f'{side}_model_n']:,.0f}; +{add:,.0f}{ci}")
        population = {"DLBCL": "DLBCL", "recorded": "index recorded", "all": "all with components"}[pop]
        rows.append(f"{name} & {population} & {end} & {int(r.n_patients):,} & {int(r.n_cases):,} & {auc} & {gain} "
                    f"& {a} & {b} & {sides[0]} & {sides[1]} \\\\")
    body = "\n".join(rows)
    out.write_text(rf"""% Written by opera.run.render_clinical_summary_matched from the held-out clinical outputs
% (scratch/R_EA_clinical_figures, scratch/R_EA_clinical_review_20261115); do not edit by hand.
\begin{{table}}[p]
\centering
\caption{{\textbf{{Adapted \opera{{}} against each clinical prognostic index on the held-out
patients.}} One row per comparison, on held-out patients with the index recorded
(computed IPI: all patients with its components; NCCN-IPI: DLBCL), at the endpoint
and horizon given (OS, overall survival; TF, treatment failure, next treatment or
death). AUC: index (ordinal category, or points for the computed IPI and
NCCN-IPI) and \opera{{}}, with the paired difference and 95\% patient-bootstrap
interval. Equal group sizes: \opera{{}}'s groups take the sizes of the index's
categories; (a) KM survival of \opera{{}}'s lowest group minus the index's lowest
category, (b) of the index's highest category minus \opera{{}}'s highest group, in
percentage points. Matched observed risk: the index's lowest (highest) category
against the largest \opera{{}} low-risk (high-risk) group with observed survival at
least as high (no higher), with the additional number of patients and its 95\%
interval; the two selections are separate and can overlap. Bold: interval above
zero. n.e., not estimable (insufficient follow-up in the index's categories, or
no supported match). Descriptive comparisons with unadjusted pointwise
intervals.}}
\label{{tab:clinical-index-summary}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.1}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lllrrlllllll@{{}}}}
\toprule
& & & & & \multicolumn{{2}}{{c}}{{Discrimination}} & \multicolumn{{2}}{{c}}{{Equal group sizes (pp)}}
  & \multicolumn{{2}}{{c}}{{Matched observed risk (patients)}} \\
\cmidrule(lr){{6-7}}\cmidrule(lr){{8-9}}\cmidrule(l){{10-11}}
Index & Population & Endpoint & Patients & Events & AUC, index / \opera{{}} & Difference
  & (a) Lowest group & (b) Highest group & Low risk & High risk \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")


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
