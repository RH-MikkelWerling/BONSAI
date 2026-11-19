"""Render the subgroup and region figures (fig_who_benefits, supp_regions) from the
server's focused_core aggregates (recover lane, 19 Nov).

Inputs are aggregate CSVs only: stratified_*_effects.csv (equal-cell paired AUC
differences per subgroup), stratified_predominant_region_models.csv (per-model
metrics per region) and the per-model metrics of every stratification (for XGBoost
per task, whose paired contrast with adapted OPERA was not exported). Every estimate is pooled over horizons (horizon_days == 0).

Usage:
  python scripts/render_subgroup_figures.py \
      scratch/download_recover_20261119/ready/recover/focused_core \
      paper/figures/results_thesis_overlay
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from opera.visualization import paper_figures as pf
from opera.visualization.style import setup_style
from opera.run.paper_story_figures import thesis_style, S_COL, XP_COL, ADAPT, OPERA, XGB_REF

# Paired comparisons shown, in the colours of Figures 4 and 5.
CONTRASTS = {
    "FINAL_R_EA_VS_S_L": (S_COL, "OPERA vs specialists"),
    "FINAL_R_EA_VS_XP": (XP_COL, "OPERA vs pooled XGBoost"),
    "E3": (ADAPT, "Adapted vs cohort encoder"),
}
MODELS = {"R_EA": (OPERA, "Adapted OPERA"), "XP": (XP_COL, "Pooled XGBoost"),
          "X": (XGB_REF, "XGBoost per task")}
COVARIATES = {
    "age_at_index": ("Age at treatment, years",
                     {"(-0.001, 50.0]": "50 or younger", "(50.0, 65.0]": "51–65",
                      "(65.0, 80.0]": "66–80", "(80.0, inf]": "Over 80"}),
    "sex": ("Sex", {"F": "Female", "M": "Male"}),
    "IPI": ("Disease's own index, risk group",
            {"Low": "Low", "Intermediate": "Intermediate", "High": "High",
             "Very high": "Very high", "missing": "No index recorded"}),
    "history_event_count": ("Recorded events before treatment",
                            {"(-1.001, 99.0]": "Fewer than 100", "(99.0, 999.0]": "100–999",
                             "(999.0, 9999.0]": "1,000–9,999", "(9999.0, inf]": "10,000 or more"}),
    "index_year": ("Year of treatment", {str(y): str(y) for y in range(2022, 2027)}),
}


def _subgroups(frame, value_column):
    rows = []
    for covariate, (_, labels) in COVARIATES.items():
        part = frame.loc[frame.covariate == covariate].copy()
        part["key"] = part[value_column].astype(str).str.removesuffix(".0")
        part = part.loc[part.key.isin(labels)]
        part["order"] = part.key.map({k: i for i, k in enumerate(labels)})
        part["subgroup"] = part.key.map(labels)
        rows.append(part.sort_values("order"))
    return pd.concat(rows, ignore_index=True)


def load(directory: Path):
    effects = pd.concat([pd.read_csv(p, low_memory=False)
                         for p in sorted(directory.glob("stratified_*_effects.csv"))], ignore_index=True)
    effects = effects.loc[effects.split.eq("held_out") & effects.horizon_days.eq(0)
                          & effects.stratum.eq("all_cells")]
    models = pd.concat([pd.read_csv(p, low_memory=False)
                        for p in sorted(directory.glob("stratified_*_models.csv"))], ignore_index=True)
    models = models.loc[models.split.eq("held_out") & models.horizon_days.eq(0)]
    panels = pd.read_csv(directory / "patient_subgroup_panels.csv", low_memory=False)
    panels = panels.loc[panels.split.eq("held_out") & panels.horizon_days.eq(0)]
    return effects, models, panels


def _vs_per_task(models, metric):
    """Adapted OPERA minus XGBoost per task within each subgroup. Both are averaged over
    the same cells (equal_cell_common_panel), so this is the paired point estimate; no
    interval was exported for it."""
    part = models.loc[models.metric.eq(metric) & models.model.isin(["R_EA", "X"])]
    wide = part.pivot_table(index=["covariate", "stratum_value"], columns="model",
                            values="estimate").reset_index()
    wide["gain"] = wide["R_EA"] - wide["X"]
    return wide.assign(contrast="VS_X", lower=np.nan, upper=np.nan)


PAIRED_X = "FINAL_R_EA_VS_X"  # exported from the 19 Nov rerun onwards


def _with_per_task(effects, models, metric):
    """Paired contrasts, plus adapted OPERA vs XGBoost per task: the paired contrast with
    its interval where exported, else the point estimate from the per-model averages."""
    paired = effects.loc[effects.metric.eq(metric) & effects.contrast.isin([*CONTRASTS, PAIRED_X])]
    if paired.contrast.eq(PAIRED_X).any():
        return paired.assign(contrast=paired.contrast.replace({PAIRED_X: "VS_X"})), True
    return pd.concat([paired, _vs_per_task(models, metric)], ignore_index=True), False


def who_benefits(effects, models, out: Path):
    panels = []
    for metric in ("auc", "ipa"):
        both, has_interval = _with_per_task(effects, models, metric)
        panels.append(_subgroups(both, "stratum_value"))
    label = "OPERA vs XGBoost per task"  # without its interval until exported: see caption
    styles = {**CONTRASTS, "VS_X": (XGB_REF, label)}
    styles = {k: styles[k] for k in ("FINAL_R_EA_VS_S_L", "VS_X", "FINAL_R_EA_VS_XP", "E3")}
    auc, ipa = panels
    order = {c: t for c, (t, _) in COVARIATES.items()}
    rows = pd.concat([auc, ipa], ignore_index=True).drop_duplicates(["covariate", "subgroup"])
    rows = rows.assign(c=rows.covariate.map(list(COVARIATES).index)).sort_values(["c", "order"])
    height = 0.20 * len(rows) + 0.30 * len(order) + 1.5
    fig, axes = plt.subplots(1, 2, figsize=(6.4, height), sharey=True)
    for ax, data, letter, title, label in zip(
            axes, [auc, ipa], "AB", ["Discrimination", "Accuracy of predicted risk"],
            ["Δ time-dependent AUC ×100 (95% CI)", "Δ IPA ×100 (95% CI)"]):
        pf.figure_subgroup_effects(data, contrast_styles=styles, covariate_order=order,
                                   xlabel=label, title_letters=letter, panel_title=title,
                                   ax=ax, row_order=rows, show_legend=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=9, fontweight="bold", pad=6)
    # One interval (69 patients with fewer than 100 recorded events) runs to about +65;
    # cap the IPA axis so the other groups stay readable (stated in the caption).
    axes[1].set_xlim(-15, 35)
    axes[1].tick_params(axis="y", labelleft=False)
    for text in list(axes[1].texts):  # covariate headers belong to the shared y-axis only
        text.remove()
    left, right, top = 0.33, 0.98, 1 - 0.95 / height
    fig.subplots_adjust(left=left, right=right, wspace=0.08, top=top, bottom=0.6 / height)
    # Centred over the two panels, just above their titles.
    fig.legend(handles=[Line2D([], [], ls="none", marker="D", color=c, label=l)
                        for c, l in styles.values()],
               loc="lower center", bbox_to_anchor=((left + right) / 2, top + 0.30 / height),
               ncol=2, frameon=False, fontsize=7.2, columnspacing=1.2, handletextpad=0.2,
               borderaxespad=0)
    _save(fig, out / "fig_who_benefits")


# English names for the five regions (the geometry and tiles are keyed by the Danish ones).
ENGLISH = {"Nordjylland": "North Denmark", "Midtjylland": "Central Denmark",
           "Syddanmark": "Southern Denmark", "Hovedstaden": "Capital Region",
           "Sjælland": "Zealand"}


def _english(ax, on_map=False):
    from matplotlib import patheffects
    for text in ax.texts:
        if text.get_text() in ENGLISH:
            text.set_text(ENGLISH[text.get_text()])
    if on_map:
        # Dark ink with a thin white halo reads on every fill and on the sea, where the
        # longer English names now reach; the default white-on-dark text vanished there.
        halo = [patheffects.withStroke(linewidth=1.8, foreground="white")]
        for text in ax.texts:
            if text.get_text() == "Bornholm":
                continue
            text.set_color("#17212B")
            text.set_path_effects(halo)
            text.set_fontsize(7.0 if text.get_text() in ENGLISH.values() else 7.4)
            x, y = text.get_position()
            if 11.4 < x < 11.6 and y > 56.5:  # Capital Region: closer to its leader line
                text.set_position((x - 0.12, y - 0.17))
    labels = [t.get_text() for t in ax.get_yticklabels()]
    if any(l in ENGLISH for l in labels):
        ax.set_yticks(ax.get_yticks(), [ENGLISH.get(l, l) for l in labels])


def regions(effects, models, out: Path):
    geometry = json.loads((Path(pf.__file__).resolve().parents[1] / "resources"
                           / "denmark_regions.geojson").read_text())
    eff = effects.loc[effects.covariate.eq("predominant_region")].copy()
    eff["region"] = eff.stratum_value.map(pf._region_name)
    mod = models.loc[models.covariate.eq("predominant_region")].rename(columns={"estimate": "gain"})
    mod["region"] = mod.stratum_value.map(pf._region_name)
    auc, has_interval = _with_per_task(eff, models.loc[models.covariate.eq("predominant_region")],
                                       "auc")
    auc["region"] = auc.stratum_value.map(pf._region_name)
    shown = {"FINAL_R_EA_VS_S_L": CONTRASTS["FINAL_R_EA_VS_S_L"],
             "VS_X": (XGB_REF, "OPERA vs XGBoost per task"),  # interval: see caption
             "FINAL_R_EA_VS_XP": CONTRASTS["FINAL_R_EA_VS_XP"], "E3": CONTRASTS["E3"]}
    fig = plt.figure(figsize=(6.6, 7.0))
    maps = fig.add_gridspec(1, 3, wspace=0.06, left=0.01, right=0.99, top=0.95, bottom=0.64)

    def map_panel(i, letter, title, rows, cmap, norm, label, fmt, reference=None):
        ax = fig.add_subplot(maps[i])
        values = rows.set_index("region").gain.to_dict()
        pf._draw_denmark(ax, values, geometry, norm, cmap, fmt=fmt, reference=reference)
        _english(ax, on_map=True)
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=9, fontweight="bold", pad=4)
        box = ax.get_position()
        bar = fig.add_axes([box.x0 + 0.03, box.y0 - 0.03, box.width - 0.06, 0.010])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar,
                          orientation="horizontal")
        cb.ax.tick_params(labelsize=6.5, length=0)
        cb.ax.xaxis.set_major_locator(MaxNLocator(3))
        cb.outline.set_visible(False)
        cb.set_label(label, fontsize=7)

    # A: adapted OPERA minus XGBoost per task. Both are averaged over the same cells in
    # each region (equal_cell_common_panel), so the difference of the two is the paired
    # point estimate; its interval was not exported (no R_EA-vs-X contrast on the server).
    if has_interval:
        vs_x = auc.loc[auc.contrast.eq("VS_X")]
    else:
        auc_models = mod.loc[mod.metric.eq("auc")].pivot_table(index="region", columns="model",
                                                               values="gain")
        vs_x = (auc_models["R_EA"] - auc_models["X"]).rename("gain").reset_index()
    span = float(np.nanmax(np.abs(vs_x.gain))) or 0.01
    map_panel(0, "A", "Gain over XGBoost per task", vs_x, pf.GAIN_CMAP,
              TwoSlopeNorm(0, -span, span), "Δ time-dependent AUC", "{:+.3f}", 0.0)
    own = mod.loc[mod.metric.eq("auc") & mod.model.eq("R_EA")]
    low, high = own.gain.min(), own.gain.max()
    pad = max((high - low) * 0.15, 0.005)
    map_panel(1, "B", "AUC", own, pf.ATLAS_CMAP, plt.Normalize(low - pad, high + pad),
              "Time-dependent AUC", "{:.3f}")
    bias = mod.loc[mod.metric.eq("calibration_bias") & mod.model.eq("R_EA")]
    span = float(np.nanmax(np.abs(bias.gain))) or 0.01
    map_panel(2, "C", "Calibration in the large", bias, pf.BIAS_CMAP,
              TwoSlopeNorm(0, -span, span), "Predicted − observed risk", "{:+.3f}", 0.0)

    lower = fig.add_gridspec(1, 3, wspace=0.12, left=0.17, right=0.98, top=0.51, bottom=0.17)
    panels = [
        ("D", "Gains", auc, [(k, "contrast", v) for k, v in shown.items()], 0.0,
         "Δ time-dependent AUC (95% CI)"),
        ("E", "AUC", mod.loc[mod.metric.eq("auc")], [(k, "model", v) for k, v in MODELS.items()],
         None, "Time-dependent AUC (95% CI)"),
        ("F", "Calibration", mod.loc[mod.metric.eq("calibration_bias")],
         [(k, "model", v) for k, v in MODELS.items()], 0.0, "Predicted − observed (95% CI)"),
    ]
    for i, (letter, title, table, series, ref, xlabel) in enumerate(panels):
        ax = fig.add_subplot(lower[i])
        pf._region_forest(ax, table, series, reference=ref)
        _english(ax)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel(xlabel, fontsize=7.5)
        if i:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=9, fontweight="bold", pad=6)
        ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(-0.02, -0.16),
                  ncol=1, handletextpad=0.1)
    _save(fig, out / "supp_regions")


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {stem}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("focused_core", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    setup_style()
    thesis_style()
    effects, models, panels = load(a.focused_core)
    who_benefits(effects, models, a.out)
    regions(effects, models, a.out)


if __name__ == "__main__":
    main()
