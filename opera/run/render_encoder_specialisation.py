"""OPERA's specialisation and its gap to pooled XGBoost, by encoder (held-out, seed 42).

A: OPERA minus the joint model on the same encoder (EXT_ARCH_L, EXT_ARCH_E, EXT_ARCH_EA).
B: OPERA minus pooled XGBoost (EXT_J_OA_OWN_L_VS_XP, EXT_R_E_VS_XP, EXT_R_EA_VS_XP).
Equal-cell AUC differences over all horizons with paired patient-bootstrap 95% intervals,
read from effect_summaries.csv.gz.

    python -m opera.run.render_encoder_specialisation \
        --bundle scratch/download_bundle_20261112/tables \
        --out paper/figures/results/fig_encoder_specialisation.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from opera.visualization.style import LABEL_SIZE, TICK_SIZE, setup_style  # noqa: E402

INDIGO, SIENNA, INK2, MUTED, GRID, ZERO = "#2D3A8C", "#B5451B", "#485463", "#657180", "#E8ECF1", "#737D8C"
ENCODERS = ["Cohort", "National", "Adapted"]
PANELS = [
    ("A", "OPERA's specialised parts", "OPERA minus the joint model",
     ["EXT_ARCH_L", "EXT_ARCH_E", "EXT_ARCH_EA"], INDIGO, "o", 3, 0.005),
    ("B", "The gap to pooled XGBoost", "OPERA minus pooled XGBoost",
     ["EXT_J_OA_OWN_L_VS_XP", "EXT_R_E_VS_XP", "EXT_R_EA_VS_XP"], SIENNA, "D", 2, 0.01),
]


def lighten(colour, amount):
    return tuple(1 - amount * (1 - c) for c in to_rgb(colour))


def rows(bundle: Path, contrasts: list[str]) -> pd.DataFrame:
    e = pd.read_csv(bundle / "effect_summaries.csv.gz", low_memory=False)
    e = e[(e.split.eq("held_out")) & e.seed.eq(42) & e.metric.eq("auc") & e.horizon_days.eq(0)
          & e.stratum.eq("all_cells")]
    out = e.set_index("contrast").loc[contrasts, ["gain", "lower", "upper"]]
    assert len(out) == 3, contrasts
    return out.reset_index()


def render(bundle: Path, out: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.10, right=0.975, top=0.76, bottom=0.17, wspace=0.32)
    for ax, (letter, title, sub, contrasts, col, mk, digits, step) in zip(axes, PANELS):
        r = rows(bundle, contrasts)
        x = list(range(3))
        ax.plot(x, r.gain, color=lighten(col, 0.55), lw=1.1, zorder=2)
        ax.errorbar(x, r.gain, yerr=[r.gain - r.lower, r.upper - r.gain], fmt="none",
                    ecolor=lighten(col, 0.5), elinewidth=2.0, capsize=2.6, capthick=1.0, zorder=2)
        ax.scatter(x, r.gain, color=col, marker=mk, s=38, zorder=3, edgecolor="white", linewidth=1.1)
        for i, g in enumerate(r.gain):
            ax.annotate(f"{g:+.3f}".replace("-", "−"), (i, g), xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")
        ax.axhline(0, color=ZERO, lw=0.8, zorder=1)
        ax.set_xticks(x, ENCODERS)
        ax.set_xlim(-0.35, 2.6)
        ax.set_xlabel("Encoder", fontsize=LABEL_SIZE)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _, k=digits: "0" if abs(v) < 1e-12 else f"{v:+.{k}f}".replace("-", "−")))
        ax.tick_params(axis="both", length=0, labelsize=TICK_SIZE)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.grid(axis="x", visible=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        pos = ax.get_position()
        fig.text(pos.x0 - 0.06, 0.97, f"{letter}  {title}", fontsize=10, weight="bold", va="top")
        fig.text(pos.x0 - 0.06 + 0.035, 0.89, sub, fontsize=8, color=MUTED, va="top", style="italic")
    axes[0].set_ylabel("Difference in AUC", fontsize=LABEL_SIZE)
    axes[1].set_ylim(-0.045, 0.006)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    render(a.bundle, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
