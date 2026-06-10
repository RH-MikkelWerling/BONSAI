"""
OPERA pairwise significance comparison runner.

Loads saved predictions.npz files from sweep output, runs DeLong's test
(AUROC) and paired bootstrap (other metrics) for all specified contrasts,
applies Benjamini-Hochberg FDR correction, and saves:
    - significance_results.csv     full results table
    - significance_table.txt       human-readable summary
    - significance_table.tex       LaTeX table for paper
    - delta_plot.png               annotated delta plot (Fig 5)

Usage
─────
python -m opera.run.compare \\
    --sweep_output_dir /results/opera_sweep \\
    --output_dir /results/comparisons \\
    [--metrics auroc,auprc,brier] \\
    [--n_bootstrap 2000] \\
    [--alpha 0.05]
"""

import argparse
from pathlib import Path
import pandas as pd

from opera.evaluation.significance import (
    run_pairwise_comparisons,
    format_significance_table,
    to_latex_significance_table,
    DEFAULT_CONTRASTS,
)


def main():
    parser = argparse.ArgumentParser(description="OPERA significance comparisons")
    parser.add_argument(
        "--sweep_output_dir", required=True, help="Root dir from sweep.py"
    )
    parser.add_argument("--output_dir", default="./results/comparisons")
    parser.add_argument(
        "--metrics", default="auroc", help="Comma-separated: auroc,auprc,brier"
    )
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR threshold")
    parser.add_argument(
        "--no_delong",
        action="store_true",
        help="Use bootstrap for AUROC too (slower, less powerful)",
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running pairwise comparisons...")
    print(f"  Contrasts: {len(DEFAULT_CONTRASTS)}")
    print(f"  Metrics:   {metrics}")
    print(
        f"  Method:    {'DeLong (AUROC) + bootstrap' if not args.no_delong else 'bootstrap only'}"
    )
    print(f"  FDR α:     {args.alpha}")
    print()

    df = run_pairwise_comparisons(
        sweep_output_dir=args.sweep_output_dir,
        metrics=metrics,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        use_delong_for_auroc=not args.no_delong,
    )

    if df.empty:
        print("No comparisons could be run. Check that predictions.npz files exist.")
        return

    # ── Save results ──────────────────────────────────────────────────
    df.to_csv(output_dir / "significance_results.csv", index=False)
    print(f"Full results: {output_dir / 'significance_results.csv'}")
    print(f"  {len(df)} tests, {df['significant'].sum()} significant after FDR")

    # Summary by contrast
    print("\n── Significance summary by contrast ──")
    for contrast, grp in df[df["metric"] == "auroc"].groupby("contrast_label"):
        n_sig = grp["significant"].sum()
        n_total = len(grp)
        mean_delta = grp["delta_mean"].mean()
        print(f"  {contrast}")
        print(f"    {n_sig}/{n_total} cells significant  |  mean Δ = {mean_delta:+.3f}")

    # Human-readable table
    txt = format_significance_table(df, metric="auroc")
    (output_dir / "significance_table.txt").write_text(txt)
    print(f"\nText table: {output_dir / 'significance_table.txt'}")

    # LaTeX table
    latex = to_latex_significance_table(df, metric="auroc")
    (output_dir / "significance_table.tex").write_text(latex)
    print(f"LaTeX table: {output_dir / 'significance_table.tex'}")

    # ── Annotated delta plot ──────────────────────────────────────────
    results_csv = Path(args.sweep_output_dir) / "results_table.csv"
    if results_csv.exists():
        try:
            from opera.visualization.analysis_plots import plot_delta_with_significance

            results_df = pd.read_csv(results_csv)
            plot_delta_with_significance(
                results_df,
                df,
                save_path=str(output_dir / "delta_plot_significant.png"),
            )
            print(f"Delta plot: {output_dir / 'delta_plot_significant.png'}")
        except Exception as e:
            print(f"Delta plot failed: {e}")

    print(f"\nDone. All outputs in {output_dir}")


if __name__ == "__main__":
    main()
