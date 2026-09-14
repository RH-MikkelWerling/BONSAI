"""Export paginated training diagnostics from the actual Lightning CSV log."""

from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd

from opera.visualization.embedding_plots import plot_training_curves


def export_training_report(logger, output_dir, *, is_global_zero=True):
    """Run only on rank zero; an export failure must not invalidate training."""
    if not is_global_zero:
        return []
    paths = []
    try:
        logger.save()
        source = Path(logger.log_dir) / "metrics.csv"
        if not source.exists():
            return paths
        frame = pd.read_csv(source)
        metrics = sorted(
            {
                column.split("/", 1)[1]
                for column in frame
                if column.startswith(("train/", "val/")) and frame[column].notna().any()
            }
        )
        for page, start in enumerate(range(0, len(metrics), 6), 1):
            path = Path(output_dir) / "plots" / f"training_diagnostics_{page:02d}.png"
            fig = plot_training_curves(
                frame,
                metrics=metrics[start : start + 6],
                smooth_window=1,
                save_path=str(path),
            )
            plt.close(fig)
            paths.append(path)
    except (OSError, ValueError, KeyError) as error:
        logging.getLogger(__name__).warning("Training figure export failed: %s", error)
    return paths
