"""Synthetic patient-embedding visual QA. Never use these as study results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opera.run.plot_patient_embeddings import plot_colored_coordinates
from opera.visualization.embedding_plots import (
    plot_clinical_variables_panel,
    plot_embedding_disease_map,
    plot_embedding_multi_outcome,
    plot_embedding_stage_metadata_grid,
    plot_embedding_survival_gradient,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(47)
    for n in (4000, 30000):
        coords = rng.normal(size=(n, 2))
        coords[:, 1] += 0.4 * coords[:, 0]
        age = np.clip(62 + 12 * coords[:, 0] + rng.normal(0, 4, n), 18, 95)
        frame = pd.DataFrame({"component_1": coords[:, 0], "component_2": coords[:, 1], "age": age})
        plot_colored_coordinates(frame, color_by="age", method=f"synthetic_{n}", output_dir=args.output_dir)
        for count in (2, 6, 13, 24):
            frame[f"cohorts_{count}"] = [f"long_named_hematologic_cohort_{i % count + 1:02d}" for i in range(n)]
            plot_colored_coordinates(
                frame, color_by=f"cohorts_{count}", method=f"synthetic_{n}",
                output_dir=args.output_dir,
            )
        metadata = {
            "Age at diagnosis": age,
            "Sex": np.where(rng.random(n) < 0.5, "Female", "Male"),
            "Stage": rng.integers(1, 5, n),
            "IPI score": rng.integers(0, 6, n),
        }
        plot_embedding_stage_metadata_grid(
            {"local": coords, "joint": coords * [1, 0.7]}, metadata,
            title="SYNTHETIC layout preview: representation stages",
            save_path=str(args.output_dir / f"stage_metadata_{n}.png"),
        )
        plot_clinical_variables_panel(
            coords, metadata, title="SYNTHETIC clinical metadata preview",
            save_path=str(args.output_dir / f"clinical_panel_{n}.png"),
        )
        outcomes = {}
        for i in range(8):
            probability = 1 / (1 + np.exp(-(0.8 * coords[:, 0] - 2 + i * 0.3)))
            labels = (rng.random(n) < probability).astype(int)
            labels[rng.random(n) < 0.1] = -1
            outcomes[f"long_named_clinical_outcome_{i + 1}_grade_2plus"] = labels
        plot_embedding_multi_outcome(
            coords, outcomes, method="pca",
            save_path=str(args.output_dir / f"outcome_scatter_{n}.png"),
        )
        plot_embedding_survival_gradient(
            coords, rng.exponential(365, n), rng.choice([0, 1, 2], n), method="pca",
            title="SYNTHETIC observed follow-up preview",
            save_path=str(args.output_dir / f"followup_{n}.png"),
        )
        plot_embedding_disease_map(
            coords, np.array([f"Cohort {i % 6 + 1}" for i in range(n)]),
            title="SYNTHETIC disease-map preview",
            save_path=str(args.output_dir / f"disease_map_{n}.png"),
        )
        plt.close("all")


if __name__ == "__main__":
    main()
