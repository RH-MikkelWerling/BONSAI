"""Generate clearly synthetic layout previews for dense and sparse figure panels.

These images are for visual QA only; they are not research results.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opera.run.plot_patient_embeddings import plot_outcome_probability_facets
from opera.visualization.outcome_forest import plot_outcome_performance_forest
from opera.visualization.disease_geometry_plots import plot_transfer_matrix
from opera.visualization.comparison_plots import plot_outcome_dot_matrix
from opera.visualization.rarity_plots import plot_rarity_delta
from opera.visualization.classification_plots import (
    plot_roc_comparison, plot_prc, plot_calibration, plot_decision_curve,
)


def _forest_tables(n_outcomes: int, n_models: int):
    rows, contrasts = [], []
    rng = np.random.default_rng(17)
    for i in range(n_outcomes):
        name = f"outcome_{i+1:02d}_grade_2plus" if i % 4 else f"longer_named_clinical_outcome_{i+1:02d}"
        family = f"Clinical domain {i // 8 + 1}"
        base = 0.58 + 0.22 * rng.random()
        events = int(rng.integers(8, 300))
        for j in range(n_models):
            estimate = min(0.98, base + 0.025 * j + rng.normal(0, 0.012))
            rows.append(dict(outcome=name, outcome_family=family, cohort="all hematology",
                             horizon_days=365, model=f"model_{j+1}", estimate=estimate,
                             lower=estimate-0.05, upper=min(1, estimate+0.05),
                             n=1200, n_events=events))
        contrasts.append(dict(outcome=name, outcome_family=family,
                              cohort="all hematology", horizon_days=365,
                              contrast="model_2_minus_model_1", comparator="model_2",
                              baseline="model_1", estimate=0.025,
                              lower=-0.02, upper=0.07,
                              probability_superior=0.84, n=1200, n_events=events))
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def _embedding_frame(n_outcomes: int, n: int = 900):
    rng = np.random.default_rng(9)
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    frame = pd.DataFrame({"component_1": x, "component_2": y})
    columns = []
    for i in range(n_outcomes):
        name = f"outcome_{i+1:02d}_within_365d"
        intercept = -2.6 + 1.1 * (i % 4)
        probability = 1 / (1 + np.exp(-(x * (0.4 + i / 10) - y * 0.25 + intercept)))
        frame[name] = rng.random(n) < probability
        columns.append(name)
    return frame, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for n_outcomes, n_models in ((3, 2), (18, 3), (35, 2)):
        estimates, contrasts = _forest_tables(n_outcomes, n_models)
        plot_outcome_performance_forest(
            estimates, contrasts,
            title="SYNTHETIC LAYOUT PREVIEW — not research results",
            save_path=args.output_dir / f"synthetic_forest_{n_outcomes:02d}.png",
        )
    for n_outcomes in (1, 4, 12, 13):
        frame, columns = _embedding_frame(n_outcomes)
        plot_outcome_probability_facets(
            frame, columns, method=f"synthetic_{n_outcomes:02d}",
            output_dir=args.output_dir, horizon_days=365,
            facets_per_page=12,
        )
    frame, columns = _embedding_frame(4, n=4000)
    for palette in ("plasma", "inferno"):
        plot_outcome_probability_facets(
            frame, columns, method=f"palette_{palette}",
            output_dir=args.output_dir, horizon_days=365, colormap=palette,
        )
    plt.close("all")
    rng = np.random.default_rng(2026)
    labels = rng.random(1500) < 0.12
    probability = 1 / (1 + np.exp(-(-2.8 + 1.5 * labels + rng.normal(size=len(labels)))))
    for model_count in (2, 6, 12):
        predictions = {
            f"model_{i+1}": (labels, np.clip(probability + rng.normal(0, 0.01 * i, len(labels)), 0, 1))
            for i in range(model_count)
        }
        plot_roc_comparison(
            predictions, title="Synthetic ROC layout preview",
            save_path=str(args.output_dir / f"synthetic_roc_{model_count:02d}.png"),
        )
        plt.close("all")
    for function, name in ((plot_prc, "prc"), (plot_calibration, "calibration"), (plot_decision_curve, "decision_curve")):
        function(labels, probability, save_path=str(args.output_dir / f"synthetic_{name}.png"))
        plt.close("all")
    for n_diseases in (4, 10, 24):
        rng = np.random.default_rng(n_diseases)
        clinical_names = [
            "diffuse_large_B_cell_lymphoma", "follicular_lymphoma",
            "acute_myeloid_leukemia", "multiple_myeloma",
            "chronic_lymphocytic_leukemia", "mantle_cell_lymphoma",
            "myelodysplastic_syndrome", "Hodgkin_lymphoma",
        ]
        names = [
            clinical_names[i] if i < len(clinical_names) else f"hematologic_disease_{i+1:02d}"
            for i in range(n_diseases)
        ]
        values = rng.uniform(0.52, 0.83, size=(n_diseases, n_diseases))
        np.fill_diagonal(values, 0.72)
        plot_transfer_matrix(
            pd.DataFrame(values, index=names, columns=names),
            save_path=str(args.output_dir / f"synthetic_transfer_{n_diseases:02d}.png"),
        )
    rng = np.random.default_rng(21)
    for n_tasks in (8, 40):
        task_rows = []
        for index in range(n_tasks):
            for model, offset in (("tabular_ehr", 0.0), ("opera", 0.025)):
                task_rows.append({
                    "cohort": f"disease_{index % 5 + 1:02d}",
                    "outcome": f"outcome_{index+1:02d}",
                    "model_family": model,
                    "auroc": float(0.62 + offset + rng.normal(0, 0.06)),
                    "n_events_train": 10 + index * 8,
                    "rarity_tier": "common" if index > n_tasks // 2 else "rare",
                })
        task_frame = pd.DataFrame(task_rows)
        plot_outcome_dot_matrix(
            task_frame,
            save_path=str(args.output_dir / f"synthetic_dot_matrix_{n_tasks:02d}.png"),
        )
        plot_rarity_delta(
            task_frame,
            save_path=str(args.output_dir / f"synthetic_rarity_delta_{n_tasks:02d}.png"),
        )


if __name__ == "__main__":
    main()
