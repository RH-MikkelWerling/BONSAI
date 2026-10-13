"""Render deterministic synthetic design proofs: python -m opera.run.render_style_gallery.

These figures demonstrate styling only and must not be used as study results.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from opera.visualization.style import save_fig  # noqa: E402


def synthetic_calibration_cohort(
    seed: int,
    *,
    n: int = 900,
    n_dims: int = 6,
    horizon: float = 365.0,
    spread: float = 0.7,
    blob_center: tuple[float, float] = (2.2, 0.0),
    blob_radius: float = 2.0,
    blob_shift: float = 0.0,
    global_shift: float = 0.0,
):
    """A held-out cohort whose true risk and predicted risk are both known.

    Panel D of the censoring-aware map asks whether the model's predicted
    cause-specific CIF matches the observed one region by region, so a design
    proof needs predictions with a *known* error: ``blob_shift`` inflates
    (or deflates) the prediction inside one ball of the embedding centred on
    ``blob_center`` and ``global_shift`` moves every prediction. Setting both
    to zero makes the model exactly calibrated, which is what panel D should
    then show.

    Everything is drawn as if the model were trained through 2022 and scored on
    2023+ patients, so every row is held out.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, n)
    angle = group * 2 * np.pi / 3
    embedding = rng.normal(0, spread, (n, n_dims))
    embedding[:, 0] += 2.2 * np.cos(angle)
    embedding[:, 1] += 2.2 * np.sin(angle)
    coords = embedding[:, :2].copy()  # the layout is an honest projection

    # True 365-day cause-specific risk varies smoothly over the embedding, and
    # leans on a dimension the two-dimensional layout does not show.
    signal = 0.9 * embedding[:, 0] - 0.6 * embedding[:, 1] + 0.5 * embedding[:, 2]
    true_risk = 0.06 + 0.62 / (1 + np.exp(-signal))
    competing_risk = 0.10 + 0.06 / (1 + np.exp(-embedding[:, 3]))

    draw = rng.random(n)
    primary = draw < true_risk
    competing = (draw >= true_risk) & (draw < true_risk + competing_risk)
    within = primary | competing
    event_time = np.where(
        within,
        rng.uniform(10.0, horizon, n),
        rng.uniform(1.05 * horizon, 1500.0, n),
    )
    late_cause = rng.choice([1, 2], n, p=[0.75, 0.25])
    cause = np.where(primary, 1, np.where(competing, 2, late_cause))
    censor_time = rng.uniform(30.0, 1400.0, n)
    censored = censor_time < event_time
    times = np.where(censored, censor_time, event_time)
    events = np.where(censored, 0, cause)

    centre = np.zeros(n_dims)
    centre[:2] = blob_center
    distance = np.linalg.norm(embedding - centre, axis=1)
    blob = 1.0 / (1.0 + np.exp((distance - blob_radius) / 0.35))
    blob[blob < 0.01] = 0.0  # a bounded region, not a tail that reaches everywhere
    predicted_cif = np.clip(true_risk + global_shift + blob_shift * blob, 0.0, 1.0)
    index_year = rng.integers(2023, 2026, n)
    return {
        "coords": coords,
        "embedding": embedding,
        "times": times,
        "events": events,
        "true_risk": true_risk,
        "predicted_cif": predicted_cif,
        "blob": blob,
        "index_year": index_year,
        "held_out": index_year >= 2023,
    }


def main():
    from opera.visualization.classification_plots import (
        plot_calibration,
        plot_evaluation_panel,
        plot_roc_comparison,
    )
    from opera.visualization.comparison_plots import plot_forest
    from opera.visualization.analysis_plots import plot_transfer_matrix
    from opera.visualization.embedding_plots import (
        plot_embedding_stage_metadata_grid,
        plot_training_curves,
    )
    from opera.visualization.survival_plots import plot_timedep_auc_curves
    from opera.visualization.censoring_aware_embeddings import (
        plot_censoring_aware_embedding_panel,
    )
    from opera.visualization.rarity_plots import plot_synthetic_rarity_delta
    from opera.visualization.vocabulary_atlas import plot_vocabulary_embedding_atlas

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/style_gallery")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    n = 600
    latent = rng.normal(size=n)
    probability = 1 / (1 + np.exp(1.2 - latent))
    event_time = rng.exponential(650 / np.exp(0.7 * latent))
    censor_time = rng.uniform(400, 1000, n)
    times = np.minimum(event_time, censor_time)
    events = (event_time <= censor_time).astype(int)
    labels = ((events == 1) & (times <= 365)).astype(int)
    models = ("opera", "dapt", "tabular_rkkp", "tabular_ehr", "ipi", "pretrain")
    results = {
        model: (
            labels,
            1 / (1 + np.exp(1.2 - latent - rng.normal(0, 0.2 + i * 0.22, n))),
        )
        for i, model in enumerate(models)
    }
    cards = []

    def export(name, fig):
        # Stamp every artifact, including PDFs, to prevent confusion with results.
        fig.text(
            0.995,
            0.005,
            "SYNTHETIC DESIGN PROOF",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#657180",
        )
        save_fig(fig, args.output_dir / f"{name}.png")
        save_fig(fig, args.output_dir / f"{name}.svg")
        plt.close(fig)
        title = html.escape(name.replace("_", " ").title())
        cards.append(
            f'<article><h2>{title}</h2><a href="{name}.pdf"><img src="{name}.png" alt="{title}"></a>'
            f'<p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p></article>'
        )

    export(
        "model_discrimination",
        plot_roc_comparison(results, title="Discrimination across models"),
    )
    export("calibration", plot_calibration(labels, probability))
    export(
        "evaluation",
        plot_evaluation_panel(
            labels,
            probability,
            times=times,
            events=events,
            window_days=365,
            outcome_name="Treatment failure",
        ),
    )
    forest = {
        model: {
            "auroc": {
                "mean": 0.82 - i * 0.025,
                "lower": 0.79 - i * 0.025,
                "upper": 0.85 - i * 0.025,
            }
        }
        for i, model in enumerate(models)
    }
    export("model_estimates", plot_forest(forest, title="Held-out discrimination"))
    names = ["Treatment failure", "Infection", "Renal event", "Cardiovascular event"]
    matrix = pd.DataFrame(rng.uniform(0.55, 0.85, (4, 4)), index=names, columns=names)
    export("outcome_transfer", plot_transfer_matrix(matrix))
    rows = [
        {
            "outcome": "treatment_failure",
            "model_family": model,
            "horizon_days": day,
            "ipcw_auc": 0.82 - i * 0.04 - j * 0.015,
            "ci_lower": 0.77 - i * 0.04 - j * 0.015,
            "ci_upper": 0.87 - i * 0.04 - j * 0.015,
        }
        for i, model in enumerate(models[:3])
        for j, day in enumerate([30, 90, 365, 730])
    ]
    export(
        "survival_discrimination",
        plot_timedep_auc_curves(pd.DataFrame(rows), "treatment_failure"),
    )
    group_index = np.resize(np.arange(3), n)
    angle = group_index * 2 * np.pi / 3
    coords = np.column_stack(
        (latent + 2 * np.cos(angle), rng.normal(0, 0.65, n) + 2 * np.sin(angle))
    )
    metadata = {
        "Age": 60 + 23 * np.tanh(latent),
        "Cohort": np.resize(["DLBCL", "FL", "CLL"], n),
        "Treatment failure": labels,
    }
    export(
        "embedding_atlas",
        plot_embedding_stage_metadata_grid(
            {"dapt": coords, "opera": coords @ np.array([[1, 0.3], [0, 0.8]])},
            metadata,
            variable_kinds={"Age": "continuous"},
            category_labels={"Treatment failure": {0: "No failure", 1: "Failure"}},
        ),
    )
    # Panel D is a spatial calibration residual, so it is only readable against
    # a known truth: one realistic run, one with a miscalibration injected into
    # a single region of the embedding, one with perfect predictions.
    calibration_runs = (
        (
            "censoring_aware_map",
            "Treatment failure",
            {"global_shift": -0.04, "blob_shift": 0.18},
        ),
        (
            "censoring_aware_map_injected_miscalibration",
            "Treatment failure \u00b7 injected under-prediction",
            {"blob_shift": -0.30},
        ),
        (
            "censoring_aware_map_calibrated",
            "Treatment failure \u00b7 perfectly calibrated",
            {},
        ),
    )
    for name, panel_title, knobs in calibration_runs:
        cohort = synthetic_calibration_cohort(2026, **knobs)
        fig, calibration_diagnostics = plot_censoring_aware_embedding_panel(
            cohort["coords"],
            cohort["times"],
            cohort["events"],
            horizon=365,
            predicted_cif=cohort["predicted_cif"],
            embedding=cohort["embedding"],
            held_out=cohort["held_out"],
            bandwidth=0.45,
            grid_size=35,
            min_effective_support=10,
            n_neighbors=40,
            residual_bootstrap=200,
            random_state=0,
            title=panel_title,
        )
        export(name, fig)
        print(f"{name}: {json.dumps(calibration_diagnostics, sort_keys=True)}")
    rarity = pd.DataFrame(
        [
            {
                "model_family": model,
                "training_fraction": fraction,
                "median_delta_auroc": 0.02 + 0.025 * (1 - fraction) + i * 0.005,
                "lower_delta_auroc": 0.005 + 0.025 * (1 - fraction) + i * 0.005,
                "upper_delta_auroc": 0.035 + 0.025 * (1 - fraction) + i * 0.005,
            }
            for i, model in enumerate(models[:2])
            for fraction in [0.05, 0.1, 0.25, 0.5, 1]
        ]
    )
    export("label_efficiency", plot_synthetic_rarity_delta(rarity))
    vocabulary = pd.DataFrame(
        {
            "atlas_x": coords[:, 0],
            "atlas_y": coords[:, 1],
            "token_family": np.resize(["Diagnosis", "Laboratory", "Treatment"], n),
            "token": [f"token_{i}" for i in range(n)],
        }
    )
    export("vocabulary_atlas", plot_vocabulary_embedding_atlas(vocabulary))
    epoch = np.arange(20)
    training = pd.DataFrame(
        {
            "epoch": epoch,
            "train/loss": 0.3 + 0.6 * np.exp(-epoch / 6),
            "val/loss": 0.4 + 0.5 * np.exp(-epoch / 7),
            "val/auroc": 0.55 + 0.22 * (1 - np.exp(-epoch / 6)),
            "val/auprc": 0.35 + 0.19 * (1 - np.exp(-epoch / 8)),
        }
    )
    export("finetuning_diagnostics", plot_training_curves(training))
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OPERA figure design proofs</title><style>
body{font:15px system-ui;color:#17212b;background:#f4f6f9;margin:40px auto;max-width:1400px;padding:0 24px}
h1{font-size:30px;letter-spacing:-.7px}header{margin-bottom:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:24px}
article{background:white;padding:20px;border:1px solid #e0e5eb;border-radius:8px}h2{font-size:16px;margin:0 0 16px}img{width:100%;height:370px;object-fit:contain}a{color:#2d3a8c}p{color:#485463}
</style><header><h1>OPERA / figure system</h1><p>Synthetic design proofs only. Click a figure for its vector PDF; SVGs retain editable text.</p></header><main>"""
    (args.output_dir / "index.html").write_text(
        page + "\n".join(cards) + "</main></html>", encoding="utf-8"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
