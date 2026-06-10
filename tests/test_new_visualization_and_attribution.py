import matplotlib.pyplot as plt
import pandas as pd

from opera.interpretability.integrated_gradients import (
    aggregate_attributions_by_namespace,
    plot_namespace_attribution,
)
from opera.visualization.comparison_plots import (
    plot_ipi_credibility,
    plot_outcome_dot_matrix,
)
from opera.visualization.disease_geometry_plots import (
    plot_disease_embedding,
    plot_survival_gradient,
    plot_transfer_matrix,
)
from opera.visualization.rarity_plots import plot_rarity_delta
from opera.visualization.survival_plots import (
    plot_calibration_comparison,
    plot_decision_curves,
    plot_timedep_auc_curves,
)


def test_namespace_attribution_aggregation_and_plot():
    attr = pd.DataFrame(
        {
            "token_id": [1, 2, 3],
            "namespace": ["lab", "lab", "drug"],
            "attribution": [0.2, -0.4, 0.4],
        }
    )
    agg = aggregate_attributions_by_namespace(attr)
    assert set(agg["namespace"]) == {"lab", "drug"}
    assert abs(agg["pct_of_total"].sum() - 1.0) < 1e-9
    fig = plot_namespace_attribution(agg)
    assert fig is not None
    plt.close(fig)


def test_new_visualization_functions_return_figures():
    emb = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.2, 1.2],
            "y": [0.0, 0.8, 0.1, 0.9],
            "disease": ["a", "b", "a", "b"],
            "survival_quantile": [0.1, 0.9, 0.2, 0.8],
        }
    )
    transfer = pd.DataFrame(
        {
            "source": ["a", "a", "b", "b"],
            "target": ["a", "b", "a", "b"],
            "auroc": [0.8, 0.7, 0.6, 0.75],
        }
    )
    results = pd.DataFrame(
        {
            "cohort": ["c"] * 8,
            "outcome": ["o"] * 8,
            "model_family": [
                "tabular_ehr",
                "opera",
                "tabular_rkkp",
                "dapt",
                "mol",
                "opera_joint",
                "ipi",
                "opera",
            ],
            "auroc": [0.7, 0.8, 0.65, 0.72, 0.73, 0.81, 0.62, 0.8],
            "evaluation_subset": ["full"] * 6 + ["ipi_complete", "ipi_complete"],
            "ipi_coverage": [None] * 6 + [0.7, 0.7],
            "n_train": [100] * 8,
            "rarity_tier": ["common"] * 8,
        }
    )
    auc = pd.DataFrame(
        {
            "outcome": ["o", "o"],
            "model_family": ["opera", "tabular_ehr"],
            "horizon_days": [30, 30],
            "ipcw_auc": [0.8, 0.7],
        }
    )
    cal = pd.DataFrame(
        {
            "outcome": ["o", "o"],
            "cohort": ["c", "c"],
            "predicted": [0.2, 0.8],
            "observed": [0.25, 0.75],
            "ece": [0.05, 0.05],
            "hl_pvalue": [0.5, 0.5],
        }
    )
    dca = pd.DataFrame(
        {
            "outcome": ["o", "o"],
            "model_family": ["opera", "opera"],
            "threshold": [0.1, 0.2],
            "net_benefit_model": [0.1, 0.08],
            "net_benefit_treat_all": [0.03, 0.02],
        }
    )

    figs = [
        plot_disease_embedding(emb),
        plot_survival_gradient(emb),
        plot_transfer_matrix(transfer),
        plot_rarity_delta(results),
        plot_outcome_dot_matrix(results),
        plot_ipi_credibility(results),
        plot_timedep_auc_curves(auc, "o"),
        plot_calibration_comparison(cal, "o"),
        plot_decision_curves(dca, "o"),
    ]
    assert all(fig is not None for fig in figs)
    for fig in figs:
        plt.close(fig)
