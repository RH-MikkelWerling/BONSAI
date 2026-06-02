import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from opera.evaluation.patient_transfer import compute_atypicality_scores
from opera.visualization.patient_benefit_plots import (
    plot_benefit_contrast_ladder,
    plot_patient_benefit,
)


def _synthetic_embeddings(n=100, d=32, cohorts=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    subject_ids = np.arange(n)
    cohort_values = np.resize(np.asarray(cohorts), n)
    data = {
        "subject_id": subject_ids,
        "cohort": cohort_values,
    }
    x = rng.normal(size=(n, d))
    for j in range(d):
        data[f"embedding_{j}"] = x[:, j]
    return pd.DataFrame(data)


def _transfer_and_atypicality(n=50, d=16, seed=1):
    rng = np.random.default_rng(seed)
    emb = _synthetic_embeddings(n=n, d=d, cohorts=("a", "b", "c"), seed=seed)
    transfer = pd.DataFrame(
        {
            "subject_id": emb["subject_id"],
            "cohort": emb["cohort"],
            "brier_gain": rng.normal(0.02, 0.1, size=n),
        }
    )
    atyp = compute_atypicality_scores(emb, emb[["subject_id", "cohort"]], min_cohort_size=5)
    return transfer, atyp, emb


def test_compute_atypicality_scores_basic():
    embeddings = _synthetic_embeddings(n=120, d=32, cohorts=("a", "b", "c"))
    result = compute_atypicality_scores(
        embeddings,
        embeddings[["subject_id", "cohort"]],
        min_cohort_size=10,
    )

    expected = {
        "subject_id",
        "cohort",
        "atypicality_own",
        "atypicality_nearest",
        "nearest_centroid_cohort",
        "own_centroid_distance_raw",
        "nearest_centroid_distance_raw",
    }
    assert len(result) == len(embeddings)
    assert expected.issubset(result.columns)
    assert result["atypicality_own"].notna().all()
    for _, group in result.groupby("cohort"):
        assert abs(group["atypicality_own"].mean()) < 1e-9
    assert set(result["nearest_centroid_cohort"]).issubset(set(result["cohort"]))


def test_compute_atypicality_scores_small_cohort():
    embeddings = pd.concat(
        [
            _synthetic_embeddings(n=30, d=12, cohorts=("large_a",), seed=1),
            _synthetic_embeddings(n=30, d=12, cohorts=("large_b",), seed=2).assign(
                subject_id=lambda df: df["subject_id"] + 100
            ),
            _synthetic_embeddings(n=5, d=12, cohorts=("small",), seed=3).assign(
                subject_id=lambda df: df["subject_id"] + 200
            ),
        ],
        ignore_index=True,
    )

    result = compute_atypicality_scores(
        embeddings,
        embeddings[["subject_id", "cohort"]],
        min_cohort_size=10,
    )

    small = result[result["cohort"] == "small"]
    assert small["atypicality_own"].isna().all()
    assert small["atypicality_nearest"].notna().all()


def test_compute_atypicality_own_centroid_identity():
    rows = []
    for cohort, center in (("cohort_A", 1.0), ("cohort_B", -1.0)):
        for i in range(12):
            rows.append(
                {
                    "subject_id": len(rows),
                    "cohort": cohort,
                    "embedding_0": center,
                    "embedding_1": 0.0,
                    "embedding_2": 0.0,
                }
            )
    embeddings = pd.DataFrame(rows)

    result = compute_atypicality_scores(
        embeddings,
        embeddings[["subject_id", "cohort"]],
        min_cohort_size=10,
    )

    assert np.allclose(result["atypicality_own"], 0.0)
    assert np.allclose(result["own_centroid_distance_raw"], 0.0)


def test_plot_patient_benefit_runs():
    transfer, atyp, emb = _transfer_and_atypicality()

    fig = plot_patient_benefit(
        transfer,
        atyp,
        emb,
        atypicality_mode="own",
        save_path=None,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_patient_benefit_both_mode():
    transfer, atyp, emb = _transfer_and_atypicality()

    fig = plot_patient_benefit(
        transfer,
        atyp,
        emb,
        atypicality_mode="both",
        save_path=None,
    )

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_benefit_contrast_ladder_runs():
    contrasts = []
    for i in range(3):
        transfer, atyp, _ = _transfer_and_atypicality(seed=i + 4)
        contrasts.append(
            {
                "contrast_name": f"contrast_{i}",
                "patient_transfer_df": transfer,
                "atypicality_df": atyp,
            }
        )

    fig = plot_benefit_contrast_ladder(contrasts, save_path=None)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_atypicality_nearest_centroid_is_correct_disease():
    rows = []
    for i in range(20):
        rows.append(
            {
                "subject_id": i,
                "cohort": "cohort_A",
                "embedding_0": 1.0 if i < 15 else -1.0,
                "embedding_1": 0.0,
            }
        )
    for i in range(20):
        rows.append(
            {
                "subject_id": 100 + i,
                "cohort": "cohort_B",
                "embedding_0": -1.0,
                "embedding_1": 0.0,
            }
        )
    embeddings = pd.DataFrame(rows)

    result = compute_atypicality_scores(
        embeddings,
        embeddings[["subject_id", "cohort"]],
        min_cohort_size=10,
    )

    moved = result[result["subject_id"].isin([15, 16, 17, 18, 19])]
    assert (moved["nearest_centroid_cohort"] == "cohort_B").all()
