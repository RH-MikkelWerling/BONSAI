import numpy as np
import pandas as pd

from opera.retrieval.retrieval import PatientRetriever


def _make_retrieval_inputs():
    patient_ids = [f"p{i}" for i in range(8)]
    diseases = ["DLBCL"] * 4 + ["FL"] * 4
    embeddings = {
        "p0": np.array([1.0, 0.0, 0.0]),
        "p1": np.array([0.9, 0.1, 0.0]),
        "p2": np.array([0.8, 0.2, 0.0]),
        "p3": np.array([0.7, 0.3, 0.0]),
        "p4": np.array([0.0, 1.0, 0.0]),
        "p5": np.array([0.0, 0.9, 0.1]),
        "p6": np.array([0.0, 0.8, 0.2]),
        "p7": np.array([0.0, 0.7, 0.3]),
    }
    outcomes = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "disease_subtype": diseases,
            "outcome_name": ["OS_2y"] * 8,
            "event_indicator": [1, 1, 0, 0, 1, 1, 0, 0],
            "time_to_event": [100.0, 140.0, 900.0, 800.0, 110.0, 150.0, 910.0, 820.0],
        }
    )
    rkkp = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "ipi_score": [4, 4, 1, 1, 3, 3, 1, 1],
            "ecog": [2, 2, 0, 0, 1, 1, 0, 0],
            "ldh_ratio": [1.8, 1.7, 1.0, 1.0, 1.5, 1.4, 1.0, 1.0],
            "ann_arbor_stage": [4, 4, 2, 2, 3, 3, 1, 1],
            "extranodal_sites": [2, 2, 0, 0, 1, 1, 0, 0],
        }
    )
    disease_cohorts = {"DLBCL": patient_ids[:4], "FL": patient_ids[4:]}
    embeddings_base = {patient_id: vector.copy() for patient_id, vector in embeddings.items()}
    embeddings_dapt = {
        patient_id: 0.6 * embeddings_base[patient_id] + 0.4 * embeddings[patient_id]
        for patient_id in patient_ids
    }
    return embeddings_base, embeddings_dapt, embeddings, outcomes, rkkp, disease_cohorts


def test_retrieve_similar_returns_ranked_neighbors():
    embeddings_base, embeddings_dapt, embeddings, outcomes, rkkp, disease_cohorts = _make_retrieval_inputs()
    retriever = PatientRetriever(
        embeddings_base=embeddings_base,
        embeddings_dapt=embeddings_dapt,
        embeddings_opera=embeddings,
        outcomes=outcomes,
        rkkp=rkkp,
        disease_cohorts=disease_cohorts,
        use_faiss=False,
    )

    neighbors = retriever.retrieve_similar("p0", k=2, restrict_to_same_disease=True)

    assert neighbors[0][0] == "p1"
    assert neighbors[1][0] == "p2"
    assert neighbors[0][2] == "DLBCL"


def test_retrieval_validation_methods_return_structured_outputs():
    embeddings_base, embeddings_dapt, embeddings, outcomes, rkkp, disease_cohorts = _make_retrieval_inputs()
    retriever = PatientRetriever(
        embeddings_base=embeddings_base,
        embeddings_dapt=embeddings_dapt,
        embeddings_opera=embeddings,
        outcomes=outcomes,
        rkkp=rkkp,
        disease_cohorts=disease_cohorts,
        use_faiss=False,
    )

    concordance = retriever.validate_outcome_concordance(k=2, n_bootstrap=20, outcome_name="OS_2y")
    gradient = retriever.validate_rkkp_gradient(max_rank=3, n_bootstrap=20)
    case = retriever.get_case_study("p0", k=2)

    assert "opera_mean_concordance" in concordance.columns
    assert "per_rank" in gradient
    assert len(gradient["per_rank"]) == 3
    assert case["query_patient"]["patient_id"] == "p0"
    assert len(case["retrieved_patients"]) == 2


def test_pairwise_rkkp_embedding_correlation_returns_cached_structured_outputs(tmp_path):
    embeddings_base, embeddings_dapt, embeddings, outcomes, rkkp, disease_cohorts = _make_retrieval_inputs()
    retriever = PatientRetriever(
        embeddings_base=embeddings_base,
        embeddings_dapt=embeddings_dapt,
        embeddings_opera=embeddings,
        outcomes=outcomes,
        rkkp=rkkp,
        disease_cohorts=disease_cohorts,
        use_faiss=False,
    )

    cache_path = tmp_path / "pairwise.json"
    result = retriever.validate_rkkp_embedding_correlation(
        n_pairs=10,
        seed=42,
        n_bootstrap=20,
        cache_path=cache_path,
    )
    cached = retriever.validate_rkkp_embedding_correlation(
        n_pairs=10,
        seed=42,
        n_bootstrap=20,
        cache_path=cache_path,
    )

    assert cache_path.exists()
    assert "summary_sentence" in result
    assert set(result["embedding_versions"].keys()) == {"base", "dapt", "opera"}
    assert set(result["scatter_plot_data"].columns) == {
        "rkkp_similarity",
        "embedding_similarity",
        "disease_subtype",
    }
    assert "comparison_scatter_data" in result
    assert len(cached["scatter_plot_data"]) == len(result["scatter_plot_data"])
