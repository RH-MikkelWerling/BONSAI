"""The router audit reads existing artifacts without launching inference."""
import json

import pandas as pd
import yaml

from opera.run.audit_tuning_router import audit, digest


def test_audit_resolves_primary_alias_and_detects_embedding_join_types(tmp_path):
    root = tmp_path / "paper"
    bundle = root / "bundles" / "stage3"
    bundle.mkdir(parents=True)
    config = bundle / "model.yaml"
    config.write_text(yaml.safe_dump({"model": {"pooling": "cls_last"},
                                   "competing_risk": {"adapter_source_mode": "both"}}))
    hash_value = digest(config)
    (bundle / "bundle.json").write_text(json.dumps({"runs": [{
        "name": "fix_R_AB", "config_path": "model.yaml", "config_sha256": hash_value,
        "run_directory": "fixed"}]}))
    predictions = root / "predictions" / "fixed" / "tuning"
    predictions.mkdir(parents=True)
    (predictions / "evaluation_complete.json").write_text(json.dumps({
        "inputs": {"config_sha256": hash_value, "split": "tuning"}}))
    analysis = root / "analysis" / "preview"
    inventory = analysis / "analysis_inventory"
    inventory.mkdir(parents=True)
    (inventory / "analysis.yaml").write_text(yaml.safe_dump({
        "route_weights": str(predictions / "source_route_weights.parquet"),
        "artifacts": [{"model": "R_L", "path": str(predictions / "direct_cr_predictions.parquet")}]}))
    plot = analysis / "figure_inputs" / "plot_tables"
    plot.mkdir(parents=True)
    pd.DataFrame({"subject_id": [1, 2]}).to_csv(plot / "gallery_coordinates.csv", index=False)
    pd.DataFrame({"subject_id": [1, 2]}).to_parquet(plot / "embedding_gallery.parquet")
    result = audit(root, analysis)
    assert result["artifacts"][0]["fit_name"] == "fix_R_AB"
    assert result["route_directory_matches_primary"] is True
    assert result["embedding_join"]["matched_rows_before_normalization"] == 0
    assert result["embedding_join"]["matched_rows_after_normalization"] == 2

    checkpoint = root / "neural_runs" / "fixed" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture")
    data = predictions / "direct_cr_predictions.parquet"
    data.write_bytes(b"prediction fixture")
    marker = predictions / "evaluation_complete.json"
    saved = json.loads(marker.read_text())
    saved["inputs"]["checkpoint_sha256"] = digest(checkpoint)
    saved["prediction_sha256"] = digest(data)
    marker.write_text(json.dumps(saved))
    checked = audit(root, analysis, verify=True)["artifacts"][0]
    assert checked["checkpoint_hash_matches"] and checked["prediction_hash_matches"]
    checkpoint.write_bytes(b"changed checkpoint")
    assert audit(root, analysis, verify=True)["artifacts"][0]["checkpoint_hash_matches"] is False


def test_audit_distinguishes_cell_median_from_decision_weighted_mean(tmp_path):
    root = tmp_path / "paper"
    analysis = root / "analysis"
    (analysis / "analysis_inventory").mkdir(parents=True)
    (analysis / "analysis_inventory" / "analysis.yaml").write_text("artifacts: []")
    directory = analysis / "prediction_analysis"
    directory.mkdir()
    pd.DataFrame([
        {"split": "tuning", "metric": "auc", "contrast": "R1", "seed": 42,
         "cohort": cohort, "outcome": outcome, "horizon_days": 365, "gain": gain,
         "n_cases": 10, "n_controls": 20}
        for cohort, outcome, gain in [("A", "a", 0.03), ("A", "b", 0.03), ("B", "a", -0.03)]
    ]).to_csv(directory / "contrasts.csv", index=False)
    row = audit(root, analysis)["point_summary_comparison_raw_auc"][0]
    assert row["cell_median"] == 0.03
    assert abs(row["equal_cell_mean"] - 0.01) < 1e-12
    assert abs(row["decision_weighted_mean_min5"]) < 1e-12


def test_recompute_compares_raw_cifs_to_saved_fine_disease_estimates(tmp_path):
    import numpy as np

    from opera.evaluation.metrics import compute_competing_risk_metrics_at_horizon
    from opera.run.audit_tuning_router import reconcile_metrics
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame({"subject_id": range(8), "cohort_fine": ["A"] * 4 + ["B"] * 4}).to_parquet(metadata)
    predictions = tmp_path / "direct_cr_predictions.parquet"
    data = pd.DataFrame({"subject_id": range(8), "cohort": "group", "outcome": "x",
                         "event": [1, 0, 2, 0] * 2, "time_days": [10, 500, 20, 30] * 2,
                         "cif_365d": np.linspace(0.1, 0.9, 8)})
    data.to_parquet(predictions)
    spec = {"path": str(predictions), "model": "R_L", "format": "direct_cif",
            "cohort_from_metadata": True, "split": "tuning", "seed": 42,
            "data_contract": "fixture"}
    config = {"artifacts": [spec], "estimand": "cif_ipcw", "patient_metadata": str(metadata)}
    directory = tmp_path / "prediction_analysis"
    directory.mkdir()
    rows = []
    for cohort, part in [("A", data.iloc[:4]), ("B", data.iloc[4:])]:
        values = compute_competing_risk_metrics_at_horizon(part.time_days.to_numpy(),
            part.event.to_numpy(), part.cif_365d.to_numpy(), 365)
        for metric in ["auc", "brier"]:
            rows.append({"split": "tuning", "seed": 42, "cohort": cohort, "outcome": "x",
                         "horizon_days": 365, "model": "R_L", "metric": metric,
                         "estimate": values["cif_" + metric]})
    saved = pd.DataFrame(rows)
    saved.to_csv(directory / "estimates.csv", index=False)
    native = compute_competing_risk_metrics_at_horizon(data.time_days.to_numpy(),
        data.event.to_numpy(), data.cif_365d.to_numpy(), 365)
    pd.DataFrame([{"outcome": "x", "horizon_days": 365, "cif_auc": native["cif_auc"]}]).to_csv(
        tmp_path / "direct_cr_pooled_metrics.csv", index=False)
    result = reconcile_metrics(config, tmp_path)[0]
    assert result["native_pooled_saved_max_abs_error"] < 1e-12
    assert abs(result["native_patient_pooled_recomputed_auc_mean"] - native["cif_auc"]) < 1e-12
    assert result["missing_saved_estimates"] == result["finite_status_mismatches"] == 0
    assert result["native_vs_analysis_max_abs_error"] < 1e-12
    assert result["recomputed_vs_saved_max_abs_error"] < 1e-12
    saved.loc[0, "estimate"] += 0.02
    saved.to_csv(directory / "estimates.csv", index=False)
    assert abs(reconcile_metrics(config, tmp_path)[0]["recomputed_vs_saved_max_abs_error"] - 0.02) < 1e-12


def test_gain_distribution_separates_sparse_exact_ties_from_supported_gains():
    from opera.run.audit_tuning_router import gain_distribution
    cells = pd.DataFrame({'gain': [0., 0., .02, -.01, 1e-14],
                          'n_cases': [1, 2, 8, 9, 8], 'n_controls': [2, 3, 9, 8, 9]})
    rows = {r['population']: r for r in gain_distribution(cells)}
    assert rows['all_finite']['exactly_zero'] == 2
    assert rows['sparse_below5']['exactly_zero'] == 2
    assert rows['supported_min5']['exactly_zero'] == 0
    assert rows['supported_min5']['nonzero_within_1e_minus12'] == 1
