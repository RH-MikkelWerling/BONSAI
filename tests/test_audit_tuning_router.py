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
