"""The stage-8 GPU smoke checker flags skipped steps, frozen parameters and
reload failures (fabricated smoke outputs, no GPU)."""

import json

import pandas as pd
import torch
import yaml

from opera.run import router_smoke
from test_stage8_integration import FITS, _build


def _fake(tmp_path, alias, *, scale, move=True):
    config = FITS[f"{alias}_L"]
    model, outcomes = _build(config)
    r = model.source_adapter_readout
    if move:
        with torch.no_grad():
            r.pool_scale.fill_(1.1); r.tau_log_ratio.fill_(0.05)
    output = tmp_path / "smoke" / alias
    output.mkdir(parents=True)
    (output / "training_complete.json").write_text("{}")
    (output / "smoke.yaml").write_text(yaml.safe_dump(config))
    torch.save({"global_step": 40, "MixedPrecision": {"scale": scale, "_growth_tracker": 3},
                "hyper_parameters": {"model_init_config": {"competing_risk_config": dict(
                    config["competing_risk"], fine_names=list(r.fine_names))}, "outcome_names": outcomes},
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, output / "best.ckpt")
    logs = output / "contrastive_multicohort_runs" / "version_0"
    logs.mkdir(parents=True)
    pd.DataFrame([{"epoch": 0, "val/cr_cif_auc_macro": 0.7, "val/cr/patient_norm": 16.0,
                   "val/cr/correction_norm/own": 2.0, "val/cr/correction_norm/borrowed": 0.3,
                   "val/cr/correction_norm/global": 1.0}]).to_csv(logs / "metrics.csv", index=False)
    return {"name": f"seed42/L/fix_{alias}", "phase": "router_fix"}


def test_check_passes_a_healthy_fit_and_flags_skipped_or_frozen_ones(tmp_path, monkeypatch):
    import opera.diagnostics.representation_gradient_conflict as evaluation

    monkeypatch.setattr(evaluation, "_load_model", lambda *a, **k: None)
    healthy = router_smoke.check_one(_fake(tmp_path, "R_PF5E", scale=4096.0), tmp_path / "smoke")
    assert healthy["status"] == "ok", healthy
    skipped = router_smoke.check_one(_fake(tmp_path / "b", "R_PF5E", scale=1e-8), tmp_path / "b" / "smoke")
    assert "steps skipped" in skipped["status"]
    stuck = router_smoke.check_one(_fake(tmp_path / "c", "R_PF5E", scale=4096.0, move=False), tmp_path / "c" / "smoke")
    assert "never moved" in stuck["status"]
    missing = router_smoke.check_one({"name": "seed42/L/fix_R_PF5", "phase": "router_fix"}, tmp_path / "smoke")
    assert missing["status"].startswith("FAIL")


def test_run_lists_every_stage8_fit_once_across_shards(tmp_path, capsys):
    rows = [{"name": f"seed42/L/fix_{a[:-2]}", "phase": "router_fix", "config_path": "x", "queue_order": i}
            for i, a in enumerate(FITS, 1)]
    (tmp_path / "bundle.json").write_text(json.dumps({"runs": rows + [
        {"name": "seed42/L/fix_R_AB", "phase": "router_fix", "config_path": "x"}]}))
    seen = []
    for shard in range(1, 7):
        selected = router_smoke.stage8_rows(tmp_path)[shard - 1::6]
        seen += [router_smoke.alias(row) for row in selected]
    assert sorted(seen) == sorted(a[:-2] for a in FITS)
