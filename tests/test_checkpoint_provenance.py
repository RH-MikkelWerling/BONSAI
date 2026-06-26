import json

import pytest
from omegaconf import OmegaConf

from opera.run.evaluate import resolve_evaluation_checkpoint


def test_resolve_evaluation_checkpoint_uses_run_dir_best_and_sidecar(tmp_path):
    run_dir = tmp_path / "finetune_run"
    run_dir.mkdir()
    best = run_dir / "best.ckpt"
    best.write_bytes(b"checkpoint")
    (run_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_metadata": {
                    "selection_split": "tuning",
                    "selection_metric": "val/AUROC",
                    "selection_mode": "max",
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create({"run_dir": str(run_dir), "ckpt_path": None})

    path, provenance = resolve_evaluation_checkpoint(cfg)

    assert path == best
    assert provenance["checkpoint_source"] == "run_dir_best"
    assert provenance["selection_split"] == "tuning"
    assert provenance["selection_metric"] == "val/AUROC"
    assert provenance["selection_mode"] == "max"


def test_resolve_evaluation_checkpoint_warns_on_explicit_override(tmp_path):
    ckpt = tmp_path / "manual.ckpt"
    cfg = OmegaConf.create({"ckpt_path": str(ckpt), "run_dir": None})

    with pytest.warns(RuntimeWarning, match="Explicit ckpt_path override"):
        path, provenance = resolve_evaluation_checkpoint(cfg)

    assert path == ckpt
    assert provenance["checkpoint_source"] == "explicit_override"
    assert "warning" in provenance
