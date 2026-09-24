import json

import pytest
from omegaconf import OmegaConf

from bonsai.functional.checkpointing import resolve_resume_checkpoint
from opera.run.evaluate import resolve_evaluation_checkpoint


def test_resolve_resume_checkpoint_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "explicit.ckpt"
    explicit.write_bytes(b"checkpoint")
    (tmp_path / "last.ckpt").write_bytes(b"last")

    resolved = resolve_resume_checkpoint(tmp_path, explicit, auto_resume=True)

    assert resolved == str(explicit)


def test_resolve_resume_checkpoint_discovers_last_only_when_enabled(tmp_path):
    last = tmp_path / "last.ckpt"
    last.write_bytes(b"checkpoint")

    assert resolve_resume_checkpoint(tmp_path, None, auto_resume=False) is None
    assert resolve_resume_checkpoint(tmp_path, None, auto_resume=True) == str(last)


def test_resolve_resume_checkpoint_returns_none_for_fresh_run(tmp_path):
    assert resolve_resume_checkpoint(tmp_path, None, auto_resume=True) is None


def test_resolve_resume_checkpoint_rejects_missing_explicit_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_resume_checkpoint(
            tmp_path,
            tmp_path / "missing.ckpt",
            auto_resume=True,
        )


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
