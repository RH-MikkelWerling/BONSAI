import json

import pytest

from bonsai.functional.checkpointing import (
    COMPLETION_MARKER,
    ENCODER_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    mark_training_complete,
    save_checkpoint_metadata_sidecar,
    should_skip_completed_training,
)


class DummyModule:
    def __init__(self):
        self.hparams = {
            "model_class": "DummyModel",
            MODEL_CONFIG_KEY: {"hidden_size": 8},
            ENCODER_CONFIG_KEY: {"hidden_size": 8},
            "checkpoint_metadata": {
                "training_stage": "unit_test",
                "source_checkpoint": "/tmp/source.ckpt",
            },
        }


def test_save_checkpoint_metadata_sidecar(tmp_path):
    path = save_checkpoint_metadata_sidecar(
        str(tmp_path),
        DummyModule(),
        extra={"run_id": "abc"},
    )

    payload = json.loads(path.read_text())
    assert payload["model_class"] == "DummyModel"
    assert payload[MODEL_CONFIG_KEY]["hidden_size"] == 8
    assert payload["checkpoint_metadata"]["training_stage"] == "unit_test"
    assert payload["extra"]["run_id"] == "abc"


def _write_complete_artifacts(path):
    (path / "best.ckpt").write_bytes(b"checkpoint")
    (path / "checkpoint_metadata.json").write_text("{}", encoding="utf-8")


def test_completed_training_is_reused_until_overwrite(tmp_path):
    cfg = {"seed": 42, "training": {"epochs": 2}, "overwrite": False}
    _write_complete_artifacts(tmp_path)
    marker = mark_training_complete(tmp_path, cfg)

    assert marker.name == COMPLETION_MARKER
    assert should_skip_completed_training(tmp_path, cfg) is True
    assert should_skip_completed_training(tmp_path, {**cfg, "overwrite": True}) is False


def test_completed_training_rejects_changed_config_without_overwrite(tmp_path):
    cfg = {"seed": 42, "training": {"epochs": 2}, "overwrite": False}
    _write_complete_artifacts(tmp_path)
    mark_training_complete(tmp_path, cfg)

    with pytest.raises(RuntimeError, match="configuration differs"):
        should_skip_completed_training(
            tmp_path,
            {"seed": 43, "training": {"epochs": 2}, "overwrite": False},
        )


def test_partial_training_is_not_reused(tmp_path):
    (tmp_path / "best.ckpt").write_bytes(b"checkpoint")

    assert should_skip_completed_training(tmp_path, {"overwrite": False}) is False
