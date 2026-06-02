import json

from bonsai.functional.checkpointing import (
    ENCODER_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    save_checkpoint_metadata_sidecar,
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
