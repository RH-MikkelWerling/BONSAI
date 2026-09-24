from pathlib import Path

import pytest

from bonsai.functional.checkpointing import CompletedEpochCheckpoint


class _Trainer:
    def __init__(self):
        self.current_epoch = 0
        self.saved = []

    def save_checkpoint(self, path, weights_only=False):
        self.saved.append((Path(path), weights_only))


def test_completed_epoch_checkpoint_uses_human_epoch_number(tmp_path):
    callback = CompletedEpochCheckpoint(tmp_path, [10])
    trainer = _Trainer()

    trainer.current_epoch = 8
    callback.on_train_epoch_end(trainer, None)
    assert trainer.saved == []

    trainer.current_epoch = 9
    callback.on_train_epoch_end(trainer, None)
    assert trainer.saved == [(tmp_path / "epoch_010.ckpt", False)]

    callback.on_train_epoch_end(trainer, None)
    assert len(trainer.saved) == 1


def test_completed_epoch_checkpoint_rejects_nonpositive_epochs(tmp_path):
    with pytest.raises(ValueError, match="positive integers"):
        CompletedEpochCheckpoint(tmp_path, [0, 10])
