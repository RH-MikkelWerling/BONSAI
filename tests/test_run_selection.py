"""Only the checkpoint a completed fit selected may be scored or evaluated."""

from argparse import Namespace
import json

import lightning as L
import pandas as pd
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from bonsai.functional.checkpointing import (
    checkpoint_selection_record,
    mark_training_complete,
    resolve_resume_checkpoint,
    verified_training_selection,
)
from opera.run.paper_prediction_queue import bundle_runs, evaluate_row
from opera.run.paper_tuning_summary import summarize
from opera.run.prepare_supervision_factorial import factorial_configs
from test_paper_workflow import bundle, complete_run
from test_supervision_factorial import base_config


class _Scheduled(L.LightningModule):
    """Validation loss follows a fixed per-epoch schedule."""

    def __init__(self, schedule):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.schedule = list(schedule)

    def training_step(self, batch, batch_idx):
        return (self.weight - batch[0]).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        value = self.schedule[self.current_epoch] + 0 * self.weight.sum()
        self.log("val/score", value, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.ones(4, 1), batch_size=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(torch.ones(2, 1), batch_size=2)


def _fit(run, schedule, epochs):
    callback = ModelCheckpoint(
        dirpath=run,
        monitor="val/score",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=[callback],
        logger=CSVLogger(run, name="logs"),
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    trainer.fit(
        _Scheduled(schedule),
        ckpt_path=resolve_resume_checkpoint(run, None, auto_resume=True),
    )
    return callback


def test_resumed_fit_keeps_best_score_from_before_the_interruption(tmp_path):
    schedule = [0.5, 0.2, 0.4, 0.3]  # best at the second epoch, before resume
    _fit(tmp_path, schedule, epochs=2)  # interrupted after two epochs
    callback = _fit(tmp_path, schedule, epochs=4)  # restarted job auto-resumes
    selection = checkpoint_selection_record(callback, tmp_path)
    assert selection["best_model_score"] == pytest.approx(0.2)
    assert selection["best_epoch"] == 1
    assert len(list((tmp_path / "logs").glob("version_*"))) == 2
    (tmp_path / "checkpoint_metadata.json").write_text("{}")
    mark_training_complete(tmp_path, {"seed": 1}, selection=selection)
    assert verified_training_selection(tmp_path) == selection


def test_selection_is_rejected_when_best_checkpoint_changes(tmp_path):
    complete_run(tmp_path, "val/x")
    assert verified_training_selection(tmp_path)["best_model_score"] == 0.25
    (tmp_path / "best.ckpt").write_bytes(b"later attempt")
    with pytest.raises(ValueError, match="best_checkpoint_changed_after_completion"):
        verified_training_selection(tmp_path)
    (tmp_path / "training_complete.json").write_text(
        json.dumps({"status": "completed"})
    )
    with pytest.raises(ValueError, match="without_selection_record"):
        verified_training_selection(tmp_path)


def test_tuning_summary_ignores_metrics_from_abandoned_attempts(tmp_path):
    directory = bundle(tmp_path)
    _, rows = bundle_runs(directory)
    (row, _, config), (other, _, _) = rows[0], rows[1]
    metric = config["training"]["eval_monitor_metric"]
    run = tmp_path / "runs" / row["run_directory"]
    complete_run(run, metric, score=0.31)
    stale = run / "contrastive_multicohort_runs" / "version_0"
    stale.mkdir(parents=True)
    pd.DataFrame({metric: [0.01], "epoch": [5]}).to_csv(stale / "metrics.csv")
    legacy = tmp_path / "runs" / other["run_directory"]
    legacy.mkdir(parents=True)
    (legacy / "best.ckpt").write_bytes(b"x")
    (legacy / "training_complete.json").write_text(json.dumps({"status": "completed"}))
    table = summarize(directory, tmp_path / "runs").set_index("name")
    assert table.loc[row["name"], "score"] == pytest.approx(0.31)
    assert table.loc[row["name"], "best_epoch"] == 3
    assert (
        table.loc[other["name"], "status"]
        == "completion_marker_without_selection_record"
    )
    assert pd.isna(table.loc[other["name"], "score"])


def test_held_out_evaluation_is_closed_for_screening_candidates(tmp_path):
    directory = bundle(tmp_path)
    _, rows = bundle_runs(directory)
    row, path, config = next(i for i in rows if i[0]["phase"] == "source_screen")
    complete_run(
        tmp_path / "runs" / row["run_directory"],
        config["training"]["eval_monitor_metric"],
    )
    options = Namespace(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "predictions",
        split="held_out",
        execute=False,
    )
    assert (
        evaluate_row(row, path, config, options, aliases=())["status"]
        == "held_out_closed_for_screening_candidate"
    )
    options.split = "tuning"
    assert evaluate_row(row, path, config, options)["status"] == "ready_to_evaluate"


def test_evaluation_requires_a_matching_selection_monitor(tmp_path):
    directory = bundle(tmp_path)
    _, rows = bundle_runs(directory)
    row, path, config = next(i for i in rows if i[0]["phase"] == "supervision")
    complete_run(tmp_path / "runs" / row["run_directory"], "val/some_other_metric")
    options = Namespace(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "predictions",
        split="tuning",
        execute=False,
    )
    assert (
        evaluate_row(row, path, config, options)["status"]
        == "selection_monitor_differs_from_config"
    )


def test_factorial_fits_resume_their_own_interrupted_run():
    configs = factorial_configs(base_config(), encoder_checkpoint="/x/best.ckpt")
    assert configs and all(
        c["training"]["auto_resume"] is True for c in configs.values()
    )
