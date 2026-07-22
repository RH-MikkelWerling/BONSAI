from omegaconf import OmegaConf
import torch

from opera.run.finetune import build_finetune_data_module
from opera.run.survival_finetune import build_survival_finetune_data_module
from opera.modules.datamodules.HybridDataModule import HybridDataModule


def _base_cfg(tmp_path, max_len=17):
    population = tmp_path / "population_full.csv"
    population.write_text("subject_id\n1\n2\n", encoding="utf-8")
    return OmegaConf.create(
        {
            "hardware": {"num_workers": 0},
            "paths": {
                "train_split": str(tmp_path / "subject_data_train.pt"),
                "val_split": str(tmp_path / "subject_data_tuning.pt"),
                "population": str(population),
            },
            "model": {"max_seqlen": 8192},
            "training": {
                "batch_size": 2,
                "max_len": max_len,
                "sampling_weight_fn": {
                    "_target_": "bonsai.functional.sampling.effective_n_samples",
                },
                "batch_sampling": {"type": "none"},
            },
        }
    )


def test_finetune_runner_datamodule_threads_configured_max_len(tmp_path):
    cfg = _base_cfg(tmp_path, max_len=23)
    outcomes = {1: {"label": 0}, 2: {"label": 1}}

    datamodule = build_finetune_data_module(
        cfg,
        {"[CLS]": 1},
        outcomes,
        outcomes,
        outcomes,
        [0, 1],
    )

    assert datamodule.max_len == 23


def test_survival_finetune_runner_datamodule_threads_configured_max_len(tmp_path):
    cfg = _base_cfg(tmp_path, max_len=31)
    outcomes = {
        1: {"label": 0, "time_days": 5.0, "event": 0},
        2: {"label": 1, "time_days": 2.0, "event": 1},
    }

    datamodule = build_survival_finetune_data_module(
        cfg,
        {"[CLS]": 1},
        outcomes,
        outcomes,
        outcomes,
    )

    assert datamodule.max_len == 31


def test_hybrid_datamodule_uses_explicit_encoder_sequence_limit(tmp_path):
    population = tmp_path / "population_full.csv"
    population.write_text("subject_id\n1\n", encoding="utf-8")
    tabular = tmp_path / "tabular.parquet"
    import pandas as pd

    pd.DataFrame({"subject_id": [1], "feature": [0.5]}).to_parquet(tabular)
    datamodule = HybridDataModule(
        path_train_data=str(tmp_path / "train.pt"),
        path_val_data=str(tmp_path / "val.pt"),
        path_population=str(population),
        path_tabular=str(tabular),
        feature_columns=["feature"],
        train_outcomes={1: {"label": 0}},
        val_outcomes={1: {"label": 0}},
        test_outcomes={},
        predict_token_id=1,
        max_len=37,
        batch_size=1,
        num_workers=0,
    )

    assert datamodule.max_len == 37


def test_outcome_splits_are_selected_after_pooling_physical_ssl_splits(tmp_path):
    cfg = _base_cfg(tmp_path, max_len=17)
    (tmp_path / "population_full.csv").write_text(
        "subject_id\n1\n2\n3\n", encoding="utf-8"
    )
    held_out_path = tmp_path / "subject_data_held_out.pt"
    cfg.paths.test_split = str(held_out_path)
    cfg.paths.subject_data_paths = [
        cfg.paths.train_split,
        cfg.paths.val_split,
        cfg.paths.test_split,
    ]

    def subject(subject_id):
        return {
            "subject_id": subject_id,
            "code": torch.tensor([1, 2]),
            "age": torch.tensor([1.0, 2.0]),
            "abspos": torch.tensor([1.0, 2.0]),
            "segment": torch.tensor([0, 1]),
        }

    # Deliberately place every patient in a physical file whose name does not
    # match the prospective outcome split.
    torch.save([subject(3)], cfg.paths.train_split)
    torch.save([subject(1)], cfg.paths.val_split)
    torch.save([subject(2)], cfg.paths.test_split)
    outcomes = {
        1: {"label": 0, "censor_abspos": 2.0},
        2: {"label": 1, "censor_abspos": 2.0},
        3: {"label": 0, "censor_abspos": 2.0},
    }

    datamodule = build_finetune_data_module(
        cfg,
        {"[CLS]": 1},
        {1: outcomes[1]},
        {2: outcomes[2]},
        {3: outcomes[3]},
        [0],
    )
    datamodule.setup("fit")
    datamodule.setup("predict")

    assert [item["subject_id"] for item in datamodule.train_dataset.subjects] == [1]
    assert [item["subject_id"] for item in datamodule.val_dataset.subjects] == [2]
    assert [item["subject_id"] for item in datamodule.predict_dataset.subjects] == [3]
