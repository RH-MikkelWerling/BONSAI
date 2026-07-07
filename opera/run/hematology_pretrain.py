"""Hematology-only pretraining entry point.

This wraps the BONSAI pretraining machinery with an OPERA-owned config so the
paper ladder can include a clean hematology-from-scratch checkpoint without
editing the shared BONSAI config tree.
"""

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.utils import get_class
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from bonsai.functional.checkpointing import save_checkpoint_metadata_sidecar
from bonsai.functional.model_config import validate_pretraining_attention
from bonsai.functional.pathing import get_experiment_output_path
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule
from bonsai.modules.lightningmodules.PretrainModule import PretrainModule
from opera.compat.bonsai import build_bonsai_pretrain

load_dotenv()


@hydra.main(
    config_path="../configs",
    config_name="hematology_pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="training_runs")
    model_save_dir = logger.log_dir

    dataset_class = get_class(cfg.paths.dataset_class)
    validate_pretraining_attention(dataset_class, causal=cfg.model.causal)
    data_module = PretrainDataModule(
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_vocab=cfg.paths.vocab,
        path_population=cfg.paths.population,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        dataset_class=dataset_class,
        masking_config=cfg.training.get("masking"),
        cutoff_date=cfg.training.get("cutoff_date"),
        max_len=cfg.training.max_len,
        train_truncation_strategy=cfg.training.get("truncation_strategy", "tail"),
        val_truncation_strategy=cfg.training.get(
            "validation_truncation_strategy", "tail"
        ),
        tail_window_probability=cfg.training.get("tail_window_probability", 1.0),
    )

    model = build_bonsai_pretrain(
        cfg.model,
        vocab_size=len(data_module.vocabulary),
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
        save_last=True,
    )

    lightning_module = PretrainModule(
        model=model,
        compile_mode=cfg.hardware.compile_mode,
        learning_rate=cfg.training.learning_rate,
        optimizer_epsilon=cfg.training.optimizer_epsilon,
        scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
        checkpoint_metadata={
            "training_stage": cfg.get("training_stage", "hematology_only_pretraining"),
            "dataset": cfg.get("dataset"),
            "tokenizer_vocab_path": cfg.paths.vocab,
            "split_identifier": "train:tuning",
            "pretraining_scope": "hematology_only",
        },
    )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[ckpt_callback],
        logger=[logger],
        max_epochs=cfg.training.epochs,
        num_nodes=cfg.hardware.num_nodes,
        precision=cfg.hardware.precision,
    )

    trainer.fit(
        model=lightning_module,
        datamodule=data_module,
        ckpt_path=cfg.paths.ckpt_path,
    )
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)


if __name__ == "__main__":
    main()
