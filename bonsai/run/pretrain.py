import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.checkpointing import save_checkpoint_metadata_sidecar
from bonsai.functional.model_config import validate_pretraining_attention
from bonsai.functional.input_contract import input_contract_metadata
from bonsai.functional.versioning import generate_unused_run_id
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule
from bonsai.modules.lightningmodules.PretrainModule import PretrainModule
from bonsai.modules.networks.bonsai_nets import BonsaiPretrain
from bonsai.paths import get_config_path

OmegaConf.register_new_resolver(
    "version", lambda: generate_unused_run_id(), use_cache=True, replace=True
)

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="pretrain",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(
        f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n"
    )

    logger = CSVLogger(get_experiment_output_path(), name=None, version=0)
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
        cutoff_date=cfg.training.cutoff_date,
        max_len=cfg.training.max_len,
        train_truncation_strategy=cfg.training.get("truncation_strategy", "tail"),
        val_truncation_strategy=cfg.training.get(
            "validation_truncation_strategy", "tail"
        ),
        tail_window_probability=cfg.training.get("tail_window_probability", 1.0),
        value_embedding_mode=cfg.model.get("value_embedding_mode", "legacy"),
        numeric_value_control=cfg.training.get("numeric_value_control", "observed"),
        ignore_target_tokens=cfg.training.get("ignore_target_tokens", []),
        ignore_same_time_targets=cfg.training.get("ignore_same_time_targets", False),
        abspos_subject_jitter_years=cfg.training.get(
            "abspos_subject_jitter_years", 0.0
        ),
    )

    model = BonsaiPretrain(
        vocab_size=len(data_module.vocabulary),
        max_seqlen=cfg.model.max_seqlen,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        bias=cfg.model.bias,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        causal=cfg.model.causal,
        attn_type=cfg.model.attn_type,
        value_bin_vocab_size=cfg.model.get("value_bin_vocab_size", 0),
        value_embedding_mode=cfg.model.get("value_embedding_mode", "legacy"),
        abspos_encoding=cfg.model.get("abspos_encoding", "legacy"),
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
        value_regression_loss_weight=cfg.training.get(
            "value_regression_loss_weight", 1.0
        ),
        checkpoint_metadata={
            "training_stage": "general_pretraining",
            "dataset": cfg.dataset,
            **input_contract_metadata(
                cfg.training.get("numeric_value_control", "observed")
            ),
            "pretraining_target_policy": {
                "ignore_target_tokens": list(
                    cfg.training.get("ignore_target_tokens", [])
                ),
                "ignore_same_time_targets": bool(
                    cfg.training.get("ignore_same_time_targets", False)
                ),
                "abspos_subject_jitter_years": float(
                    cfg.training.get("abspos_subject_jitter_years", 0.0)
                ),
            },
            "pretraining_window_policy": {
                "training_strategy": str(
                    cfg.training.get("truncation_strategy", "tail")
                ),
                "validation_strategy": str(
                    cfg.training.get("validation_truncation_strategy", "tail")
                ),
                "tail_window_probability": float(
                    cfg.training.get("tail_window_probability", 1.0)
                ),
                "event_boundary_preserving": True,
            },
        },
    )

    callbacks = [ckpt_callback]
    if cfg.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=int(cfg.training.early_stopping_patience),
                min_delta=float(cfg.training.get("early_stopping_min_delta", 0.0)),
            )
        )

    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=cfg.hardware.num_devices,
        limit_val_batches=cfg.training.limit_val_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=callbacks,
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
