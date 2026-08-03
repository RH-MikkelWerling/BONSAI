"""
Domain-Adaptive Pretraining (DAPT) for OPERA.

Initialises a BonsaiPretrain model from an existing pretrained checkpoint
and continues MLM/AR pretraining on a domain-specific (e.g. hematology)
cohort.

Two vocabulary modes:

  expand_vocab: false  (default)
      Uses the base pretrained vocabulary unchanged.  Safe, clean,
      directly comparable to the base model.

  expand_vocab: true
      Loads an additional domain vocabulary (e.g. RKKP tokens), merges
      it with the base vocab, resizes the embedding + decoder layers,
      and trains with differential LR so the new embeddings can catch
      up without destabilising the pretrained ones.

      Two sub-strategies for handling the pretrained embedding rows:
        freeze_pretrained_embeds: false  (default)
            Moderate LR on embedding/decoder layers, base LR elsewhere.
        freeze_pretrained_embeds: true
            Gradient hooks zero out gradients for pretrained rows, so
            a higher LR only affects new tokens.  More precise but
            adds hook complexity.

Usage:
    # Standard DAPT (no vocab expansion):
    python -m opera.run.dapt pretrain_ckpt=/path/to/base.ckpt dataset=hema

    # DAPT with vocabulary expansion:
    python -m opera.run.dapt pretrain_ckpt=/path/to/base.ckpt dataset=hema \
        expand_vocab=true \
        paths.domain_vocab=/path/to/rkkp_vocabulary.pt \
        vocab_expansion.domain_prefix=RKKP
"""

import logging
import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from bonsai.functional.pathing import get_experiment_output_path
from bonsai.functional.model_config import validate_pretraining_attention
from bonsai.functional.checkpointing import (
    clean_lightning_state_dict,
    get_saved_encoder_config,
    save_checkpoint_metadata_sidecar,
)
from bonsai.functional.checkpointing import (
    mark_training_complete,
    should_skip_completed_training,
)
from opera.compat.bonsai import build_bonsai_pretrain
from bonsai.modules.datamodules.PretrainDataModule import PretrainDataModule

load_dotenv()


@hydra.main(
    config_path="../configs",
    config_name="dapt",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger = CSVLogger(get_experiment_output_path(), name="dapt_runs")
    model_save_dir = get_experiment_output_path()
    if should_skip_completed_training(model_save_dir, cfg):
        return

    # ── Checkpoint and base vocabulary ───────────────────────────────
    ckpt = torch.load(cfg.pretrain_ckpt, map_location="cpu", weights_only=False)
    pretrain_hparams = ckpt["hyper_parameters"]
    model_cfg = get_saved_encoder_config(pretrain_hparams)

    # ── Vocabulary handling ──────────────────────────────────────────
    base_vocab = torch.load(cfg.paths.vocab)
    old_vocab_size = len(base_vocab)
    checkpoint_vocab_size = int(model_cfg["vocab_size"])
    if old_vocab_size != checkpoint_vocab_size:
        raise ValueError(
            "DAPT base vocabulary/checkpoint mismatch: "
            f"paths.vocab has {old_vocab_size} tokens but the checkpoint expects "
            f"{checkpoint_vocab_size}. For joined lab-bin suffixes, keep "
            "paths.vocab pointed at the checkpoint's original vocabulary and "
            "supply the hematology/bin vocabulary through vocab_expansion."
        )
    expand = cfg.get("expand_vocab", False)

    if expand:
        from opera.functional.vocab_expansion import (
            merge_vocabularies,
            merge_multiple_vocabularies,
            expand_model_vocab,
        )

        vcfg = cfg.vocab_expansion

        # Support single or multiple domain vocabs
        if vcfg.get("domain_vocabs"):
            # Multiple sources: {"RKKP": path, "FLOW": path, ...}
            domain_vocabs = {}
            sources = OmegaConf.to_container(vcfg.domain_vocabs, resolve=True)
            for prefix, path in sources.items():
                domain_vocabs[prefix] = torch.load(path)
            merged_vocab, n_base, n_new = merge_multiple_vocabularies(
                base_vocab, domain_vocabs
            )
        else:
            # Single source
            domain_vocab = torch.load(vcfg.domain_vocab_path)
            merged_vocab, n_base, n_new = merge_vocabularies(
                base_vocab,
                domain_vocab,
                domain_prefix=vcfg.get("domain_prefix"),
            )

        dapt_vocab = merged_vocab
        logging.info(f"Expanded vocabulary: {old_vocab_size} → {len(dapt_vocab)}")

        # Save expanded vocab for downstream stages
        expanded_vocab_path = f"{model_save_dir}/vocabulary_expanded.pt"
        torch.save(dapt_vocab, expanded_vocab_path)
        logging.info(f"Saved expanded vocabulary to {expanded_vocab_path}")
    else:
        dapt_vocab = base_vocab
        n_new = 0

    # ── Data ─────────────────────────────────────────────────────────
    # NOTE: The domain data must already be tokenized with the expanded
    # vocabulary.  If expand_vocab=true, you need to re-run create_data
    # with the expanded vocab first.  The DataModule loads the vocab from
    # the path specified in paths.vocab.
    #
    # When expand_vocab=true, override paths.vocab to point to the
    # expanded vocabulary, OR pass it via the command line.

    # For the DataModule, we temporarily save the vocab if expanded
    if expand:
        vocab_path_for_dm = expanded_vocab_path
    else:
        vocab_path_for_dm = cfg.paths.vocab

    dataset_class = get_class(cfg.paths.dataset_class)
    validate_pretraining_attention(dataset_class, causal=model_cfg["causal"])
    data_module = PretrainDataModule(
        path_train_data=cfg.paths.train_split,
        path_val_data=cfg.paths.val_split,
        path_vocab=vocab_path_for_dm,
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
        value_embedding_mode=model_cfg.get("value_embedding_mode", "legacy"),
    )

    # ── Model ────────────────────────────────────────────────────────
    # First: create model with OLD vocab size to load pretrained weights
    model = build_bonsai_pretrain(model_cfg, vocab_size=old_vocab_size)

    # Load pretrained weights
    model_state = clean_lightning_state_dict(ckpt["state_dict"])
    model.load_state_dict(model_state, strict=True)

    # Then: expand vocab if needed (preserves loaded weights)
    if expand and n_new > 0:
        model = expand_model_vocab(
            model,
            old_vocab_size=old_vocab_size,
            new_vocab_size=len(dapt_vocab),
            init_std=cfg.vocab_expansion.get("init_std", 0.02),
        )

    # ── Lightning module ─────────────────────────────────────────────
    if expand and n_new > 0:
        from opera.modules.lightningmodules.DAPTPretrainModule import DAPTPretrainModule

        lightning_module = DAPTPretrainModule(
            model=model,
            learning_rate=cfg.training.learning_rate,
            optimizer_epsilon=cfg.training.optimizer_epsilon,
            scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
            old_vocab_size=old_vocab_size,
            new_embed_lr_multiplier=cfg.vocab_expansion.get(
                "new_embed_lr_multiplier", 5.0
            ),
            freeze_pretrained_embeds=cfg.vocab_expansion.get(
                "freeze_pretrained_embeds", False
            ),
            value_bin_loss_weight=cfg.training.get("value_bin_loss_weight", 1.0),
            value_regression_loss_weight=cfg.training.get(
                "value_regression_loss_weight", 1.0
            ),
            checkpoint_metadata={
                "training_stage": "hematology_domain_adaptation",
                "source_checkpoint": cfg.pretrain_ckpt,
                "dataset": cfg.dataset,
                "tokenizer_vocab_path": vocab_path_for_dm,
                "vocab_expanded": True,
            },
        )
    else:
        # No expansion — use the standard BONSAI PretrainModule
        from bonsai.modules.lightningmodules.PretrainModule import PretrainModule

        lightning_module = PretrainModule(
            model=model,
            learning_rate=cfg.training.learning_rate,
            optimizer_epsilon=cfg.training.optimizer_epsilon,
            scheduler_warmup_epochs=cfg.training.scheduler_warmup_epochs,
            value_bin_loss_weight=cfg.training.get("value_bin_loss_weight", 1.0),
            value_regression_loss_weight=cfg.training.get(
                "value_regression_loss_weight", 1.0
            ),
            checkpoint_metadata={
                "training_stage": "hematology_domain_adaptation",
                "source_checkpoint": cfg.pretrain_ckpt,
                "dataset": cfg.dataset,
                "tokenizer_vocab_path": vocab_path_for_dm,
                "vocab_expanded": False,
            },
        )

    # ── Training ─────────────────────────────────────────────────────
    ckpt_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
        save_last=True,
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
        ckpt_path=cfg.paths.get("ckpt_path"),
    )
    save_checkpoint_metadata_sidecar(model_save_dir, lightning_module)
    mark_training_complete(model_save_dir, cfg)


if __name__ == "__main__":
    main()
