"""Checkpoint helpers for BONSAI and OPERA training runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from bonsai.functional.model_config import (
    config_to_dict,
    require_native_checkpoint_config,
)

MODEL_CONFIG_KEY = "model_config"
ENCODER_CONFIG_KEY = "encoder_config"
MODEL_INIT_CONFIG_KEY = "model_init_config"
COMPLETION_MARKER = "training_complete.json"


def resolve_resume_checkpoint(
    output_dir: str | Path,
    configured_checkpoint: str | Path | None,
    *,
    auto_resume: bool = False,
) -> str | None:
    """Resolve an explicit checkpoint or an existing ``last.ckpt``.

    Automatic discovery is deliberately opt-in and scoped to the current run
    directory. This prevents an invocation from silently inheriting weights
    from a different experiment while making scheduler requeues restart-safe.
    """
    if configured_checkpoint not in (None, ""):
        checkpoint = Path(configured_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Configured resume checkpoint does not exist: {checkpoint}"
            )
        print(f"Resuming training from configured checkpoint: {checkpoint}")
        return str(checkpoint)
    if not auto_resume:
        return None
    checkpoint = Path(output_dir) / "last.ckpt"
    if checkpoint.is_file():
        print(f"Auto-resuming interrupted training from: {checkpoint}")
        return str(checkpoint)
    return None


class CompletedEpochCheckpoint(Callback):
    """Save named checkpoints after exact numbers of completed epochs.

    Lightning's ``current_epoch`` is zero-based. Exposing completed-epoch names
    avoids the common ambiguity where ``epoch=10`` is actually the eleventh
    pass through the training data.
    """

    def __init__(self, dirpath: str | Path, completed_epochs: list[int] | tuple[int, ...]):
        super().__init__()
        epochs = sorted({int(value) for value in completed_epochs})
        if any(value <= 0 for value in epochs):
            raise ValueError("Checkpoint milestone epochs must be positive integers.")
        self.dirpath = Path(dirpath)
        self.completed_epochs = tuple(epochs)
        self._saved: set[int] = set()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        completed = int(trainer.current_epoch) + 1
        if completed not in self.completed_epochs or completed in self._saved:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            self.dirpath / f"epoch_{completed:03d}.ckpt",
            weights_only=False,
        )
        self._saved.add(completed)


def _completion_config(cfg: DictConfig | dict) -> dict:
    """Return the stable, resolved config used to identify a completed run."""
    if isinstance(cfg, DictConfig):
        # Remove invocation-only fields *before* resolving interpolations.  In
        # particular, the shared training config defines
        # ``run_id: ${version:}``, while OPERA entry points do not need to
        # register BONSAI's ``version`` resolver.  Resolving that field only to
        # discard it made otherwise successful training fail while writing the
        # completion marker.
        stable_cfg = OmegaConf.create(
            {
                key: value
                for key, value in cfg.items_ex(resolve=False)
                if key not in {"overwrite", "run_id"}
            }
        )
        payload = OmegaConf.to_container(stable_cfg, resolve=True)
    else:
        payload = dict(cfg)
    if not isinstance(payload, dict):
        raise TypeError("Training config must resolve to a mapping.")
    # These control invocation identity rather than the fitted model.
    payload.pop("overwrite", None)
    payload.pop("run_id", None)
    return payload


def training_config_fingerprint(cfg: DictConfig | dict) -> str:
    """Hash a resolved training config for safe completed-run reuse."""
    canonical = json.dumps(
        _completion_config(cfg),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def should_skip_completed_training(
    output_dir: str | Path,
    cfg: DictConfig | dict,
) -> bool:
    """Return whether a compatible, complete training run can be reused.

    A marker-backed run is reused only when its config fingerprint matches.
    Legacy runs containing both the checkpoint and metadata sidecar are also
    considered complete, but cannot be checked for exact config identity.
    """
    output_dir = Path(output_dir)
    overwrite = bool(cfg.get("overwrite", False))
    if overwrite:
        return False

    required = (output_dir / "best.ckpt", output_dir / "checkpoint_metadata.json")
    marker_path = output_dir / COMPLETION_MARKER
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return False
        current = training_config_fingerprint(cfg)
        recorded = marker.get("config_fingerprint")
        if recorded != current:
            raise RuntimeError(
                f"Completed training output already exists at {output_dir}, but "
                "its configuration differs from this invocation. Choose a new "
                "output directory or set overwrite=true explicitly."
            )
        print(
            f"Completed training run found at {output_dir}; reusing best.ckpt "
            "(set overwrite=true to rerun)."
        )
        return True

    if all(path.is_file() for path in required):
        print(
            f"Legacy completed training run found at {output_dir}; reusing "
            "best.ckpt (set overwrite=true to rerun)."
        )
        return True
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_selection_record(checkpoint_callback, output_dir: str | Path) -> dict:
    """Describe the checkpoint a ModelCheckpoint callback actually selected.

    Lightning restores the callback state on resume, so ``best_model_score``
    spans the interrupted and resumed segments of one fit. Logged metric files
    from abandoned attempts are deliberately not consulted.
    """
    best_path = Path(checkpoint_callback.best_model_path or "")
    expected = Path(output_dir) / "best.ckpt"
    if not best_path.is_file() or best_path.resolve() != expected.resolve():
        raise RuntimeError(
            f"Selected checkpoint {best_path} is not this run's best.ckpt ({expected})"
        )
    score = checkpoint_callback.best_model_score
    if score is None:
        raise RuntimeError("Checkpoint callback recorded no selection score")
    score = float(score)
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite checkpoint selection score: {score}")
    state = torch.load(expected, map_location="cpu", weights_only=False)
    return {
        "monitor": str(checkpoint_callback.monitor),
        "mode": str(checkpoint_callback.mode),
        "best_model_score": score,
        "best_epoch": int(state.get("epoch", -1)),
        "best_global_step": int(state.get("global_step", -1)),
        "best_checkpoint": expected.name,
        "best_checkpoint_sha256": _file_sha256(expected),
    }


def verified_training_selection(output_dir: str | Path) -> dict:
    """Return the completed run's selection record after checking best.ckpt.

    Raises ``ValueError`` when the run is incomplete, predates selection
    records, or its best.ckpt no longer matches the checkpoint that completed.
    """
    output_dir = Path(output_dir)
    marker_path = output_dir / COMPLETION_MARKER
    if not marker_path.is_file():
        raise ValueError("training_incomplete")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    selection = marker.get("selection")
    if not selection:
        raise ValueError("completion_marker_without_selection_record")
    checkpoint = output_dir / selection["best_checkpoint"]
    if not checkpoint.is_file():
        raise ValueError("missing_best_checkpoint")
    if _file_sha256(checkpoint) != selection["best_checkpoint_sha256"]:
        raise ValueError("best_checkpoint_changed_after_completion")
    return selection


def mark_training_complete(
    output_dir: str | Path,
    cfg: DictConfig | dict,
    selection: dict | None = None,
) -> Path:
    """Atomically write the marker used by resumable checkpoint generation."""
    output_dir = Path(output_dir)
    required = (output_dir / "best.ckpt", output_dir / "checkpoint_metadata.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot mark training complete; required artifacts are missing: "
            + ", ".join(missing)
        )
    path = output_dir / COMPLETION_MARKER
    temporary = output_dir / f".{COMPLETION_MARKER}.tmp"
    payload = {
        "status": "completed",
        "config_fingerprint": training_config_fingerprint(cfg),
        "artifacts": [item.name for item in required],
    }
    if selection is not None:
        payload["selection"] = selection
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def attach_model_config(module: Any, model: Any) -> None:
    """Store model architecture config and class name in module.hparams.

    Supports native models exposing ``hparams`` and wrapped models whose
    encoder exposes them. Legacy ``config`` objects remain readable for clear
    checkpoint error reporting.
    """
    if hasattr(model, "hparams"):
        config_dict = config_to_dict(model.hparams)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "hparams"):
        config_dict = config_to_dict(model.encoder.hparams)
    elif hasattr(model, "config"):
        config_dict = config_to_dict(model.config)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        config_dict = config_to_dict(model.encoder.config)
    else:
        module.hparams["model_class"] = model.__class__.__name__
        return
    module.hparams[MODEL_CONFIG_KEY] = config_dict
    module.hparams["architecture_version"] = config_dict.get(
        "architecture_version", "legacy-modernbert"
    )
    module.hparams["model_class"] = model.__class__.__name__


def attach_checkpoint_metadata(
    module: Any,
    checkpoint_metadata: Optional[dict],
) -> None:
    """Store training-stage metadata in module.hparams."""
    if checkpoint_metadata is not None:
        module.hparams["checkpoint_metadata"] = checkpoint_metadata


def save_checkpoint_metadata_sidecar(
    output_dir: str,
    module: Any,
    extra: Optional[dict] = None,
) -> Path:
    """Save a JSON sidecar alongside a Lightning checkpoint.

    The sidecar carries model class, architecture config, training-stage
    metadata, and any caller-supplied extras, making checkpoints
    self-describing without loading the full .ckpt file.
    """
    hparams = module.hparams
    payload: dict = {}
    for key in (
        "model_class",
        MODEL_CONFIG_KEY,
        ENCODER_CONFIG_KEY,
        MODEL_INIT_CONFIG_KEY,
        "checkpoint_metadata",
    ):
        if key in hparams:
            payload[key] = hparams[key]
    if extra:
        payload["extra"] = extra

    path = Path(output_dir) / "checkpoint_metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def get_saved_encoder_config(hparams: dict) -> dict:
    """Extract model/encoder architecture config from a checkpoint's hparams.

    Handles both structured checkpoints (which embed config under
    MODEL_CONFIG_KEY) and flat config dicts (random_init case where
    the caller passes model config directly as hparams).
    """
    config = hparams[MODEL_CONFIG_KEY] if MODEL_CONFIG_KEY in hparams else hparams
    return require_native_checkpoint_config(config)


def clean_lightning_state_dict(state_dict: dict) -> dict:
    """Extract a bare model state dict from a Lightning checkpoint.

    Lightning modules commonly persist loss buffers and metric state beside
    the wrapped model under keys such as ``train_loss.pos_weight``. When the
    checkpoint contains ``model.*`` keys, only that namespace belongs to the
    reconstructable network and all wrapper state must be excluded.

    Bare state dictionaries without a ``model.`` namespace remain supported.
    """
    model_items = {
        key[len("model.") :]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    return model_items or dict(state_dict)


def extract_encoder_state_dict(state_dict: dict, prefix: str = "model.") -> dict:
    """Extract native backbone tensors and discard stage-specific heads."""
    head_prefixes = (
        "head.",
        "decoder.",
        "cls.",
        "classifier.",
        "pretrain_head.",
        "value_head.",
        "value_bin_head.",
        "finetune_head.",
    )
    encoder_state = {}
    for key, value in state_dict.items():
        if prefix and not key.startswith(prefix):
            continue
        clean_key = key[len(prefix) :] if prefix else key
        if clean_key.startswith(head_prefixes):
            continue
        encoder_state[clean_key] = value
    return encoder_state


def load_state_dict_checked(
    model: Any,
    state_dict: dict,
    strict: bool = True,
) -> None:
    """Load a state dict, raising a clear RuntimeError on mismatch."""
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"State dict mismatch loading {type(model).__name__}: {exc}"
        ) from exc
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"State dict mismatch loading {type(model).__name__}. "
            f"Missing keys ({len(missing)}): {missing[:5]}... "
            f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}..."
        )


def load_pretrained_encoder_checked(model: Any, state_dict: dict) -> None:
    """Load BONSAI encoder weights while allowing a newly initialized task head."""
    encoder_state = extract_encoder_state_dict(state_dict)

    if not encoder_state:
        raise RuntimeError("Checkpoint contains no BONSAI encoder weights.")

    try:
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Encoder state mismatch loading {type(model).__name__}: {exc}"
        ) from exc

    meaningful_missing = [
        key
        for key in missing
        if not key.startswith(("cls.", "finetune_head.", "pretrain_head."))
    ]
    if meaningful_missing or unexpected:
        raise RuntimeError(
            f"Encoder state mismatch loading {type(model).__name__}. "
            f"Missing encoder keys: {meaningful_missing[:10]}; "
            f"unexpected keys: {unexpected[:10]}."
        )


def load_finetune_model_from_checkpoint(
    ckpt_path: str,
    strict: bool = True,
    map_location: str = "cpu",
    attn_type: str | None = None,
):
    """Reconstruct a BonsaiFinetune model from a Lightning checkpoint."""
    from bonsai.modules.networks.bonsai_nets import BonsaiFinetune

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    if MODEL_CONFIG_KEY not in hparams:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} is missing '{MODEL_CONFIG_KEY}'. "
            "Ensure the checkpoint was saved with attach_model_config()."
        )
    model_config = require_native_checkpoint_config(hparams[MODEL_CONFIG_KEY])
    if attn_type is not None:
        if attn_type not in {"flash", "sdpa"}:
            raise ValueError("attn_type override must be 'flash' or 'sdpa'.")
        model_config["attn_type"] = attn_type
    predict_token_id = hparams[MODEL_CONFIG_KEY].get(
        "predict_token_id", hparams.get("predict_token_id")
    )
    if predict_token_id is None:
        raise ValueError(f"Checkpoint {ckpt_path!r} is missing 'predict_token_id'.")
    model = BonsaiFinetune(
        **model_config,
        predict_token_id=int(predict_token_id),
    )
    clean_state = clean_lightning_state_dict(ckpt["state_dict"])
    load_state_dict_checked(model, clean_state, strict=strict)
    return model


def load_joint_model_from_checkpoint(
    ckpt_path: str,
    strict: bool = True,
    map_location: str = "cpu",
    attn_type: str | None = None,
):
    """Reconstruct a JointFinetuneModel from a Lightning checkpoint."""
    from opera.modules.networks.joint_finetune_net import JointFinetuneModel
    from opera.compat.bonsai import BonsaiEncoder

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    if MODEL_CONFIG_KEY not in hparams:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} is missing '{MODEL_CONFIG_KEY}'. "
            "Ensure the checkpoint was saved with attach_model_config()."
        )
    model_config = require_native_checkpoint_config(hparams[MODEL_CONFIG_KEY])
    if attn_type is not None:
        if attn_type not in {"flash", "sdpa"}:
            raise ValueError("attn_type override must be 'flash' or 'sdpa'.")
        model_config["attn_type"] = attn_type
    outcome_names = list(hparams.get("outcome_names", []))
    if not outcome_names:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} is missing non-empty 'outcome_names'."
        )
    model_init_config = dict(hparams.get(MODEL_INIT_CONFIG_KEY, {}))
    encoder = BonsaiEncoder(**model_config)
    model = JointFinetuneModel(
        encoder=encoder,
        outcome_names=outcome_names,
        hidden_size=model_init_config.get(
            "hidden_size",
            model_config.get("hidden_size", 768),
        ),
        pooling=model_init_config.get("pooling", "bigru"),
        freeze_encoder=model_init_config.get("freeze_encoder", False),
        dropout=model_init_config.get("dropout", 0.1),
        cross_outcome_config=model_init_config.get("cross_outcome_config"),
    )
    clean_state = clean_lightning_state_dict(ckpt["state_dict"])
    load_state_dict_checked(model, clean_state, strict=strict)
    return model
