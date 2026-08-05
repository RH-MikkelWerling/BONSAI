"""
Lightning module for the OPERA contrastive learning stage.

Handles:
  - Multi-outcome survival-informed contrastive training with Kendall sigma weighting
  - Logging of per-outcome losses, sigma values, precision terms, and valid-pair counts
  - Separate param groups for encoder vs projection head (different LR)
"""

import logging
import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from typing import Dict, List
import numpy as np
from bonsai.functional.checkpointing import (
    MODEL_INIT_CONFIG_KEY,
    attach_checkpoint_metadata,
    attach_model_config,
)
from bonsai.functional.scheduling import optimizer_with_warmup

LOGGER = logging.getLogger(__name__)


class OperaContrastiveModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        outcome_names: List[str],
        learning_rate: float = 1e-4,
        encoder_lr_multiplier: float = 0.1,
        optimizer_epsilon: float = 1e-6,
        scheduler_warmup_epochs: int = 1,
        dapt_anchor_weight: float = 0.2,
        gradient_cache: bool = False,
        probe_every_n_epochs: int = 1,
        enable_validation_probe: bool = True,
        log_per_outcome_metrics: bool = True,
        checkpoint_metadata: dict = None,
    ):
        super().__init__()
        self.model = model
        self.dapt_anchor_weight = dapt_anchor_weight
        if hasattr(self.model, "dapt_anchor_weight"):
            self.model.dapt_anchor_weight = dapt_anchor_weight
        elif dapt_anchor_weight != 0.0:
            LOGGER.warning(
                "dapt_anchor_weight=%s was requested but the model does not "
                "expose a dapt_anchor_weight attribute; anchor loss is disabled.",
                dapt_anchor_weight,
            )
        self.save_hyperparameters(ignore=["model"])
        attach_model_config(self, model)
        if hasattr(model, "model_init_config"):
            self.hparams[MODEL_INIT_CONFIG_KEY] = dict(model.model_init_config)
        attach_checkpoint_metadata(self, checkpoint_metadata)
        self.outcome_names = outcome_names
        self.learning_rate = learning_rate
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.optimizer_epsilon = optimizer_epsilon
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.gradient_cache = bool(gradient_cache)
        self.probe_every_n_epochs = max(1, int(probe_every_n_epochs))
        self.enable_validation_probe = bool(enable_validation_probe)
        self.log_per_outcome_metrics = bool(log_per_outcome_metrics)
        if self.gradient_cache:
            self.automatic_optimization = False

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Logical batches are lists of independently padded CPU microbatches.
        # Moving the complete list here would defeat gradient caching's memory bound.
        if self.gradient_cache and isinstance(batch, list):
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        return value

    def _should_log_metric(self, key: str) -> bool:
        if self.log_per_outcome_metrics:
            return True
        detailed_prefixes = (
            "loss/",
            "loss_sigma_input/",
            "target_entropy/",
            "excess_loss/",
            "contrastive_headroom/",
            "n_valid/",
            "n_effective_pairs/",
            "effective_pair_fraction/",
            "class_balance_factor/",
            "support_weight/",
            "cross_outcome_weight/",
            "sigma/",
            "precision/",
            "cr/loss/",
            "cr/n_valid/",
            "cr/n_target/",
            "cr/n_death/",
            "cr/smoothness/",
        )
        return not key.startswith(detailed_prefixes)

    def _cached_training_step(self, microbatches: list[dict]) -> torch.Tensor:
        """Exact logical-batch gradient using memory-sized encoder passes."""
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.model.eval()  # both encoder passes must describe the same network
        pooled_parts = []
        survival_parts = {name: {"times": [], "events": []} for name in self.outcome_names}
        subject_parts = []
        with torch.no_grad():
            for cpu_batch in microbatches:
                batch = self._to_device(cpu_batch)
                pooled_parts.append(self.model._pool(batch))
                survival = self._build_outcome_survival(batch)
                for name in self.outcome_names:
                    if name in survival:
                        survival_parts[name]["times"].append(survival[name]["times"])
                        survival_parts[name]["events"].append(survival[name]["events"])
                subject_parts.append(batch["subject_id"])
        pooled = torch.cat(pooled_parts).detach().requires_grad_(True)
        outcome_survival = {
            name: {key: torch.cat(parts) for key, parts in fields.items()}
            for name, fields in survival_parts.items()
            if fields["times"]
        }
        result = self.model.forward_from_pooled(
            pooled, outcome_survival, torch.cat(subject_parts)
        )
        loss = result["loss"]
        self.manual_backward(loss)
        pooled_grad = pooled.grad.detach()
        offset = 0
        for cpu_batch in microbatches:
            batch = self._to_device(cpu_batch)
            recomputed = self.model._pool(batch)
            count = recomputed.shape[0]
            self.manual_backward((recomputed * pooled_grad[offset : offset + count]).sum())
            offset += count
        optimizer.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            # Gradient caching uses manual optimization, so Lightning does not
            # advance an interval="step" scheduler for us.
            scheduler.step()
        logical_size = len(pooled)
        self.log(
            "train/loss", loss.detach(), prog_bar=True,
            on_step=True, on_epoch=True, batch_size=logical_size,
        )
        self.log(
            "train/logical_batch_size", float(logical_size),
            on_step=True, on_epoch=True, batch_size=logical_size,
        )
        auxiliary_metrics = {
            f"train/{key}": value
            for key, value in result.items()
            if key != "loss" and self._should_log_metric(key)
        }
        self.log_dict(
            auxiliary_metrics,
            on_step=True,
            on_epoch=True,
            batch_size=logical_size,
        )
        return loss.detach()

    def _build_outcome_survival(
        self, batch: dict
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collect per-outcome survival dicts from the batch.

        Looks for ``time_<name>`` and ``event_<name>`` tensors.
        Falls back gracefully if only binary ``outcome_<name>`` is present
        (treats label as event indicator, time = -1 → missing → pair skipped).
        """
        outcome_survival = {}
        for name in self.outcome_names:
            time_key = f"time_{name}"
            event_key = f"event_{name}"
            label_key = f"outcome_{name}"

            if time_key in batch and event_key in batch:
                outcome_survival[name] = {
                    "times": batch[time_key],
                    "events": batch[event_key],
                }
            elif label_key in batch:
                # Fallback: binary label only — treat as event, time unknown
                labels = batch[label_key]
                outcome_survival[name] = {
                    "times": torch.full_like(labels, -1, dtype=torch.float),
                    "events": labels,
                }
        return outcome_survival

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        outcome_survival = self._build_outcome_survival(batch)
        log_dict = self.model(batch, outcome_survival)
        loss = log_dict["loss"]

        self.log(f"{prefix}/loss", loss, prog_bar=True)
        for k, v in log_dict.items():
            if k != "loss" and self._should_log_metric(k):
                self.log(f"{prefix}/{k}", v)

        return loss

    def training_step(self, batch, batch_idx):
        if self.gradient_cache:
            return self._cached_training_step(batch)
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if self.gradient_cache and isinstance(batch, list):
            pooled_parts, subject_parts = [], []
            survival_parts = {name: {"times": [], "events": []} for name in self.outcome_names}
            with torch.no_grad():
                for cpu_batch in batch:
                    microbatch = self._to_device(cpu_batch)
                    pooled_parts.append(self.model._pool(microbatch))
                    survival = self._build_outcome_survival(microbatch)
                    for name, fields in survival.items():
                        survival_parts[name]["times"].append(fields["times"])
                        survival_parts[name]["events"].append(fields["events"])
                    subject_parts.append(microbatch["subject_id"])
                pooled = torch.cat(pooled_parts)
                result = self.model.forward_from_pooled(
                    pooled,
                    {name: {key: torch.cat(values) for key, values in fields.items()}
                     for name, fields in survival_parts.items() if fields["times"]},
                    torch.cat(subject_parts),
                )
            self.log("val/loss", result["loss"], prog_bar=True, batch_size=len(pooled))
            for key, value in result.items():
                if key != "loss" and self._should_log_metric(key):
                    self.log(f"val/{key}", value, batch_size=len(pooled))
            if not self.enable_validation_probe:
                return result["loss"]
            if not hasattr(self, "_val_probe_emb_store"):
                self._val_probe_embs = {name: [] for name in self.outcome_names}
                self._val_probe_labels = {name: [] for name in self.outcome_names}
                self._val_probe_emb_store = []
            self._val_probe_emb_store.append(self.model.projection(pooled).detach().cpu())
            for name in self.outcome_names:
                key = f"outcome_{name}"
                labels = [micro[key].detach().cpu() for micro in batch if key in micro]
                if labels:
                    self._val_probe_labels[name].append(torch.cat(labels))
            return result["loss"]
        loss = self._shared_step(batch, "val")

        if not self.enable_validation_probe:
            return loss

        # Accumulate embeddings + labels for end-of-epoch linear probe
        with torch.no_grad():
            emb = self.model.get_embeddings(batch).cpu()

        if not hasattr(self, "_val_probe_embs"):
            self._val_probe_embs: Dict[str, list] = {n: [] for n in self.outcome_names}
            self._val_probe_labels: Dict[str, list] = {
                n: [] for n in self.outcome_names
            }
            self._val_probe_emb_store: list = []

        self._val_probe_emb_store.append(emb)
        for name in self.outcome_names:
            label_key = f"outcome_{name}"
            if label_key in batch:
                self._val_probe_labels[name].append(batch[label_key].cpu())

        return loss

    def on_validation_epoch_end(self):
        """
        Frozen linear probe AUROC on val embeddings, logged per outcome.
        This directly measures representation quality without training overhead —
        the probe is a logistic regression on already-computed frozen embeddings.
        """
        if not hasattr(self, "_val_probe_emb_store") or not self._val_probe_emb_store:
            return

        is_last_epoch = self.current_epoch + 1 >= self.trainer.max_epochs
        should_probe = (
            (self.current_epoch + 1) % self.probe_every_n_epochs == 0
            or is_last_epoch
        )
        if not should_probe:
            self._val_probe_emb_store = []
            self._val_probe_embs = {n: [] for n in self.outcome_names}
            self._val_probe_labels = {n: [] for n in self.outcome_names}
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            all_embs = torch.cat(self._val_probe_emb_store, dim=0).numpy()

            probe_aurocs = []
            for name in self.outcome_names:
                labels_list = self._val_probe_labels.get(name, [])
                if not labels_list:
                    continue
                labels = torch.cat(labels_list, dim=0).numpy()
                valid = labels >= 0
                if valid.sum() < 40 or len(np.unique(labels[valid])) < 2:
                    continue

                X_all = all_embs[valid]
                y_all = labels[valid]
                try:
                    # Split 80/20 so the probe is scored on held-out data.
                    # stratify ensures both splits have both classes.
                    X_tr, X_ev, y_tr, y_ev = train_test_split(
                        X_all, y_all, test_size=0.2, random_state=0, stratify=y_all
                    )
                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ev)) < 2:
                        continue
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=1.0, max_iter=200, solver="lbfgs", warm_start=False
                        ),
                    )
                    clf.fit(X_tr, y_tr)
                    probs = clf.predict_proba(X_ev)[:, 1]
                    auroc = float(roc_auc_score(y_ev, probs))
                    self.log(
                        f"val/diagnostic_ever_event_projection_auroc/{name}",
                        auroc,
                        prog_bar=False,
                    )
                    probe_aurocs.append(auroc)
                except Exception:
                    pass

            if probe_aurocs:
                self.log(
                    "val/diagnostic_ever_event_projection_auroc_mean",
                    float(np.mean(probe_aurocs)),
                    prog_bar=True,
                )

        except ImportError:
            pass  # scikit-learn not available
        finally:
            # Clear accumulators
            self._val_probe_emb_store = []
            self._val_probe_embs = {n: [] for n in self.outcome_names}
            self._val_probe_labels = {n: [] for n in self.outcome_names}

    def configure_optimizers(self):
        encoder_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = [{"params": head_params, "lr": self.learning_rate}]
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": self.learning_rate * self.encoder_lr_multiplier,
                }
            )

        optimizer = AdamW(param_groups, eps=self.optimizer_epsilon)

        return optimizer_with_warmup(
            optimizer,
            self.trainer,
            self.scheduler_warmup_epochs,
        )
