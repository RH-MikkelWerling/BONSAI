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
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List
import numpy as np
from bonsai.functional.checkpointing import (
    MODEL_INIT_CONFIG_KEY,
    attach_checkpoint_metadata,
    attach_model_config,
)

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
        dapt_anchor_weight: float = 0.0,
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
            if k != "loss":
                self.log(f"{prefix}/{k}", v)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")

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

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            all_embs = torch.cat(self._val_probe_emb_store, dim=0).numpy()
            scaler = StandardScaler()
            all_embs_s = scaler.fit_transform(all_embs)

            probe_aurocs = []
            for name in self.outcome_names:
                labels_list = self._val_probe_labels.get(name, [])
                if not labels_list:
                    continue
                labels = torch.cat(labels_list, dim=0).numpy()
                valid = labels >= 0
                if valid.sum() < 40 or len(np.unique(labels[valid])) < 2:
                    continue

                X_all = all_embs_s[valid]
                y_all = labels[valid]
                try:
                    # Split 80/20 so the probe is scored on held-out data.
                    # stratify ensures both splits have both classes.
                    X_tr, X_ev, y_tr, y_ev = train_test_split(
                        X_all, y_all, test_size=0.2, random_state=0, stratify=y_all
                    )
                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ev)) < 2:
                        continue
                    clf = LogisticRegression(
                        C=1.0, max_iter=200, solver="lbfgs", warm_start=False
                    )
                    clf.fit(X_tr, y_tr)
                    probs = clf.predict_proba(X_ev)[:, 1]
                    auroc = float(roc_auc_score(y_ev, probs))
                    self.log(f"val/probe_auroc_{name}", auroc, prog_bar=False)
                    probe_aurocs.append(auroc)
                except Exception:
                    pass

            if probe_aurocs:
                self.log(
                    "val/probe_auroc_mean", float(np.mean(probe_aurocs)), prog_bar=True
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

        steps_per_epoch = max(
            1, self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(steps_per_epoch * self.scheduler_warmup_epochs),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
