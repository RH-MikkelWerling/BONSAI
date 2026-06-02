"""
Embedding & prediction extraction for evaluation and visualization.

Given a checkpoint and a dataset, this module extracts:
  - Pooled embeddings (pre-projection or post-projection)
  - Logits / probabilities
  - Labels
  - Subject IDs

Works with all OPERA model types: finetune, contrastive, hybrid.
"""

from typing import Dict, List, Optional, Literal
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    model_type: Literal["finetune", "contrastive", "hybrid"] = "finetune",
) -> Dict[str, np.ndarray]:
    """
    Run inference and collect predictions + embeddings.

    Parameters
    ----------
    model : nn.Module
        A BonsaiFinetune, OperaContrastiveModel, or HybridClassifier.
    dataloader : DataLoader
    device : str
    model_type : str
        Determines how to call the model and what to extract.

    Returns
    -------
    dict with keys:
        "subject_ids": (N,) int array
        "labels": (N,) int array
        "logits": (N,) float array  (finetune/hybrid only)
        "probabilities": (N,) float array  (finetune/hybrid only)
        "embeddings": (N, D) float array
    """
    model = model.to(device)
    model.eval()

    all_subject_ids = []
    all_labels = []
    all_logits = []
    all_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        subject_ids = batch["subject_id"].cpu().numpy()
        labels = batch["target"].cpu().numpy().squeeze()

        if model_type == "finetune":
            # BonsaiFinetune: forward returns logits via FineTuneHead
            # We also want embeddings, so we call the encoder + head manually
            outputs = model.forward.__self__  # the BonsaiFinetune instance
            encoder_out = torch.nn.Module.forward(
                model.__class__.__bases__[0], model, batch
            )
            # Simpler approach: just call the model normally for logits
            logits = model(batch)
            if hasattr(logits, "squeeze"):
                logits = logits.squeeze(-1)

            # Get embeddings from the cls head (BiGRU) with return_embedding=True
            enc_out = model.__class__.__bases__[0].forward(model, batch)
            hidden = enc_out[0]
            emb = model.cls(hidden, batch["attention_mask"], return_embedding=True)

            all_logits.append(logits.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())

        elif model_type == "contrastive":
            # OperaContrastiveModel: get embeddings
            emb = model.get_embeddings(batch, return_pre_projection=False)
            all_embeddings.append(emb.cpu().numpy())

        elif model_type == "hybrid":
            logits = model(batch).squeeze(-1)
            all_logits.append(logits.cpu().numpy())

            # Get embeddings: encoder + pool (before MLP)
            with torch.set_grad_enabled(False):
                enc_out = model.encoder(batch)
            hidden = enc_out[0]
            if model.pooling == "bigru":
                emb = model.pooler(hidden, batch["attention_mask"],
                                    return_embedding=True)
            else:
                lengths = batch["attention_mask"].sum(dim=1) - 1
                emb = hidden[torch.arange(hidden.size(0)), lengths]
            all_embeddings.append(emb.cpu().numpy())

        all_subject_ids.append(subject_ids)
        all_labels.append(labels)

    result = {
        "subject_ids": np.concatenate(all_subject_ids),
        "labels": np.concatenate(all_labels),
        "embeddings": np.concatenate(all_embeddings),
    }
    if all_logits:
        logits_arr = np.concatenate(all_logits)
        result["logits"] = logits_arr
        result["probabilities"] = 1.0 / (1.0 + np.exp(-logits_arr))

    return result


def extract_from_finetune_simple(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Simplified extraction for BonsaiFinetune models that avoids the complex
    inheritance call chain.  Just gets logits and labels.
    """
    model = model.to(device)
    model.eval()

    all_sids, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting predictions"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits = model(batch)
            if hasattr(logits, "squeeze"):
                logits = logits.squeeze(-1)

            all_sids.append(batch["subject_id"].cpu().numpy())
            all_labels.append(batch["target"].cpu().numpy().squeeze())
            all_logits.append(logits.cpu().numpy())

    logits_arr = np.concatenate(all_logits)
    return {
        "subject_ids": np.concatenate(all_sids),
        "labels": np.concatenate(all_labels),
        "logits": logits_arr,
        "probabilities": 1.0 / (1.0 + np.exp(-logits_arr)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MC-Dropout uncertainty estimation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_uncertainty(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_samples: int = 30,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Estimate per-patient predictive uncertainty via MC Dropout.

    The model's encoder dropout is activated during inference by passing
    ``enable_dropout=True`` to ``get_embeddings``.  Running N forward
    passes and measuring the variance of the projected embeddings gives a
    principled uncertainty estimate without requiring an ensemble.

    Why embedding variance?
    -----------------------
    For the OPERA contrastive model we do not have a scalar prediction
    head at this stage.  The natural uncertainty measure is the *spread*
    of the embedding under stochastic dropout: a patient whose
    representation jumps around in embedding space across forward passes
    is one the model is unsure how to represent — i.e., a hard case.

    This aligns exactly with the clinician-validation experiment: the
    high-uncertainty patients are presented to hematologists who rate
    their own confidence, allowing empirical validation of whether model
    uncertainty tracks clinical difficulty.

    Parameters
    ----------
    model      : OperaContrastiveModel (or any model with ``get_embeddings``
                 that accepts ``enable_dropout=True``).
    dataloader : DataLoader over the target patient population.
    n_samples  : Number of MC forward passes (30 is usually sufficient;
                 use 50+ for publication-quality estimates).
    device     : "cuda" | "cpu".

    Returns
    -------
    dict with keys:
        "subject_ids"      : (N,) int array
        "mean_embedding"   : (N, D) float array   — mean embedding
        "uncertainty"      : (N,) float array      — scalar uncertainty per patient
                                                     (mean variance across embedding dims)
        "uncertainty_std"  : (N,) float array      — std of per-dim variances
                                                     (high = directionally uncertain)
        "labels"           : (N,) int array        — targets if present
    """
    model = model.to(device)
    model.eval()  # disables batch norm updates but we'll re-enable dropout manually

    all_subject_ids: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    # First pass: collect subject IDs and labels
    sample_runs: List[List[np.ndarray]] = []

    for sample_idx in tqdm(range(n_samples), desc="MC Dropout samples"):
        run_embeddings: List[np.ndarray] = []
        first_pass = sample_idx == 0

        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            emb = model.get_embeddings(batch, enable_dropout=True)
            run_embeddings.append(emb.cpu().numpy())

            if first_pass:
                all_subject_ids.append(batch["subject_id"].cpu().numpy())
                if "target" in batch:
                    all_labels.append(batch["target"].cpu().numpy().squeeze())

        sample_runs.append([e for e in run_embeddings])

    # Concatenate runs: shape (n_samples, N, D)
    runs_concat = np.stack(
        [np.concatenate(run) for run in sample_runs], axis=0
    )  # (S, N, D)

    mean_emb  = runs_concat.mean(axis=0)       # (N, D)
    var_emb   = runs_concat.var(axis=0)         # (N, D)

    uncertainty     = var_emb.mean(axis=-1)     # (N,) — mean variance across dims
    uncertainty_std = var_emb.std(axis=-1)      # (N,) — spread of per-dim variance

    result = {
        "subject_ids":    np.concatenate(all_subject_ids),
        "mean_embedding": mean_emb,
        "uncertainty":    uncertainty,
        "uncertainty_std": uncertainty_std,
    }
    if all_labels:
        result["labels"] = np.concatenate(all_labels)

    return result


def rank_by_uncertainty(
    uncertainty_results: Dict[str, np.ndarray],
    top_k: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Return the top-k most uncertain patients, sorted descending.

    This is the input to the clinician-validation experiment:
    present these patients to hematologists and ask them to rate
    their own confidence in predicting the outcome.  A positive
    correlation between model uncertainty and clinician uncertainty
    validates that the model has learned what makes a case hard.

    Parameters
    ----------
    uncertainty_results : output of ``extract_uncertainty``.
    top_k               : number of patients to return.

    Returns
    -------
    dict with same keys as input, filtered and sorted by uncertainty (desc).
    """
    u = uncertainty_results["uncertainty"]
    idx = np.argsort(u)[::-1][:top_k]

    return {k: v[idx] for k, v in uncertainty_results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# DAPT-prior embedding store construction
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_dapt_embedding_store(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> Dict[int, torch.Tensor]:
    """
    Extract and store DAPT-stage embeddings for all patients.

    These frozen embeddings are used as the cohort-similarity prior
    in ``MultiOutcomeSurvivalLoss._compute_dapt_weights``.  The idea:
    two patients whose *clinical histories* (as understood by the DAPT
    model) are similar should have their contrastive pairs upweighted,
    regardless of disease label.

    This is pre-computed *once* from the DAPT checkpoint and stored on
    disk to avoid redundant forward passes during contrastive training.

    Parameters
    ----------
    model      : DAPT-stage model with ``get_embeddings`` method.
                 Should be the OperaContrastiveModel loaded from the
                 DAPT checkpoint (pre-projection embeddings used).
    dataloader : DataLoader over the full contrastive training population.
    device     : "cuda" | "cpu".
    save_path  : if provided, save the store as a .pt file.

    Returns
    -------
    dict mapping subject_id (int) -> (D,) float tensor.
    """
    model = model.to(device)
    model.eval()

    store: Dict[int, torch.Tensor] = {}

    for batch in tqdm(dataloader, desc="Building DAPT embedding store"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Use pre-projection embeddings (768d) as the prior
        embs = model.get_embeddings(batch, return_pre_projection=True)
        sids = batch["subject_id"].cpu().tolist()

        for sid, emb in zip(sids, embs.cpu()):
            store[int(sid)] = emb.clone()

    if save_path is not None:
        torch.save(store, save_path)
        print(f"DAPT embedding store saved to {save_path}  ({len(store)} patients)")

    return store
