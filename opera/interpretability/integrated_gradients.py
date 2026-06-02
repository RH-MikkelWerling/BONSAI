"""Generic integrated-gradients tooling for OPERA token attributions.

This module intentionally separates the attribution math from the concrete
dataset/model plumbing. The current repository context does not expose one
stable token-level inference API for per-patient pre-index sequences, so the
caller provides lightweight callback adapters:

- ``batch_getter(patient_id)`` returns the model-ready patient batch.
- ``forward_fn(model, batch, outcome_head)`` returns a scalar logit/risk score.
- ``token_metadata_getter(patient_id)`` returns one row per token position.

That keeps the attribution engine reusable once the OPERA inference path is
finalized on the cluster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from opera.visualization.style import FIG_HALF, save_fig, setup_style

LOGGER = logging.getLogger(__name__)


@dataclass
class IntegratedGradientsExplainer:
    """Adapter-based integrated gradients explainer."""

    model: Any
    batch_getter: Callable[[Any], Dict[str, Any]]
    forward_fn: Callable[[Any, Dict[str, Any], str], Any]
    token_metadata_getter: Callable[[Any], pd.DataFrame]
    cohort_mean_embedding: Optional[Any] = None
    device: Optional[str] = None

    def compute_ig(
        self,
        patient_id: Any,
        outcome_head: str,
        baseline: str = "mean_cohort",
        n_steps: int = 50,
    ) -> pd.DataFrame:
        """Compute token-level integrated gradients for one patient."""
        import torch

        batch = self.batch_getter(patient_id)
        token_embeddings = batch["token_embeddings"]
        if not torch.is_tensor(token_embeddings):
            raise TypeError("batch['token_embeddings'] must be a torch.Tensor.")
        token_embeddings = token_embeddings.detach().to(self.device or token_embeddings.device)
        batch = dict(batch)
        batch["token_embeddings"] = token_embeddings

        baseline_tensor = self._baseline_tensor(
            token_embeddings=token_embeddings,
            baseline=baseline,
        )

        scaled_inputs = [
            baseline_tensor + (float(step) / n_steps) * (token_embeddings - baseline_tensor)
            for step in range(1, n_steps + 1)
        ]

        grads = []
        self.model.eval()
        for scaled in scaled_inputs:
            scaled = scaled.clone().detach().requires_grad_(True)
            local_batch = dict(batch)
            local_batch["token_embeddings"] = scaled
            output = self.forward_fn(self.model, local_batch, outcome_head)
            if torch.is_tensor(output):
                scalar = output.squeeze()
            else:
                raise TypeError("forward_fn must return a torch.Tensor scalar/logit.")
            self.model.zero_grad(set_to_none=True)
            scalar.backward(retain_graph=False)
            grads.append(scaled.grad.detach().cpu())

        mean_grad = torch.stack(grads, dim=0).mean(dim=0)
        attributions = ((token_embeddings.detach().cpu() - baseline_tensor.detach().cpu()) * mean_grad).sum(dim=-1)
        metadata = self.token_metadata_getter(patient_id).copy()
        metadata = metadata.reset_index(drop=True)
        if len(metadata) != len(attributions):
            raise ValueError(
                f"Token metadata length mismatch for patient_id={patient_id!r}: "
                f"{len(metadata)} rows vs {len(attributions)} attribution positions."
            )
        metadata["patient_id"] = patient_id
        metadata["outcome_head"] = outcome_head
        metadata["attribution"] = attributions.numpy().astype(float)
        metadata["abs_attribution"] = np.abs(metadata["attribution"])
        return metadata

    def aggregate_ig_population(
        self,
        patient_ids: Sequence[Any],
        outcome_head: str,
        *,
        groupby: Sequence[str] = ("event_category", "time_window"),
        batch_size: int = 32,
        baseline: str = "mean_cohort",
        n_steps: int = 50,
        risk_scores: Optional[Mapping[Any, float]] = None,
    ) -> Dict[str, Any]:
        """Aggregate token attributions for a patient set."""
        rows: List[pd.DataFrame] = []
        iterator = tqdm(range(0, len(patient_ids), batch_size), desc=f"ig:{outcome_head}")
        for start in iterator:
            batch_ids = patient_ids[start : start + batch_size]
            for patient_id in batch_ids:
                rows.append(
                    self.compute_ig(
                        patient_id=patient_id,
                        outcome_head=outcome_head,
                        baseline=baseline,
                        n_steps=n_steps,
                    )
                )
        if not rows:
            return {"aggregate": pd.DataFrame(), "top_event_codes": pd.DataFrame()}

        attributions = pd.concat(rows, axis=0, ignore_index=True)
        attributions["time_window"] = attributions["timestamp_relative_to_index"].apply(_time_window_label)
        attributions["risk_group"] = "all"
        if risk_scores is not None:
            risk_series = pd.Series(risk_scores, dtype=float)
            q1 = float(risk_series.quantile(0.25))
            q3 = float(risk_series.quantile(0.75))
            attributions["risk_score"] = attributions["patient_id"].map(risk_scores)
            attributions.loc[attributions["risk_score"] <= q1, "risk_group"] = "low_risk"
            attributions.loc[attributions["risk_score"] >= q3, "risk_group"] = "high_risk"

        aggregate = (
            attributions.groupby([*groupby, "risk_group"], dropna=False)["abs_attribution"]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mean_abs_attribution",
                    "sum": "total_abs_attribution",
                    "count": "n_tokens",
                }
            )
        )

        high_risk = attributions.loc[attributions["risk_group"] == "high_risk"].copy()
        top_event_codes = (
            high_risk.groupby("event_code", dropna=False)["abs_attribution"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
            .rename(columns={"abs_attribution": "mean_abs_attribution"})
        )

        return {
            "aggregate": aggregate,
            "top_event_codes": top_event_codes,
            "raw_attributions": attributions,
        }

    def _baseline_tensor(self, token_embeddings: Any, baseline: str) -> Any:
        import torch

        if baseline == "zero":
            return torch.zeros_like(token_embeddings)
        if baseline == "mean_cohort":
            if self.cohort_mean_embedding is None:
                raise ValueError(
                    "cohort_mean_embedding is required when baseline='mean_cohort'."
                )
            base = self.cohort_mean_embedding
            if not torch.is_tensor(base):
                base = torch.tensor(base, dtype=token_embeddings.dtype, device=token_embeddings.device)
            if base.ndim == 1:
                base = base.unsqueeze(0).expand_as(token_embeddings)
            return base.to(token_embeddings.device, dtype=token_embeddings.dtype)
        raise ValueError("baseline must be 'mean_cohort' or 'zero'.")


def compute_ig(
    model: Any,
    patient_id: Any,
    outcome_head: str,
    *,
    batch_getter: Callable[[Any], Dict[str, Any]],
    forward_fn: Callable[[Any, Dict[str, Any], str], Any],
    token_metadata_getter: Callable[[Any], pd.DataFrame],
    cohort_mean_embedding: Optional[Any] = None,
    baseline: str = "mean_cohort",
    n_steps: int = 50,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper for one-patient integrated gradients."""
    explainer = IntegratedGradientsExplainer(
        model=model,
        batch_getter=batch_getter,
        forward_fn=forward_fn,
        token_metadata_getter=token_metadata_getter,
        cohort_mean_embedding=cohort_mean_embedding,
        device=device,
    )
    return explainer.compute_ig(
        patient_id=patient_id,
        outcome_head=outcome_head,
        baseline=baseline,
        n_steps=n_steps,
    )


def aggregate_ig_population(
    patient_ids: Sequence[Any],
    outcome_head: str,
    *,
    model: Any,
    batch_getter: Callable[[Any], Dict[str, Any]],
    forward_fn: Callable[[Any, Dict[str, Any], str], Any],
    token_metadata_getter: Callable[[Any], pd.DataFrame],
    cohort_mean_embedding: Optional[Any] = None,
    groupby: Sequence[str] = ("event_category", "time_window"),
    batch_size: int = 32,
    baseline: str = "mean_cohort",
    n_steps: int = 50,
    device: Optional[str] = None,
    risk_scores: Optional[Mapping[Any, float]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for population-level integrated gradients."""
    explainer = IntegratedGradientsExplainer(
        model=model,
        batch_getter=batch_getter,
        forward_fn=forward_fn,
        token_metadata_getter=token_metadata_getter,
        cohort_mean_embedding=cohort_mean_embedding,
        device=device,
    )
    return explainer.aggregate_ig_population(
        patient_ids=patient_ids,
        outcome_head=outcome_head,
        groupby=groupby,
        batch_size=batch_size,
        baseline=baseline,
        n_steps=n_steps,
        risk_scores=risk_scores,
    )


def _time_window_label(value: Any) -> str:
    days = abs(float(value))
    if days <= 30:
        return "0-30d"
    if days <= 90:
        return "1-3m"
    if days <= 180:
        return "3-6m"
    return "6-12m"


def aggregate_attributions_by_namespace(attr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate token attributions into source namespaces.

    Inputs are token-level rows with `token_id`, `attribution`, and `namespace`.
    The output quantifies mean/median absolute attribution, token count, and
    share of total absolute attribution. Scientifically, this reports which EHR
    namespaces drive OPERA predictions rather than individual sparse codes.
    """
    required = {"token_id", "attribution", "namespace"}
    missing = required - set(attr_df.columns)
    if missing:
        raise ValueError(f"Attribution frame is missing columns: {sorted(missing)}")
    frame = attr_df.copy()
    frame["abs_attribution"] = frame["attribution"].abs()
    total = float(frame["abs_attribution"].sum())
    agg = (
        frame.groupby("namespace", dropna=False)["abs_attribution"]
        .agg(
            mean_abs_attribution="mean",
            median_abs_attribution="median",
            n_tokens="count",
            total_abs_attribution="sum",
        )
        .reset_index()
    )
    agg["pct_of_total"] = (
        agg["total_abs_attribution"] / total if total > 0 else 0.0
    )
    return agg.drop(columns=["total_abs_attribution"])


def plot_namespace_attribution(
    agg_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot namespace-level attribution strength.

    Input is the output of `aggregate_attributions_by_namespace`. The returned
    figure ranks namespaces by mean absolute attribution to communicate which
    clinical data namespaces contribute most to model risk scores.
    """
    setup_style()
    df = agg_df.sort_values("mean_abs_attribution", ascending=True)
    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.barh(df["namespace"].astype(str), df["mean_abs_attribution"].astype(float))
    ax.set_xlabel("Mean absolute attribution")
    ax.set_ylabel("Namespace")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, save_path)
    return fig


__all__ = [
    "IntegratedGradientsExplainer",
    "aggregate_attributions_by_namespace",
    "aggregate_ig_population",
    "compute_ig",
    "plot_namespace_attribution",
]
