"""Integrated-gradients explanations of the locked best model, computed once.

One pass per patient explains every requested target column at once: the
predicted risk of every outcome (at one or more horizons) and, for a routed
model, the router weight on borrowed sources. The encoder forward along the
integration path is shared, and each column costs one backward pass. All later
figures and tables (any outcome, subtype or patient group) are summaries of
these tables; nothing is recomputed.

Written inside the secure environment (never exported):

``token_attributions_INTERNAL/part-*.parquet``
    one row per patient and attributed event token (code, position, event,
    days before the index date) with one float32 column per target column,
    ``attr|<kind>|<outcome>|<horizon>``, in absolute risk. NaN where the
    patient is not eligible for that outcome.
``patient_attributions_INTERNAL.parquet``
    one row per patient and column: prediction, baseline prediction (same
    timeline, average events), completeness gap, steps, observed time and
    event, and the sampling weight.
``faithfulness_INTERNAL.parquet`` and ``sanity_INTERNAL.parquet``
    perturbation and randomisation checks for the prespecified check columns.

Patients are a stratified random sample of the evaluation split: up to
``--per-subtype`` from every fine subtype, then a uniform sample of the rest up
to ``--max-patients``. Each patient carries ``weight = 1 / inclusion
probability`` so population summaries are not tilted towards rare subtypes.
"""

from __future__ import annotations

import argparse
import copy
import json
import hashlib
import random
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from opera.evaluation.input_restriction import keep_tokens
from opera.interpretability.attribution_summaries import (
    SOURCE_ORDER,
    bin_counts,
    source_index,
)
from opera.interpretability.token_attribution import (
    attributable_mask,
    baseline_tensor,
    combined_target,
    event_ids,
    gradients_through_encoder,
    integrated_gradients,
    mean_content,
    non_attributable_ids,
    replaced_target,
    split_embeddings,
)

DEFAULT_CHECK_OUTCOMES = (
    "overall_survival,serious_infection_composite,neutropenia_g3plus"
)
HOURS_PER_DAY = 24.0


def column_name(kind, outcome, horizon):
    return f"attr|{kind}|{outcome}|{'' if horizon is None else int(horizon)}"


def parse_column(name):
    _, kind, outcome, horizon = name.split("|")
    return kind, outcome, (float(horizon) if horizon else None)


def _move(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    from scipy.stats import spearmanr

    if torch.all(a == a[0]) or torch.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a.numpy(), b.numpy()).statistic)


def randomized_copy(model, scope, seed=0):
    """Model copy with re-drawn readout (``head``) or readout and encoder layers (``model``).

    Each tensor is redrawn from a normal distribution with its own standard
    deviation, so the scale of every layer is kept and only what was learned is
    lost (Adebayo et al., cascading randomisation). Embeddings are kept, so the
    baseline is unchanged.
    """
    clone = copy.deepcopy(model)
    modules = [clone.competing_risk_head]
    if getattr(clone, "source_adapter_readout", None) is not None:
        modules.append(clone.source_adapter_readout)
    if scope == "model":
        modules += [*clone.encoder.layers, clone.encoder.layernorm]
    elif scope != "head":
        raise ValueError("scope must be 'head' or 'model'")
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in modules:
            for parameter in module.parameters():
                if parameter.numel() > 1 and float(parameter.std()) > 0:
                    noise = torch.randn(parameter.shape, generator=generator)
                    parameter.copy_((noise * parameter.std().cpu()).to(parameter))
    return clone.eval()


def stratified_sample(pool, *, per_subtype, total, seed):
    """Indices into ``pool`` (dicts with ``subtype``) and their sampling weights.

    Stage 1 takes ``m_s = min(per_subtype, N_s)`` patients at random from every
    subtype ``s``; stage 2 samples a fraction ``f`` of all remaining patients
    uniformly, up to ``total``. A patient of subtype ``s`` is included with
    probability ``(m_s + (N_s - m_s) f) / N_s`` and weighted by its inverse.
    """
    if total < 1 or per_subtype < 0 or not pool:
        raise ValueError(
            "Need a nonempty population, positive cap and nonnegative subtype quota"
        )
    rng = np.random.default_rng(seed)
    subtypes = pd.Series([p["subtype"] for p in pool])
    first, remainder = [], []
    for _, part in subtypes.groupby(subtypes, sort=True):
        index = rng.permutation(part.index.to_numpy())
        first += index[:per_subtype].tolist()
        remainder += index[per_subtype:].tolist()
    if len(first) > total:
        raise ValueError(
            "Subtype quotas exceed max-patients; reduce per-subtype or increase the explicit cap"
        )
    extra = min(max(0, total - len(first)), len(remainder))
    second = rng.choice(remainder, size=extra, replace=False).tolist() if extra else []
    fraction = extra / len(remainder) if remainder else 0.0
    size = subtypes.value_counts()
    chosen = first + second
    weights = []
    for i in chosen:
        n = int(size[subtypes[i]])
        m = min(per_subtype, n)
        weights.append(n / (m + (n - m) * fraction))
    return chosen, weights


class Explainer:
    """Per-patient attributions for many target columns from one integration path."""

    def __init__(
        self,
        model,
        vocabulary,
        *,
        collate,
        columns,
        check_columns=(),
        steps=8,
        precision="fp32",
        token_budget=65536,
        tolerance=0.05,
        baseline="mean",
        ks=(1, 5, 10, 20),
        random_repeats=5,
        deletion=True,
        seed=2026,
    ):
        self.model = model.eval()
        self.collate = collate
        self.names = {int(i): str(name) for name, i in vocabulary.items()}
        self.fixed = non_attributable_ids(vocabulary)
        lookup = source_index(vocabulary)
        table = torch.full(
            (max(lookup) + 1,), SOURCE_ORDER.index("Other"), dtype=torch.long
        )
        for index, source in lookup.items():
            table[index] = source
        self.source_table = table
        self.columns = [tuple(c) for c in columns]
        self.check_columns = [tuple(c) for c in check_columns]
        unknown = set(self.check_columns) - set(self.columns)
        if unknown:
            raise ValueError(
                f"Check columns must be explained columns: {sorted(unknown)}"
            )
        if steps < 1 or token_budget < 1 or tolerance <= 0 or random_repeats < 1:
            raise ValueError(
                "Positive steps, token budget, tolerance and random repeats required"
            )
        self.seed = int(seed)
        self.steps = int(steps)
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        self.precision = precision
        self.token_budget = int(token_budget)
        self.tolerance = float(tolerance)
        self.baseline_kind = baseline
        self.ks = tuple(int(k) for k in ks)
        if any(k < 1 for k in self.ks) or len(self.ks) != len(set(self.ks)):
            raise ValueError(
                "Perturbation k values must be unique positive event counts"
            )
        self.random_repeats = int(random_repeats)
        self.deletion = bool(deletion)
        self.rng = np.random.default_rng(seed)
        self.global_mean = None
        self.source_means = None

    # Baselines -----------------------------------------------------------
    def groups(self, codes):
        return self.source_table.to(codes.device)[
            codes.clamp(0, len(self.source_table) - 1)
        ]

    def fit_baseline(self, batches):
        """Average event content over training batches (global and per source)."""
        mean, by_source = mean_content(
            self.model.encoder, batches, self.fixed, groups=self.groups
        )
        by_source[-1] = mean
        self.global_mean, self.source_means = mean, by_source
        return mean, by_source

    def baseline_for(self, kind, content, mask, codes):
        if kind == "zero":
            return baseline_tensor(content, mask, "zero")
        if self.global_mean is None:
            raise ValueError("fit_baseline must run before mean baselines are used")
        if kind == "mean":
            return baseline_tensor(content, mask, self.global_mean)
        if kind == "source_mean":
            return baseline_tensor(
                content, mask, self.source_means, codes=codes, groups=self.groups
            )
        raise ValueError(f"Unknown baseline {kind!r}")

    def _chunk(self, length):
        return max(1, self.token_budget // max(1, length))

    def attribute(self, model, batch, columns, kind=None):
        mask = attributable_mask(batch, self.fixed)
        full, content = split_embeddings(model.encoder, batch)
        baseline = self.baseline_for(
            kind or self.baseline_kind, content, mask, batch["code"]
        )
        target = combined_target(model, columns)
        steps = self.steps
        # Half precision is opt-in: the completeness check below decides whether
        # it is accurate enough for this model (refined or flagged otherwise).
        from contextlib import nullcontext

        device = batch["code"].device.type
        precision = (nullcontext() if self.precision == "fp32" else torch.autocast(
            device, dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16))
        with torch.no_grad(), precision:
            native_prediction = target(batch).float()
        with gradients_through_encoder(model), precision:
            while True:
                attributions, f_input, f_base = integrated_gradients(
                    target,
                    batch,
                    full,
                    content,
                    baseline,
                    mask,
                    steps=steps,
                    chunk=self._chunk(full.shape[1]),
                )
                attributions, f_input, f_base = (
                    attributions.float(), f_input.float(), f_base.float())
                difference = f_input - f_base
                gap = (attributions.sum(-1) - difference) / difference.abs().clamp_min(
                    1e-3
                )
                if gap.abs().max() <= self.tolerance or steps >= 4 * self.steps:
                    break
                steps *= 4
        close = (dict(atol=1e-6, rtol=1e-5) if self.precision == "fp32"
                 else dict(atol=2e-3, rtol=2e-2))
        if not torch.allclose(native_prediction, f_input, **close):
            raise ValueError(
                "Injected embeddings do not reproduce native model predictions"
            )
        if not all(torch.isfinite(t).all() for t in (attributions, f_input, f_base)):
            raise ValueError(
                "Nonfinite attributions or probabilities; reduce chunk size or inspect the checkpoint"
            )
        return dict(
            mask=mask,
            full=full,
            content=content,
            baseline=baseline,
            target=target,
            attributions=attributions,
            f_input=f_input,
            f_base=f_base,
            gap=gap,
            steps=steps,
        )

    def eligible(self, sample, columns):
        return [
            c
            for c in columns
            if int(sample.get(f"event_{c[1]}", -1)) >= 0
            and float(sample.get(f"time_{c[1]}", -1)) >= 0
        ]

    # One patient ----------------------------------------------------------
    def explain(
        self,
        sample,
        *,
        meta,
        sanity_models=(),
        alternative_baseline=None,
        run_checks=True,
    ):
        """Wide token rows, patient rows and check rows for one (unpadded) sample."""
        patient_seed = int.from_bytes(
            hashlib.sha256(str(meta["subject_id"]).encode()).digest()[:4], "little"
        )
        self.rng = np.random.default_rng([self.seed, patient_seed])
        device = next(self.model.parameters()).device
        columns = self.eligible(sample, self.columns)
        out = {"tokens": [], "patients": [], "faithfulness": [], "sanity": []}
        if not columns:
            return out
        batch = _move(self.collate([sample]), device)
        result = self.attribute(self.model, batch, columns)
        mask = result["mask"][0]
        positions = torch.nonzero(mask).flatten()
        abspos = batch["abspos"][0]
        last = int(batch["attention_mask"][0].sum()) - 1
        events = event_ids(batch, result["mask"])[0]
        token_frame = {
            "position": positions.cpu().numpy().astype(np.int32),
            "event": events[positions].cpu().numpy().astype(np.int32),
            "code": [
                self.names.get(int(c), "[UNK]")
                for c in batch["code"][0, positions].tolist()
            ],
            "days_before_index": ((abspos[last] - abspos[positions]) / HOURS_PER_DAY)
            .cpu()
            .numpy()
            .astype(np.float32),
        }
        values = (
            result["attributions"][0][:, positions].cpu().numpy().astype(np.float32)
        )
        for column in self.columns:
            token_frame[column_name(*column)] = np.full(
                len(positions), np.nan, np.float32
            )
        for j, column in enumerate(columns):
            token_frame[column_name(*column)] = values[j]
        frame = pd.DataFrame(token_frame)
        for key, value in meta.items():
            frame.insert(0, key, value)
        out["tokens"].append(frame)
        n_events = int(events.max()) + 1 if len(positions) else 0
        for j, (kind, outcome, horizon) in enumerate(columns):
            out["patients"].append(
                {
                    **meta,
                    "column": column_name(kind, outcome, horizon),
                    "kind": kind,
                    "outcome": outcome,
                    "horizon_days": horizon,
                    "prediction": float(result["f_input"][0, j]),
                    "baseline_prediction": float(result["f_base"][0, j]),
                    "attribution_sum": float(result["attributions"][0, j].sum()),
                    "completeness_gap": float(result["gap"][0, j]),
                    "completeness_ok": bool(
                        torch.isfinite(result["gap"][0, j])
                        and result["gap"][0, j].abs() <= self.tolerance
                    ),
                    "absolute_gap": float(
                        (
                            result["attributions"][0, j].sum()
                            - (result["f_input"][0, j] - result["f_base"][0, j])
                        ).abs()
                    ),
                    "steps": result["steps"],
                    "n_tokens": len(positions),
                    "n_events": n_events,
                    "time_days": float(sample[f"time_{outcome}"]),
                    "event": int(sample[f"event_{outcome}"]),
                }
            )
        checks = [c for c in columns if c in set(self.check_columns)]
        for column in checks if run_checks else ():
            j = columns.index(column)
            out["faithfulness"] += self._perturb(
                sample, batch, result, j, column, events, meta
            )
        if checks:
            index = [columns.index(c) for c in checks]
            for name, model in sanity_models:
                other = self.attribute(model, batch, checks)
                out["sanity"] += self._agreement(
                    result, other, index, checks, positions, name, meta, absolute=True
                )
            if alternative_baseline:
                other = self.attribute(
                    self.model, batch, checks, kind=alternative_baseline
                )
                out["sanity"] += self._agreement(
                    result,
                    other,
                    index,
                    checks,
                    positions,
                    f"baseline_{alternative_baseline}",
                    meta,
                    absolute=False,
                )
        return out

    def _agreement(
        self, result, other, index, checks, positions, name, meta, *, absolute
    ):
        rows = []
        for j_other, (j, column) in enumerate(zip(index, checks)):
            a = result["attributions"][0, j, positions].cpu()
            b = other["attributions"][0, j_other, positions].cpu()
            if absolute:
                a, b = a.abs(), b.abs()
            rows.append(
                {
                    **meta,
                    "column": column_name(*column),
                    "outcome": column[1],
                    "check": name,
                    "spearman": _spearman(a, b),
                }
            )
        return rows

    def _selections(self, scores):
        """Event sets to remove: top, bottom and random, for each k."""
        n = len(scores)
        order = torch.argsort(scores, descending=True).tolist()
        chosen = []
        for k in self.ks:
            if k >= n:
                continue
            chosen.append(("top", k, 0, order[:k]))
            chosen.append(("bottom", k, 0, order[-k:]))
            for repeat in range(self.random_repeats):
                chosen.append(
                    (
                        "random",
                        k,
                        repeat,
                        self.rng.choice(n, size=k, replace=False).tolist(),
                    )
                )
        return chosen

    def _perturb(self, sample, batch, result, j, column, events, meta):
        n_events = int(events.max()) + 1 if bool((events >= 0).any()) else 0
        if not n_events or not self.ks:
            return []
        valid = events >= 0
        scores = torch.zeros(n_events, device=events.device).index_add_(
            0, events[valid], result["attributions"][0, j][valid]
        )
        selections = self._selections(scores)
        if not selections:
            return []
        name = column_name(*column)
        masks = torch.stack(
            [
                torch.isin(events, torch.tensor(chosen, device=events.device))
                for *_, chosen in selections
            ]
        )
        per_pass = self._chunk(events.shape[0])
        values = []
        for begin in range(0, len(selections), per_pass):
            part = masks[begin : begin + per_pass].unsqueeze(1)
            values.append(
                replaced_target(
                    result["target"],
                    batch,
                    result["full"],
                    result["content"],
                    result["baseline"],
                    part,
                )[:, 0, j]
            )
        rows = [
            {
                **meta,
                "column": name,
                "outcome": column[1],
                "perturbation": "replace",
                "ranking": ranking,
                "k": k,
                "repeat": repeat,
                "prediction_after": after,
            }
            for (ranking, k, repeat, _), after in zip(
                selections, torch.cat(values).tolist()
            )
        ]
        if self.deletion:
            device = batch["code"].device
            for begin in range(0, len(selections), per_pass):
                stop = min(begin + per_pass, len(selections))
                deleted = _move(
                    self.collate(
                        [
                            keep_tokens(sample, ~masks[i].cpu())
                            for i in range(begin, stop)
                        ]
                    ),
                    device,
                )
                with torch.no_grad():
                    after = result["target"](deleted)[:, j].tolist()
                rows += [
                    {
                        **meta,
                        "column": name,
                        "outcome": column[1],
                        "perturbation": "delete",
                        "ranking": ranking,
                        "k": k,
                        "repeat": repeat,
                        "prediction_after": a,
                    }
                    for (ranking, k, repeat, _), a in zip(selections[begin:stop], after)
                ]
        return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="The locked best-performing model.",
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--split", choices=["tuning", "held_out"], default="held_out")
    parser.add_argument(
        "--outcomes",
        default="all",
        help="Comma-separated outcomes to explain, or 'all' (one pass explains all).",
    )
    parser.add_argument("--horizons", default="365")
    parser.add_argument(
        "--borrowed-outcomes",
        default="",
        help="Routed model: also attribute the borrowed router weight for "
        "these outcomes ('all', a list, or '' for none).",
    )
    parser.add_argument(
        "--check-outcomes",
        default=DEFAULT_CHECK_OUTCOMES,
        help="Outcomes for faithfulness and randomisation checks.",
    )
    parser.add_argument("--max-patients", type=int, default=1500)
    parser.add_argument("--per-subtype", type=int, default=40)
    parser.add_argument("--steps", type=int, default=8,
                        help="Gauss-Legendre nodes; patients failing completeness are "
                        "recomputed with four times as many.")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
                        help="Opt-in half precision on GPU, guarded by the completeness "
                        "check; compare a pilot against fp32 before using it.")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=65536,
        help="Tokens per forward/backward pass (sequences x length).",
    )
    parser.add_argument(
        "--baseline", choices=["mean", "source_mean", "zero"], default="mean"
    )
    parser.add_argument(
        "--alternative-baseline",
        default="source_mean",
        help="Second baseline for the stability check ('' to skip).",
    )
    parser.add_argument("--baseline-patients", type=int, default=1000)
    parser.add_argument("--k", default="1,5,10")
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--no-deletion", action="store_true")
    parser.add_argument("--faithfulness-patients", type=int, default=200)
    parser.add_argument("--sanity-patients", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-new-patients",
        type=int,
        help="Pilot cap; rerun --resume without it to finish the SAME planned sample",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse hash-verified completed patient exports",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _names(value, available):
    if value.strip() == "all":
        return list(available)
    names = [v.strip() for v in value.split(",") if v.strip()]
    if len(names) != len(set(names)):
        raise ValueError("Outcome names must be unique")
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown outcomes: {sorted(unknown)}")
    return names


def _eligible(dataset, outcomes):
    for index, subject in enumerate(dataset.subjects):
        sid = subject["subject_id"]
        if any(sid in dataset.outcome_dicts.get(o, {}) for o in outcomes):
            yield index


def _run(args):
    from dotenv import load_dotenv
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from bonsai.functional.checkpointing import get_saved_encoder_config
    from bonsai.functional.input_contract import resolve_numeric_value_control
    from opera.diagnostics.representation_gradient_conflict import _load_model
    from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
    from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
        MultiCohortContrastiveDataModule,
        compute_pooled_event_time_probability_grids,
    )

    load_dotenv()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base="1.2"
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)
    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    available = sorted(outcome_configs)
    outcomes = _names(args.outcomes, available)
    horizons = [float(h) for h in args.horizons.split(",") if h.strip()]
    if (
        not outcomes
        or not horizons
        or any(not np.isfinite(h) or h <= 0 for h in horizons)
        or len(horizons) != len(set(horizons))
    ):
        raise ValueError("Choose nonempty outcomes and unique positive horizons")
    borrowed = (
        _names(args.borrowed_outcomes, available) if args.borrowed_outcomes else []
    )
    check_outcomes = _names(args.check_outcomes, available)
    # Log-odds for every outcome (comparable across outcomes and risk levels),
    # absolute risk for the check outcomes (clinical examples, removal checks).
    columns = [("logit", o, h) for o in outcomes for h in horizons]
    columns += [("risk", o, h) for o in check_outcomes if o in outcomes for h in horizons]
    columns += [("borrowed", o, None) for o in borrowed]
    check_columns = [
        ("risk", o, h) for o in check_outcomes if o in outcomes for h in horizons
    ]

    required = bool(cfg.training.get("require_all_configured_cells", True))
    time_grids, probability_grids = compute_pooled_event_time_probability_grids(
        cohort_configs,
        outcome_configs,
        split="train",
        require_all_configured_cells=required,
    )
    vocabulary = torch.load(
        cfg.paths.vocabulary, map_location="cpu", weights_only=False
    )
    hparams = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get(
        "hyper_parameters", {}
    )
    saved_outcomes = hparams.get("outcome_names")
    if saved_outcomes is not None and list(saved_outcomes) != available:
        raise ValueError(
            "Checkpoint outcome order differs from the supplied configuration"
        )
    saved_readout = hparams.get("model_init_config", {}).get(
        "competing_risk_config", {}
    )
    if saved_readout.get("fine_names") is not None and list(
        saved_readout["fine_names"]
    ) != list(cfg.competing_risk.get("fine_names") or []):
        raise ValueError(
            "Checkpoint fine-disease order differs from the supplied configuration"
        )
    encoder_config = get_saved_encoder_config(hparams)
    adapters = cfg.competing_risk.get("readout_mode") in {
        "source_anchored_adapters",
        "unanchored_adapters",
    }
    fine_names = list(cfg.competing_risk.get("fine_names") or []) if adapters else []
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs=cohort_configs,
        outcome_configs=outcome_configs,
        predict_token_id=int(vocabulary["[CLS]"]),
        batch_size=1,
        num_workers=0,
        require_all_configured_cells=required,
        require_min_followup_val=False,
        max_len=int(encoder_config["max_seqlen"]),
        numeric_value_control=resolve_numeric_value_control(
            cfg.training.get("numeric_value_control", "inherit"), hparams
        ),
        fine_names=fine_names or None,
        fine_column=str(cfg.competing_risk.get("fine_column", "cohort_fine")),
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model = (
        _load_model(
            args.checkpoint,
            cfg,
            available,
            time_grids,
            probability_grids,
            device,
            execution_attn_type="sdpa",
        )
        .float()
        .eval()
    )
    if model.competing_risk_head is None:
        raise ValueError("Checkpoint has no direct competing-risk head")
    if borrowed and model.source_adapter_readout is None:
        raise ValueError("--borrowed-outcomes needs a routed checkpoint")

    explainer = Explainer(
        model,
        vocabulary,
        collate=contrastive_collate,
        columns=columns,
        check_columns=check_columns,
        steps=args.steps,
        precision=args.precision,
        token_budget=args.token_budget,
        tolerance=args.tolerance,
        baseline=args.baseline,
        ks=[int(k) for k in args.k.split(",") if k.strip()],
        random_repeats=args.random_repeats,
        deletion=not args.no_deletion,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    # Baseline: an average event, from training patients only.
    baseline_stats = {"patients": 0, "event_tokens": 0}

    def train_batches():
        for name, cohort_cfg in cohort_configs.items():
            dataset = data_module._build_dataset_for_cohort(name, cohort_cfg, "train")
            if dataset is None:
                continue
            take = rng.permutation(len(dataset))[
                : max(1, args.baseline_patients // len(cohort_configs))
            ]
            for i in take:
                batch = _move(contrastive_collate([dataset[int(i)]]), device)
                baseline_stats["patients"] += 1
                baseline_stats["event_tokens"] += int(
                    attributable_mask(batch, explainer.fixed).sum()
                )
                yield batch

    verify_intent(args, cfg, vocabulary)
    baseline_cache = args.output_dir / "baseline_INTERNAL.pt"
    baseline_digest = args.output_dir / "baseline_sha256.txt"
    if baseline_cache.exists() and baseline_digest.exists():
        if sha256(baseline_cache) != baseline_digest.read_text().strip():
            raise ValueError("Cached baseline changed")
        saved = torch.load(baseline_cache, map_location="cpu", weights_only=True)
        explainer.global_mean, explainer.source_means = saved["mean"], saved["sources"]
        baseline_stats = saved.get("statistics", baseline_stats)
    else:
        explainer.fit_baseline(train_batches())
        temporary = baseline_cache.with_suffix(".tmp")
        torch.save(
            {
                "mean": explainer.global_mean.cpu(),
                "sources": {k: v.cpu() for k, v in explainer.source_means.items()},
                "statistics": baseline_stats,
            },
            temporary,
        )
        os.replace(temporary, baseline_cache)
        baseline_digest.write_text(sha256(baseline_cache))

    pool = []
    for name, cohort_cfg in cohort_configs.items():
        dataset = data_module._build_dataset_for_cohort(name, cohort_cfg, args.split)
        if dataset is None:
            continue
        fine = getattr(dataset, "fine_index_by_subject", None) or {}
        for i in _eligible(dataset, outcomes):
            sid = dataset.subjects[i]["subject_id"]
            index = fine.get(sid, -1)
            subtype = fine_names[index] if 0 <= index < len(fine_names) else name
            pool.append(
                {"cohort": name, "dataset": dataset, "index": i, "subtype": subtype}
            )
    pool_ids = [p["dataset"].subjects[p["index"]]["subject_id"] for p in pool]
    if len(pool_ids) != len(set(pool_ids)):
        raise ValueError(
            "Patient appears in multiple attribution cohorts; resolve population membership"
        )
    chosen, weights = stratified_sample(
        pool, per_subtype=args.per_subtype, total=args.max_patients, seed=args.seed
    )
    sanity_models = (
        [
            ("random_head", randomized_copy(model, "head", args.seed)),
            ("random_model", randomized_copy(model, "model", args.seed)),
        ]
        if args.sanity_patients
        else []
    )
    order = rng.permutation(len(chosen))  # check patients are a random subset
    token_dir = args.output_dir / "token_attributions_INTERNAL"
    token_dir.mkdir(parents=True, exist_ok=True)
    tables = {"patients": [], "faithfulness": [], "sanity": []}
    shard, shard_index = [], 0
    new_patients = 0
    for rank, j in enumerate(order.tolist()):
        is_cached = (
            args.output_dir
            / "patient_cache_INTERNAL"
            / f"patient-{rank:06d}"
            / "complete.json"
        ).exists()
        if not is_cached:
            if (
                args.max_new_patients is not None
                and new_patients >= args.max_new_patients
            ):
                break
            new_patients += 1
        entry, weight = pool[chosen[j]], weights[j]
        sample = entry["dataset"][entry["index"]]
        meta = {
            "subject_id": int(sample["subject_id"]),
            "cohort": entry["cohort"],
            "subtype": entry["subtype"],
            "weight": float(weight),
        }
        check = rank < args.sanity_patients

        def compute():
            rows = explainer.explain(
                sample,
                meta=meta,
                sanity_models=sanity_models if check else (),
                alternative_baseline=(args.alternative_baseline or None)
                if check
                else None,
                run_checks=rank < args.faithfulness_patients,
            )
            return rows

        rows = cached_explanation(args.output_dir, rank, meta, compute)
        shard += rows["tokens"]
        for key in tables:
            tables[key] += rows.get(key, [])
        if len(shard) >= args.shard_size:
            pd.concat(shard).to_parquet(
                token_dir / f"part-{shard_index:05d}.parquet", index=False
            )
            shard, shard_index = [], shard_index + 1
        if (rank + 1) % 50 == 0:
            print(f"explained {rank + 1}/{len(order)} patients", flush=True)
    if shard:
        pd.concat(shard).to_parquet(
            token_dir / f"part-{shard_index:05d}.parquet", index=False
        )
    names = {
        "patients": "patient_attributions",
        "faithfulness": "faithfulness",
        "sanity": "sanity",
    }
    for key, rows in tables.items():
        pd.DataFrame(rows).to_parquet(
            args.output_dir / f"{names[key]}_INTERNAL.parquet", index=False
        )
    (args.output_dir / "lab_bin_counts.json").write_text(
        json.dumps(bin_counts(vocabulary), indent=2, sort_keys=True)
    )
    (args.output_dir / "attribution_run.json").write_text(
        json.dumps(
            {
                "output_sha256": {
                    str(p.relative_to(args.output_dir)): sha256(p)
                    for p in [
                        *args.output_dir.glob("*_INTERNAL.parquet"),
                        *token_dir.glob("part-*.parquet"),
                    ]
                },
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256(args.checkpoint),
                "execution_backend": "sdpa_float32",
                "config_name": args.config_name,
                "split": args.split,
                "columns": [column_name(*c) for c in columns],
                "check_columns": [column_name(*c) for c in check_columns],
                "n_patients": len({r["subject_id"] for r in tables["patients"]}),
                "planned_patients": int(len(chosen)),
                "complete": len(
                    list(
                        (args.output_dir / "patient_cache_INTERNAL").glob(
                            "patient-*/complete.json"
                        )
                    )
                )
                == len(chosen),
                "cost": _cost(args.output_dir, len(chosen)),
                "per_subtype": args.per_subtype,
                "steps": args.steps,
                "integration": "gauss-legendre",
                "precision": args.precision,
                "tolerance": args.tolerance,
                "baseline": args.baseline,
                "baseline_statistics": baseline_stats,
                "alternative_baseline": args.alternative_baseline,
                "k": args.k,
                "random_repeats": args.random_repeats,
                "deletion": not args.no_deletion,
                "sanity_patients": int(min(args.sanity_patients, len(chosen))),
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n"
    )


def _cost(output_dir, planned):
    """Measured GPU seconds per patient so far, and the projection for the plan.

    A pilot (``--max-new-patients 50``) therefore answers how long the full
    sample will take before committing to it.
    """
    seconds = []
    for marker in (Path(output_dir) / "patient_cache_INTERNAL").glob("patient-*/complete.json"):
        value = json.loads(marker.read_text()).get("wall_seconds")
        if value is not None:
            seconds.append(float(value))
    if not seconds:
        return {}
    seconds = np.asarray(seconds)
    return {"patients_timed": int(len(seconds)),
            "median_seconds_per_patient": float(np.median(seconds)),
            "p90_seconds_per_patient": float(np.quantile(seconds, 0.9)),
            "projected_hours_for_plan": float(seconds.mean() * planned / 3600)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def implementation_files():
    root = Path(__file__).resolve().parents[2]
    explicit = [
        Path(__file__).resolve(),
        root / "opera/interpretability/token_attribution.py",
        root / "opera/evaluation/input_restriction.py",
        root / "opera/diagnostics/representation_gradient_conflict.py",
    ]
    paths = list((root / "bonsai/modules/networks").rglob("*.py"))
    paths += list((root / "opera/modules/networks").rglob("*.py"))
    paths += list((root / "opera/modules/datamodules").rglob("*.py"))
    return sorted(set(explicit + paths))


def verify_intent(args, cfg, vocabulary):
    from omegaconf import OmegaConf

    settings = {
        k: v
        for k, v in vars(args).items()
        if k
        not in {
            "resume",
            "device",
            "token_budget",
            "shard_size",
            "output_dir",
            "max_new_patients",
        }
    }

    def referenced_files(tree):
        if isinstance(tree, dict):
            for value in tree.values():
                yield from referenced_files(value)
        elif isinstance(tree, list):
            for value in tree:
                yield from referenced_files(value)
        elif isinstance(tree, str) and len(tree) < 4096:
            path = Path(tree)
            try:
                if path.is_file():
                    yield path.resolve()
            except OSError:
                return

    resolved = OmegaConf.to_container(cfg, resolve=True)
    identity = {
        "torch_version": torch.__version__,
        "settings": json.loads(json.dumps(settings, default=str)),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": resolved,
        "data_sha256": {
            str(p): sha256(p) for p in sorted(set(referenced_files(resolved)))
        },
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(vocabulary, sort_keys=True).encode()
        ).hexdigest(),
        "implementation_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
            for p in implementation_files()
        },
    }

    path = args.output_dir / "attribution_intent.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Attribution inputs/settings changed; use a fresh output directory"
            )
    else:
        path.write_text(json.dumps(identity, indent=2))


def cached_explanation(directory, rank, meta, compute):
    cache = Path(directory) / "patient_cache_INTERNAL" / f"patient-{rank:06d}"
    marker = cache / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["meta"] != meta:
            raise ValueError("Cached attribution patient roster changed")
        for name, digest in record["hashes"].items():
            if sha256(cache / name) != digest:
                raise ValueError("Cached attribution export changed")
        result = {}
        for file_name in record["hashes"]:
            key = Path(file_name).stem
            frame = pd.read_parquet(cache / (key + ".parquet"))
            result[key] = (
                [frame]
                if key == "tokens" and len(frame)
                else ([] if key == "tokens" else frame.to_dict("records"))
            )
        return result
    started = time.perf_counter()
    result = compute()
    elapsed = time.perf_counter() - started
    cache.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for key, rows in result.items():
        frame = (
            pd.concat(rows, ignore_index=True)
            if key == "tokens" and rows
            else pd.DataFrame(rows)
        )
        file = cache / (key + ".parquet")
        frame.to_parquet(file, index=False)
        hashes[file.name] = sha256(file)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"meta": meta, "hashes": hashes, "wall_seconds": elapsed})
    )
    os.replace(temporary, marker)
    return result


def main():
    import fcntl

    args = build_parser().parse_args()
    if (
        args.max_patients < 1
        or args.per_subtype < 0
        or args.baseline_patients < 1
        or args.shard_size < 1
        or min(args.sanity_patients, args.faithfulness_patients)
        < 0
    ):
        raise ValueError("Invalid patient, baseline, check or shard budget")
    if args.max_new_patients is not None and args.max_new_patients < 1:
        raise ValueError("max-new-patients must be positive")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("Use --resume for an existing attribution directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".attribution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(args)


if __name__ == "__main__":
    main()
