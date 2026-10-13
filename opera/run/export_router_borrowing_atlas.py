"""Stream a tuning-only, all-task router borrowing atlas without patient-row exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bonsai.functional.checkpointing import get_saved_encoder_config
from bonsai.functional.input_contract import resolve_numeric_value_control
from opera.diagnostics.representation_gradient_conflict import _load_model
from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
    MultiCohortContrastiveDataModule,
    compute_pooled_event_time_probability_grids,
)
from opera.run.evaluate_direct_competing_risk import _device, _move
from opera.visualization.style import setup_style


setup_style()


class RouteAccumulator:
    """Patient-balanced means for each fine disease × eligible target × source."""

    def __init__(self, n_disease: int, n_outcome: int):
        self.n_disease = n_disease
        self.n_outcome = n_outcome
        self.n_source = 1 + n_disease + n_outcome
        self.counts = np.zeros((n_disease, n_outcome), dtype=np.int64)
        self.sums = np.zeros((n_disease, n_outcome, self.n_source), dtype=np.float64)
        self.sumsq = np.zeros_like(self.sums)
        self.unknown_patients = 0
        self.noneligible_target_rows = 0

    def update(self, weights: torch.Tensor, fine_index: torch.Tensor,
               eligible: torch.Tensor) -> None:
        if weights.ndim != 3 or weights.shape[1:] != (self.n_outcome, self.n_source):
            raise ValueError("Route tensor has the wrong target/source shape")
        if fine_index.shape != (len(weights),) or eligible.shape != weights.shape[:2]:
            raise ValueError("Fine identity or target eligibility shape differs from routes")
        if not torch.isfinite(weights).all() or not torch.allclose(
            weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-3
        ):
            raise ValueError("Router weights must be finite and sum to one")
        w = weights.detach().float().cpu().numpy()
        fine = fine_index.detach().long().cpu().numpy()
        valid = eligible.detach().bool().cpu().numpy()
        self.unknown_patients += int(((fine < 0) | (fine >= self.n_disease)).sum())
        self.noneligible_target_rows += int((~valid).sum())
        for d in np.unique(fine):
            if d < 0 or d >= self.n_disease:
                continue
            selected = fine == d
            mask = valid[selected].astype(np.float64)
            values = w[selected].astype(np.float64)
            self.counts[d] += mask.sum(axis=0).astype(np.int64)
            self.sums[d] += np.einsum("bo,bos->os", mask, values)
            self.sumsq[d] += np.einsum("bo,bos->os", mask, values * values)

    def mean(self) -> np.ndarray:
        return np.divide(self.sums, self.counts[..., None],
                         out=np.full_like(self.sums, np.nan),
                         where=self.counts[..., None] > 0)


def budget_matrices(means: np.ndarray, counts: np.ndarray, *, min_patients: int) -> dict[str, np.ndarray]:
    n_disease, n_outcome, n_source = means.shape
    if n_source != 1 + n_disease + n_outcome or counts.shape != (n_disease, n_outcome):
        raise ValueError("Route atlas dimensions are inconsistent")
    disease = means[:, :, 1:1 + n_disease]
    outcome = means[:, :, 1 + n_disease:]
    own_d = np.stack([disease[d, :, d] for d in range(n_disease)])
    own_o = np.stack([outcome[:, o, o] for o in range(n_outcome)], axis=1)
    result = {
        "shared": means[:, :, 0].copy(),
        "own_disease": own_d,
        "borrowed_disease": disease.sum(axis=2) - own_d,
        "own_outcome": own_o,
        "borrowed_outcome": outcome.sum(axis=2) - own_o,
    }
    for value in result.values():
        value[counts < min_patients] = np.nan
    return result


def source_flows(means: np.ndarray, counts: np.ndarray, families: dict[str, list[str]],
                 outcome_names: list[str], *, min_patients: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-task means; outcomes in a source family count per adapter."""
    n_disease, n_outcome, _ = means.shape
    valid = counts >= min_patients
    dflow = np.full((n_disease, n_disease), np.nan)
    for target_d in range(n_disease):
        for source_d in range(n_disease):
            if target_d != source_d:
                values = means[target_d, valid[target_d], 1 + source_d]
                if values.size:
                    dflow[target_d, source_d] = float(values.mean())
    by_outcome = {name: i for i, name in enumerate(outcome_names)}
    family_names = list(families)
    fflow = np.full((len(family_names), len(family_names)), np.nan)
    for target_f, target_family in enumerate(family_names):
        for source_f, source_family in enumerate(family_names):
            cell_values = []
            source_indices = [by_outcome[name] for name in families[source_family]]
            for target_name in families[target_family]:
                target_o = by_outcome[target_name]
                off_home = [s for s in source_indices if s != target_o]
                if not off_home:
                    continue
                for d in range(n_disease):
                    if valid[d, target_o]:
                        cell_values.append(float(means[d, target_o,
                                                       1 + n_disease + np.array(off_home)].mean()))
            if cell_values:
                fflow[target_f, source_f] = float(np.mean(cell_values))
    return dflow, fflow


def _heatmap(ax, values: np.ndarray, ylabels: list[str], xlabels: list[str],
             title: str, vmax: float, *, cmap: str):
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax,
                      aspect="auto", interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=90, fontsize=5.3)
    ax.tick_params(length=0)
    return image


def plot_atlas(budgets: dict[str, np.ndarray], dflow: np.ndarray, fflow: np.ndarray,
               disease_names: list[str], outcome_names: list[str],
               family_names: list[str], output: Path) -> None:
    disease_borrow = budgets["borrowed_disease"]
    outcome_borrow = budgets["borrowed_outcome"]
    finite = np.concatenate([x[np.isfinite(x)] for x in (disease_borrow, outcome_borrow)])
    vmax = max(float(np.quantile(finite, .99)) if finite.size else 0, .01)
    fig, axes = plt.subplots(2, 1, figsize=(21, 15.5), sharex=True)
    for ax, values, title, cmap in (
        (axes[0], disease_borrow, "A  Borrowing from other disease adapters", "BuGn"),
        (axes[1], outcome_borrow, "B  Borrowing from other outcome adapters", "OrRd"),
    ):
        image = _heatmap(ax, values, disease_names, outcome_names, title, vmax, cmap=cmap)
        ax.set_ylabel("Patient's fine disease")
        fig.colorbar(image, ax=ax, fraction=.012, pad=.012, label="Mean off-home route weight")
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Target outcome (each cell pools eligible tuning patients)")
    fig.suptitle("A map of where the interacting router borrows", x=.08, y=.995,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.08, .015, "All configured fine diseases × outcomes; white = fewer than the minimum eligible tuning patients. "
             "Common color scale, capped at the 99th percentile. Gate mass is not predictive benefit.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.95, top=.955, bottom=.19, hspace=.25)
    fig.savefig(output / "router_full_task_borrowing_atlas.png", dpi=220)
    fig.savefig(output / "router_full_task_borrowing_atlas.pdf")
    plt.close(fig)

    finite_flow = np.concatenate([x[np.isfinite(x)] for x in (dflow, fflow)])
    flow_max = max(float(np.quantile(finite_flow, .99)) if finite_flow.size else 0, .01)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    image_d = _heatmap(axes[0], dflow, disease_names, disease_names,
                     "A  Disease → disease source preference", flow_max, cmap="BuGn")
    image_f = _heatmap(axes[1], fflow, family_names, family_names,
             "B  Outcome-family → outcome-family preference", flow_max, cmap="OrRd")
    for ax in axes:
        ax.set_xlabel("Source adapter (borrowed from)")
        ax.set_ylabel("Target task (borrows)")
    fig.colorbar(image_d, ax=axes[0], location="bottom", shrink=.66, pad=.19,
                 label="Mean off-home weight per disease adapter")
    fig.colorbar(image_f, ax=axes[1], location="bottom", shrink=.66, pad=.19,
                 label="Mean off-home weight per outcome adapter")
    fig.suptitle("Which sources receive the borrowing weight?", x=.07, y=.98,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(.07, .02, "Rows borrow from columns. Equal weight per eligible task cell; "
             "outcome-family entries are per source adapter. Home source excluded. "
             "Descriptive gate preference, not a performance effect.", fontsize=8)
    fig.subplots_adjust(left=.09, right=.97, top=.9, bottom=.22, wspace=.4)
    fig.savefig(output / "router_full_source_flow.png", dpi=220)
    fig.savefig(output / "router_full_source_flow.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-patients", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.batch_size < 1 or args.num_workers < 0 or args.min_patients < 1:
        raise ValueError("Batch size, workers or minimum patients are invalid")
    if args.config_dir.resolve().parent != args.checkpoint.resolve().parent:
        raise ValueError("Saved config directory and checkpoint must belong to the same run")
    load_dotenv()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base="1.2"):
        cfg = compose(config_name=args.config_name)
    scale = float(cfg.competing_risk.get("adapter_residual_scale", 0))
    if (cfg.competing_risk.get("readout_mode") not in {"source_anchored_adapters", "unanchored_adapters"}
            or cfg.competing_risk.get("adapter_query_mode") != "interacting"
            or not 0 < scale <= 1):
        raise ValueError("Full atlas requires an interacting source router with valid residual scale")
    cohorts = OmegaConf.to_container(cfg.cohorts, resolve=True)
    outcomes = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcomes)
    time_grids, probability_grids = compute_pooled_event_time_probability_grids(
        cohorts, outcomes, split="train",
        require_all_configured_cells=bool(cfg.training.get("require_all_configured_cells", True)),
    )
    vocabulary = torch.load(cfg.paths.vocabulary, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_hparams = checkpoint.get("hyper_parameters", {})
    encoder_config = get_saved_encoder_config(checkpoint_hparams)
    del checkpoint
    data = MultiCohortContrastiveDataModule(
        cohort_configs=cohorts, outcome_configs=outcomes,
        predict_token_id=int(vocabulary["[CLS]"]),
        batch_size=args.batch_size, num_workers=args.num_workers,
        require_all_configured_cells=bool(cfg.training.get("require_all_configured_cells", True)),
        require_min_followup_val=False, max_len=int(encoder_config["max_seqlen"]),
        numeric_value_control=resolve_numeric_value_control(
            cfg.training.get("numeric_value_control", "inherit"),
            checkpoint_hparams,
        ),
        fine_names=cfg.competing_risk.fine_names,
        fine_column=str(cfg.competing_risk.get("fine_column", "cohort_fine")),
    )
    device = _device(args.device)
    model = _load_model(args.checkpoint, cfg, outcome_names,
                        time_grids, probability_grids, device)
    readout = model.source_adapter_readout
    if readout is None or tuple(readout.outcome_names) != tuple(outcome_names):
        raise ValueError("Checkpoint router source inventory differs from config")
    disease_names = list(readout.fine_names)
    accumulator = RouteAccumulator(len(disease_names), len(outcome_names))
    seen_subject_ids: set[int] = set()
    model.eval()
    for cohort_name, cohort_cfg in cohorts.items():
        dataset = data._build_dataset_for_cohort(cohort_name, cohort_cfg, "tuning")
        if dataset is None:
            continue
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=args.num_workers,
                            collate_fn=contrastive_collate)
        n_cohort = 0
        with torch.inference_mode():
            for raw_batch in loader:
                ids = raw_batch["subject_id"].detach().cpu().tolist()
                duplicates = seen_subject_ids.intersection(ids)
                if duplicates or len(ids) != len(set(ids)):
                    raise ValueError("Patient appears more than once in tuning route atlas")
                seen_subject_ids.update(ids)
                batch = _move(raw_batch, device)
                representations = model.get_objective_representations(batch)
                weights = readout.route_weights(representations, batch["cohort_fine_index"])
                eligible = torch.stack(
                    [(batch[f"event_{name}"] >= 0) & (batch[f"time_{name}"] >= 0)
                     for name in outcome_names], dim=1,
                )
                accumulator.update(weights, batch["cohort_fine_index"], eligible)
                n_cohort += len(ids)
        print(f"Tuning route atlas: {cohort_name}: {n_cohort} patients", flush=True)
    if not seen_subject_ids or not accumulator.counts.any():
        raise ValueError("No eligible tuning patient-target cells were found")
    means = accumulator.mean()
    budgets = budget_matrices(means, accumulator.counts, min_patients=args.min_patients)
    families = dict(OmegaConf.to_container(cfg.cross_outcome.outcome_families, resolve=True))
    family_members = [name for members in families.values() for name in members]
    if sorted(family_members) != outcome_names:
        raise ValueError("Outcome families do not partition router targets")
    dflow, fflow = source_flows(means, accumulator.counts, families, outcome_names,
                               min_patients=args.min_patients)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "router_route_cube_INTERNAL.npz",
                        mean=means, count=accumulator.counts,
                        sumsq=accumulator.sumsq,
                        disease_names=np.array(disease_names), outcome_names=np.array(outcome_names),
                        source_names=np.array(["shared", *["disease:" + d for d in disease_names],
                                               *["outcome:" + o for o in outcome_names]]))
    for name, values in budgets.items():
        pd.DataFrame(values, index=disease_names, columns=outcome_names).to_csv(
            args.output_dir / f"{name}_by_task.csv")
    pd.DataFrame(accumulator.counts, index=disease_names, columns=outcome_names).to_csv(
        args.output_dir / "eligible_tuning_patients_by_task.csv")
    pd.DataFrame(dflow, index=disease_names, columns=disease_names).to_csv(
        args.output_dir / "disease_source_flow.csv")
    pd.DataFrame(fflow, index=list(families), columns=list(families)).to_csv(
        args.output_dir / "outcome_family_source_flow.csv")
    plot_atlas(budgets, dflow, fflow, disease_names, outcome_names,
               list(families), args.output_dir)
    report = {
        "status": "full_tuning_router_borrowing_atlas",
        "checkpoint": str(args.checkpoint.resolve()), "split": "tuning",
        "n_tuning_patients": len(seen_subject_ids),
        "n_target_cells_with_eligible_patients": int((accumulator.counts > 0).sum()),
        "n_target_cells_plotted": int((accumulator.counts >= args.min_patients).sum()),
        "n_eligible_patient_target_rows": int(accumulator.counts.sum()),
        "n_unknown_fine_patients": accumulator.unknown_patients,
        "min_patients_for_display": args.min_patients,
        "estimand": "Mean gate weight among eligible tuning patients for each fine disease × target outcome; "
                    "equal patient weight within cells; source-flow matrices equal-weight task cells.",
        "caveat": "Route preferences are not predictive effects, causal source contributions, "
                  "or uncertainty across model seeds. No held-out labels were loaded.",
    }
    (args.output_dir / "atlas_report_INTERNAL.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
