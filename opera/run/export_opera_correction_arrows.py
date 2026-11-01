"""Where each outcome moves a patient's representation (aggregates only; GPU or CPU).

The new OPERA reads the encoded record once into a shared summary (the CLS
representation drawn on the paper's patient map). Each outcome then adds what
its own attention query reads, gated by tanh(g), and own low-rank corrections
(diagnosis, outcome, global). This step places the shared summary and each
outcome's final representation on the *same* fitted map as the calibration maps
(``build_paper_embedding_tables``: standardisation, PCA and UMAP fitted on tuning
patients, applied unchanged; UMAP ``transform`` places the displaced vectors, as
for the retired router arrows) and writes only binned means:

* ``correction_arrows.csv``: per outcome and component (``total``: shared ->
  final; ``reading``: shared -> after reading; ``corrections``: after reading ->
  final), each map bin with at least ``--min-count`` (>= 20) patients: bin
  centre (mean position), mean displacement, patient count and coherence
  (|mean shift| / mean |shift|).
* ``correction_arrow_density.csv``: patient counts on a regular grid over the
  central 98% of the map, cells with fewer than ``--min-count`` patients
  omitted (the grey backdrop).
* ``correction_arrows.json``: projection method, patients, outcomes, settings.

No patient-level coordinates or vectors are written.

    python -m opera.run.export_opera_correction_arrows --checkpoint <best.ckpt> \\
        --config-name <run config> --embedding-config <embedding.yaml> \\
        --output-dir <analysis root>/figure_inputs/plot_tables [--outcomes a,b,c]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MIN_BIN_PATIENTS = 20
COMPONENTS = ("total", "reading", "corrections")


def representation_parts(model, batch, outcome_indices) -> dict:
    """Shared summary, after outcome-specific reading, and final representation
    of the chosen outcomes: (b, h), (b, k, h), (b, k, h)."""
    readout = getattr(model, "outcome_attention_readout", None)
    if readout is None or not hasattr(readout, "outcome_queries"):
        raise ValueError("Correction arrows need an outcome-attention (J_OA*) checkpoint")
    if getattr(model, "outcome_readout_mode", None) == "factorized_cohort_outcome_prototypes":
        raise ValueError("Factorised prototype readouts are not supported")
    hidden = model._encode_hidden(batch)
    shared = model._pool_hidden(hidden, batch)
    read = readout(hidden, shared, batch["attention_mask"])
    final = read
    if getattr(model, "attention_own_corrections", None) is not None:
        final = model.attention_own_corrections(read, batch["cohort_fine_index"])
    index = torch.as_tensor(list(outcome_indices), device=read.device, dtype=torch.long)
    return {"shared": shared.float(), "read": read[:, index].float(), "final": final[:, index].float()}


def arrow_aggregates(before, after: dict, *, bins=None, min_count=MIN_BIN_PATIENTS,
                     density_bins=None):
    """Binned mean displacement on the map; ``after`` maps (outcome, component)
    to start and end map positions ``(start, end)`` of the same patients, or to
    end positions starting at ``before``. Bins with fewer than ``min_count``
    patients are dropped; nothing patient-level is returned."""
    from opera.visualization.censoring_aware_embeddings import _binned_router_displacement

    if min_count < MIN_BIN_PATIENTS:
        raise ValueError(f"Aggregates need at least {MIN_BIN_PATIENTS} patients per bin")
    before = np.asarray(before, float)
    rows = []
    for (outcome, component), value in after.items():
        start, end = value if isinstance(value, tuple) else (before, value)
        # Bins are always defined on the shared map positions, so every
        # component of an outcome uses the same bins and centres.
        shift = np.asarray(end, float) - np.asarray(start, float)
        arrows = _binned_router_displacement(before, before + shift, bins=bins, min_count=min_count)
        for centre, vector, count, coherence in zip(arrows["centers"], arrows["vectors"],
                                                     arrows["counts"], arrows["coherence"]):
            rows.append({"outcome": outcome, "component": component,
                         "center_x": centre[0], "center_y": centre[1],
                         "dx": vector[0], "dy": vector[1], "count": int(count),
                         "coherence": coherence})
    arrows = pd.DataFrame(rows, columns=["outcome", "component", "center_x", "center_y",
                                         "dx", "dy", "count", "coherence"])
    if density_bins is None:  # fine enough to outline the map; sparse cells drop out
        density_bins = int(np.clip(np.sqrt(len(before) / 8), 10, 40))
    low = np.quantile(before, 0.01, axis=0)
    high = np.quantile(before, 0.99, axis=0)
    pad = 0.08 * (high - low)
    x_edges = np.linspace(low[0] - pad[0], high[0] + pad[0], density_bins + 1)
    y_edges = np.linspace(low[1] - pad[1], high[1] + pad[1], density_bins + 1)
    counts, _, _ = np.histogram2d(before[:, 0], before[:, 1], bins=[x_edges, y_edges])
    i, j = np.nonzero(counts >= min_count)
    density = pd.DataFrame({"x": (x_edges[i] + x_edges[i + 1]) / 2,
                            "y": (y_edges[j] + y_edges[j + 1]) / 2,
                            "width": np.diff(x_edges)[i], "height": np.diff(y_edges)[j],
                            "count": counts[i, j].astype(int)})
    if (arrows["count"] < min_count).any() or (density["count"] < min_count).any():
        raise AssertionError("small bins must never be written")
    return arrows, density


def project_parts(projection, parts: dict, outcomes) -> tuple[np.ndarray, dict]:
    """Map positions of the shared summary and of each outcome's representations."""
    before = projection.transform(parts["shared"])
    after = {}
    for k, outcome in enumerate(outcomes):
        read = projection.transform(parts["read"][:, k])
        final = projection.transform(parts["final"][:, k])
        after[(outcome, "total")] = (before, final)
        after[(outcome, "reading")] = (before, read)
        after[(outcome, "corrections")] = (read, final)
    return before, after


def _collect(args, outcomes, keep_ids):
    """Run the checkpoint over the split; keep the displayed patients' vectors
    in memory only."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from bonsai.functional.checkpointing import get_saved_encoder_config
    from bonsai.functional.input_contract import resolve_numeric_value_control
    from opera.diagnostics.representation_gradient_conflict import _load_model
    from opera.functional.direct_cr_contract import needs_fine_index
    from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
    from opera.modules.datamodules.MultiCohortContrastiveDataModule import (
        MultiCohortContrastiveDataModule, compute_pooled_event_time_probability_grids)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base="1.2"):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    cohort_configs = OmegaConf.to_container(cfg.cohorts, resolve=True)
    outcome_configs = OmegaConf.to_container(cfg.outcomes, resolve=True)
    outcome_names = sorted(outcome_configs)
    time_grids, probability_grids = compute_pooled_event_time_probability_grids(
        cohort_configs, outcome_configs, split="train",
        require_all_configured_cells=bool(cfg.training.get("require_all_configured_cells", True)))
    vocabulary = torch.load(cfg.paths.vocabulary, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    competing = OmegaConf.to_container(cfg.competing_risk, resolve=True)
    data_module = MultiCohortContrastiveDataModule(
        cohort_configs=cohort_configs, outcome_configs=outcome_configs,
        predict_token_id=int(vocabulary["[CLS]"]), batch_size=args.batch_size,
        num_workers=args.num_workers,
        require_all_configured_cells=bool(cfg.training.get("require_all_configured_cells", True)),
        require_min_followup_val=False,
        max_len=int(get_saved_encoder_config(hparams)["max_seqlen"]),
        numeric_value_control=resolve_numeric_value_control(
            cfg.training.get("numeric_value_control", "inherit"), hparams),
        fine_names=cfg.competing_risk.get("fine_names") if needs_fine_index(competing) else None,
        fine_column=str(cfg.competing_risk.get("fine_column", "cohort_fine")))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    model = _load_model(args.checkpoint, cfg, outcome_names, time_grids, probability_grids, device)
    model.eval()
    names = list(model.outcome_attention_readout.outcome_names)
    missing = [o for o in outcomes if o not in names]
    if missing:
        raise ValueError(f"Outcomes not in the checkpoint: {missing}")
    indices = [names.index(o) for o in outcomes]
    ids, parts = [], {"shared": [], "read": [], "final": []}
    checked = False
    for cohort_name, cohort_cfg in cohort_configs.items():
        dataset = data_module._build_dataset_for_cohort(cohort_name, cohort_cfg, args.data_split)
        if dataset is None:
            continue
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, collate_fn=contrastive_collate)
        with torch.inference_mode():
            for raw in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
                subject = np.asarray(batch["subject_id"].detach().cpu().numpy()).astype(str)
                wanted = np.isin(subject, list(keep_ids))
                if not wanted.any():
                    continue
                result = representation_parts(model, batch, indices)
                if not checked:  # the decomposition must reproduce the model's own path
                    full = model.get_objective_representations(batch)[:, indices].float()
                    if not torch.allclose(full, result["final"], atol=1e-4, rtol=1e-3):
                        raise ValueError("Decomposed representation does not match the model")
                    checked = True
                ids.append(subject[wanted])
                for key, value in result.items():
                    parts[key].append(value[torch.as_tensor(wanted, device=value.device)].cpu().numpy())
    if not ids:
        raise ValueError("No displayed patients found in the evaluation split")
    ids = np.concatenate(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("A patient appears twice across cohorts")
    return ids, {key: np.concatenate(value) for key, value in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--embedding-config", type=Path, required=True,
                        help="The build_paper_embedding_tables config: its calibration-map "
                             "stage is the map, refitted here exactly as there.")
    parser.add_argument("--outcomes", help="Comma-separated registry outcomes (default: the "
                                           "calibration-map outcomes of the embedding config)")
    parser.add_argument("--rehearse-on-tuning", action="store_true",
                        help="Tuning-only exports: fit on half the tuning patients, show the other half")
    parser.add_argument("--min-count", type=int, default=MIN_BIN_PATIENTS)
    parser.add_argument("--bins", type=int, help="Arrow bins per axis (default as the router arrows)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import yaml

    from opera.run.build_paper_embedding_tables import fit_projection, load_stage

    if args.min_count < MIN_BIN_PATIENTS:
        raise ValueError(f"--min-count must be at least {MIN_BIN_PATIENTS}")
    config = yaml.safe_load(args.embedding_config.read_text())
    maps = config["calibration_maps"]
    outcomes = args.outcomes.split(",") if args.outcomes else list(maps["outcomes"])
    frame = load_stage(config["stages"][maps["stage"]])
    projection, shown = fit_projection(frame, method=config.get("method", "umap"),
                                       rehearse=args.rehearse_on_tuning)
    args.data_split = "tuning" if args.rehearse_on_tuning else "held_out"
    ids, parts = _collect(args, outcomes, set(shown.subject_id.astype(str)))
    reference = shown.set_index(shown.subject_id.astype(str)).loc[ids]
    if not np.allclose(parts["shared"], np.stack(reference.embedding), atol=1e-4, rtol=1e-3):
        raise ValueError("The map stage was not exported from this checkpoint "
                         "(shared representations differ); arrows need the same checkpoint")
    before, after = project_parts(projection, parts, outcomes)
    del parts
    arrows, density = arrow_aggregates(before, after, bins=args.bins, min_count=args.min_count)
    from opera.visualization.outcome_names import outcome_display_name

    arrows["outcome_id"] = arrows.outcome
    arrows["outcome"] = arrows.outcome.map(outcome_display_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrows.to_csv(args.output_dir / "correction_arrows.csv", index=False)
    density.to_csv(args.output_dir / "correction_arrow_density.csv", index=False)
    meta = {"checkpoint": str(args.checkpoint), "projection": projection.method,
            "map_stage": maps["stage"], "split": args.data_split, "n_patients": int(len(ids)),
            "outcomes": [outcome_display_name(o) for o in outcomes], "min_count": args.min_count,
            "components": list(COMPONENTS),
            "representation": "shared CLS summary -> + tanh(g) x outcome reading -> + own corrections"}
    (args.output_dir / "correction_arrows.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("projection", "n_patients", "outcomes")}, indent=2))


if __name__ == "__main__":
    main()
