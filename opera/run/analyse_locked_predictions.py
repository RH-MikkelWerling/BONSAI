"""Run secondary OPERA analyses from an explicit locked prediction inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opera.evaluation.operating_points import (
    DEFAULT_RULES,
    operating_points,
    suppress_small_counts,
)
from opera.evaluation.secondary_analysis import (
    CELL,
    ESTIMANDS,
    ROUTER_GATED_CONTRASTS,
    annotate_router_gate,
    audit_router_identity,
    calibration_summary,
    coverage_inventory,
    effect_summaries,
    disk_backed_draw_matrix,
    pool_cells_over_horizons,
    negative_transfer_strata,
    normalize_predictions,
    paired_analysis,
    panel_estimates,
    panel_subgroup_gains,
    patient_bootstrap,
    reliability_tables,
    route_summaries,
    router_null_equivalence,
    seed_averaged_effects,
    subgroup_labels,
    subgroup_patient_gains,
    tuning_selected_strategies,
)


DEFAULT_PANEL_ESTIMATE_GROUPS = {
    "encoder_origin": ["J_L", "R_L", "J_E", "R_E", "J_EA", "R_EA"],
}


def read_table(path):
    path = Path(path)
    # Sniff the format: some tables (patient metadata) are parquet whatever
    # the configured file name says.
    with path.open("rb") as handle:
        is_parquet = handle.read(4) == b"PAR1"
    return (
        pd.read_parquet(path)
        if is_parquet or path.suffix in {".parquet", ".pq"}
        else pd.read_csv(path, dtype={"subject_id": str})
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_artifact(specification, metadata=None, estimand="binary_known_status_unweighted", *,
                  allow_training=False):
    """Adapt native wide CIFs or explicitly keyed long prediction tables."""
    frame = read_table(specification["path"])
    frame["subject_id"] = frame.subject_id.astype(str)
    if specification.get("cohort_from_metadata", False):
        if metadata is None or "cohort_fine" not in metadata:
            raise ValueError("Pooled native exports need metadata.cohort_fine")
        frame = frame.merge(
            metadata[["subject_id", "cohort_fine"]].rename(
                columns={"cohort_fine": "analysis_cohort"}
            ),
            on="subject_id",
            how="left",
            validate="many_to_one",
        )
        if frame.analysis_cohort.isna().any():
            raise ValueError("Missing fine disease membership")
        frame["cohort"] = frame.pop("analysis_cohort")
    if specification.get("format") == "direct_cif":
        prefix = re.escape(specification.get("probability_prefix", "cif_"))
        columns = [column for column in frame if re.fullmatch(prefix + r"\d+d", column)]
        if specification.get("include_horizons_days"):
            wanted = {int(day) for day in specification["include_horizons_days"]}
            columns = [
                column
                for column in columns
                if int(re.search(r"(\d+)d$", column).group(1)) in wanted
            ]
            present = {
                int(re.search(r"(\d+)d$", column).group(1)) for column in columns
            }
            if present != wanted:
                raise ValueError(
                    "Native CIF artifact is missing a requested analysis horizon"
                )
        if not columns or not {"time_days", "event"} <= set(frame):
            raise ValueError("Native CIF export needs cif_*d, time_days and event")
        frame = frame.melt(
            id_vars=[column for column in frame if column not in columns],
            value_vars=columns,
            var_name="horizon",
            value_name="probability",
        )
        frame["horizon_days"] = (
            frame.pop("horizon").str.extract(r"(\d+)")[0].astype(int)
        )
        positive = (frame.event == 1) & (frame.time_days <= frame.horizon_days)
        frame["label"] = positive.astype(int)
        frame["known"] = (
            positive
            | (frame.time_days >= frame.horizon_days)
            | ((frame.event == 2) & (frame.time_days <= frame.horizon_days))
        )
    for key in (
        "model",
        "seed",
        "split",
        "cohort",
        "outcome",
        "horizon_days",
        "data_contract",
    ):
        if key in specification:
            if key in frame and not frame[key].eq(specification[key]).all():
                raise ValueError(f"Artifact declaration conflicts with rows: {key}")
            frame[key] = specification[key]
    return normalize_predictions(frame, estimand, allow_training=allow_training)


_READ_CONTEXT = {}


def _init_reader(metadata, estimand):
    _READ_CONTEXT.update(metadata=metadata, estimand=estimand)


def _read_one(specification):
    try:
        data, summary = read_artifact(
            specification, _READ_CONTEXT["metadata"], _READ_CONTEXT["estimand"])
    except ValueError as error:
        raise ValueError(f"{specification['path']}: {error}") from error
    return data, summary, file_hash(specification["path"])


def read_artifacts(specifications, metadata, estimand):
    """Read every artifact, in order; in parallel with OPERA_ANALYSIS_WORKERS."""
    import sys
    import time

    from opera.evaluation.secondary_analysis import analysis_workers

    started, total, workers = time.time(), len(specifications), analysis_workers()
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_reader,
                                   initargs=(metadata, estimand))
        results = pool.map(_read_one, specifications, chunksize=16)
    else:
        pool = None
        _init_reader(metadata, estimand)
        results = map(_read_one, specifications)
    try:
        for done, result in enumerate(results, 1):
            if done % 500 == 0 or done == total:
                print(f"reading predictions: {done}/{total} artifacts, "
                      f"{(time.time() - started) / 60:.1f} min", file=sys.stderr, flush=True)
            yield result
    finally:
        if pool is not None:
            pool.shutdown()


def json_safe(value):
    """Preserve YAML infinite bin edges in a strict JSON provenance manifest."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


# Router decomposition (lock addenda 2026-10-26), for every router-fix design
# whose fits were analysed: own corrections (home-only vs global-only),
# borrowing (routed vs home-only) and both (routed vs global-only).
ROUTER_DESIGNS = {"O": ("R_O_L", "RBH_O_L"), "AB": ("R_AB_L", "RBH_AB_L")}
GLOBAL_ONLY = "N_A_L"


# The configuration chosen on tuning data (router-fix rule, then the national
# arm) under one name, so every figure can show "the best model" against
# XGBoost without knowing which fit won. Set ``opera_model`` in the analysis
# config; absent or unavailable, no OPERA rows or contrasts are made.
OPERA_CONTRASTS = {"O1": "gain(OPERA,X)", "O2": "gain(OPERA,J_L)", "O3": "gain(OPERA,S_L)",
                   "O4": "gain(OPERA,D_L)", "O5": "gain(OPERA,O_L)",
                   "O6": "gain(OPERA,XP)", "O7": "gain(OPERA,LR)"}


# Sampling sensitivity (lock addendum 2026-10-27): IPW event sampling against
# natural sampling, same design and seed.
SAMPLING_SENSITIVITY = {"IPW_R": "gain(R_AB_IPW_L,R_AB_L)", "IPW_J": "gain(J_IPW_L,J_L)"}
ORIGINAL_ROUTER = {"R1_orig": "gain(R_ORIG_L,J_L)", "T1_orig": "gain(R_ORIG_L,X)",
                   "R_orig": "gain(R_ORIG_L,R_L)"}

# Partial-pooling router (lock amendment 2026-10-30), computed when its fits are
# present. R_L is the primary routed model (R_AB unless R_PP is selected).
# Task-routed candidates (31 October 2026), computed when their fits are present.
TASK_ROUTED = {
    "PT5_AB": "gain(R_PT5_L,R_L)",            # selection rule: R_PT5 against the primary router
    "PT3V_AB": "gain(R_PT3V_L,R_L)",          # selection rule: R_PT3V against the primary router
    "PT5_J": "gain(R_PT5_L,J_L)",             # against the joint model
    "PT3V_J": "gain(R_PT3V_L,J_L)",
    "PT5_FZ": "gain(R_PT5_L,R_PP_FROZEN_L)",  # learned routing (identical even-mix control)
    "FZ_AB": "gain(R_PP_FROZEN_L,R_L)",       # structural change, no learned routing
    "PT3V_PT5": "gain(R_PT3V_L,R_PT5_L)",     # between the candidates
    "NF_AB": "gain(R_AB_s43_L,R_L)",          # seed noise floor of the routed design
}

# Full own corrections (stage 8, 31 October, evening): a one-factor-at-a-time
# sweep around the anchor R_PF5E, computed when the fits are present. Each
# variant is compared with the primary router (_AB) and with the anchor (_E).
PF_VARIANTS = {"PF5": "R_PF5_L", "PF5B": "R_PF5B_L", "PF10E": "R_PF10E_L", "PF5EP": "R_PF5EP_L",
               "PF5EL": "R_PF5EL_L", "PF5ED": "R_PF5ED_L", "PF5EO": "R_PF5EO_L",
               "PFLR": "R_PF5E_LR10_L", "PFW0": "R_PF5E_W0_L", "PFR32": "R_PF5E_R32_L",
               "PHG": "R_PH_G_L", "PHGE": "R_PH_GE_L", "PHGEP": "R_PH_GE_PF5E_L",
               "PF5E43": "R_PF5E_s43_L", "PFLIVE": "R_PF5E_LIVE_L", "PL8": "R_PL8_L"}
FULL_OWN = {
    "PF5E_AB": "gain(R_PF5E_L,R_L)",                # selection rule: the anchor against the primary router
    "PF5E_J": "gain(R_PF5E_L,J_L)",                 # against the joint model
    "PF5E_FZ": "gain(R_PF5E_L,R_PF5E_FROZEN_L)",    # learned routing (identical frozen control)
    "PF5E_OWN": "gain(R_PF5E_L,R_PF_OFF_L)",        # borrowing on top of own corrections
    "PFOFF_RBHO": "gain(R_PF_OFF_L,RBH_O_L)",       # implementation check: should be ~0
    "PFOFF_AB": "gain(R_PF_OFF_L,R_L)",             # own-outside floor against the primary router
    "PFR32_OWN": "gain(R_PF5E_R32_L,R_PF_OFF_R32_L)",  # borrowing at rank 32
    "PFOFF32_OFF": "gain(R_PF_OFF_R32_L,R_PF_OFF_L)",  # capacity alone (own-only, rank 32 vs locked)
    "PHG_OFF": "gain(R_PH_G_L,R_PF_OFF_L)",         # lineage hierarchy on top of own corrections
    "PHGE_OFF": "gain(R_PH_GE_L,R_PF_OFF_L)",       # lineage + endpoint hierarchy
    "PHGEP_PHGE": "gain(R_PH_GE_PF5E_L,R_PH_GE_L)",  # routing on top of the hierarchy
    "PFLIVE_OFF": "gain(R_PF5E_LIVE_L,R_PF_OFF_L)",  # learned sharing on top of own corrections
    # Scarce data (10% of training patients; J_L_F10 is the same subset).
    "F10_OFF_J": "gain(R_PF_OFF_F10_L,J_L_F10)",     # own corrections when data are scarce
    "F10_PF5E_OFF": "gain(R_PF5E_F10_L,R_PF_OFF_F10_L)",  # routed borrowing when data are scarce
    "F10_PHGE_OFF": "gain(R_PH_GE_F10_L,R_PF_OFF_F10_L)",  # hierarchy when data are scarce
    "F10_LIVE_OFF": "gain(R_PF5E_LIVE_F10_L,R_PF_OFF_F10_L)",  # learned sharing when data are scarce
    "F10_PF5E_J": "gain(R_PF5E_F10_L,J_L_F10)",
    "PL8_OFF": "gain(R_PL8_L,R_PF_OFF_L)",           # learned grouping on top of own corrections
    "PL8_PHGE": "gain(R_PL8_L,R_PH_GE_L)",           # learned against known grouping
    "F10_PL8_OFF": "gain(R_PL8_F10_L,R_PF_OFF_F10_L)",
    "F10_PL8_PHGE": "gain(R_PL8_F10_L,R_PH_GE_F10_L)",
    **{f"{code}_AB": f"gain({model},R_L)" for code, model in PF_VARIANTS.items()},
    **{f"{code}_E": f"gain({model},R_PF5E_L)" for code, model in PF_VARIANTS.items()},
}

# Task-conditioned pooling (stage 9, 1 November 2026), when the fits are present.
TASK_POOLING = {
    "OA_J": "gain(J_OA_L,J_L)",    # one attention query per outcome, against the joint model
    "SP_J": "gain(J_SP_L,J_L)",    # shared attention maps mixed per outcome
    "FP_J": "gain(J_FP_L,J_L)",    # mixture by disease group x outcome
    "OA_AB": "gain(J_OA_L,R_L)",   # against the primary router
    "SP_AB": "gain(J_SP_L,R_L)",
    "FP_AB": "gain(J_FP_L,R_L)",
    "FP_OA": "gain(J_FP_L,J_OA_L)",
    "FPF_J": "gain(J_FPF_L,J_L)",            # mixture by diagnosis x outcome
    "FPF_FP": "gain(J_FPF_L,J_FP_L)",        # diagnosis against disease group
    "OAOWN_J": "gain(J_OA_OWN_L,J_L)",       # attention + own corrections
    "OAOWN_AB": "gain(J_OA_OWN_L,R_L)",
    "OAOWN_OA": "gain(J_OA_OWN_L,J_OA_L)",   # own corrections on top of attention
    # The ladder in the main text (1 November 2026): J -> own corrections ->
    # + own reading, and learned routing on top of own corrections.
    "OFF_J": "gain(R_PF_OFF_L,J_L)",         # own diagnosis and outcome corrections
    "OAOWN_OFF": "gain(J_OA_OWN_L,R_PF_OFF_L)",  # own reading on top of own corrections
    "AB_OFF": "gain(R_AB_L,R_PF_OFF_L)",     # learned routing on top of own corrections
    "OAOWNPH_OAOWN": "gain(J_OA_OWN_PH_L,J_OA_OWN_L)",  # lineage corrections on top
    "OAOWNPH_AB": "gain(J_OA_OWN_PH_L,R_L)",
    "FPFOWN_AB": "gain(J_FPF_OWN_L,R_L)",
    "FPFOWN_FPF": "gain(J_FPF_OWN_L,J_FPF_L)",
    "FPFPH_FPF": "gain(J_FPF_PH_L,J_FPF_L)",            # diagnosis terms tied to lineage
    "FPFPHOWN_AB": "gain(J_FPF_PH_OWN_PH_L,R_L)",
    "FPFPHOWN_FPFOWN": "gain(J_FPF_PH_OWN_PH_L,J_FPF_OWN_L)",
    "OALR_OA": "gain(J_OA_LR_L,J_OA_L)",                # outcomes tied through a shared query basis
    "OALROWN_OAOWN": "gain(J_OA_LR_OWN_L,J_OA_OWN_L)",
    "OALROWN_AB": "gain(J_OA_LR_OWN_L,R_L)",
}

# Encoder learning rate (stage 10, 1 November 2026), when the fits are present.
ENCODER_LR = {
    "ELR_J": "gain(J_ELR10_L,J_L)",          # joint model, encoder at the head's learning rate
    "ELR_AB": "gain(R_AB_ELR10_L,R_L)",      # primary router, encoder at the head's learning rate
    "ELR3_AB": "gain(R_AB_ELR3_L,R_L)",      # primary router, encoder at 0.3x
    "ELR_FPFPHOWN": "gain(J_FPF_PH_OWN_PH_ELR10_L,J_FPF_PH_OWN_PH_L)",  # stage 11
    "ELR_OAOWN": "gain(J_OA_OWN_ELR10_L,J_OA_OWN_L)",
    "ELR_FPFPHOWN_AB": "gain(J_FPF_PH_OWN_PH_ELR10_L,R_L)",
    "ELR_OAOWN_AB": "gain(J_OA_OWN_ELR10_L,R_L)",
}

# Stage 12 (1 November 2026): what the leading reading design is made of.
TASK_ABLATION = {
    "OAOWN_NF": "gain(J_OA_OWN_L,J_OA_OWN_s43_L)",              # seed noise floor of the design
    "OAOWN_R32": "gain(J_OA_OWN_R32_L,J_OA_OWN_L)",             # rank 32 vs 8 corrections
    "OAOWN_DIS": "gain(J_OA_OWN_L,J_OA_OWN_NODIS_L)",           # what the diagnosis correction adds
    "OAOWN_OUT": "gain(J_OA_OWN_L,J_OA_OWN_NOOUT_L)",           # what the outcome correction adds
    "F10_OAOWN_OFF": "gain(J_OA_OWN_F10_L,R_PF_OFF_F10_L)",     # reading at 10% vs corrections only
    "F10_OAOWN_PHGE": "gain(J_OA_OWN_F10_L,R_PH_GE_F10_L)",
}

# Stage 13 (1 November 2026): design changes to the leading reading design.
TASK_DESIGN = {
    "SW20_OAOWN": "gain(J_OA_OWN_SW20_L,J_OA_OWN_L)",        # cell shrinkage, tau 20
    "SW100_OAOWN": "gain(J_OA_OWN_SW100_L,J_OA_OWN_L)",      # cell shrinkage, tau 100
    "X_OAOWN": "gain(J_OA_OWN_X_L,J_OA_OWN_L)",              # diagnosis x outcome correction
    "XSW_OAOWN": "gain(J_OA_OWN_X_SW_L,J_OA_OWN_L)",
    "PE_OAOWN": "gain(J_OA_OWN_PE_L,J_OA_OWN_L)",            # endpoint-family corrections
    "PHX_OAOWN": "gain(J_OA_OWN_PH_X_L,J_OA_OWN_L)",
    "PEXSW_OAOWN": "gain(J_OA_OWN_PE_X_SW_L,J_OA_OWN_L)",
    "XSW_AB": "gain(J_OA_OWN_X_SW_L,R_L)",
    "PEXSW_AB": "gain(J_OA_OWN_PE_X_SW_L,R_L)",
    "MAX_OAOWN": "gain(J_OA_OWN_MAX_L,J_OA_OWN_L)",          # moonshot: everything + rank 32
    "MAX_AB": "gain(J_OA_OWN_MAX_L,R_L)",
}

# Stage 14 (1 November 2026): seed spread of the design and of J.
TASK_SEEDS = {
    "OAOWN_NF44": "gain(J_OA_OWN_L,J_OA_OWN_s44_L)",
    "OAOWN_NF45": "gain(J_OA_OWN_L,J_OA_OWN_s45_L)",
    "OAOWN_NF4345": "gain(J_OA_OWN_s43_L,J_OA_OWN_s45_L)",
    "J_NF": "gain(J_L,J_s43_L)",                              # seed noise floor of J
    "OAOWN_J_s43": "gain(J_OA_OWN_s43_L,J_s43_L)",            # the main gain on a second seed pair
    "OFF_J_s43": "gain(R_PF_OFF_s43_L,J_s43_L)",              # ladder steps on seed 43 (stage 15)
    "OAOWN_OFF_s43": "gain(J_OA_OWN_s43_L,R_PF_OFF_s43_L)",
    "AB_OFF_s43": "gain(R_AB_s43_L,R_PF_OFF_s43_L)",
    # Seed averages (lock amendment, 2 November 2026): OPERA over its four
    # seeds against each reference over its available seeds. The s42 fit stays
    # primary; these show where it sits among its seeds.
    "OAOWN4_J": "gain(J_OA_OWN_L|J_OA_OWN_s43_L|J_OA_OWN_s44_L|J_OA_OWN_s45_L,J_L|J_s43_L)",
    "OAOWN4_OFF": "gain(J_OA_OWN_L|J_OA_OWN_s43_L|J_OA_OWN_s44_L|J_OA_OWN_s45_L,R_PF_OFF_L|R_PF_OFF_s43_L)",
    "OAOWN4_AB": "gain(J_OA_OWN_L|J_OA_OWN_s43_L|J_OA_OWN_s44_L|J_OA_OWN_s45_L,R_AB_L|R_AB_s43_L)",
    "OAOWN4_X": "gain(J_OA_OWN_L|J_OA_OWN_s43_L|J_OA_OWN_s44_L|J_OA_OWN_s45_L,X)",
    # Runner-up on more seeds (stage 16): its seed check against R_AB, its own
    # seed difference, and the parsimony comparison on seed 43.
    "OAOWNPH_AB_s43": "gain(J_OA_OWN_PH_s43_L,R_AB_s43_L)",
    "OAOWNPH_OAOWN_s43": "gain(J_OA_OWN_PH_s43_L,J_OA_OWN_s43_L)",
    "OAOWNPH_NF": "gain(J_OA_OWN_PH_L,J_OA_OWN_PH_s43_L)",
    "OAOWNPH_OAOWN_s44": "gain(J_OA_OWN_PH_s44_L,J_OA_OWN_s44_L)",
    "OAOWN4_S42": "gain(J_OA_OWN_L,J_OA_OWN_L|J_OA_OWN_s43_L|J_OA_OWN_s44_L|J_OA_OWN_s45_L)",  # s42 vs its seed mean
}

# The design search (supplementary figure, 1 November 2026): every tuned design
# against the selected reading design (J_OA_OWN) and against the primary router
# (R_L = R_AB), computed for the fits present. Contrast names are
# "DS_<model>__OAOWN" and "DS_<model>__AB"; display labels and groups live in
# build_paper_plot_tables.DESIGN_LABELS. The seed repeats give the seed band.
# ``design_search: false`` in the analysis config switches the family off.
DESIGN_SEARCH_REFERENCES = {"OAOWN": "J_OA_OWN_L", "AB": "R_L"}
DESIGN_SEARCH_MODELS = (
    # Learned routing and its controls (stages 3-8)
    "N_A_L", "RBH_O_L", "RBH_AB_L", "R_PF_OFF_L", "R_O_L", "R_L", "R_ORIG_L",
    "R_PP_L", "R_PP_FROZEN_L", "R_PT5_L", "R_PT3V_L", "R_PF5E_L", "R_PF5_L",
    "R_PF5B_L", "R_PF10E_L", "R_PF5EP_L", "R_PF5EL_L", "R_PF5ED_L", "R_PF5EO_L",
    "R_PF5E_LR10_L", "R_PF5E_W0_L", "R_PF5E_LIVE_L", "R_PF5E_FROZEN_L",
    # Known and learned structure (stage 8)
    "R_PH_G_L", "R_PH_GE_L", "R_PH_GE_PF5E_L", "R_PL8_L",
    # Capacity
    "R_PF_OFF_R32_L", "R_PF5E_R32_L", "J_OA_OWN_R32_L",
    # Readouts (stage 9)
    "J_L", "J_OA_L", "J_OA_LR_L", "J_SP_L", "J_FP_L", "J_FPF_L", "J_FPF_PH_L",
    # Reading + own corrections (stages 9 and 12)
    "J_OA_OWN_L", "J_OA_OWN_PH_L", "J_OA_LR_OWN_L", "J_FPF_OWN_L", "J_FPF_PH_OWN_PH_L",
    "J_OA_OWN_NODIS_L", "J_OA_OWN_NOOUT_L",
    # Shrinkage, cross and endpoint-family terms (stage 13)
    "J_OA_OWN_SW20_L", "J_OA_OWN_SW100_L", "J_OA_OWN_X_L", "J_OA_OWN_X_SW_L",
    "J_OA_OWN_PE_L", "J_OA_OWN_PH_X_L", "J_OA_OWN_PE_X_SW_L", "J_OA_OWN_MAX_L",
    # Encoder learning rate (stages 10-11)
    "J_ELR10_L", "R_AB_ELR10_L", "R_AB_ELR3_L", "J_OA_OWN_ELR10_L",
    "J_FPF_PH_OWN_PH_ELR10_L",
    # Seed repeats
    "J_OA_OWN_s43_L", "J_OA_OWN_s44_L", "J_OA_OWN_s45_L", "R_AB_s43_L", "J_s43_L",
    "R_PF5E_s43_L", "R_PF_OFF_s43_L",
)


def design_search_contrasts(models) -> dict:
    """``DS_<model>__<reference>`` for every design and reference present."""
    models = set(models)
    contrasts = {}
    for tag, reference in DESIGN_SEARCH_REFERENCES.items():
        if reference not in models:
            continue
        for model in DESIGN_SEARCH_MODELS:
            if model in models and model != reference:
                contrasts[f"DS_{model}__{tag}"] = f"gain({model},{reference})"
    return contrasts


PARTIAL_POOLING = {
    "PP1": "gain(R_PP_L,R_L)",            # selection rule: R_PP against the primary router
    "PP2": "gain(R_PP_L,R_PP_FROZEN_L)",  # gain from learned routing
    "PP3": "gain(R_PP_FROZEN_L,R_L)",     # gain from the structural changes
    "PP4": "gain(R_PP_L,J_L)",            # against the joint model
    "PP0": "gain(R_PP_L,R_PP_s43_L)",     # seed noise floor
}


def with_opera_alias(predictions, model):
    if not model or model not in set(predictions.model):
        return predictions, {}
    alias = predictions.loc[predictions.model == model].assign(model="OPERA")
    present = set(predictions.model) | {"OPERA"}
    contrasts = {name: expression for name, expression in OPERA_CONTRASTS.items()
                 if expression[5:-1].split(",")[1] in present}
    # Encoder identity follows the frozen selected alias, never test performance.
    origin = "EA" if model.endswith("_EA") else "E" if model.endswith("_E") else "L" if model.endswith("_L") else None
    if origin:
        references = {"O2_MATCHED": f"J_{origin}",
                      "O_READ_MATCHED": f"R_PF_OFF_{origin}",
                      "O_ROUTING_MATCHED": f"R_AB_{origin}"}
        if model == "J_OA_OWN_L":
            references["O_CORRECTIONS_MATCHED"] = "J_OA_L"
        for name, reference in references.items():
            if reference in present and reference != model:
                contrasts[name] = f"gain(OPERA,{reference})"
    if {"J_OA_OWN_L", "J_L"} <= present:
        contrasts["O2_LOCAL"] = "gain(J_OA_OWN_L,J_L)"
    return pd.concat([predictions, alias], ignore_index=True), contrasts


ALWAYS_ANALYSED = {"J_L", "R_L", "S_L", "D_L", "O_L", "X", "XP", "XR", "LR", "OPERA",
                   "BEST_TUNING"}


def analysed_models_only(predictions, contrasts, config, candidates=()):
    """Drop fits no part of the analysis refers to (e.g. router variants that only
    the router decision used): every estimate, draw and calibration fit is per
    model, so they cost time and change nothing. ``keep_all_models: true`` keeps
    them. Returns the predictions and the dropped model names."""
    if config.get("keep_all_models"):
        return predictions, set()
    used = set(ALWAYS_ANALYSED) | set(candidates) | models_for(contrasts)
    for members in panel_groups(config).values():
        used |= set(members)
    for spec in (config.get("stratified_reanalysis") or {}).values():
        used |= set(spec.get("models", []))
    for pair in (config.get("operating_points") or {}).get(
            "pairs", [["J_L", "X"], ["J_L", "S_L"], ["R_L", "J_L"]]):
        used |= set(pair)
    present = set(predictions.model)
    unused = present - used
    if unused:
        print(f"models not analysed (no contrast or summary uses them): {sorted(unused)}",
              flush=True)
        # Boolean selection already creates the selected table. A subsequent
        # deep copy consolidates all object columns and doubles peak memory.
        predictions = predictions.loc[predictions.model.isin(used)]
    return predictions, unused


def router_decomposition(models):
    contrasts = {}
    for tag, (routed, home) in ROUTER_DESIGNS.items():
        if {routed, home, GLOBAL_ONLY} <= set(models):
            contrasts |= {f"router_own_{tag}": f"gain({home},{GLOBAL_ONLY})",
                          f"router_borrow_{tag}": f"gain({routed},{home})",
                          f"router_both_{tag}": f"gain({routed},{GLOBAL_ONLY})"}
    return contrasts


def read_route_artifact(config, metadata):
    """Accept long gates or the native JSON-bank route export."""
    frame = read_table(config["route_weights"])
    if "shared_weight" not in frame:
        return frame
    if metadata is None or "cohort_fine" not in metadata:
        raise ValueError("Native route exports need subject-to-fine metadata")
    frame["subject_id"] = frame.subject_id.astype(str)
    frame = frame.merge(
        metadata[["subject_id", "cohort_fine"]].rename(
            columns={"cohort_fine": "analysis_cohort"}
        ),
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    if frame.analysis_cohort.isna().any():
        raise ValueError("Native route export has unknown fine membership")
    rows = []
    for record in frame.to_dict("records"):
        identity = {key: record[key] for key in ("subject_id", "split", "outcome")}
        identity.update(
            model=config["route_model"],
            seed=config["route_seed"],
            cohort=record["analysis_cohort"],
        )
        rows.append(
            {
                **identity,
                "source": "shared",
                "bank": "shared",
                "weight": record["shared_weight"],
            }
        )
        for bank in ("disease", "outcome"):
            for source, weight in json.loads(record[bank + "_weights"]).items():
                rows.append(
                    {**identity, "source": source, "bank": bank, "weight": weight}
                )
    return pd.DataFrame(rows)


_STARTED = None


def stage(name):
    """Progress line with elapsed time (stderr), so a slow stage is visible."""
    import sys
    import time

    global _STARTED
    _STARTED = _STARTED or time.time()
    print(f"analysis stage: {name} ({(time.time() - _STARTED) / 60:.1f} min)",
          file=sys.stderr, flush=True)


# National arm of the secondary model (lock, 3 November): learned routing
# (R_AB) on the national encoder against its local fit and XGBoost.
NATIONAL_ROUTING = {
    "E2_AB": "gain(R_AB_E,R_AB_L)", "E4_AB": "gain(R_AB_EA,R_AB_L)",
    "A2_AB": "gain(R_AB_EA,R_AB_E)", "T5E_AB": "gain(R_AB_E,X)", "T5_AB": "gain(R_AB_EA,X)",
    "I1_AB": "gain(R_AB_E,J_E)-gain(R_AB_L,J_L)", "I3_AB": "gain(R_AB_EA,J_EA)-gain(R_AB_L,J_L)",
    # OPERA against learned routing on the same national encoder.
    "OAOWN_AB_E": "gain(R_E,R_AB_E)", "OAOWN_AB_EA": "gain(R_EA,R_AB_EA)",
    # National mechanism ladder (3 Nov): the same steps as Table 2 on each
    # national encoder, the ablations on the adapted one, and whether each
    # step grows from the cohort to the adapted national encoder.
    "OFF_J_E": "gain(R_PF_OFF_E,J_E)", "OAOWN_OFF_E": "gain(R_E,R_PF_OFF_E)",
    "AB_OFF_E": "gain(R_AB_E,R_PF_OFF_E)",
    "OFF_J_EA": "gain(R_PF_OFF_EA,J_EA)", "OAOWN_OFF_EA": "gain(R_EA,R_PF_OFF_EA)",
    "AB_OFF_EA": "gain(R_AB_EA,R_PF_OFF_EA)",
    "OAOWN_DIS_EA": "gain(R_EA,J_OA_OWN_NODIS_EA)", "OAOWN_OUT_EA": "gain(R_EA,J_OA_OWN_NOOUT_EA)",
    "OFF_GROWTH_EA": "gain(R_PF_OFF_EA,J_EA)-gain(R_PF_OFF_L,J_L)",
    "READ_GROWTH_EA": "gain(R_EA,R_PF_OFF_EA)-gain(J_OA_OWN_L,R_PF_OFF_L)",
}
# Contrasts of the 14 October manifest whose local reference R_L stands for the
# primary OPERA: with a reading-design primary, the national OPERA (R_E, R_EA)
# carries that design, while R_L stays learned routing (lock, section 1).
NATIONAL_OPERA_CONTRASTS = ("E2", "E4", "I1", "I3")


def national_counterparts(contrasts, opera_model):
    """Point the national-vs-local OPERA contrasts at the local primary design."""
    local_model = "J_OA_OWN_L" if opera_model in {"R_E", "R_EA"} else opera_model
    if not local_model or not str(local_model).startswith("J_"):
        return contrasts
    return {name: (expression.replace("R_L)", f"{local_model})").replace("(R_L,", f"({local_model},")
                   if name in NATIONAL_OPERA_CONTRASTS else expression)
            for name, expression in contrasts.items()}


def panel_groups(config):
    """Encoder-origin panels: the local OPERA is the primary design (as in the
    national contrasts), and learned routing gets its own row group."""
    groups = dict(config.get("panel_estimate_groups", DEFAULT_PANEL_ESTIMATE_GROUPS))
    opera = config.get("opera_model")
    opera = "J_OA_OWN_L" if opera in {"R_E", "R_EA"} else opera
    if opera and str(opera).startswith("J_") and "encoder_origin" in groups:
        groups["encoder_origin"] = [opera if m == "R_L" else m for m in groups["encoder_origin"]]
        groups.setdefault("encoder_origin_routing", ["R_AB_L", "R_AB_E", "R_AB_EA"])
    return groups


def collapse_seed_repeats(predictions, config):
    """Put every model on the primary seed's key when each model has one seed.

    Cells are keyed by training seed, so a contrast between a seed-42 fit and a
    seed-43 fit (seed spread, noise floors, seed averages) would find no cell.
    Seed repeats are distinct models by name (``J_OA_OWN_s43_L``), patients and
    labels do not depend on the seed, and the patient bootstrap is shared across
    seeds, so one key changes no estimate; it only lets such contrasts pair. A
    model fitted on several seeds under one name keeps the per-seed keys.
    """
    if not config.get("collapse_seed_repeats", True) or predictions.seed.nunique() < 2:
        return predictions
    if (predictions.groupby("model").seed.nunique() > 1).any():
        return predictions
    primary = int(config.get("primary_seed", 42))
    result = predictions.copy(deep=False)
    result["seed"] = primary
    return result


def models_for(contrasts, extra=()):
    """Models a set of contrasts (and any extra summaries) needs: a restricted
    re-analysis evaluates only these, which gives the same contrasts."""
    from opera.evaluation.secondary_analysis import contrast_coefficients

    needed = set(extra)
    for expression in contrasts.values():
        needed |= set(contrast_coefficients(expression))
    return needed


def prepare_analysis_predictions(config, estimand):
    """Validate/normalize once; context is restartable inside the secure workspace."""
    metadata = (
        read_table(config["patient_metadata"])
        if config.get("patient_metadata")
        else None
    )
    if metadata is not None:
        mapping = config.get("patient_metadata_column_map", {})
        if any(source not in metadata for source in mapping):
            raise ValueError(
                "Patient metadata column mapping names absent source columns"
            )
        if any(
            target in metadata and source != target
            for source, target in mapping.items()
        ):
            raise ValueError(
                "Patient metadata mapping would overwrite existing columns"
            )
        metadata = metadata.rename(columns=mapping)
        metadata["subject_id"] = metadata.subject_id.astype(str)
        if "index_date" in metadata and "index_year" not in metadata:
            metadata["index_year"] = pd.to_datetime(
                metadata.index_date, errors="raise"
            ).dt.year
        if metadata.subject_id.duplicated().any():
            raise ValueError("Patient metadata has duplicate subject IDs")
    inputs, audit, hashes = [], [], {}
    for specification, (data, summary, digest) in zip(
            config["artifacts"], read_artifacts(config["artifacts"], metadata, estimand)):
        inputs.append(data)
        audit.append({"path": specification["path"], **summary})
        hashes[specification["path"]] = digest
    # Validate configured strata before any expensive bootstrap work. Only the
    # evaluated patients determine category counts, never unrelated registry rows.
    if metadata is not None:
        evaluated_ids = pd.concat(
            [part[["subject_id"]].drop_duplicates() for part in inputs],
            ignore_index=True,
        ).drop_duplicates()
        specifications = dict(config.get("patient_groups") or {})
        for column in config.get("stratified_reanalysis") or {}:
            specifications.setdefault(column, {"provenance": "pre_index"})
        available = {k: v for k, v in specifications.items() if k in metadata}
        for _ in subgroup_labels(evaluated_ids, metadata, available):
            pass
    stage("censoring weights")
    predictions = pd.concat(inputs, ignore_index=True)
    # The concatenated table now owns these rows. Retaining every input table
    # throughout normalization doubles the prediction payload at cohort scale.
    inputs.clear()
    del data
    predictions, combined_audit = normalize_predictions(predictions, estimand, copy=False)
    predictions = collapse_seed_repeats(predictions, config)
    if config.get("analysis_horizons_days"):
        keep_horizons = predictions.horizon_days.isin(config["analysis_horizons_days"])
        if not keep_horizons.all():
            predictions = predictions.loc[keep_horizons]
        if predictions.empty:
            raise ValueError("No predictions at the requested analysis horizons")
    contrast_manifest = yaml.safe_load(Path(config["contrast_manifest"]).read_text())
    contrasts = {
        name: value
        for family in contrast_manifest["essential_contrasts"].values()
        for name, value in family.items()
    }
    contrasts = national_counterparts(contrasts, config.get("opera_model"))
    contrasts.update({name: expression for name, expression in NATIONAL_ROUTING.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update(config.get("extra_contrasts", {}))
    contrasts.update(router_decomposition(set(predictions.model)))
    predictions, opera_contrasts = with_opera_alias(predictions, config.get("opera_model"))
    contrasts.update(opera_contrasts)
    contrasts.update({name: expression for name, expression in SAMPLING_SENSITIVITY.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in TASK_ROUTED.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in TASK_DESIGN.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in TASK_SEEDS.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in TASK_ABLATION.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in ENCODER_LR.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in TASK_POOLING.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in FULL_OWN.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    contrasts.update({name: expression for name, expression in PARTIAL_POOLING.items()
                      if models_for({name: expression}) <= set(predictions.model)})
    if config.get("design_search", True):
        contrasts.update(design_search_contrasts(set(predictions.model)))
    if "R_ORIG_L" in set(predictions.model):
        # The originally locked router, kept as a sensitivity model when another
        # router fit is primary (lock amendment 2026-10-27).
        contrasts.update({name: expression for name, expression in ORIGINAL_ROUTER.items()
                          if models_for({name: expression}) <= set(predictions.model)})
    candidates = config.get("strategy_candidates", ["S_L", "D_L", "O_L"])
    stage("strategy selection")
    selected, selection = tuning_selected_strategies(predictions, candidates)
    # A separate descriptive policy adds joint and the declared OPERA model.
    # Preserve BEST_TUNING and its original S/D/O comparator unchanged.
    inclusive_candidates = list(dict.fromkeys([*candidates, 'J_L', 'OPERA']))
    if set(inclusive_candidates) <= set(predictions.model):
        _, inclusive_selection = tuning_selected_strategies(predictions, inclusive_candidates)
    else:
        inclusive_selection = pd.DataFrame(columns=[
            'seed', 'cohort', 'outcome', 'status', 'selected_model',
            'candidate_models', 'selection_metric'])
    if not selected.empty:
        predictions = pd.concat([predictions, selected], ignore_index=True)
        contrasts.update(
            {
                "J_vs_best_tuning": "gain(J_L,BEST_TUNING)",
                "R_vs_best_tuning": "gain(R_L,BEST_TUNING)",
            }
        )
    predictions, unused_models = analysed_models_only(predictions, contrasts, config,
                                                     candidates)
    replicates = int(config.get("bootstrap_replicates", 1000))
    if replicates < 1:
        raise ValueError("At least one paired bootstrap replicate is required")
    bootstrap_method = config.get('bootstrap_method', 'multiplier')
    performance_support = dict(min_cases=int(config.get('performance_min_cases', 5)),
                               min_controls=int(config.get('performance_min_controls', 5)))
    draws = patient_bootstrap(
        predictions, replicates, int(config.get("bootstrap_seed", 2026)), method=bootstrap_method
    )
    context = dict(
        metadata=metadata,
        audit=audit,
        hashes=hashes,
        combined_audit=combined_audit,
        contrasts=contrasts,
        selection=selection,
        inclusive_selection=inclusive_selection,
        unused_models=unused_models,
        replicates=replicates,
        bootstrap_method=bootstrap_method,
        performance_support=performance_support,
        draws=draws,
        contrast_manifest=contrast_manifest,
        selected_present=not selected.empty,
    )
    return predictions, context


def run(config, output_dir):
    """Produce reviewable tables with explicit missing-input/coverage statuses."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            "Use a new analysis directory; existing results are immutable"
        )
    if not config.get("artifacts"):
        raise ValueError("Provide explicit artifact paths and run identities")
    estimand = config.get("estimand")
    if estimand not in ESTIMANDS:
        raise ValueError(
            f"Declare the analysis estimand explicitly: one of {ESTIMANDS}. "
            "Known-status and full-risk-set IPCW results are separate analyses."
        )
    import os
    checkpoint_root = None
    if os.environ.get('OPERA_ANALYSIS_CHECKPOINT_DIR'):
        from opera.evaluation.distributed_analysis import signature, load_prepared, save_prepared
        checkpoint_root = Path(os.environ['OPERA_ANALYSIS_CHECKPOINT_DIR']) / signature(config)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        # The coordinator publishes the active immutable generation for workers.
        from opera.evaluation.distributed_analysis import save_json
        save_json(checkpoint_root.parent / 'active.json', {'generation': checkpoint_root.name})
    if checkpoint_root is not None and (checkpoint_root / 'prepared/prepared.json').exists():
        stage('loading prepared analysis snapshot (no raw prediction reread)')
        predictions, context = load_prepared(checkpoint_root / 'prepared')
    else:
        predictions, context = prepare_analysis_predictions(config, estimand)
        if checkpoint_root is not None:
            stage('saving prepared analysis snapshot')
            save_prepared(checkpoint_root / 'prepared', predictions, context)
    metadata = context['metadata']
    audit = context['audit']
    hashes = context['hashes']
    combined_audit = context['combined_audit']
    contrasts = context['contrasts']
    selection = context['selection']
    inclusive_selection = context['inclusive_selection']
    unused_models = context['unused_models']
    replicates = context['replicates']
    bootstrap_method = context['bootstrap_method']
    performance_support = context['performance_support']
    draws = context['draws']
    contrast_manifest = context['contrast_manifest']
    selected_present = context['selected_present']
    del context
    from opera.evaluation.distributed_analysis import cached_call
    stage("paired analysis")
    tables = paired_analysis(
        predictions,
        contrasts,
        replicates=replicates,
        draws=draws,
        patient_gain_contrasts=config.get("patient_gain_contrasts"),
        checkpoint_dir=checkpoint_root / 'paired/primary' if checkpoint_root is not None else None,
        **performance_support,
    )
    # Absolute metrics on a common panel, e.g. joint/routed under each encoder origin.
    stage("panel estimates")
    model_bootstrap_draws = tables.pop("model_bootstrap_draws")
    tables["panel_estimates"] = cached_call(checkpoint_root, 'panel_estimates', lambda: panel_estimates(
        tables["estimates"], model_bootstrap_draws, panel_groups(config)))
    del model_bootstrap_draws
    cell_metadata = read_table(config["cell_metadata"])
    for column, definition in config.get("cell_groups", {}).items():
        if definition.get("provenance") != "train_only":
            raise ValueError("Cell heterogeneity groups must be train-only")
        if "values" in definition:
            # Membership group, e.g. the subtypes without a prognostic index.
            members = {str(value) for value in definition["values"]}
            cell_metadata[column] = np.where(
                cell_metadata[definition["column"]].astype(str).isin(members),
                definition.get("label", "yes"), "other")
            continue
        cell_metadata[column] = pd.cut(
            pd.to_numeric(cell_metadata[definition["column"]]),
            definition["bins"],
            include_lowest=True,
        ).astype(str)
    stage("cells pooled over horizons")
    tables["contrasts_all_horizons"] = cached_call(checkpoint_root, 'pooled_horizons',
        lambda: pool_cells_over_horizons(tables["contrasts"], tables["bootstrap_draws"]))
    stage("effect summaries")
    annotated, summaries = cached_call(checkpoint_root, 'effect_summaries', lambda: effect_summaries(
        tables["contrasts"],
        tables["bootstrap_draws"],
        cell_metadata,
        strata=config.get("cell_summary_strata") or list(dict.fromkeys(
            ["support_bin", "outcome_family", "endpoint_group", "cohort",
             *config.get("cell_groups", {})])),
        checkpoint_dir=checkpoint_root / 'tasks/equal_cell' if checkpoint_root is not None else None,
    ))
    from opera.evaluation.headline_holm import headline_holm
    tables["headline_holm"] = headline_holm(summaries, bootstrap_method=bootstrap_method)
    stage("disease-balanced effect summaries")
    _, tables["effect_summaries_disease_balanced"] = cached_call(
        checkpoint_root, 'disease_balanced', lambda: effect_summaries(
        tables["contrasts"], tables["bootstrap_draws"], cell_metadata,
        strata=config.get("cell_summary_strata") or list(dict.fromkeys(
            ["support_bin", "outcome_family", "endpoint_group", "cohort",
             *config.get("cell_groups", {})])), aggregation="disease_balanced",
        checkpoint_dir=checkpoint_root / 'tasks/disease_balanced' if checkpoint_root is not None else None))
    stage("calibration, strategies, negative transfer")
    if checkpoint_root is not None:
        from opera.evaluation.distributed_analysis import distributed_calibration
        calibration, calibration_metrics = cached_call(checkpoint_root, 'calibration_combined',
            lambda: distributed_calibration(predictions, checkpoint_root / 'tasks/calibration'))
    else:
        calibration, calibration_metrics = reliability_tables(predictions), calibration_summary(predictions)
    tables.update(
        contrasts=annotated,
        effect_summaries=summaries,
        calibration=calibration,
        calibration_summary=calibration_metrics,
        strategy_selection=selection,
        strategy_selection_inclusive=inclusive_selection,
        negative_transfer=negative_transfer_strata(
            tables["contrasts"], **config.get("rescue", {})
        ),
    )
    # Decision points fixed on tuning data: confusion counts, sensitivity,
    # specificity, predictive values and MCC, paired through the same draws.
    stage("operating points")
    point_settings = config.get("operating_points", {})
    point_tables, difference_tables = [], []
    rules = list(point_settings.get("rules", DEFAULT_RULES))
    pairs = [tuple(pair) for pair in point_settings.get(
        "pairs", [["J_L", "X"], ["J_L", "S_L"], ["R_L", "J_L"]])]
    minimum_events = int(point_settings.get("minimum_tuning_events", 5))
    if checkpoint_root is not None:
        from opera.evaluation.distributed_analysis import distributed_operating_points
        operating_results = cached_call(checkpoint_root, 'operating_all',
            lambda: distributed_operating_points(predictions, draws,
                checkpoint_root / 'tasks/operating_points', rules=rules, pairs=pairs,
                minimum_events=minimum_events))
    else:
        operating_results = [operating_points(predictions, draws, rule=rule,
            pairs=pairs, minimum_events=minimum_events) for rule in rules]
    for points, differences in operating_results:
        point_tables.append(points)
        difference_tables.append(differences)
    points = pd.concat(point_tables, ignore_index=True)
    tables["operating_points_counts"] = points
    tables["operating_points"] = suppress_small_counts(
        points, int(point_settings.get("minimum_cell_count", 5))
    )
    tables["operating_point_differences"] = pd.concat(difference_tables, ignore_index=True)
    stage("seed-averaged effects")
    tables["seed_averaged_effects"] = seed_averaged_effects(
        tables["contrasts"], tables["bootstrap_draws"]
    )
    stage("router gate")
    gate_settings = {
        **contrast_manifest.get("evaluation", {}).get("router_null_equivalence", {}),
        **config.get("router_null_equivalence", {}),
    }
    gate_contrast = gate_settings.get("contrast", "R0")
    gated = tuple(gate_settings.get("gates", ROUTER_GATED_CONTRASTS))
    tables["router_null_equivalence"] = router_null_equivalence(
        tables["effect_summaries"],
        gate_settings.get("tolerance"),
        contrast=gate_contrast,
        gated=gated,
    )
    tables["contrasts"] = annotate_router_gate(
        tables["contrasts"], tables["router_null_equivalence"], gated=gated
    )
    tables["effect_summaries"] = annotate_router_gate(
        tables["effect_summaries"], tables["router_null_equivalence"], gated=gated
    )
    if "n_events_train" in annotated:
        rarity = annotated.loc[
            annotated.metric.isin(["auc", "ipa", "brier", "log_loss"])
        ].copy()
        rarity["metric"] = rarity.metric.replace(
            {"auc": "auroc", "brier": "brier_score"}
        )
        rarity["difference"] = rarity.gain
        rarity["minority_class"] = rarity[["n_cases", "n_controls"]].min(axis=1)
        rarity["cell_id"] = (
            rarity[["cohort", "outcome", "horizon_days", "contrast"]]
            .astype(str)
            .agg("::".join, axis=1)
        )
        tables["rarity_inputs"] = rarity
    if not tables["negative_transfer"].empty:
        _, tables["negative_transfer_summary"] = effect_summaries(
            tables["negative_transfer"],
            tables["bootstrap_draws"],
            cell_metadata[["cohort", "outcome", "horizon_days"]],
            strata=["stratum"],
            checkpoint_dir=checkpoint_root / 'tasks/negative_transfer' if checkpoint_root is not None else None,
        )
    stage("identity and subgroup sensitivity analyses")
    statuses = [{"analysis": "models_not_analysed", "status": "dropped",
                 "models": sorted(unused_models)}] if unused_models else []
    if metadata is not None:
        tables["router_identity_audit"] = audit_router_identity(metadata)
        needs_review = (
            tables["router_identity_audit"]
            .status.isin(
                [
                    "missing_identity_time_provenance",
                    "missing_identity_evidence",
                    "same_or_postindex_evidence_needs_review",
                ]
            )
            .any()
        )
        statuses.append(
            {
                "analysis": "router_identity_provenance",
                "status": "needs_review"
                if needs_review
                else "evidence_recorded_not_independently_certified",
            }
        )
    if (
        metadata is not None
        and config.get("identity_date_sensitivity", True)
        and (
            "cohort_identity_post_index" in metadata
            or {"index_date", "cohort_identity_evidence_date"} <= set(metadata)
        )
    ):
        if "cohort_identity_post_index" in metadata:
            # Flag from how the subtype was assigned (first-line pipeline pass).
            late = metadata.cohort_identity_post_index.astype(bool)
            retained_ids = set(metadata.loc[~late, "subject_id"])
        else:
            index = pd.to_datetime(metadata.index_date, errors="raise")
            diagnosis = pd.to_datetime(
                metadata.cohort_identity_evidence_date, errors="raise"
            )
            retained_ids = set(
                metadata.loc[
                    diagnosis.notna() & index.notna() & (diagnosis <= index),
                    "subject_id",
                ]
            )
        from opera.evaluation.secondary_analysis import bounded_identity_counts
        tables["identity_sensitivity_counts"] = bounded_identity_counts(predictions, retained_ids)
        has_retained = bool(tables["identity_sensitivity_counts"].n_known_retained.gt(0).any())
        if not has_retained:
            statuses.append(
                {
                    "analysis": "late_diagnosis_exclusion_sensitivity",
                    "status": "no_dated_patients_retained",
                }
            )
        else:
            sensitivity = paired_analysis(
                predictions,
                contrasts,
                subject_ids=retained_ids, model_names=models_for(contrasts),
                replicates=replicates,
                draws=draws,
                patient_gain_contrasts=[],
                checkpoint_dir=checkpoint_root / 'paired/identity' if checkpoint_root is not None else None,
                **performance_support,
            )
            tables["late_diagnosis_exclusion_sensitivity"] = sensitivity["contrasts"]
            statuses.append(
                {
                    "analysis": "late_diagnosis_exclusion_sensitivity",
                    "status": "computed_same_fits_restricted_evaluation_population",
                }
            )
    # Same fits, evaluation restricted to one stratum at a time (e.g. the region
    # where patients were treated): panel effects and absolute model metrics,
    # each with intervals from the shared patient resamples.
    for column, spec in (config.get("stratified_reanalysis") or {}).items():
        if metadata is None or column not in metadata:
            statuses.append({"analysis": f"stratified_{column}",
                             "status": "missing_covariate"})
            continue
        wanted = {k: v for k, v in contrasts.items() if k in spec.get("contrasts", contrasts)}
        effects, absolute = [], []
        wanted = {k: v for k, v in wanted.items()
                  if models_for({k: v}) <= set(predictions.model)}
        summary_models = [m for m in spec.get("models", ["J_L", "X"])
                          if m in set(predictions.model)]
        needed = models_for(wanted, summary_models)
        relevant_ids = predictions.loc[predictions.model.isin(needed), "subject_id"].drop_duplicates()
        # Use exactly the bins and missing-value labels of the Brier subgroups.
        group_spec = (config.get("patient_groups") or {}).get(
            column, {"provenance": "pre_index"})
        _, labelled = next(subgroup_labels(
            relevant_ids.to_frame(), metadata, {column: group_spec}))
        for value, ids in labelled.groupby("subgroup").subject_id:
            try:
                result = paired_analysis(predictions, wanted, replicates=replicates, draws=draws,
                                         subject_ids=set(ids), model_names=needed,
                                         patient_gain_contrasts=[],
                                         checkpoint_dir=(checkpoint_root / 'paired' /
                                             ('stratum_' + hashlib.sha256(repr((column, value)).encode()).hexdigest()[:24]))
                                             if checkpoint_root is not None else None,
                                         **performance_support)
            except ValueError:
                continue
            _, summary = effect_summaries(result["contrasts"], result["bootstrap_draws"],
                                          cell_metadata, strata=["no_strata"],
                                          stable_cells_only=True,
                                          checkpoint_dir=(checkpoint_root / 'tasks' /
                                              ('stratum_summary_' + hashlib.sha256(repr((column, value)).encode()).hexdigest()[:24]))
                                              if checkpoint_root is not None else None)
            effects.append(summary.assign(covariate=column, stratum_value=str(value),
                                          n_subjects=len(set(ids))))
            absolute.append(panel_estimates(
                result["estimates"], result["model_bootstrap_draws"],
                {column: summary_models}, stable_cells_only=True, pooled_horizons=True,
            ).assign(covariate=column, stratum_value=str(value)))
        if effects:
            tables[f"stratified_{column}_effects"] = pd.concat(effects, ignore_index=True)
            tables[f"stratified_{column}_models"] = pd.concat(absolute, ignore_index=True)
        statuses.append({"analysis": f"stratified_{column}",
                         "status": "computed" if effects else "no_estimable_stratum"})
    # Patient groups whose covariate is not in the metadata yet (e.g. national
    # vocabulary coverage before the national arm exists) are skipped and
    # recorded, as for the stratified analyses above.
    patient_groups = {}
    for column, spec in (config.get("patient_groups") or {}).items():
        if metadata is not None and column in metadata:
            patient_groups[column] = spec
        else:
            statuses.append({"analysis": f"patient_group_{column}",
                             "status": "missing_covariate"})
    if (
        metadata is not None
        and patient_groups
        and not tables["patient_gains"].empty
    ):
        if checkpoint_root is not None:
            from opera.evaluation.distributed_analysis import distributed_patient_subgroups
            subgroup_tables = cached_call(checkpoint_root, 'patient_subgroups',
                lambda: distributed_patient_subgroups(tables["patient_gains"], metadata,
                    patient_groups, draws, checkpoint_root / 'tasks/patient_subgroups'))
            tables["patient_subgroups"], tables["patient_subgroup_panels"] = subgroup_tables
        else:
            tables["patient_subgroups"] = subgroup_patient_gains(
                tables["patient_gains"], metadata, patient_groups, draws=draws)
            tables["patient_subgroup_panels"] = panel_subgroup_gains(
                tables["patient_gains"], metadata, patient_groups, draws=draws)
        statuses.append({"analysis": "patient_characteristics", "status": "computed"})
    else:
        statuses.append(
            {
                "analysis": "patient_characteristics",
                "status": "missing_metadata_or_group_definitions",
            }
        )
    if config.get("coverage_roster"):
        tables["model_coverage"] = coverage_inventory(
            predictions, read_table(config["coverage_roster"])
        )
        statuses.append({"analysis": "specialist_coverage", "status": "computed"})
    else:
        statuses.append(
            {
                "analysis": "specialist_coverage",
                "status": "missing_planned_coverage_roster",
            }
        )
    if config.get("route_weights"):
        tables["route_weights"] = route_summaries(read_route_artifact(config, metadata))
        statuses.append({"analysis": "route_descriptions", "status": "computed"})
    else:
        statuses.append(
            {"analysis": "route_descriptions", "status": "missing_route_export"}
        )
    if config.get("compute_inventory"):
        tables["compute_inventory"] = read_table(config["compute_inventory"])
        compute = tables["compute_inventory"]
        keys = [key for key in ("model", "seed", "cohort", "outcome") if key in compute]
        if not {"model", "seed"} <= set(keys) or compute.duplicated(keys).any():
            raise ValueError(
                "Compute inventory needs unique model/seed and optional cohort/outcome fit keys"
            )
        tables["cost_performance"] = tables["estimates"].merge(
            compute, on=keys, how="left", validate="many_to_one"
        )
    gate = tables["router_null_equivalence"]
    if not gate.empty:
        statuses.append(
            {
                "analysis": "router_null_equivalence",
                "status": "passed"
                if gate.status.eq("equivalent_within_tolerance").all()
                else "bank_effects_not_interpretable",
                "detail": ",".join(sorted(set(gate.status))),
            }
        )
    available = set(tables["contrasts"].contrast)
    for analysis, required in {
        "pooling_axes_and_scarcity": {"P1", "P2", "P3", "P4"},
        "dapt_discrimination_and_calibration": {"A1", "A2"},
        "dapt_by_router_interaction": {"I2"},
    }.items():
        missing = required - available
        statuses.append(
            {
                "analysis": analysis,
                "status": "computed" if not missing else "missing_contrasts",
                "detail": ",".join(sorted(missing)),
            }
        )
    statuses.append(
        {
            "analysis": "negative_transfer_rescue",
            "status": "computed"
            if not tables["negative_transfer"].empty
            and tables["negative_transfer"].stratum.ne("missing_tuning_evidence").any()
            else "missing_tuning_or_held_out_pairs",
        }
    )
    statuses.append(
        {
            "analysis": "tuning_selected_cell_strategies",
            "status": "computed" if selected_present else "missing_strategy_inputs",
        }
    )
    statuses.append(
        {
            "analysis": "horizon_heterogeneity",
            "status": "computed"
            if predictions.horizon_days.nunique() > 1
            else "single_horizon_only",
        }
    )
    if "patient_subgroups" in tables:
        calendar_columns = set(config.get("calendar_covariates", ["index_year"]))
        tables["calendar_heterogeneity"] = tables["patient_subgroups"].loc[
            tables["patient_subgroups"].covariate.isin(calendar_columns)
        ]
        mismatch_columns = set(config.get("domain_mismatch_covariates", []))
        tables["domain_mismatch"] = tables["patient_subgroups"].loc[
            tables["patient_subgroups"].covariate.isin(mismatch_columns)
            & tables["patient_subgroups"].contrast.isin(
                ["A1", "A2", "E1", "E2", "E3", "E4"]
            )
        ]
    for name in ("calendar_heterogeneity", "domain_mismatch"):
        statuses.append(
            {
                "analysis": name,
                "status": "computed"
                if name in tables and not tables[name].empty
                else "missing_covariates_or_contrasts",
            }
        )
    removal_names = set(config.get("source_removal_contrasts", []))
    tables["source_removal_sensitivity"] = tables["contrasts"].loc[
        tables["contrasts"].contrast.isin(removal_names)
    ]
    statuses.append(
        {
            "analysis": "source_removal_sensitivity",
            "status": "computed"
            if not tables["source_removal_sensitivity"].empty
            else "missing_source_removal_inference_exports",
        }
    )
    statuses.append(
        {
            "analysis": "compute_cost",
            "status": "inventory_saved"
            if "compute_inventory" in tables
            else "missing_benchmark_inventory",
        }
    )
    tables["analysis_status"] = pd.DataFrame(statuses)
    tables["input_audit"] = pd.DataFrame(audit)
    for key in (
        "patient_metadata",
        "cell_metadata",
        "coverage_roster",
        "route_weights",
        "compute_inventory",
        "contrast_manifest",
    ):
        if config.get(key):
            hashes[config[key]] = file_hash(config[key])
    stage("writing outputs")
    # Publish the complete output directory atomically. If writing is killed,
    # its unpublished staging directory cannot block a checkpointed restart.
    import uuid
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    write_dir = output_dir.with_name(output_dir.name + '.writing_' + uuid.uuid4().hex)
    write_dir.mkdir()
    for name, table in tables.items():
        if name == "bootstrap_draws":
            np.savez_compressed(
                write_dir / "paired_bootstrap_draws_INTERNAL.npz",
                gains=disk_backed_draw_matrix(table, "gain_draws"),
            )
            table.drop(columns="gain_draws").to_csv(
                write_dir / "paired_bootstrap_index.csv", index=False
            )
            continue
        suffix = (
            "_INTERNAL"
            if name in {"patient_gains", "bootstrap_draws", "operating_points_counts"}
            else ""
        )
        table.to_csv(write_dir / f"{name}{suffix}.csv", index=False)
    summary = {
        "estimand": config["estimand"],
        "bootstrap_replicates": replicates,
        "bootstrap_method": bootstrap_method,
        "p_value_interpretation": ('twice the smaller posterior sign-tail probability; not a frequentist p-value'
                                   if bootstrap_method == 'bayesian' else 'paired frequentist bootstrap sign-tail measure'),
        "performance_support": performance_support,
        "interval_interpretation": ('Bayesian patient-weight credible intervals conditional on fixed IPCW weights'
                                    if bootstrap_method == 'bayesian' else
                                    ('Frequentist exchangeably weighted patient-bootstrap confidence intervals conditional on fixed IPCW weights'
                                     if bootstrap_method == 'multiplier' else
                                     'Multinomial patient-bootstrap intervals conditional on fixed IPCW weights')),
        "contrast_expressions": contrasts,
        "shared_bootstrap_axes": ["models", "outcomes", "horizons", "training_seeds"],
        "bootstrap_unit": ["split", "cohort", "subject_id"],
        "input_sha256": hashes,
        "config": json_safe(config),
        "combined_input_audit": combined_audit,
        "intervals": "patient_uncertainty_conditional_on_selected_fits_per_seed",
        "aggregation": "equal_cell_descriptive; endpoint/disease weighting must be locked separately",
        "tables": {name: len(table) for name, table in tables.items()},
    }
    (write_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_dir.replace(output_dir)
    return summary


def main():
    import faulthandler
    import signal

    # `kill -USR1 <pid>` prints where the analysis is (to the log).
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    summary = run(config, args.output_dir)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "tables": summary["tables"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
