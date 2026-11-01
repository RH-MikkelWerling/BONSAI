"""Prepare a staged paper run inventory from approved server inputs.

Configs may be prepared before the checkpoint exists. Execution audits the
actual checkpoint and requires the tuning decisions appropriate to each phase.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from opera.functional.cohort_groups import ALL_EVALUATED_FINE, fine_to_grouped
from opera.functional.direct_cr_contract import validate_direct_cr_config
from opera.run.prepare_supervision_factorial import factorial_configs

ADAPTER_FIELDS = (
    "readout_mode",
    "adapter_residual_scale",
    "adapter_home_floor",
    "adapter_event_tau",
    "adapter_shared_logit_init",
)


def router_config(shared, reference, *, source_mode, rank, query_mode, query_width):
    """Change readout only; preserve the matched shared training protocol."""
    policy = reference["competing_risk"]
    if policy.get("readout_mode") not in {
        "source_anchored_adapters",
        "unanchored_adapters",
    }:
        raise ValueError(
            "Supply an explicitly chosen anchored/borrowed router reference"
        )
    missing = set(ADAPTER_FIELDS) - set(policy)
    if missing:
        raise ValueError(f"Router reference must explicitly record: {sorted(missing)}")
    result = copy.deepcopy(shared)
    risk = result["competing_risk"]
    risk.update({key: policy[key] for key in ADAPTER_FIELDS})
    risk.update(
        adapter_source_mode=source_mode,
        adapter_bottleneck=rank,
        adapter_query_mode=query_mode,
        adapter_query_hidden_dim=query_width,
        fine_column="cohort_fine",
        fine_names=[
            fine
            for fine in ALL_EVALUATED_FINE
            if fine_to_grouped(fine) in shared["cohorts"]
        ],
    )
    result["adapter_initialization"] = "dapt_fresh"
    result["source_opera_ckpt"] = None
    # Every router parameter (adapters, keys, queries, query interaction, shared
    # logit) uses the small adapter epsilon: their gradients sit far below the
    # default 1e-6, which otherwise damps their updates. Router fit audit,
    # 2026-10-23: the interaction's first layer had 75-99% of entries below
    # 1e-6 and 19-31% below 1e-8; none below 1e-10 in any routing parameter
    # (exact zeros of switched-off sources aside).
    epsilon = reference.get("training", {}).get("adapter_optimizer_epsilon")
    result["training"]["adapter_optimizer_epsilon"] = (
        ROUTER_EPSILON if epsilon is None else min(float(epsilon), ROUTER_EPSILON))
    result["training"]["adapter_epsilon_scope"] = "router"
    validate_direct_cr_config(result)
    return result


ROUTER_EPSILON = 1e-10  # Adam epsilon for every router parameter (see router_config)
SENSITIVITY_SHARED_LOGIT = 3.0  # control_RSI: the shared slot starts with a larger share (~15-18% of 94 sources vs ~1%)

ROUTER_CONTROLS = {
    # alias: (source bank, borrowing, disease embedding)
    "JDE": ("none", True, True),           # joint + disease identity only
    "RDH": ("disease_only", False, False),  # own-disease adapter, no borrowing
    "ROH": ("outcome_only", False, False),  # own-outcome adapter, no borrowing
    # The selected source bank without borrowing (own disease and own outcome
    # adapter only): R - RBH is the borrowing of the locked router itself, whichever
    # source bank was selected (added 2026-10-24 when the screen selected "both").
    "RBH": ("both", False, False),
}


# Router fix (lock addendum 2026-10-26). The stage-2 router collapsed onto one
# outcome's adapter within the first epoch. Each fit is the locked router
# (source bank, rank, query) with:
#   alias: (source bank, borrowing, global adapter, warm-up epochs, logit scale, seed)
# A = a global adapter outside the routed mix; B = 2 epochs of router warm-up;
# T = scores bounded as 3 * cos(query, key). R_ABT is used only if R_AB fails
# the prespecified routing-health criteria. R_AB_s43 repeats R_AB on another
# seed: the noise floor for the borrowing contrast.
ROUTER_FIX_WARMUP, ROUTER_FIX_LOGIT_SCALE = 2, 3.0
ROUTER_FIX = {
    "R_AB": ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 42),
    "RBH_AB": ("selected", False, True, ROUTER_FIX_WARMUP, 0.0, 42),  # own sources only
    "N_A": ("none", True, True, ROUTER_FIX_WARMUP, 0.0, 42),  # global adapter, no sources
    "R_A": ("selected", True, True, 0, 0.0, 42),
    "R_ABT": ("selected", True, True, ROUTER_FIX_WARMUP, ROUTER_FIX_LOGIT_SCALE, 42),
    "R_AB_s43": ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 43),
}


# Own sources outside the mix (stage 4; same addendum, after the hub audits):
# own disease and own outcome adapters at full weight outside the routed mix,
# which holds only borrowed sources; with A and B, and no home floor.
ROUTER_OWN = {
    "R_O": ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 42),
    "RBH_O": ("selected", False, True, ROUTER_FIX_WARMUP, 0.0, 42),  # no borrowed mix
    "R_O_s43": ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 43),
}


# Partial-pooling router (stage 6; lock amendment 2026-10-30, before any test
# data): R_AB's design with soft-normalised sources, own sources outside the mix
# at w_own = n/(n + tau) and a learned temperature. R_PP_s43 is the noise floor.
ROUTER_PP = {
    "R_PP": ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 42),
    "R_PP_s43": ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 43),
}
# Task-routed candidates (stage 7; 31 October 2026, before any test data): R_PP
# with routing from disease and outcome only and bounded logits (bound x cosine);
# R_PT3V adds the variance-preserving borrowed mixture. Values: (bound, mix).
ROUTER_PT = {"R_PT5": (5.0, "per_source"), "R_PT3V": (3.0, "variance_preserving")}


def task_routing(config, bound, mix):
    partial_pooling(config)
    config["competing_risk"].update(adapter_task_routing=True, adapter_logit_bound=bound,
                                    adapter_borrow_mix=mix)
    return config


# Stage 8 (31 October, evening): the R_PT candidates finished ~0.004 below R_AB,
# with tau falling and the correction scale growing (they wanted more of their
# own correction). Own corrections are applied as in R_O (full weight,
# unbounded); only the borrowed part is pooled, with task routing bounded at
# 5 x cosine. The stage is a one-factor-at-a-time sweep around the anchor
# R_PF5E; each fit changes one design choice (lock amendment 2026-10-31,
# evening). Keys: values (borrowed corrections raw, equalised, bounded or off),
# bound (routing logit bound), patient_gate, amount (support-based or learned
# per-task borrowed share), borrow_from (source banks), router_lr (learning-rate
# multiplier for routing parameters), warmup (router warm-up epochs), rank
# (adapter rank), groups (hierarchical lineage / endpoint-group adapters),
# live (borrowed sources with live parameters: a task-level mixture of
# experts), latent (K learned group adapters with task routing over them),
# fraction (share of training patients, the data-efficiency subset)
# and seed. Order is priority: workers claim fits in bundle order.
PF_ANCHOR = {"values": "equalised", "bound": 5.0, "patient_gate": False, "amount": "support",
             "borrow_from": "both", "router_lr": 1.0, "warmup": ROUTER_FIX_WARMUP,
             "rank": None, "groups": None, "live": False, "latent": 0, "fraction": 1.0, "seed": 42}
ROUTER_PF = {
    "R_PF5E": {},                         # anchor
    # Hierarchical shrinkage over known structure, no routing; then both.
    "R_PH_G": {"values": "off", "groups": "disease"},        # lineage adapters
    "R_PH_GE": {"values": "off", "groups": "both"},          # + endpoint-group adapters
    "R_PH_GE_PF5E": {"groups": "both"},                      # hierarchy + routed borrowing
    # Learned grouping: 8 shared group adapters, routing learns who shares them.
    "R_PL8": {"values": "off", "latent": 8},
    # Capacity: the corrections at rank 32 instead of the locked rank.
    "R_PF5E_R32": {"rank": 32},
    "R_PF5": {"values": "raw"},           # borrowed corrections as they are
    "R_PF5B": {"values": "bounded"},      # borrowed corrections capped as in R_PT
    "R_PF10E": {"bound": 10.0},           # looser routing
    "R_PF5EP": {"patient_gate": True},    # patient-specific routing, gated from 0
    "R_PF5EL": {"amount": "learned"},     # free borrowed share per disease x outcome
    "R_PF5ED": {"borrow_from": "disease"},  # borrow from diseases only
    "R_PF5EO": {"borrow_from": "outcome"},  # borrow from outcomes only
    "R_PF5E_LR10": {"router_lr": 10.0},   # routing parameters at 10x learning rate
    "R_PF5E_W0": {"warmup": 0},           # no router warm-up
    "R_PF5E_s43": {"seed": 43},           # noise floor of the anchor
    "R_PF5E_LIVE": {"live": True},        # learned sharing: borrowers train the sources
    # Scarce data (10% of training patients, the J_L_F10 subset): where borrowing
    # should matter if it can.
    "R_PF5E_F10": {"fraction": 0.1},
    "R_PH_GE_F10": {"values": "off", "groups": "both", "fraction": 0.1},
    "R_PF5E_LIVE_F10": {"live": True, "fraction": 0.1},
    "R_PL8_F10": {"values": "off", "latent": 8, "fraction": 0.1},
}
# Controls, never selected: routing and tau frozen for the whole fit; nothing
# borrowed (the own-only control RBH_O implemented in this code).
ROUTER_PF_CONTROLS = {"R_PF_OFF": {"values": "off"}, "R_PF5E_FROZEN": {"warmup": None},
                      "R_PF_OFF_R32": {"values": "off", "rank": 32},
                      "R_PF_OFF_F10": {"values": "off", "fraction": 0.1}}


# Stage 9 (1 November 2026): task-conditioned pooling. The joint model with
# outcome-conditioned attention over the encoder's token states instead of one
# shared summary vector (zero-gated, so each fit starts exactly at J). Every
# correction tried in stages 3-8 acts on that one shared vector; these test
# whether letting each outcome read the record differently is the bigger lever.
# Values: (readout mode, condition on the diagnosis, own corrections, lineage
# groups, outcome query rank). Lineage groups tie the diagnosis terms of the
# factorized attention to the lineage (FPF) and add lineage corrections (OWN).
# A query rank ties outcomes: each outcome's query is a combination of that many
# shared basis queries plus its own deviation.
TASK_POOLING = {
    "J_OA": ("outcome_attention", False, False, False, 0),                     # one query per outcome
    "J_OA_LR": ("outcome_attention", False, False, False, 8),                  # outcome queries tied (rank 8)
    "J_SP": ("shared_attention_prototypes", False, False, False, 0),           # shared maps, mixed per outcome
    "J_FP": ("factorized_cohort_outcome_prototypes", False, False, False, 0),  # mixture by disease group x outcome
    "J_FPF": ("factorized_cohort_outcome_prototypes", True, False, False, 0),  # mixture by diagnosis x outcome
    "J_OA_OWN": ("outcome_attention", False, True, False, 0),                  # + own disease/outcome corrections
    "J_OA_LR_OWN": ("outcome_attention", False, True, False, 8),               # tied queries + own corrections
    "J_OA_OWN_PH": ("outcome_attention", False, True, True, 0),                # + lineage corrections (hierarchy)
    "J_FPF_OWN": ("factorized_cohort_outcome_prototypes", True, True, False, 0),
    "J_FPF_PH": ("factorized_cohort_outcome_prototypes", True, False, True, 0),   # diagnosis terms tied to lineage
    "J_FPF_PH_OWN_PH": ("factorized_cohort_outcome_prototypes", True, True, True, 0),  # + own and lineage corrections
}


# Stage 10 (1 November 2026): is the representation the limit? Every readout
# and borrowing variant in stages 3-9 reached about the same tuning AUC, so the
# encoder is fine-tuned harder: its learning rate as a multiple of the head's
# (0.1 in every other fit). Values: (model, encoder learning-rate multiplier).
ENCODER_LR = {"J_ELR10": ("J", 1.0), "R_AB_ELR10": ("R_AB", 1.0), "R_AB_ELR3": ("R_AB", 0.3)}
# Stage 11 (1 November 2026): the stage-9 reading models with own corrections
# at the full encoder rate. Reading attends over token states, so a
# harder-trained encoder may matter more for them than for J and R_AB.
ENCODER_LR_READING = {"J_FPF_PH_OWN_PH_ELR10": ("J_FPF_PH_OWN_PH", 1.0),
                      "J_OA_OWN_ELR10": ("J_OA_OWN", 1.0)}


def encoder_lr_entries(entries, *, seed=42, designs=None, first_order=-100):
    """Copies of existing fits that differ only in the encoder learning rate."""
    by_name = {e["name"]: e for e in entries}
    sources = {"J": f"seed{seed}/L/J_all_diseases_all_outcomes"}
    new = []
    for order, (alias, (model, multiplier)) in enumerate((designs or ENCODER_LR).items()):
        source = by_name[sources.get(model, f"seed{seed}/L/fix_{model}")]
        config = copy.deepcopy(source["config"])
        config.setdefault("training", {})["encoder_lr_multiplier"] = float(multiplier)
        new.append({"name": f"seed{seed}/L/fix_{alias}", "phase": "router_fix", "origin": "L",
                    "seed": seed, "condition": {}, "config": config,
                    # claimed before every remaining stage-8 and stage-9 fit
                    "queue_order": first_order + order})
    return new


# Stage 12 (1 November 2026): what the leading reading design is made of.
# Sensitivity and ablation fits around J_OA_OWN (the simplest leader; tied
# queries added nothing measurable); never selected. Values: (source, changes).
TASK_ABLATION = {
    "J_OA_OWN_s43": ("J_OA_OWN", {"seed": 43}),                  # seed noise floor (claimed first)
    "J_OA_OWN_R32": ("J_OA_OWN", {"rank": 32}),                  # task-specific capacity
    "J_OA_OWN_NODIS": ("J_OA_OWN", {"own_disease": False}),      # reading + outcome corrections
    "J_OA_OWN_NOOUT": ("J_OA_OWN", {"own_outcome": False}),      # reading + diagnosis corrections
    "J_OA_OWN_F10": ("J_OA_OWN", {"fraction": 0.1}),             # scarce data (the J_L_F10 subset)
}


# Stage 13 (1 November 2026): design changes aimed at the diagnosis of what
# makes the reading designs better (own corrections overfit rare outcomes and
# are not shrunk by support; the gain is synergy between reading and own
# corrections). Candidates under the locked rule. Values: (source, changes).
TASK_DESIGN = {
    "J_OA_OWN_SW20": ("J_OA_OWN", {"shrink_tau": 20.0}),         # cell shrinkage, weak prior
    "J_OA_OWN_SW100": ("J_OA_OWN", {"shrink_tau": 100.0}),       # cell shrinkage, strong prior
    "J_OA_OWN_X": ("J_OA_OWN", {"cross": True}),                 # diagnosis x outcome correction
    "J_OA_OWN_X_SW": ("J_OA_OWN", {"cross": True, "shrink_tau": 50.0}),
    "J_OA_OWN_PE": ("J_OA_OWN", {"endpoint": True}),             # endpoint-family corrections
    "J_OA_OWN_PH_X": ("J_OA_OWN", {"cross": True, "lineage": True}),
    "J_OA_OWN_PE_X_SW": ("J_OA_OWN", {"endpoint": True, "cross": True, "shrink_tau": 50.0}),
    # Moonshot: every mechanism with a positive sign, plus capacity, reined in by
    # shrinkage where cells are small.
    "J_OA_OWN_MAX": ("J_OA_OWN", {"endpoint": True, "lineage": True, "cross": True,
                                  "shrink_tau": 50.0, "rank": 32}),
}


def task_design_entries(base, reference, checkpoints, selected_router, *, epochs, patience, seed=42):
    from opera.visualization.paper_taxonomy import fine_cohort_groups

    entries = []
    for order, (alias, (design, change)) in enumerate(TASK_DESIGN.items()):
        shared = factorial_configs(base, encoder_checkpoint=checkpoints["L"], epochs=epochs,
                                   seed=seed, patience=patience)["J_all_diseases_all_outcomes"]
        config = task_pooling_config(shared, design, reference, selected_router)
        risk = config["competing_risk"]
        if change.get("cross"):
            risk["attention_own_cross"] = True
        if "shrink_tau" in change:
            risk["attention_own_shrink_tau"] = float(change["shrink_tau"])
        if "rank" in change:
            risk["attention_own_rank"] = int(change["rank"])
        if change.get("lineage"):
            lineage = fine_cohort_groups()
            risk["attention_own_disease_groups"] = {n: lineage[n] for n in risk["fine_names"]}
        if change.get("endpoint"):
            from opera.visualization.paper_taxonomy import outcome_endpoint_groups

            family = outcome_endpoint_groups()
            risk["attention_own_outcome_groups"] = {o: family[o] for o in config["outcomes"]}
        entries.append({"name": f"seed{seed}/L/fix_{alias}", "phase": "router_fix", "origin": "L",
                        "seed": seed, "condition": {}, "config": config,
                        "queue_order": order - 140})
    return entries


def task_ablation_entries(base, reference, checkpoints, selected_router, *, epochs, patience, seed=42):
    entries = []
    for order, (alias, (design, change)) in enumerate(TASK_ABLATION.items()):
        fit_seed = int(change.get("seed", seed))
        shared = factorial_configs(base, encoder_checkpoint=checkpoints["L"], epochs=epochs,
                                   seed=fit_seed, patience=patience)["J_all_diseases_all_outcomes"]
        config = task_pooling_config(shared, design, reference, selected_router)
        risk = config["competing_risk"]
        if "rank" in change:
            risk["attention_own_rank"] = int(change["rank"])
        for key in ("own_disease", "own_outcome"):
            if key in change:
                risk[f"attention_{key}"] = bool(change[key])
        if "fraction" in change:
            config["training"].update(train_patient_fraction=float(change["fraction"]),
                                      train_fraction_seed=seed)
        entries.append({"name": f"seed{fit_seed}/L/fix_{alias}", "phase": "router_fix", "origin": "L",
                        "seed": fit_seed, "condition": {}, "config": config,
                        # claimed before every other remaining fit; the seed repeat
                        # first of all (it is what lets the gain over R_AB be claimed)
                        "queue_order": order - 150 if "seed" in change else order - 130})
    return entries


# Stage 14 (1 November 2026): seed spread. J_OA_OWN s43 trailed s42 by 0.005
# at almost every epoch, while R_AB's two seeds agreed to 0.0002; two seeds
# cannot say which was the unusual one. Two more seeds of the design, and a
# second seed of J (the reference of the main contrast). Values: (model, seed).
TASK_SEEDS = {"J_OA_OWN_s44": ("J_OA_OWN", 44), "J_OA_OWN_s45": ("J_OA_OWN", 45), "J_s43": ("J", 43)}


def task_seed_entries(base, reference, checkpoints, selected_router, *, epochs, patience):
    entries = []
    for order, (alias, (design, seed)) in enumerate(TASK_SEEDS.items()):
        config = factorial_configs(base, encoder_checkpoint=checkpoints["L"], epochs=epochs,
                                   seed=seed, patience=patience)["J_all_diseases_all_outcomes"]
        if design != "J":
            config = task_pooling_config(config, design, reference, selected_router)
        entries.append({"name": f"seed{seed}/L/fix_{alias}", "phase": "router_fix", "origin": "L",
                        "seed": seed, "condition": {}, "config": config,
                        "queue_order": order - 200})  # claimed before every remaining fit
    return entries


# Stage 15 (1 November 2026, night): the Table 2 ladder on a second seed.
# J, R_AB and J_OA_OWN have seed 43 already (J_s43 in stage 14); the own-
# corrections rung (R_PF_OFF) gets it here, so every mechanism step has a
# second seed pair. Values: (stage-8 design, seed).
LADDER_SEEDS = {"R_PF_OFF_s43": ("R_PF_OFF", 43)}


def ladder_seed_entries(base, reference, checkpoints, selected_router, *, epochs, patience):
    entries = []
    for order, (alias, (design, seed)) in enumerate(LADDER_SEEDS.items()):
        entry = router_fix_entries(base, reference, checkpoints, selected_router,
                                   epochs=epochs, patience=patience,
                                   fits={alias: ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, seed)},
                                   pooling=True)[0]
        full_own(entry["config"], {**ROUTER_PF, **ROUTER_PF_CONTROLS}[design])
        entry["queue_order"] = order - 190  # after the stage-14 seeds, before the rest
        entries.append(entry)
    return entries


def task_pooling_design(routed_model):
    """J_OA_LR_OWN_L -> J_OA_LR_OWN when the primary model is a stage-9 design, else None."""
    alias = (routed_model or "").removesuffix("_L")
    return alias if alias in TASK_POOLING else None


def task_pooling_config(shared, alias, reference, selected_router):
    """The joint model's config with one stage-9 readout (a copy)."""
    from opera.visualization.paper_taxonomy import fine_cohort_groups

    mode, by_diagnosis, own, groups, query_rank = TASK_POOLING[alias]
    config = copy.deepcopy(shared)
    risk = config["competing_risk"]
    risk["readout_mode"] = mode
    if by_diagnosis or own:
        risk.update(fine_column="cohort_fine", fine_names=[
            fine for fine in ALL_EVALUATED_FINE if fine_to_grouped(fine) in config["cohorts"]])
    if by_diagnosis:
        risk["prototype_condition"] = "fine"
    if query_rank:
        risk["outcome_attention_query_rank"] = int(query_rank)
    if own:
        risk.update(attention_own_corrections=True, attention_own_rank=int(selected_router["rank"]),
                    adapter_residual_scale=float(reference["competing_risk"]["adapter_residual_scale"]))
    if groups:
        lineage = fine_cohort_groups()
        tied = {name: lineage[name] for name in risk["fine_names"]}
        if own:
            risk["attention_own_disease_groups"] = tied
        if by_diagnosis:
            risk["prototype_condition_groups"] = tied
    config.setdefault("training", {})["enable_validation_probe"] = False
    return config


def task_pooling_entries(base, reference, checkpoints, selected_router, *, epochs, patience, seed=42):
    # Claimed before the remaining stage-8 fits (negative order), the designs the
    # token-level toy favoured first.
    order = ["J_OA_OWN", "J_OA_LR_OWN", "J_OA", "J_OA_LR", "J_OA_OWN_PH", "J_FPF_PH_OWN_PH",
             "J_FPF_PH", "J_FPF", "J_FP", "J_FPF_OWN", "J_SP"]
    entries = []
    for alias in TASK_POOLING:
        shared = factorial_configs(base, encoder_checkpoint=checkpoints["L"], epochs=epochs,
                                   seed=seed, patience=patience)["J_all_diseases_all_outcomes"]
        config = task_pooling_config(shared, alias, reference, selected_router)
        entries.append({"name": f"seed{seed}/L/fix_{alias}", "phase": "router_fix", "origin": "L",
                        "seed": seed, "condition": {}, "config": config,
                        "queue_order": order.index(alias) - len(order)})
    return entries


def full_own(config, spec):
    """Settings of one stage-8 design (anchor plus the spec's changes)."""
    design = {**PF_ANCHOR, **spec}
    task_routing(config, design["bound"], "per_source")
    risk = config["competing_risk"]
    risk.update(adapter_own_full=True, adapter_borrow_values=design["values"])
    if design["patient_gate"]:
        risk["adapter_patient_gate"] = True
    if design["amount"] != "support":
        risk["adapter_borrow_amount"] = design["amount"]
    if design["borrow_from"] != "both":
        risk["adapter_borrow_from"] = design["borrow_from"]
    if design["router_lr"] != 1.0:
        config.setdefault("training", {})["router_lr_multiplier"] = design["router_lr"]
    if design["latent"]:
        risk["adapter_latent_groups"] = int(design["latent"])
    if design["live"]:
        risk["adapter_live_sources"] = True
    if design["fraction"] != 1.0:
        config.setdefault("training", {}).update(
            train_patient_fraction=float(design["fraction"]),
            train_fraction_seed=int(design["seed"]))
    if design["rank"] is not None:
        risk["adapter_bottleneck"] = int(design["rank"])
    if design["groups"] is not None:
        from opera.visualization.paper_taxonomy import fine_cohort_groups, outcome_endpoint_groups

        lineage = fine_cohort_groups()
        risk["adapter_disease_groups"] = {name: lineage[name] for name in risk["fine_names"]}
        if design["groups"] == "both":
            endpoint = outcome_endpoint_groups()
            risk["adapter_outcome_groups"] = {name: endpoint[name] for name in config["outcomes"]}
    if "warmup" in spec:
        # None: frozen for the whole fit (the training length).
        risk["adapter_router_warmup_epochs"] = (
            spec["warmup"] if spec["warmup"] is not None else int(config["training"]["epochs"]))
    return config


# Control, never a primary candidate: R_PP with the router frozen for the whole
# fit (warm-up = training length), so tau and the borrowed mix stay at their
# starting values. R_PP minus this control is the gain from learned routing.
ROUTER_PP_CONTROLS = {
    "R_PP_FROZEN": ("selected", True, True, None, 0.0, 42),
}


def partial_pooling(config):
    config["competing_risk"].update(adapter_partial_pooling=True, adapter_own_outside=True,
                                    adapter_home_floor=0.0)
    return config


def apply_router_fix(config, fix):
    """The readout settings of a router-fix design (e.g. R_AB, or R_PP)."""
    designs = {**ROUTER_FIX, **ROUTER_PP,
               **{name: ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 42)
                  for name in (*ROUTER_PT, *ROUTER_PF)}}
    _, borrowing, global_adapter, warmup, scale, _ = designs[fix]
    config["competing_risk"].update(adapter_global=global_adapter,
                                    adapter_router_warmup_epochs=warmup)
    if not borrowing:
        config["competing_risk"]["adapter_borrowing"] = False
    if scale:
        config["competing_risk"]["adapter_logit_scale"] = scale
    if fix in ROUTER_PP:
        partial_pooling(config)
    if fix in ROUTER_PT:
        task_routing(config, *ROUTER_PT[fix])
    if fix in ROUTER_PF:
        full_own(config, ROUTER_PF[fix])
    return config


def routed_fix(routed_model):
    """R_AB_L -> R_AB (a ROUTER_FIX design); None for the original router."""
    if not routed_model or routed_model == "R_L":
        return None
    fix = routed_model.removesuffix("_L")
    if fix in ENCODER_LR:  # stage 10: the R_AB readout, encoder trained harder
        return "R_AB"
    if fix not in ROUTER_FIX and fix not in ROUTER_PP and fix not in ROUTER_PT and fix not in ROUTER_PF:
        raise ValueError(f"Unknown primary routed model {routed_model}")
    return fix


def external_routed(config, routed_model):
    """National routed fits use the primary router's readout (lock amendment
    2026-10-27: R_AB primary), so local and national routed models compare like
    with like."""
    fix = routed_fix(routed_model)
    return apply_router_fix(config, fix) if fix else config


def router_fix_entries(base, reference, checkpoints, selected_router, *, epochs, patience,
                       fits=None, own_outside=False, pooling=False):
    """The router-fix fits (local encoder): ROUTER_FIX, or ROUTER_OWN with own_outside."""
    entries = []
    for alias, (source, borrowing, global_adapter, warmup, scale, seed) in (fits or ROUTER_FIX).items():
        shared = factorial_configs(base, encoder_checkpoint=checkpoints["L"], epochs=epochs,
                                   seed=seed, patience=patience)["J_all_diseases_all_outcomes"]
        config = router_config(
            shared, reference,
            source_mode=selected_router["source_mode"] if source == "selected" else source,
            rank=int(selected_router["rank"]), query_mode=selected_router["query_mode"],
            query_width=int(selected_router["query_width"]))
        config["competing_risk"].update(
            adapter_global=global_adapter,
            # None: frozen for the whole fit (a control, e.g. R_PP_FROZEN).
            adapter_router_warmup_epochs=epochs if warmup is None else warmup)
        if not borrowing:
            config["competing_risk"]["adapter_borrowing"] = False
        if scale:
            config["competing_risk"]["adapter_logit_scale"] = scale
        if own_outside:
            config["competing_risk"].update(adapter_own_outside=True, adapter_home_floor=0.0)
        if pooling:
            partial_pooling(config)
        entries.append({"name": f"seed{seed}/L/fix_{alias}", "phase": "router_fix",
                        "origin": "L", "seed": seed, "condition": {}, "config": config})
    return entries


# Sampling sensitivity (stage 5; lock addendum 2026-10-27, before any test data):
# the primary routed model and the joint model trained with importance-weighted
# event sampling. Batches see more events, and each patient's loss is divided by
# its sampling weight, so the likelihood (and calibration) stays that of the
# natural population. The sampler requires a zero DAPT anchor weight.
IPW_SAMPLING = {"event_fraction": 0.5, "min_unique_events": 10, "min_unique_controls": 20}


def ipw_sampling(config, seed):
    config["training"]["batch_sampling"] = {
        "type": "ipw_event_horizon_bce", **IPW_SAMPLING, "seed": seed,
        "horizons_days": list(config["training"]["evaluation_horizons_days"])}
    config.setdefault("model", {})["dapt_anchor_weight"] = 0.0
    return config


def sampling_entries(base, reference, checkpoints, selected_router, *, epochs, patience,
                     seed=42):
    """J_IPW (joint) and R_AB_IPW (the primary routed design) with IPW sampling."""
    shared = factorial_configs(base, encoder_checkpoint=checkpoints["L"], epochs=epochs,
                               seed=seed, patience=patience)["J_all_diseases_all_outcomes"]
    routed = apply_router_fix(router_config(
        copy.deepcopy(shared), reference, source_mode=selected_router["source_mode"],
        rank=int(selected_router["rank"]), query_mode=selected_router["query_mode"],
        query_width=int(selected_router["query_width"])), "R_AB")
    return [{"name": f"seed{seed}/L/fix_{alias}", "phase": "router_fix", "origin": "L",
             "seed": seed, "condition": {}, "config": ipw_sampling(config, seed)}
            for alias, config in (("J_IPW", copy.deepcopy(shared)), ("R_AB_IPW", routed))]


def external_sequences(config, sequence_dir):
    """Point a national arm at our patients tokenised in the national format.

    Outcomes, membership and splits stay the paper's (each cohort's
    ``data_dir``); only the sequences and vocabulary are the national ones.
    """
    if sequence_dir is None:
        return config
    sequence_dir = Path(sequence_dir).resolve()
    config["paths"]["vocabulary"] = str(sequence_dir / "vocabulary.pt")
    files = [str(sequence_dir / f"subject_data_{split}.pt") for split in ("train", "tuning")]
    for cohort in config["cohorts"].values():
        cohort["subject_data_paths"] = files
    return config


# Data efficiency: the joint model trained on nested shares of the training
# patients, from the local and from the adapted national encoder.
DATA_EFFICIENCY_FRACTIONS = (0.1, 0.25, 0.5)
# Adapter ranks screened. The encoder is four times wider than the 64-wide
# model on which rank 8 was chosen, so rank 32 keeps the same rank-to-width
# ratio (8/64 = 32/256); the screen prefers the smallest rank unless a larger
# one is clearly better on tuning data.
ROUTER_RANKS = (8, 16, 32)


def build_bundle(
    base, reference, checkpoints, *, seeds=(42,), epochs=30, patience=5,
    selected_router=None,
    router_controls=True,
    external_sequence_dir=None,
    data_efficiency=True,
    router_fix=False,
    router_own=False,
    routed_model=None,
    sampling_ipw=False,
    router_pp=False,
    router_pt=False,
    router_pf=False,
    task_pooling=False,
    encoder_lr=False,
    encoder_lr_reading=False,
    external_encoder_lr=None,
    task_ablation=False,
    task_design=False,
    task_seeds=False,
    ladder_seeds=False,
):
    """All supervision configs and conditional bounded router candidates."""
    if not checkpoints.get("L") or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Need a local checkpoint path and unique nonempty seeds")
    if bool(checkpoints.get("E")) != bool(checkpoints.get("EA")):
        raise ValueError("Matched external matrix needs both E and EA checkpoint paths")
    if selected_router is not None:
        required = {"source_mode", "rank", "query_mode", "query_width"}
        if required - set(selected_router):
            raise ValueError(
                "Final lock needs source_mode, rank, query_mode and query_width"
            )
        if selected_router["source_mode"] not in {
            "none",
            "disease_only",
            "outcome_only",
            "both",
        }:
            raise ValueError("Unknown locked source bank")
        if selected_router["rank"] not in ROUTER_RANKS or (
            selected_router["query_mode"],
            selected_router["query_width"],
        ) not in {("additive", 128), ("interacting", 128), ("interacting", 512)}:
            raise ValueError("Router lock is outside the prespecified bounded screen")
    entries = []
    for seed in seeds:
        factorial = factorial_configs(
            base, encoder_checkpoint=checkpoints["L"], epochs=epochs, seed=seed,
            patience=patience,
        )
        shared = factorial["J_all_diseases_all_outcomes"]
        for name, config in factorial.items():
            entries.append(
                {
                    "name": f"seed{seed}/L/{name}",
                    "phase": "supervision",
                    "origin": "L",
                    "seed": seed,
                    "condition": {},
                    "config": config,
                }
            )
        # Joint model from a randomly initialised encoder of the same
        # architecture: does pretraining help at all? One fit (first seed).
        if seed == seeds[0]:
            random_init = copy.deepcopy(shared)
            random_init["encoder_initialization"] = "random"
            entries.append(
                {
                    "name": f"seed{seed}/RI/J_all_diseases_all_outcomes",
                    "phase": "pretraining_control",
                    "origin": "RI",
                    "seed": seed,
                    "condition": {},
                    "config": random_init,
                }
            )
            # Joint model on the frozen local encoder: only the readout is
            # trained, separating what fine-tuning adds from what pretraining
            # already represents.
            frozen = copy.deepcopy(shared)
            frozen.setdefault("model", {})["freeze_encoder"] = True
            entries.append(
                {
                    "name": f"seed{seed}/FZ/J_all_diseases_all_outcomes",
                    "phase": "pretraining_control",
                    "origin": "FZ",
                    "seed": seed,
                    "condition": {},
                    "config": frozen,
                }
            )
        # Only the first seed is used for bounded method selection. All final
        # source controls are replicated once an explicit router lock is given.
        if seed == seeds[0]:
            for source in ("none", "disease_only", "outcome_only", "both"):
                config = router_config(
                    shared,
                    reference,
                    source_mode=source,
                    rank=8,
                    query_mode="interacting",
                    query_width=128,
                )
                entries.append(
                    {
                        "name": f"seed{seed}/L/source_{source}_r8_q128",
                        "phase": "source_screen",
                        "origin": "L",
                        "seed": seed,
                        "condition": {},
                        "config": config,
                    }
                )
            for source in ("disease_only", "outcome_only", "both"):
                for rank in ROUTER_RANKS:
                    config = router_config(
                        shared,
                        reference,
                        source_mode=source,
                        rank=rank,
                        query_mode="interacting",
                        query_width=128,
                    )
                    entries.append(
                        {
                            "name": f"seed{seed}/L/rank_{source}_r{rank}_q128",
                            "phase": "rank_screen",
                            "origin": "L",
                            "seed": seed,
                            "condition": {"source_mode": source},
                            "config": config,
                        }
                    )
                    for mode, width in (
                        ("additive", 128),
                        ("interacting", 128),
                        ("interacting", 512),
                    ):
                        config = router_config(
                            shared,
                            reference,
                            source_mode=source,
                            rank=rank,
                            query_mode=mode,
                            query_width=width,
                        )
                        entries.append(
                            {
                                "name": f"seed{seed}/L/query_{source}_r{rank}_{mode}_q{width}",
                                "phase": "query_screen",
                                "origin": "L",
                                "seed": seed,
                                "condition": {"source_mode": source, "rank": rank},
                                "config": config,
                            }
                        )
        if selected_router is not None and router_controls and seed == seeds[0]:
            # Decompose routing: identity, own-adapter capacity, then borrowing.
            for alias, (source, borrowing, embedding) in ROUTER_CONTROLS.items():
                config = router_config(
                    shared, reference, source_mode=source,
                    rank=int(selected_router["rank"]),
                    query_mode=selected_router["query_mode"],
                    query_width=int(selected_router["query_width"]),
                )
                config["competing_risk"].update(
                    adapter_borrowing=borrowing, adapter_disease_embedding=embedding)
                entries.append({
                    "name": f"seed{seed}/L/control_{alias}",
                    "phase": "router_controls", "origin": "L", "seed": seed,
                    "condition": {}, "config": config,
                })
            # Sensitivity (added 2026-10-22): on tuning the locked router kept almost
            # no weight on the shared slot and spread the rest nearly evenly; this
            # fit starts the shared slot at the code default (logit 3.0) to see
            # whether a router that starts from the shared path borrows selectively.
            config = router_config(
                shared, reference, source_mode=selected_router["source_mode"],
                rank=int(selected_router["rank"]),
                query_mode=selected_router["query_mode"],
                query_width=int(selected_router["query_width"]),
            )
            config["competing_risk"]["adapter_shared_logit_init"] = SENSITIVITY_SHARED_LOGIT
            entries.append({
                "name": f"seed{seed}/L/control_RSI",
                "phase": "router_controls", "origin": "L", "seed": seed,
                # No gating condition: conditions name recorded decisions, and
                # this key is never one (it would silently skip the fit).
                "condition": {},
                "config": config,
            })
        if selected_router is not None:
            for source in ("none", "disease_only", "outcome_only", "both"):
                config = router_config(
                    shared,
                    reference,
                    source_mode=source,
                    rank=int(selected_router["rank"]),
                    query_mode=selected_router["query_mode"],
                    query_width=int(selected_router["query_width"]),
                )
                entries.append(
                    {
                        "name": f"seed{seed}/L/final_source_{source}",
                        "phase": "final_sources",
                        "origin": "L",
                        "seed": seed,
                        "condition": {},
                        "config": config,
                    }
                )
            for origin in ("E", "EA"):
                if not checkpoints.get(origin):
                    continue
                external_shared = external_sequences(
                    copy.deepcopy(shared), external_sequence_dir)
                external_shared["dapt_ckpt"] = checkpoints[origin]
                design = task_pooling_design(routed_model)
                if design:
                    # Primary OPERA is a stage-9 design (lock closure, Nov 2026): the
                    # national fit uses its readout on the national encoder.
                    national_routed = task_pooling_config(
                        external_shared, design, reference, selected_router)
                else:
                    national_routed = external_routed(router_config(
                        external_shared,
                        reference,
                        source_mode=selected_router["source_mode"],
                        rank=int(selected_router["rank"]),
                        query_mode=selected_router["query_mode"],
                        query_width=int(selected_router["query_width"]),
                    ), routed_model)
                for mode, config in (("J", external_shared), ("R", national_routed)):
                    if external_encoder_lr is not None:
                        # Lock closure: the national arm is fine-tuned at the encoder
                        # rate that won on tuning data (stage 10).
                        config = copy.deepcopy(config)
                        config.setdefault("training", {})["encoder_lr_multiplier"] = float(external_encoder_lr)
                    entries.append(
                        {
                            "name": f"seed{seed}/{origin}/{mode}",
                            "phase": "external",
                            "origin": origin,
                            "seed": seed,
                            "condition": {},
                            "config": config,
                        }
                    )
    if router_fix:
        if selected_router is None:
            raise ValueError("The router-fix fits need the router lock")
        entries += router_fix_entries(base, reference, checkpoints, selected_router,
                                      epochs=epochs, patience=patience)
    if router_own:
        if not router_fix:
            raise ValueError("Stage 4 repeats the stage-3 router-fix fits")
        entries += router_fix_entries(base, reference, checkpoints, selected_router,
                                      epochs=epochs, patience=patience,
                                      fits=ROUTER_OWN, own_outside=True)
    if sampling_ipw:
        if not router_own:
            raise ValueError("Stage 5 repeats the stage-4 fits")
        entries += sampling_entries(base, reference, checkpoints, selected_router,
                                    epochs=epochs, patience=patience, seed=seeds[0])
    if router_pp:
        if not sampling_ipw:
            raise ValueError("Stage 6 repeats the stage-5 fits")
        entries += router_fix_entries(base, reference, checkpoints, selected_router,
                                      epochs=epochs, patience=patience,
                                      fits={**ROUTER_PP, **ROUTER_PP_CONTROLS}, pooling=True)
    if router_pt:
        if not router_pp:
            raise ValueError("Stage 7 repeats the stage-6 fits")
        for alias, (bound, mix) in ROUTER_PT.items():
            entry = router_fix_entries(base, reference, checkpoints, selected_router,
                                       epochs=epochs, patience=patience,
                                       fits={alias: ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, 42)},
                                       pooling=True)[0]
            task_routing(entry["config"], bound, mix)
            entries.append(entry)
    if router_pf:
        if not router_pt:
            raise ValueError("Stage 8 repeats the stage-7 fits")
        # Decisive fits first: anchor, own-only floor, frozen routing, hierarchy.
        # Order revised after the multi-outcome toy (31 October, night): the
        # designs it favoured or that test its open questions first; the ones
        # it disfavoured (live sources, bound 10, patient gate, one source bank)
        # last, still run so real data, not the toy, rules them out.
        order = ["R_PF5E", "R_PF_OFF", "R_PH_GE", "R_PL8", "R_PF5E_FROZEN", "R_PF_OFF_R32",
                 "R_PF5E_R32", "R_PF_OFF_F10", "R_PH_GE_F10", "R_PF5E_F10", "R_PL8_F10",
                 "R_PH_G", "R_PH_GE_PF5E", "R_PF5E_s43", "R_PF5EL", "R_PF5", "R_PF5B",
                 "R_PF5E_LR10", "R_PF5E_W0", "R_PF5ED", "R_PF5EO", "R_PF10E", "R_PF5EP",
                 "R_PF5E_LIVE", "R_PF5E_LIVE_F10"]
        designs = {**ROUTER_PF, **ROUTER_PF_CONTROLS}
        for queue_order, alias in enumerate([*order, *(a for a in designs if a not in order)], 1):
            spec = designs[alias]
            seed = int(spec.get("seed", 42))
            entry = router_fix_entries(base, reference, checkpoints, selected_router,
                                       epochs=epochs, patience=patience,
                                       fits={alias: ("selected", True, True, ROUTER_FIX_WARMUP, 0.0, seed)},
                                       pooling=True)[0]
            full_own(entry["config"], spec)
            entry["queue_order"] = queue_order  # workers claim stage 8 in this order
            entries.append(entry)
    if task_pooling:
        if not router_pf:
            raise ValueError("Stage 9 repeats the stage-8 fits")
        entries += task_pooling_entries(base, reference, checkpoints, selected_router,
                                        epochs=epochs, patience=patience, seed=seeds[0])
    if encoder_lr:
        if not task_pooling:
            raise ValueError("Stage 10 repeats the stage-9 fits")
        entries += encoder_lr_entries(entries, seed=seeds[0])
    if encoder_lr_reading:
        if not encoder_lr:
            raise ValueError("Stage 11 repeats the stage-10 fits")
        entries += encoder_lr_entries(entries, seed=seeds[0], designs=ENCODER_LR_READING, first_order=-110)
    if task_ablation:
        if not encoder_lr_reading:
            raise ValueError("Stage 12 repeats the stage-11 fits")
        entries += task_ablation_entries(base, reference, checkpoints, selected_router,
                                         epochs=epochs, patience=patience, seed=seeds[0])
    if task_design:
        if not task_ablation:
            raise ValueError("Stage 13 repeats the stage-12 fits")
        entries += task_design_entries(base, reference, checkpoints, selected_router,
                                       epochs=epochs, patience=patience, seed=seeds[0])
    if task_seeds:
        if not task_design:
            raise ValueError("Stage 14 repeats the stage-13 fits")
        entries += task_seed_entries(base, reference, checkpoints, selected_router,
                                     epochs=epochs, patience=patience)
    if ladder_seeds:
        if not task_seeds:
            raise ValueError("Stage 15 repeats the stage-14 fits")
        entries += ladder_seed_entries(base, reference, checkpoints, selected_router,
                                       epochs=epochs, patience=patience)
    if data_efficiency and checkpoints.get("EA") and selected_router is not None:
        seed = seeds[0]
        local = next(e for e in entries if e["name"] == f"seed{seed}/L/J_all_diseases_all_outcomes")
        national = next(e for e in entries if e["name"] == f"seed{seed}/EA/J")
        for origin, source in (("L", local), ("EA", national)):
            for fraction in DATA_EFFICIENCY_FRACTIONS:
                config = copy.deepcopy(source["config"])
                config.pop("paper_run", None)
                config["training"]["train_patient_fraction"] = fraction
                config["training"]["train_fraction_seed"] = seed
                entries.append({
                    "name": f"seed{seed}/{origin}/J_frac{round(fraction * 100)}",
                    "phase": "data_efficiency",
                    "origin": origin,
                    "seed": seed,
                    "condition": {},
                    "config": config,
                })
    for entry in entries:
        entry["config"]["paper_run"] = {key: entry[key] for key in ("origin", "seed")}
        if entry["origin"] == "EA":
            entry["config"]["paper_run"]["direct_external_checkpoint"] = checkpoints[
                "E"
            ]
    return entries


def write_bundle(entries, output, *, provenance, selected_router=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Use a fresh bundle directory; existing configs are immutable"
        )
    output.mkdir(parents=True)
    manifest = []
    for entry in entries:
        path = output / "configs" / (entry["name"] + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# @package _global_\n" + yaml.safe_dump(entry["config"], sort_keys=False)
        )
        scientific_hash = hashlib.sha256(
            json.dumps(entry["config"], sort_keys=True, default=str).encode()
        ).hexdigest()
        manifest.append(
            {key: value for key, value in entry.items() if key != "config"}
            | {
                "regime": entry["config"]["supervision_factorial"]["regime"],
                "config_path": str(path.relative_to(output)),
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scientific_hash": scientific_hash,
                "run_directory": f"{entry['origin']}/seed{entry['seed']}/{scientific_hash[:20]}",
            }
        )
    report = {
        "status": "prepared_not_checkpoint_audited_or_launched",
        "provenance": provenance,
        "selected_router": selected_router,
        "runs": manifest,
        "phase_counts": {
            phase: sum(row["phase"] == phase for row in manifest)
            for phase in sorted({row["phase"] for row in manifest})
        },
    }
    (output / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_resolved_config(path, sequence_data_dir=None):
    """Freeze scientific env-dependent paths before generating immutable configs.

    ``sequence_data_dir`` replaces the config's ``paths.sequence_data_dir``; the
    vocabulary and every cohort's ``data_dir`` interpolate from it, so all point
    at the sequences the encoder was pretrained on.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    path = Path(path).resolve()
    OmegaConf.register_new_resolver("version", lambda: 0, replace=True)
    with initialize_config_dir(config_dir=str(path.parent), version_base="1.2"):
        overrides = ([f"paths.sequence_data_dir={Path(sequence_data_dir).resolve()}"]
                     if sequence_data_dir else [])
        config = compose(config_name=path.stem, overrides=overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    # Relative paths are relative to the invocation directory, as in the
    # existing runners. Save their absolute identities in the paper bundle.
    for key in ("vocabulary", "sequence_data_dir"):
        if resolved.get("paths", {}).get(key):
            resolved["paths"][key] = str(Path(resolved["paths"][key]).resolve())
    for cohort in resolved.get("cohorts", {}).values():
        for key in ("data_dir", "population_file"):
            if cohort.get(key):
                cohort[key] = str(Path(cohort[key]).resolve())
    directories = {
        cohort["data_dir"]
        for cohort in resolved.get("cohorts", {}).values()
        if cohort.get("data_dir")
    }
    if len(directories) == 1:
        outcome_directory = Path(next(iter(directories))) / "outcomes"
        for outcome in resolved.get("outcomes", {}).values():
            for key in (
                "outcome_path",
                "outcome_file",
                "filename",
                "competing_outcome_path",
                "competing_outcome_file",
                "eligibility_path",
                "eligibility_file",
            ):
                if outcome.get(key):
                    value = Path(outcome[key])
                    outcome[key] = str(
                        value.resolve()
                        if value.is_absolute()
                        else (outcome_directory / value).resolve()
                    )
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--release-registry", type=Path, required=True)
    parser.add_argument("--router-reference-config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--external-direct-checkpoint")
    parser.add_argument("--external-dapt-checkpoint")
    parser.add_argument("--router-lock", type=Path)
    parser.add_argument("--sequence-data-dir", type=Path,
                        help="Sequences the local encoder was pretrained on (overrides the base config).")
    parser.add_argument("--external-sequence-dir", type=Path,
                        help="Our patients tokenised in the national encoder's own format "
                        "(prepare_external_sequences); used by the national arms.")
    parser.add_argument("--no-data-efficiency", action="store_true",
                        help="Omit the training-fraction runs (local and national joint).")
    parser.add_argument("--no-router-controls", action="store_true",
                        help="Omit the routing decomposition controls from stage 2.")
    parser.add_argument("--router-fix", action="store_true",
                        help="Add the six router-fix fits (stage 3 onwards; lock addendum 2026-10-26).")
    parser.add_argument("--router-own", action="store_true",
                        help="Add the three own-outside router fits (stage 4 onwards).")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                        help="Epochs without tuning improvement before stopping (every arm)")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override physical batch size consistently in every arm",
    )
    parser.add_argument("--num-workers", type=int,
                        help="Override data-loading workers consistently in every arm")
    parser.add_argument("--sampling-ipw", action="store_true",
                        help="stage 5: J and R_AB with importance-weighted event sampling")
    parser.add_argument("--router-pt", action="store_true",
                        help="stage 7: task-routed candidates R_PT5 and R_PT3V")
    parser.add_argument("--task-design", action="store_true",
                        help="stage 13: shrinkage and diagnosis x outcome corrections for the reading design")
    parser.add_argument("--ladder-seeds", action="store_true",
                        help="stage 15: R_PF_OFF on seed 43 (the Table 2 ladder on a second seed)")
    parser.add_argument("--task-seeds", action="store_true",
                        help="stage 14: two more seeds of J_OA_OWN and a second seed of J")
    parser.add_argument("--task-ablation", action="store_true",
                        help="stage 12: seed repeat, capacity and ablations of the leading reading design")
    parser.add_argument("--external-encoder-lr", type=float,
                        help="encoder learning-rate multiplier for the national (E, EA) fits")
    parser.add_argument("--encoder-lr-reading", action="store_true",
                        help="stage 11: the stage-9 reading models with own corrections at the full encoder rate")
    parser.add_argument("--encoder-lr", action="store_true",
                        help="stage 10: J and R_AB with the encoder fine-tuned at a higher learning rate")
    parser.add_argument("--task-pooling", action="store_true",
                        help="stage 9: the joint model with task-conditioned attention pooling")
    parser.add_argument("--router-pf", action="store_true",
                        help="stage 8: full own corrections, design sweep around R_PF5E")
    parser.add_argument("--router-pp", action="store_true",
                        help="stage 6: the partial-pooling router R_PP and its seed-43 repeat")
    parser.add_argument("--routed-model",
                        help="primary routed model (e.g. R_AB_L): national routed fits "
                             "use its readout")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    base = load_resolved_config(args.base_config, args.sequence_data_dir)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        base["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("num-workers must be nonnegative")
        base.setdefault("hardware", {})["num_workers"] = args.num_workers
    registry = yaml.safe_load(args.release_registry.read_text())
    if (
        set(base["outcomes"]) != set(registry["outcomes"])
        or len(base["outcomes"]) != 69
    ):
        raise ValueError("Paper bundle requires the approved 69-outcome release roster")
    reference = load_resolved_config(args.router_reference_config)
    selected = json.loads(args.router_lock.read_text()) if args.router_lock else None
    checkpoints = {
        "L": args.encoder_checkpoint,
        "E": args.external_direct_checkpoint,
        "EA": args.external_dapt_checkpoint,
    }
    seeds = tuple(int(value) for value in args.seeds.split(","))
    entries = build_bundle(
        base,
        reference,
        checkpoints,
        seeds=seeds,
        epochs=args.epochs,
        patience=args.early_stopping_patience,
        selected_router=selected,
        router_controls=not args.no_router_controls,
        external_sequence_dir=args.external_sequence_dir,
        data_efficiency=not args.no_data_efficiency,
        router_fix=args.router_fix,
        router_own=args.router_own,
        routed_model=args.routed_model,
        sampling_ipw=args.sampling_ipw,
        router_pp=args.router_pp,
        router_pt=args.router_pt,
        router_pf=args.router_pf,
        task_pooling=args.task_pooling,
        encoder_lr=args.encoder_lr,
        encoder_lr_reading=args.encoder_lr_reading,
        external_encoder_lr=args.external_encoder_lr,
        task_ablation=args.task_ablation,
        task_design=args.task_design,
        task_seeds=args.task_seeds,
        ladder_seeds=args.ladder_seeds,
    )
    paths = [args.base_config, args.release_registry, args.router_reference_config]
    if args.router_lock:
        paths.append(args.router_lock)
    report = write_bundle(
        entries,
        args.output_dir,
        selected_router=selected,
        provenance={
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    )
    print(
        json.dumps(
            {"output": str(args.output_dir), "phase_counts": report["phase_counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
