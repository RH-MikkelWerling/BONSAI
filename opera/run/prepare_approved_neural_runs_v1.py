"""Prepare and verify matched architecture candidates for the approved panel.

No training is launched. All candidates retain horizon BCE, checkpoint metric,
roster, seed, metadata and sequence paths. The plain shared controls use
per-outcome macro loss; the other candidates use endpoint-group macro loss.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from opera.run.decide_outcome_panel import sha256_file
from opera.run.generate_sweep_configs import build_dapt_fresh_router_config


VARIANTS = (
    "shared", "time2vec_grouped_shared_bce",
    "fourier_plain_shared_bce", "time2vec_plain_shared_bce",
    "fourier_plain_factorized_bce", "time2vec_plain_factorized_bce",
    "family_trunks", "time2vec_family_trunks",
    "router_additive_e2e", "time2vec_router_additive_e2e",
    "router_interacting_e2e", "time2vec_router_interacting_e2e",
    "time2vec_router_disease_only_e2e", "time2vec_router_outcome_only_e2e",
    "time2vec_router_no_sources_e2e",
    "time2vec_router_interacting_scale02_e2e",
    "time2vec_router_interacting_scale03_e2e",
    "outcome_attention",
)
SOURCE_FACTORIAL_VARIANTS = tuple(
    name for name in VARIANTS
    if name not in {
        "time2vec_router_interacting_scale02_e2e",
        "time2vec_router_interacting_scale03_e2e",
    }
)
LEGACY_VARIANTS = tuple(
    name for name in SOURCE_FACTORIAL_VARIANTS
    if name not in {
        "time2vec_router_disease_only_e2e",
        "time2vec_router_outcome_only_e2e",
        "time2vec_router_no_sources_e2e",
    }
)


def candidate_configs(base: dict, registry: dict) -> dict[str, dict]:
    if base["competing_risk"]["objective"] != "horizon_bce":
        raise ValueError("Candidates require the approved multi-horizon BCE objective")
    if base["cross_outcome"]["aggregation"] != "endpoint_group_macro":
        raise ValueError("Family proposal must retain endpoint-group macro loss")
    if base["training"]["eval_monitor_metric"] != "val/cr_cif_auc_endpoint_group_macro":
        raise ValueError("Family proposal must retain endpoint-group checkpoint selection")
    if list(base["outcomes"]) != registry["outcomes"]:
        raise ValueError("Candidate base roster/order differs from release")
    if base.get("source_opera_ckpt") is not None or base["model"]["freeze_encoder"]:
        raise ValueError("Candidates must start from DAPT with a trainable encoder")
    result = {}
    for name in ("shared", "family_trunks", "outcome_attention"):
        config = copy.deepcopy(base)
        cr = config["competing_risk"]
        cr["readout_mode"] = "outcome_attention" if name == "outcome_attention" else "shared"
        cr["head_mode"] = "family_trunks" if name == "family_trunks" else "linear"
        result[name] = config
    result["time2vec_grouped_shared_bce"] = copy.deepcopy(result["shared"])
    result["time2vec_family_trunks"] = copy.deepcopy(result["family_trunks"])
    for name in ("fourier_plain_shared_bce", "time2vec_plain_shared_bce"):
        config = copy.deepcopy(result["shared"])
        # Plain per-outcome macro BCE is the ungrouped control. Preserve the
        # endpoint groups for the *same* tuning checkpoint metric.
        config["cross_outcome"]["aggregation"] = "macro"
        result[name] = config
    for name in ("fourier_plain_factorized_bce", "time2vec_plain_factorized_bce"):
        config = copy.deepcopy(base)
        config["cross_outcome"]["aggregation"] = "macro"
        result[name] = config
    router_template = build_dapt_fresh_router_config(registry)
    template_cr = router_template["competing_risk"]
    base_cr = base["competing_risk"]
    changed = {key for key in set(template_cr) | set(base_cr)
               if template_cr.get(key) != base_cr.get(key)}
    allowed = {
        "readout_mode", "fine_names", "fine_column", "adapter_bottleneck",
        "adapter_residual_scale", "adapter_home_floor", "adapter_event_tau",
        "adapter_query_mode", "adapter_query_hidden_dim",
    }
    if changed != allowed:
        raise ValueError(f"Router template changed unexpected fields: {changed ^ allowed}")
    for name, query_mode in (("router_additive_e2e", "additive"),
                             ("router_interacting_e2e", "interacting")):
        config = copy.deepcopy(base)
        config["competing_risk"] = copy.deepcopy(template_cr)
        config["competing_risk"]["adapter_query_mode"] = query_mode
        config["adapter_initialization"] = "dapt_fresh"
        config["source_opera_ckpt"] = None
        config["model"]["freeze_encoder"] = False
        result[name] = config
        result[f"time2vec_{name}"] = copy.deepcopy(config)
    for name, mode in (
        ("time2vec_router_disease_only_e2e", "disease_only"),
        ("time2vec_router_outcome_only_e2e", "outcome_only"),
        ("time2vec_router_no_sources_e2e", "none"),
    ):
        config = copy.deepcopy(result["time2vec_router_interacting_e2e"])
        config["competing_risk"]["adapter_source_mode"] = mode
        result[name] = config
    for name, scale in (
        ("time2vec_router_interacting_scale02_e2e", 0.2),
        ("time2vec_router_interacting_scale03_e2e", 0.3),
    ):
        config = copy.deepcopy(result["time2vec_router_interacting_e2e"])
        config["competing_risk"]["adapter_residual_scale"] = scale
        result[name] = config
    return result


def _inputs(release: Path, family_dir: Path) -> tuple[dict, dict, dict]:
    registry_path = release / "experiment_registry_primary_v2.yaml"
    base_path = release / "generated_configs" / "direct_cr_bce_lowrank_endpoint_groups_v1.yaml"
    family_path = family_dir / "outcome_training_families_primary_v1.yaml"
    family_config_path = family_dir / "direct_cr_bce_lowrank_endpoint_groups_training_families_v1.yaml"
    audit_path = family_dir / "training_family_audit_INTERNAL.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "training_family_proposal_not_active"
            or audit.get("n_outcomes") != 69 or audit.get("n_endpoint_groups") != 52
            or audit.get("n_training_families") != 12):
        raise ValueError("Expected successful 69-outcome, 52-group, 12-family audit")
    if (audit.get("source_sha256", {}).get("registry") != sha256_file(registry_path)
            or audit.get("source_sha256", {}).get("base_config") != sha256_file(base_path)
            or audit.get("families_sha256") != sha256_file(family_path)
            or audit.get("candidate_config_sha256") != sha256_file(family_config_path)):
        raise ValueError("Family proposal inputs differ from their audited hashes")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    family_config = yaml.safe_load(family_config_path.read_text(encoding="utf-8"))
    families = yaml.safe_load(family_path.read_text(encoding="utf-8"))
    expected = copy.deepcopy(base)
    expected["cross_outcome"]["outcome_families"] = families
    if family_config != expected:
        raise ValueError("Family config changes more than clinical family membership")
    return registry, family_config, {
        "registry_sha256": sha256_file(registry_path),
        "release_base_config_sha256": sha256_file(base_path),
        "family_audit_sha256": sha256_file(audit_path),
        "family_config_sha256": sha256_file(family_config_path),
    }


def prepare(release: Path, family_dir: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Candidate directory already exists: {output}")
    registry, base, sources = _inputs(release, family_dir)
    configs = candidate_configs(base, registry)
    output.mkdir(parents=True)
    files = {}
    for name, config in configs.items():
        path = output / f"{name}.yaml"
        path.write_text("# @package _global_\n" + yaml.safe_dump(config, sort_keys=False),
                        encoding="utf-8")
        files[name] = sha256_file(path)
    manifest = {"status": "approved_neural_architecture_candidates_not_launched",
                "release_dir": str(release.resolve()),
                "family_dir": str(family_dir.resolve()),
                "sources": sources, "configs": files,
                "loss_by_variant": {
                    name: config["cross_outcome"]["aggregation"]
                    for name, config in configs.items()
                },
                "selection": "val/cr_cif_auc_endpoint_group_macro",
                "source_checkpoint": "Variant prefix time2vec selects scaled-Time2Vec DAPT; all others Fourier DAPT; no outcome-head warm start"}
    (output / "candidate_manifest_INTERNAL.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"output_dir": str(output.resolve()), "variants": list(files),
            "status": manifest["status"]}


def verify(release: Path, family_dir: Path, output: Path, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    registry, base, sources = _inputs(release, family_dir)
    manifest = json.loads((output / "candidate_manifest_INTERNAL.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "approved_neural_architecture_candidates_not_launched"
            or manifest.get("release_dir") != str(release.resolve())
            or manifest.get("family_dir") != str(family_dir.resolve())
            or manifest.get("sources") != sources
            or frozenset(manifest.get("configs", {})) not in
            {frozenset(VARIANTS), frozenset(SOURCE_FACTORIAL_VARIANTS),
             frozenset(LEGACY_VARIANTS)}):
        raise ValueError("Candidate manifest is stale or incomplete")
    path = output / f"{variant}.yaml"
    if manifest["configs"][variant] != sha256_file(path):
        raise ValueError(f"Candidate config hash differs: {variant}")
    if not path.read_text(encoding="utf-8").startswith("# @package _global_\n"):
        raise ValueError(f"Candidate config lacks Hydra global package: {variant}")
    actual = yaml.safe_load(path.read_text(encoding="utf-8"))
    if actual != candidate_configs(base, registry)[variant]:
        raise ValueError(f"Candidate config differs from controlled variant: {variant}")
    return {"status": "candidate_config_verified", "variant": variant,
            "config": str(path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--family-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-variant", choices=VARIANTS)
    args = parser.parse_args()
    result = (verify(args.release_dir, args.family_dir, args.output_dir, args.verify_variant)
              if args.verify_variant else
              prepare(args.release_dir, args.family_dir, args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
