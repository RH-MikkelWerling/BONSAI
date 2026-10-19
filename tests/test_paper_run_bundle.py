"""Staged run readiness, conditional queues and direct external export."""

import copy
import json

import pytest
import torch

from bonsai.run.export_external_encoder import export_external_encoder
from bonsai.functional.checkpointing import extract_encoder_state_dict
from opera.run.prepare_paper_run_bundle import build_bundle, write_bundle
from opera.run.run_paper_queue import audit_encoder, file_hash, select_runs
from test_supervision_factorial import base_config
from test_pretrain_transfer import _bundle


def reference():
    return {
        "competing_risk": {
            "readout_mode": "source_anchored_adapters",
            "adapter_residual_scale": 0.7,
            "adapter_home_floor": 0.05,
            "adapter_event_tau": 10.0,
            "adapter_shared_logit_init": 0.0,
        },
        "training": {"adapter_optimizer_epsilon": 1e-8},
    }


def test_bundle_preserves_inputs_and_separates_supervision_from_sources(tmp_path):
    base = base_config()
    original = copy.deepcopy(base)
    entries = build_bundle(
        base, reference(), {"L": "/future/best.ckpt"}, seeds=(42, 43)
    )
    assert base == original
    assert all(
        entry["config"]["training"]["eval_monitor_metric"]
        == "val/cr_horizon_bce_endpoint_group_macro"
        for entry in entries
    )
    assert all(
        not entry["config"]["competing_risk"]["curriculum"]["enabled"]
        for entry in entries
    )
    assert all(
        entry["seed"] == 42 for entry in entries if entry["phase"] != "supervision"
    )
    report = write_bundle(entries, tmp_path / "bundle", provenance={})
    source = select_runs(report, "source_screen")
    assert len(source) == 4
    assert all(entry["condition"] == {} for entry in source)
    with pytest.raises(ValueError, match="tuning decisions"):
        select_runs(report, "rank_screen")
    ranks = select_runs(report, "rank_screen", {"source_mode": "disease_only"})
    assert len(ranks) == 2
    query = select_runs(
        report, "query_screen", {"source_mode": "disease_only", "rank": 8}
    )
    assert len(query) == 3
    assert set(row["scientific_hash"] for row in ranks) & set(
        row["scientific_hash"] for row in query
    )
    # Same scientific candidate in two phases uses identical config and run root.
    shared_hash = next(
        iter(
            set(row["scientific_hash"] for row in ranks)
            & set(row["scientific_hash"] for row in query)
        )
    )
    matching = [row for row in ranks + query if row["scientific_hash"] == shared_hash]
    assert matching[0]["run_directory"] == matching[1]["run_directory"]
    shards = [
        select_runs(report, "supervision", shard_count=3, shard_index=i)
        for i in range(3)
    ]
    assert sum(map(len, shards)) == len(select_runs(report, "supervision"))
    assert len({row["scientific_hash"] for shard in shards for row in shard}) == sum(
        map(len, shards)
    )
    with pytest.raises(FileExistsError):
        write_bundle(entries, tmp_path / "bundle", provenance={})


def test_final_external_matrix_requires_lock_and_uses_same_shared_protocol(tmp_path):
    lock = {
        "source_mode": "disease_only",
        "rank": 8,
        "query_mode": "interacting",
        "query_width": 128,
    }
    entries = build_bundle(
        base_config(),
        reference(),
        {"L": "/L.ckpt", "E": "/E.ckpt", "EA": "/EA.ckpt"},
        seeds=(42,),
        selected_router=lock,
    )
    external = [row for row in entries if row["phase"] == "external"]
    assert len(external) == 4
    assert {row["origin"] for row in external} == {"E", "EA"}
    assert all(
        row["config"]["dapt_ckpt"] == f"/{row['origin']}.ckpt" for row in external
    )
    assert all(
        row["config"]["competing_risk"]["adapter_source_mode"] == "disease_only"
        for row in external
        if row["name"].endswith("/R")
    )
    efficiency = [row for row in entries if row["phase"] == "data_efficiency"]
    assert sorted(row["name"] for row in efficiency) == [
        f"seed42/{o}/J_frac{f}" for o in ("EA", "L") for f in (10, 25, 50)]
    assert {row["config"]["dapt_ckpt"] for row in efficiency} == {"/L.ckpt", "/EA.ckpt"}
    assert {row["config"]["training"]["train_patient_fraction"] for row in efficiency} == {
        0.1, 0.25, 0.5}


def test_external_export_remaps_rows_without_heads_optimizer_or_dapt(tmp_path):
    source, source_vocab, target_vocab, checkpoint = _bundle(tmp_path)
    output = tmp_path / "direct"
    exported = export_external_encoder(
        checkpoint, source_vocab, target_vocab, output, minimum_token_coverage=0.5
    )
    payload = torch.load(exported["checkpoint"], weights_only=False)
    state = extract_encoder_state_dict(payload["state_dict"])
    assert torch.equal(
        state["embeddings.code_embedding.weight"][2],
        source.embeddings.code_embedding.weight[1],
    )
    assert not any("head" in name for name in state)
    assert "optimizer_states" not in payload
    assert exported["local_dapt_performed"] is False
    config = {
        "dapt_ckpt": exported["checkpoint"],
        "paths": {"vocabulary": str(target_vocab)},
    }
    assert (
        audit_encoder(config, "E", expected_width=8)["status"]
        == "strict_load_and_forward_passed"
    )
    with pytest.raises(FileExistsError):
        export_external_encoder(
            checkpoint, source_vocab, target_vocab, output, minimum_token_coverage=0.5
        )
    # Completion sidecar is mandatory for a local fit; a selected checkpoint
    # alone may exist while pretraining is still running.
    with pytest.raises(ValueError):
        audit_encoder(
            {"dapt_ckpt": str(checkpoint), "paths": {"vocabulary": str(source_vocab)}},
            "L",
            expected_width=8,
        )


def test_dapt_requires_complete_training_and_the_exact_direct_export(tmp_path):
    _, source_vocab, target_vocab, checkpoint = _bundle(tmp_path)
    direct = export_external_encoder(
        checkpoint,
        source_vocab,
        target_vocab,
        tmp_path / "E",
        minimum_token_coverage=0.5,
    )
    payload = torch.load(direct["checkpoint"], weights_only=False)
    metadata = payload["hyper_parameters"]["checkpoint_metadata"]
    metadata["training_stage"] = "domain_adaptive_pretraining"
    metadata["source_pretraining_transfer"]["checkpoint_sha256"] = file_hash(
        direct["checkpoint"]
    )
    adapted = tmp_path / "EA"
    adapted.mkdir()
    torch.save(payload, adapted / "best.ckpt")
    sidecar = adapted / "checkpoint_metadata.json"
    sidecar.write_text(json.dumps({"extra": {"completed_epochs": 9}}))
    config = {
        "dapt_ckpt": str(adapted / "best.ckpt"),
        "paths": {"vocabulary": str(target_vocab)},
        "paper_run": {"direct_external_checkpoint": direct["checkpoint"]},
    }
    with pytest.raises(ValueError, match="require 10"):
        audit_encoder(config, "EA", expected_width=8)
    sidecar.write_text(json.dumps({"extra": {"completed_epochs": 10}}))
    assert (
        audit_encoder(config, "EA", expected_width=8)["status"]
        == "strict_load_and_forward_passed"
    )
    metadata["source_pretraining_transfer"]["checkpoint_sha256"] = "wrong"
    torch.save(payload, adapted / "best.ckpt")
    with pytest.raises(ValueError, match="exact direct external export"):
        audit_encoder(config, "EA", expected_width=8)


def test_width_probe_accepts_a_named_milestone_of_a_running_fit(tmp_path):
    """The 256 run is still training when the 512 probe ends: its epoch-10
    milestone is accepted by the probe only, and its epoch must match its name."""
    from test_pretrain_transfer import BonsaiPretrain

    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4, "DIAGNOSIS//A": 5}
    vocab_path = tmp_path / "vocabulary.pt"
    torch.save(vocab, vocab_path)
    model = BonsaiPretrain(vocab_size=len(vocab), max_seqlen=16, hidden_size=64, num_layers=4,
                           num_attention_heads=1, bias=False, dropout=0.0,
                           attention_dropout=0.0, causal=True, attn_type="sdpa",
                           abspos_encoding="scaled_time2vec")
    payload = {"hyper_parameters": {"model_config": model.hparams},
               "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
               "epoch": 9}  # Lightning stores the zero-based epoch: ten completed
    milestone = tmp_path / "version_0" / "epoch_010.ckpt"
    milestone.parent.mkdir()
    torch.save(payload, milestone)
    config = {"dapt_ckpt": str(milestone), "paths": {"vocabulary": str(vocab_path)}}
    assert audit_encoder(config, "L", expected_width=64, minimum_local_epochs=10,
                         allow_milestone=True)["status"] == "strict_load_and_forward_passed"
    with pytest.raises(ValueError, match="sidecar"):  # paper runs still need the sidecar
        audit_encoder(config, "L", expected_width=64, minimum_local_epochs=10)
    with pytest.raises(ValueError, match="require 20"):
        audit_encoder(config, "L", expected_width=64, minimum_local_epochs=20,
                      allow_milestone=True)
    renamed = milestone.with_name("epoch_011.ckpt")
    milestone.rename(renamed)
    with pytest.raises(ValueError, match="records 10"):
        audit_encoder({**config, "dapt_ckpt": str(renamed)}, "L", expected_width=64,
                      minimum_local_epochs=10, allow_milestone=True)


def test_width_decision_waits_for_the_reference_probe(tmp_path):
    from opera.run.width_probe import compare

    (tmp_path / "w512_e10").mkdir()
    decision = compare(tmp_path)
    assert decision["decision"].startswith("pending")
    assert (tmp_path / "width_decision.json").is_file()


def test_width_decision_compares_macro_auc_at_each_selected_epoch(tmp_path, monkeypatch):
    """Selection is on tuning loss; the rule compares macro AUC at that epoch."""
    import bonsai.functional.checkpointing as checkpointing
    from opera.run.width_probe import compare

    curves = {"w256_e10": [(0, 0.60), (1, 0.66), (2, 0.65)],
              "w512_e10": [(0, 0.62), (1, 0.69), (2, 0.70)]}
    best = {"w256_e10": 1, "w512_e10": 1}  # the loss-selected epochs
    for label, curve in curves.items():
        logs = tmp_path / label / "contrastive_multicohort_runs" / "version_0"
        logs.mkdir(parents=True)
        rows = ["epoch,step,val/cr_cif_auc_macro,val/loss"]
        rows += [f"{e},{10 * e},{auc},0.3" for e, auc in curve]
        (logs / "metrics.csv").write_text("\n".join(rows) + "\n")
        (tmp_path / label / "encoder_audit.json").write_text("{}")
    monkeypatch.setattr(checkpointing, "verified_training_selection", lambda run: {
        "best_epoch": best[run.name], "monitor": "val/cr_horizon_bce_endpoint_group_macro",
        "mode": "min", "best_model_score": 0.2})
    decision = compare(tmp_path)
    assert decision["gain_w512_e10_vs_w256_e10"] == 0.03  # 0.69 - 0.66, not epoch 2
    assert decision["decision"] == "switch to w512_e10"
    best["w512_e10"] = 0  # 0.62 - 0.66: below the threshold
    assert compare(tmp_path)["decision"] == "keep 256"


def test_probe_curves_report_the_loss_minimum_one_based(tmp_path):
    from opera.run.width_probe import curves

    logs = tmp_path / "w512_e10" / "contrastive_multicohort_runs" / "version_0"
    logs.mkdir(parents=True)
    logs.joinpath("metrics.csv").write_text(
        "epoch,val/cr_horizon_bce_endpoint_group_macro,val/cr_cif_auc_macro,train/loss\n"
        "0,0.30,0.66,\n0,,,0.9\n1,0.25,0.69,\n2,0.27,0.70,\n")
    report = curves(tmp_path)["w512_e10"]
    assert report["best_epoch_by_loss"] == 2 and report["auc_at_best"] == 0.69
    assert report["epochs_validated"] == 3
