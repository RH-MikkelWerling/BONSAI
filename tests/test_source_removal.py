"""Borrowing design: source terms, one evaluation, leave-one-in/out, claims."""

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from opera.modules.networks.source_anchored_readout import (
    SourceAnchoredReadout,
    borrowing_design,
)
from opera.run.paper_prediction_queue import bundle_runs, evaluate_row
from opera.run.source_removal import (
    ALL_OUTCOMES,
    endpoint_groups,
    hypothesis_table,
    hypothesis_test,
    lineage_summary,
    reading,
    shrink,
    source_effects,
    borrowing_levels,
    variant_file,
)
from test_paper_workflow import bundle, complete_run

OUTCOMES = {"overall_survival": "Disease course and later malignancy",
            "sepsis": "Infection and antimicrobial use"}


def test_dropping_a_source_term_changes_nothing_else():
    torch.manual_seed(5)
    readout = SourceAnchoredReadout(12, ["CLL", "DLBCL", "MM"], ["a", "b"], bottleneck=3,
                                    query_mode="interacting", query_hidden_dim=16,
                                    source_mode="disease_only")
    readout.set_training_support([10] * 5, [10] * 5, [10] * 5)
    torch.nn.init.normal_(readout.up, std=0.5)
    readout.eval()
    patient, fine = torch.randn(4, 12), torch.tensor([0, 1, 2, 1])
    with torch.no_grad():
        output = readout(patient, fine)
        base, terms = readout.disease_terms(patient, fine)
        labels, masks = borrowing_design(readout.fine_names, fine)
    assert torch.allclose(base + terms.sum(2), output, atol=1e-6)
    assert terms.abs().sum() > 0
    assert len(labels) == 2 + 2 * 3 and masks.shape == (8, 4, 3)
    variant = {label: base + torch.einsum("bd,bodh->boh", mask, terms)
               for label, mask in zip(labels, masks)}
    assert torch.allclose(variant["full"], output, atol=1e-6)
    # Leaving MM out drops exactly MM's term; the other weights are untouched.
    assert torch.allclose(variant["out:MM"], output - terms[:, :, 2], atol=1e-6)
    own = terms[torch.arange(4), :, fine]
    assert torch.allclose(variant["none"], base + own, atol=1e-6)
    # MM alone for a DLBCL patient: own term plus MM's; for an MM patient, none.
    assert torch.allclose(variant["in:MM"][1], (base + own + terms[:, :, 2])[1], atol=1e-6)
    assert torch.allclose(variant["in:MM"][2], variant["none"][2], atol=1e-6)


def test_the_design_is_one_evaluation_in_its_own_directory(tmp_path, monkeypatch):
    import opera.run.paper_prediction_queue as queue

    _, rows = bundle_runs(bundle(tmp_path))
    row, path, config = next(item for item in rows if item[0]["phase"] == "supervision")
    complete_run(tmp_path / "runs" / row["run_directory"],
                 config["training"]["eval_monitor_metric"])
    option = Namespace(runs_dir=tmp_path / "runs", output_dir=tmp_path / "predictions",
                       split="tuning", execute=True, horizons="30,365", batch_size=1,
                       num_workers=0, device="cpu", export_routes=False)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        (output / "direct_cr_predictions.parquet").write_bytes(b"predictions")
        if "--borrowing-design" in command:
            (output / "source_removal").mkdir()
            (output / "source_removal" / "design.json").write_text("{}")

    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    result = evaluate_row(row, path, config, option, borrowing_design=True)
    assert result["status"] == "evaluated" and result["borrowing_design"]
    assert Path(result["prediction_directory"]).name == "tuning_source_removal"
    assert "--borrowing-design" in commands[0] and "--export-embeddings" not in commands[0]
    # The ordinary evaluation is a separate, untouched directory.
    assert evaluate_row(row, path, config, option)["status"] == "evaluated"
    assert "--borrowing-design" not in commands[1]
    assert evaluate_row(row, path, config, option, borrowing_design=True)["status"] \
        == "reused_verified_evaluation"


def design(tmp_path, n=300, seed=0):
    """CLL borrows from MM: without MM its CIF is uninformative. MM borrows nothing."""
    rng = np.random.default_rng(seed)
    base, variants, members = [], {}, []
    labels = ["full", "none", "in:CLL", "in:MM", "out:CLL", "out:MM"]
    for cohort in ("CLL", "MM"):
        for i in range(n):
            subject = f"{cohort}{i}"
            members.append({"subject_id": subject, "cohort_fine": cohort})
            for outcome in OUTCOMES:
                risk = rng.uniform(0.05, 0.7)
                event = int(rng.uniform() < risk)
                base.append({"subject_id": subject, "cohort": "grouped",
                             "outcome": outcome,
                             "time_days": rng.uniform(5, 300) if event else 800.0,
                             "event": event})
                blind = rng.uniform(0.05, 0.7)
                for label in labels:
                    uses_mm = label in ("full", "in:MM", "out:CLL")
                    value = risk if cohort == "MM" or uses_mm else blind
                    variants.setdefault(label, []).append(value)
    directory = tmp_path / "tuning_source_removal"
    for ablation in ("mean", "zero"):
        (directory / "source_removal" / ablation).mkdir(parents=True)
    frame = pd.DataFrame(base)
    frame.to_parquet(directory / "direct_cr_predictions.parquet", index=False)
    for label, values in variants.items():
        cif = frame[["subject_id", "outcome"]].copy()
        for h in (30, 60, 90, 180, 365, 730):
            cif[f"cif_{h}d"] = values
        for ablation in ("mean", "zero"):
            cif.to_parquet(directory / "source_removal" / ablation
                           / f"{label.replace(':', '_')}.parquet", index=False)
    (directory / "source_removal" / "design.json").write_text(json.dumps({"variants": labels}))
    return directory, pd.DataFrame(members)


def test_leave_one_in_and_out_bracket_a_source(tmp_path):
    directory, members = design(tmp_path)
    effects, cells, draws = source_effects(directory, members, "MM", OUTCOMES,
                                           split="tuning", replicates=200)
    rows = effects.set_index(["target", "endpoint_group"])
    cll = rows.loc["CLL"]
    assert set(cll.index) == {ALL_OUTCOMES, *OUTCOMES.values()}
    assert (cll.in_effect > 1).all() and (cll.in_lower > 0).all()  # AUC points
    assert (cll.out_effect > 1).all() and (cll.out_lower > 0).all()
    assert np.isclose(cll.loc[ALL_OUTCOMES, "in_effect"],
                      cll.drop(ALL_OUTCOMES).in_effect.mean())
    # A disease alone with itself is no borrowing: no leave-one-in row.
    assert rows.loc["MM"].in_effect.isna().all()
    assert np.allclose(rows.loc["MM"].out_effect, 0)
    assert (cells.source == "MM").all()
    assert {"in_ipa_effect", "out_ipa_lower"} <= set(effects)
    assert set(draws) == {("CLL", d) for d in [ALL_OUTCOMES, *OUTCOMES.values()]}
    assert all(len(v) == 200 for v in draws.values())
    unused, _, _ = source_effects(directory, members, "CLL", OUTCOMES, split="tuning",
                                  replicates=200)
    assert np.allclose(unused.set_index("target").loc["MM"].in_effect, 0)
    # References scored by ordinary evaluations, rows in another order: the
    # joint model behaves like no borrowing, the own-adapter router like full.
    references = {}
    for name, variant in (("J", "none"), ("RDH", "full")):
        frame = pd.read_parquet(variant_file(directory, variant)).sample(frac=1, random_state=3)
        references[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(references[name], index=False)
    levels, _ = borrowing_levels(directory, members, OUTCOMES, split="tuning",
                                 references=references, replicates=200)
    assert set(levels.metric) == {"auc", "ipa"}
    levels = levels.loc[(levels.endpoint_group == ALL_OUTCOMES) & (levels.metric == "auc")]
    levels = levels.set_index(["contrast", "target"])
    # Every disease weighing equally: half of CLL's dependence, MM adds none.
    assert np.isclose(levels.loc[("dependence", "All diseases"), "effect"],
                      levels.loc[("dependence", "CLL"), "effect"] / 2)
    assert levels.loc[("dependence", "CLL"), "effect"] > 1
    assert np.isclose(levels.loc[("dependence", "MM"), "effect"], 0)
    assert levels.loc[("routing", "CLL"), "effect"] > 1
    assert np.isclose(levels.loc[("trained_borrowing", "CLL"), "effect"], 0)
    # Without reference predictions only the dependence is estimated.
    alone, _ = borrowing_levels(directory, members, OUTCOMES, split="tuning",
                                references={"J": tmp_path / "missing.parquet"},
                                replicates=50)
    assert set(alone.contrast) == {"dependence"}


def test_paper_domains_cover_the_test_outcomes():
    domains = endpoint_groups()
    assert all(domains[o] == g for o, g in OUTCOMES.items())


def test_readings_follow_the_two_extremes():
    assert reading(0.5, 0.5) == "main source"
    assert reading(0.5, -0.1) == "useful, redundant with other sources"
    assert reading(-0.1, 0.5) == "useful in combination"
    assert reading(np.nan, -0.1) == "contribution not shown"


def map_rows(effects, se, *, same_group=True):
    return pd.DataFrame([{"target": f"T{i}", "source": "S", "endpoint_group": ALL_OUTCOMES,
                          "target_group": "g", "source_group": "g" if same_group else "h",
                          "effect": e, "lower": e - 1.96 * se, "upper": e + 1.96 * se,
                          "se": se}
                         for i, e in enumerate(effects)])


def test_noisy_null_pairs_shrink_to_zero_and_real_ones_are_kept():
    rng = np.random.default_rng(1)
    effects = [*rng.normal(0, 0.2, 60), 3.0, 2.5]
    table = shrink(map_rows(effects, se=0.2))
    assert np.allclose(table.raw_effect, effects)
    assert table.effect.iloc[:60].abs().max() < 0.15  # nulls pulled to zero
    assert table.effect.iloc[60] > 2.7 and table.lower.iloc[61] > 1.5  # real ones kept
    assert (table.pool == "pairs").all() and table.prior_sd.notna().all()
    own = map_rows([3.0, 3.0, 3.0], se=1.0).assign(source=["T0", "T1", "T2"])
    assert np.allclose(shrink(own).effect, 3.0) and (shrink(own).pool == "own").all()


def test_pooling_ignores_lineage():
    within = map_rows([0.1] * 5, se=0.3)
    across = map_rows([0.1] * 5, se=0.3, same_group=False).assign(
        target=[f"U{i}" for i in range(5)])
    table = shrink(pd.concat([within, across], ignore_index=True))
    assert np.allclose(table.effect.iloc[:5], table.effect.iloc[5:])


def pair(target, source, effect, se, out):
    return {"target": target, "source": source, "endpoint_group": ALL_OUTCOMES,
            "in_effect": effect, "in_lower": effect - 1.96 * se,
            "in_upper": effect + 1.96 * se, "in_p_value": 0.5, "in_n_cells": 10,
            "out_effect": out, "out_lower": out - 0.5, "out_upper": out + 0.5}


def test_hypotheses_are_claimed_on_leave_one_in():
    raw = pd.DataFrame([pair("CLL", "MM", 2.0, 0.3, 0.1),
                        pair("HL", "AMYLOIDOSIS", 0.05, 0.1, 0.0)])
    plan = {"hypotheses": [{"target": "CLL", "source": "MM", "domain": ALL_OUTCOMES,
                            "strength": "moderate", "rationale": "r", "evidence": ["a", "b"]},
                           {"target": "LBL", "source": "BL", "domain": ALL_OUTCOMES}],
            "negative_controls": [{"target": "HL", "source": "AMYLOIDOSIS",
                                   "domain": ALL_OUTCOMES, "rationale": "none"}]}
    table = hypothesis_table(raw, plan).set_index("target")
    assert table.loc["CLL", "claim"] == "better, above margin"
    assert table.loc["CLL", "reading"] == "useful, redundant with other sources"
    assert table.loc["CLL", "evidence"] == "a; b"
    assert table.loc["HL", "claim"] == "equivalent" and table.loc["HL", "kind"] == "negative control"
    assert table.loc["LBL", "claim"] == "not estimable"


def test_masked_hazards_through_the_routed_network():
    from test_attribution_real_model import prepared, routed_model

    model = routed_model()
    batch = prepared(model)[0]
    with torch.no_grad():
        pooled = model.get_objective_representations(batch)
        fine = batch["cohort_fine_index"]
        hazards = model.competing_risk_log_hazards(pooled, fine)
        labels, masks = borrowing_design(model.source_adapter_readout.fine_names, fine)
        variants = model.competing_risk_log_hazard_masks(pooled, fine, masks)
    assert variants.shape == (len(labels), *hazards.shape)
    torch.testing.assert_close(variants[labels.index("full")], hazards, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(variants[labels.index("none")], hazards, atol=1e-4)
    # Mean ablation: a source switched off is replaced by its mean term. If the
    # mean equals this patient's own terms, switching off changes nothing.
    with torch.no_grad():
        _, terms = model.borrowing_hazard_terms(pooled, fine)
        n_fine = len(model.source_adapter_readout.fine_names)
        means = torch.zeros((n_fine, *terms.shape[1:]))
        means[fine[0]] = terms[0]
        ablated = model.competing_risk_log_hazard_masks(pooled, fine, masks, means=means)
    torch.testing.assert_close(ablated[labels.index("none"), 0], hazards[0], atol=1e-5,
                               rtol=1e-5)


def test_the_prespecified_pairs_are_tested_as_a_set():
    rng = np.random.default_rng(0)
    rows = [pair(f"T{i}", "S", rng.normal(0, 0.1), 0.1, 0.0) for i in range(200)]
    rows += [pair("CLL", "MM", 0.4, 0.1, 0.0), pair("SLL", "CLL", 0.5, 0.1, 0.0),
             pair("HL", "AMYLOIDOSIS", 0.02, 0.1, 0.0)]
    plan = {"hypotheses": [{"target": "CLL", "source": "MM", "domain": ALL_OUTCOMES},
                           {"target": "SLL", "source": "CLL", "domain": ALL_OUTCOMES},
                           {"target": "LBL", "source": "BL", "domain": ALL_OUTCOMES}],
            "negative_controls": [{"target": "HL", "source": "AMYLOIDOSIS",
                                   "domain": ALL_OUTCOMES}]}
    test = hypothesis_test(pd.DataFrame(rows), plan, permutations=2000).set_index("kind")
    assert test.loc["hypothesis", "n_pairs"] == 2  # LBL <- BL not estimable
    assert test.loc["hypothesis", "difference"] > 0.3
    assert test.loc["hypothesis", "p_value"] < 0.01
    assert test.loc["negative control", "p_value"] > 0.1


def test_lineage_summary_keeps_the_pairing_of_the_bootstrap():
    rng = np.random.default_rng(2)
    shared = rng.normal(0, 1, 500)  # one patient resample moves every pair alike
    groups = {"A1": "a", "A2": "a", "B1": "b", "B2": "b"}
    effects, draws = {}, {}
    for target in groups:
        for source in groups:
            if target == source:
                continue
            true = 0.5 if groups[target] == groups[source] else 0.0
            effects[(target, source, "Infection")] = true
            draws[(target, source, "Infection")] = true + 0.3 * shared + rng.normal(0, 0.05, 500)
    table = lineage_summary(effects, draws, groups).set_index("kind")
    assert np.isclose(table.loc["within", "effect"], 0.5)
    assert np.isclose(table.loc["across", "effect"], 0.0)
    # The shared resampling noise cancels in the difference, so it is precise.
    difference = table.loc["within - across"]
    assert difference.lower > 0.4 and difference.upper < 0.6
    assert table.loc["within", "upper"] - table.loc["within", "lower"] > 1.0
