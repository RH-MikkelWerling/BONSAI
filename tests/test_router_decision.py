"""The router rules: gate and noise floor, defaults unless clearly better."""

import zlib

import numpy as np
import pandas as pd
import pytest

from opera.evaluation.secondary_analysis import normalize_predictions
from opera.run.router_decision import decide, roles


def predictions(noise_sd, *, n=600, seed=0):
    """Tuning predictions per model alias: the true risk plus noise of the given
    size, so a smaller noise is a better and still calibrated model. N shares
    J's noise draw (the same model, initialised differently), A shares D's."""
    rng = np.random.default_rng(seed)
    rows = []
    for cohort in ("CLL", "DLBCL"):
        risk = rng.uniform(0.05, 0.6, n)
        event = np.where(rng.uniform(size=n) < risk, 1, 0)
        time = np.where(event == 1, rng.uniform(10, 700, n), 800.0)
        for model, sd in noise_sd.items():
            base = {"N": "J", "A": "D"}.get(model, model)
            z = np.random.default_rng(zlib.crc32(f"{base}{cohort}".encode())).normal(0, 1, n)
            probability = np.clip(risk + sd * z, 0.01, 0.99)
            for outcome in ("infection", "death"):
                for horizon in (365, 730):
                    rows.append(pd.DataFrame({
                        "subject_id": [f"{cohort}{i}" for i in range(n)],
                        "split": "tuning", "seed": 42, "cohort": cohort,
                        "outcome": outcome, "horizon_days": horizon, "model": model,
                        "probability": probability, "data_contract": "test",
                        "time_days": time, "event": event}))
    frame, _ = normalize_predictions(pd.concat(rows, ignore_index=True), "cif_ipcw")
    return frame


def run(noise_sd, **kwargs):
    return decide(predictions(noise_sd), joint="J", null="N", options=["D", "A"],
                  replicates=200, **kwargs)


def test_the_best_option_is_chosen_even_by_a_small_margin():
    result = run({"J": 0.2, "N": 0.2, "D": 0.2, "A": 0.19})
    assert result["noise"]["floor_auc"] < 0.005 and result["noise"]["machinery_check"]["passed"]
    gain = result["options"][1]["auc"]["gain"]
    assert 0 < gain < 0.01 and result["chosen"] == "A" and result["status"] == "best"


def test_options_within_the_noise_floor_go_to_the_simplest():
    result = run({"J": 0.2, "N": 0.25, "D": 0.2, "A": 0.199})
    assert result["noise"]["floor_auc"] > abs(result["options"][1]["auc"]["gain"])
    assert result["chosen"] == "D"


def test_a_much_worse_calibrated_option_is_not_eligible():
    rows = predictions({"J": 0.2, "N": 0.2, "D": 0.2, "A": 0.17})
    worse = rows.model.eq("A")
    rows.loc[worse, "probability"] = (rows.loc[worse, "probability"] * 3).clip(0, 0.999)
    result = decide(rows, joint="J", null="N", options=["D", "A"], replicates=200)
    assert not result["options"][1]["eligible"] and result["chosen"] == "D"


def test_options_are_ordered_simplest_first_and_need_earlier_decisions():
    assert roles("source_screen", {})[0][0] == "disease_only"
    options = roles("query_screen", {"source_mode": "disease_only", "rank": 16})
    assert options[0] == ("disease_only", 16, "additive", 128) and len(options) == 3
    assert [key[1] for key in roles("rank_screen", {"source_mode": "both"})] == [8, 16, 32]
    with pytest.raises(ValueError, match="source_mode"):
        roles("rank_screen", {})


def test_intervals_survive_cells_without_cases_in_some_resamples():
    rows = predictions({"J": 0.2, "N": 0.2, "D": 0.2, "A": 0.19}, n=40)
    result = decide(rows, joint="J", null="N", options=["D", "A"], replicates=200)
    a = result["options"][1]
    assert np.isfinite(a["auc"]["lower"]) and np.isfinite(a["vs_joint"]["auc"]["upper"])


def test_init_rule_keeps_r_unless_rsi_is_clearly_better():
    # R_L primary, RSI_L challenger: a gain within the noise floor keeps R.
    rows = predictions({"J": 0.2, "N": 0.25, "D": 0.2, "A": 0.199})
    rows["model"] = rows.model.map({"J": "J_L", "N": "N_L", "D": "R_L", "A": "RSI_L"})
    result = decide(rows, joint="J_L", null="N_L", options=["R_L", "RSI_L"], replicates=200)
    assert result["chosen"] == "R_L"


def test_routing_diagnostics_split_own_and_borrowed_weight(tmp_path):
    import json
    import math

    from opera.run.router_decision import routing_diagnostics

    # One patient, CLL, predicting infection: 0.1 shared, 0.6 on CLL, 0.1 on MM,
    # 0.2 on its own outcome.
    # The export names the disease group (CLL_SLL); the own disease is found by
    # the fine index.
    pd.DataFrame([{
        "subject_id": 1, "cohort": "CLL_SLL", "split": "tuning", "outcome": "infection",
        "fine_index": 1, "shared_weight": 0.1,
        "disease_weights": json.dumps({"disease:MM": 0.1, "disease:CLL": 0.6}),
        "outcome_weights": json.dumps({"outcome:infection": 0.2, "outcome:sepsis": 0.0}),
    }]).to_parquet(tmp_path / "routes.parquet")
    result = routing_diagnostics(tmp_path / "routes.parquet")
    assert result["top_sources"][0] == {"source": "disease:CLL", "share_of_rows": 1.0}
    summary = result["overall"]
    assert summary["own_disease"] == 0.6 and summary["borrowed_disease"] == 0.1
    assert summary["own_outcome"] == 0.2 and summary["borrowed_outcome"] == 0.0
    entropy = -sum(w * math.log(w) for w in (0.1, 0.6, 0.1, 0.2))
    assert summary["entropy"] == pytest.approx(round(entropy, 4))
    assert summary["effective_sources"] == pytest.approx(round(math.exp(entropy), 4))


def _routes(tmp_path, rows):
    import json

    diseases = ["disease:A", "disease:B", "disease:C", "disease:D"]
    outcomes = ["outcome:os", "outcome:inf"]
    records = []
    for fine_index, outcome, weights in rows:
        records.append({"fine_index": fine_index, "outcome": outcome, "shared_weight": 0.0,
                        "disease_weights": json.dumps(dict(zip(diseases, weights[:4]))),
                        "outcome_weights": json.dumps(dict(zip(outcomes, weights[4:])))})
    pd.DataFrame(records).to_parquet(tmp_path / "routes.parquet")
    (tmp_path / "support.json").write_text(json.dumps(
        {"sources": diseases + outcomes, "shrinkage": [1.0] * 6}))
    return tmp_path / "routes.parquet", tmp_path / "support.json"


def test_a_collapsed_router_fails_the_health_criteria(tmp_path):
    from opera.run.router_decision import routing_health

    # Every row puts 90% on the overall-survival adapter.
    rows = [(d, "inf", [0.05 if i == d else 0 for i in range(4)] + [0.9, 0.05])
            for d in range(4) for _ in range(5)]
    health = routing_health(*_routes(tmp_path, rows))
    assert not health["healthy"]
    assert health["criteria"]["top_source_share"]["source"] == "outcome:os"


def test_selective_routing_passes_the_health_criteria(tmp_path):
    from opera.run.router_decision import routing_health

    # Each disease leans on a different relative, with real weight on its own.
    partner = {0: 1, 1: 2, 2: 3, 3: 0}
    rows = []
    for d in range(4):
        weights = [0.0] * 4
        weights[d], weights[partner[d]] = 0.4, 0.35
        rows += [(d, "inf", weights + [0.1, 0.15])] * 5
    health = routing_health(*_routes(tmp_path, rows))
    assert health["healthy"], health
    assert health["criteria"]["distinct_top_borrowed_across_diseases"]["count"] == 4
