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
