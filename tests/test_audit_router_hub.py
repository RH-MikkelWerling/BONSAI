import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


def test_the_hub_and_its_lookalikes_are_found(tmp_path):
    # Three sources; source 1 carries the routed weight, source 2 is a copy of it.
    torch.manual_seed(0)
    down, up = torch.randn(3, 2, 4), torch.randn(3, 4, 2)
    down[2], up[2] = down[1], up[1]
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"state_dict": {"model.source_adapter_readout.down": down,
                               "model.source_adapter_readout.up": up,
                               "model.source_adapter_readout.source_shrinkage": torch.ones(3)}},
               run / "best.ckpt")
    pred = tmp_path / "tuning"
    pred.mkdir()
    np.savez(pred / "patient_embeddings.npz", embeddings=np.random.default_rng(0).normal(size=(50, 4)))
    pd.DataFrame([{"fine_index": 0, "outcome": "inf",
                   "disease_weights": json.dumps({"disease:A": 0.05}),
                   "outcome_weights": json.dumps({"outcome:os": 0.9, "outcome:alb": 0.05})}] * 10
                 ).to_parquet(pred / "source_route_weights.parquet")
    out = tmp_path / "hub.json"
    subprocess.run([sys.executable, "-m", "opera.run.audit_router_hub", "--run-dir", str(run),
                    "--prediction-dir", str(pred), "--output", str(out)], check=True,
                   capture_output=True)
    report = json.loads(out.read_text())
    assert report["hub"]["source"] == "outcome:os"
    assert report["most_like_hub"][0] == {"source": "outcome:alb", "mean_cosine": 1.0}
    # Disease A borrows from the two outcome sources only (its own disease is excluded).
    top = dict(report["borrowing"]["top_borrowed_by_target"]["A"])
    assert top["outcome:os"] > 0.9 and top.get("disease:A", 0.0) == 0.0
