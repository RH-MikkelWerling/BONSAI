"""
Task 8 — Tabular baseline tuning and matched-fraction parity.

Tests:
1. ``tune_estimator_params`` selects from ``TUNING_GRIDS`` on a tiny synthetic
   validation set and returns a valid param dict.
2. ``build_tabular_fraction_cmd`` produces a command that:
   - contains ``opera.run.train_tabular_baselines`` as the module to run,
   - carries the supplied seed verbatim (so it matches the encoder cell), and
   - includes ``--tune`` and ``--tune_split`` when ``tune=True``.
3. The label_efficiency main loop calls ``subprocess.run`` for the tabular
   baseline at matched (fraction, seed) cells when ``--tabular_features`` is
   given — verified by monkeypatching subprocess.run.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────
# 1. tune_estimator_params
# ──────────────────────────────────────────────────────────────────


def _make_binary_dfs(n_train=200, n_val=80, seed=7):
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, 4))
    y_tr = (X_tr[:, 0] + rng.standard_normal(n_train) * 0.5 > 0).astype(int)
    X_va = rng.standard_normal((n_val, 4))
    y_va = (X_va[:, 0] + rng.standard_normal(n_val) * 0.5 > 0).astype(int)
    cols = [f"f{i}" for i in range(4)]
    train = pd.DataFrame(X_tr, columns=cols).assign(
        label=y_tr, subject_id=range(n_train)
    )
    val = pd.DataFrame(X_va, columns=cols).assign(
        label=y_va, subject_id=range(n_train, n_train + n_val)
    )
    return train, val, cols


def test_tune_estimator_params_logistic_returns_c_value():
    from opera.run.train_tabular_baselines import tune_estimator_params, TUNING_GRIDS

    train, val, cols = _make_binary_dfs()
    best = tune_estimator_params("logistic", train, val, cols, seed=0)

    assert isinstance(best, dict)
    assert "C" in best
    # Best C must come from the tuning grid.
    valid_c_values = {p["C"] for p in TUNING_GRIDS["logistic"]}
    assert best["C"] in valid_c_values


def test_tune_estimator_params_xgboost_returns_depth_lr():
    pytest.importorskip("xgboost")
    from opera.run.train_tabular_baselines import tune_estimator_params, TUNING_GRIDS

    train, val, cols = _make_binary_dfs()
    best = tune_estimator_params("xgboost", train, val, cols, seed=0)

    assert isinstance(best, dict)
    assert "max_depth" in best
    valid = {(p["max_depth"], p["learning_rate"]) for p in TUNING_GRIDS["xgboost"]}
    assert (best["max_depth"], best["learning_rate"]) in valid


def test_tune_estimator_params_unsupported_model_returns_empty():
    """tabpfn and survival models are not in TUNING_GRIDS → empty dict."""
    from opera.run.train_tabular_baselines import tune_estimator_params

    train, val, cols = _make_binary_dfs()
    assert tune_estimator_params("tabpfn", train, val, cols) == {}
    assert tune_estimator_params("cox", train, val, cols) == {}


def test_tune_estimator_params_falls_back_on_single_class(tmp_path):
    """Single-class val split → return empty dict without crashing."""
    from opera.run.train_tabular_baselines import tune_estimator_params

    train, _, cols = _make_binary_dfs()
    # val with only one class
    val_single = pd.DataFrame(
        np.zeros((20, 4)), columns=cols
    ).assign(label=0, subject_id=range(200, 220))

    result = tune_estimator_params("logistic", train, val_single, cols)
    assert result == {}


# ──────────────────────────────────────────────────────────────────
# 2. build_tabular_fraction_cmd
# ──────────────────────────────────────────────────────────────────


def test_build_tabular_fraction_cmd_contains_module_and_seed(tmp_path):
    from opera.run.label_efficiency import build_tabular_fraction_cmd

    cmd = build_tabular_fraction_cmd(
        features_path="/data/features.csv",
        outcome_parquet="/data/outcome_subsampled.parquet",
        output_dir=tmp_path / "cell",
        cohort="dlbcl",
        outcome_name="mortality",
        seed=43,
        models="xgboost",
        tune=False,
    )

    cmd_str = " ".join(cmd)
    assert "opera.run.train_tabular_baselines" in cmd_str
    assert "--seed" in cmd
    assert "43" in cmd
    assert "--tune" not in cmd


def test_build_tabular_fraction_cmd_with_tune_flag(tmp_path):
    from opera.run.label_efficiency import build_tabular_fraction_cmd

    cmd = build_tabular_fraction_cmd(
        features_path="/data/features.csv",
        outcome_parquet="/data/outcome.parquet",
        output_dir=tmp_path / "cell",
        cohort="dlbcl",
        outcome_name="mortality",
        seed=42,
        tune=True,
        tune_split="tuning",
    )

    assert "--tune" in cmd
    assert "--tune_split" in cmd
    tune_split_idx = cmd.index("--tune_split")
    assert cmd[tune_split_idx + 1] == "tuning"


def test_build_tabular_fraction_cmd_seed_matches_encoder_seed(tmp_path):
    """The tabular cmd seed must exactly match the encoder finetuning seed."""
    from opera.run.label_efficiency import build_tabular_fraction_cmd

    for encoder_seed in (42, 43, 100):
        cmd = build_tabular_fraction_cmd(
            features_path="/data/features.csv",
            outcome_parquet="/data/outcome.parquet",
            output_dir=tmp_path / "cell",
            cohort="c",
            outcome_name="o",
            seed=encoder_seed,
        )
        seed_idx = cmd.index("--seed")
        assert cmd[seed_idx + 1] == str(encoder_seed), (
            f"Expected seed {encoder_seed} in cmd, got {cmd[seed_idx + 1]}"
        )


# ──────────────────────────────────────────────────────────────────
# 3. label_efficiency main: tabular subprocess called at matched fractions
# ──────────────────────────────────────────────────────────────────


def _make_minimal_sweep_config(tmp_path: Path) -> Path:
    """Minimal sweep config that label_efficiency.main accepts without real data."""
    cfg = {
        "cohorts": {
            "dlbcl": {
                "data_dir": str(tmp_path / "data"),
                "registry_start_date": None,
            }
        },
        "outcomes": {
            "mortality": {
                "outcome_file": "mortality.parquet",
                "n_hours_start_include": 1,
                "n_hours_end_include": 8760,
            }
        },
        "model_variants": {
            "opera": {
                "encoder_ckpt": str(tmp_path / "encoder.ckpt"),
                "encoder_source": "opera",
            }
        },
    }
    config_path = tmp_path / "sweep_config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return config_path


def test_label_efficiency_calls_tabular_at_each_fraction_seed(
    tmp_path, monkeypatch
):
    """When --tabular_features is set, subprocess.run is called for each cell."""
    import yaml
    from opera.run.label_efficiency import (
        build_tabular_fraction_cmd,
        subsample_outcome_parquet,
    )

    # Build a tiny outcome parquet the subsampler accepts.
    outcome_df = pd.DataFrame(
        {
            "subject_id": range(20),
            "split": ["train"] * 14 + ["tuning"] * 3 + ["held_out"] * 3,
            "label": [1, 0] * 7 + [1, 0, 0] + [1, 0, 0],
        }
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outcome_path = data_dir / "mortality.parquet"
    outcome_df.to_parquet(outcome_path)

    features_path = tmp_path / "features.csv"
    features_path.write_text("subject_id,f1\n" + "\n".join(f"{i},{i*0.1}" for i in range(20)))

    config_path = _make_minimal_sweep_config(tmp_path)

    # Track subprocess calls without executing them.
    captured_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        captured_calls.append(list(cmd))
        # Simulate successful finetune/evaluate by writing a dummy metrics file.
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--config-name":
                continue
            if "hydra.run.dir=" in arg:
                output_dir = Path(arg.split("=", 1)[1])
                break
            if arg == "hydra.run.dir" and i + 1 < len(cmd):
                output_dir = Path(cmd[i + 1])
                break
        if output_dir and "train_tabular_baselines" not in " ".join(cmd):
            # Encoder call: write dummy metrics so the cache check passes.
            (output_dir / "eval").mkdir(parents=True, exist_ok=True)
            with open(output_dir / "eval" / "metrics.json", "w") as f:
                json.dump({"discrimination": {"auroc": 0.70}}, f)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_efficiency",
            "--sweep_config", str(config_path),
            "--tasks", "dlbcl:mortality",
            "--fractions", "0.5",
            "--seeds", "42,43",
            "--output_dir", str(tmp_path / "out"),
            "--tabular_features", str(features_path),
            "--tabular_models", "xgboost",
        ],
    )

    # Import and monkeypatch subsample_outcome_parquet to avoid parquet overhead.
    def fake_subsample(outcome_path, fraction, seed, output_path, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        outcome_df.to_parquet(output_path)

    monkeypatch.setattr(
        "opera.run.label_efficiency.subsample_outcome_parquet", fake_subsample
    )

    from opera.run.label_efficiency import main

    main()

    # Check that tabular baseline subprocess was called once per (fraction, seed).
    tabular_calls = [
        c for c in captured_calls if "train_tabular_baselines" in " ".join(c)
    ]
    # 1 fraction × 2 seeds = 2 tabular calls
    assert len(tabular_calls) == 2, (
        f"Expected 2 tabular calls (1 frac × 2 seeds), got {len(tabular_calls)}"
    )
    # Each call must carry the correct seed.
    for call in tabular_calls:
        assert "--seed" in call
        seed_idx = call.index("--seed")
        assert call[seed_idx + 1] in ("42", "43")
