"""Reproducible low-n/high-p simulations for OPERA tabular defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, roc_auc_score

from opera.run.train_tabular_baselines import train_one_model


PROFILES: dict[str, tuple[str, dict]] = {
    "logistic_ridge": ("logistic", {"C": 0.1, "l1_ratio": 0.0}),
    "logistic_elastic": ("logistic", {"C": 0.1, "l1_ratio": 0.5}),
    "logistic_sparse": ("logistic", {"C": 0.1, "l1_ratio": 1.0}),
    "xgboost_legacy": (
        "xgboost",
        {
            "n_estimators": 200,
            "max_depth": 3,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "colsample_bytree": 0.9,
        },
    ),
    "xgboost_regularized": (
        "xgboost",
        {
            "n_estimators": 200,
            "max_depth": 2,
            "min_child_weight": 2,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "colsample_bytree": 0.5,
        },
    ),
}


def _frame(x: np.ndarray, y: np.ndarray, offset: int) -> pd.DataFrame:
    frame = pd.DataFrame(x, columns=[f"x_{idx}" for idx in range(x.shape[1])])
    frame.insert(0, "subject_id", np.arange(offset, offset + len(frame)))
    frame["label"] = y
    return frame


def simulate_cell(
    n_train: int,
    n_features: int,
    seed: int,
    n_tune: int = 200,
    n_test: int = 1000,
) -> list[dict]:
    """Fit fixed candidate profiles and score tuning and untouched test sets."""
    total = n_train + n_tune + n_test
    x, y = make_classification(
        n_samples=total,
        n_features=n_features,
        n_informative=min(20, max(5, n_features // 20)),
        n_redundant=min(10, max(0, n_features // 40)),
        weights=[0.8, 0.2],
        class_sep=0.8,
        flip_y=0.03,
        random_state=seed,
        shuffle=True,
    )
    train = _frame(x[:n_train], y[:n_train], 0)
    tune = _frame(x[n_train : n_train + n_tune], y[n_train : n_train + n_tune], n_train)
    test = _frame(x[-n_test:], y[-n_test:], n_train + n_tune)
    columns = [col for col in train if col.startswith("x_")]
    rows = []
    for profile, (model_name, params) in PROFILES.items():
        try:
            tune_predictions, pipeline = train_one_model(
                model_name,
                train,
                tune,
                columns,
                categorical_columns=None,
                seed=seed,
                tune_params=params,
            )
        except (ModuleNotFoundError, SystemExit):
            continue
        test_probability = pipeline.predict_proba(test[columns])[:, 1]
        for split, labels, probability in (
            ("tuning", tune["label"], tune_predictions["probability"]),
            ("held_out", test["label"], test_probability),
        ):
            rows.append(
                {
                    "seed": seed,
                    "n_train": n_train,
                    "n_features": n_features,
                    "features_per_row": n_features / n_train,
                    "profile": profile,
                    "split": split,
                    "auroc": roc_auc_score(labels, probability),
                    "log_loss": log_loss(labels, probability),
                }
            )
    return rows


def run_simulation(repeats: int = 5) -> pd.DataFrame:
    """Run representative clinical low-n/high-p scenarios."""
    rows = []
    for n_train, n_features in ((100, 500), (250, 1000), (500, 2000)):
        for seed in range(repeats):
            rows.extend(simulate_cell(n_train, n_features, seed))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = run_simulation(args.repeats)
    results.to_csv(output / "tabular_low_n_simulation.csv", index=False)
    summary = (
        results.groupby(["n_train", "n_features", "profile", "split"])[
            ["auroc", "log_loss"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, map(str, col))) for col in summary]
    summary.to_csv(output / "tabular_low_n_simulation_summary.csv", index=False)
    with open(output / "simulation_contract.json", "w") as handle:
        json.dump(
            {
                "repeats": args.repeats,
                "profiles": PROFILES,
                "selection_split": "tuning",
                "final_evaluation_split": "held_out",
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
