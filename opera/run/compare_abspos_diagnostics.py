"""Plot an apples-to-apples comparison of vocabulary diagnostic runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Diagnostic inputs must be LABEL=DIRECTORY.")
    label, directory = value.split("=", 1)
    return label, Path(directory)


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, n_bootstrap: int = 2000
) -> tuple[float, float]:
    """Return a percentile bootstrap CI for a sample mean."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    )
    return tuple(np.quantile(means, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=10000)
    args = parser.parse_args()
    runs = [_spec(value) for value in args.diagnostic]
    if not runs:
        raise ValueError("Provide at least one --diagnostic LABEL=DIRECTORY input.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_frames, probe_frames, event_frames = [], [], []
    for label, directory in runs:
        details = pd.read_csv(directory / "contextual_sensitivity_subjects.csv")
        details["checkpoint"] = label
        detail_frames.append(details)
        probes = pd.read_csv(directory / "contextual_linear_probes.csv")
        probes["checkpoint"] = label
        probe_frames.append(probes)
        events = pd.read_parquet(directory / "contextual_event_sample.parquet")
        event_frames.append(events)

    details = pd.concat(detail_frames, ignore_index=True)
    rows = []
    for keys, frame in details.groupby(["checkpoint", "pooling", "perturbation"]):
        values = frame["cosine_distance"].to_numpy(float)
        lower, upper = bootstrap_mean_ci(values, seed=args.seed)
        rows.append({
            "checkpoint": keys[0], "pooling": keys[1], "perturbation": keys[2],
            "mean_cosine_distance": values.mean(), "ci_lower": lower,
            "ci_upper": upper, "n_subjects": frame["subject_id"].nunique(),
        })
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(args.output_dir / "abspos_contextual_sensitivity.csv", index=False)

    calendar = pd.concat(probe_frames, ignore_index=True)
    calendar = calendar.loc[calendar["target"].eq("calendar_abspos")].copy()
    calendar.to_csv(args.output_dir / "calendar_decodability.csv", index=False)

    focus = sensitivity.loc[sensitivity["perturbation"].eq("calendar_plus_5y")]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    poolings = list(dict.fromkeys(focus["pooling"]))
    labels = [label for label, _ in runs]
    width = 0.8 / len(labels)
    x = np.arange(len(poolings))
    indexed = focus.set_index(["checkpoint", "pooling"])
    for index, label in enumerate(labels):
        frame = indexed.loc[label].reindex(poolings)
        y = frame["mean_cosine_distance"].to_numpy()
        lo = frame["ci_lower"].to_numpy()
        hi = frame["ci_upper"].to_numpy()
        axis.bar(
            x + (index - (len(labels) - 1) / 2) * width, y, width, label=label,
            yerr=np.vstack([y - lo, hi - y]), capsize=3,
        )
    axis.set_xticks(x, poolings)
    axis.set_ylabel("Cosine distance after +5 calendar years")
    axis.set_title("Counterfactual calendar sensitivity (95% bootstrap CI)")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "calendar_shift_sensitivity.png", dpi=200)
    fig.savefig(args.output_dir / "calendar_shift_sensitivity.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(
        len(runs), 2, figsize=(10, 4.2 * len(runs)), squeeze=False
    )
    rng = np.random.default_rng(args.seed)
    for row, ((label, _), events) in enumerate(zip(runs, event_frames)):
        if len(events) > args.max_events:
            events = events.iloc[rng.choice(len(events), args.max_events, replace=False)]
        year = pd.to_datetime(events["abspos"], unit="h", origin="unix").dt.year
        for column, representation in enumerate(("raw", "contextual")):
            axis = axes[row, column]
            embedding_columns = [
                name for name in events if name.startswith(f"{representation}_")
            ]
            coords = PCA(n_components=2, random_state=args.seed).fit_transform(
                events[embedding_columns]
            )
            scatter = axis.scatter(
                coords[:, 0], coords[:, 1], c=year, s=5, alpha=0.45,
                cmap="viridis", rasterized=True,
            )
            axis.set_title(f"{label}: {representation}")
            axis.set_xlabel("PCA 1")
            axis.set_ylabel("PCA 2")
            fig.colorbar(scatter, ax=axis, label="Calendar year")
    fig.suptitle("Event embeddings colored by calendar era")
    fig.tight_layout()
    fig.savefig(args.output_dir / "contextual_embeddings_by_era.png", dpi=200)
    fig.savefig(args.output_dir / "contextual_embeddings_by_era.pdf")
    plt.close(fig)
    print(f"Wrote checkpoint comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
