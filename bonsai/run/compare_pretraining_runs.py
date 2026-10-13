"""Compare two BONSAI pretraining CSV logs without reading patient data.

Usage: python -m bonsai.run.compare_pretraining_runs --fourier RUN --time2vec RUN
RUN may be the Hydra run directory or its version_0 CSVLogger directory.
"""

import argparse
import csv
import math
from pathlib import Path


METRICS = ("val/loss", "val/code_loss", "val/Precision@1", "val/Precision@10")


def metrics_path(run: str) -> Path:
    root = Path(run).expanduser()
    candidates = (root / "metrics.csv", root / "version_0" / "metrics.csv")
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one metrics.csv under {root}; found {found}")
    return found[0]


def epoch_metrics(path: Path) -> dict[int, dict[str, float]]:
    epochs: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"Missing epoch column in {path}")
        for row in reader:
            if not row["epoch"]:
                continue
            epoch = int(float(row["epoch"])) + 1  # Lightning logs zero-based epoch.
            values = epochs.setdefault(epoch, {})
            for metric in METRICS:
                raw = row.get(metric, "")
                if raw:
                    number = float(raw)
                    if math.isfinite(number):
                        values[metric] = number
    result = {epoch: values for epoch, values in epochs.items() if "val/loss" in values}
    if not result:
        raise ValueError(f"No completed validation epochs in {path}")
    return result


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.5f}"


def describe(name: str, path: Path, epochs: dict[int, dict[str, float]]) -> None:
    best = min(epochs, key=lambda epoch: epochs[epoch]["val/loss"])
    last = max(epochs)
    print(f"{name}: {path}")
    print(f"  validated epochs: {len(epochs)}; last={last}; best={best}")
    print(
        "  best val/loss=" + fmt(epochs[best]["val/loss"])
        + "; last val/loss=" + fmt(epochs[last]["val/loss"])
        + "; best P@1=" + fmt(epochs[best].get("val/Precision@1"))
        + "; last P@1=" + fmt(epochs[last].get("val/Precision@1"))
    )
    if last < 20:
        print("  NOTE: 20-epoch run is not complete or its final validation is absent.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourier", required=True)
    parser.add_argument("--time2vec", required=True)
    args = parser.parse_args()
    fourier_path, time2vec_path = metrics_path(args.fourier), metrics_path(args.time2vec)
    fourier, time2vec = epoch_metrics(fourier_path), epoch_metrics(time2vec_path)
    describe("Fourier", fourier_path, fourier)
    describe("scaled-Time2Vec", time2vec_path, time2vec)
    print("\nMatched completed validation epochs (lower loss, higher P@1 are better):")
    print("epoch  Fourier loss  Time2Vec loss  T2V - Fourier  Fourier P@1  Time2Vec P@1")
    for epoch in sorted(fourier.keys() & time2vec.keys()):
        f, t = fourier[epoch], time2vec[epoch]
        print(
            f"{epoch:>5}  {fmt(f['val/loss']):>12}  {fmt(t['val/loss']):>13}"
            f"  {t['val/loss'] - f['val/loss']:>13.5f}"
            f"  {fmt(f.get('val/Precision@1')):>11}"
            f"  {fmt(t.get('val/Precision@1')):>12}"
        )
    if not (fourier.keys() & time2vec.keys()):
        print("No epochs overlap; do not compare best values as a matched run.")
    print("\nThis compares next-code validation, not downstream clinical performance.")


if __name__ == "__main__":
    main()
