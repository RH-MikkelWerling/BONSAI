"""Small synthetic CPU/GPU cost check for the source-adapter readout.

Run with ``python -m scripts.benchmark_source_adapters --device cuda``.
This does not load patient data or produce scientific results.
"""

import argparse
import time

import torch

from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    device = torch.device(args.device)
    readout = SourceAnchoredReadout(
        args.hidden_size,
        [f"disease_{index}" for index in range(24)],
        [f"outcome_{index}" for index in range(87)],
    ).to(device)
    readout.set_training_support([100] * 111, [100] * 111, [100] * 111)
    features = torch.randn(args.batch_size, args.hidden_size, device=device)
    fine = torch.randint(0, 24, (args.batch_size,), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        readout.zero_grad(set_to_none=True)
        readout(features, fine).square().mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print({
        "adapter_parameters": sum(p.numel() for p in readout.parameters()),
        "seconds_per_step": elapsed / args.steps,
        "peak_cuda_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else None
        ),
    })


if __name__ == "__main__":
    main()
