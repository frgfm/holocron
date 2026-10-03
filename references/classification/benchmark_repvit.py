# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Compare unfused and fused RepViT CPU forwards using the same trained weights."""

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from holocron.models.classification import repvit_m0_9


@torch.inference_mode()
def benchmark(args):
    if min(args.threads, args.iterations, *args.sizes) < 1 or args.warmup < 0:
        raise ValueError("threads, iterations and input sizes must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = repvit_m0_9(num_classes=args.num_classes).eval()
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    fused = copy.deepcopy(model)
    fused.reparametrize()
    measurements = []
    for size in args.sizes:
        image = torch.randn(1, 3, size, size)
        expected, actual = model(image), fused(image)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        row = {
            "input_shape": list(image.shape),
            "max_absolute_logit_difference": (expected - actual).abs().max().item(),
        }
        candidates = [("unfused", model), ("fused", fused)]
        for _ in range(args.warmup):
            for _, candidate in candidates:
                candidate(image)
        timings = {name: [] for name, _ in candidates}
        for iteration in range(args.iterations):
            # Alternate the first model to reduce drift from a shared CPU host.
            for name, candidate in candidates[:: -1 if iteration % 2 else 1]:
                started = time.perf_counter()
                candidate(image)
                timings[name].append((time.perf_counter() - started) * 1000)
        for name, elapsed in timings.items():
            elapsed.sort()
            row[name] = {"median_ms": statistics.median(elapsed), "p95_ms": elapsed[math.ceil(0.95 * len(elapsed)) - 1]}
        measurements.append(row)
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = platform.processor()
    if cpu_info.is_file():
        cpu_model = next(
            (
                line.partition(":")[2].strip()
                for line in cpu_info.read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    results = {
        "scope": "CPU forward-only microbenchmark; excludes preprocessing and data transfer; no GPU claim",
        "model": "repvit_m0_9",
        "num_classes": args.num_classes,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": "cpu",
        "cpu_model": cpu_model,
        "dtype": "float32",
        "intraop_threads": args.threads,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "measurement_order": "paired forwards, alternating which model runs first",
        "parameters_before": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_after": sum(parameter.numel() for parameter in fused.parameters()),
        "batchnorm_layers_after": sum(
            isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) for module in fused.modules()
        ),
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/repvit-digits.pth"))
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("references/classification/results/repvit-deployment.json"))
    benchmark(parser.parse_args())
