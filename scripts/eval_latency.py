# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Holocron model latency benchmark
"""

import argparse
import importlib
import json
import platform
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch.utils import benchmark

from holocron import models


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_evaluation(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    min_run_time: float = 1.0,
    warmup_it: int = 10,
    num_threads: int = 1,
) -> dict[str, float | int]:
    """Benchmark a callable with accelerator-aware synchronization.

    Returns:
        Machine-readable timing statistics.
    """
    for _ in range(warmup_it):
        fn()
    _synchronize(device)

    measurement = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=min_run_time)
    return {
        "mean_ms": 1000 * measurement.mean,
        "median_ms": 1000 * measurement.median,
        "iqr_ms": 1000 * measurement.iqr,
        "measurements": len(measurement.raw_times),
        "runs_per_measurement": measurement.number_per_run,
        "threads": num_threads,
    }


def _format_measurement(name: str, result: dict[str, float | int]) -> str:
    return f"{name} - median {result['median_ms']:.2f}ms, mean {result['mean_ms']:.2f}ms, IQR {result['iqr_ms']:.2f}ms"


@torch.inference_mode()
def main(args: argparse.Namespace) -> dict[str, Any]:
    try:
        model = models.get_model(args.arch, pretrained=args.pretrained).eval()
    except ValueError as exc:
        raise TypeError(f"unknown architecture: {args.arch}") from exc
    if hasattr(model, "reparametrize"):
        model.reparametrize()

    input_shape = (args.batch_size, 3, args.size, args.size)
    cpu = torch.device("cpu")
    cpu_input = torch.rand(input_shape)
    results: dict[str, dict[str, float | int]] = {
        "pytorch_cpu": run_evaluation(
            lambda: model(cpu_input),
            device=cpu,
            min_run_time=args.min_run_time,
            warmup_it=args.warmup,
            num_threads=args.num_threads,
        )
    }

    onnxruntime = importlib.import_module("onnxruntime")

    with TemporaryDirectory(prefix="holocron-onnx-") as tmp_dir:
        onnx_path = Path(tmp_dir) / "model.onnx"
        torch.onnx.export(
            model,
            (cpu_input,),
            onnx_path,
            export_params=True,
            opset_version=20,
            dynamo=True,
            input_names=("input",),
            output_names=("output",),
            dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
            verbose=False,
        )
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = args.num_threads
        session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=("CPUExecutionProvider",),
        )
        ort_input = {session.get_inputs()[0].name: cpu_input.numpy()}
        results["onnxruntime_cpu"] = run_evaluation(
            lambda: session.run(None, ort_input),
            device=cpu,
            min_run_time=args.min_run_time,
            warmup_it=args.warmup,
            num_threads=args.num_threads,
        )

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type != "cpu":
        device_model = model.to(device=device)
        device_input = cpu_input.to(device=device)
        results[f"pytorch_{device.type}"] = run_evaluation(
            lambda: device_model(device_input),
            device=device,
            min_run_time=args.min_run_time,
            warmup_it=args.warmup,
            num_threads=args.num_threads,
        )

    report: dict[str, Any] = {
        "architecture": args.arch,
        "input_shape": list(input_shape),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "onnxruntime": onnxruntime.__version__,
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{args.arch} ({input_shape[0]}x3x{args.size}x{args.size})")
        labels = {
            "pytorch_cpu": "PyTorch CPU",
            "onnxruntime_cpu": "ONNX Runtime CPU",
            "pytorch_cuda": "PyTorch CUDA",
            "pytorch_mps": "PyTorch MPS",
        }
        for name, result in results.items():
            print(_format_measurement(labels.get(name, name), result))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holocron model latency benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("arch", type=str, help="Architecture to use")
    parser.add_argument("--size", type=int, default=224, help="The square image input size")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument("--device", type=str, help="Optional accelerator device, such as cuda:0 or mps")
    parser.add_argument("--min-run-time", type=float, default=1.0, help="Minimum seconds per benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-threads", type=int, default=1, help="PyTorch CPU threads")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model-zoo weights")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    main(parser.parse_args())
