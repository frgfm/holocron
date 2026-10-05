# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Measure model inference in fresh processes, with optional JSON results."""

# Keep runtime imports inside workers so ONNX memory does not include PyTorch.
# ruff: noqa: PLC0415

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess  # noqa: S404
import sys
import tempfile
import time
from pathlib import Path


def synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(fn, batch_size, num_it, warmup_it, sync=lambda: None):
    # Measure the first call before any warm-up.
    sync()
    start = time.perf_counter()
    output = fn()
    sync()
    first_ms = 1000 * (time.perf_counter() - start)

    for _ in range(warmup_it):
        fn()
    sync()

    timings = []
    for _ in range(num_it):
        start = time.perf_counter()
        fn()
        sync()
        timings.append(1000 * (time.perf_counter() - start))

    # A separate loop measures sustained throughput, without a wait after each call.
    start = time.perf_counter()
    for _ in range(num_it):
        fn()
    sync()
    elapsed = time.perf_counter() - start

    # Nearest-rank percentile also works for short smoke runs.
    p95_index = (95 * num_it + 99) // 100 - 1
    return {
        "first_ms": first_ms,
        "median_ms": statistics.median(timings),
        "p95_ms": sorted(timings)[p95_index],
        "throughput_per_s": batch_size * num_it / elapsed,
    }, output


def peak_rss_mib():
    if sys.platform == "win32":
        return None
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def prepare_model(args):
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    from holocron import models

    model = models.__dict__[args.arch](pretrained=args.pretrained).eval()
    if hasattr(model, "reparametrize"):
        model.reparametrize()
    image = torch.rand((args.batch_size, 3, args.size, args.size))
    return model, image


def export_onnx(args):
    import numpy as np
    import torch

    model, image = prepare_model(args)
    with torch.inference_mode():
        torch.onnx.export(model, image, args.export_to, export_params=True, opset_version=20, dynamo=False)
        np.savez(args.export_to.with_suffix(".npz"), image=image.numpy(), output=model(image).numpy())


def evaluate(args):
    if args.backend == "torch":
        import torch

        model, image = prepare_model(args)
        device = torch.device(args.device)
        if device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("Use a CPU, CUDA, or MPS device")
        model, image = model.to(device), image.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            result, output = measure(
                lambda: model(image), args.batch_size, args.it, args.warmup, lambda: synchronize(device)
            )
        result["peak_rss_mib"] = peak_rss_mib()
        result["runtime"] = {"device_name": str(device)}
        if device.type == "cuda":
            result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
            result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
            result["runtime"].update({
                "device_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            })
        if not torch.isfinite(output).all():
            raise ValueError("Model output contains non-finite values")
    else:
        import numpy as np
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            str(args.onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        with np.load(args.onnx_path.with_suffix(".npz"), allow_pickle=False) as fixture:
            inputs = {session.get_inputs()[0].name: fixture["image"]}
            result, output = measure(lambda: session.run(None, inputs), args.batch_size, args.it, args.warmup)
            result["peak_rss_mib"] = peak_rss_mib()
            if not np.isfinite(output[0]).all():
                raise ValueError("Model output contains non-finite values")
            np.testing.assert_allclose(output[0], fixture["output"], rtol=1e-3, atol=1e-5, equal_nan=False)
        result["runtime"] = {"providers": session.get_providers()}
    result["pid"] = os.getpid()
    return result


def run_child(args, *extra):
    command = [sys.executable, str(Path(__file__).resolve()), args.arch]
    for name in ("backend", "device", "size", "batch_size", "it", "warmup", "threads", "seed"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.pretrained:
        command.append("--pretrained")
    return subprocess.run(command + list(extra), check=True, stdout=subprocess.PIPE, text=True).stdout  # noqa: S603


def environment():
    spec = importlib.util.find_spec("holocron")
    root = Path(spec.origin).resolve().parents[1]
    git = shutil.which("git")
    revision = None
    dirty = None
    if git:
        revision_result = subprocess.run(  # noqa: S603
            [git, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        revision = revision_result.stdout.strip() or None
        if revision:
            status = subprocess.run(  # noqa: S603
                [git, "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=False,
            )
            dirty = bool(status.stdout) if status.returncode == 0 else None
    versions = {}
    for package in ("pylocron", "torch", "torchvision", "numpy", "onnx", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "hostname": platform.node(),
        "revision": revision,
        "dirty": dirty,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": versions,
    }


def main(args):
    if args.export_to:
        export_onnx(args)
        return
    if args.worker:
        print(json.dumps(evaluate(args), allow_nan=False))
        return

    with tempfile.TemporaryDirectory(prefix="holocron-latency-") as directory:
        extra = ["--worker"]
        if args.backend == "onnx":
            path = str(Path(directory) / "model.onnx")
            run_child(args, "--export-to", path)
            extra.extend(["--onnx-path", path])
        runs = [json.loads(run_child(args, *extra)) for _ in range(args.repeat)]

    summary = {}
    # Keep the largest memory peak; use medians for timings and throughput.
    for key in runs[0]:
        if key not in {"pid", "runtime"}:
            values = [run[key] for run in runs if run[key] is not None]
            summary[key] = (max(values) if "peak" in key else statistics.median(values)) if values else None
    summary["median_range_ms"] = [min(run["median_ms"] for run in runs), max(run["median_ms"] for run in runs)]
    config = {
        key: value for key, value in vars(args).items() if key not in {"output", "worker", "export_to", "onnx_path"}
    }
    report = {"schema_version": 1, "config": config, "environment": environment(), "runs": runs, "summary": summary}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    rss = "unavailable" if summary["peak_rss_mib"] is None else f"{summary['peak_rss_mib']:.1f} MiB"
    print(f"{args.arch}: {args.backend}, {args.device}, batch {args.batch_size}, {args.repeat} fresh processes")
    print(
        f"First call: {summary['first_ms']:.2f} ms; median: {summary['median_ms']:.2f} ms; "
        f"p95 (median across runs): {summary['p95_ms']:.2f} ms; "
        f"throughput: {summary['throughput_per_s']:.2f} images/s; peak RSS: {rss}"
    )
    print(f"Run median range: {summary['median_range_ms'][0]:.2f}-{summary['median_range_ms'][1]:.2f} ms")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("arch", help="Classification architecture to use")
    parser.add_argument("--backend", choices=("torch", "onnx"), default="torch", help="Runtime to measure")
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda[:index], or mps; ONNX uses cpu")
    parser.add_argument("--size", type=positive_int, default=224, help="Square image size")
    parser.add_argument("--batch-size", type=positive_int, default=1, help="Images per batch")
    parser.add_argument("--it", type=positive_int, default=100, help="Iterations in each latency and throughput loop")
    parser.add_argument("--warmup", type=nonnegative_int, default=10, help="Warm-up calls after the first call")
    parser.add_argument("--repeat", type=positive_int, default=5, help="Fresh worker processes")
    parser.add_argument(
        "--threads", type=positive_int, default=1, help="Intra-op CPU threads; inter-op threads stay at 1"
    )
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="Seed for model initialization and input")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model weights")
    parser.add_argument("--output", type=Path, help="Save settings, versions, trials, and summary as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--export-to", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--onnx-path", type=Path, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.backend == "onnx" and args.device != "cpu":
        parser.error("--backend onnx supports only --device cpu")
    main(args)
