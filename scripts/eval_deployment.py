# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Replay a frozen batch-one classification sweep in separate accuracy and timing processes."""

# Keep optional runtime imports inside workers; deployment does not require Holocron or PyTorch.
# ruff: noqa: PLC0415

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import statistics
import subprocess  # noqa: S404
import sys
import tempfile
import time
import traceback
from importlib.metadata import distributions, version
from pathlib import Path

if __package__:
    from .eval_latency import environment, peak_rss_mib, save_report
else:
    from eval_latency import environment, peak_rss_mib, save_report

PACKAGES = {
    "onnx": ("numpy", "Pillow", "onnxruntime"),
    "ncnn": ("numpy", "Pillow", "ncnn"),
    "coreml": ("numpy", "Pillow", "coremltools"),
}
NCNN_FLAGS = ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic", "use_bf16_storage", "use_bf16_packed")


class UnsupportedError(RuntimeError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes bundle: {name}")
    return root / path


def validate_manifest(bundle):  # noqa: PLR0912 -- keep bundle validation in one place
    if bundle["schema_version"] != 1:
        raise ValueError("Unsupported bundle version")
    prep = bundle["preprocessing"]
    shape = prep["input_shape"]
    if len(shape) != 4 or shape[:2] != [1, 3] or shape[2] != shape[3] or shape[2] < 1 or prep["resize"] < shape[2]:
        raise ValueError("Expected a fixed batch-one RGB square crop and resize >= crop")
    if (
        len(prep["mean"]) != 3
        or len(prep["std"]) != 3
        or any(not math.isfinite(value) for value in prep["mean"] + prep["std"])
        or any(value <= 0 for value in prep["std"])
    ):
        raise ValueError("Expected three normalization means and positive standard deviations")
    indices = bundle["accuracy_indices"]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(bundle["samples"]):
        raise ValueError("Invalid accuracy subset indices")
    if not bundle["categories"] or len(set(bundle["categories"])) != len(bundle["categories"]):
        raise ValueError("Expected unique ordered category names")
    if any(not 0 <= sample["label"] < len(bundle["categories"]) for sample in bundle["samples"]):
        raise ValueError("Invalid class label")
    for name, case in bundle["cases"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or case["backend"] not in PACKAGES:
            raise ValueError(f"Invalid case: {name}")
        allowed = {
            "backend",
            "artifact",
            "param",
            "requested_precision",
            "threads",
            "compute_units",
            "require_arm_fp16",
            "minimum_target",
            "reduce_range",
            *NCNN_FLAGS,
        }
        if set(case) - allowed or not case.get("requested_precision"):
            raise ValueError(f"Unknown case settings or missing requested_precision: {name}")
        if case["backend"] != "coreml" and case.get("threads", 1) != 1:
            raise ValueError("This protocol uses one CPU inference thread")
        if case["backend"] == "ncnn" and "param" not in case:
            raise ValueError("NCNN requires a param file")
        if case["backend"] == "coreml" and case.get("compute_units") not in {
            "CPU_ONLY",
            "CPU_AND_GPU",
            "CPU_AND_NE",
            "ALL",
        }:
            raise ValueError("Invalid Core ML compute-unit allowance")
        if any(not isinstance(case[flag], bool) for flag in NCNN_FLAGS if flag in case):
            raise ValueError("NCNN options must be booleans")
    for value in bundle["protocol"].values():
        if not isinstance(value, int) or value < 1:
            raise ValueError("Protocol counts must be positive integers")


def load_bundle(root):
    bundle = json.loads((root / "bundle.json").read_text())
    validate_manifest(bundle)
    for name, expected in bundle["files"].items():
        path = safe_path(root, name)
        if path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    required = [bundle["reference"], *(sample["path"] for sample in bundle["samples"])]
    for case in bundle["cases"].values():
        for field in ("artifact", "param"):
            if field in case:
                path = safe_path(root, case[field])
                if path.is_dir():
                    required.extend(p.relative_to(root).as_posix() for p in path.rglob("*") if p.is_file())
                else:
                    required.append(case[field])
    required.extend(f"scripts/{name}" for name in ("eval_deployment.py", "eval_latency.py"))
    if any(name not in bundle["files"] for name in required):
        raise ValueError("An input is missing from the bundle hash catalog")
    for name in ("eval_deployment.py", "eval_latency.py"):
        if digest(Path(__file__).with_name(name)) != bundle["files"][f"scripts/{name}"]:
            raise ValueError("Runner source differs from frozen bundle; run bundle/scripts/eval_deployment.py")
    return bundle


def preprocess(data, prep):
    import numpy as np
    from PIL import Image

    resize, size = prep["resize"], prep["input_shape"][2]
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
        width, height = image.size
        dimensions = (
            (resize, int(resize * height / width)) if width <= height else (int(resize * width / height), resize)
        )
        image = image.resize(dimensions, Image.Resampling.BILINEAR)
        width, height = image.size
        left, top = round((width - size) / 2), round((height - size) / 2)
        pixels = np.asarray(image.crop((left, top, left + size, top + size)), np.float32).transpose(2, 0, 1)
    mean = np.asarray(prep["mean"], np.float32)[:, None, None]
    std = np.asarray(prep["std"], np.float32)[:, None, None]
    return np.ascontiguousarray((pixels * np.float32(1 / 255) - mean) / std)[None]


def temperature():
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    return float(path.read_text(encoding="utf-8")) / 1000 if path.exists() else None


def cool(record, limit=68, target=60, timeout=600):
    current = temperature()
    if current is not None and current >= limit:
        start = time.monotonic()
        while temperature() > target:
            if time.monotonic() - start >= timeout:
                raise TimeoutError("Device did not cool within the 600-second thermal budget")
            time.sleep(3)
        record["cooldown_seconds"] = record.get("cooldown_seconds", 0) + time.monotonic() - start


def runtime(root, case, mode, output, record):
    import numpy as np

    backend = case["backend"]
    if backend == "coreml" and sys.platform != "darwin":
        raise UnsupportedError("Core ML prediction requires macOS")
    for package, expected in record["requirements"].items():
        if version(package) != expected:
            raise RuntimeError(f"Runtime version mismatch: {package} requires {expected}, found {version(package)}")
    start = time.monotonic()
    if backend == "onnx":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
        if mode == "probe":
            options.enable_profiling = True
            options.profile_file_prefix = str(output / "profile")
        try:
            model = ort.InferenceSession(str(root / case["artifact"]), options, providers=["CPUExecutionProvider"])
        except ort.capi.onnxruntime_pybind11_state.NotImplemented as exc:
            raise UnsupportedError(str(exc)) from exc
        record["execution"] = {
            "providers": model.get_providers(),
            "observed_precision": "See probe operator input types",
        }

        def predict(image):
            return model.run(None, {model.get_inputs()[0].name: image})[0]

    elif backend == "ncnn":
        import ncnn

        model = ncnn.Net()
        model.opt.num_threads, model.opt.use_vulkan_compute = 1, False
        for flag in NCNN_FLAGS:
            setattr(model.opt, flag, case.get(flag, False))
        if case.get("require_arm_fp16") and not ncnn.cpu_support_arm_asimdhp():
            raise UnsupportedError("Requested ARM FP16 storage/arithmetic unavailable in this CPU/runtime")
        if model.load_param(str(root / case["param"])) or model.load_model(str(root / case["artifact"])):
            raise RuntimeError("NCNN model loading failed")
        record["execution"] = {
            "vulkan": False,
            "options": {flag: getattr(model.opt, flag) for flag in NCNN_FLAGS},
            "observed_precision": "See probe weighted-layer output storage; arithmetic unverified",
        }

        def predict(image):
            with model.create_extractor() as extractor:
                if extractor.input(model.input_names()[0], ncnn.Mat(image[0])):
                    raise RuntimeError("NCNN input failed")
                code, result = extractor.extract(model.output_names()[0])
                if code:
                    raise RuntimeError(f"NCNN extraction failed: {code}")
                return np.array(result)

    else:
        import coremltools as ct

        model = ct.models.MLModel(str(root / case["artifact"]), compute_units=ct.ComputeUnit[case["compute_units"]])
        record["execution"] = {
            "compute_units_allowed": case["compute_units"],
            "observed_precision": "Per-operation precision and device placement unverified",
        }

        def predict(image):
            return model.predict({"images": image})["logits"]

    record["load_seconds"] = time.monotonic() - start

    def checked(image):
        result = np.asarray(predict(image), dtype=np.float32).reshape(-1).copy()
        if result.shape != (record["classes"],) or not np.isfinite(result).all():
            raise ValueError("Wrong output shape or non-finite logits")
        return result

    return model, checked


def probe_precision(model, image, root, case, record):
    if case["backend"] == "onnx":
        path = Path(model.end_profiling())
        operators = {}
        for event in json.loads(path.read_text(encoding="utf-8")):
            args = event.get("args", {})
            if "op_name" in args:
                types = sorted({key for value in args.get("input_type_shape", []) for key in value})
                operators.setdefault(args["op_name"], []).append(types)
        record["execution"]["operator_input_types"] = {
            key: sorted({tuple(types) for types in values}) for key, values in operators.items()
        }
    elif case["backend"] == "ncnn":
        import ncnn

        storage = {}
        with model.create_extractor() as extractor:
            extractor.input(model.input_names()[0], ncnn.Mat(image[0]))
            for line in (root / case["param"]).read_text().splitlines()[2:]:
                parts = line.split()
                if parts and parts[0] in {"Convolution", "ConvolutionDepthWise", "InnerProduct"}:
                    code, blob = extractor.extract(parts[4 + int(parts[2])], 1)
                    if code:
                        raise RuntimeError("NCNN precision probe failed")
                    storage[parts[1]] = blob.elembits()
        record["execution"]["weighted_output_storage_bits"] = storage


def score(logits, reference, truth):
    import numpy as np

    logits, reference, truth = np.asarray(logits), np.asarray(reference), np.asarray(truth)
    actual, expected = logits.argmax(1), reference.argmax(1)
    correct, baseline = actual == truth, expected == truth
    return {
        "samples": len(truth),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "reference_correct": int(baseline.sum()),
        "source_agreement": float((actual == expected).mean()),
        "regressions": int((baseline & ~correct).sum()),
        "improvements": int((~baseline & correct).sum()),
        "max_logit_difference": float(np.abs(logits - reference).max()),
    }


def accuracy(args, bundle, output, record, predict):
    import numpy as np

    indices, chunks = bundle["accuracy_indices"], []
    chunk_size = bundle["protocol"]["chunk_size"]
    for offset in range(0, len(indices), chunk_size):
        path = output / f"chunk-{offset:06d}.json"
        selected = indices[offset : offset + chunk_size]
        if args.resume and path.exists():
            chunk = json.loads(path.read_text())
            if chunk["identity"] != record["identity"] or chunk["indices"] != selected or chunk["case"] != args.case:
                raise ValueError("Accuracy chunk identity mismatch")
            logits = np.asarray(chunk["logits"], np.float32)
        else:
            values = []
            for index, sample_index in enumerate(selected):
                if index % 20 == 0:
                    cool(record)
                data = (args.bundle / bundle["samples"][sample_index]["path"]).read_bytes()
                values.append(predict(preprocess(data, bundle["preprocessing"])))
            logits = np.stack(values)
            save_report(
                path,
                {"identity": record["identity"], "case": args.case, "indices": selected, "logits": logits.tolist()},
            )
        if logits.shape != (len(selected), record["classes"]) or not np.isfinite(logits).all():
            raise ValueError("Invalid saved accuracy chunk")
        chunks.append(logits)
        record["processed"] = min(offset + chunk_size, len(indices))
        save_report(output / "result.json", record)
    reference = np.load(args.bundle / bundle["reference"], allow_pickle=False)
    truth = [bundle["samples"][index]["label"] for index in indices]
    record.update(score(np.concatenate(chunks), reference, truth))


def timing(args, bundle, record, predict, image):
    protocol = bundle["protocol"]
    cool(record, limit=64)
    for _ in range(protocol["warmup"]):
        predict(image)
    inference, images = [], []
    for index in range(protocol["iterations"]):
        if index % 20 == 0:
            cool(record, limit=65)
        sample = bundle["samples"][(index * 37) % len(bundle["samples"])]
        data = (args.bundle / sample["path"]).read_bytes()
        image = preprocess(data, bundle["preprocessing"])
        begin = time.perf_counter()
        predict(image)
        inference.append((time.perf_counter() - begin) * 1000)
        begin = time.perf_counter()
        predict(preprocess(data, bundle["preprocessing"]))
        images.append((time.perf_counter() - begin) * 1000)
    record.update(raw_inference_ms=inference, raw_image_ms=images)
    for name, values in (("inference", inference), ("image_to_logits", images)):
        record[name] = {
            "median_ms": statistics.median(values),
            "p95_ms": sorted(values)[math.ceil(0.95 * len(values)) - 1],
        }


def worker(args, bundle, output, identity):
    case = bundle["cases"][args.case]
    record = {
        "identity": identity,
        "case": args.case,
        "mode": args.worker,
        "settings": case,
        "batch_size": 1,
        "input_shape": bundle["preprocessing"]["input_shape"],
        "classes": len(bundle["categories"]),
        "requirements": bundle["requirements"][case["backend"]],
        "started_at": time.time(),
        "pid": os.getpid(),
        "status": "running",
        "temperature_start_c": temperature(),
    }
    save_report(output / "result.json", record)
    start = time.monotonic()
    try:  # noqa: PLW0717 -- worker lifecycle boundary
        cool(record)
        model, predict = runtime(args.bundle, case, args.worker, output, record)
        image = preprocess((args.bundle / bundle["samples"][0]["path"]).read_bytes(), bundle["preprocessing"])
        first = time.perf_counter()
        predict(image)
        record["first_call_ms"] = (time.perf_counter() - first) * 1000
        if args.worker == "probe":
            probe_precision(model, image, args.bundle, case, record)
        elif args.worker == "accuracy":
            accuracy(args, bundle, output, record, predict)
        else:
            timing(args, bundle, record, predict, image)
        record["status"] = "completed"
    except UnsupportedError as exc:
        record.update(status="unsupported", error=str(exc))
    except Exception as exc:  # noqa: BLE001 -- preserve failures at the worker process boundary
        record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    record.update(
        elapsed_seconds=time.monotonic() - start,
        peak_rss_mib=peak_rss_mib(),
        temperature_end_c=temperature(),
        torch_imported="torch" in sys.modules,
    )
    save_report(output / "result.json", record)
    return record


def run(args):  # noqa: PLR0915 -- the coordinator owns process execution and reporting
    bundle = load_bundle(args.bundle)
    env = environment()
    env["packages"] = dict(sorted((dist.metadata["Name"], dist.version) for dist in distributions()))
    env["thread_environment"] = {
        name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    identity = hashlib.sha256(
        json.dumps({"bundle": digest(args.bundle / "bundle.json"), "environment": env}, sort_keys=True).encode()
    ).hexdigest()
    if args.worker:
        return worker(args, bundle, args.output, identity)["status"] == "failed"
    selected = args.cases or list(bundle["cases"])
    if set(selected) - bundle["cases"].keys():
        raise ValueError("Unknown case name")
    selected = [name for name in selected if bundle["cases"][name]["backend"] in args.backends]
    if not selected:
        raise ValueError("No cases selected")
    root = args.output / identity[:16]
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "identity": identity,
        "expected_cases": selected,
        "complete": False,
        "bundle_sha256": digest(args.bundle / "bundle.json"),
        "environment": env,
        "scope": {
            "latency": "Batch one, paired inference-only and memory-resident JPEG-to-logits samples; excludes disk IO, loading and thermal waits",
            "memory": "Fresh timing process lifetime peak RSS, including Python, imports, loading and preprocessing; excludes GPU/ANE memory",
            "thermal": "Bounded bursts; cool at 65C between 20 timing samples until <=60C, deadline 600s; no rewarm after cooling",
        },
        "cases": {},
    }
    save_report(root / "report.json", report)
    failed = False
    for name in selected:
        results = {}
        for mode in ("probe", "accuracy", *(f"timing-{i + 1}" for i in range(bundle["protocol"]["repeats"]))):
            output = root / name / mode
            output.mkdir(parents=True, exist_ok=True)
            path = output / "result.json"
            cached = json.loads(path.read_text()) if mode == "accuracy" and args.resume and path.exists() else None
            if (
                cached
                and cached["identity"] == identity
                and cached["case"] == name
                and cached["mode"] == mode
                and cached["status"] == "completed"
            ):
                result = {**cached, "reused": True}
            else:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(args.bundle),
                    str(output),
                    "--case",
                    name,
                    "--worker",
                    mode.split("-")[0],
                ]
                if args.resume:
                    command.append("--resume")
                with (output / "worker.log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)  # noqa: S603
                result = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"status": "failed", "error": "Worker exited without a result"}
                )
                if completed.returncode and result["status"] != "failed":
                    result = {**result, "status": "failed", "error": f"Worker exited {completed.returncode}"}
            results[mode] = result
            failed |= result["status"] == "failed"
            print(f"{platform.node()} {name} {mode}: {result['status']}", flush=True)
            if result["status"] != "completed":
                break
        report["cases"][name] = results
        save_report(root / "report.json", report)
    report["complete"] = True
    save_report(root / "report.json", report)
    rows = []
    for name, results in report["cases"].items():
        accuracy = results.get("accuracy", {})
        timing = [
            value for key, value in results.items() if key.startswith("timing-") and value["status"] == "completed"
        ]
        row = {
            "case": name,
            "status": next(
                (value["status"] for value in results.values() if value["status"] != "completed"), "completed"
            ),
        }
        if accuracy.get("status") == "completed" and timing:
            row.update(
                correct=accuracy["correct"],
                samples=accuracy["samples"],
                image_median_ms=statistics.median(value["image_to_logits"]["median_ms"] for value in timing),
                inference_median_ms=statistics.median(value["inference"]["median_ms"] for value in timing),
                peak_rss_mib=max(value["peak_rss_mib"] for value in timing),
                timing_repeats=len(timing),
            )
        rows.append(row)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case",
                "status",
                "correct",
                "samples",
                "image_median_ms",
                "inference_median_ms",
                "peak_rss_mib",
                "timing_repeats",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(root / "report.json", flush=True)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", help="Case names; default all")
    parser.add_argument("--backends", nargs="+", choices=tuple(PACKAGES), default=list(PACKAGES))
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching accuracy chunks/results; always measure fresh timings"
    )
    parser.add_argument("--worker", choices=("probe", "accuracy", "timing"), help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.bundle, args.output = args.bundle.resolve(), args.output.resolve()
    if args.worker:
        sys.exit(run(args))
    # One sweep per user/host, including invocations using different output directories.
    with (Path(tempfile.gettempdir()) / f"holocron-deployment-{os.getuid()}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another deployment sweep is running on this host\n")
        sys.exit(run(args))
