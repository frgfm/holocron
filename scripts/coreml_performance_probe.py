"""One-off FP32 app deployment probe; deliberately outside the feature PR."""

import gc
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def rss():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) * 1024


def emit(label, result):
    print(label + " " + json.dumps(result, sort_keys=True), flush=True)


def torch_worker(directory, arch, fused, threads):
    import numpy as np
    import torch
    from holocron import models

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    baseline = rss()
    model = models.get_model(arch, num_classes=3).eval()
    if fused:
        model.reparametrize()
    state = torch.load(directory / "checkpoint.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    inputs = [torch.from_numpy(np.fromfile(directory / f"input{i}.bin", dtype=np.float32).reshape(1, 3, 224, 224)) for i in range(3)]
    gc.collect()
    with torch.inference_mode():
        outputs = [model(value).numpy().copy().tolist()[0] for value in inputs]
        for i in range(30):
            model(inputs[i % 3]).numpy().copy()
        warm = rss()
        durations = []
        for i in range(150):
            start = time.perf_counter_ns()
            model(inputs[i % 3]).numpy().copy()
            durations.append((time.perf_counter_ns() - start) / 1e6)
    durations.sort()
    print(json.dumps({"backend": "pytorch_cpu", "threads": threads,
                      "median_ms": durations[75], "p95_ms": durations[142],
                      "baseline_rss_bytes": baseline, "warm_rss_bytes": warm,
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "outputs": outputs}), flush=True)


def prepare(directory, arch, fused):
    import numpy as np
    import torch
    from holocron import models
    from holocron.models.coreml import export_coreml

    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = models.get_model(arch, num_classes=3).eval()
    if fused:
        model.reparametrize()
    images = torch.rand(1, 3, 224, 224)
    samples = [images, torch.zeros_like(images), torch.rand(images.shape, generator=torch.Generator().manual_seed(0))]
    with torch.inference_mode():
        references = [model(sample).numpy().copy() for sample in samples]
    for i, reference in enumerate(references):
        assert np.isfinite(reference).all()
        assert not np.allclose(reference, references[(i + 1) % 3], rtol=1e-3, atol=3e-5)
    torch.save(model.state_dict(), directory / "checkpoint.pth")
    for i, sample in enumerate(samples):
        sample.numpy().tofile(directory / f"input{i}.bin")
    # Verification uses actual Core ML CPU prediction at the measured 224x224 shape.
    export_coreml(model, images, directory / "model.mlpackage")
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(directory / "model.mlpackage"), str(directory)], check=True)
    (directory / "references.json").write_text(json.dumps([value.tolist()[0] for value in references]))


def main():
    import coremltools as ct
    import numpy as np
    import torch
    import torchvision

    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    environment = {"platform": platform.platform(), "python": platform.python_version(),
                   "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "coremltools": ct.__version__, "numpy": np.__version__,
                   "devices": [type(device).__name__ for device in ct.models.MLModel.get_available_compute_devices()],
                   "logical_cpus": os.cpu_count(), "mps_available": torch.backends.mps.is_available(),
                   "shape": [1, 3, 224, 224], "classes": 3, "precision": "FP32",
                   "warmups": 30, "iterations": 150, "repetitions": 3}
    emit("BENCH_ENV", environment)
    results = []
    with tempfile.TemporaryDirectory() as root:
        for arch, fused in [("resnet18", False), ("mobileone_s0", False), ("mobileone_s0", True)]:
            label = arch + ("_reparameterized" if fused else "")
            directory = Path(root) / label
            directory.mkdir()
            subprocess.run([sys.executable, __file__, "prepare", str(directory), arch, str(int(fused))], check=True)
            references = np.asarray(json.loads((directory / "references.json").read_text()), dtype=np.float32)
            rounds = []

            def run(backend, threads=0):
                command = ([sys.executable, __file__, "torch", str(directory), arch, str(int(fused)), str(threads)]
                           if backend == "pytorch_cpu" else ["/tmp/coreml-probe", str(directory), backend.removeprefix("coreml_")])
                output = subprocess.check_output(command, text=True)
                result = json.loads(output.strip().splitlines()[-1])
                actual = np.asarray(result.pop("outputs"), dtype=np.float32)
                assert actual.shape == references.shape and np.isfinite(actual).all()
                np.testing.assert_allclose(actual, references, rtol=1e-3, atol=3e-5)
                error = np.abs(actual - references)
                result.update(configuration=label, max_abs=float(error.max()),
                              max_rel=float((error / np.maximum(np.abs(references), 1e-12)).max()))
                emit("BENCH_SAMPLE", result)
                results.append(result)
                return result

            sweep = [run("pytorch_cpu", threads) for threads in range(1, min(4, os.cpu_count()) + 1)]
            fastest = min(sweep, key=lambda result: result["median_ms"])
            threads = fastest["threads"]
            selected = {"pytorch_cpu": [fastest], "coreml_cpu": [], "coreml_all": []}
            for repetition in range(3):
                # Reverse order in alternate repetitions to reduce ordering bias.
                order = ["pytorch_cpu", "coreml_cpu", "coreml_all"]
                if repetition % 2:
                    order.reverse()
                for backend in order:
                    if backend == "pytorch_cpu" and repetition == 0:
                        continue
                    selected[backend].append(run(backend, threads))
            for backend, samples in selected.items():
                summary = {"configuration": label, "backend": backend, "threads": threads if backend == "pytorch_cpu" else "managed",
                           "median_ms": float(np.median([item["median_ms"] for item in samples])),
                           "median_range_ms": [min(item["median_ms"] for item in samples), max(item["median_ms"] for item in samples)],
                           "p95_ms": float(np.median([item["p95_ms"] for item in samples])),
                           "warm_rss_mib": float(np.median([item["warm_rss_bytes"] for item in samples])) / 2**20,
                           "incremental_rss_mib": float(np.median([item["warm_rss_bytes"] - item["baseline_rss_bytes"] for item in samples])) / 2**20,
                           "peak_rss_mib": float(np.median([item["peak_rss_bytes"] for item in samples])) / 2**20,
                           "max_abs": max(item["max_abs"] for item in samples), "max_rel": max(item["max_rel"] for item in samples)}
                emit("BENCH_SUMMARY", summary)
            package_size = sum(path.stat().st_size for path in (directory / "model.mlpackage").rglob("*") if path.is_file())
            emit("BENCH_SIZE", {"configuration": label, "checkpoint_mib": (directory / "checkpoint.pth").stat().st_size / 2**20,
                                "package_mib": package_size / 2**20})
    Path("/tmp/coreml-performance-results.json").write_text(json.dumps({"environment": environment, "samples": results}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "prepare":
        prepare(Path(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4])))
    else:
        torch_worker(Path(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4])), int(sys.argv[5]))
