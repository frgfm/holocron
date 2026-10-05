import json
import subprocess  # noqa: S404
import sys
from pathlib import Path

import pytest
import torch

from scripts import eval_latency as benchmark


def test_measure_waits_for_work_and_uses_warmup(monkeypatch):
    clock = 0
    calls = []

    def forward():
        nonlocal clock
        clock += 0.002
        calls.append("forward")
        return "output"

    def sync():
        nonlocal clock
        clock += 0.001
        calls.append("sync")

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock)
    result, output = benchmark.measure(forward, batch_size=2, num_it=3, warmup_it=2, sync=sync)

    assert output == "output"
    assert calls == [
        "sync",
        "forward",
        "sync",  # First call.
        "forward",
        "forward",
        "sync",  # Warm-up.
        "forward",
        "sync",
        "forward",
        "sync",
        "forward",
        "sync",  # Latency.
        "forward",
        "forward",
        "forward",
        "sync",  # Throughput.
    ]
    assert result["first_ms"] == pytest.approx(3)
    assert result["median_ms"] == pytest.approx(3)
    assert result["p95_ms"] == pytest.approx(3)
    assert result["throughput_per_s"] == pytest.approx(6 / 0.007)


def test_synchronize_selects_requested_accelerator(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    for name in ("cpu", "cuda:1", "mps"):
        benchmark.synchronize(torch.device(name))
    assert calls == ["cuda:1", "mps"]


@pytest.mark.parametrize("option", ["--size", "--batch-size", "--it", "--repeat", "--threads"])
def test_reject_zero_measurement_settings(option):
    with pytest.raises(SystemExit):
        benchmark.get_parser().parse_args(["rexnet1_0x", option, "0"])


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_cli_records_isolated_trials(tmp_path, backend):
    if backend == "onnx":
        pytest.importorskip("onnxruntime")
    output = tmp_path / "results.json"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(Path(benchmark.__file__).resolve()),
            "rexnet1_0x",
            "--backend",
            backend,
            "--size",
            "32",
            "--batch-size",
            "2",
            "--it",
            "2",
            "--warmup",
            "0",
            "--repeat",
            "2",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["config"]["warmup"] == 0
    assert report["config"]["backend"] == backend
    assert report["config"]["batch_size"] == 2
    assert len({run["pid"] for run in report["runs"]}) == 2
    assert report["summary"]["median_ms"] > 0
    assert report["summary"]["throughput_per_s"] > 0
    if sys.platform != "win32":
        assert report["summary"]["peak_rss_mib"] > 0
    assert report["environment"]["versions"]["torch"]
    assert len(report["environment"]["harness_sha256"]) == 64
    assert list(tmp_path.iterdir()) == [output]
