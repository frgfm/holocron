import io
import json
import subprocess  # noqa: S404
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper
from PIL import Image
from torchvision.transforms import v2

from scripts import eval_deployment as deployment
from scripts.prepare_deployment import prepare


@pytest.fixture
def experiment(tmp_path):
    inputs = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    outputs = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["logits"], axes=[2, 3])], "rgb", [inputs], [outputs]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, tmp_path / "original.onnx")
    (tmp_path / "export.onnx").symlink_to("original.onnx")
    manifest = {
        "input_shape": [1, 3, 8, 8],
        "resize": 9,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "categories": ["r", "g", "b"],
        "samples": [],
    }
    reference = []
    session = ort.InferenceSession(str(tmp_path / "export.onnx"), providers=["CPUExecutionProvider"])
    for index, label in enumerate((0, 1, 2, 0)):
        path = tmp_path / f"image-{index}.png"
        color = [0, 0, 0]
        color[label] = 200 + index
        Image.new("RGB", (11 + index, 15 - index), tuple(color)).save(path)
        manifest["samples"].append({"path": path.name, "label": label, "sha256": deployment.digest(path)})
        image = deployment.preprocess(path.read_bytes(), manifest)
        reference.append(session.run(None, {"images": image})[0].reshape(-1))
    np.save(tmp_path / "reference.npy", np.stack(reference)[[3, 1]])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "accuracy.json").write_text(
        json.dumps({**manifest, "samples": [manifest["samples"][i] for i in (3, 1)]})
    )
    cases = {"onnx-fp32": {"backend": "onnx", "artifact": "export.onnx", "requested_precision": "fp32", "threads": 1}}
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    config = {
        "schema_version": 1,
        "manifest": "manifest.json",
        "accuracy_manifest": "accuracy.json",
        "reference": "reference.npy",
        "cases": "cases.json",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def make_bundle(config, destination):
    root = prepare(config, destination)
    path = root / "bundle.json"
    bundle = json.loads(path.read_text())
    bundle["protocol"] = {"warmup": 1, "iterations": 3, "repeats": 2, "chunk_size": 1}
    path.write_text(json.dumps(bundle))
    return root


def invoke(bundle, output, *extra):
    return subprocess.run(  # noqa: S603 -- run the frozen test fixture with this Python interpreter
        [sys.executable, str(bundle / "scripts/eval_deployment.py"), str(bundle), str(output), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("shape", [(17, 30), (30, 17), (19, 20)])
def test_preprocessing_matches_checkpoint_transforms(shape):
    pixels = np.random.default_rng(4).integers(0, 256, (*shape, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    data = io.BytesIO()
    image.save(data, format="PNG")
    prep = {"input_shape": [1, 3, 8, 8], "resize": 9, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    transform = v2.Compose([
        v2.Resize(9),
        v2.CenterCrop(8),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(prep["mean"], prep["std"]),
    ])
    np.testing.assert_array_equal(deployment.preprocess(data.getvalue(), prep)[0], transform(image).numpy())


def test_freeze_is_portable_and_repeatable(experiment, tmp_path):
    first = prepare(experiment, tmp_path / "first")
    second = prepare(experiment, tmp_path / "second")
    assert (first / "bundle.json").read_bytes() == (second / "bundle.json").read_bytes()
    bundle = deployment.load_bundle(first)
    assert bundle["accuracy_indices"] == [3, 1]
    assert not any(path.is_symlink() for path in first.rglob("*"))
    (tmp_path / "original.onnx").unlink()
    for path in tmp_path.glob("image-*.png"):
        path.unlink()
    deployment.load_bundle(first)
    with pytest.raises(FileExistsError):
        prepare(experiment, first)
    artifact = first / bundle["cases"]["onnx-fp32"]["artifact"]
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        deployment.load_bundle(first)


def test_calibration_leakage_and_input_hashes(experiment, tmp_path):
    config = json.loads(experiment.read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    (tmp_path / "calibration.json").write_text(json.dumps({"samples": manifest["samples"][:1]}))
    experiment.write_text(json.dumps({**config, "calibration": "calibration.json"}))
    with pytest.raises(ValueError, match="overlap"):
        prepare(experiment, tmp_path / "leaking")
    assert not (tmp_path / "leaking").exists()
    experiment.write_text(json.dumps(config))
    (tmp_path / "image-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(experiment, tmp_path / "changed")


def test_fresh_processes_resume_and_failures(experiment, tmp_path):
    bundle, output = make_bundle(experiment, tmp_path / "bundle"), tmp_path / "results"
    result = invoke(bundle, output)
    assert result.returncode == 0, result.stderr + result.stdout
    report_path = next(output.glob("*/report.json"))
    assert json.loads(report_path.read_text())["complete"]
    report = json.loads(report_path.read_text())["cases"]["onnx-fp32"]
    assert report["accuracy"]["correct"] == report["accuracy"]["samples"] == 2
    assert len({record["pid"] for record in report.values()}) == 4
    assert not any(record["torch_imported"] for record in report.values())
    assert all(len(report[mode]["raw_image_ms"]) == 3 for mode in ("timing-1", "timing-2"))
    resumed = invoke(bundle, output, "--resume")
    assert resumed.returncode == 0, resumed.stderr
    fresh = json.loads(report_path.read_text())["cases"]["onnx-fp32"]
    assert fresh["accuracy"]["reused"]
    assert fresh["accuracy"]["pid"] == report["accuracy"]["pid"]
    assert fresh["timing-1"]["pid"] != report["timing-1"]["pid"]

    accuracy_dir = report_path.parent / "onnx-fp32" / "accuracy"
    first_chunk = accuracy_dir / "chunk-000000.json"
    saved = first_chunk.stat().st_mtime_ns
    (accuracy_dir / "result.json").unlink()
    (accuracy_dir / "chunk-000001.json").unlink()
    assert invoke(bundle, output, "--resume").returncode == 0
    assert first_chunk.stat().st_mtime_ns == saved
    (accuracy_dir / "result.json").unlink()
    chunk = json.loads(first_chunk.read_text())
    first_chunk.write_text(json.dumps({**chunk, "case": "another-model"}))
    failed = invoke(bundle, output, "--resume")
    assert failed.returncode == 1
    record = json.loads((accuracy_dir / "result.json").read_text())
    assert record["status"] == "failed"
    assert "identity mismatch" in record["error"]


def test_invalid_model_is_failed_not_unsupported(experiment, tmp_path):
    (tmp_path / "original.onnx").write_bytes(b"not an ONNX model")
    bundle = make_bundle(experiment, tmp_path / "bundle")
    result = invoke(bundle, tmp_path / "results")
    assert result.returncode == 1
    report = json.loads(next((tmp_path / "results").glob("*/report.json")).read_text())
    assert report["cases"]["onnx-fp32"]["probe"]["status"] == "failed"


def test_unsupported_thermal_budget_and_path_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(deployment.sys, "platform", "linux")
    with pytest.raises(deployment.UnsupportedError, match="requires macOS"):
        deployment.runtime(tmp_path, {"backend": "coreml"}, "probe", tmp_path, {})
    monkeypatch.setattr(deployment, "temperature", lambda: 70)
    ticks = iter((0, 10))
    monkeypatch.setattr(deployment.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="thermal budget"):
        deployment.cool({}, timeout=1)
    with pytest.raises(ValueError, match="escapes"):
        deployment.safe_path(tmp_path, "../outside")


def test_shell_rejects_unsafe_targets_before_launch(tmp_path):
    (tmp_path / "bundle.json").write_text("{}")
    targets = tmp_path / "targets.tsv"
    targets.write_text("local\t/bin/false\tunused\nbad;host\t/bin/python\twork\n")
    script = Path(__file__).resolve().parents[1] / "scripts/run_deployment_sweep.sh"
    result = subprocess.run(  # noqa: S603 -- all arguments are controlled test paths
        ["/bin/sh", str(script), str(tmp_path), str(targets), str(tmp_path / "results")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Invalid target" in result.stderr
    assert not (tmp_path / "results/local/launcher.log").exists()


def test_shell_requires_a_manifest_before_launch(tmp_path):
    targets = tmp_path / "targets.tsv"
    targets.write_text("local\t/bin/false\tunused\n")
    script = Path(__file__).resolve().parents[1] / "scripts/run_deployment_sweep.sh"
    result = subprocess.run(  # noqa: S603 -- controlled test paths; no remote target
        ["/bin/sh", str(script), str(tmp_path), str(targets), str(tmp_path / "results")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "results/local").exists()
