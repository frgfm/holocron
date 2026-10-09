import copy
import os
import platform
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from holocron import models
from holocron.models.coreml import export_coreml


@pytest.fixture(scope="module")
def coreml():
    required = os.environ.get("HOLOCRON_REQUIRE_COREML_RUNTIME") == "1"
    if required:
        assert platform.system() == "Darwin", "macOS runtime is required"
        assert platform.machine() == "arm64", "Apple Silicon runtime is required"
        import coremltools as ct  # noqa: PLC0415
    else:
        ct = pytest.importorskip("coremltools")
    return ct


@pytest.mark.parametrize(
    ("arch", "reparameterized", "trainer_checkpoint"),
    [("resnet18", False, False), ("mobileone_s0", False, True), ("mobileone_s0", True, False)],
)
def test_coreml_classifier(arch, reparameterized, trainer_checkpoint, coreml, tmp_path):  # noqa: PLR0915
    torch.manual_seed(42)
    model = models.get_model(arch, num_classes=3).eval()
    images = torch.rand(1, 3, 64, 64)
    samples = (images, torch.zeros_like(images), torch.rand(images.shape, generator=torch.Generator().manual_seed(0)))
    with torch.no_grad():
        if reparameterized:
            model.reparametrize()
        references = [model(sample) for sample in samples]
    for index, expected in enumerate(references):
        assert torch.isfinite(expected).all()
        assert not torch.allclose(expected, references[(index + 1) % 3], rtol=1e-3, atol=3e-5)
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "epoch": 1} if trainer_checkpoint else model.state_dict(), checkpoint)
    path = tmp_path / "model.mlpackage"
    verify = platform.system() == "Darwin" and platform.machine() == "arm64"
    command = [
        sys.executable,
        "scripts/export_to_coreml.py",
        arch,
        "--checkpoint",
        str(checkpoint),
        "--num-classes",
        "3",
        "--height",
        "64",
        "--width",
        "64",
        "--path",
        str(path),
    ]
    if reparameterized:
        command.append("--reparameterized")
    if not verify:
        command.append("--unverified")
    result = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True, check=True)  # noqa: S603
    assert ("match PyTorch" if verify else "NOT checked") in result.stdout
    # A wrong class count must be rejected by strict checkpoint loading.
    command[command.index("--num-classes") + 1] = "4"
    rejected = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False)  # noqa: S603
    assert rejected.returncode != 0
    assert "size mismatch" in rejected.stderr
    shutil.rmtree(path)
    original = copy.deepcopy(model.state_dict())
    architecture = repr(model)
    model.train()
    for iteration in range(2 if reparameterized else 1):
        if not verify:
            with pytest.warns(UserWarning, match="inference was not checked"):
                export_coreml(model, images, path, verify=False)
        else:
            export_coreml(model, images, path)
        assert model.training
        assert repr(model) == architecture
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        if reparameterized and iteration == 0:
            shutil.rmtree(path)
    model.eval()
    package = coreml.models.MLModel(str(path), skip_model_load=not verify, compute_units=coreml.ComputeUnit.CPU_ONLY)
    spec = package.get_spec()
    assert spec.WhichOneof("Type") == "mlProgram"
    assert list(spec.description.input[0].type.multiArrayType.shape) == list(images.shape)
    if verify:
        with torch.no_grad():
            for index, (sample, expected) in enumerate(zip(samples, references, strict=True)):
                actual = torch.from_numpy(package.predict({"images": sample.numpy()})["logits"])
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-5)
                error = (actual - expected).abs()
                print(  # noqa: T201
                    f"{arch} reparameterized={reparameterized} input={index}: max_abs={error.max().item():.8g} "
                    f"max_rel={(error / expected.abs().clamp_min(1e-12)).max().item():.8g}"
                )
    shutil.rmtree(path)


@pytest.mark.parametrize("frozen", [False, True])
def test_coreml_rejects_incorrect_trace(frozen, coreml, tmp_path):  # noqa: ARG001
    class BrokenClassifier(nn.Module):
        def forward(self, images):  # noqa: PLR6301
            if frozen:
                return (images * 0 if torch.jit.is_tracing() else images).mean(dim=(2, 3))
            return images.mean(dim=(2, 3)) + (1 if images.sum() > 0 else 0)

    with (
        pytest.warns(UserWarning, match="inference was not checked"),
        pytest.raises(RuntimeError, match="Tracing verification input"),
    ):
        export_coreml(BrokenClassifier(), torch.ones(1, 3, 4, 4), tmp_path / "model.mlpackage", verify=False)
    assert not list(tmp_path.iterdir())


def test_coreml_failures_preserve_destination(coreml, monkeypatch, tmp_path):
    path = tmp_path / "model.mlpackage"
    path.mkdir()
    marker = path / "weights"
    marker.write_bytes(b"existing package")
    images = torch.ones(1, 3, 4, 4)
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_coreml(model, images, path)
    assert marker.read_bytes() == b"existing package"
    shutil.rmtree(path)

    def fail(*_args, **_kwargs):
        raise ValueError("unsupported operation")

    monkeypatch.setattr(coreml, "convert", fail)
    with (
        pytest.warns(UserWarning, match="inference was not checked"),
        pytest.raises(RuntimeError, match="conversion failed: unsupported"),
    ):
        export_coreml(model, images, path, verify=False)
    assert not list(tmp_path.iterdir())


def test_coreml_runtime_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match=r"requires Apple Silicon.*verify=False"):
        export_coreml(nn.Identity(), torch.ones(1, 3, 4, 4), tmp_path / "model.mlpackage")


def test_coreml_rejects_wrong_runtime(coreml, monkeypatch, tmp_path):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Core ML runtime test requires Apple Silicon")
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
    images = torch.ones(1, 3, 4, 4)
    convert = coreml.convert

    def incorrect(_traced, **kwargs):
        wrong = nn.Sequential(model, nn.Linear(3, 3, bias=False))
        with torch.no_grad():
            wrong[1].weight.copy_(-torch.eye(3))
        return convert(torch.jit.trace(wrong, images), **kwargs)

    monkeypatch.setattr(coreml, "convert", incorrect)
    with pytest.raises(RuntimeError, match="Core ML CPU verification input 0"):
        export_coreml(model, images, tmp_path / "model.mlpackage")
    assert not list(tmp_path.iterdir())
