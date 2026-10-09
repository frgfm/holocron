import math
from argparse import ArgumentParser, Namespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from holocron.optim import AdamP
from holocron.trainer import ClassificationTrainer, DetectionTrainer, SegmentationTrainer, resolve_device
from holocron.utils import CTCCodec
from holocron.utils.data import Mixup
from references._common import add_loading_args, create_loader  # noqa: PLC2701
from references.classification.benchmark_repvit_imagenette import count_macs
from references.recognition.train import ctc_loss

DEVICES = [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amp", [False, True])
def test_classification_training_device(device, amp, tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5))
    loader = DataLoader(TensorDataset(torch.randn(8, 3, 8, 8), torch.arange(8) % 5), batch_size=2)
    criterion = nn.CrossEntropyLoss(weight=torch.ones(5))
    learner = ClassificationTrainer(
        model,
        loader,
        loader,
        criterion,
        torch.optim.SGD(model.parameters(), lr=0.01),
        device=device,
        amp=amp,
        gradient_acc=2,
        output_file=str(tmp_path / "weights.pth"),
    )
    before = model[0].weight.detach().cpu().clone()
    learner.fit_n_epochs(1, 0.01, sched_type="cosine")
    assert next(model.parameters()).device.type == torch.device(device).type
    assert criterion.weight.device.type == torch.device(device).type
    assert not torch.equal(before, model[0].weight.detach().cpu())
    assert all(math.isfinite(value) for value in learner.evaluate().values())
    assert learner.scaler.is_enabled() == (amp and torch.device(device).type != "cpu") if amp else True
    loaded = torch.load(tmp_path / "weights.pth", map_location="cpu", weights_only=True)
    assert all(value.device.type == "cpu" for value in loaded["model"].values())


@pytest.mark.parametrize("device", DEVICES)
def test_segmentation_training_device(device, tmp_path):
    model = nn.Conv2d(3, 2, 1)
    targets = torch.randint(2, (4, 8, 8))
    targets[:, 0] = 255
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 8, 8), targets), batch_size=2)
    learner = SegmentationTrainer(
        model,
        loader,
        loader,
        nn.CrossEntropyLoss(ignore_index=255),
        torch.optim.SGD(model.parameters(), lr=0.01),
        device=device,
        num_classes=2,
        output_file=str(tmp_path / "weights.pth"),
    )
    learner.fit_n_epochs(1, 0.01, sched_type="cosine")
    assert all(math.isfinite(value) for value in learner.evaluate().values())


@pytest.mark.parametrize("device", DEVICES)
def test_detection_target_device(device):
    model = nn.Linear(1, 1)
    learner = DetectionTrainer(model, None, None, None, torch.optim.SGD(model.parameters(), lr=0.01), device=device)
    images, targets = learner.to_device(
        [torch.zeros(3, 8, 8)],
        [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}],
    )
    assert images[0].device.type == torch.device(device).type
    assert all(value.device.type == torch.device(device).type for value in targets[0].values())
    legacy_images, _ = learner.to_cuda(images, targets)
    torch.testing.assert_close(legacy_images[0], images[0])


def test_device_selection_and_legacy_cuda_indices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("auto") == torch.device("cuda:0")
    assert resolve_device("cuda") == torch.device("cuda:0")
    assert resolve_device(1) == resolve_device("1") == resolve_device("cuda:1")
    with pytest.raises(ValueError, match="Invalid device index"):
        resolve_device(2)
    with pytest.raises(ValueError, match="Invalid device index"):
        resolve_device(-1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == resolve_device(None) == torch.device("cpu")
    with pytest.raises(ValueError, match="not available"):
        resolve_device("mps")
    with pytest.raises(AssertionError, match="cannot access"):
        resolve_device("cuda")


def test_mixup_loader_works_with_spawn():
    dataset = TensorDataset(torch.randn(4, 3, 8, 8), torch.arange(4) % 2)
    args = Namespace(batch_size=2, workers=1, device="cpu")
    loader = create_loader(
        dataset,
        args,
        training=True,
        collate_fn=Mixup(2).collate,
        multiprocessing_context="spawn",
    )
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 8, 8)
    assert targets.shape == (2, 2)
    torch.testing.assert_close(targets.sum(1), torch.ones(2))
    assert not loader.pin_memory


@pytest.mark.parametrize("value", ["cpu", "mps", "cuda", "cuda:1", "0", "auto"])
def test_reference_cli_accepts_portable_devices(value):
    parser = ArgumentParser()
    add_loading_args(parser)
    assert parser.parse_args(["--device", value]).device == value


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("delta", [0.0, 10.0])
def test_adamp_device_projection_matches_reference(device, delta):
    value = torch.tensor([[1.0, 2.0]])
    gradient = torch.tensor([[3.0, 4.0]])
    update = gradient / (gradient.abs() + 1e-8)
    if delta:
        normalized = value / (value.norm() + 1e-8)
        update -= (normalized * update).sum() * normalized
    parameter = nn.Parameter(value.to(device).clone())
    parameter.grad = gradient.to(device)
    optimizer = AdamP([parameter], lr=0.1, betas=(0.0, 0.0), delta=delta)
    optimizer.step()
    torch.testing.assert_close(parameter.detach().cpu(), value - 0.1 * update)


@pytest.mark.parametrize("device", DEVICES)
def test_ctc_loss_preserves_accelerator_gradients(device):
    probabilities = torch.randn(5, 1, 3, device=device).log_softmax(-1).detach().requires_grad_()
    loss = ctc_loss(probabilities, torch.tensor([5]), ["AB"], CTCCodec("AB"))
    loss.backward()
    assert torch.isfinite(loss)
    assert probabilities.grad is not None
    assert probabilities.grad.device == probabilities.device
    assert torch.isfinite(probabilities.grad).all()


def test_benchmark_macs_count_convolution_and_linear():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)).eval()
    assert count_macs(model) == 222 * 222 * 4 * 3 * 3 * 3 + 5 * 4
    assert not model[0]._forward_hooks
