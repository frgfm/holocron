import copy

import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from holocron.models.segmentation import YOLO26Semantic, yolo26n_sem
from holocron.trainer import SegmentationTrainer


@pytest.mark.parametrize("shape", [(64, 96), (35, 47)])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_yolo26_semantic_shapes_and_gradients(shape, auxiliary):
    model = yolo26n_sem(num_classes=4, in_channels=1, auxiliary=auxiliary)
    images = torch.randn(2, 1, *shape)
    output = model(images)
    scores = output["out"] if auxiliary else output
    assert scores.shape == (2, 4, *shape)
    target = torch.randint(4, (2, *shape))
    target[:, :2] = 255
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss = criterion(scores, target)
    if auxiliary:
        assert output["aux"].shape == scores.shape
        loss += 0.5 * criterion(output["aux"], target)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        assert model(images).shape == scores.shape


def test_yolo26_semantic_fusion_and_checkpoint():
    torch.manual_seed(0)
    model = yolo26n_sem().eval()
    inputs = torch.randn(1, 3, 65, 79)
    reference = model(inputs)
    loaded = yolo26n_sem().eval()
    loaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(loaded(inputs), reference)
    fused = copy.deepcopy(model).fuse()
    torch.testing.assert_close(fused(inputs), reference, atol=1e-5, rtol=1e-4)
    assert fused.aux_classifier is None
    assert not any(isinstance(module, nn.BatchNorm2d) for module in fused.modules())
    assert fused.fuse() is fused
    reloaded_fused = yolo26n_sem().eval().fuse()
    reloaded_fused.load_state_dict(fused.state_dict())
    torch.testing.assert_close(reloaded_fused(inputs), reference, atol=1e-5, rtol=1e-4)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_632_902
    assert sum(parameter.numel() for parameter in fused.parameters()) == 1_552_795


def test_yolo26_semantic_factory_validation():
    with pytest.raises(ValueError, match="Pretrained"):
        yolo26n_sem(pretrained=True)
    with pytest.raises(ValueError, match="positive"):
        YOLO26Semantic(num_classes=0)
    with pytest.raises(ValueError, match="positive"):
        YOLO26Semantic(in_channels=0)
    with pytest.raises(ValueError, match="eval"):
        yolo26n_sem().fuse()


def test_yolo26_semantic_trainer_ignored_pixels_and_heldout_evaluation():
    generator = torch.Generator().manual_seed(9)
    images = torch.randn(4, 3, 32, 40, generator=generator)
    targets = torch.randint(3, (4, 32, 40), generator=generator)
    targets[:, :3] = 255
    train_loader = DataLoader(TensorDataset(images[:2], targets[:2]), batch_size=2)
    val_loader = DataLoader(TensorDataset(images[2:], targets[2:]), batch_size=1)
    model = yolo26n_sem(num_classes=3)
    learner = SegmentationTrainer(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(ignore_index=255),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        num_classes=3,
    )
    model.train()
    images, targets = next(iter(train_loader))
    loss = learner._get_loss(images, targets)
    loss.backward()
    assert model.classifier[-1].weight.grad is not None
    assert model.aux_classifier[-1].weight.grad is not None
    metrics = learner.evaluate()
    assert 0 <= metrics["mean_iou"] <= 1
    assert 0 <= metrics["acc_global"] <= 1
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
    model.train()
    model.zero_grad()
    learner._get_loss(images, torch.full_like(targets, 255)).backward()
    assert all(parameter.grad is not None and not parameter.grad.any() for parameter in model.parameters())


def test_yolo26_semantic_onnx_export(tmp_path):
    model = yolo26n_sem(num_classes=3).eval().fuse()
    images = torch.rand(1, 3, 35, 47)
    path = tmp_path / "yolo26-semantic.onnx"
    with torch.inference_mode():
        expected = model(images)
        torch.onnx.export(model, images, path, opset_version=20, dynamo=False, input_names=["images"])
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    output = ReferenceEvaluator(graph).run(None, {"images": images.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(output), expected, atol=1e-5, rtol=1e-4)
