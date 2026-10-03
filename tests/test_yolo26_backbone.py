import math

import pytest
import torch
from torch import nn

from holocron.models._yolo26 import C3k2, ConvNormAct, YOLO26Backbone, fuse_model  # noqa: PLC2701


@pytest.mark.parametrize("shape", [(64, 64), (65, 79)])
def test_yolo26_backbone_features_and_gradients(shape):
    model = YOLO26Backbone(in_channels=1)
    inputs = torch.rand(2, 1, *shape, requires_grad=True)
    features = model(inputs)
    for output, channels, stride in zip(features, model.out_channels, (8, 16, 32), strict=True):
        assert output.shape == (2, channels, math.ceil(shape[0] / stride), math.ceil(shape[1] / stride))
    sum(output.square().mean() for output in features).backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("in_channels", [0, -1, 1.5, True])
def test_yolo26_backbone_invalid_channels(in_channels):
    with pytest.raises(ValueError, match="positive integer"):
        YOLO26Backbone(in_channels)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_yolo26_fusion(dtype):
    torch.manual_seed(3)
    model = YOLO26Backbone().to(dtype).eval()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean.uniform_(-0.1, 0.1)
            layer.running_var.uniform_(0.5, 1.5)
    inputs = torch.randn(2, 3, 65, 79, dtype=dtype)
    with torch.no_grad():
        reference = model(inputs)
        fuse_model(model)
        actual = model(inputs)
        fuse_model(model)
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    assert all(parameter.dtype == dtype for parameter in model.parameters())
    for expected, output in zip(reference, actual, strict=True):
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)


def test_yolo26_fusion_requires_eval_without_partial_mutation():
    model = YOLO26Backbone()
    with pytest.raises(ValueError, match=r"eval\(\)"):
        fuse_model(model)
    model.eval()
    model.stage5.train()
    with pytest.raises(ValueError, match=r"eval\(\)"):
        fuse_model(model)
    assert isinstance(model.stem[0][1], nn.BatchNorm2d)


def test_yolo26_attention_refinement_and_depthwise_fusion():
    block = nn.Sequential(C3k2(256, 256, attention=True), ConvNormAct(256, 256, groups=256)).eval()
    inputs = torch.randn(2, 256, 3, 4)
    with torch.no_grad():
        expected = block(inputs)
        fuse_model(block)
        actual = block(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
