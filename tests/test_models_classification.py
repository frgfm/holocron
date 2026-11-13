from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from holocron.models import classification
from holocron.models.classification.convnext import LayerNorm2d
from holocron.models.classification.darknetv2 import DarknetBodyV2
from holocron.models.classification.darknetv3 import DarknetBodyV3
from holocron.models.classification.darknetv4 import DarknetBodyV4
from holocron.models.classification.repvit import _RepVGGDW  # noqa: PLC2701
from holocron.models.classification.tridentnet import TridentConv2d
from holocron.optim import AdamP
from holocron.trainer import ClassificationTrainer


@pytest.mark.parametrize(
    ("factory", "channels"),
    [
        (partial(LayerNorm2d, 4), 4),
        (partial(DarknetBodyV2, [(4, 1), (8, 1)], stem_channels=2), 3),
        (partial(DarknetBodyV2, [(4, 1), (8, 1)], stem_channels=2, passthrough=True), 3),
        (partial(DarknetBodyV3, [(4, 1), (8, 1)], stem_channels=2), 3),
        (partial(DarknetBodyV3, [(4, 1), (8, 1)], stem_channels=2, num_features=2), 3),
        (partial(DarknetBodyV4, [(4, 1), (8, 1)], stem_channels=2), 3),
        (partial(DarknetBodyV4, [(4, 1), (8, 1)], stem_channels=2, num_features=2), 3),
        (partial(TridentConv2d, 2, 2, 3, padding=1), 6),
    ],
)
def test_forward_helpers_accept_pytorch_input_keyword(factory, channels):
    model = factory().eval()
    images = torch.rand(2, channels, 16, 16)
    with torch.no_grad():
        expected = model(images)
        torch.testing.assert_close(model(x=images), expected)
        torch.testing.assert_close(model(input=images), expected)
    with pytest.raises(TypeError, match="either input or x"):
        model(input=images, x=images)
    with pytest.raises(TypeError, match="input tensor is required"):
        model()


@pytest.mark.parametrize("block", [classification.BasicBlock, classification.Bottleneck])
@pytest.mark.parametrize("drop_layer", [None, nn.Dropout2d])
@pytest.mark.parametrize("norm_layer", [nn.BatchNorm2d, partial(nn.GroupNorm, 1)])
def test_resnet_zero_init_residual(block, drop_layer, norm_layer):
    model = classification.ResNet(
        block, [1], [4], zero_init_residual=True, drop_layer=drop_layer, norm_layer=norm_layer
    ).eval()
    residual = next(module for module in model.modules() if isinstance(module, block))
    norms = [module for module in residual.conv if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm))]
    assert torch.count_nonzero(norms[-1].weight) == 0
    assert torch.all(norms[0].weight == 1)
    images = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(residual.conv(images), torch.zeros(2, 4 * block.expansion, 8, 8))
        torch.testing.assert_close(residual(images), residual.activation(residual.downsample(images)))


def _test_classification_model(name, num_classes, pretrained):
    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = classification.__dict__[name](pretrained=pretrained, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[-1] == num_classes

    # Check backprop is OK
    target = torch.zeros(batch_size, dtype=torch.long)
    model.train()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()


def test_repvgg_reparametrize():
    torch.manual_seed(0)
    num_classes = 10
    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = classification.repvgg_a0(pretrained=False, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    # Reparametrize
    model.reparametrize()
    # Check that there is no longer any Conv1x1 or BatchNorm
    for mod in model.modules():
        assert not isinstance(mod, nn.BatchNorm2d)
        if isinstance(mod, nn.Conv2d):
            assert mod.weight.data.shape[2:] == (3, 3)
    # Check that values are still matching
    with torch.no_grad():
        torch.testing.assert_close(out, model(x), rtol=1e-4, atol=1e-5)  # logit score, not prob


def test_mobileone_reparametrize():
    torch.manual_seed(0)
    num_classes = 10
    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = classification.mobileone_s0(pretrained=False, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    # Reparametrize
    model.reparametrize()
    # Check that there is no longer any Conv1x1 or BatchNorm
    for mod in model.modules():
        assert not isinstance(mod, nn.BatchNorm2d)
    # Check that values are still matching
    with torch.no_grad():
        torch.testing.assert_close(out, model(x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("arch", "training_params", "deployment_params"),
    [
        ("repvit_m0_9", 5_103_560, 5_067_056),
        ("repvit_m1_0", 6_852_900, 6_810_312),
        ("repvit_m1_1", 8_288_888, 8_244_312),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repvit_reparametrize(arch, training_params, deployment_params, dtype, tmp_path):
    torch.manual_seed(0)
    x = torch.rand((1, 3, 64, 64), dtype=dtype)
    model = classification.__dict__[arch](pretrained=False, num_classes=1000).to(dtype=dtype)
    assert sum(p.numel() for p in model.parameters()) == training_params

    loader = DataLoader(TensorDataset(torch.rand((8, 3, 64, 64), dtype=dtype), torch.arange(8) % 7), batch_size=4)
    initial_weights = model.features[0][0][0].weight.detach().clone()
    trainer = ClassificationTrainer(
        model,
        loader,
        loader,
        nn.CrossEntropyLoss(label_smoothing=0.1),
        AdamP(model.parameters(), lr=1e-3),
        output_file=str(tmp_path / "checkpoint.pth"),
    )
    trainer.fit_n_epochs(2, 1e-3)
    assert torch.isfinite(torch.tensor(trainer.min_loss))
    assert not torch.equal(initial_weights, model.features[0][0][0].weight)
    model.load_state_dict(torch.load(trainer.output_file, weights_only=True)["model"])
    with torch.no_grad():
        model.eval()
        out = model(x)
    model.reparametrize()
    model.reparametrize()

    assert sum(p.numel() for p in model.parameters()) == deployment_params
    assert not any(isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, _RepVGGDW)) for mod in model.modules())
    assert all(not mod.training for mod in model.modules())
    assert all(p.dtype == dtype for p in model.parameters())
    with torch.no_grad():
        torch.testing.assert_close(out, model(x), rtol=1e-4, atol=1e-5)


def test_repvit_reparametrize_requires_eval():
    model = classification.repvit_m0_9().eval()
    modules = tuple(model.modules())
    model.head[0].train()
    with pytest.raises(ValueError, match="call eval"):
        model.reparametrize()
    assert tuple(model.modules()) == modules


def test_repvit_custom_channels():
    model = classification.RepViT([72, 144, 288, 576], [1, 2, 2, 2], num_classes=7, in_channels=1)
    # The official SE rounding keeps 72 / 4 = 18 channels rounded to 16, not 24.
    assert model.features[1][0].token_mixer[1].fc1.out_channels == 16
    x = torch.rand((2, 1, 64, 64))
    assert model.features(x).shape == (2, 576, 2, 2)
    out = model(x)
    assert out.shape == (2, 7)
    nn.functional.cross_entropy(out, torch.tensor([0, 1])).backward()


@pytest.mark.parametrize(
    ("arch", "pretrained"),
    [
        ("darknet24", True),
        ("darknet19", True),
        ("darknet53", True),
        ("cspdarknet53", True),
        ("cspdarknet53_mish", True),
        ("resnet18", True),
        ("resnet34", True),
        ("resnet50", True),
        ("resnet101", True),
        ("resnet152", True),
        ("resnext50_32x4d", True),
        ("resnext101_32x8d", True),
        ("resnet50d", True),
        ("res2net50_26w_4s", True),
        ("tridentnet50", True),
        ("pyconv_resnet50", True),
        ("pyconvhg_resnet50", True),
        ("rexnet1_0x", True),
        ("rexnet1_3x", False),
        ("rexnet1_5x", False),
        ("rexnet2_0x", False),
        ("rexnet2_2x", False),
        ("sknet50", True),
        ("sknet101", True),
        ("sknet152", True),
        ("repvgg_a0", True),
        ("repvgg_b0", False),
        ("convnext_atto", True),
        ("convnext_femto", False),
        ("convnext_pico", False),
        ("convnext_nano", False),
        ("convnext_tiny", False),
        ("convnext_small", False),
        ("convnext_base", False),
        ("convnext_large", False),
        ("convnext_xl", False),
        ("mobileone_s0", True),
        ("mobileone_s1", False),
        ("mobileone_s2", False),
        ("mobileone_s3", False),
        ("repvit_m0_9", False),
        ("repvit_m1_0", False),
        ("repvit_m1_1", False),
    ],
)
def test_classification_model(arch, pretrained):
    num_classes = 1000 if arch.startswith("rexnet") else 10
    _test_classification_model(arch, num_classes, pretrained)
