from types import SimpleNamespace

import pytest
import torch
from torch import nn

from holocron.models import utils
from holocron.nn import SAM, BlurPool2d, DropBlock2d


def _test_conv_seq(conv_seq, expected_classes, expected_channels):
    assert len(conv_seq) == len(expected_classes)
    for layer, mod_class in zip(conv_seq, expected_classes, strict=False):
        assert isinstance(layer, mod_class)

    input_t = torch.rand(1, conv_seq[0].in_channels, 224, 224)
    out = torch.nn.Sequential(*conv_seq)(input_t)
    assert out.shape[:2] == (1, expected_channels)
    out.sum().backward()


def test_conv_sequence():
    mod = utils.conv_sequence(
        3,
        32,
        kernel_size=3,
        act_layer=nn.ReLU(inplace=True),
        norm_layer=nn.BatchNorm2d,
        drop_layer=DropBlock2d,
        attention_layer=SAM,
    )

    _test_conv_seq(mod, [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, SAM, DropBlock2d], 32)
    assert mod[0].kernel_size == (3, 3)

    mod = utils.conv_sequence(
        3,
        32,
        kernel_size=3,
        stride=2,
        act_layer=nn.ReLU(inplace=True),
        norm_layer=nn.BatchNorm2d,
        drop_layer=DropBlock2d,
        blurpool=True,
    )
    _test_conv_seq(mod, [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, BlurPool2d, DropBlock2d], 32)
    assert mod[0].kernel_size == (3, 3)
    assert mod[0].stride == (1, 1)
    assert mod[3].stride == 2
    assert mod[0].bias is None
    # Ensures that bias is added when there is no BN
    mod = utils.conv_sequence(3, 32, kernel_size=3, stride=2, act_layer=nn.ReLU(inplace=True))
    assert isinstance(mod[0].bias, nn.Parameter)


def test_fuse_conv_bn():
    # Check the channel verification
    with pytest.raises(AssertionError):
        utils.fuse_conv_bn(nn.Conv2d(3, 5, 3), nn.BatchNorm2d(3))

    # Prepare candidate modules
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False).eval()
    bn = nn.BatchNorm2d(8).eval()
    bn.weight.data = torch.rand(8)

    # Create the fused version
    fused_conv = nn.Conv2d(3, 8, 3, padding=1, bias=True).eval()
    k, b = utils.fuse_conv_bn(conv, bn)
    fused_conv.weight.data = k
    fused_conv.bias.data = b

    # Check values
    batch_size = 2
    x = torch.rand((batch_size, 3, 32, 32))
    with torch.no_grad():
        assert torch.allclose(bn(conv(x)), fused_conv(x), atol=1e-6)

    # Check the warning when there is already a bias
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=True).eval()
    k, b = utils.fuse_conv_bn(conv, bn)
    fused_conv.weight.data = k
    fused_conv.bias.data = b
    with torch.no_grad():
        assert torch.allclose(bn(conv(x)), fused_conv(x), atol=1e-6)


def test_remote_checkpoint_loading_is_safe(monkeypatch, tmp_path):
    model = nn.Linear(2, 2)
    model.default_cfg = None
    config_path = tmp_path / "config.json"
    config_path.write_text('{"arch": "test_model", "classes": ["a", "b"]}')
    checkpoint_path = tmp_path / "pytorch_model.bin"
    torch.save(model.state_dict(), checkpoint_path)

    url_kwargs = {}

    def load_url(_url, **kwargs):
        url_kwargs.update(kwargs)
        return model.state_dict()

    monkeypatch.setattr(utils, "load_state_dict_from_url", load_url)
    utils.load_pretrained_params(model, "https://example.org/model.pth")
    assert url_kwargs["weights_only"] is True

    downloads = []

    def download(_repo_id, filename, **kwargs):
        downloads.append((filename, kwargs.get("revision"), kwargs.get("dry_run", False)))
        if kwargs.get("dry_run"):
            return SimpleNamespace(commit_hash="resolved-sha")
        return config_path if filename == "config.json" else checkpoint_path

    torch_load = torch.load

    def load_checkpoint(path, **kwargs):
        assert kwargs["weights_only"] is True
        return torch_load(path, **kwargs)

    monkeypatch.setitem(utils.models.__dict__, "test_model", lambda **_kwargs: model)
    monkeypatch.setattr(utils, "hf_hub_download", download)
    monkeypatch.setattr(utils.torch, "load", load_checkpoint)

    assert utils.model_from_hf_hub("owner/model", revision="main") is model
    assert downloads == [
        ("config.json", "main", True),
        ("config.json", "resolved-sha", False),
        ("pytorch_model.bin", "resolved-sha", False),
    ]
