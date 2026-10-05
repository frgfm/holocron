from collections import OrderedDict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from holocron.models import classification


def _official_backbone(model):
    # Write the authors' flat block names from an independently initialized backbone.
    blocks = {
        f"features.{stage_idx}.{block_idx}": idx
        for idx, (stage_idx, block_idx) in enumerate(
            (
                (stage_idx, block_idx)
                for stage_idx in range(1, 5)
                for block_idx in range(len(model.features[stage_idx]))
            ),
            1,
        )
    }
    result = OrderedDict()
    for key, tensor in model.state_dict().items():
        if key.startswith("head."):
            continue
        parts = key.split(".")
        if parts[1] == "0":
            prefix, suffix = "features.0", ".".join(parts[2:])
        else:
            prefix, suffix = f"features.{blocks['.'.join(parts[:3])]}", ".".join(parts[3:])
        suffix = suffix.replace("conv3.0.", "conv.c.").replace("conv3.1.", "conv.bn.")
        suffix = suffix.replace("token_mixer.0.norm.", "token_mixer.0.bn.")
        suffix = suffix.replace("channel_mixer.block.", "channel_mixer.m.")
        for path in ("0", "2", "token_mixer.0", "token_mixer.2", "channel_mixer.m.0", "channel_mixer.m.2"):
            suffix = suffix.replace(f"{path}.0.", f"{path}.c.").replace(f"{path}.1.", f"{path}.bn.")
        result[f"{prefix}.{suffix}"] = tensor.detach().clone()
    return result


def _official_head(channels, classes):
    head = nn.Sequential(OrderedDict([("bn", nn.BatchNorm1d(channels)), ("l", nn.Linear(channels, classes))]))
    with torch.no_grad():
        head.bn.weight.uniform_(0.5, 1.5)
        head.bn.bias.normal_()
        head.bn.running_mean.normal_()
        head.bn.running_var.uniform_(0.5, 1.5)
    return head.eval()


@pytest.mark.parametrize("arch", ["repvit_m0_9", "repvit_m1_0", "repvit_m1_1"])
@pytest.mark.parametrize("distilled", [False, True])
def test_official_checkpoint_preserves_single_or_distilled_logits(arch, distilled):
    torch.manual_seed(1)
    reference = classification.__dict__[arch](num_classes=7).eval()
    with torch.no_grad():
        for module in reference.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(std=0.1)
                module.running_mean.normal_(std=0.1)
                module.running_var.uniform_(0.5, 1.5)
    width = reference.head[1].in_features
    heads = [_official_head(width, 7) for _ in range(2 if distilled else 1)]
    state = _official_backbone(reference)
    for name, head in zip(("classifier", "classifier_dist")[: len(heads)], heads, strict=True):
        state.update({f"classifier.{name}.{key}": value for key, value in head.state_dict().items()})
    source_before = {key: value.clone() for key, value in state.items()}
    model = classification.__dict__[arch](num_classes=7).eval()
    model.load_official_state_dict(state)
    images = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        features = reference.pool(reference.features(images))
        expected = torch.stack([head(features) for head in heads]).mean(0)
        torch.testing.assert_close(model(images), expected, rtol=1e-4, atol=1e-5)
        model.reparametrize()
        torch.testing.assert_close(model(images), expected, rtol=1e-4, atol=1e-5)
    assert all(torch.equal(value, source_before[key]) for key, value in state.items())


def test_backbone_import_keeps_custom_head_trainable():
    reference = classification.repvit_m0_9(num_classes=1000)
    state = _official_backbone(reference)
    model = classification.repvit_m0_9(num_classes=3)
    head_before = {key: value.clone() for key, value in model.head.state_dict().items()}
    model.load_official_state_dict(state, include_head=False)
    assert all(torch.equal(value, head_before[key]) for key, value in model.head.state_dict().items())
    images = torch.randn(2, 3, 32, 32)
    F.cross_entropy(model(images), torch.tensor([0, 2])).backward()
    assert model.features[0][0][0].weight.grad is not None
    assert model.head[1].weight.grad is not None


@pytest.mark.parametrize("bad_key", ["unknown", "features.100.token_mixer.0.c.weight"])
def test_official_import_rejects_unknown_keys_before_copying(bad_key):
    model = classification.repvit_m0_9()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = _official_backbone(model)
    state[bad_key] = torch.ones(1)
    with pytest.raises(ValueError, match=r"unexpected|different number"):
        model.load_official_state_dict(state, include_head=False)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_official_import_rejects_incompatible_or_fused_models():
    source = classification.repvit_m0_9()
    state = _official_backbone(source)
    incompatible = classification.repvit_m1_0()
    with pytest.raises(ValueError, match="does not match"):
        incompatible.load_official_state_dict(state, include_head=False)
    with pytest.raises(ValueError, match="classifier is incomplete"):
        source.load_official_state_dict(state)
    source.eval().reparametrize()
    with pytest.raises(ValueError, match="before reparametrizing"):
        source.load_official_state_dict(state, include_head=False)
