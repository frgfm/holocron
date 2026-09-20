from copy import deepcopy
from typing import Any

import pytest
import torch
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

from holocron import optim


def _test_optimizer(name: str, **kwargs: Any) -> None:
    lr = 1e-4
    input_shape = (3, 224, 224)
    num_batches = 4
    # Get model and optimizer
    model = mobilenet_v3_small(num_classes=10)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier[3].parameters():
        p.requires_grad_(True)
    optimizer = optim.__dict__[name](model.classifier[3].parameters(), lr=lr, **kwargs)

    # Save param value
    p_ = model.classifier[3].weight
    p_val = p_.data.clone()

    # Random inputs
    input_t = torch.rand((num_batches, *input_shape), dtype=torch.float32)
    target = torch.zeros(num_batches, dtype=torch.long)

    # Update
    optimizer.zero_grad()
    output = model(input_t)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    # Test
    assert p_.grad is not None
    assert not torch.equal(p_.data, p_val)


def test_lars():
    _test_optimizer("LARS", momentum=0.9, weight_decay=2e-5)


def test_lamb():
    _test_optimizer("LAMB", weight_decay=2e-5)


def test_ralars():
    _test_optimizer("RaLars", weight_decay=2e-5)


def test_tadam():
    _test_optimizer("TAdam")


def test_adabelief():
    _test_optimizer("AdaBelief")


def test_adamp():
    _test_optimizer("AdamP")


def test_adan():
    _test_optimizer("Adan")


def test_adan_betas_and_state():
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    betas = (0.8, 0.9, 0.95)
    optimizer = optim.Adan([
        {"params": [params[0]]},
        {"params": [params[1]], "betas": betas, "amsgrad": True},
    ])
    assert optimizer.param_groups[0]["betas"] == (0.98, 0.92, 0.99)
    assert optimizer.param_groups[1]["betas"] == betas
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    assert all(torch.all(param < 1) for param in params)
    assert "max_exp_avg_delta" in optimizer.state[params[1]]

    restored_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    restored = optim.Adan([{"params": [param]} for param in restored_params])
    state = deepcopy(optimizer.state_dict())
    del state["param_groups"][0]["amsgrad"]
    state["state"][0]["step"] = torch.tensor(state["state"][0]["step"])
    restored.load_state_dict(state)
    assert restored.param_groups[0]["amsgrad"] is False
    assert restored.param_groups[1]["betas"] == betas
    assert restored.param_groups[1]["amsgrad"] is True
    for param in restored_params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    restored.step()
    for actual, expected in zip(restored_params, params, strict=True):
        torch.testing.assert_close(actual, expected)

    for kwargs in (
        {"lr": -1.0},
        {"eps": -1.0},
        {"weight_decay": -1.0},
        {"betas": (0.9, 0.99)},
        {"betas": (0.8, 0.9, 1.0)},
        {"betas": (0.8, -0.1, 0.99)},
    ):
        with pytest.raises(ValueError):
            optim.Adan(params, **kwargs)


def test_ademamix():
    _test_optimizer("AdEMAMix")
