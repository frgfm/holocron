import pytest
import torch
from torch.nn import functional as F
from torch.optim import SGD
from torchvision.models import mobilenet_v3_small

from holocron.optim import wrapper


def _test_wrapper(name: str) -> None:
    lr = 1e-4
    input_shape = (3, 224, 224)
    num_batches = 4
    # Get model, optimizer and criterion
    model = mobilenet_v3_small(num_classes=10)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier[3].parameters():
        p.requires_grad_(True)
    # Pick an optimizer whose update is easy to verify
    optimizer = SGD(model.classifier[3].parameters(), lr=lr)

    # Wrap the optimizer
    opt_wrapper = wrapper.__dict__[name](optimizer)

    # Check gradient reset
    opt_wrapper.zero_grad()
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                assert torch.all(p.grad == 0.0)

    # Check update step
    p_ = model.classifier[3].weight
    p_val = p_.data.clone()

    # Random inputs
    input_t = torch.rand((num_batches, *input_shape), dtype=torch.float32)
    target = torch.zeros(num_batches, dtype=torch.long)

    # Update
    for _ in range(10):
        output = model(input_t)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt_wrapper.step()
    # Check update rule
    assert not torch.equal(p_.data, p_val)
    assert not torch.equal(p_.data, p_val - lr * p_.grad)

    # Repr
    assert len(repr(opt_wrapper).split("\n")) == len(repr(optimizer).split("\n")) + 4


def test_lookahead():
    _test_wrapper("Lookahead")


def test_scout():
    _test_wrapper("Scout")

    p = torch.nn.Parameter(torch.ones(2))
    optimizer = wrapper.Scout(SGD([p], lr=0.125), sync_period=3)
    for gradient in (1.0, 2.0, 3.0):
        p.grad = torch.tensor([gradient, 0.0])
        optimizer.step()
    torch.testing.assert_close(p, torch.tensor([0.625, 1.0]))


@pytest.mark.parametrize("optimizer_cls", [wrapper.Lookahead, wrapper.Scout])
@pytest.mark.parametrize("saved_steps", [2, 3])
def test_wrapper_checkpoint_resume(optimizer_cls, saved_steps, tmp_path):
    params = [torch.nn.Parameter(torch.tensor([1.0, -2.0])), torch.nn.Parameter(torch.tensor([3.0]))]
    base = SGD([{"params": [p], "lr": lr} for p, lr in zip(params, [0.1, 0.2], strict=True)], momentum=0.9)
    original = optimizer_cls(base, sync_rate=0.8, sync_period=3)
    if optimizer_cls is wrapper.Lookahead:
        params.append(torch.nn.Parameter(torch.tensor([0.5])))
        original.add_param_group({"params": [params[-1]], "lr": 0.05})

    def step(optimizer, parameters, index):
        for p in parameters:
            p.grad = p.detach().square() + index + 1
        optimizer.step()

    for index in range(saved_steps):
        step(original, params, index)
    path = tmp_path / "checkpoint.pth"
    torch.save({"params": params, "optimizer": original.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in checkpoint["params"]]
    restored = optimizer_cls(SGD([{"params": [p]} for p in restored_params], lr=0.01))
    restored.load_state_dict(checkpoint["optimizer"])

    # Continue through the next synchronization and a complete subsequent cycle.
    for index in range(saved_steps, saved_steps + 4):
        step(original, params, index)
        step(restored, restored_params, index)
        for expected, actual in zip(params, restored_params, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
