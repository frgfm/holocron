import pytest
import torch
from torch import nn

from holocron import trainer


def test_freeze_bn():
    # Simple module with BN
    mod = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
    nb = mod[1].num_batches_tracked.clone()
    rm = mod[1].running_mean.clone()
    rv = mod[1].running_var.clone()
    # Freeze & forward
    for p in mod.parameters():
        p.requires_grad_(False)
    trainer.freeze_bn(mod)
    for _ in range(10):
        _ = mod(torch.rand((1, 3, 32, 32)))
    # Check that stats were not updated
    assert torch.equal(mod[1].num_batches_tracked, nb)
    assert torch.equal(mod[1].running_mean, rm)
    assert torch.equal(mod[1].running_var, rv)


@pytest.mark.parametrize(
    ("norm_layer", "shape"),
    [(nn.BatchNorm1d, (4, 2)), (nn.BatchNorm2d, (4, 2, 3, 3)), (nn.BatchNorm3d, (4, 2, 3, 3, 3))],
)
@pytest.mark.parametrize("track_running_stats", [False, True])
def test_freeze_model_batchnorm_transitions(norm_layer, shape, track_running_stats):
    bn = norm_layer(2, track_running_stats=track_running_stats)
    model = nn.Sequential(bn)
    inputs = torch.full(shape, 3.0)
    initial_stats = {name: value.clone() for name, value in bn.named_buffers()}

    trainer.freeze_model(model.train(), "0")
    assert not any(p.requires_grad for p in bn.parameters())
    assert not bn.training
    assert bn.track_running_stats is track_running_stats
    model(inputs)

    # Validation and the next training epoch must both leave frozen statistics intact.
    for _ in range(2):
        model.eval()
        model(inputs)
        trainer.freeze_bn(model.train())
        model(inputs)
        for name, value in bn.named_buffers():
            torch.testing.assert_close(value, initial_stats[name], rtol=0, atol=0)

    trainer.freeze_model(model.train())
    assert all(p.requires_grad for p in bn.parameters())
    assert bn.training
    assert bn.track_running_stats is track_running_stats
    assert torch.isfinite(model(inputs)).all()
    if track_running_stats:
        assert bn.num_batches_tracked.item() == 1
        assert not torch.equal(bn.running_mean, initial_stats["running_mean"])
        assert not torch.equal(bn.running_var, initial_stats["running_var"])
    else:
        assert bn.running_mean is bn.running_var is bn.num_batches_tracked is None


@pytest.mark.parametrize("track_running_stats", [False, True])
def test_freeze_model_allows_batchnorm_stat_updates(track_running_stats):
    bn = nn.BatchNorm1d(2, track_running_stats=track_running_stats)
    model = nn.Sequential(bn)
    trainer.freeze_model(model.train(), "0")
    trainer.freeze_model(model.train(), "0", frozen_bn_stat_update=True)

    assert not any(p.requires_grad for p in bn.parameters())
    assert bn.training
    assert bn.track_running_stats is track_running_stats
    model(torch.full((4, 2), 3.0))
    if track_running_stats:
        assert bn.num_batches_tracked.item() == 1
        torch.testing.assert_close(bn.running_mean, torch.full((2,), 0.3))
    else:
        assert bn.running_mean is bn.running_var is bn.num_batches_tracked is None


def test_freeze_model():
    # Simple model
    mod = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True))
    trainer.freeze_model(mod, "0")
    # Check that the correct layers were frozen
    assert not any(p.requires_grad for p in mod[0].parameters())
    assert all(p.requires_grad for p in mod[2].parameters())
    with pytest.raises(ValueError):
        trainer.freeze_model(mod, "wrong_layer")

    # Freeze last layer
    for p in mod[-1].parameters():
        p.requires_grad_(False)
    trainer.freeze_model(mod, "0")
    # Ensure the last layer is now unfrozen
    assert all(p.requires_grad for p in mod[-1].parameters())
