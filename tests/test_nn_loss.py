import pytest
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.nn.functional import cross_entropy

from holocron import nn
from holocron.nn import functional as F


def _test_loss_function(loss_fn, same_loss=0.0, multi_label=False):
    num_batches = 2
    num_classes = 4
    # 4 classes
    x = torch.ones(num_batches, num_classes)
    x[:, 0, ...] = 100
    x.requires_grad_(True)

    # Identical target
    if multi_label:
        target = torch.zeros_like(x)
        target[:, 0] = 1.0
    else:
        target = torch.zeros(num_batches, dtype=torch.long)
    assert abs(loss_fn(x, target).item() - same_loss) < 1e-3
    assert torch.allclose(
        loss_fn(x, target, reduction="none"), same_loss * torch.ones(num_batches, dtype=x.dtype), atol=1e-3
    )

    # Check that class rescaling works
    x = torch.rand(num_batches, num_classes, requires_grad=True)
    target = torch.rand(x.shape) if multi_label else (num_classes * torch.rand(num_batches)).to(torch.long)
    weights = torch.ones(num_classes)
    assert loss_fn(x, target).item() == loss_fn(x, target, weight=weights).item()

    # Check that ignore_index works
    assert loss_fn(x, target).item() == loss_fn(x, target, ignore_index=num_classes).item()
    # Ignore an index we are certain to be in the target
    ignore_index = torch.unique(target.argmax(dim=1))[0].item() if multi_label else torch.unique(target)[0].item()
    assert loss_fn(x, target).item() != loss_fn(x, target, ignore_index=ignore_index)
    # Check backprop
    loss = loss_fn(x, target, ignore_index=0)
    loss.backward()

    # Test reduction
    assert torch.allclose(loss_fn(x, target, reduction="sum"), loss_fn(x, target, reduction="none").sum(), atol=1e-6)
    assert torch.allclose(
        loss_fn(x, target, reduction="mean"), loss_fn(x, target, reduction="sum") / target.shape[0], atol=1e-6
    )


def test_focal_loss():
    # Common verification
    _test_loss_function(F.focal_loss)

    num_batches = 2
    num_classes = 4
    x = torch.rand(num_batches, num_classes, 20, 20)
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

    # Value check
    assert torch.allclose(F.focal_loss(x, target, gamma=0), cross_entropy(x, target), atol=1e-5)
    # Equal probabilities
    x = torch.ones(num_batches, num_classes, 20, 20)
    assert torch.allclose(
        (1 - 1 / num_classes) * F.focal_loss(x, target, gamma=0), F.focal_loss(x, target, gamma=1), atol=1e-5
    )

    assert repr(nn.FocalLoss()) == "FocalLoss(gamma=2.0, reduction='mean')"


@pytest.mark.parametrize("ignore_index", [255, -100, 0])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_focal_loss_ignored_segmentation_pixels(ignore_index, reduction):
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    target = torch.ones(2, 4, 5, dtype=torch.long)
    target[:, 0] = ignore_index
    # Exercise non-contiguous masks as well as ignored labels outside the class range.
    target = target.transpose(-1, -2)
    logits = logits.transpose(-1, -2)
    logits.retain_grad()
    loss = F.focal_loss(logits, target, ignore_index=ignore_index, gamma=0, reduction=reduction)
    expected = cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)
    assert torch.allclose(loss, expected)
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.permute(0, 2, 3, 1)[target == ignore_index] == 0).all()


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_focal_loss_all_ignored_and_weighted_single_pixel(reduction):
    logits = torch.randn(1, 3, 1, 1, requires_grad=True)
    ignored = torch.full((1, 1, 1), 255, dtype=torch.long)
    weights = torch.tensor([0.5, 1.0, 2.0])
    loss = F.focal_loss(logits, ignored, weight=weights, ignore_index=255, reduction=reduction)
    assert loss.sum().item() == 0
    loss.sum().backward()
    assert (logits.grad == 0).all()
    target = torch.full_like(ignored, 2)
    assert torch.isfinite(F.focal_loss(logits, target, weight=weights, reduction=reduction)).all()


def test_multilabel_cross_entropy():
    num_batches = 2
    num_classes = 4

    # Common verification
    _test_loss_function(F.multilabel_cross_entropy, multi_label=True)

    x = torch.rand(num_batches, num_classes, 20, 20)
    target = torch.zeros_like(x)
    target[:, 0] = 1.0

    # Value check
    assert torch.allclose(F.multilabel_cross_entropy(x, target), cross_entropy(x, target.argmax(dim=1)), atol=1e-5)

    assert repr(nn.MultiLabelCrossEntropy()) == "MultiLabelCrossEntropy(reduction='mean')"


def test_complement_cross_entropy():
    num_batches = 2
    num_classes = 4

    x = torch.rand((num_batches, num_classes, 20, 20), requires_grad=True)
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

    # Backprop
    out = F.complement_cross_entropy(x, target, ignore_index=0)
    out.backward()

    assert repr(nn.ComplementCrossEntropy()) == "ComplementCrossEntropy(gamma=-1, reduction='mean')"


def test_mc_loss():
    num_batches = 2
    num_classes = 4
    xi = 2
    # 4 classes
    x = torch.ones(num_batches, xi * num_classes)
    x[:, 0, ...] = 10
    target = torch.zeros(num_batches, dtype=torch.long)

    mod = Linear(xi * num_classes, xi * num_classes)

    # Check backprop
    for reduction in ["mean", "sum", "none"]:
        for p in mod.parameters():
            p.grad = None
        train_loss = F.mutual_channel_loss(mod(x), target, ignore_index=0, reduction=reduction)
        if reduction == "none":
            assert train_loss.shape == (num_batches,)
            train_loss = train_loss.sum()
        train_loss.backward()
        assert isinstance(mod.weight.grad, torch.Tensor)

    # Check type casting of weights
    for p in mod.parameters():
        p.grad = None
    class_weights = torch.ones(num_classes, dtype=torch.float16)
    ignore_index = 0

    criterion = nn.MutualChannelLoss(weight=class_weights, ignore_index=ignore_index, xi=xi)
    train_loss = criterion(mod(x), target)
    train_loss.backward()
    assert isinstance(mod.weight.grad, torch.Tensor)
    assert repr(criterion) == f"MutualChannelLoss(reduction='mean', xi={xi}, alpha=1)"


def test_cb_loss():
    num_batches = 2
    num_classes = 4
    x = torch.rand(num_batches, num_classes, 20, 20)
    beta = 0.99
    num_samples = 10 * torch.ones(num_classes, dtype=torch.long)

    # Identical target
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)
    base_criterion = CrossEntropyLoss()
    base_loss = base_criterion(x, target)
    criterion = nn.ClassBalancedWrapper(base_criterion, num_samples, beta=beta)

    assert isinstance(criterion.criterion, CrossEntropyLoss)
    assert criterion.criterion.weight is not None

    # Value tests
    loss_value = criterion(x, target)
    assert loss_value.shape == base_loss.shape
    # assert torch.allclose(criterion(x, target), (1 - beta) / (1 - beta ** num_samples[0]) * base_loss, atol=1e-5)
    # With pre-existing weights
    base_criterion = CrossEntropyLoss(weight=torch.ones(num_classes, dtype=torch.float32))
    base_weights = base_criterion.weight.clone()
    criterion = nn.ClassBalancedWrapper(base_criterion, num_samples, beta=beta)
    assert not torch.equal(base_weights, criterion.criterion.weight)
    # assert torch.allclose(criterion(x, target), (1 - beta) / (1 - beta ** num_samples[0]) * base_loss, atol=1e-5)

    assert repr(criterion) == "ClassBalancedWrapper(CrossEntropyLoss(), beta=0.99)"


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_mc_loss_ignores_void_pixels_in_both_terms(reduction, all_ignored):
    logits = torch.randn(2, 9, 4, 5, requires_grad=True)
    target = torch.randint(3, (2, 4, 5))
    target[:, :2] = 255
    if all_ignored:
        target.fill_(255)
        logits = torch.full_like(logits, 3e38, requires_grad=True)
    loss = F.mutual_channel_loss(logits, target, ignore_index=255, xi=3, reduction=reduction)
    assert torch.isfinite(loss).all()
    if all_ignored:
        assert (loss == 0).all()
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.permute(0, 2, 3, 1)[target == 255] == 0).all()


def test_mc_loss_evaluation_is_deterministic():
    criterion = nn.MutualChannelLoss(xi=3).eval()
    logits = torch.randn(2, 9, 4, 5)
    target = torch.randint(3, (2, 4, 5))
    first = criterion(logits, target)
    torch.manual_seed(123)
    second = criterion(logits, target)
    assert torch.equal(first, second)
    assert torch.allclose(first, F.mutual_channel_loss(logits, target, xi=3, training=False))


def test_mc_loss_channel_mask_preserves_negative_logits():
    logits = torch.tensor([[-4.0, -4.0, -1.0, -1.0]], requires_grad=True)
    target = torch.tensor([0])
    expected = cross_entropy(torch.tensor([[-4.0, -1.0]]), target)
    actual = F.mutual_channel_loss(logits, target, xi=2, alpha=0)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("loss", ["focal", "mc"])
def test_dense_segmentation_losses_accumulate_in_float32(dtype, loss):
    channels = 3 if loss == "focal" else 6
    logits = torch.zeros(1, channels, 128, 128, dtype=dtype)
    logits[:, 1 if loss == "focal" else slice(2, 4)] = -8
    logits.requires_grad_(True)
    target = torch.ones(1, 128, 128, dtype=torch.long)
    criterion = F.focal_loss if loss == "focal" else lambda x, y: F.mutual_channel_loss(x, y, xi=2, training=False)
    actual = criterion(logits, target)
    expected = criterion(logits.float(), target)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_dice_loss():
    num_batches = 2
    num_classes = 4

    x = torch.rand((num_batches, num_classes, 20, 20), requires_grad=True)
    target = torch.rand(num_batches, num_classes, 20, 20)

    # Backprop
    out = F.dice_loss(x, target)
    out.backward()

    # Weighted loss
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2
    out = F.dice_loss(x, target, weight=class_weights)
    out.backward()

    assert repr(nn.DiceLoss()) == "DiceLoss(reduction='mean', gamma=1.0, eps=1e-08)"


def test_poly_loss():
    _test_loss_function(F.poly_loss)
    _test_loss_function(F.poly_loss, multi_label=True)

    num_batches = 2
    num_classes = 4

    x = torch.rand((num_batches, num_classes, 20, 20), requires_grad=True)
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

    # Backprop
    out = F.poly_loss(x, target)
    out.backward()

    # Weighted loss
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2
    out = F.poly_loss(x, target, weight=class_weights)
    out.backward()

    assert repr(nn.PolyLoss()) == "PolyLoss(eps=2.0, reduction='mean')"
