import math

import pytest
import torch

from holocron import ops


@pytest.fixture
def boxes():
    return torch.tensor(
        [[0, 0, 100, 100], [50, 50, 100, 100], [50, 50, 150, 150], [100, 100, 200, 200]], dtype=torch.float32
    )


def test_iou_penalty(boxes):
    penalty = ops.boxes.iou_penalty(boxes, boxes)

    # Check shape
    assert penalty.shape == (boxes.shape[0], boxes.shape[0])
    # Unit tests
    for idx in range(boxes.shape[0]):
        assert penalty[idx, idx].item() == 0

    assert penalty[0, 1].item() == 25**2 / 100**2
    assert penalty[0, 3].item() == 100**2 / 200**2
    assert penalty[0, 2].item() == penalty[2, 3].item()


def test_diou_loss(boxes):
    diou = ops.boxes.diou_loss(boxes, boxes)

    # Check shape
    assert diou.shape == (boxes.shape[0], boxes.shape[0])
    # Unit tests
    for idx in range(boxes.shape[0]):
        assert diou[idx, idx].item() == 0.0

    assert diou[0, 1].item() == 1 - 0.25 + 25**2 / 100**2
    assert diou[0, 3].item() == 1 + 100**2 / 200**2
    assert diou[0, 2].item() == diou[2, 3].item()


def test_box_giou(boxes):
    giou = ops.boxes.box_giou(boxes, boxes)

    # Check shape
    assert giou.shape == (boxes.shape[0], boxes.shape[0])
    # Unit tests
    for idx in range(boxes.shape[0]):
        assert giou[idx, idx].item() == 1.0

    assert giou[0, 1].item() == 0.25
    assert giou[0, 3].item() == -(200**2 - 2 * 100**2) / 200**2
    assert giou[0, 2].item() == giou[2, 3].item()


def test_aspect_ratio(boxes):
    # All boxes are squares so arctan should yield Pi / 4
    assert torch.allclose(ops.boxes.aspect_ratio(boxes), math.pi / 4 * torch.ones(boxes.shape[0]))


def test_aspect_ratio_consistency(boxes):
    # All boxes have the same aspect ratio
    assert torch.equal(ops.boxes.aspect_ratio_consistency(boxes, boxes), torch.zeros(boxes.shape[0], boxes.shape[0]))


def test_ciou_loss(boxes):
    ciou = ops.boxes.ciou_loss(boxes, boxes)

    # Check shape
    assert ciou.shape == (boxes.shape[0], boxes.shape[0])
    # Unit tests
    for idx in range(boxes.shape[0]):
        assert ciou[idx, idx].item() == 0.0
    assert ciou[0, 2].item() == ciou[2, 3].item()


def test_ciou_loss_aspect_ratio_term():
    boxes1 = torch.tensor([[-1, -0.5, 1, 0.5]], dtype=torch.float32, requires_grad=True)
    boxes2 = torch.tensor([[-0.5, -1, 0.5, 1]], dtype=torch.float32)

    iou = 1 / 3
    v = 4 / math.pi**2 * (math.atan(2) - math.atan(0.5)) ** 2
    expected = 1 - iou + v**2 / (1 - iou + v)
    loss = ops.boxes.ciou_loss(boxes1, boxes2)

    assert loss.item() == pytest.approx(expected)
    assert loss.item() > ops.boxes.diou_loss(boxes1, boxes2).item()
    loss.backward()
    assert torch.isfinite(boxes1.grad).all()
