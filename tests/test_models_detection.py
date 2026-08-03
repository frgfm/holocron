import math
from pathlib import Path

import pytest
import torch

from holocron.models import detection
from holocron.models.detection.yolo import _post_process  # noqa: PLC2701
from holocron.models.detection.yolov4 import YoloLayer, Yolov4Head


def _test_detection_model(name, input_size):
    num_classes = 10
    batch_size = 2
    x = torch.rand((batch_size, 3, *input_size))
    model = detection.__dict__[name](pretrained=True, num_classes=num_classes).eval()
    # Check backbone pretrained
    model = detection.__dict__[name](pretrained_backbone=True, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, list)
    assert len(out) == x.shape[0]
    if len(out) > 0:
        assert isinstance(out[0].get("boxes"), torch.Tensor)
        assert isinstance(out[0].get("scores"), torch.Tensor)
        assert isinstance(out[0].get("labels"), torch.Tensor)

    # Check that list of Tensors does not change output
    x_list = [torch.rand(3, *input_size) for _ in range(batch_size)]
    with torch.no_grad():
        out_list = model(x_list)
        assert len(out_list) == len(out)

    # Training mode without target
    model = model.train()
    with pytest.raises(ValueError):
        model(x)
    # Generate targets
    num_boxes = [3, 4]
    gt_boxes = []
    for num in num_boxes:
        boxes = torch.rand((num, 4), dtype=torch.float)
        # Ensure format xmin, ymin, xmax, ymax
        boxes[:, :2] *= boxes[:, 2:]
        # Ensure some anchors will be assigned
        boxes[0, :2] = 0
        boxes[0, 2:] = 1
        # Check cases where cell can get two assignments
        boxes[1, :2] = 0.2
        boxes[1, 2:] = 0.8
        gt_boxes.append(boxes)
    gt_labels = [(num_classes * torch.rand(num)).to(dtype=torch.long) for num in num_boxes]

    # Loss computation
    loss = model(x, [{"boxes": boxes, "labels": labels} for boxes, labels in zip(gt_boxes, gt_labels, strict=False)])
    assert isinstance(loss, dict)
    for subloss in loss.values():
        assert isinstance(subloss, torch.Tensor)
        assert subloss.requires_grad
        assert not torch.isnan(subloss)

    # Loss computation with no GT
    gt_boxes = [torch.zeros((0, 4)) for _ in num_boxes]
    gt_labels = [torch.zeros(0, dtype=torch.long) for _ in num_boxes]
    loss = model(x, [{"boxes": boxes, "labels": labels} for boxes, labels in zip(gt_boxes, gt_labels, strict=False)])
    sum(v for v in loss.values()).backward()


@pytest.mark.parametrize(
    ("arch", "input_shape"),
    [
        ("yolov1", (448, 448)),
        ("yolov2", (416, 416)),
        ("yolov4", (608, 608)),
    ],
)
def test_detection_model(arch, input_shape):
    _test_detection_model(arch, input_shape)


@pytest.mark.parametrize(
    ("arch", "input_shape"),
    [
        ("yolov1", (448, 448)),
        ("yolov2", (416, 416)),
        ("yolov4", (608, 608)),
    ],
)
def test_detection_onnx_export(arch, input_shape, tmpdir_factory):
    model = detection.__dict__[arch](pretrained=False, num_classes=10).eval()
    tmp_path = Path(str(tmpdir_factory.mktemp("onnx"))).joinpath(f"{arch}.onnx")
    img_tensor = torch.rand((1, 3, *input_shape))
    with torch.no_grad():
        torch.onnx.export(
            model, img_tensor, tmp_path, export_params=True, opset_version=20, dynamo=False, verbose=False
        )


@torch.inference_mode()
def test_yolov1():
    input_shape = (448, 448)
    n, h, w = 2, 7, 7
    num_anchors = 2
    num_classes = 10
    model = detection.yolov1(num_classes=10, pretrained_backbone=False)

    # Forward
    t = torch.rand((n, 3, *input_shape), dtype=torch.float32)
    out = model._forward(t)
    assert out.shape == (n, h * w * (num_anchors * 5 + num_classes))

    # Format outputs
    t = torch.rand((n, h * w * (num_anchors * 5 + num_classes)), dtype=torch.float32)
    b_coords, b_o, b_scores = model._format_outputs(t)
    assert b_coords.shape == (n, h, w, num_anchors, 4)
    assert b_o.shape == (n, h, w, num_anchors)
    assert b_scores.shape == (n, h, w, 1, num_classes)
    assert torch.all(b_coords <= 1)
    assert torch.all(b_coords >= 0)
    assert torch.all(b_o <= 1)
    assert torch.all(b_o >= 0)
    assert torch.allclose(b_scores.sum(-1), torch.ones(1))

    # Compute loss
    target = [
        {
            "boxes": torch.tensor([[0, 0, 1 / 7, 1 / 7]], dtype=torch.float32),
            "labels": torch.zeros((1,), dtype=torch.long),
        }
    ]
    pred_boxes = torch.zeros((1, h, w, num_anchors, 4), dtype=torch.float32)
    pred_boxes[..., :2] = 0.5
    pred_boxes[..., 2:] = 1 / 7
    pred_boxes[0, 0, 0, 1, 0] = 0.8
    pred_o = torch.zeros((1, h, w, num_anchors), dtype=torch.float32)
    pred_o[0, 0, 0, 0] = 0.5
    pred_o[0, -1, -1, 0] = 0.5
    pred_scores = torch.zeros((1, h, w, 1, num_classes), dtype=torch.float32)
    pred_scores[0, 0, 0, 0, 0] = 0.5
    pred_scores[0, 0, 0, 0, 1:] = 0.5 / (num_classes - 1)
    loss_dict = model._compute_losses(pred_boxes, pred_o, pred_scores, target, ignore_high_iou=True)
    assert loss_dict["obj_loss"].item() == model.lambda_obj * 0.5**2
    assert loss_dict["noobj_loss"].item() == model.lambda_noobj * 0.5**2
    assert loss_dict["bbox_loss"].item() == 0
    assert (
        abs(
            loss_dict["clf_loss"].item()
            - model.lambda_class * (0.5**2 + (num_classes - 1) * (0.5 / (num_classes - 1)) ** 2)
        )
        < 1e-7
    )

    # Post process
    b_coords = torch.zeros((n, h * w * num_anchors, 4), dtype=torch.float32)
    b_coords[..., :2] = 0.5
    b_coords[..., 2:] = 1 / h
    b_o = torch.zeros((n, h * w * num_anchors), dtype=torch.float32)
    b_o[:, ::2] = 0.5
    b_scores = torch.zeros((n, h * w * num_anchors, num_classes), dtype=torch.float32)
    b_scores[..., 0] = 0.5
    b_scores[..., 1:] = 0.5 / (num_classes - 1)
    dets = model.post_process(b_coords, b_o, b_scores, (h, w))
    assert dets[0]["labels"].shape[0] == b_o.shape[1] // 2
    assert torch.all(dets[0]["labels"] == 0)
    assert torch.all(dets[0]["scores"] == 0.25)
    assert torch.equal(dets[0]["boxes"][0], torch.tensor([0, 0, 1 / 7, 1 / 7]))
    assert torch.allclose(dets[0]["boxes"][-1], torch.tensor([6 / 7, 6 / 7, 1, 1]))


@torch.inference_mode()
def test_yolov2():
    input_shape = (416, 416)
    n, h, w = 2, 13, 13
    num_anchors = 5
    num_classes = 10
    model = detection.yolov2(num_classes=10, pretrained_backbone=False)

    # Forward
    t = torch.rand((n, 3, *input_shape), dtype=torch.float32)
    out = model._forward(t)
    assert out.shape == (n, num_anchors * (5 + num_classes), h, w)

    # Format outputs
    t = torch.rand((n, num_anchors * (5 + num_classes), h, w), dtype=torch.float32)
    b_coords, b_o, b_scores = model._format_outputs(t)
    assert b_coords.shape == (n, h, w, num_anchors, 4)
    assert b_o.shape == (n, h, w, num_anchors)
    assert b_scores.shape == (n, h, w, num_anchors, num_classes)
    assert torch.all(b_coords[..., :2] <= 1)
    assert torch.all(b_coords >= 0)
    assert torch.all(b_o <= 1)
    assert torch.all(b_o >= 0)
    assert torch.allclose(b_scores.sum(-1), torch.ones(1))

    # Compute loss
    target = [
        {"boxes": torch.tensor([[0, 0, 1, 1]], dtype=torch.float32), "labels": torch.zeros((1,), dtype=torch.long)}
    ]
    pred_boxes = torch.zeros((1, h, w, num_anchors, 4), dtype=torch.float32)
    pred_boxes[..., :2] = 0.5
    pred_boxes[..., 2:] = 1
    pred_boxes[0, -1, -1, 0, 0] = (w - 1) / w
    pred_boxes[0, -1, -1, 0, 1] = (h - 1) / h
    pred_boxes[0, -1, -1, 0, 2] = 1 / w
    pred_boxes[0, -1, -1, 0, 3] = 1 / h
    pred_o = torch.zeros((1, h, w, num_anchors), dtype=torch.float32)
    pred_o[0, h // 2, w // 2, 0] = 0.5
    pred_o[0, -1, -1, 0] = 0.5
    pred_scores = torch.zeros((1, h, w, 1, num_classes), dtype=torch.float32)
    pred_scores[0, h // 2, w // 2, 0, 0] = 0.5
    pred_scores[0, h // 2, w // 2, 0, 1:] = 0.5 / (num_classes - 1)
    loss_dict = model._compute_losses(pred_boxes, pred_o, pred_scores, target, ignore_high_iou=True)
    assert loss_dict["obj_loss"].item() == model.lambda_obj * 0.5**2
    assert loss_dict["noobj_loss"].item() == model.lambda_noobj * 0.5**2
    assert loss_dict["bbox_loss"].item() == 0
    assert (
        abs(
            loss_dict["clf_loss"].item()
            - model.lambda_class * (0.5**2 + (num_classes - 1) * (0.5 / (num_classes - 1)) ** 2)
        )
        < 1e-7
    )

    # Post process
    b_coords = torch.zeros((n, h * w * num_anchors, 4), dtype=torch.float32)
    b_coords[..., :2] = 0.5
    b_coords[..., 2:] = 1
    b_o = torch.zeros((n, h * w * num_anchors), dtype=torch.float32)
    b_o[:, ::2] = 0.5
    b_scores = torch.zeros((n, h * w * num_anchors, num_classes), dtype=torch.float32)
    b_scores[..., 0] = 0.5
    b_scores[..., 1:] = 0.5 / (num_classes - 1)
    dets = model.post_process(b_coords, b_o, b_scores, (h, w))
    assert dets[0]["labels"].shape[0] == 1
    assert torch.all(dets[0]["labels"] == 0)
    assert torch.all(dets[0]["scores"] == 0.25)
    assert torch.equal(dets[0]["boxes"][0], torch.tensor([0, 0, 1, 1], dtype=torch.float32))


def _yolov4_layers(num_classes=4):
    anchors = (
        torch.tensor(
            [
                [[12, 16], [19, 36], [40, 28]],
                [[36, 75], [76, 55], [72, 146]],
                [[142, 110], [192, 243], [459, 401]],
            ],
            dtype=torch.float32,
        )
        / 608
    )
    all_anchors = anchors.reshape(-1, 2)
    return [
        YoloLayer(
            anchors[idx],
            num_classes=num_classes,
            all_anchors=all_anchors,
            anchor_mask=torch.arange(3 * idx, 3 * (idx + 1)),
        )
        for idx in range(3)
    ]


def test_yolov4_global_anchor_assignment():
    layers = _yolov4_layers()
    pred_boxes = torch.zeros((1, 4, 4, 3, 4))
    b_o = torch.zeros((1, 4, 4, 3))
    b_scores = torch.zeros((1, 4, 4, 3, 4))
    boxes = torch.tensor([[0.0975, 0.0975, 0.1025, 0.1025], [0.45, 0.45, 0.55, 0.55]], dtype=torch.float32)
    target = [{"boxes": boxes, "labels": torch.tensor([1, 3])}]
    expected = [
        {(0, 0, 0, 0), (0, 2, 2, 2)},
        {(0, 2, 2, 0), (0, 2, 2, 1), (0, 2, 2, 2)},
        {(0, 2, 2, 0)},
    ]

    for layer, expected_positions in zip(layers, expected, strict=True):
        target_boxes, target_scores, obj_mask, noobj_mask = layer._build_targets(pred_boxes, b_o, b_scores, target)
        positions = {tuple(pos) for pos in obj_mask.nonzero().tolist()}
        assert positions == expected_positions
        assert torch.all(~noobj_mask[obj_mask])
        for position in positions:
            expected_idx = 0 if position[1:3] == (0, 0) else 1
            assert torch.equal(target_boxes[position], boxes[expected_idx])
            assert target_scores[position].argmax().item() == target[0]["labels"][expected_idx]
            assert target_scores[position].sum().item() == 1

    assert set(layers[0].state_dict()) == {"anchors"}


def test_yolov4_ignore_mask():
    layer = _yolov4_layers()[0]
    pred_boxes = torch.zeros((1, 2, 2, 3, 4))
    box = torch.tensor([[0.45, 0.45, 0.55, 0.55]])
    pred_boxes[0, 0, 0, 0] = box[0]
    b_o = torch.zeros((1, 2, 2, 3))
    b_scores = torch.zeros((1, 2, 2, 3, 4))

    _, _, obj_mask, noobj_mask = layer._build_targets(
        pred_boxes, b_o, b_scores, [{"boxes": box, "labels": torch.tensor([2])}]
    )

    assert not noobj_mask[0, 0, 0, 0]
    assert noobj_mask[0, 0, 0, 1]
    assert obj_mask[0, 1, 1, 2]
    assert not noobj_mask[0, 1, 1, 2]


def test_yolov4_empty_targets_and_objectness_gradients():
    layer = YoloLayer(torch.tensor([[0.2, 0.2]]), num_classes=2).train()
    empty_output = torch.zeros((1, 7, 2, 2), requires_grad=True)
    losses = layer(empty_output, [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}])
    total_loss = sum(losses.values())
    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert torch.isfinite(empty_output.grad).all()

    output = torch.zeros((1, 7, 1, 1), requires_grad=True)
    target = [{"boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]]), "labels": torch.tensor([1])}]
    layer(output, target)["obj_loss"].backward()
    assert torch.count_nonzero(output.grad[:, 4]) > 0
    assert torch.count_nonzero(output.grad[:, :4]) == 0
    assert torch.count_nonzero(output.grad[:, 5:]) == 0


@torch.inference_mode()
def test_yolo_post_process_combined_confidence_and_class_aware_nms():
    low_objectness = _post_process(
        torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        torch.tensor([[0.4]]),
        torch.tensor([[[0.9, 0.1]]]),
        box_score_thresh=0.3,
    )
    assert low_objectness[0]["scores"].item() == pytest.approx(0.36)

    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]] * 3])
    detections = _post_process(
        boxes,
        torch.tensor([[0.9, 0.8, 0.9]]),
        torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]]),
    )
    assert detections[0]["boxes"].shape[0] == 2
    assert set(detections[0]["labels"].tolist()) == {0, 1}


@torch.inference_mode()
def test_yolov4_cross_scale_nms():
    head = Yolov4Head(num_classes=2).eval()
    output_layers = [head.head1[-1], head.head2_2[-1], head.head3[-1]]
    for output_layer in output_layers:
        output_layer.weight.zero_()
        output_layer.bias.fill_(-20)

    for yolo_layer, output_layer in zip((head.yolo1, head.yolo2), output_layers[:2], strict=True):
        output_layer.bias.reshape(3, 7)[0] = torch.tensor([
            0,
            0,
            math.log(0.1 / yolo_layer.anchors[0, 0].item()),
            math.log(0.1 / yolo_layer.anchors[0, 1].item()),
            10,
            10,
            -10,
        ])

    detections = head([torch.zeros((1, 128, 1, 1)), torch.zeros((1, 256, 1, 1)), torch.zeros((1, 512, 1, 1))])
    assert detections[0]["boxes"].shape[0] == 1
