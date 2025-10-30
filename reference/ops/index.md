# holocron.ops

`holocron.ops` implements operators that are specific for Computer Vision.

Note

Those operators currently do not support TorchScript.

## Boxes

### box_giou

```python
box_giou(boxes1: Tensor, boxes2: Tensor) -> Tensor
```

Computes the Generalized-IoU as described in ["Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"](https://arxiv.org/pdf/1902.09630.pdf). This implementation was adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py

The generalized IoU is defined as follows:

[ GIoU = IoU - \\frac{|C - A \\cup B|}{|C|} ]

where (\\IoU) is the Intersection over Union, (A \\cup B) is the area of the boxes' union, and (C) is the area of the smallest enclosing box covering the two boxes.

| PARAMETER | DESCRIPTION                                       |
| --------- | ------------------------------------------------- |
| `boxes1`  | bounding boxes of shape [M, 4] **TYPE:** `Tensor` |
| `boxes2`  | bounding boxes of shape [N, 4] **TYPE:** `Tensor` |

| RETURNS  | DESCRIPTION                     |
| -------- | ------------------------------- |
| `Tensor` | Generalized-IoU of shape [M, N] |

| RAISES           | DESCRIPTION                                     |
| ---------------- | ----------------------------------------------- |
| `AssertionError` | if the boxes are in incorrect coordinate format |

Source code in `holocron/ops/boxes.py`

```python
def box_giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    r"""Computes the Generalized-IoU as described in ["Generalized Intersection over Union: A Metric and A Loss
    for Bounding Box Regression"](https://arxiv.org/pdf/1902.09630.pdf). This implementation was adapted
    from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py

    The generalized IoU is defined as follows:

    $$
    GIoU = IoU - \frac{|C - A \cup B|}{|C|}
    $$

    where $\IoU$ is the Intersection over Union,
    $A \cup B$ is the area of the boxes' union,
    and $C$ is the area of the smallest enclosing box covering the two boxes.

    Args:
        boxes1: bounding boxes of shape [M, 4]
        boxes2: bounding boxes of shape [N, 4]

    Returns:
        Generalized-IoU of shape [M, N]

    Raises:
        AssertionError: if the boxes are in incorrect coordinate format
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if torch.any(boxes1[:, 2:] < boxes1[:, :2]) or torch.any(boxes2[:, 2:] < boxes2[:, :2]):
        raise AssertionError("Incorrect coordinate format")
    iou, union = _box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
```

### diou_loss

```python
diou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor
```

Computes the Distance-IoU loss as described in ["Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"](https://arxiv.org/pdf/1911.08287.pdf).

The loss is defined as follows:

[ \\mathcal{L}\_{DIoU} = 1 - IoU + \\frac{\\rho^2(b, b^{GT})}{c^2} ]

where (\\IoU) is the Intersection over Union, (b) and (b^{GT}) are the centers of the box and the ground truth box respectively, (c) c is the diagonal length of the smallest enclosing box covering the two boxes, and (\\rho(.)) is the Euclidean distance.

| PARAMETER | DESCRIPTION                                       |
| --------- | ------------------------------------------------- |
| `boxes1`  | bounding boxes of shape [M, 4] **TYPE:** `Tensor` |
| `boxes2`  | bounding boxes of shape [N, 4] **TYPE:** `Tensor` |

| RETURNS  | DESCRIPTION                       |
| -------- | --------------------------------- |
| `Tensor` | Distance-IoU loss of shape [M, N] |

Source code in `holocron/ops/boxes.py`

```python
def diou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    r"""Computes the Distance-IoU loss as described in ["Distance-IoU Loss: Faster and Better Learning for
    Bounding Box Regression"](https://arxiv.org/pdf/1911.08287.pdf).

    The loss is defined as follows:

    $$
    \mathcal{L}_{DIoU} = 1 - IoU + \frac{\rho^2(b, b^{GT})}{c^2}
    $$

    where $\IoU$ is the Intersection over Union,
    $b$ and $b^{GT}$ are the centers of the box and the ground truth box respectively,
    $c$ c is the diagonal length of the smallest enclosing box covering the two boxes,
    and $\rho(.)$ is the Euclidean distance.

    ![Distance-IoU loss](https://github.com/frgfm/Holocron/releases/download/v0.1.3/diou_loss.png)

    Args:
        boxes1: bounding boxes of shape [M, 4]
        boxes2: bounding boxes of shape [N, 4]

    Returns:
        Distance-IoU loss of shape [M, N]
    """
    return 1 - box_iou(boxes1, boxes2) + iou_penalty(boxes1, boxes2)
```

### ciou_loss

```python
ciou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor
```

Computes the Complete IoU loss as described in ["Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"](https://arxiv.org/pdf/1911.08287.pdf).

The loss is defined as follows:

[ \\mathcal{L}\_{CIoU} = 1 - IoU + \\frac{\\rho^2(b, b^{GT})}{c^2} + \\alpha v ]

where (\\IoU) is the Intersection over Union, (b) and (b^{GT}) are the centers of the box and the ground truth box respectively, (c) c is the diagonal length of the smallest enclosing box covering the two boxes, (\\rho(.)) is the Euclidean distance, (\\alpha) is a positive trade-off parameter, and (v) is the aspect ratio consistency.

More specifically:

[ v = \\frac{4}{\\pi^2} \\Big(\\arctan{\\frac{w^{GT}}{h^{GT}}} - \\arctan{\\frac{w}{h}}\\Big)^2 ]

and

[ \\alpha = \\frac{v}{(1 - IoU) + v} ]

| PARAMETER | DESCRIPTION                                       |
| --------- | ------------------------------------------------- |
| `boxes1`  | bounding boxes of shape [M, 4] **TYPE:** `Tensor` |
| `boxes2`  | bounding boxes of shape [N, 4] **TYPE:** `Tensor` |

| RETURNS  | DESCRIPTION                       |
| -------- | --------------------------------- |
| `Tensor` | Complete IoU loss of shape [M, N] |

Example

```python
import torch
from holocron.ops.boxes import box_ciou
boxes1 = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200]], dtype=torch.float32)
boxes2 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
box_ciou(boxes1, boxes2)
```

Source code in `holocron/ops/boxes.py`

````python
def ciou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    r"""Computes the Complete IoU loss as described in ["Distance-IoU Loss: Faster and Better Learning for
    Bounding Box Regression"](https://arxiv.org/pdf/1911.08287.pdf).

    The loss is defined as follows:

    $$
    \mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{GT})}{c^2} + \alpha v
    $$

    where $\IoU$ is the Intersection over Union,
    $b$ and $b^{GT}$ are the centers of the box and the ground truth box respectively,
    $c$ c is the diagonal length of the smallest enclosing box covering the two boxes,
    $\rho(.)$ is the Euclidean distance,
    $\alpha$ is a positive trade-off parameter,
    and $v$ is the aspect ratio consistency.

    More specifically:

    $$
    v = \frac{4}{\pi^2} \Big(\arctan{\frac{w^{GT}}{h^{GT}}} - \arctan{\frac{w}{h}}\Big)^2
    $$

    and

    $$
    \alpha = \frac{v}{(1 - IoU) + v}
    $$

    Args:
        boxes1: bounding boxes of shape [M, 4]
        boxes2: bounding boxes of shape [N, 4]

    Returns:
        Complete IoU loss of shape [M, N]

    Example:
        ```python
        import torch
        from holocron.ops.boxes import box_ciou
        boxes1 = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200]], dtype=torch.float32)
        boxes2 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        box_ciou(boxes1, boxes2)
        ```
    """
    iou = box_iou(boxes1, boxes2)
    v = aspect_ratio_consistency(boxes1, boxes2)

    ciou_loss = 1 - iou + iou_penalty(boxes1, boxes2)

    # Check
    filter_ = (v != 0) & (iou != 0)
    ciou_loss[filter_].addcdiv_(v[filter_], 1 - iou[filter_] + v[filter_])

    return ciou_loss
````
