# holocron.models

The models subpackage contains definitions of models for addressing different tasks, including: image classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection and video classification.

## Classification

Classification models expect a 4D image tensor as an input (N x C x H x W) and returns a 2D output (N x K). The output represents the classification scores for each output classes.

### Supported architectures

- [ResNet](../classification/resnet/)
- [ResNeXt](../classification/resnext/)
- [Res2Net](../classification/res2net/)
- [TridentNet](../classification/tridentnet/)
- [ConvNeXt](../classification/convnext/)
- [PyConvResNet](../classification/pyconv_resnet/)
- [ReXNet](../classification/rexnet/)
- [SKNet](../classification/sknet/)
- [DarkNet](../classification/darknet/)
- [DarkNetV2](../classification/darknetv2/)
- [DarkNetV3](../classification/darknetv3/)
- [DarkNetV4](../classification/darknetv4/)
- [RepVGG](../classification/repvgg/)
- [MobileOne](../classification/mobileone/)

### Available checkpoints

Here is the list of available checkpoints:

| **Checkpoint**                          | **Acc@1** | **Acc@5** | **Params** | **Size (MB)** |
| --------------------------------------- | --------- | --------- | ---------- | ------------- |
| CSPDarknet53_Checkpoint.IMAGENETTE      | 94.50%    | 99.64%    | 26.6M      | 101.8         |
| CSPDarknet53_Mish_Checkpoint.IMAGENETTE | 94.65%    | 99.69%    | 26.6M      | 101.8         |
| ConvNeXt_Atto_Checkpoint.IMAGENETTE     | 87.59%    | 98.32%    | 3.4M       | 12.9          |
| Darknet19_Checkpoint.IMAGENETTE         | 93.86%    | 99.36%    | 19.8M      | 75.7          |
| Darknet53_Checkpoint.IMAGENETTE         | 94.17%    | 99.57%    | 40.6M      | 155.1         |
| MobileOne_S0_Checkpoint.IMAGENETTE      | 88.08%    | 98.83%    | 4.3M       | 16.9          |
| MobileOne_S1_Checkpoint.IMAGENETTE      | 91.26%    | 99.18%    | 3.6M       | 13.9          |
| MobileOne_S2_Checkpoint.IMAGENETTE      | 91.31%    | 99.21%    | 5.9M       | 22.8          |
| MobileOne_S3_Checkpoint.IMAGENETTE      | 91.06%    | 99.31%    | 8.1M       | 31.5          |
| ReXNet1_0x_Checkpoint.IMAGENET1K        | 77.86%    | 93.87%    | 4.8M       | 13.7          |
| ReXNet1_0x_Checkpoint.IMAGENETTE        | 94.39%    | 99.62%    | 3.5M       | 13.7          |
| ReXNet1_3x_Checkpoint.IMAGENET1K        | 79.50%    | 94.68%    | 7.6M       | 13.7          |
| ReXNet1_3x_Checkpoint.IMAGENETTE        | 94.88%    | 99.39%    | 5.9M       | 22.8          |
| ReXNet1_5x_Checkpoint.IMAGENET1K        | 80.31%    | 95.17%    | 9.7M       | 13.7          |
| ReXNet1_5x_Checkpoint.IMAGENETTE        | 94.47%    | 99.62%    | 7.8M       | 30.2          |
| ReXNet2_0x_Checkpoint.IMAGENET1K        | 80.31%    | 95.17%    | 16.4M      | 13.7          |
| ReXNet2_0x_Checkpoint.IMAGENETTE        | 95.24%    | 99.57%    | 13.8M      | 53.1          |
| ReXNet2_2x_Checkpoint.IMAGENETTE        | 95.44%    | 99.46%    | 16.7M      | 64.1          |
| RepVGG_A0_Checkpoint.IMAGENETTE         | 92.92%    | 99.46%    | 24.7M      | 94.6          |
| RepVGG_A1_Checkpoint.IMAGENETTE         | 93.78%    | 99.18%    | 30.1M      | 115.1         |
| RepVGG_A2_Checkpoint.IMAGENETTE         | 93.63%    | 99.39%    | 48.6M      | 185.8         |
| RepVGG_B0_Checkpoint.IMAGENETTE         | 92.69%    | 99.21%    | 31.8M      | 121.8         |
| RepVGG_B1_Checkpoint.IMAGENETTE         | 93.96%    | 99.39%    | 100.8M     | 385.1         |
| RepVGG_B2_Checkpoint.IMAGENETTE         | 94.14%    | 99.57%    | 157.5M     | 601.2         |
| Res2Net50_26w_4s_Checkpoint.IMAGENETTE  | 93.94%    | 99.41%    | 23.7M      | 90.6          |
| ResNeXt50_32x4d_Checkpoint.IMAGENETTE   | 94.55%    | 99.49%    | 23.0M      | 88.1          |
| ResNet18_Checkpoint.IMAGENETTE          | 93.61%    | 99.46%    | 11.2M      | 42.7          |
| ResNet34_Checkpoint.IMAGENETTE          | 93.81%    | 99.49%    | 21.3M      | 81.3          |
| ResNet50D_Checkpoint.IMAGENETTE         | 94.65%    | 99.52%    | 23.5M      | 90.1          |
| ResNet50_Checkpoint.IMAGENETTE          | 93.78%    | 99.54%    | 23.5M      | 90            |
| SKNet50_Checkpoint.IMAGENETTE           | 94.37%    | 99.54%    | 35.2M      | 134.7         |

## Object Detection

Object detection models expect a 4D image tensor as an input (N x C x H x W) and returns a list of dictionaries. Each dictionary has 3 keys: box coordinates, classification probability, classification label.

```python
import holocron.models as models
yolov2 = models.yolov2(num_classes=10)
```

### YOLO family

#### YOLOv1

```python
YOLOv1(layout: list[list[int]], num_classes: int = 20, in_channels: int = 3, stem_channels: int = 64, num_anchors: int = 2, lambda_obj: float = 1, lambda_noobj: float = 0.5, lambda_class: float = 1, lambda_coords: float = 5.0, rpn_nms_thresh: float = 0.7, box_score_thresh: float = 0.05, head_hidden_nodes: int = 512, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, backbone_norm_layer: Callable[[int], Module] | None = None)
```

Source code in `holocron/models/detection/yolo.py`

```python
def __init__(
    self,
    layout: list[list[int]],
    num_classes: int = 20,
    in_channels: int = 3,
    stem_channels: int = 64,
    num_anchors: int = 2,
    lambda_obj: float = 1,
    lambda_noobj: float = 0.5,
    lambda_class: float = 1,
    lambda_coords: float = 5.0,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    head_hidden_nodes: int = 512,  # In the original paper, 4096
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    backbone_norm_layer: Callable[[int], nn.Module] | None = None,
) -> None:
    super().__init__(
        num_classes, rpn_nms_thresh, box_score_thresh, lambda_obj, lambda_noobj, lambda_class, lambda_coords
    )

    if act_layer is None:
        act_layer = nn.LeakyReLU(0.1, inplace=True)

    if backbone_norm_layer is None and norm_layer is not None:
        backbone_norm_layer = norm_layer

    self.backbone = DarknetBodyV1(layout, in_channels, stem_channels, act_layer, backbone_norm_layer)

    self.block4 = nn.Sequential(
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024 * 7**2, head_hidden_nodes),
        act_layer,
        nn.Dropout(0.5),
        nn.Linear(head_hidden_nodes, 7**2 * (num_anchors * 5 + num_classes)),
    )
    self.num_anchors: int = num_anchors

    init_module(self.block4, "leaky_relu")
    init_module(self.classifier, "leaky_relu")
```

#### yolov1

```python
yolov1(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv1
```

YOLO model from ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf).

YOLO's particularity is to make predictions in a grid (same size as last feature map). For each grid cell, the model predicts classification scores and a fixed number of boxes (default: 2). Each box in the cell gets 5 predictions: an objectness score, and 4 coordinates. The 4 coordinates are composed of: the 2-D coordinates of the predicted box center (relative to the cell), and the width and height of the predicted box (relative to the whole image).

For training, YOLO uses a multi-part loss whose components are computed by:

\[ \\mathcal{L}_{coords} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits\_{j=0}^{B} \\mathbb{1}_{ij}^{obj} \\Big\[ (x_{ij} - \\hat{x}_{ij})² + (y_{ij} - \\hat{y}_{ij})² + (\\sqrt{w_{ij}} - \\sqrt{\\hat{w}_{ij}})² + (\\sqrt{h_{ij}} - \\sqrt{\\hat{h}\_{ij}})² \\Big\] \]

where: (S) is size of the output feature map (7 for an input size ((448, 448))), (B) is the number of anchor boxes per grid cell (default: 2), (\\mathbb{1}_{ij}^{obj}) equals to 1 if a GT center falls inside the i-th grid cell and among the anchor boxes of that cell, has the highest IoU with the j-th box else 0, ((x_{ij}, y\_{ij}, w\_{ij}, h\_{ij})) are the coordinates of the ground truth assigned to the j-th anchor box of the i-th grid cell, ((\\hat{x}_{ij}, \\hat{y}_{ij}, \\hat{w}_{ij}, \\hat{h}_{ij})) are the coordinate predictions for the j-th anchor box of the i-th grid cell.

\[ \\mathcal{L}_{objectness} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits\_{j=0}^{B} \\Big\[ \\mathbb{1}_{ij}^{obj} \\Big(C_{ij} - \\hat{C}\_{ij} \\Big)^2

- \\lambda\_{noobj} \\mathbb{1}_{ij}^{noobj} \\Big(C_{ij} - \\hat{C}\_{ij} \\Big)^2 \\Big\] \]

where (\\lambda\_{noobj}) is a positive coefficient (default: 0.5), (\\mathbb{1}_{ij}^{noobj} = 1 - \\mathbb{1}_{ij}^{obj}), (C\_{ij}) equals the Intersection Over Union between the j-th anchor box in the i-th grid cell and its matched ground truth box if that box is matched with a ground truth else 0, and (\\hat{C}\_{ij}) is the objectness score of the j-th anchor box in the i-th grid cell..

\[ \\mathcal{L}_{classification} = \\sum\\limits_{i=0}^{S^2} \\mathbb{1}_{i}^{obj} \\sum\\limits_{c \\in classes} (p_i(c) - \\hat{p}\_i(c))^2 \]

where (\\mathbb{1}\_{i}^{obj}) equals to 1 if a GT center falls inside the i-th grid cell else 0, (p_i(c)) equals 1 if the assigned ground truth to the i-th cell is classified as class (c), and (\\hat{p}\_i(c)) is the predicted probability of class (c) in the i-th cell.

And the full loss is given by:

\[ \\mathcal{L}_{YOLOv1} = \\lambda_{coords} \\cdot \\mathcal{L}_{coords} + \\mathcal{L}_{objectness} + \\mathcal{L}\_{classification} \]

where (\\lambda\_{coords}) is a positive coefficient (default: 5).

| PARAMETER             | DESCRIPTION                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on ImageNet **TYPE:** `bool` **DEFAULT:** `False`                    |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`           |
| `pretrained_backbone` | If True, backbone parameters will have been pretrained on Imagenette **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`              | keyword args of YOLOv1 **TYPE:** `Any` **DEFAULT:** `{}`                                                  |

| RETURNS  | DESCRIPTION      |
| -------- | ---------------- |
| `YOLOv1` | detection module |

Source code in `holocron/models/detection/yolo.py`

```python
def yolov1(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv1:
    r"""YOLO model from
    ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf).

    YOLO's particularity is to make predictions in a grid (same size as last feature map). For each grid cell,
    the model predicts classification scores and a fixed number of boxes (default: 2). Each box in the cell gets
    5 predictions: an objectness score, and 4 coordinates. The 4 coordinates are composed of: the 2-D coordinates of
    the predicted box center (relative to the cell), and the width and height of the predicted box (relative to
    the whole image).

    For training, YOLO uses a multi-part loss whose components are computed by:

    $$
    \mathcal{L}_{coords} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
    \mathbb{1}_{ij}^{obj} \Big[
    (x_{ij} - \hat{x}_{ij})² + (y_{ij} - \hat{y}_{ij})² +
    (\sqrt{w_{ij}} - \sqrt{\hat{w}_{ij}})² + (\sqrt{h_{ij}} - \sqrt{\hat{h}_{ij}})²
    \Big]
    $$

    where:
    $S$ is size of the output feature map (7 for an input size $(448, 448)$),
    $B$ is the number of anchor boxes per grid cell (default: 2),
    $\mathbb{1}_{ij}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    $(x_{ij}, y_{ij}, w_{ij}, h_{ij})$ are the coordinates of the ground truth assigned to
    the j-th anchor box of the i-th grid cell,
    $(\hat{x}_{ij}, \hat{y}_{ij}, \hat{w}_{ij}, \hat{h}_{ij})$ are the coordinate predictions
    for the j-th anchor box of the i-th grid cell.

    $$
    \mathcal{L}_{objectness} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
    \Big[ \mathbb{1}_{ij}^{obj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
    + \lambda_{noobj} \mathbb{1}_{ij}^{noobj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
    \Big]
    $$

    where $\lambda_{noobj}$ is a positive coefficient (default: 0.5),
    $\mathbb{1}_{ij}^{noobj} = 1 - \mathbb{1}_{ij}^{obj}$,
    $C_{ij}$ equals the Intersection Over Union between the j-th anchor box in the i-th grid cell and its
    matched ground truth box if that box is matched with a ground truth else 0,
    and $\hat{C}_{ij}$ is the objectness score of the j-th anchor box in the i-th grid cell..

    $$
    \mathcal{L}_{classification} = \sum\limits_{i=0}^{S^2}
    \mathbb{1}_{i}^{obj} \sum\limits_{c \in classes}
    (p_i(c) - \hat{p}_i(c))^2
    $$

    where $\mathbb{1}_{i}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell else 0,
    $p_i(c)$ equals 1 if the assigned ground truth to the i-th cell is classified as class $c$,
    and $\hat{p}_i(c)$ is the predicted probability of class $c$ in the i-th cell.

    And the full loss is given by:

    $$
    \mathcal{L}_{YOLOv1} = \lambda_{coords} \cdot \mathcal{L}_{coords} +
    \mathcal{L}_{objectness} + \mathcal{L}_{classification}
    $$

    where $\lambda_{coords}$ is a positive coefficient (default: 5).

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv1`][holocron.models.detection.yolo.YOLOv1]

    Returns:
        detection module
    """
    return _yolo(
        "yolov1",
        pretrained,
        progress,
        pretrained_backbone,
        [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
        **kwargs,
    )
```

#### YOLOv2

```python
YOLOv2(layout: list[tuple[int, int]], num_classes: int = 20, in_channels: int = 3, stem_chanels: int = 32, anchors: Tensor | None = None, passthrough_ratio: int = 8, lambda_obj: float = 1, lambda_noobj: float = 0.5, lambda_class: float = 1, lambda_coords: float = 5, rpn_nms_thresh: float = 0.7, box_score_thresh: float = 0.05, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, backbone_norm_layer: Callable[[int], Module] | None = None)
```

Source code in `holocron/models/detection/yolov2.py`

```python
def __init__(
    self,
    layout: list[tuple[int, int]],
    num_classes: int = 20,
    in_channels: int = 3,
    stem_chanels: int = 32,
    anchors: Tensor | None = None,
    passthrough_ratio: int = 8,
    lambda_obj: float = 1,
    lambda_noobj: float = 0.5,
    lambda_class: float = 1,
    lambda_coords: float = 5,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    backbone_norm_layer: Callable[[int], nn.Module] | None = None,
) -> None:
    super().__init__(
        num_classes, rpn_nms_thresh, box_score_thresh, lambda_obj, lambda_noobj, lambda_class, lambda_coords
    )

    if act_layer is None:
        act_layer = nn.LeakyReLU(0.1, inplace=True)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    if backbone_norm_layer is None:
        backbone_norm_layer = norm_layer

    # Priors computed using K-means
    if anchors is None:
        # cf. https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg#L242
        anchors = (
            torch.tensor([
                [1.3221, 1.73145],
                [3.19275, 4.00944],
                [5.05587, 8.09892],
                [9.47112, 4.84053],
                [11.2364, 10.0071],
            ])
            / 13
        )

    self.backbone = DarknetBodyV2(
        layout, in_channels, stem_chanels, True, act_layer, backbone_norm_layer, drop_layer, conv_layer
    )

    self.block5 = nn.Sequential(
        *conv_sequence(
            layout[-1][0],
            layout[-1][0],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            layout[-1][0],
            layout[-1][0],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
    )

    self.passthrough_layer = nn.Sequential(
        *conv_sequence(
            layout[-2][0],
            layout[-2][0] // passthrough_ratio,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=1,
            bias=(norm_layer is None),
        ),
        ConcatDownsample2d(scale_factor=2),
    )

    self.block6 = nn.Sequential(
        *conv_sequence(
            layout[-1][0] + layout[-2][0] // passthrough_ratio * 2**2,
            layout[-1][0],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        )
    )

    # Each box has P_objectness, 4 coords, and score for each class
    self.head = nn.Conv2d(layout[-1][0], anchors.shape[0] * (5 + num_classes), 1)

    # Register losses
    self.register_buffer("anchors", anchors)

    init_module(self.block5, "leaky_relu")
    init_module(self.passthrough_layer, "leaky_relu")
    init_module(self.block6, "leaky_relu")
    # Initialize the head like a linear (default Conv2D init is the same as Linear)
    if self.head.bias is not None:
        self.head.bias.data.zero_()
```

#### yolov2

```python
yolov2(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv2
```

YOLOv2 model from ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

YOLOv2 improves upon YOLO by raising the number of boxes predicted by grid cell (default: 5), introducing bounding box priors and predicting class scores for each anchor box in the grid cell.

For training, YOLOv2 uses the same multi-part loss as YOLO apart from its classification loss:

\[ \\mathcal{L}_{classification} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits\_{j=0}^{B} \\mathbb{1}_{ij}^{obj} \\sum\\limits_{c \\in classes} (p\_{ij}(c) - \\hat{p}\_{ij}(c))^2 \]

where:

- (S) is size of the output feature map (13 for an input size ((416, 416))),
- (B) is the number of anchor boxes per grid cell (default: 5),
- (\\mathbb{1}\_{ij}^{obj}) equals to 1 if a GT center falls inside the i-th grid cell and among the anchor boxes of that cell, has the highest IoU with the j-th box else 0,
- (p\_{ij}(c)) equals 1 if the assigned ground truth to the j-th anchor box of the i-th cell is classified as class (c),
- (\\hat{p}\_{ij}(c)) is the predicted probability of class (c) for the j-th anchor box in the i-th cell.

| PARAMETER             | DESCRIPTION                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on ImageNet **TYPE:** `bool` **DEFAULT:** `False`                    |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`           |
| `pretrained_backbone` | If True, backbone parameters will have been pretrained on Imagenette **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`              | keyword args of YOLOv2 **TYPE:** `Any` **DEFAULT:** `{}`                                                  |

| RETURNS  | DESCRIPTION      |
| -------- | ---------------- |
| `YOLOv2` | detection module |

Source code in `holocron/models/detection/yolov2.py`

```python
def yolov2(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv2:
    r"""YOLOv2 model from
    ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

    YOLOv2 improves upon YOLO by raising the number of boxes predicted by grid cell (default: 5), introducing
    bounding box priors and predicting class scores for each anchor box in the grid cell.

    For training, YOLOv2 uses the same multi-part loss as YOLO apart from its classification loss:

    $$
    \mathcal{L}_{classification} = \sum\limits_{i=0}^{S^2}  \sum\limits_{j=0}^{B}
    \mathbb{1}_{ij}^{obj} \sum\limits_{c \in classes}
    (p_{ij}(c) - \hat{p}_{ij}(c))^2
    $$

    where:
    - $S$ is size of the output feature map (13 for an input size $(416, 416)$),
    - $B$ is the number of anchor boxes per grid cell (default: 5),
    - $\mathbb{1}_{ij}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    - $p_{ij}(c)$ equals 1 if the assigned ground truth to the j-th anchor box of the i-th cell is classified
    as class $c$,
    - $\hat{p}_{ij}(c)$ is the predicted probability of class $c$ for the j-th anchor box
    in the i-th cell.

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv2`][holocron.models.detection.yolov2.YOLOv2]

    Returns:
        detection module
    """
    if pretrained_backbone:
        kwargs["backbone_norm_layer"] = FrozenBatchNorm2d

    return _yolo(
        "yolov2",
        pretrained,
        progress,
        pretrained_backbone,
        [(64, 0), (128, 1), (256, 1), (512, 2), (1024, 2)],
        **kwargs,
    )
```

#### YOLOv4

```python
YOLOv4(layout: list[tuple[int, int]], num_classes: int = 80, in_channels: int = 3, stem_channels: int = 32, anchors: Tensor | None = None, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, backbone_norm_layer: Callable[[int], Module] | None = None)
```

Source code in `holocron/models/detection/yolov4.py`

```python
def __init__(
    self,
    layout: list[tuple[int, int]],
    num_classes: int = 80,
    in_channels: int = 3,
    stem_channels: int = 32,
    anchors: Tensor | None = None,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    backbone_norm_layer: Callable[[int], nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.Mish(inplace=True)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    if backbone_norm_layer is None:
        backbone_norm_layer = norm_layer
    if drop_layer is None:
        drop_layer = DropBlock2d

    # backbone
    self.backbone = DarknetBodyV4(
        layout, in_channels, stem_channels, 3, act_layer, backbone_norm_layer, drop_layer, conv_layer
    )
    # neck
    self.neck = Neck([1024, 512, 256], act_layer, norm_layer, drop_layer, conv_layer)
    # head
    self.head = Yolov4Head(num_classes, anchors, act_layer, norm_layer, drop_layer, conv_layer)

    init_module(self.neck, "leaky_relu")
    init_module(self.head, "leaky_relu")
```

#### yolov4

```python
yolov4(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv4
```

YOLOv4 model from ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf).

The architecture improves upon YOLOv3 by including: the usage of [DropBlock](https://arxiv.org/pdf/1810.12890.pdf) regularization, [Mish](https://arxiv.org/pdf/1908.08681.pdf) activation, [CSP](https://arxiv.org/pdf/2004.10934.pdf) and [SAM](https://arxiv.org/pdf/1807.06521.pdf) in the backbone, [SPP](https://arxiv.org/pdf/1406.4729.pdf) and [PAN](https://arxiv.org/pdf/1803.01534.pdf) in the neck.

For training, YOLOv4 uses the same multi-part loss as YOLOv3 apart from its box coordinate loss:

\[ \\mathcal{L}_{coords} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits\_{j=0}^{B} \\min\\limits\_{k \\in [1, M]} C\_{IoU}(\\hat{loc}\_{ij}, loc^{GT}\_k) \]

where:

- (S) is size of the output feature map (13 for an input size ((416, 416))),
- (B) is the number of anchor boxes per grid cell (default: 3),
- (M) is the number of ground truth boxes,
- (C\_{IoU}) is the complete IoU loss,
- (\\hat{loc}\_{ij}) is the predicted bounding box for grid cell (i) at anchor (j), and (loc^{GT}\_k) is the k-th ground truth bounding box.

| PARAMETER             | DESCRIPTION                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on ImageNet **TYPE:** `bool` **DEFAULT:** `False`                    |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`           |
| `pretrained_backbone` | If True, backbone parameters will have been pretrained on Imagenette **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`              | keyword args of YOLOv4 **TYPE:** `Any` **DEFAULT:** `{}`                                                  |

| RETURNS  | DESCRIPTION      |
| -------- | ---------------- |
| `YOLOv4` | detection module |

Source code in `holocron/models/detection/yolov4.py`

```python
def yolov4(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv4:
    r"""YOLOv4 model from
    ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf).

    The architecture improves upon YOLOv3 by including: the usage of [DropBlock](https://arxiv.org/pdf/1810.12890.pdf) regularization, [Mish](https://arxiv.org/pdf/1908.08681.pdf) activation, [CSP](https://arxiv.org/pdf/2004.10934.pdf) and [SAM](https://arxiv.org/pdf/1807.06521.pdf) in the
    backbone, [SPP](https://arxiv.org/pdf/1406.4729.pdf) and [PAN](https://arxiv.org/pdf/1803.01534.pdf) in the
    neck.

    For training, YOLOv4 uses the same multi-part loss as YOLOv3 apart from its box coordinate loss:

    $$
        \mathcal{L}_{coords} = \sum\limits_{i=0}^{S^2}  \sum\limits_{j=0}^{B}
        \min\limits_{k \in [1, M]} C_{IoU}(\hat{loc}_{ij}, loc^{GT}_k)
    $$

    where:
    - $S$ is size of the output feature map (13 for an input size $(416, 416)$),
    - $B$ is the number of anchor boxes per grid cell (default: 3),
    - $M$ is the number of ground truth boxes,
    - $C_{IoU}$ is the complete IoU loss,
    - $\hat{loc}_{ij}$ is the predicted bounding box for grid cell $i$ at anchor $j$,
    and $loc^{GT}_k$ is the k-th ground truth bounding box.

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv4`][holocron.models.detection.yolov4.YOLOv4]

    Returns:
        detection module
    """
    if pretrained_backbone:
        kwargs["backbone_norm_layer"] = FrozenBatchNorm2d

    return _yolo(
        "yolov4",
        pretrained,
        progress,
        pretrained_backbone,
        [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)],
        **kwargs,
    )
```

## Semantic Segmentation

Semantic segmentation models expect a 4D image tensor as an input (N x C x H x W) and returns a classification score tensor of size (N x K x Ho x Wo).

```python
import holocron.models as models
unet = models.unet(num_classes=10)
```

### U-Net family

#### UNet

```python
UNet(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, same_padding: bool = True, bilinear_upsampling: bool = True)
```

Implements a U-Net architecture

| PARAMETER             | DESCRIPTION                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| `layout`              | number of channels after each contracting block **TYPE:** `list[int]`                                  |
| `in_channels`         | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3`                                |
| `num_classes`         | number of output classes **TYPE:** `int` **DEFAULT:** `10`                                             |
| `act_layer`           | activation layer **TYPE:** \`Module                                                                    |
| `norm_layer`          | normalization layer **TYPE:** \`Callable\[[int], Module\]                                              |
| `drop_layer`          | dropout layer **TYPE:** \`Callable[..., Module]                                                        |
| `conv_layer`          | convolutional layer **TYPE:** \`Callable[..., Module]                                                  |
| `same_padding`        | enforces same padding in convolutions **TYPE:** `bool` **DEFAULT:** `True`                             |
| `bilinear_upsampling` | replaces transposed conv by bilinear interpolation for upsampling **TYPE:** `bool` **DEFAULT:** `True` |

Source code in `holocron/models/segmentation/unet.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    same_padding: bool = True,
    bilinear_upsampling: bool = True,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    # Contracting path
    self.encoder = nn.ModuleList([])
    layout_ = [in_channels, *layout]
    pool = False
    for in_chan, out_chan in pairwise(layout_):
        self.encoder.append(
            down_path(in_chan, out_chan, pool, int(same_padding), act_layer, norm_layer, drop_layer, conv_layer)
        )
        pool = True

    self.bridge = nn.Sequential(
        nn.MaxPool2d((2, 2)),
        *conv_sequence(
            layout[-1], 2 * layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
        *conv_sequence(
            2 * layout[-1], layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout_ = [chan // 2 if bilinear_upsampling else chan for chan in layout[::-1][:-1]] + [layout[0]]
    for in_chan, out_chan in zip([2 * layout[-1], *layout[::-1][:-1]], layout_, strict=True):
        self.decoder.append(
            UpPath(
                in_chan,
                out_chan,
                bilinear_upsampling,
                int(same_padding),
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
            )
        )

    # Classifier
    self.classifier = nn.Conv2d(layout[0], num_classes, 1)

    init_module(self, "relu")
```

#### unet

```python
unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNet **TYPE:** `Any` **DEFAULT:** `{}`                                          |

| RETURNS | DESCRIPTION                 |
| ------- | --------------------------- |
| `UNet`  | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)

    ![UNet explanation](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNet`][holocron.models.segmentation.unet.UNet]

    Returns:
        semantic segmentation model
    """
    return _unet("unet", pretrained, progress, **kwargs)
```

#### DynamicUNet

```python
DynamicUNet(encoder: IntermediateLayerGetter, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, same_padding: bool = True, input_shape: tuple[int, int, int] | None = None, final_upsampling: bool = False)
```

Implements a dymanic U-Net architecture

| PARAMETER          | DESCRIPTION                                                                                                      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `encoder`          | feature extractor used for encoding **TYPE:** `IntermediateLayerGetter`                                          |
| `num_classes`      | number of output classes **TYPE:** `int` **DEFAULT:** `10`                                                       |
| `act_layer`        | activation layer **TYPE:** \`Module                                                                              |
| `norm_layer`       | normalization layer **TYPE:** \`Callable\[[int], Module\]                                                        |
| `drop_layer`       | dropout layer **TYPE:** \`Callable[..., Module]                                                                  |
| `conv_layer`       | convolutional layer **TYPE:** \`Callable[..., Module]                                                            |
| `same_padding`     | enforces same padding in convolutions **TYPE:** `bool` **DEFAULT:** `True`                                       |
| `input_shape`      | shape of the input tensor **TYPE:** \`tuple[int, int, int]                                                       |
| `final_upsampling` | if True, replaces transposed conv by bilinear interpolation for upsampling **TYPE:** `bool` **DEFAULT:** `False` |

Source code in `holocron/models/segmentation/unet.py`

```python
def __init__(
    self,
    encoder: IntermediateLayerGetter,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    same_padding: bool = True,
    input_shape: tuple[int, int, int] | None = None,
    final_upsampling: bool = False,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    self.encoder = encoder
    # Determine all feature map shapes
    training_mode = self.encoder.training
    self.encoder.eval()
    input_shape = (3, 256, 256) if input_shape is None else input_shape
    with torch.no_grad():
        shapes = [v.shape[1:] for v in self.encoder(torch.zeros(1, *input_shape)).values()]
    chans = [s[0] for s in shapes]
    if training_mode:
        self.encoder.train()

    # Middle layers
    self.bridge = nn.Sequential(
        nn.BatchNorm2d(chans[-1]) if norm_layer is None else norm_layer(chans[-1]),
        act_layer,
        *conv_sequence(
            chans[-1], 2 * chans[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
        *conv_sequence(
            2 * chans[-1], chans[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout = [*chans[::-1][1:], chans[0]]
    for up_chan, out_chan in zip(chans[::-1], layout, strict=True):
        self.decoder.append(
            UBlock(up_chan, up_chan, out_chan, int(same_padding), act_layer, norm_layer, drop_layer, conv_layer)
        )

    # Final upsampling if sizes don't match
    self.upsample: nn.Sequential | None = None
    if final_upsampling:
        self.upsample = nn.Sequential(
            *conv_sequence(chans[0], chans[0] * 2**2, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
        )

    # Classifier
    self.classifier = nn.Conv2d(chans[0], num_classes, 1)

    init_module(self, "relu")
```

#### unet2

```python
unet2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet
```

Modified version of U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) that includes a more advanced upscaling block inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

| PARAMETER     | DESCRIPTION                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------- |
| `pretrained`  | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`    | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `in_channels` | number of input channels **TYPE:** `int` **DEFAULT:** `3`                                       |
| `kwargs`      | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                   |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet:
    """Modified version of U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    that includes a more advanced upscaling block inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    ![UNet architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        in_channels: number of input channels
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    backbone = UNetBackbone(default_cfgs["unet2"]["encoder_layout"], in_channels=in_channels).features

    return _dynamic_unet("unet2", backbone, pretrained, progress, **kwargs)  # ty: ignore[invalid-argument-type]
```

#### unet_tvvgg11

```python
unet_tvvgg11(pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any) -> DynamicUNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) with a VGG-11 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

| PARAMETER             | DESCRIPTION                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`            |
| `pretrained_backbone` | If True, the encoder will load pretrained parameters from ImageNet **TYPE:** `bool` **DEFAULT:** `True` |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`         |
| `kwargs`              | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                           |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet_tvvgg11(
    pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a VGG-11 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    weights = get_model_weights("vgg11").DEFAULT if pretrained_backbone and not pretrained else None  # ty: ignore[unresolved-attribute]
    backbone: nn.Module = get_model("vgg11", weights=weights).features  # ty: ignore[invalid-assignment]

    return _dynamic_unet("unet_vgg11", backbone, pretrained, progress, **kwargs)
```

#### unet_tvresnet34

```python
unet_tvresnet34(pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any) -> DynamicUNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) with a ResNet-34 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

| PARAMETER             | DESCRIPTION                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`            |
| `pretrained_backbone` | If True, the encoder will load pretrained parameters from ImageNet **TYPE:** `bool` **DEFAULT:** `True` |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`         |
| `kwargs`              | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                           |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet_tvresnet34(
    pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a ResNet-34 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    weights = get_model_weights("resnet34").DEFAULT if pretrained_backbone and not pretrained else None  # ty: ignore[unresolved-attribute]
    backbone = get_model("resnet34", weights=weights)
    kwargs["final_upsampling"] = kwargs.get("final_upsampling", True)

    return _dynamic_unet("unet_tvresnet34", backbone, pretrained, progress, **kwargs)
```

#### unet_rexnet13

```python
unet_rexnet13(pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) with a ReXNet-1.3x backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet).

| PARAMETER             | DESCRIPTION                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`            |
| `pretrained_backbone` | If True, the encoder will load pretrained parameters from ImageNet **TYPE:** `bool` **DEFAULT:** `True` |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`         |
| `in_channels`         | the number of input channels **TYPE:** `int` **DEFAULT:** `3`                                           |
| `kwargs`              | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                           |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet_rexnet13(
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    progress: bool = True,
    in_channels: int = 3,
    **kwargs: Any,
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a ReXNet-1.3x backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet).

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        in_channels: the number of input channels
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    backbone = rexnet1_3x(pretrained=pretrained_backbone and not pretrained, in_channels=in_channels).features
    kwargs["final_upsampling"] = kwargs.get("final_upsampling", True)
    kwargs["act_layer"] = kwargs.get("act_layer", nn.SiLU(inplace=True))
    # hotfix of https://github.com/pytorch/vision/issues/3802
    backbone[21] = nn.SiLU(inplace=True)  # ty: ignore[possibly-missing-implicit-call]

    return _dynamic_unet("unet_rexnet13", backbone, pretrained, progress, **kwargs)  # ty: ignore[invalid-argument-type]
```

#### UNetp

```python
UNetp(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None)
```

Implements a UNet+ architecture

| PARAMETER     | DESCRIPTION                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `layout`      | number of channels after each contracting block **TYPE:** `list[int]`   |
| `in_channels` | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3` |
| `num_classes` | number of output classes **TYPE:** `int` **DEFAULT:** `10`              |
| `act_layer`   | activation layer **TYPE:** \`Module                                     |
| `norm_layer`  | normalization layer **TYPE:** \`Callable\[[int], Module\]               |
| `drop_layer`  | dropout layer **TYPE:** \`Callable[..., Module]                         |
| `conv_layer`  | convolutional layer **TYPE:** \`Callable[..., Module]                   |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    # Contracting path
    self.encoder = nn.ModuleList([])
    layout_ = [in_channels, *layout]
    pool = False
    for in_chan, out_chan in pairwise(layout_):
        self.encoder.append(down_path(in_chan, out_chan, pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
        pool = True

    self.bridge = nn.Sequential(
        nn.MaxPool2d((2, 2)),
        *conv_sequence(
            layout[-1], 2 * layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
        *conv_sequence(
            2 * layout[-1], layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout_ = [layout[-1], *layout[1:][::-1]]
    for left_chan, up_chan, num_cells in zip(layout[::-1], layout_, range(1, len(layout) + 1), strict=True):
        self.decoder.append(
            nn.ModuleList([
                UpPath(left_chan + up_chan, left_chan, True, 1, act_layer, norm_layer, drop_layer, conv_layer)
                for _ in range(num_cells)
            ])
        )

    # Classifier
    self.classifier = nn.Conv2d(layout[0], num_classes, 1)

    init_module(self, "relu")
```

#### unetp

```python
unetp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetp
```

UNet+ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNetp **TYPE:** `Any` **DEFAULT:** `{}`                                         |

| RETURNS | DESCRIPTION                 |
| ------- | --------------------------- |
| `UNetp` | semantic segmentation model |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def unetp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetp:
    """UNet+ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

    ![UNet+ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetp.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNetp`][holocron.models.segmentation.unetpp.UNetp]

    Returns:
        semantic segmentation model
    """
    return _unet("unetp", pretrained, progress, **kwargs)  # type: ignore[return-value]
```

#### UNetpp

```python
UNetpp(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None)
```

Implements a UNet++ architecture

| PARAMETER     | DESCRIPTION                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `layout`      | number of channels after each contracting block **TYPE:** `list[int]`   |
| `in_channels` | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3` |
| `num_classes` | number of output classes **TYPE:** `int` **DEFAULT:** `10`              |
| `act_layer`   | activation layer **TYPE:** \`Module                                     |
| `norm_layer`  | normalization layer **TYPE:** \`Callable\[[int], Module\]               |
| `drop_layer`  | dropout layer **TYPE:** \`Callable[..., Module]                         |
| `conv_layer`  | convolutional layer **TYPE:** \`Callable[..., Module]                   |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    # Contracting path
    self.encoder = nn.ModuleList([])
    layout_ = [in_channels, *layout]
    pool = False
    for in_chan, out_chan in pairwise(layout_):
        self.encoder.append(down_path(in_chan, out_chan, pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
        pool = True

    self.bridge = nn.Sequential(
        nn.MaxPool2d((2, 2)),
        *conv_sequence(
            layout[-1], 2 * layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
        *conv_sequence(
            2 * layout[-1], layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout_ = [layout[-1], *layout[1:][::-1]]
    for left_chan, up_chan, num_cells in zip(layout[::-1], layout_, range(1, len(layout) + 1), strict=True):
        self.decoder.append(
            nn.ModuleList([
                UpPath(
                    up_chan + (idx + 1) * left_chan,
                    left_chan,
                    True,
                    1,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                )
                for idx in range(num_cells)
            ])
        )

    # Classifier
    self.classifier = nn.Conv2d(layout[0], num_classes, 1)

    init_module(self, "relu")
```

#### unetpp

```python
unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp
```

UNet++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNetpp **TYPE:** `Any` **DEFAULT:** `{}`                                        |

| RETURNS  | DESCRIPTION                 |
| -------- | --------------------------- |
| `UNetpp` | semantic segmentation model |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp:
    """UNet++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

    ![UNet++ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetpp.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNetpp`][holocron.models.segmentation.unetpp.UNetpp]

    Returns:
        semantic segmentation model
    """
    return _unet("unetpp", pretrained, progress, **kwargs)  # type: ignore[return-value]
```

#### UNet3p

```python
UNet3p(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None)
```

Implements a UNet3+ architecture

| PARAMETER     | DESCRIPTION                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `layout`      | number of channels after each contracting block **TYPE:** `list[int]`   |
| `in_channels` | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3` |
| `num_classes` | number of output classes **TYPE:** `int` **DEFAULT:** `10`              |
| `act_layer`   | activation layer **TYPE:** \`Module                                     |
| `norm_layer`  | normalization layer **TYPE:** \`Callable\[[int], Module\]               |
| `drop_layer`  | dropout layer **TYPE:** \`Callable[..., Module]                         |
| `conv_layer`  | convolutional layer **TYPE:** \`Callable[..., Module]                   |

Source code in `holocron/models/segmentation/unet3p.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    # Contracting path
    self.encoder = nn.ModuleList([])
    layout_ = [in_channels, *layout]
    pool = False
    for in_chan, out_chan in pairwise(layout_):
        self.encoder.append(down_path(in_chan, out_chan, pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
        pool = True

    # Expansive path
    self.decoder = nn.ModuleList([])
    for row in range(len(layout) - 1):
        self.decoder.append(
            FSAggreg(
                layout[:row],
                layout[row],
                [len(layout) * layout[0]] * (len(layout) - 2 - row) + layout[-1:],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
            )
        )

    # Classifier
    self.classifier = nn.Conv2d(len(layout) * layout[0], num_classes, 1)

    init_module(self, "relu")
```

#### unet3p

```python
unet3p(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet3p
```

UNet3+ from ["UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation"](https://arxiv.org/pdf/2004.08790.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNet3p **TYPE:** `Any` **DEFAULT:** `{}`                                        |

| RETURNS  | DESCRIPTION                 |
| -------- | --------------------------- |
| `UNet3p` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet3p.py`

```python
def unet3p(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet3p:
    """UNet3+ from
    ["UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation"](https://arxiv.org/pdf/2004.08790.pdf)

    ![UNet 3+ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet3p.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNet3p`][holocron.models.segmentation.unet3p.UNet3p]

    Returns:
        semantic segmentation model
    """
    return _unet("unet3p", pretrained, progress, **kwargs)  # type: ignore[return-value]
```
