# holocron.models

The models subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection and video classification.


.. currentmodule:: holocron.models

## Classification

Classification models expect a 4D image tensor as an input (N x C x H x W) and returns a 2D output (N x K).
The output represents the classification scores for each output classes.

```python
import holocron.models as models
darknet19 = models.darknet19(num_classes=10)
```

### Supported architectures
* [ResNet](./classification/resnet.md)
* [ResNeXt](./classification/resnext.md)
* [Res2Net](./classification/res2net.md)
* [TridentNet](./classification/tridentnet.md)
* [ConvNeXt](./classification/convnext.md)
* [PyConvResNet](./classification/pyconv_resnet.md)
* [ReXNet](./classification/rexnet.md)
* [SKNet](./classification/sknet.md)
* [DarkNet](./classification/darknet.md)
* [DarkNetV2](./classification/darknetv2.md)
* [DarkNetV3](./classification/darknetv3.md)
* [DarkNetV4](./classification/darknetv4.md)
* [RepVGG](./classification/repvgg.md)
* [MobileOne](./classification/mobileone.md)

### Available checkpoints

Here is the list of available checkpoints:

.. include:: generated/classification_table.rst


## Object Detection

Object detection models expect a 4D image tensor as an input (N x C x H x W) and returns a list of dictionaries.
Each dictionary has 3 keys: box coordinates, classification probability, classification label.

```python
import holocron.models as models
yolov2 = models.yolov2(num_classes=10)
```

.. currentmodule:: holocron.models.detection

### YOLO

::: holocron.models.detection.yolo.yolov1
    options:
        heading_level: 4

::: holocron.models.detection.yolov2.yolov2
    options:
        heading_level: 4

::: holocron.models.detection.yolov4.yolov4
    options:
        heading_level: 4


## Semantic Segmentation

Semantic segmentation models expect a 4D image tensor as an input (N x C x H x W) and returns a classification score
tensor of size (N x K x Ho x Wo).

```python
import holocron.models as models
unet = models.unet(num_classes=10)
```

### U-Net

::: holocron.models.segmentation.unet.unet
    options:
        heading_level: 4

::: holocron.models.segmentation.unetpp.unetp
    options:
        heading_level: 4

::: holocron.models.segmentation.unetpp.unetpp
    options:
        heading_level: 4

::: holocron.models.segmentation.unet3p.unet3p
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.unet2
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.unet_tvvgg11
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.unet_tvresnet34
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.unet_rexnet13
    options:
        heading_level: 4
