# holocron.models

The models subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection and video classification.


## Classification

Classification models expect a 4D image tensor as an input (N x C x H x W) and returns a 2D output (N x K).
The output represents the classification scores for each output classes.

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

| **Checkpoint** | **Acc@1** | **Acc@5** | **Params** | **Size (MB)** |
|---|---|---|---|---|
| [`CSPDarknet53_Checkpoint.IMAGENETTE`][holocron.models.classification.CSPDarknet53_Checkpoint.IMAGENETTE] | 94.50% | 99.64% | 26.6M | 101.8 |
| [`CSPDarknet53_Mish_Checkpoint.IMAGENETTE`][holocron.models.classification.CSPDarknet53_Mish_Checkpoint.IMAGENETTE] | 94.65% | 99.69% | 26.6M | 101.8 |
| [`ConvNeXt_Atto_Checkpoint.IMAGENETTE`][holocron.models.classification.ConvNeXt_Atto_Checkpoint.IMAGENETTE] | 87.59% | 98.32% | 3.4M | 12.9 |
| [`Darknet19_Checkpoint.IMAGENETTE`][holocron.models.classification.Darknet19_Checkpoint.IMAGENETTE] | 93.86% | 99.36% | 19.8M | 75.7 |
| [`Darknet53_Checkpoint.IMAGENETTE`][holocron.models.classification.Darknet53_Checkpoint.IMAGENETTE] | 94.17% | 99.57% | 40.6M | 155.1 |
| [`MobileOne_S0_Checkpoint.IMAGENETTE`][holocron.models.classification.MobileOne_S0_Checkpoint.IMAGENETTE] | 88.08% | 98.83% | 4.3M | 16.9 |
| [`MobileOne_S1_Checkpoint.IMAGENETTE`][holocron.models.classification.MobileOne_S1_Checkpoint.IMAGENETTE] | 91.26% | 99.18% | 3.6M | 13.9 |
| [`MobileOne_S2_Checkpoint.IMAGENETTE`][holocron.models.classification.MobileOne_S2_Checkpoint.IMAGENETTE] | 91.31% | 99.21% | 5.9M | 22.8 |
| [`MobileOne_S3_Checkpoint.IMAGENETTE`][holocron.models.classification.MobileOne_S3_Checkpoint.IMAGENETTE] | 91.06% | 99.31% | 8.1M | 31.5 |
| [`ReXNet1_0x_Checkpoint.IMAGENET1K`][holocron.models.classification.ReXNet1_0x_Checkpoint.IMAGENET1K] | 77.86% | 93.87% | 4.8M | 13.7 |
| [`ReXNet1_0x_Checkpoint.IMAGENETTE`][holocron.models.classification.ReXNet1_0x_Checkpoint.IMAGENETTE] | 94.39% | 99.62% | 3.5M | 13.7 |
| [`ReXNet1_3x_Checkpoint.IMAGENET1K`][holocron.models.classification.ReXNet1_3x_Checkpoint.IMAGENET1K] | 79.50% | 94.68% | 7.6M | 13.7 |
| [`ReXNet1_3x_Checkpoint.IMAGENETTE`][holocron.models.classification.ReXNet1_3x_Checkpoint.IMAGENETTE] | 94.88% | 99.39% | 5.9M | 22.8 |
| [`ReXNet1_5x_Checkpoint.IMAGENET1K`][holocron.models.classification.ReXNet1_5x_Checkpoint.IMAGENET1K] | 80.31% | 95.17% | 9.7M | 13.7 |
| [`ReXNet1_5x_Checkpoint.IMAGENETTE`][holocron.models.classification.ReXNet1_5x_Checkpoint.IMAGENETTE] | 94.47% | 99.62% | 7.8M | 30.2 |
| [`ReXNet2_0x_Checkpoint.IMAGENET1K`][holocron.models.classification.ReXNet2_0x_Checkpoint.IMAGENET1K] | 80.31% | 95.17% | 16.4M | 13.7 |
| [`ReXNet2_0x_Checkpoint.IMAGENETTE`][holocron.models.classification.ReXNet2_0x_Checkpoint.IMAGENETTE] | 95.24% | 99.57% | 13.8M | 53.1 |
| [`ReXNet2_2x_Checkpoint.IMAGENETTE`][holocron.models.classification.ReXNet2_2x_Checkpoint.IMAGENETTE] | 95.44% | 99.46% | 16.7M | 64.1 |
| [`RepVGG_A0_Checkpoint.IMAGENETTE`][holocron.models.classification.RepVGG_A0_Checkpoint.IMAGENETTE] | 92.92% | 99.46% | 24.7M | 94.6 |
| [`RepVGG_A1_Checkpoint.IMAGENETTE`][holocron.models.classification.RepVGG_A1_Checkpoint.IMAGENETTE] | 93.78% | 99.18% | 30.1M | 115.1 |
| [`RepVGG_A2_Checkpoint.IMAGENETTE`][holocron.models.classification.RepVGG_A2_Checkpoint.IMAGENETTE] | 93.63% | 99.39% | 48.6M | 185.8 |
| [`RepVGG_B0_Checkpoint.IMAGENETTE`][holocron.models.classification.RepVGG_B0_Checkpoint.IMAGENETTE] | 92.69% | 99.21% | 31.8M | 121.8 |
| [`RepVGG_B1_Checkpoint.IMAGENETTE`][holocron.models.classification.RepVGG_B1_Checkpoint.IMAGENETTE] | 93.96% | 99.39% | 100.8M | 385.1 |
| [`RepVGG_B2_Checkpoint.IMAGENETTE`][holocron.models.classification.RepVGG_B2_Checkpoint.IMAGENETTE] | 94.14% | 99.57% | 157.5M | 601.2 |
| [`Res2Net50_26w_4s_Checkpoint.IMAGENETTE`][holocron.models.classification.Res2Net50_26w_4s_Checkpoint.IMAGENETTE] | 93.94% | 99.41% | 23.7M | 90.6 |
| [`ResNeXt50_32x4d_Checkpoint.IMAGENETTE`][holocron.models.classification.ResNeXt50_32x4d_Checkpoint.IMAGENETTE] | 94.55% | 99.49% | 23.0M | 88.1 |
| [`ResNet18_Checkpoint.IMAGENETTE`][holocron.models.classification.ResNet18_Checkpoint.IMAGENETTE] | 93.61% | 99.46% | 11.2M | 42.7 |
| [`ResNet34_Checkpoint.IMAGENETTE`][holocron.models.classification.ResNet34_Checkpoint.IMAGENETTE] | 93.81% | 99.49% | 21.3M | 81.3 |
| [`ResNet50D_Checkpoint.IMAGENETTE`][holocron.models.classification.ResNet50D_Checkpoint.IMAGENETTE] | 94.65% | 99.52% | 23.5M | 90.1 |
| [`ResNet50_Checkpoint.IMAGENETTE`][holocron.models.classification.ResNet50_Checkpoint.IMAGENETTE] | 93.78% | 99.54% | 23.5M | 90 |
| [`SKNet50_Checkpoint.IMAGENETTE`][holocron.models.classification.SKNet50_Checkpoint.IMAGENETTE] | 94.37% | 99.54% | 35.2M | 134.7 |




## Object Detection

Object detection models expect a 4D image tensor as an input (N x C x H x W) and returns a list of dictionaries.
Each dictionary has 3 keys: box coordinates, classification probability, classification label.

```python
import holocron.models as models
yolov2 = models.yolov2(num_classes=10)
```

### YOLO

::: holocron.models.detection
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - YOLOv1
            - yolov1
            - YOLOv2
            - yolov2
            - YOLOv4
            - yolov4

## Semantic Segmentation

Semantic segmentation models expect a 4D image tensor as an input (N x C x H x W) and returns a classification score
tensor of size (N x K x Ho x Wo).

```python
import holocron.models as models
unet = models.unet(num_classes=10)
```

### U-Net

::: holocron.models.segmentation
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - UNet
            - DynamicUNet
            - unet
            - unet2
            - unet_tvvgg11
            - unet_tvresnet34
            - unet_rexnet13

::: holocron.models.segmentation
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - UNetp
            - unetp
            - UNetpp
            - unetpp

::: holocron.models.segmentation
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - UNet3p
            - unet3p
