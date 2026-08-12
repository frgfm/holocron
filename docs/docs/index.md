# Holocron: a Deep Learning toolbox for PyTorch

<p align="center">
    <img src="img/logo_text.svg" alt="Holocron logo" width="50%">
</p>

Holocron is meant to bridge the gap between [PyTorch](https://pytorch.org/) and latest research papers. It brings training components that are not available yet in PyTorch with a similar interface.

!!! warning "Development documentation"
    These pages follow the `main` branch and Holocron `0.2.2.dev0`. The stable
    PyPI release is `0.2.1`, and some APIs differ. See the
    [installation options](getting-started/installation.md).

This project is meant for:

* :zap: **speed**: architectures in this repo are picked for both pure performances and minimal latency
* :woman_scientist: **research**: train your models easily to SOTA standards

## Installation

Create and activate a virtual environment and then install Holocron:

```shell
uv pip install "pylocron @ git+https://github.com/frgfm/holocron.git"
```

For stable `0.2.1` and system-wide options, see the
[installation guide](getting-started/installation.md).

## Quick start

Load a checkpoint and use its preprocessing and category metadata:

<!-- quickstart-example-start -->
```python
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize
from holocron.models.classification import ResNet18_Checkpoint, resnet18

checkpoint = ResNet18_Checkpoint.DEFAULT.value
model = resnet18(checkpoint=checkpoint).eval()

image = Image.open(path_to_an_image).convert("RGB")
preprocessing = checkpoint.pre_processing

transform = Compose([
    Resize(preprocessing.input_shape[1:], interpolation=preprocessing.interpolation),
    PILToTensor(),
    ConvertImageDtype(torch.float32),
    Normalize(preprocessing.mean, preprocessing.std),
])

input_tensor = transform(image).unsqueeze(0)

with torch.inference_mode():
    probabilities = model(input_tensor).squeeze(0).softmax(dim=0)

class_idx = probabilities.argmax().item()
label = checkpoint.meta.categories[class_idx]
confidence = probabilities[class_idx].item()
print(label, confidence)
```
<!-- quickstart-example-end -->

To adapt this checkpoint to your own classes, follow the
[classification and transfer-learning guide](getting-started/classification.md).

## Model zoo

Holocron implements all three tasks below, but they do not have the same level
of checkpoint and benchmark coverage. See the
[capability and maturity matrix](reference/models/models.md#support-status)
before choosing a model.

### Image classification — preview checkpoints

Published checkpoints and metrics cover Imagenette, plus selected ReXNet
ImageNet-1K variants. These historical checkpoints predate schema-v1 run
manifests and verified export evidence, so v0.3 labels them `preview` rather
than overstating their reproducibility.

* TridentNet from ["Scale-Aware Trident Networks for Object Detection"](https://arxiv.org/pdf/1901.01892.pdf)
* SKNet from ["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf)
* PyConvResNet from ["Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"](https://arxiv.org/pdf/2006.11538.pdf)
* ReXNet from ["ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"](https://arxiv.org/pdf/2007.00992.pdf)
* RepVGG from ["RepVGG: Making VGG-style ConvNets Great Again"](https://arxiv.org/pdf/2101.03697.pdf)

### Semantic segmentation — unbenchmarked

* U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
* U-Net++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)
* UNet3+ from ["UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation"](https://arxiv.org/pdf/2004.08790.pdf)

### Object detection — experimental

* YOLO from ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf)
* YOLOv2 from ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
* YOLOv4 from ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf)
