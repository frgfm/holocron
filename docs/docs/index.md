# Holocron: a Deep Learning toolbox for PyTorch

<p align="center">
    <img src="img/logo_text.svg" alt="Holocron logo" width="50%">
</p>

Holocron is meant to bridge the gap between [PyTorch](https://pytorch.org/) and latest research papers. It brings training components that are not available yet in PyTorch with a similar interface.

This project is meant for:

* :zap: **speed**: architectures in this repo are picked for both pure performances and minimal latency
* :woman_scientist: **research**: train your models easily to SOTA standards

## Installation

Create and activate a virtual environment and then install Holocron:

```shell
pip install pylocron
```

Check out the [installation guide](getting-started/installation.md) for more options

## Quick start

Load a model and use it just like any PyTorch model:

```python hl_lines="5 8"
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize
from torchvision.transforms.v2.functional import InterpolationMode
from holocron.models.classification import repvgg_a0

# Load your model (weights pretrained on Imagenette, a 10-class subset of ImageNet)
model = repvgg_a0(pretrained=True).eval()

# Read your image
img = Image.open(path_to_an_image).convert("RGB")

# Preprocessing (model.default_cfg is the Checkpoint the pretrained weights came from)
config = model.default_cfg
transform = Compose([
    Resize(config.pre_processing.input_shape[1:], interpolation=InterpolationMode.BILINEAR),
    PILToTensor(),
    ConvertImageDtype(torch.float32),
    Normalize(config.pre_processing.mean, config.pre_processing.std),
])

input_tensor = transform(img).unsqueeze(0)

# Inference
with torch.inference_mode():
    output = model(input_tensor)
print(config.meta.categories[output.squeeze(0).argmax().item()], output.squeeze(0).softmax(dim=0).max())
```

## Model zoo

Holocron implements architectures directly from their papers and trains its own weights: most classification models on [Imagenette](https://github.com/fastai/imagenette) (a 10-class subset of ImageNet), and the ReXNet family on full ImageNet-1k. These weights load through Holocron's own `pretrained=True` and are **not** interchangeable with torchvision/`timm` checkpoints. Top-1 accuracy is measured on the listed dataset's validation split, so Imagenette (10 classes) numbers are not comparable to ImageNet-1k (1000 classes) ones.

The classification architectures below ship pretrained weights:

<!-- AUTOGEN:MODEL_ZOO START - edit via .github/generate_model_zoo.py -->

| Model | Input | Training dataset | Top-1 acc (%) | Params (M) |
| --- | --- | --- | --- | --- |
| `convnext_atto` | 224×224 | Imagenette (10) | 87.6 | 3.4 |
| `cspdarknet53` | 224×224 | Imagenette (10) | 94.5 | 26.6 |
| `cspdarknet53_mish` | 224×224 | Imagenette (10) | 94.7 | 26.6 |
| `darknet19` | 224×224 | Imagenette (10) | 93.9 | 19.8 |
| `darknet24` | 224×224 | Imagenette (10) | — | — |
| `darknet53` | 224×224 | Imagenette (10) | 94.2 | 40.6 |
| `mobileone_s0` | 224×224 | Imagenette (10) | 88.1 | 4.3 |
| `mobileone_s1` | 224×224 | Imagenette (10) | 91.3 | 3.6 |
| `mobileone_s2` | 224×224 | Imagenette (10) | 91.3 | 5.9 |
| `mobileone_s3` | 224×224 | Imagenette (10) | 91.1 | 8.1 |
| `repvgg_a0` | 224×224 | Imagenette (10) | 92.9 | 24.7 |
| `repvgg_a1` | 224×224 | Imagenette (10) | 93.8 | 30.1 |
| `repvgg_a2` | 224×224 | Imagenette (10) | 93.6 | 48.6 |
| `repvgg_b0` | 224×224 | Imagenette (10) | 92.7 | 31.8 |
| `repvgg_b1` | 224×224 | Imagenette (10) | 94.0 | 100.8 |
| `repvgg_b2` | 224×224 | Imagenette (10) | 94.1 | 157.5 |
| `res2net50_26w_4s` | 224×224 | Imagenette (10) | 93.9 | 23.7 |
| `resnet18` | 224×224 | Imagenette (10) | 93.6 | 11.2 |
| `resnet34` | 224×224 | Imagenette (10) | 93.8 | 21.3 |
| `resnet50` | 224×224 | Imagenette (10) | 93.8 | 23.5 |
| `resnet50d` | 224×224 | Imagenette (10) | 94.7 | 23.5 |
| `resnext50_32x4d` | 224×224 | Imagenette (10) | 94.5 | 23.0 |
| `rexnet1_0x` | 224×224 | ImageNet-1k (1000) | 77.9 | 4.8 |
| `rexnet1_3x` | 224×224 | ImageNet-1k (1000) | 79.5 | 7.6 |
| `rexnet1_5x` | 224×224 | ImageNet-1k (1000) | 80.3 | 9.7 |
| `rexnet2_0x` | 224×224 | ImageNet-1k (1000) | 80.3 | 16.4 |
| `rexnet2_2x` | 224×224 | Imagenette (10) | 95.4 | 16.7 |
| `sknet50` | 224×224 | Imagenette (10) | 94.4 | 35.2 |
| `tridentnet50` | 224×224 | Imagenette (10) | — | — |

_Rows showing `—` are legacy checkpoints whose accuracy/params are not recorded in metadata._

<!-- AUTOGEN:MODEL_ZOO END -->

`unet_rexnet13` is the only segmentation model with pretrained weights, and the detection models (`yolov1`, `yolov2`, `yolov4`) ship none yet. Requesting `pretrained=True` for any architecture without a checkpoint emits a warning and falls back to random initialization — train your own with the [reference scripts](https://github.com/frgfm/Holocron/tree/main/references).

Holocron implements the following architectures (with reference papers):

### Image classification
* TridentNet from ["Scale-Aware Trident Networks for Object Detection"](https://arxiv.org/pdf/1901.01892.pdf)
* SKNet from ["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf)
* PyConvResNet from ["Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"](https://arxiv.org/pdf/2006.11538.pdf)
* ReXNet from ["ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"](https://arxiv.org/pdf/2007.00992.pdf)
* RepVGG from ["RepVGG: Making VGG-style ConvNets Great Again"](https://arxiv.org/pdf/2101.03697.pdf)

### Semantic segmentation
* U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
* U-Net++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)
* UNet3+ from ["UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation"](https://arxiv.org/pdf/2004.08790.pdf)

### Object detection
* YOLO from ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf)
* YOLOv2 from ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
* YOLOv4 from ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf)
