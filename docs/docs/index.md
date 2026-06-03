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
from holocron.models.classification import darknet24

# Load your model (weights pretrained on Imagenette, a 10-class subset of ImageNet)
model = darknet24(pretrained=True).eval()

# Read your image
img = Image.open(path_to_an_image).convert("RGB")

# Preprocessing
config = model.default_cfg
transform = Compose([
    Resize(config["input_shape"][1:], interpolation=InterpolationMode.BILINEAR),
    PILToTensor(),
    ConvertImageDtype(torch.float32),
    Normalize(config["mean"], config["std"]),
])

input_tensor = transform(img).unsqueeze(0)

# Inference
with torch.inference_mode():
    output = model(input_tensor)
print(config["classes"][output.squeeze(0).argmax().item()], output.squeeze(0).softmax(dim=0).max())
```

## Model zoo

Holocron implements architectures directly from their papers and trains its own weights: most classification models on [Imagenette](https://github.com/fastai/imagenette) (a 10-class subset of ImageNet), and the ReXNet family on full ImageNet-1k. These weights load through Holocron's own `pretrained=True` and are **not** interchangeable with torchvision/`timm` checkpoints. Top-1 accuracy is reported on each model's own training set, so Imagenette (10 classes) numbers are not comparable to ImageNet-1k (1000 classes) ones.

### Image classification

| Model | Input | Training dataset | Top-1 acc (%) | Params (M) |
| --- | --- | --- | --- | --- |
| `convnext_atto` | 224×224 | Imagenette (10) | 87.6 | 3.4 |
| `cspdarknet53` | 224×224 | Imagenette (10) | 94.5 | 26.6 |
| `cspdarknet53_mish` | 224×224 | Imagenette (10) | 94.7 | 26.6 |
| `darknet19` | 224×224 | Imagenette (10) | 93.9 | 19.8 |
| `darknet24` | 224×224 | Imagenette (10) | — | 22.4 |
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
| `tridentnet50` | 224×224 | Imagenette (10) | — | 45.8 |

### Semantic segmentation

Only `unet_rexnet13` (~9.3M params) currently ships pretrained weights.

### Object detection

The detection models (`yolov1`, `yolov2`, `yolov4`) ship **no pretrained weights yet** — train them with the [reference scripts](https://github.com/frgfm/Holocron/tree/main/references/detection).

Every other architecture is available **untrained** (randomly initialized); calling it with `pretrained=True` emits a warning and falls back to random initialization. See the [project README](https://github.com/frgfm/Holocron#paper-references) for the full catalogue of implemented architectures.
