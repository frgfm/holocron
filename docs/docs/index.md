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

```python hl_lines="4 7"
from PIL import Image
from torchvision.transforms.v2 import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize
from torchvision.transforms.v2.functional import InterpolationMode
from holocron.models.classification import repvgg_a0

# Load your model
model = repvgg_a0(pretrained=True).eval()

# Read your image
img = Image.open(path_to_an_image).convert("RGB")

# Preprocessing
config = model.default_cfg
transform = Compose([
    Resize(config['input_shape'][1:], interpolation=InterpolationMode.BILINEAR),
    PILToTensor(),
    ConvertImageDtype(torch.float32),
    Normalize(config['mean'], config['std'])
])

input_tensor = transform(img).unsqueeze(0)

# Inference
with torch.inference_mode():
    output = model(input_tensor)
print(config['classes'][output.squeeze(0).argmax().item()], output.squeeze(0).softmax(dim=0).max())
```

## Model zoo

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
