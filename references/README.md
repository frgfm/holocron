# Holocron training scripts

## Installation

Use Python 3.11 or higher, [Git](https://git-scm.com/), and [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the training dependencies:

```shell
git clone https://github.com/frgfm/Holocron.git
cd Holocron
uv sync --locked --extra training
```

Shared CLI setup and training actions live in [`_common.py`](_common.py). They belong to the reference recipes; the installed `holocron.trainer` package provides the training primitives.

## CPU, CUDA, and Apple Silicon

Classification, segmentation, detection, and character-training recipes accept
`--device cpu`, `--device cuda:0`, or `--device mps`. The shared recipes default to
`--device auto`, which chooses CUDA, then MPS, then CPU. The older `--device 0`
CUDA spelling still works. An unavailable explicitly requested accelerator raises
an error instead of silently changing devices.

```shell
uv run --no-sync python -m references.classification.train DATA --arch repvit_m0_9 --device cpu -j 4
uv run --no-sync python -m references.classification.train DATA --arch repvit_m0_9 --device cuda:0 --amp -j 4
uv run --no-sync python -m references.classification.train DATA --arch repvit_m0_9 --device mps --amp -j 4
```

CPU training uses FP32 by default; `--amp` opts into BF16 autocasting. CUDA and
MPS use FP16 autocasting with gradient scaling when `--amp` is enabled. Measure
CPU mixed precision on the target processor: it is not faster on every CPU.
Pinned host memory is used only for CUDA. Start with two to four loading workers
on a Mac, or `-j 0` when debugging. The Mixup loader supports macOS worker spawning.

For line recognition, use `--device mps` with the recognition script. Its CTC
loss runs on CPU because PyTorch has no MPS CTC kernel; gradients return to the
MPS model. Other training and inference remain on the requested device.

The reproducible [RepViT comparison](classification/benchmark_repvit_imagenette.py)
uses the same reference recipe for all three RepViT variants and MobileOne-S2:

```shell
uv run --no-sync python -m references.classification.benchmark_repvit_imagenette \
  /path/to/imagenette2-320 /path/to/fresh-results --device mps
```

The default is 20 epochs, seed 0, AMP, and effective batch size 32. Smaller
physical batches use gradient accumulation. Results include selected and final
validation accuracy, model size, convolution/linear MACs, synchronized deployment
latency, elapsed wall time, and backend-specific memory readings. MPS memory is
sampled tensor/driver allocation, not CUDA VRAM or an exact peak. Checkpoints and
progress records stay in the requested output directory, which must be empty.

## Available tasks

### Image classification

Refer to the [`./classification`](classification) folder

### Semantic segmentation

Refer to the [`./segmentation`](segmentation) folder

### Object detection

Refer to the [`./detection`](detection) folder

### Character and text recognition

Refer to the [`./recognition`](recognition) folder for a reproducible character-pretraining, CTC line-recognition, and synthetic-page OCR experiment.
