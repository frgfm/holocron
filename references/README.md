# Holocron training scripts

## Installation

Use Python 3.11 or higher, [Git](https://git-scm.com/), and [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the training dependencies:

```shell
git clone https://github.com/frgfm/Holocron.git
cd Holocron
uv sync --locked --extra training
```

Shared CLI setup and training actions live in [`_common.py`](_common.py). They belong to the reference recipes; the installed `holocron.trainer` package provides the training primitives.

## Available tasks

### Image classification

Refer to the [`./classification`](classification) folder

### Semantic segmentation

Refer to the [`./segmentation`](segmentation) folder

### Object detection

Refer to the [`./detection`](detection) folder

### Character and text recognition

Refer to the [`./recognition`](recognition) folder for a reproducible character-pretraining, CTC line-recognition, and synthetic-page OCR experiment.
