# RepViT

RepViT is based on the
["RepViT: Revisiting Mobile CNN From ViT Perspective"](https://arxiv.org/abs/2307.09283)
paper and its [official implementation](https://github.com/THU-MIG/RepViT).

## Architecture overview

RepViT adapts mobile CNN blocks using design choices associated with efficient
vision transformers. Each block separates spatial token mixing from channel
mixing, uses squeeze-excitation selectively, and can fuse its training-time
depthwise branches for deployment.

Call `model.eval()` and then `model.reparametrize()` before exporting or
benchmarking the deployment form.

## Paper evidence

These ImageNet-1K results are teacher-distilled scores reported by the authors,
not Holocron benchmark results.

| Model | Parameters | MACs | Top-1 |
|---|---:|---:|---:|
| RepViT-M0.9 | 5.1M | 0.8G | 78.7% |
| RepViT-M1.0 | 6.8M | 1.1G | 80.0% |
| RepViT-M1.1 | 8.2M | 1.3G | 80.7% |

## Controlled Holocron benchmark

The Holocron comparison trains from scratch on Imagenette without a teacher:
176px training crops, 232px resize and 224px validation crops, 20 epochs,
effective batch size 32, AMP, AdamP at `1e-3`, OneCycle, Mixup `0.2`, and label
smoothing `0.1`. MobileOne-S2 uses the identical command as the baseline.

CUDA measurements remain a separate acceptance gate for
[issue #499](https://github.com/frgfm/holocron/issues/499); they are not inferred
from local CPU or MPS checks.

| Model | Parameters before/after fusion | MACs | Top-1 | Top-5 | Status |
|---|---:|---:|---:|---:|---|
| RepViT-M0.9 | Pending | Pending | Pending | Pending | CUDA run required |
| RepViT-M1.0 | Pending | Pending | Pending | Pending | CUDA run required |
| RepViT-M1.1 | Pending | Pending | Pending | Pending | CUDA run required |
| MobileOne-S2 | Pending rerun | Pending | Pending | Pending | CUDA run required |

No pretrained RepViT checkpoint is published with this implementation.

## Model builders

All builders rely on [`RepViT`][holocron.models.RepViT] and accept a custom
class count through `num_classes`.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - RepViT
            - repvit_m0_9
            - repvit_m1_0
            - repvit_m1_1
