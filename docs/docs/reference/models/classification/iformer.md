# iFormer

iFormer is based on the ICLR 2025 paper
["iFormer: Integrating ConvNet and Transformer for Mobile Application"](https://proceedings.iclr.cc/paper_files/paper/2025/hash/396a3fc4560b1fe85574ebebe2b2a739-Abstract-Conference.html)
and its [official implementation](https://github.com/ChuanyangZheng/iFormer).

## Architecture overview

The compact variants use a fused inverted-bottleneck stem, depthwise
convolutional blocks for early local processing, and self-modulation attention
with convolutional feed-forward blocks in their final stages. The Holocron
implementation follows the official T/S/M layouts using standard PyTorch
operators and exposes `forward_features`, `forward_head`, `get_classifier`, and
`reset_classifier`.

## Paper evidence

These are ImageNet-1K results reported by the authors for 224px inputs and a
300-epoch, non-distilled recipe. They are not Holocron benchmark results.

| Model | Parameters | MACs | Top-1 |
|---|---:|---:|---:|
| iFormer-T | 2.9M | 0.53G | 74.1% |
| iFormer-S | 6.5M | 1.09G | 78.8% |
| iFormer-M | 8.9M | 1.64G | 80.4% |

The exact 1,000-class parameter counts below match the official implementation.

| Model | Parameters | Holocron maturity | Holocron checkpoint | Holocron metrics |
|---|---:|---|---|---|
| iFormer-T | 2,886,456 | Preview | Not published | Not measured |
| iFormer-S | 6,563,368 | Preview | Not published | Not measured |
| iFormer-M | 8,907,424 | Preview | Not published | Not measured |

## Low-memory Imagenette command

The existing classification reference supports a physical batch of 16 with two
gradient-accumulation steps for an effective batch size of 32:

```shell
uv run --extra training python references/classification/train.py ./imagenette2-320 \
  --arch iformer_t --batch-size 16 --grad-acc 2 --amp --device 0 --epochs 20 \
  --lr 1e-3 --opt adamp --sched onecycle --mixup-alpha 0.2 \
  --label-smoothing 0.1 --seed 0
```

This is runnable guidance, not a completed benchmark. Record the completed run
manifest, hardware/software versions, peak memory, elapsed time, top-1, and
top-5 before promoting a variant from preview.

## Model builders

All builders rely on [`IFormer`][holocron.models.IFormer] and accept a custom
class count through `num_classes`.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - IFormer
            - iformer_t
            - iformer_s
            - iformer_m
