# YOLO26 nano: training and validation

YOLO26 nano predicts object boxes and classes with a small neural network.
Three feature scales help it handle different object sizes. Two prediction
branches train together. The inference branch learns one match per object and
does not apply NMS. `to_deploy()` removes the other branch and folds batch norms.

This is an original implementation of the published nano architecture. Its
80-class parameter counts match the public configuration: **2,572,280** during
training and **2,408,932** after conversion to one fused inference branch.
There are no Holocron pretrained weights. This does not reproduce the paper's
full STAL, Progressive Loss, MuSGD, augmentation, or pretraining recipe.

The [model documentation](../../docs/docs/reference/models/detection/yolo26.md)
describes the loss, local numerical guards, input format, and architecture
source. Use `--arch yolo26n` with the existing VOC training script; do not pass
`--pretrained`.

## CPU experiment

`check_yolo26.py` trains the entire model from random weights. It uses AdamW,
weight decay 0.0001, cosine decay to one tenth of the initial LR, clipping at 10,
batch size 8, seed 42, and one CPU thread. Images are resized to squares and
scaled to [0, 1]. Random horizontal flip is the only augmentation.

Each run has separate training, validation, and test data. The checkpoint with
the highest validation AP50 is selected. The optional `--calibrate` flag then
selects a score threshold by maximum validation F1 from
0.05/0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9. Ties select the higher threshold.
The test split is evaluated only after these choices. Calibration uses the
same cached predictions; it does not update network weights.

AP50 uses 101-point interpolation at IoU 0.5 and predictions with score ≥0.05.
Each target can match at most one prediction. Duplicate boxes count as false
positives. These figures are not COCO AP50:95. Precision and recall show the
effect of the score threshold in the normal NMS-free inference path.

| Recipe | Epochs | Image size | Initial learning rate |
| --- | ---: | ---: | ---: |
| Synthetic rectangles | 25 | 64 | 0.001 |
| Initial PennFudan check | 30 | 128 | 0.001 |
| Longer PennFudan check | 100 | 160 | 0.003 |

## Synthetic generalization check

```bash
uv run --no-sync python references/detection/check_yolo26.py \
  --epochs 25 --image-size 64 \
  --output references/detection/results/yolo26-rectangles.json
```

This uses 96 training, 32 validation, and 32 test images, generated with seeds
100, 200, and 300. Rectangle position, size, color, and background noise vary.
The data includes blank images and images with two objects. It is a simple
generalization check, not a natural-image accuracy claim.

The model completed 300 updates with finite losses and gradients. Validation
selected epoch 23. The mean training loss fell from 9.3905 to 2.0855 over the
25-epoch run. See the saved JSON for exact values and every epoch.

| Metric | Before training: validation | Selected checkpoint: validation | Held-out test |
| --- | ---: | ---: | ---: |
| AP50 | 0.00% | 95.30% | 99.29% |
| Precision, score ≥0.05 | 0.00% | 52.31% | 55.74% |
| Recall, score ≥0.05 | 0.00% | 100.00% | 100.00% |
| True positives / false positives | 0 / 0 | 34 / 31 | 34 / 27 |

The low precision at 0.05 matters: this small run still returns extra boxes.
High AP alone does not make this a production detector. This synthetic run
was not used to tune a test-set threshold.

## Real pedestrian images

The real-data check uses 170 PennFudan images and their instance masks from
the pinned public mirror. Boxes are computed from each nonzero instance ID.
Images are kept local and are not redistributed with this code.

```bash
git clone https://github.com/swallan/PennFudanPed.git /tmp/PennFudanPed
git -C /tmp/PennFudanPed checkout ec1d4583fb436b14e2062587c8b28a5018668a5e
uv run --no-sync python references/detection/check_yolo26.py \
  --data /tmp/PennFudanPed --epochs 30 --image-size 128 --calibrate \
  --output references/detection/results/yolo26-pennfudan.json \
  --checkpoint /tmp/yolo26-pennfudan.pth
```

The fixed split uses a seed-2026 permutation of sorted image names: 120 train,
25 validation, and 25 test images. The saved report lists every image in each
split. It is a small custom split, not a standard PennFudan benchmark protocol.
Square resizing changes image aspect ratios. Neither this check nor the
synthetic run estimates the accuracy of the published COCO checkpoint.

All 450 updates had finite losses and gradients. Mean training loss fell from
12.1940 to 4.5002. Validation AP selected epoch 21. The validation-only F1 rule
selected a score threshold of 0.1 before test evaluation.

| Metric | Before training: validation | Selected checkpoint: validation | Held-out test |
| --- | ---: | ---: | ---: |
| AP50, score ≥0.05 | 0.00% | 34.66% | 26.94% |
| Precision, score ≥0.05 | 0.00% | 24.41% | 23.46% |
| Recall, score ≥0.05 | 0.00% | 51.67% | 55.07% |
| Precision, selected score ≥0.1 | — | 38.36% | 32.97% |
| Recall, selected score ≥0.1 | — | 46.67% | 43.48% |
| F1, selected score ≥0.1 | — | 42.11% | 37.50% |
| True positives / false positives, score ≥0.1 | — | 28 / 45 | 30 / 61 |

These results show that the model learns from real images, but the test
precision and recall are poor. This short run is **not ready for production**.
The synthetic score must not be used as evidence of natural-image accuracy.
The saved [real-data report](results/yolo26-pennfudan.json) and
[synthetic report](results/yolo26-rectangles.json) contain all measurements.

The weak initial validation result led to a second, longer training recipe.
It keeps the same split and seed, raises input resolution to 160, and trains
for 100 epochs with initial LR 0.003. The initial and final recipes both use
the same test split. This is therefore not a single blind test-set estimate;
all trial results are retained, and checkpoint and threshold selection use
validation data only.

```bash
uv run --no-sync python references/detection/check_yolo26.py \
  --data /tmp/PennFudanPed --epochs 100 --image-size 160 --lr 0.003 --calibrate \
  --output references/detection/results/yolo26-pennfudan-100ep.json \
  --checkpoint /tmp/yolo26-pennfudan-100ep.pth
```

All 1,500 updates had finite losses and gradients. The 100-epoch run took
557.4 seconds on one CPU thread, including validation. Mean training loss
fell from 11.7972 to 2.3025. Validation selected epoch 42 and score threshold
0.1. The final epoch had lower validation AP50 (38.19%), so the saved
checkpoint uses epoch 42. No further recipe was tried after this run.

| Metric | Before training: validation | Selected checkpoint: validation | Held-out test |
| --- | ---: | ---: | ---: |
| AP50, score ≥0.05 | 0.00% | 47.40% | 55.04% |
| Precision, score ≥0.05 | 0.00% | 41.18% | 49.47% |
| Recall, score ≥0.05 | 0.00% | 58.33% | 68.12% |
| Precision, selected score ≥0.1 | — | 54.84% | 56.00% |
| Recall, selected score ≥0.1 | — | 56.67% | 60.87% |
| F1, selected score ≥0.1 | — | 55.74% | 58.33% |
| True positives / false positives, score ≥0.1 | — | 34 / 28 | 42 / 33 |

The longer recipe improved test AP50 from 26.94% to 55.04%, but test precision
and recall still fall short of a reliable pedestrian detector. This model
remains experimental. A full training recipe and larger, standard benchmark
evaluation are needed before claiming competitive accuracy. The
[100-epoch report](results/yolo26-pennfudan-100ep.json) retains every epoch,
the calibration candidates, and the selected checkpoint metrics.

## Published reference

The authors' released, Objects365-pretrained YOLO26n model reports **40.1 COCO
AP50:95** and **1.7 ms T4 TensorRT10 latency** at 640 pixels for its NMS-free
branch (batch 1). The NMS branch reports 40.9 AP. These are **external reference
figures**, not results for this implementation. See
[the paper](https://arxiv.org/abs/2606.03748) and the pinned source links in the
model documentation.

## Checks and limits

Tests cover losses and gradients with mixed and empty targets, independent
branch gradients, tiny-object candidates, class matching, optional NMS,
duplicate behavior, invalid inputs, tuple collation in `DetectionTrainer`,
deployment parity, and state-dictionary restoration. A static 64×64, batch-1
ONNX opset-20 export passes the ONNX checker. Dynamic export and ONNX Runtime
parity are not validated.

GPU latency, full COCO training, published checkpoint accuracy, and transfer
from upstream weights remain unverified. Reported CPU training time includes
validation and is not an inference speed benchmark.
