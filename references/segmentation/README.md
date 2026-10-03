# Semantic segmentation

The sample training script trains semantic segmentation models on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Getting started

Follow the [shared installation instructions](../README.md#installation). Run the commands below from the repository root.

No need to download the dataset, torchvision will handle [this](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCSegmentation) for you! From there, you can run your training with the following command

```bash
uv run --no-sync python references/segmentation/train.py VOC2012 --arch unet3p -b 4 -j 4 --opt radam --lr 1e-3 --sched onecycle --epochs 20 --img-size 256
```

`--arch unet` and `--arch unetpp` use the same pipeline. Existing `VOCdevkit/VOC2012` directories are reused without downloading.

`--img-size` controls the training crop and validation resolution. `--norm-wd` sets normalization weight decay, and `--grad-acc` accumulates microbatches, including partial batches at epoch end. Holocron retains partial training batches; torchvision drops them to avoid singleton BatchNorm failures. Cross-entropy, focal, and mutual-channel losses support ignored labels (`255`); mutual-channel evaluation uses deterministic class scores. Validation loss averages labelled images independently of batch grouping and rejects entirely unlabelled datasets. Mean IoU excludes classes absent from both targets and predictions.

Weights & Biases and CodeCarbon are optional; enable `--wb` or `--track-emissions` to use them. `--resume` restores model weights and trainer counters; optimizer and scheduler state restart.

Regression tests exercise decoder shapes and gradients, ignored-label losses, validation metrics, optimizer groups, and short training/checkpoint reloads for all three architectures:

```bash
OMP_NUM_THREADS=2 MPLBACKEND=Agg pytest tests/test_models_segmentation.py tests/test_segmentation_training.py tests/test_nn_loss.py tests/test_trainer_utils.py
```

Some model-zoo tests download pretrained backbones. CPU learning checks also trained the full default models on synthetic masks and a small microscopy sample; CUDA AMP and full VOC accuracy require separate validation.

## YOLO26 nano

`yolo26n_sem` is an independent implementation of the YOLO26 nano semantic
model. It has 1,632,902 parameters for 19 classes during training and 1,552,795
after fusion and auxiliary-head removal. The latter count rounds to the
published 1.6 million. Holocron has no pretrained weights for this model.

The main and auxiliary heads return full-resolution scores during training.
The existing trainer uses both heads and ignores label `255`. Evaluation
returns the main score tensor. For example, train on VOC with:

```bash
uv run --no-sync python references/segmentation/train.py VOC2012 --arch yolo26n_sem -b 4 -j 4 --opt radam --lr 3e-3 --sched cosine --epochs 30 --img-size 256
```

### Reproduce the learning checks

The synthetic check uses 48 training images and 24 independently generated
validation images. Shapes, locations, sizes, brightness, and noise vary.
It exercises the complete model and trainer from random weights.

```bash
OMP_NUM_THREADS=2 uv run --no-sync python references/segmentation/check_yolo26.py --output references/segmentation/results/yolo26-semantic-synthetic.json
```

The real-image check uses the 170-image Penn-Fudan pedestrian dataset. This
small dataset has a custom, fixed split of 120 training, 25 validation, and
25 test images. The validation set selects the checkpoint. The test set is
used only after that selection. Instance masks are merged into a binary
person/background mask, and images and masks are resized to 128 by 128.
Only the training images receive flip and brightness augmentation.

```bash
git clone https://github.com/swallan/PennFudanPed.git /tmp/PennFudanPed
git -C /tmp/PennFudanPed checkout ec1d4583fb436b14e2062587c8b28a5018668a5e
OMP_NUM_THREADS=2 uv run --no-sync python references/segmentation/check_yolo26_pennfudan.py /tmp/PennFudanPed --output references/segmentation/results/yolo26-semantic-pennfudan.json
```

Use `--checkpoint /path/model.pth` to keep the selected checkpoint. Dataset
images are not included in this repository; see the source dataset's terms.
The JSON files record the split, seeds, training settings, validation history,
and measured results. These are small learning checks, not reproductions of
the published Cityscapes result or a standard Penn-Fudan benchmark.

| Check | Setup | Result |
| --- | --- | --- |
| [Synthetic validation](results/yolo26-semantic-synthetic.json) | 48 train / 24 validation images; 64 px; 3 classes; 360 updates | Mean IoU **5.74% → 95.02%**; final pixel accuracy **98.51%** |
| [Penn-Fudan validation](results/yolo26-semantic-pennfudan.json) | 120 train / 25 validation images; 128 px; 2 classes; 600 updates | Mean IoU **9.65% → 75.62%**; checkpoint selected at epoch 23 of 30 |
| Penn-Fudan test | 25 images held out from training and model selection | Mean IoU **76.01%**; **person IoU 59.94%**; background IoU **92.08%** |
| Trained model fusion | Penn-Fudan checkpoint, one validation image | Maximum score difference **2.87e-6** |

Both runs train all model parameters from scratch with FP32 on CPU and two
threads. The real-image run took 98 seconds on an Intel Xeon Platinum 8370C.
This elapsed time is a training-run measurement, not an inference benchmark.

Focused checks cover odd input dimensions, gradients in both heads,
ignored-label losses, the VOC reference script, checkpoint reload, parameter
counts, fusion, and ONNX reference output parity:

```bash
OMP_NUM_THREADS=2 MPLBACKEND=Agg uv run --no-sync pytest tests/test_yolo26_backbone.py tests/test_models_yolo26_semantic.py tests/test_segmentation_training.py
```


## Personal leaderboard

### PASCAL VOC 2012

Performances are evaluated on the validation set of the dataset using the mean IoU metric.

| Size (px) | Epochs | args                                                         | mean IoU | # Runs |
| --------- | ------ | ------------------------------------------------------------ | -------- | ------ |
| 256       | 200    | VOC2012 --arch unet_rexnet13 -b 16 --loss crossentropy --label-smoothing 0.1 --opt adamp --device 0 --lr 2e-3 --epochs 200 | 32.14    | 1      |
| 256       | 20     | VOC2012 --arch unet3p -b 4 -j 16 --opt radam --lr 1e-5 --sched onecycle --epochs 20 | 14.17    | 1      |



## Model zoo

| Model         | mean IoU | Param # | MACs | Interpolation | Image size |
| ------------- | -------- | ------- | ---- | ------------- | ---------- |
| unet          |          | 18.11M |      | bilinear      | 256        |
| unetp         |          | 28.28M  |      | bilinear      | 256        |
| unetpp        |          | 29.54M  |      | bilinear      | 256        |
| unet3p        |          | 26.93M  |      | bilinear      | 256    |
| unet_tvvgg11  |          | 32.17M |      | bilinear      | 256        |
| unet_tvresnet34 |     | 36.25M |      | bilinear      | 256        |
| unet_rexnet13 | 32.14    | 9.34M |      | bilinear      | 256        |
