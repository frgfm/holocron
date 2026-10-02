# Semantic segmentation

The sample training script trains semantic segmentation models on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Getting started

Ensure that you have holocron installed

```bash
git clone https://github.com/frgfm/Holocron.git
pip install -e "Holocron/.[training]"
```

No need to download the dataset, torchvision will handle [this](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCSegmentation) for you! From there, you can run your training with the following command

```bash
python references/segmentation/train.py VOC2012 --arch unet3p -b 4 -j 4 --opt radam --lr 1e-3 --sched onecycle --epochs 20 --img-size 256
```

Run from the repository root. `--arch unet` and `--arch unetpp` use the same pipeline. Existing `VOCdevkit/VOC2012` directories are reused without downloading.

`--img-size` controls the training crop and validation resolution. `--norm-wd` sets normalization weight decay, and `--grad-acc` accumulates microbatches, including partial batches at epoch end. Cross-entropy, focal, and mutual-channel losses support ignored labels (`255`); mutual-channel evaluation uses deterministic class scores. Mean IoU excludes classes absent from both targets and predictions.

Weights & Biases and CodeCarbon are optional; enable `--wb` or `--track-emissions` to use them. `--resume` restores model weights and trainer counters; optimizer and scheduler state restart.

Regression tests exercise decoder shapes and gradients, ignored-label losses, validation metrics, optimizer groups, and short training/checkpoint reloads for all three architectures:

```bash
OMP_NUM_THREADS=2 MPLBACKEND=Agg pytest tests/test_models_segmentation.py tests/test_segmentation_training.py tests/test_nn_loss.py tests/test_trainer_utils.py
```

Some model-zoo tests download pretrained backbones. CPU learning checks also trained the full default models on synthetic masks and a small microscopy sample; CUDA AMP and full VOC accuracy require separate validation.


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
