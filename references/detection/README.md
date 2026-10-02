# Object detection

The sample training script was made to train object detection models on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Validation status

YOLOv4's corrected training implementation is verified by regression tests and the CPU learning check below: finite losses and gradients, correct localization and labels under normal inference, and a saved checkpoint. The matched synthetic control reaches 0% detection error with corrected initialization, versus 99.43% when the old initialization is restored.

Full CUDA/VOC training on the final repaired implementation remains pending. This does not claim validation-set convergence or paper-level accuracy. YOLOv1/v2 loss behavior is preserved, and their shared inference changes have regression coverage. YOLOv3 is not implemented as a detector. There are no published pretrained detection checkpoints; YOLOv4 uses a pretrained Imagenette backbone by default.

## Getting started

Ensure that you have holocron installed

```bash
git clone https://github.com/frgfm/Holocron.git
cd Holocron
uv sync --locked --extra training
cd references/detection
```

No need to download the dataset, torchvision will handle [this](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCDetection) for you! From there, you can run your training with the following command

```bash
uv run --project ../.. --no-sync python train.py VOC2012 --arch yolov2 --lr 1e-5 -b 32 -j 16 --epochs 20 --opt radam --sched onecycle
```

### YOLOv4 smoke gate

This is the pending hardware acceptance gate. Run the five-epoch correctness smoke from `references/detection`:

```bash
uv run --project ../.. --no-sync python train.py VOC2012 --arch yolov4 --img-size 608 --lr 1.3e-3 -b 8 --grad-acc 8 -j 2 --epochs 5 --opt sgd --momentum 0.949 --wd 5e-4 --sched onecycle --freeze-until backbone --amp --device 0 --output-file ./checkpoints/yolov4-voc-smoke.pth
```

Accept it only when all losses remain finite, the checkpoint is created, and epoch-five localization and detection errors are lower than epoch one. This smoke does not claim paper-level accuracy.

YOLOv4 loads the pretrained Imagenette CSPDarknet53-Mish backbone by default, even without `--pretrained`; `--freeze-until backbone` freezes those weights. The detector head is trained from scratch.

Before tuning the learning rate, use `--check-setup --grad-acc 1` to check fixed-batch overfitting. The LR finder plots training loss and accepts `--find-lr-start` and `--find-lr-end`. Each optimizer requires its own sweep. CodeCarbon is quiet by default; enable its logs with `--verbose-codecarbon`.

The reported `val_loss` is localization error, not the training loss evaluated on validation images. Classification error is conditional on localized matches; a low classification error alone does not establish detection quality.

### Reproducible CPU learning check

From the repository root, run:

```bash
uv run --no-sync python references/detection/check_yolov4.py
```

This uses the pretrained backbone and trains the actual neck and head for 500 SGD updates on two synthetic 96-pixel rectangle images repeated to a batch of eight. Frozen backbone features are cached, and DropBlock is disabled only for this memorization check. The script requires finite losses and gradients, a substantial loss decrease, and zero detection error under normal evaluation and NMS. It saves a checkpoint and JSON report under `checkpoints/`.

A CPU run with seed 42, SGD LR `1e-3`, momentum `0.949`, weight decay `5e-4`, gradient clip `1.0`, and cosine decay produced:

| Metric after 500 updates | Redundant parent initialization restored | Corrected initialization |
| --- | --- | --- |
| Total training loss | 23.1761 | 1.7893 |
| Area-weighted box loss | 0.239717 | 0.00002360 |
| Best correctly labeled IoU, images 1 / 2 | 0.4373 / 0.5166 | 0.8113 / 0.7945 |
| Detections per image, images 1 / 2 | 178 / 172 | 1 / 1 |
| False positives per image, images 1 / 2 | 178 / 171 | 0 / 0 |
| Fixed-batch detection error | 99.43% | 0% |

Both runs used the same examples and optimization settings. This control isolates the constructor's accidental overwrite of the zero-initialized prediction layers. These are fixed-batch learning results, not validation-set accuracy; the full CUDA/VOC smoke remains a separate gate.



## Personal leaderboard

### PASCAL VOC 2012

Performances are evaluated on the validation set of the dataset. Since the mAP does not allow easy interpretation by humans, the performance metrics have been changed here.

A prediction is considered as correct if it checks two criteria:

- Localization: it is the best acceptable localization candidate (highest IoU among predictions with the GT, and IoU >= 0.5)
- Classification: the top predicted probabilities is for the class label of the matched ground truth object.

Then we define:

- **Localization error rate**: with loc_recall being the matching rate of ground truth boxes, and loc_precision being the matching rate of predicted boxes, we define the localization error as 1 - (harmonic mean of localization loc_recall & loc_precision)
- **Classification error rate**: classification error rate of matched predictions.
- **Detection error rate**: with det_recall being the correctness rate of ground truth boxes, and det_precision being the correctness rate of predicted boxes, we define the localization error as 1 - (harmonic mean of localization det_recall & det_precision)

Here, the recall being the ratio of correctly predicted ground truth predictions by the total number of ground truth objects, and the precision being the ratio of correctly predicted ground truth predictions by the total number of predicted boxes.

| Size (px) | Epochs | args                                                         | Loc@.5 | Clf@.5 | Det@.5 | # Runs |
| --------- | ------ | ------------------------------------------------------------ | ------ | ------ | ------ | ------ |
| 416       | 40     | VOC2012 --arch yolov2 --img-size 416 --lr 5e-4 -b 64 -j 16 --epochs 40 --opt tadam --freeze-backbone --sched onecycle | 83.09  | 52.82  | 92.02  | 1      |



## Model zoo

| Model  | Loc@.5 | Clf@.5 | Det@.5 | Param # | MACs | Interpolation | Image size |
| ------ | ------ | ------ | ------ | ------- | ---- | ------------- | ---------- |
| yolov2 | 83.09  | 52.82  | 92.02  | 50.65M  |      | bilinear      | 416        |
