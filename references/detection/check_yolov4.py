# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""CPU fixed-batch YOLOv4 learning check; this is not a VOC accuracy benchmark."""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from holocron.models.detection import yolov4
from holocron.nn import DropBlock2d
from holocron.trainer import DetectionTrainer
from holocron.trainer.utils import freeze_model


def check_gradients(optimizer, _args, _kwargs):
    norms = torch.stack([
        parameter.grad.norm()
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ])
    if not torch.isfinite(norms).all():
        raise ValueError("Non-finite parameter gradients")


def main(args):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    images = torch.rand(2, 3, 96, 96) * 0.08 + 0.15
    targets = []
    for label, (left, top, right, bottom) in enumerate(((16, 16, 64, 64), (40, 24, 80, 72))):
        images[label, :, top:bottom, left:right] = 0.1
        images[label, label, top:bottom, left:right] = 0.95
        targets.append({
            "boxes": torch.tensor([[left, top, right, bottom]], dtype=torch.float32) / 96,
            "labels": torch.tensor([label]),
        })
    images = (images - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(
        1, 3, 1, 1
    )
    # Repeat the two unique examples to reduce BatchNorm bias on 3x3 feature maps.
    images = images.repeat(4, 1, 1, 1)
    targets = [{key: value.clone() for key, value in target.items()} for _ in range(4) for target in targets]
    loader = DataLoader([(images, targets)], batch_size=None)
    model = yolov4(num_classes=2, pretrained_backbone=True, progress=False)
    freeze_model(model.train(), "backbone")
    model.backbone.eval()
    for module in model.modules():
        if isinstance(module, DropBlock2d):
            module.p = 0  # Disable stochastic regularization only for this memorization check.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.949, weight_decay=5e-4)
    trainer = DetectionTrainer(model, loader, loader, nn.Identity(), optimizer, gradient_clip=1.0)
    trainer.optimizer.register_step_pre_hook(check_gradients)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, args.steps)
    with torch.no_grad():
        # Cache the unchanged pretrained backbone features; train the actual neck and head.
        features = [feature.detach() for feature in model.backbone(images)]
    initial_losses = None
    for step in range(args.steps):
        losses = model.head(model.neck(features), targets)
        if not all(torch.isfinite(value) for value in losses.values()):
            raise ValueError(f"Non-finite loss at step {step + 1}")
        scalars = {key: value.item() for key, value in losses.items()}
        if initial_losses is None:
            initial_losses = scalars
        trainer._backprop_step(sum(losses.values()))
        trainer.step += 1
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": sum(scalars.values()), **scalars}), flush=True)

    metrics = trainer.evaluate()
    with torch.inference_mode():
        detections = model(images)
    predictions = []
    for detection, target in zip(detections, targets, strict=True):
        correct = detection["labels"] == target["labels"][0]
        ious = box_iou(target["boxes"], detection["boxes"])[0]
        predictions.append({
            "detections": len(ious),
            "correct_label_iou": ious[correct].max().item() if correct.any() else 0,
        })
    trainer.save(str(args.output))
    result = {
        "pretrained_backbone": True,
        "unique_images": 2,
        "batch_size": 8,
        "steps": args.steps,
        "initial_losses": initial_losses,
        "final_losses": scalars,
        "finite_loss_and_gradients": True,
        "metrics": metrics,
        "predictions": predictions,
        "checkpoint": str(args.output),
        "checkpoint_bytes": args.output.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if metrics["det_err"] != 0 or sum(scalars.values()) >= 0.1 * sum(initial_losses.values()):
        raise ValueError("Fixed-batch learning check failed; inspect the saved checkpoint and JSON report")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("./checkpoints/yolov4-fixed-batch.pth"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    main(args)
