# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Train YOLO26n from scratch on split rectangle or PennFudan data; not a COCO benchmark."""

import copy
import json
import math
import platform
import time
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor

from holocron.models.detection import yolo26n


def rectangles(count, image_size, seed):
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for index in range(count):
        image = 0.08 + torch.rand(3, image_size, image_size, generator=generator) * 0.08
        boxes, labels = [], []
        # Blanks and two-object scenes test both background rejection and recall.
        objects = 0 if index % 8 == 0 else 2 if index % 5 == 0 else 1
        for object_index in range(objects):
            label = int(torch.randint(2, (), generator=generator))
            width = int(torch.randint(image_size // 5, image_size // 3, (), generator=generator))
            height = int(torch.randint(image_size // 4, image_size // 2, (), generator=generator))
            if objects == 2:
                left = (
                    int(torch.randint(2, image_size // 2 - width, (), generator=generator))
                    + object_index * image_size // 2
                )
            else:
                left = int(torch.randint(2, image_size - width - 2, (), generator=generator))
            top = int(torch.randint(2, image_size - height - 2, (), generator=generator))
            image[:, top : top + height, left : left + width] = 0.15
            image[label, top : top + height, left : left + width] = 0.8 + 0.2 * torch.rand((), generator=generator)
            boxes.append((left, top, left + width, top + height))
            labels.append(label)
        samples.append((
            image,
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4) / image_size,
                "labels": torch.tensor(labels, dtype=torch.long),
            },
        ))
    return samples


def pennfudan(directory, image_size):
    paths = sorted((directory / "PNGImages").glob("*.png"))
    if len(paths) != 170:
        raise ValueError("Expected all 170 PennFudan PNGImages")
    samples = []
    for path in paths:
        with Image.open(path) as source:
            image = pil_to_tensor(source.convert("RGB")).float() / 255
        with Image.open(directory / "PedMasks" / f"{path.stem}_mask.png") as source:
            mask = torch.from_numpy(np.array(source, copy=True))
        height, width = mask.shape
        boxes = []
        for identity in mask.unique().tolist():
            if identity == 0:
                continue
            y, x = (mask == identity).nonzero(as_tuple=True)
            boxes.append((x.min() / width, y.min() / height, (x.max() + 1) / width, (y.max() + 1) / height))
        image = F.interpolate(image[None], size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
        samples.append((image, {"boxes": torch.tensor(boxes), "labels": torch.zeros(len(boxes), dtype=torch.long)}))
    order = torch.randperm(170, generator=torch.Generator().manual_seed(2026)).tolist()
    splits = [order[:120], order[120:145], order[145:]]
    return [[samples[index] for index in split] for split in splits], [
        [paths[index].name for index in split] for split in splits
    ]


@torch.inference_mode()
def predict(model, samples, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(samples), batch_size):
        predictions.extend(model(torch.stack([image for image, _ in samples[start : start + batch_size]])))
    return predictions


def evaluate(model, samples, batch_size, num_classes, predictions=None, threshold=0.05):
    if predictions is None:
        predictions = predict(model, samples, batch_size)
    average_precisions, total_tp, total_predictions, total_gt = [], 0, 0, 0
    for label in range(num_classes):
        ground_truth = [target["boxes"][target["labels"] == label] for _, target in samples]
        count = sum(len(boxes) for boxes in ground_truth)
        detections = []
        for index, prediction in enumerate(predictions):
            selected = (prediction["labels"] == label) & (prediction["scores"] >= threshold)
            for box, score in zip(
                prediction["boxes"][selected],
                prediction["scores"][selected],
                strict=True,
            ):
                detections.append((float(score), index, box))
        if count == 0:
            # Exclude an absent class from mean AP, but retain its false alarms.
            total_predictions += len(detections)
            continue
        detections.sort(key=itemgetter(0), reverse=True)
        used = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in ground_truth]
        true_positive = []
        for _, index, box in detections:
            correct = False
            if len(ground_truth[index]):
                overlaps = box_iou(box[None], ground_truth[index])[0].masked_fill(used[index], -1)
                overlap, match = overlaps.max(0)
                if overlap >= 0.5:
                    used[index][match] = True
                    correct = True
            true_positive.append(int(correct))
        cumulative = torch.tensor(true_positive, dtype=torch.float32).cumsum(0)
        precision = cumulative / torch.arange(1, len(cumulative) + 1)
        recall = cumulative / count
        # Standard 101-point interpolation, at IoU=.5 only.
        ap = (
            sum(
                float(precision[recall >= threshold].max()) if (recall >= threshold).any() else 0.0
                for threshold in torch.linspace(0, 1, 101)
            )
            / 101
        )
        average_precisions.append(ap)
        total_tp += sum(true_positive)
        total_predictions += len(detections)
        total_gt += count
    metrics = {
        "ap50": sum(average_precisions) / max(len(average_precisions), 1),
        "precision": total_tp / max(total_predictions, 1),
        "recall": total_tp / max(total_gt, 1),
        "true_positives": total_tp,
        "false_positives": total_predictions - total_tp,
        "ground_truth_objects": total_gt,
    }
    if math.isclose(threshold, 0.05):
        metrics["precision_at_score_005"] = metrics.pop("precision")
        metrics["recall_at_score_005"] = metrics.pop("recall")
    return metrics


def operating_point(model, samples, batch_size, num_classes, predictions, threshold):
    metrics = evaluate(model, samples, batch_size, num_classes, predictions, threshold)
    precision = metrics.get("precision", metrics.get("precision_at_score_005", 0))
    recall = metrics.get("recall", metrics.get("recall_at_score_005", 0))
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
    }


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed + 1)
    if args.data is not None:
        (training, validation, testing), names = pennfudan(args.data, args.image_size)
        num_classes, dataset = 1, "PennFudanPed"
    else:
        training, validation, testing = (
            rectangles(count, args.image_size, seed) for count, seed in ((96, 100), (32, 200), (32, 300))
        )
        names = {"train_seed": 100, "validation_seed": 200, "test_seed": 300}
        num_classes, dataset = 2, "synthetic colored rectangles"
    model = yolo26n(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)
    before = evaluate(model, validation, args.batch_size, num_classes)
    started, history, best_ap, best_epoch, best_state = time.monotonic(), [], -1.0, 0, None
    steps = 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(training), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), args.batch_size):
            images, targets = [], []
            for index in order[start : start + args.batch_size]:
                image, source = training[index]
                target = {key: value.clone() for key, value in source.items()}
                if torch.rand((), generator=generator) < 0.5:
                    image = image.flip(-1)
                    target["boxes"][:, [0, 2]] = 1 - target["boxes"][:, [2, 0]]
                images.append(image)
                targets.append(target)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model(torch.stack(images), targets).values())
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at step {steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics = evaluate(model, validation, args.batch_size, num_classes)
        history.append({"epoch": epoch + 1, "training_loss": sum(losses) / len(losses), "validation": metrics})
        if metrics["ap50"] > best_ap:
            best_ap, best_epoch, best_state = metrics["ap50"], epoch + 1, copy.deepcopy(model.state_dict())
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    validation_predictions = predict(model, validation, args.batch_size)
    after = evaluate(model, validation, args.batch_size, num_classes, validation_predictions)
    calibration = None
    if args.calibrate:
        candidates = [
            operating_point(model, validation, args.batch_size, num_classes, validation_predictions, threshold)
            for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ]
        chosen = max(candidates, key=itemgetter("f1", "threshold"))
        calibration = {
            "selection": "Maximum validation F1; ties select the higher threshold",
            "validation": chosen,
            "candidates": candidates,
        }
    # Select weights and the optional score threshold before opening the test split.
    test_predictions = predict(model, testing, args.batch_size)
    test = evaluate(model, testing, args.batch_size, num_classes, test_predictions)
    if calibration is not None:
        calibration["test"] = operating_point(
            model, testing, args.batch_size, num_classes, test_predictions, calibration["validation"]["threshold"]
        )
    deployed = model.to_deploy()
    report = {
        "dataset": dataset,
        "data_revision": "swallan/PennFudanPed@ec1d4583fb436b14e2062587c8b28a5018668a5e" if args.data else None,
        "split": {"training": len(training), "validation": len(validation), "test": len(testing), "identifiers": names},
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "precision": "float32",
            "threads": args.threads,
        },
        "pretrained": False,
        "training_steps": steps,
        "best_epoch": best_epoch,
        "parameter_count": {
            "training": sum(parameter.numel() for parameter in model.parameters()),
            "deployment": sum(parameter.numel() for parameter in deployed.parameters()),
        },
        "before_validation": before,
        "after_validation": after,
        "test": test,
        "calibration": calibration,
        "finite_loss_and_gradients": True,
        "training_elapsed_seconds": time.monotonic() - started,
        "history": history,
        "limitations": "Small single-seed CPU check; resize distorts aspect ratios. AP50 at score>=0.05 is not COCO AP50:95. Full published training recipe and pretrained weights are not reproduced.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "num_classes": num_classes, "image_size": args.image_size}, args.checkpoint)
    print(json.dumps({"before": before, "after": after, "test": test, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="PennFudanPed directory; omit for rectangles")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--calibrate", action="store_true", help="Select a score threshold by validation F1 before testing"
    )
    parser.add_argument("--output", type=Path, default=Path("yolo26-check.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    arguments = parser.parse_args()
    if (
        min(arguments.epochs, arguments.batch_size, arguments.threads) < 1
        or arguments.image_size < 64
        or arguments.image_size % 32
    ):
        parser.error("Use positive epochs/batch/threads and an image size >=64 divisible by32")
    main(arguments)
