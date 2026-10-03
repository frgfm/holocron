# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Reproducible CPU learning check for full RepViT-M0.9 on real handwritten digits.

This is a small-dataset implementation check, not an ImageNet reproduction. The
original 8x8 digits are resized to 32x32 and repeated across three input channels.
Only validation loss selects the checkpoint; the test partition is evaluated once.
"""

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from holocron.models.classification import repvit_m0_9

DIGITS_URL = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/sklearn/datasets/data/digits.csv.gz"
DIGITS_SHA256 = "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"


def load_digits(cache_dir: Path):
    path = cache_dir / "digits.csv.gz"
    if path.is_file():
        payload = path.read_bytes()
    else:
        with urlopen(DIGITS_URL, timeout=30) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != DIGITS_SHA256:
        raise ValueError("digits dataset SHA-256 mismatch; remove the corrupt cached file and retry")
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    data = np.loadtxt(gzip.decompress(payload).splitlines(), delimiter=",", dtype=np.float32)
    if data.shape != (1797, 65):
        raise ValueError("unexpected digits dataset shape")
    # The pixel scale is specified by the dataset, not estimated from held-out data.
    images = torch.from_numpy(data[:, :64].copy()).reshape(-1, 1, 8, 8) / 8 - 1
    images = F.interpolate(images, size=(32, 32), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    labels = torch.from_numpy(data[:, -1].astype(np.int64))
    return images, labels


def stratified_split(labels: torch.Tensor, seed: int):
    """Split each class into 60% train, 20% validation, and the remainder test.

    Returns:
        Three disjoint tensors of sample indices covering the input dataset.

    Raises:
        ValueError: if any class contains fewer than five examples.
    """
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[], [], []]
    for label in labels.unique().tolist():
        indices = np.flatnonzero(labels.numpy() == label)
        if len(indices) < 5:
            raise ValueError("each class needs at least five examples for a three-way split")
        rng.shuffle(indices)
        train_end = int(0.6 * len(indices))
        validation_end = train_end + int(0.2 * len(indices))
        for partition, selected in zip(partitions, np.split(indices, [train_end, validation_end]), strict=True):
            partition.extend(selected.tolist())
    return tuple(torch.tensor(rng.permutation(indices), dtype=torch.long) for indices in partitions)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    if not count:
        raise ValueError("cannot evaluate an empty partition")
    return {"loss": loss_sum / count, "accuracy": correct / count, "correct": correct, "count": count}


def run(args: argparse.Namespace):
    if min(args.epochs, args.threads) < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError("epochs, threads, and learning rate must be positive; batch size must be at least two")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    images, labels = load_digits(args.cache_dir)
    train_indices, validation_indices, test_indices = stratified_split(labels, args.seed)
    if args.batch_size > len(train_indices):
        raise ValueError("batch size cannot exceed the training partition")
    loaders = [
        DataLoader(
            TensorDataset(images[indices], labels[indices]),
            batch_size=args.batch_size,
            shuffle=index == 0,
            drop_last=index == 0,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for index, indices in enumerate((train_indices, validation_indices, test_indices))
    ]
    train_loader, validation_loader, test_loader = loaders
    model = repvit_m0_9(pretrained=False, num_classes=10)
    initial_stem = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 10
    )
    baseline = evaluate(model, validation_loader)
    print(f"Initial validation: {baseline}", flush=True)
    best_loss, best_epoch = float("inf"), 0
    history = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_labels)
            correct += (logits.detach().argmax(1) == batch_labels).sum().item()
            count += len(batch_labels)
        validation = evaluate(model, validation_loader)
        if not math.isfinite(validation["loss"]):
            raise RuntimeError(f"non-finite validation loss in epoch {epoch}")
        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "training": {"loss": loss_sum / count, "accuracy": correct / count, "count": count},
            "validation": validation,
            "seconds": time.perf_counter() - epoch_started,
        })
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed}, args.checkpoint)
        scheduler.step()
        print(json.dumps(history[-1]), flush=True)
    training_seconds = time.perf_counter() - training_started

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    # This is the only test evaluation. Neither stopping nor checkpoint selection uses it.
    test_metrics = evaluate(model, test_loader)
    selected_validation = history[best_epoch - 1]["validation"]
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = (
        next(
            (
                line.partition(":")[2].strip()
                for line in cpu_info.read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if cpu_info.is_file()
        else platform.processor()
    )
    results = {
        "experiment": "RepViT-M0.9 full-model supervised CPU learning check on handwritten digits",
        "scope": "Not an ImageNet reproduction or a deployment benchmark; random example split, not writer-disjoint.",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "UCI optical recognition of handwritten digits (scikit-learn packaged 1797-example subset)",
            "source_url": DIGITS_URL,
            "source_sha256": DIGITS_SHA256,
            "preprocessing": "8x8 grayscale / 8 - 1; bilinear resize to 32x32; repeat to three channels",
            "unique_images": int(np.unique(images[:, 0].flatten(1).numpy(), axis=0).shape[0]),
            "split": {
                name: {
                    "count": len(indices),
                    "class_counts": torch.bincount(labels[indices], minlength=10).tolist(),
                    "indices_sha256": hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest(),
                }
                for name, indices in zip(
                    ("train", "validation", "test"), (train_indices, validation_indices, test_indices), strict=True
                )
            },
        },
        "configuration": {
            "architecture": "repvit_m0_9",
            "num_classes": 10,
            "input_shape": [3, 32, 32],
            "pretrained": False,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "drop_last_training_batch": True,
            "optimizer": "AdamW",
            "initial_learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "scheduler": "cosine decay to one tenth of initial learning rate",
            "augmentation": "none",
            "checkpoint_selection": "minimum validation cross-entropy over the fixed epoch budget",
        },
        "baseline_validation": baseline,
        "history": history,
        "selected_epoch": best_epoch,
        "selected_validation": selected_validation,
        "final_test": test_metrics,
        "test_evaluations": 1,
        "selected_stem_weight_mean_absolute_change": (next(model.parameters()).detach() - initial_stem)
        .abs()
        .mean()
        .item(),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        },
        "runtime": {
            "training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "device": "cpu",
            "dtype": "float32",
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "command": [sys.executable, *sys.argv],
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Selected epoch {best_epoch}; final test: {test_metrics}; metrics: {args.metrics}", flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/repvit-digits"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/repvit-digits.pth"))
    parser.add_argument("--metrics", type=Path, default=Path("references/classification/results/repvit-digits.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
