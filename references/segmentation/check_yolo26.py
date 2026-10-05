# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Train YOLO26 semantic segmentation on independent synthetic train/validation images."""

import copy
import json
import platform
import time
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from holocron.models.segmentation import yolo26n_sem
from holocron.trainer import SegmentationTrainer


def make_shapes(count, image_size, seed):
    generator = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing="ij")
    images, targets = [], []
    colors = torch.tensor([[0.16, 0.16, 0.18], [0.85, 0.20, 0.15], [0.15, 0.80, 0.25]])
    for _ in range(count):
        target = torch.zeros(image_size, image_size, dtype=torch.long)
        left, top = torch.randint(4, image_size // 3, (2,), generator=generator).tolist()
        width, height = torch.randint(image_size // 3, image_size // 2, (2,), generator=generator).tolist()
        target[top : top + height, left : left + width] = 1
        center_x, center_y = torch.randint(image_size // 3, 3 * image_size // 4, (2,), generator=generator).tolist()
        radius = int(torch.randint(image_size // 6, image_size // 3, (1,), generator=generator))
        target[(xx - center_x).square() + (yy - center_y).square() <= radius**2] = 2
        brightness = 0.75 + 0.5 * torch.rand((), generator=generator)
        noise = 0.04 * torch.randn(3, image_size, image_size, generator=generator)
        image = (brightness * colors[target].permute(2, 0, 1) + noise).clamp(0, 1)
        target[:2] = 255
        images.append((image - 0.5) / 0.25)
        targets.append(target)
    return TensorDataset(torch.stack(images), torch.stack(targets))


def check_gradients(optimizer, _args, _kwargs):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError("Every trainable parameter must have a finite gradient")


def main(args):
    if min(args.train_images, args.val_images, args.batch_size, args.epochs) < 1 or args.image_size < 32:
        raise ValueError("Use positive counts and an image size of at least 32")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_seed, val_seed = args.seed + 1, args.seed + 10001
    train_data = make_shapes(args.train_images, args.image_size, train_seed)
    val_data = make_shapes(args.val_images, args.image_size, val_seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    model = yolo26n_sem(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer.register_step_pre_hook(check_gradients)
    history = []
    started = time.monotonic()
    with TemporaryDirectory(prefix="holocron-yolo26-sem-") as directory:
        learner = SegmentationTrainer(
            model,
            train_loader,
            val_loader,
            nn.CrossEntropyLoss(ignore_index=255),
            optimizer,
            num_classes=3,
            gradient_clip=5.0,
            output_file=str(Path(directory) / "best.pth"),
            on_epoch_end=lambda metrics: history.append(dict(metrics)),
        )
        initial = learner.evaluate()
        learner.fit_n_epochs(args.epochs, args.lr, sched_type="cosine", norm_weight_decay=0)
        final = learner.evaluate()
        learner.val_loader = DataLoader(train_data, batch_size=args.batch_size)
        train_metrics = learner.evaluate()
        final_path = Path(directory) / "final.pth"
        learner.save(str(final_path))
        restored = yolo26n_sem(num_classes=3).eval()
        restored.load_state_dict(torch.load(final_path, weights_only=True)["model"])
        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            learner.save(str(args.checkpoint))

    example = val_data.tensors[0][:1]
    with torch.inference_mode():
        expected = model(example)
        restored_error = (restored(example) - expected).abs().max().item()
        fused = copy.deepcopy(model).eval().fuse()
        fused_error = (fused(example) - expected).abs().max().item()
    reference = yolo26n_sem(num_classes=19).eval()
    train_params = sum(parameter.numel() for parameter in reference.parameters())
    reference.fuse()
    result = {
        "task": "synthetic semantic segmentation learning check; not a Cityscapes or ADE20K benchmark",
        "architecture": "yolo26n_sem",
        "pretrained": False,
        "device": "cpu",
        "precision": "float32",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": args.seed,
        "train_seed": train_seed,
        "validation_seed": val_seed,
        "train_images": args.train_images,
        "validation_images": args.val_images,
        "image_size": args.image_size,
        "classes": ["background", "rectangle", "circle"],
        "ignore_index": 255,
        "epochs": args.epochs,
        "steps": learner.step,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "scheduler": "cosine",
        "optimizer": "AdamW",
        "threads": args.threads,
        "initial_validation": initial,
        "final_validation": final,
        "final_training": train_metrics,
        "finite_gradients": True,
        "checkpoint_reload_max_abs_error": restored_error,
        "fused_max_abs_error": fused_error,
        "parameters_19_classes_training": train_params,
        "parameters_19_classes_fused": sum(parameter.numel() for parameter in reference.parameters()),
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)


def get_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("yolo26-semantic-check.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-images", type=int, default=48)
    parser.add_argument("--val-images", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
