# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Small real-image training check with separate Penn-Fudan train, validation and test sets."""

import copy
import hashlib
import json
import platform
import time
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from holocron.models.segmentation import yolo26n_sem
from holocron.trainer import SegmentationTrainer


class Pedestrians(Dataset):
    def __init__(self, root, names, image_size, augment=False):
        self.names = names
        self.augment = augment
        self.samples = []
        for name in names:
            with Image.open(root / "PNGImages" / name) as source:
                image = TF.pil_to_tensor(
                    source.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                )
            with Image.open(root / "PedMasks" / f"{Path(name).stem}_mask.png") as source:
                mask = source.resize((image_size, image_size), Image.Resampling.NEAREST)
                target = torch.from_numpy((np.array(mask) > 0).astype(np.int64))
            self.samples.append((image.float() / 255, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, target = self.samples[index]
        if self.augment:
            if torch.rand(()) < 0.5:
                image, target = image.flip(-1), target.flip(-1)
            image = (image * (0.8 + 0.4 * torch.rand(()))).clamp(0, 1)
        return (image - 0.5) / 0.25, target


def split_images(root, seed):
    names = sorted(path.name for path in (root / "PNGImages").glob("*.png"))
    if len(names) != 170:
        raise ValueError("This check expects the 170-image Penn-Fudan dataset")
    indices = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [names[index] for index in indices]
    return {"train": shuffled[:120], "validation": shuffled[120:145], "test": shuffled[145:]}


def cpu_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    split = split_images(args.data, args.seed)
    training = Pedestrians(args.data, split["train"], args.image_size, augment=True)
    validation = Pedestrians(args.data, split["validation"], args.image_size)
    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    model = yolo26n_sem(num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    finite_gradients = True

    def check_gradients(optim, _args, _kwargs):
        nonlocal finite_gradients
        finite_gradients = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for group in optim.param_groups
            for parameter in group["params"]
        )
        if not finite_gradients:
            raise ValueError("Non-finite or missing parameter gradients")

    optimizer.register_step_pre_hook(check_gradients)
    history = []
    best_iou, best_epoch = -1.0, 0
    started = time.monotonic()
    with TemporaryDirectory(prefix="holocron-pennfudan-sem-") as directory:
        best_path = Path(directory) / "best-iou.pth"

        def select_checkpoint(metrics):
            nonlocal best_iou, best_epoch
            history.append(dict(metrics))
            if metrics["mean_iou"] > best_iou:
                best_iou, best_epoch = metrics["mean_iou"], len(history)
                learner.save(str(best_path))

        learner = SegmentationTrainer(
            model,
            train_loader,
            val_loader,
            nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]), ignore_index=255),
            optimizer,
            num_classes=2,
            gradient_clip=5.0,
            output_file=str(Path(directory) / "best-loss.pth"),
            on_epoch_end=select_checkpoint,
        )
        initial = learner.evaluate()
        learner.fit_n_epochs(args.epochs, args.lr, sched_type="cosine", norm_weight_decay=0)
        model.load_state_dict(torch.load(best_path, weights_only=True)["model"])
        checkpoint_sha256 = hashlib.sha256(best_path.read_bytes()).hexdigest()
        selected_validation = learner.evaluate()
        with torch.inference_mode():
            image = validation[0][0].unsqueeze(0)
            fused = copy.deepcopy(model).eval().fuse()
            fused_error = (fused(image) - model(image)).abs().max().item()
        # Test images are used only after the validation-selected model is fixed.
        test_data = Pedestrians(args.data, split["test"], args.image_size)
        learner.val_loader = DataLoader(test_data, batch_size=args.batch_size)
        test_metrics = learner.evaluate()
        conf_mat = torch.zeros(2, 2, dtype=torch.int64)
        with torch.inference_mode():
            for image, target in learner.val_loader:
                predicted = model(image).argmax(1)
                conf_mat += torch.bincount((2 * target + predicted).flatten(), minlength=4).reshape(2, 2)
        per_class_iou = conf_mat.diag() / (conf_mat.sum(0) + conf_mat.sum(1) - conf_mat.diag())
        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_bytes(best_path.read_bytes())

    result = {
        "task": "Penn-Fudan binary semantic segmentation; a local small-data check, not an official benchmark split",
        "dataset_repository": "https://github.com/swallan/PennFudanPed",
        "dataset_commit": "ec1d4583fb436b14e2062587c8b28a5018668a5e",
        "architecture": "yolo26n_sem",
        "pretrained": False,
        "device": "cpu",
        "precision": "float32",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "threads": args.threads,
        "seed": args.seed,
        "image_size": args.image_size,
        "split": split,
        "classes": ["background", "person"],
        "target": "all nonzero instance-mask labels merged into foreground",
        "augmentation": "training only: horizontal flip p=0.5, brightness factor uniform [0.8, 1.2]",
        "loss": "weighted cross entropy [1, 2] plus 0.5 times the auxiliary-head loss",
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "normalization_weight_decay": 0,
        "learning_rate": args.lr,
        "scheduler": "cosine",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps": learner.step,
        "selection": "highest validation mean IoU; test used only after validation-based model selection",
        "selected_epoch": best_epoch,
        "checkpoint_loaded_from_disk": True,
        "checkpoint_sha256": checkpoint_sha256,
        "fused_max_abs_error": fused_error,
        "initial_validation": initial,
        "selected_validation": selected_validation,
        "test": test_metrics,
        "test_per_class_iou": dict(zip(("background", "person"), per_class_iou.tolist(), strict=True)),
        "test_confusion_matrix": conf_mat.tolist(),
        "finite_gradients": finite_gradients,
        "parameters_2_classes_training": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"history", "split"}}, indent=2))


def get_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("yolo26-semantic-pennfudan.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
