# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Training script for semantic segmentation"""

import datetime
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.datasets import VOCSegmentation
from torchvision.models import segmentation as tv_segmentation
from torchvision.transforms import v2 as T
from torchvision.transforms.v2.functional import InterpolationMode, to_pil_image

if __package__:
    from .transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomResize, Resize, ToTensor
else:
    from transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomResize, Resize, ToTensor

import holocron
from holocron.models import segmentation
from holocron.trainer import SegmentationTrainer
from holocron.trainer._reference import add_loading_args, create_loader, create_optimizer
from holocron.utils.misc import find_image_size

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def plot_samples(images, targets, ignore_index=None):
    # Unnormalize image
    nb_samples = min(4, len(images))
    _, axes = plt.subplots(2, nb_samples, figsize=(5 * nb_samples, 5), squeeze=False)
    for idx in range(nb_samples):
        img = images[idx].clone()
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = to_pil_image(img)
        target = targets[idx].clone()
        if isinstance(ignore_index, int):
            target[target == ignore_index] = 0

        axes[0][idx].imshow(img)
        axes[0][idx].axis("off")
        axes[0][idx].set_title("Input image")
        axes[1][idx].imshow(target)
        axes[1][idx].axis("off")
        axes[1][idx].set_title("Target")
    plt.show()


def plot_predictions(images, preds, targets, ignore_index=None):
    # Unnormalize image
    nb_samples = min(4, len(images))
    _, axes = plt.subplots(3, nb_samples, figsize=(5 * nb_samples, 5), squeeze=False)
    for idx in range(nb_samples):
        img = images[idx].clone()
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = to_pil_image(img)
        # Target
        target = targets[idx].clone()
        if isinstance(ignore_index, int):
            target[target == ignore_index] = 0
        # Prediction
        pred = preds[idx].detach().cpu().argmax(dim=0)

        axes[0][idx].imshow(img)
        axes[0][idx].axis("off")
        axes[0][idx].set_title("Input image")
        axes[1][idx].imshow(target)
        axes[1][idx].axis("off")
        axes[1][idx].set_title("Target")
        axes[2][idx].imshow(pred)
        axes[2][idx].axis("off")
        axes[2][idx].set_title("Prediction")
    plt.show()


def main(args):
    print(args)

    if args.img_size < 16 or args.batch_size < 1 or args.grad_acc < 1:
        raise ValueError("Image size must be at least 16; batch size and gradient accumulation must be positive")
    if args.test_only and any((args.show_samples, args.show_preds, args.find_lr, args.check_setup, args.find_size)):
        raise ValueError("--test-only cannot be combined with training-data actions")
    source = segmentation if args.source == "holocron" else tv_segmentation
    if args.arch not in source.__dict__ or not callable(source.__dict__[args.arch]):
        raise ValueError(f"Unknown {args.source} segmentation architecture: {args.arch}")
    if args.wb:
        import wandb  # noqa: PLC0415

    torch.backends.cudnn.benchmark = True

    # Data loading
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop_size = args.img_size
    base_size = round(1.25 * crop_size)
    min_size, max_size = int(0.5 * base_size), int(2.0 * base_size)

    interpolation_mode = InterpolationMode.BILINEAR

    train_loader, val_loader = None, None
    if not args.test_only:
        st = time.time()
        train_set = VOCSegmentation(
            args.data_path,
            image_set="train",
            download=not (Path(args.data_path) / "VOCdevkit" / "VOC2012").is_dir(),
            transforms=Compose([
                RandomResize(min_size, max_size, interpolation_mode),
                RandomCrop(crop_size),
                RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
                ToTensor(),
                normalize,
            ]),
        )

        # Suggest size
        if args.find_size:
            print("Looking for optimal image size")
            train_set.transforms = None
            find_image_size(train_set)
            return

        train_loader = create_loader(
            train_set,
            args,
            training=True,
            drop_last=args.source == "torchvision",
            pin_memory=isinstance(args.device, int),
        )

        print(
            f"Training set loaded in {time.time() - st:.2f}s ({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target, ignore_index=255)
        return

    if not (args.find_lr or args.check_setup):
        st = time.time()
        val_set = VOCSegmentation(
            args.data_path,
            image_set="val",
            download=not (Path(args.data_path) / "VOCdevkit" / "VOC2012").is_dir(),
            transforms=Compose([
                Resize((crop_size, crop_size), interpolation_mode),
                ToTensor(),
                normalize,
            ]),
        )

        val_loader = create_loader(
            val_set, args, training=False, drop_last=False, pin_memory=isinstance(args.device, int)
        )

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    num_classes = len(VOC_CLASSES) * (3 if args.loss == "mc" else 1)
    if args.source == "holocron":
        model = segmentation.__dict__[args.arch](pretrained=args.pretrained, num_classes=num_classes)
    else:
        model = tv_segmentation.__dict__[args.arch](
            weights="DEFAULT" if args.pretrained else None,
            weights_backbone=None,
            num_classes=num_classes,
        )

    # Loss setup
    loss_weight = None
    if args.bg_factor != 1:
        loss_weight = torch.ones(len(VOC_CLASSES))
        loss_weight[0] = args.bg_factor
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255, label_smoothing=args.label_smoothing)
    elif args.loss == "focal":
        criterion = holocron.nn.FocalLoss(weight=loss_weight, ignore_index=255)
    elif args.loss == "mc":
        criterion = holocron.nn.MutualChannelLoss(weight=loss_weight, ignore_index=255, xi=3)

    # Optimizer setup
    optimizer = create_optimizer(model, args)

    log_wb = wandb.log if args.wb else None
    trainer = SegmentationTrainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.device,
        args.output_file,
        num_classes=len(VOC_CLASSES),
        gradient_acc=args.grad_acc,
        amp=args.amp,
        on_epoch_end=log_wb,
    )
    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        trainer.load(checkpoint)

    if args.show_preds:
        x, target = next(iter(train_loader))
        with torch.no_grad():
            x, target = trainer.to_cuda(x, target)
            trainer.model.eval()
            _, preds = trainer._get_loss(x, target, return_logits=True)
        plot_predictions(x.cpu(), preds.cpu(), target.cpu(), ignore_index=255)
        return

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(trainer._eval_metrics_str(eval_metrics))
        return

    if args.find_lr:
        print("Looking for optimal LR")
        trainer.find_lr(args.freeze_until, norm_weight_decay=args.norm_wd, num_it=min(len(train_loader), 100))
        trainer.plot_recorder()
        return

    if args.check_setup:
        print("Checking batch overfitting")
        trainer.check_setup(args.freeze_until, args.lr, norm_weight_decay=args.norm_wd, num_it=100)
        return

    # Training monitoring
    current_time = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}-{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:
        run = wandb.init(
            name=exp_name,
            project="holocron-semantic-segmentation",
            config={
                "learning_rate": args.lr,
                "scheduler": args.sched,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "source": args.source,
                "input_size": args.img_size,
                "optimizer": args.opt,
                "dataset": "Pascal VOC2012 Segmentation",
                "loss": args.loss,
            },
        )

    print("Start training")
    start_time = time.time()
    trainer.fit_n_epochs(args.epochs, args.lr, args.freeze_until, args.sched, norm_weight_decay=args.norm_wd)
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time_str}")

    if args.wb:
        run.finish()


def get_parser():
    parser = ArgumentParser(description="Holocron Segmentation Training", formatter_class=ArgumentDefaultsHelpFormatter)

    # Data & model
    group = parser.add_argument_group("Data & model")
    group.add_argument("data_path", type=str, help="path to dataset folder")
    group.add_argument("--arch", default="unet", type=str, help="architecture to use")
    group.add_argument("--source", choices=("holocron", "torchvision"), default="holocron", help="model source")
    group.add_argument("--pretrained", action="store_true", help="Use pre-trained models from the modelzoo")
    group.add_argument("--output-file", default="./checkpoints/model.pth", help="path where to save")
    group.add_argument("--resume", default="", help="resume from checkpoint")
    add_loading_args(parser)
    # Transformations
    group = parser.add_argument_group("Transformations")
    group.add_argument("--img-size", default=256, type=int, help="training crop and validation image size")
    # Optimization
    group = parser.add_argument_group("Optimization")
    group.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")
    group.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    group.add_argument("--freeze-until", default=None, type=str, help="Last layer to freeze")
    group.add_argument("--grad-acc", default=1, type=int, help="Number of batches to accumulate the gradient of")
    group.add_argument("--opt", default="adamp", choices=("sgd", "radam", "adamp", "adabelief"), help="optimizer")
    group.add_argument("--loss", default="crossentropy", choices=("crossentropy", "focal", "mc"), help="loss")
    group.add_argument("--bg-factor", default=1, type=float, help="Class weight of background in the loss")
    group.add_argument("--sched", default="onecycle", choices=("onecycle", "cosine"), help="Scheduler to be used")
    group.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    group.add_argument("--norm-wd", default=None, type=float, help="weight decay of norm parameters")
    group.add_argument("--label-smoothing", default=0.1, type=float, help="label smoothing")
    # Actions
    group = parser.add_argument_group("Actions")
    group.add_argument("--find-lr", action="store_true", help="Should you run LR Finder")
    group.add_argument("--find-size", dest="find_size", action="store_true", help="Should you run Image size Finder")
    group.add_argument("--check-setup", action="store_true", help="Check your training setup")
    group.add_argument("--show-samples", action="store_true", help="Whether training samples should be displayed")
    group.add_argument("--test-only", help="Only test the model", action="store_true")
    group.add_argument("--show-preds", action="store_true", help="Whether one batch predictions should be displayed")
    # Experiment tracking
    group = parser.add_argument_group("Experiment tracking")
    group.add_argument("--wb", action="store_true", help="Log to Weights & Biases")
    group.add_argument("--track-emissions", action="store_true", help="Track emissions with optional CodeCarbon")
    group.add_argument("--name", type=str, default=None, help="Name of your training experiment")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.track_emissions:
        from codecarbon import track_emissions

        track_emissions()(main)(args)
    else:
        main(args)
