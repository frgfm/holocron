# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

# ruff: noqa: T201
"""Common setup for the reference training scripts."""

import datetime
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Any

import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer, RAdam
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from ..optim import AdaBelief, AdamP, AdEMAMix
from .core import Trainer


def load_checkpoint(trainer: Trainer, path: str) -> None:
    """Load the requested reference checkpoint before a task-specific action."""
    if path:
        print(f"Resuming {path}")
        trainer.load(torch.load(path, map_location="cpu", weights_only=True))


def run_training(
    trainer: Trainer,
    args: Namespace,
    *,
    project: str,
    config: dict[str, Any],
    setup_iterations: int | None = None,
    **scheduler_kwargs: Any,
) -> None:
    """Run the common reference action and finish optional experiment tracking."""
    if args.test_only:
        print("Running evaluation")
        print(trainer._eval_metrics_str(trainer.evaluate()))  # noqa: SLF001
        return
    if args.find_lr:
        print("Looking for optimal LR")
        trainer.find_lr(
            args.freeze_until,
            start_lr=getattr(args, "find_lr_start", 1e-7),
            end_lr=getattr(args, "find_lr_end", 1),
            norm_weight_decay=args.norm_wd,
            num_it=min(len(trainer.train_loader), 100),
        )
        trainer.plot_recorder()
        return
    if args.check_setup:
        print("Checking batch overfitting")
        trainer.check_setup(
            args.freeze_until,
            args.lr,
            norm_weight_decay=args.norm_wd,
            num_it=min(len(trainer.train_loader), 100) if setup_iterations is None else setup_iterations,
        )
        return

    run = None
    if args.wb:
        import wandb  # ty: ignore[unresolved-import]  # noqa: PLC0415

        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
        name = f"{args.arch}-{timestamp}" if args.name is None else args.name
        run = wandb.init(name=name, project=project, config=config)
        trainer.on_epoch_end = wandb.log

    print("Start training")
    start_time = time.time()
    try:
        trainer.fit_n_epochs(
            args.epochs, args.lr, args.freeze_until, args.sched, norm_weight_decay=args.norm_wd, **scheduler_kwargs
        )
        print(f"Training time {datetime.timedelta(seconds=int(time.time() - start_time))}")
    finally:
        if run is not None:
            run.finish()


def create_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """Create the reference optimizer with the task's learning rate and weight decay.

    Returns:
        Optimizer over the model's trainable parameters.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    optimizers = {
        "sgd": SGD,
        "radam": RAdam,
        "adamw": AdamW,
        "adamp": AdamP,
        "adabelief": AdaBelief,
        "ademamix": AdEMAMix,
    }
    if args.opt not in optimizers:
        raise ValueError(f"Unknown optimizer: {args.opt}")
    options = {}
    if args.opt == "sgd":
        options["momentum"] = getattr(args, "momentum", 0.9)
    elif args.opt != "adamw":
        options.update(betas=(0.95, 0.99, 0.9999) if args.opt == "ademamix" else (0.95, 0.99), eps=1e-6)
    return optimizers[args.opt](
        [p for p in model.parameters() if p.requires_grad], args.lr, weight_decay=args.weight_decay, **options
    )


def create_loader(dataset: Dataset, args: Namespace, *, training: bool, **kwargs: Any) -> DataLoader:
    """Create a reference loader, retaining task-specific collation and memory options.

    Returns:
        Loader with random training or sequential validation sampling.
    """
    options = {"drop_last": training, "pin_memory": True, **kwargs}
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset) if training else SequentialSampler(dataset),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        **options,
    )


def add_loading_args(parser: ArgumentParser) -> None:
    """Add the common hardware and data-loading argument groups."""
    group = parser.add_argument_group("Hardware")
    group.add_argument("--device", default=None, type=int, help="device")
    group.add_argument("--amp", help="Use Automatic Mixed Precision", action="store_true")
    group = parser.add_argument_group("Data loading")
    group.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    group.add_argument(
        "-j", "--workers", default=min(os.cpu_count(), 16), type=int, help="number of data loading workers"
    )
