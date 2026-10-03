# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Common setup for the reference training scripts."""

import os
from argparse import ArgumentParser, Namespace
from typing import Any

from torch import nn
from torch.optim import SGD, AdamW, Optimizer, RAdam
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from ..optim import AdaBelief, AdamP, AdEMAMix


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
