# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib import import_module
from typing import Any

from torch import nn

from .checkpoints import Checkpoint

__all__ = ["ModelInfo", "get_model", "get_model_info", "list_checkpoints", "list_models"]

_TASKS = ("classification", "detection", "recognition", "segmentation")


@dataclass(frozen=True)
class ModelInfo:
    """Task and built-in weight availability for a public model factory."""

    name: str
    task: str
    pretrained: bool


@cache
def _model_factories() -> dict[str, tuple[str, Callable[..., nn.Module]]]:
    factories = {}
    for task in _TASKS:
        module = import_module(f"{__package__}.{task}")
        for name, factory in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(factory):
                continue
            if not factory.__module__.startswith(f"{module.__name__}."):
                continue
            return_type = inspect.get_annotations(factory, eval_str=True).get("return")
            if not inspect.isclass(return_type) or not issubclass(return_type, nn.Module):
                continue
            if name in factories:
                raise ValueError(f"duplicate model factory: {name}")
            factories[name] = (task, factory)
    return factories


def _factory(name: str) -> tuple[str, Callable[..., nn.Module]]:
    try:
        return _model_factories()[name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {name}") from exc


def list_checkpoints(name: str) -> tuple[Checkpoint, ...]:
    """List existing typed checkpoints for a model, without downloading weights.

    Args:
        name: public model factory name

    Returns:
        Checkpoints in declaration order; empty for models with only legacy metadata or no weights.
    """
    _, factory = _factory(name)
    module = import_module(factory.__module__)
    return tuple(
        member.value
        for enum in vars(module).values()
        if inspect.isclass(enum) and enum.__module__ == module.__name__ and issubclass(enum, Enum)
        for member in enum
        if isinstance(member.value, Checkpoint) and member.value.meta.arch == name
    )


def get_model_info(name: str) -> ModelInfo:
    """Return the task and built-in weight availability for a model.

    Args:
        name: public model factory name

    Returns:
        Metadata from the factory's typed checkpoints or legacy configuration.
    """
    task, factory = _factory(name)
    configs = getattr(import_module(factory.__module__), "default_cfgs", {})
    return ModelInfo(name, task, bool(list_checkpoints(name)) or bool(configs.get(name, {}).get("url")))


def list_models(task: str | None = None, pretrained: bool | None = None) -> list[str]:
    """List public model factories, optionally filtered by task and built-in weights.

    Args:
        task: optional model task
        pretrained: optional built-in weight availability

    Returns:
        Sorted factory names matching both filters.

    Raises:
        ValueError: if the task is unknown
    """
    if task is not None and task not in _TASKS:
        raise ValueError(f"unknown task: {task}")
    return sorted(
        name
        for name, (model_task, _) in _model_factories().items()
        if (task is None or model_task == task)
        and (pretrained is None or get_model_info(name).pretrained is pretrained)
    )


def get_model(name: str, *, checkpoint: Checkpoint | None = None, **kwargs: Any) -> nn.Module:
    """Instantiate a model through its existing factory and checkpoint loader.

    Args:
        name: public model factory name
        checkpoint: optional typed checkpoint matching the model
        kwargs: arguments forwarded to the factory

    Returns:
        The requested model.

    Raises:
        TypeError: if checkpoint is not a Checkpoint
        ValueError: if the model is unknown or the checkpoint is incompatible
    """
    _, factory = _factory(name)
    if checkpoint is not None:
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError("checkpoint must be a Checkpoint")
        if checkpoint.meta.arch != name:
            raise ValueError(f"checkpoint architecture {checkpoint.meta.arch!r} does not match {name!r}")
        if "checkpoint" not in inspect.signature(factory).parameters:
            raise ValueError(f"model {name!r} does not accept typed checkpoints")
        kwargs["checkpoint"] = checkpoint
    return factory(**kwargs)
