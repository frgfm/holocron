# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Any

from torch import nn

from .checkpoints import Checkpoint, Maturity

__all__ = ["Maturity", "ModelInfo", "get_model", "get_model_info", "list_checkpoints", "list_models"]


@dataclass(frozen=True)
class ModelInfo:
    """Public metadata for a model factory."""

    name: str
    task: str
    maturity: Maturity
    pretrained: bool


_TASKS = ("classification", "detection", "segmentation")


@cache
def _task_modules() -> dict[str, ModuleType]:
    return {task: import_module(f"{__package__}.{task}") for task in _TASKS}


@cache
def _model_factories() -> dict[str, tuple[str, Callable[..., nn.Module]]]:
    factories: dict[str, tuple[str, Callable[..., nn.Module]]] = {}
    for task, module in _task_modules().items():
        for name, factory in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(factory):
                continue
            if not factory.__module__.startswith(f"{module.__name__}."):
                continue
            return_type = inspect.get_annotations(factory, eval_str=True).get("return")
            if not inspect.isclass(return_type) or not issubclass(return_type, nn.Module):
                continue
            if name in factories:
                raise RuntimeError(f"duplicate model factory: {name}")
            factories[name] = (task, factory)
    return factories


@cache
def _checkpoint_map() -> dict[str, tuple[Checkpoint, ...]]:
    checkpoints: dict[str, list[Checkpoint]] = {}
    modules = {factory.__module__ for _, factory in _model_factories().values()}
    for module_name in modules:
        module = import_module(module_name)
        for enum_cls in vars(module).values():
            if not inspect.isclass(enum_cls) or enum_cls.__module__ != module_name or not issubclass(enum_cls, Enum):
                continue
            for member in enum_cls:
                if isinstance(member.value, Checkpoint):
                    checkpoints.setdefault(member.value.meta.arch, []).append(member.value)
    return {name: tuple(values) for name, values in checkpoints.items()}


def _has_legacy_weights(name: str, factory: Callable[..., nn.Module]) -> bool:
    configs = getattr(import_module(factory.__module__), "default_cfgs", {})
    config = configs.get(name, {}) if isinstance(configs, dict) else {}
    return isinstance(config, dict) and bool(config.get("url"))


def get_model_info(name: str) -> ModelInfo:
    """Return task, maturity, and weight availability for a model.

    Args:
        name: public model factory name.

    Returns:
        Metadata discovered from the public factory and its checkpoints.

    Raises:
        ValueError: if the model name is unknown.
    """
    try:
        task, factory = _model_factories()[name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {name}") from exc

    checkpoints = _checkpoint_map().get(name, ())
    pretrained = bool(checkpoints) or _has_legacy_weights(name, factory)
    if task == "detection":
        maturity = Maturity.EXPERIMENTAL
    elif task == "segmentation" or not checkpoints:
        maturity = Maturity.PREVIEW
    else:
        maturity = min(checkpoints, key=lambda checkpoint: list(Maturity).index(checkpoint.maturity)).maturity
    return ModelInfo(name=name, task=task, maturity=maturity, pretrained=pretrained)


def list_models(
    task: str | None = None,
    maturity: Maturity | str | None = None,
    pretrained: bool | None = None,
) -> list[str]:
    """List public model factories, optionally filtered by evidence and task.

    Args:
        task: optional model task.
        maturity: optional evidence maturity.
        pretrained: optional pretrained-weight availability.

    Returns:
        Sorted model factory names matching every filter.

    Raises:
        ValueError: if the task or maturity is unknown.
    """
    if task is not None and task not in _TASKS:
        raise ValueError(f"unknown task: {task}")
    if maturity is not None:
        maturity = Maturity(maturity)

    names = []
    for name in _model_factories():
        info = get_model_info(name)
        if task is not None and info.task != task:
            continue
        if maturity is not None and info.maturity != maturity:
            continue
        if pretrained is not None and info.pretrained is not pretrained:
            continue
        names.append(name)
    return sorted(names)


def list_checkpoints(name: str) -> tuple[Checkpoint, ...]:
    """List typed checkpoints declared for a model.

    Args:
        name: public model factory name.

    Returns:
        Checkpoints whose metadata identifies the requested architecture.
    """
    get_model_info(name)
    return _checkpoint_map().get(name, ())


def get_model(name: str, *, checkpoint: Checkpoint | None = None, **kwargs: Any) -> nn.Module:
    """Instantiate a public model by name.

    Args:
        name: public model factory name.
        checkpoint: optional typed checkpoint matching the model.
        kwargs: arguments forwarded to the model factory.

    Returns:
        Instantiated model.

    Raises:
        TypeError: if checkpoint is not checkpoint metadata.
        ValueError: if the name is unknown or the checkpoint is incompatible.
    """
    try:
        _, factory = _model_factories()[name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {name}") from exc

    if checkpoint is not None:
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError("checkpoint must be a Checkpoint")
        if checkpoint.meta.arch != name:
            raise ValueError(f"checkpoint architecture {checkpoint.meta.arch!r} does not match {name!r}")
        if "checkpoint" not in inspect.signature(factory).parameters:
            raise ValueError(f"model {name!r} does not accept typed checkpoints")
        kwargs["checkpoint"] = checkpoint
    return factory(**kwargs)
