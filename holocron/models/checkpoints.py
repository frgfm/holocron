# Copyright (C) 2023-2025, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import warnings
from dataclasses import dataclass
from enum import StrEnum

from torchvision.transforms.functional import InterpolationMode

__all__ = [
    "Checkpoint",
    "Dataset",
    "Evaluation",
    "LoadingMeta",
    "Metric",
    "PreProcessing",
    "PretrainedWeightsUnavailableWarning",
    "TrainingRecipe",
]


class PretrainedWeightsUnavailableWarning(UserWarning):
    """Warns that ``pretrained=True`` was requested but no checkpoint is available for that model.

    Silence it with ``warnings.filterwarnings("ignore", category=PretrainedWeightsUnavailableWarning)``.
    """


def _warn_pretrained_unavailable(model_name: str | None = None, *, stacklevel: int = 3) -> None:
    """Emit a single, actionable warning when no pretrained checkpoint is available.

    Centralizes the message used by both ``_handle_legacy_pretrained`` and
    ``holocron.models.utils.load_pretrained_params`` so the wording and links stay in sync.

    Args:
        model_name: name of the model class; included in the message when provided.
        stacklevel: forwarded to ``warnings.warn`` so the warning is attributed to the model builder
            (e.g. ``repvgg_a0``) rather than to a Holocron internal. The default of 3 suits the two
            call sites above; do not lower it.
    """
    target = f" for {model_name}" if model_name else ""
    warnings.warn(
        f"No pretrained weights are available{target}; the model will be randomly initialized. "
        "Browse the models that ship pretrained weights in the model zoo "
        "(https://frgfm.github.io/holocron/) or train your own with the reference scripts "
        "(https://github.com/frgfm/Holocron/tree/main/references).",
        PretrainedWeightsUnavailableWarning,
        stacklevel=stacklevel,
    )


@dataclass
class TrainingRecipe:
    """Implements a training recipe.

    Args:
        commit_hash: the commit that was used to train the model.
        args: the argument values that were passed to the reference script to train this.
    """

    commit: str | None
    script: str | None
    args: str | None


class Metric(StrEnum):
    """Evaluation metric"""

    TOP1_ACC = "top1-accuracy"
    TOP5_ACC = "top5-accuracy"


class Dataset(StrEnum):
    """Training/evaluation dataset"""

    IMAGENET1K = "imagenet-1k"
    IMAGENETTE = "imagenette"
    CIFAR10 = "cifar10"


@dataclass
class Evaluation:
    """Results of model evaluation"""

    dataset: Dataset
    results: dict[Metric, float]


@dataclass
class LoadingMeta:
    """Metadata to load the model"""

    url: str
    sha256: str
    size: int
    arch: str
    num_params: int
    categories: list[str]


@dataclass
class PreProcessing:
    """Preprocessing metadata for the model"""

    input_shape: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR


@dataclass
class Checkpoint:
    """Data required to run a model in the exact same condition than the checkpoint"""

    # What to expect
    evaluation: Evaluation
    # How to load it
    meta: LoadingMeta
    # How to use it
    pre_processing: PreProcessing
    # How to reproduce
    recipe: TrainingRecipe


def _handle_legacy_pretrained(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    default_checkpoint: Checkpoint | None = None,
) -> Checkpoint | None:
    checkpoint = checkpoint or (default_checkpoint if pretrained else None)

    if pretrained and checkpoint is None:
        _warn_pretrained_unavailable(stacklevel=3)

    return checkpoint
