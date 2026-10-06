# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Native paired transforms with VOC mask typing and right/bottom padding."""

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F

Resize = T.Resize
RandomHorizontalFlip = T.RandomHorizontalFlip


class Compose(T.Compose):
    """Type the semantic mask while retaining its two spatial axes."""

    def forward(self, image, target):
        mask = tv_tensors.Mask(target)
        if mask.ndim == 3:
            mask = tv_tensors.Mask(mask.squeeze(0))
        return super().forward(image, mask)


def RandomResize(min_size, max_size=None, interpolation=InterpolationMode.BILINEAR):  # noqa: N802
    """Use native resizing, including the fixed-size reference configuration.

    Returns:
        A native fixed or random resize transform.
    """
    if max_size is None or min_size == max_size:
        return T.Resize(min_size, interpolation=interpolation)
    return T.RandomResize(min_size, max_size, interpolation=interpolation)


class RandomCrop(T.RandomCrop):
    """Pad the right and bottom with ignored mask labels before a native crop."""

    def forward(self, image, target):
        height, width = F.get_size(image)
        padding = [0, 0, max(self.size[1] - width, 0), max(self.size[0] - height, 0)]
        if any(padding):
            image, target = T.Pad(padding, fill={tv_tensors.Mask: 255})(image, target)
        return super().forward(image, target)


class ToTensor(T.Compose):
    """Return plain float images and integer masks without scaling class labels."""

    def __init__(self):
        super().__init__([
            T.ToImage(),
            T.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
            T.ToPureTensor(),
        ])
