# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Transformation for object detection"""

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class VOCTargetTransform:
    """Decode VOC boxes and labels for native paired transforms."""

    def __init__(self, classes):
        self.class_map = {label: idx for idx, label in enumerate(classes)}

    def __call__(self, image, target):
        # Format boxes properly
        boxes = torch.tensor(
            [
                [
                    int(obj["bndbox"]["xmin"]),
                    int(obj["bndbox"]["ymin"]),
                    int(obj["bndbox"]["xmax"]),
                    int(obj["bndbox"]["ymax"]),
                ]
                for obj in target["annotation"]["object"]
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        # Encode class labels
        labels = torch.tensor([self.class_map[obj["name"]] for obj in target["annotation"]["object"]], dtype=torch.long)

        return image, {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=tuple(F.get_size(image))),
            "labels": labels,
        }


def convert_to_relative(image, target):
    """Convert boxes into the normalized coordinates expected by Holocron.

    Returns:
        The image and target with clipped relative box coordinates.
    """
    boxes = target["boxes"].as_subclass(torch.Tensor)
    height, width = F.get_size(image)
    boxes = (boxes / boxes.new_tensor([width, height, width, height])).clamp(0, 1)
    return image, {**target, "boxes": boxes}
