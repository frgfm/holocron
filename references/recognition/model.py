# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""A shared character backbone and compact CNN/BiGRU CTC recognizer."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharacterBackbone(nn.Module):
    """Preserve baseline information and horizontal resolution for transfer to OCR."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = 1
        for output, pool in [(24, (2, 2)), (48, (2, 2)), (96, (2, 1)), (96, None)]:
            layers.extend([nn.Conv2d(channels, output, 3, padding=1, bias=False), nn.BatchNorm2d(output), nn.ReLU()])
            if pool is not None:
                layers.append(nn.MaxPool2d(pool))
            channels = output
        self.features = nn.Sequential(*layers)

    def forward(self, images):
        return self.features(images)


class CharacterClassifier(nn.Module):
    """Pretrain the very same visual features used by the sequence recognizer."""

    def __init__(self, classes):
        super().__init__()
        self.backbone = CharacterBackbone()
        self.head = nn.Linear(384, classes)

    def forward(self, images, lengths):
        features = self.backbone(images)
        mask = torch.arange(features.shape[-1], device=images.device)[None, :] < lengths.to(images.device)[:, None]
        pooled = (features * mask[:, None, None, :]).sum(-1) / lengths.to(images.device)[:, None, None]
        return self.head(pooled.flatten(1))


class CTCRecognizer(nn.Module):
    """Read variable-width lines using visual context and CTC, with no lexicon."""

    def __init__(self, classes):
        super().__init__()
        self.backbone = CharacterBackbone()
        self.projection = nn.Sequential(nn.Linear(384, 128), nn.ReLU())
        self.context = nn.GRU(128, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(192, classes + 1)

    def forward(self, images, lengths):
        features = self.backbone(images).permute(0, 3, 1, 2).flatten(2)
        packed = pack_padded_sequence(self.projection(features), lengths.cpu(), batch_first=True, enforce_sorted=False)
        context, _ = self.context(packed)
        context, _ = pad_packed_sequence(context, batch_first=True, total_length=features.shape[1])
        return self.head(context).log_softmax(-1).transpose(0, 1)
