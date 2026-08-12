# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from torch import Tensor, nn


class _ClassifierMixin:
    """Common feature-extraction interface for sequential classifiers."""

    def _head(self) -> tuple[str, nn.Module]:
        for name in ("head", "classifier"):
            if name in self._modules:
                return name, self._modules[name]
        raise RuntimeError("classifier has no head module")

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the spatial representation produced by the backbone.

        Args:
            x: input tensor

        Returns:
            spatial feature tensor
        """
        return self.features(x)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        """Pool features and optionally return the embedding before classification.

        Args:
            x: spatial feature tensor
            pre_logits: whether to return the embedding before the final classifier

        Returns:
            logits or pre-logits embedding
        """
        name, head = self._head()
        if list(self._modules).index(name) < list(self._modules).index("pool"):
            if pre_logits:
                return self.pool(x)
            return self.pool(head(x))

        x = self.pool(x)
        if isinstance(head, nn.Sequential):
            modules = list(head.children())
            for module in modules[:-1]:
                x = module(x)
            return x if pre_logits else modules[-1](x)
        return x if pre_logits else head(x)

    def get_classifier(self) -> nn.Module:
        """Return the final classification module.

        Returns:
            final linear or convolutional classification module
        """
        _, head = self._head()
        if isinstance(head, nn.Sequential):
            return head[-1]
        return head

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the final classification module.

        Args:
            num_classes: number of outputs for the new classifier

        Raises:
            TypeError: if the existing classifier is not linear or convolutional
        """
        name, head = self._head()
        classifier = self.get_classifier()
        if isinstance(classifier, nn.Linear):
            replacement: nn.Module = nn.Linear(
                classifier.in_features,
                num_classes,
                bias=classifier.bias is not None,
                device=classifier.weight.device,
                dtype=classifier.weight.dtype,
            )
        elif isinstance(classifier, nn.Conv2d):
            replacement = nn.Conv2d(
                classifier.in_channels,
                num_classes,
                classifier.kernel_size,
                classifier.stride,
                classifier.padding,
                classifier.dilation,
                classifier.groups,
                bias=classifier.bias is not None,
                padding_mode=classifier.padding_mode,
                device=classifier.weight.device,
                dtype=classifier.weight.dtype,
            )
        else:
            raise TypeError(f"unsupported classifier type: {type(classifier).__name__}")
        replacement.train(classifier.training)

        if isinstance(head, nn.Sequential):
            head[-1] = replacement
        else:
            setattr(self, name, replacement)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))
