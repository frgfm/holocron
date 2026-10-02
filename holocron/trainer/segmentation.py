# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, cast

import torch
from torch import Tensor

from ..nn import MutualChannelLoss
from .core import Trainer

__all__ = ["SegmentationTrainer"]


class SegmentationTrainer(Trainer):
    """Semantic segmentation trainer class.

    Args:
        *args: args of [`Trainer`][holocron.trainer.core.Trainer]
        num_classes: number of output classes
        **kwargs: keyword args of [`Trainer`][holocron.trainer.core.Trainer]
    """

    def __init__(self, *args: Any, num_classes: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def _get_loss(self, x: Tensor, target: Tensor, return_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.train(self.model.training)
        with torch.amp.autocast("cuda", enabled=self.amp):
            outputs = self.model(x)
            outputs = outputs if isinstance(outputs, dict) else {"out": outputs}
            valid = target != getattr(self.criterion, "ignore_index", 255)
            losses = {
                name: cast(Tensor, self.criterion(logits, target)) if valid.any() else logits.sum() * 0
                for name, logits in outputs.items()
            }
            loss = losses["out"]
            if "aux" in losses:
                loss += 0.5 * losses["aux"]
            out = outputs["out"]
            if isinstance(self.criterion, MutualChannelLoss):
                out = out.reshape(out.shape[0], self.num_classes, self.criterion.xi, *out.shape[2:]).amax(dim=2)
        return (loss, out) if return_logits else loss

    @torch.inference_mode()
    def evaluate(self, ignore_index: int | None = None) -> dict[str, float]:
        """Evaluate the model on the validation set

        Args:
            ignore_index: target value to exclude from metrics; defaults to the criterion's ignore_index

        Returns:
            evaluation metrics (validation loss, global accuracy, mean IoU)

        Raises:
            ValueError: if validation has no batches or no finite loss
        """
        self.model.eval()

        ignore_index = getattr(self.criterion, "ignore_index", 255) if ignore_index is None else ignore_index
        val_loss, num_valid_samples, num_batches, num_labeled_batches = 0.0, 0, 0, 0
        conf_mat = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=next(self.model.parameters()).device
        )
        for x, target in self.val_loader:
            num_batches += 1
            x, target = self.to_cuda(x, target)

            loss, out = self._get_loss(x, target, return_logits=True)  # ty: ignore[invalid-argument-type]
            labeled = (target != getattr(self.criterion, "ignore_index", 255)).any()
            num_labeled_batches += int(labeled)

            # Safeguard for NaN loss
            if torch.isfinite(loss) and labeled:
                val_loss += loss.item() * x.shape[0]
                num_valid_samples += x.shape[0]

            # borrowed from https://github.com/pytorch/vision/blob/master/references/segmentation/train.py
            pred = out.argmax(dim=1).flatten()
            target = target.flatten()
            k = (target >= 0) & (target < self.num_classes) & (target != ignore_index)
            inds = self.num_classes * target[k].to(torch.int64) + pred[k]
            nc = self.num_classes
            conf_mat += torch.bincount(inds, minlength=nc**2).reshape(nc, nc)

        if num_batches == 0 or (num_labeled_batches > 0 and num_valid_samples == 0):
            raise ValueError("Validation requires at least one batch with a finite loss")
        val_loss /= max(1, num_valid_samples)
        true_positive = torch.diag(conf_mat)
        union = conf_mat.sum(1) + conf_mat.sum(0) - true_positive
        present = union > 0
        acc_global = (true_positive.sum() / conf_mat.sum().clamp_min(1)).item()
        mean_iou = (true_positive[present] / union[present]).mean().item() if present.any() else 0.0

        return {"val_loss": val_loss, "acc_global": acc_global, "mean_iou": mean_iou}

    @staticmethod
    def _eval_metrics_str(eval_metrics: dict[str, float]) -> str:
        return (
            f"Validation loss: {eval_metrics['val_loss']:.4} "
            f"(Acc: {eval_metrics['acc_global']:.2%} | Mean IoU: {eval_metrics['mean_iou']:.2%})"
        )
