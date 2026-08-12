# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


import contextlib
import io

import torch
from torch import Tensor
from torchvision.ops.boxes import box_iou

from .core import Trainer

__all__ = ["DetectionTrainer", "detection_average_precision"]


def detection_average_precision(  # noqa: PLR0912, PLR0915
    predictions: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
) -> dict[str, float | None]:
    """Compute COCO AP for normalized ``xyxy`` detections.

    Image IDs are assigned from list order. Boxes therefore need no pixel-size
    metadata: IoU and the all-area AP metrics are scale invariant.

    Returns:
        AP at IoU 0.50 and mean AP at IoUs 0.50 through 0.95. Both are ``None``
            when the inputs contain no targets.

    Raises:
        ImportError: if pycocotools is not installed
        ValueError: if predictions and targets do not follow the detection format
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of images")

    labels: set[int] = set()
    num_targets = 0
    num_predictions = 0
    for records, with_scores in ((targets, False), (predictions, True)):
        for record in records:
            required = {"boxes", "labels", *(("scores",) if with_scores else ())}
            if missing := required.difference(record):
                raise ValueError(f"missing detection fields: {', '.join(sorted(missing))}")
            boxes = record["boxes"]
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("boxes must have shape (N, 4)")
            record_labels = record["labels"]
            scores = record["scores"] if with_scores else None
            if record_labels.ndim != 1 or (scores is not None and scores.ndim != 1):
                raise ValueError("labels and scores must have shape (N,)")
            if record_labels.shape[0] != boxes.shape[0] or (scores is not None and scores.shape[0] != boxes.shape[0]):
                raise ValueError("boxes, labels and scores must describe the same number of detections")
            if torch.is_floating_point(record_labels) or torch.is_complex(record_labels):
                raise ValueError("labels must use an integer dtype")
            if torch.any(boxes[:, 2:] < boxes[:, :2]):
                raise ValueError("boxes must use xyxy order")
            labels.update(record_labels.detach().cpu().tolist())
            if with_scores:
                num_predictions += boxes.shape[0]
            else:
                num_targets += boxes.shape[0]

    if num_targets == 0:
        return {"ap50": None, "ap50_95": None}
    if num_predictions == 0:
        return {"ap50": 0.0, "ap50_95": 0.0}

    try:
        from pycocotools.coco import COCO  # ty: ignore[unresolved-import]  # noqa: PLC0415
        from pycocotools.cocoeval import COCOeval  # ty: ignore[unresolved-import]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "COCO AP evaluation requires pycocotools; install the evaluation extra with "
            "`pip install pylocron[evaluation]`."
        ) from exc

    category_ids = {label: idx for idx, label in enumerate(sorted(labels), start=1)}
    images = [{"id": idx, "width": 1, "height": 1} for idx in range(1, len(targets) + 1)]
    annotations: list[dict[str, int | float | list[float]]] = []
    results: list[dict[str, int | float | list[float]]] = []
    annotation_id = 1

    for image_id, (prediction, target) in enumerate(zip(predictions, targets, strict=True), start=1):
        for box, label in zip(
            target["boxes"].detach().cpu().tolist(), target["labels"].detach().cpu().tolist(), strict=True
        ):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[label],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1

        for box, label, score in zip(
            prediction["boxes"].detach().cpu().tolist(),
            prediction["labels"].detach().cpu().tolist(),
            prediction["scores"].detach().cpu().tolist(),
            strict=True,
        ):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": image_id,
                "category_id": category_ids[label],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        ground_truth = COCO()
        ground_truth.dataset = {
            "info": {},
            "images": images,
            "categories": [{"id": category_id} for category_id in category_ids.values()],
            "annotations": annotations,
        }
        ground_truth.createIndex()
        coco_eval = COCOeval(ground_truth, ground_truth.loadRes(results), "bbox")
        coco_eval.params.imgIds = [image["id"] for image in images]
        coco_eval.params.catIds = list(category_ids.values())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return {"ap50": float(coco_eval.stats[1]), "ap50_95": float(coco_eval.stats[0])}


def assign_iou(gt_boxes: Tensor, pred_boxes: Tensor, iou_threshold: float = 0.5) -> tuple[Tensor, Tensor]:
    """Assigns boxes by IoU

    Args:
        gt_boxes: ground truth boxes
        pred_boxes: predicted boxes
        iou_threshold: IoU threshold for assignment

    Returns:
        tuple of ground truth indices and predicted indices
    """
    iou = box_iou(gt_boxes, pred_boxes)
    iou = iou.max(dim=1)
    gt_kept = iou.values >= iou_threshold
    kept_pred_indices = iou.indices[gt_kept]
    assign_unique = torch.unique(kept_pred_indices)
    kept_gt_indices = torch.arange(gt_boxes.shape[0], device=gt_boxes.device)[gt_kept]
    # Filter
    if kept_pred_indices.shape[0] == assign_unique.shape[0]:
        return kept_gt_indices, kept_pred_indices

    gt_indices, pred_indices = [], []
    for pred_idx in assign_unique:
        candidates = kept_pred_indices == pred_idx
        selection = iou.values[gt_kept][candidates].argmax()
        gt_indices.append(kept_gt_indices[candidates][selection])
        pred_indices.append(pred_idx)
    return torch.stack(gt_indices), torch.stack(pred_indices)


class DetectionTrainer(Trainer):
    """Object detection trainer class.

    Args:
        model: model to train
        train_loader: training loader
        val_loader: validation loader
        criterion: loss criterion
        optimizer: parameter optimizer
        gpu: index of the GPU to use
        output_file: path where checkpoints will be saved
        amp: whether to use automatic mixed precision
        skip_nan_loss: whether the optimizer step should be skipped when the loss is NaN
        nan_tolerance: number of consecutive batches with NaN loss before stopping the training
        gradient_acc: number of batches to accumulate the gradient of before performing the update step
        gradient_clip: the gradient clip value
        on_epoch_end: callback triggered at the end of an epoch
    """

    @staticmethod
    def _to_cuda(  # type: ignore[override]
        x: list[Tensor], target: list[dict[str, Tensor]]
    ) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
        x = [x_.cuda(non_blocking=True) for x_ in x]
        target = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in target]
        return x, target

    def _get_loss(self, x: list[Tensor], target: list[dict[str, Tensor]]) -> Tensor:  # type: ignore[override]
        # AMP
        if self.amp:
            with torch.amp.autocast("cuda"):
                # Forward & loss computation
                loss_dict = self.model(x, target)
                return sum(loss_dict.values())
        # Forward & loss computation
        loss_dict = self.model(x, target)
        return sum(loss_dict.values())

    @staticmethod
    def _eval_metrics_str(eval_metrics: dict[str, float | None]) -> str:
        loc_str = f"{eval_metrics['loc_err']:.2%}" if isinstance(eval_metrics["loc_err"], float) else "N/A"
        clf_str = f"{eval_metrics['clf_err']:.2%}" if isinstance(eval_metrics["clf_err"], float) else "N/A"
        det_str = f"{eval_metrics['det_err']:.2%}" if isinstance(eval_metrics["det_err"], float) else "N/A"
        ap50_str = f"{eval_metrics['ap50']:.2%}" if isinstance(eval_metrics["ap50"], float) else "N/A"
        ap_str = f"{eval_metrics['ap50_95']:.2%}" if isinstance(eval_metrics["ap50_95"], float) else "N/A"
        return (
            f"Loc error: {loc_str} | Clf error: {clf_str} | Det error: {det_str} | AP50: {ap50_str} | AP50:95: {ap_str}"
        )

    @torch.inference_mode()
    def evaluate(self, iou_threshold: float = 0.5) -> dict[str, float | None]:
        """Evaluate the model on the validation set.

        Args:
            iou_threshold: IoU threshold for pair assignment

        Returns:
            evaluation metrics (validation loss, localization error rate, classification error rate, detection error rate,
            AP50 and AP50:95)
        """
        self.model.eval()

        loc_assigns = 0
        correct, clf_error, loc_fn, loc_fp, num_samples = 0, 0, 0, 0, 0
        ap_predictions: list[dict[str, Tensor]] = []
        ap_targets: list[dict[str, Tensor]] = []

        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            if self.amp:
                with torch.amp.autocast("cuda"):
                    detections = self.model(x)
            else:
                detections = self.model(x)

            for dets, t in zip(detections, target, strict=True):
                ap_predictions.append({key: dets[key].detach().cpu() for key in ("boxes", "labels", "scores")})
                ap_targets.append({key: t[key].detach().cpu() for key in ("boxes", "labels")})
                if t["boxes"].shape[0] > 0 and dets["boxes"].shape[0] > 0:
                    gt_indices, pred_indices = assign_iou(t["boxes"], dets["boxes"], iou_threshold)
                    loc_assigns += len(gt_indices)
                    correct_ = (t["labels"][gt_indices] == dets["labels"][pred_indices]).sum().item()
                else:
                    gt_indices, pred_indices = [], []
                    correct_ = 0
                correct += correct_
                clf_error += len(gt_indices) - correct_
                loc_fn += t["boxes"].shape[0] - len(gt_indices)
                loc_fp += dets["boxes"].shape[0] - len(pred_indices)
            num_samples += sum(t["boxes"].shape[0] for t in target)

        nb_preds = num_samples - loc_fn + loc_fp
        # Localization
        loc_err = 1 - 2 * loc_assigns / (nb_preds + num_samples) if nb_preds + num_samples > 0 else None
        # Classification
        clf_err = 1 - correct / loc_assigns if loc_assigns > 0 else None
        # End-to-end
        det_err = 1 - 2 * correct / (nb_preds + num_samples) if nb_preds + num_samples > 0 else None
        return {
            "loc_err": loc_err,
            "clf_err": clf_err,
            "det_err": det_err,
            "val_loss": loc_err,
            **detection_average_precision(ap_predictions, ap_targets),
        }
