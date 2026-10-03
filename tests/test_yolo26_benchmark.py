import torch

from references.detection.check_yolo26 import evaluate, operating_point


def test_detection_metric_counts_duplicate_as_false_positive():
    target = {"boxes": torch.tensor([[0.1, 0.1, 0.6, 0.8]]), "labels": torch.tensor([0])}
    predictions = [
        {
            "boxes": torch.tensor([[0.1, 0.1, 0.6, 0.8], [0.1, 0.1, 0.6, 0.8], [0.7, 0.7, 0.9, 0.9]]),
            "labels": torch.tensor([0, 0, 0]),
            "scores": torch.tensor([0.95, 0.5, 0.2]),
        }
    ]
    samples = [(torch.zeros(3, 64, 64), target)]
    metrics = evaluate(None, samples, 1, 1, predictions)
    assert metrics["ap50"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["precision_at_score_005"] == 1 / 3
    selected = operating_point(None, samples, 1, 1, predictions, 0.9)
    assert selected["precision"] == selected["recall"] == selected["f1"] == 1


def test_detection_metric_empty_predictions():
    target = {"boxes": torch.tensor([[0.1, 0.1, 0.6, 0.8]]), "labels": torch.tensor([0])}
    predictions = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long), "scores": torch.empty(0)}]
    metrics = evaluate(None, [(torch.zeros(3, 64, 64), target)], 1, 1, predictions)
    assert metrics["ap50"] == metrics["precision_at_score_005"] == metrics["recall_at_score_005"] == 0


def test_detection_metric_counts_absent_classes_and_empty_scenes():
    box = torch.tensor([[0.1, 0.1, 0.6, 0.8]])
    target = {"boxes": box, "labels": torch.tensor([0])}
    prediction = {"boxes": box.repeat(2, 1), "labels": torch.tensor([0, 1]), "scores": torch.tensor([0.9, 0.9])}
    metrics = evaluate(None, [(torch.zeros(3, 64, 64), target)], 1, 2, [prediction])
    assert metrics["ap50"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["precision_at_score_005"] == 0.5
    target = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    metrics = evaluate(None, [(torch.zeros(3, 64, 64), target)], 1, 2, [prediction])
    assert metrics["false_positives"] == 2
    assert metrics["precision_at_score_005"] == metrics["recall_at_score_005"] == 0
