import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SPEC = importlib.util.spec_from_file_location(
    "train_repvit_digits", Path(__file__).parents[1] / "references/classification/train_repvit_digits.py"
)
digits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(digits)


def test_stratified_split_is_disjoint_complete_balanced_and_reproducible():
    labels = torch.arange(10).repeat_interleave(21)
    partitions = digits.stratified_split(labels, seed=42)
    repeated = digits.stratified_split(labels, seed=42)
    for actual, expected, per_class in zip(partitions, repeated, (12, 4, 5), strict=True):
        torch.testing.assert_close(actual, expected)
        assert torch.bincount(labels[actual]).tolist() == [per_class] * 10
    combined = torch.cat(partitions)
    torch.testing.assert_close(combined.sort().values, torch.arange(len(labels)))
    assert not torch.equal(partitions[0], digits.stratified_split(labels, seed=43)[0])


def test_stratified_split_rejects_too_small_classes():
    with pytest.raises(ValueError, match="at least five"):
        digits.stratified_split(torch.tensor([0, 0, 0, 0]), seed=42)


def test_dataset_hash_rejects_corrupt_cached_payload(tmp_path):
    (tmp_path / "digits.csv.gz").write_bytes(b"corrupt")
    assert hashlib.sha256(b"corrupt").hexdigest() != digits.DIGITS_SHA256
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        digits.load_digits(tmp_path)


def test_evaluation_weights_unequal_batches_and_disables_gradients():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    model = nn.Identity()
    result = digits.evaluate(model, loader)
    assert result["loss"] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["correct"] == 2
    assert result["count"] == 3
    assert not model.training
