import json
from collections import Counter

import pytest

from references.classification import prepare_workflow_comparison as subset


@pytest.fixture
def imagenette(tmp_path):
    source = tmp_path / "source"
    for label in range(10):
        for split, count in (("train", 100), ("val", 40)):
            directory = source / split / str(label)
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / f"{index:03}.jpg").write_bytes(f"{split}/{label}/{index}".encode())
    return source


def test_freezes_repeatable_balanced_disjoint_subsets_without_decoding(imagenette, tmp_path):
    first = subset.prepare(imagenette, tmp_path / "first")
    second = subset.prepare(imagenette, tmp_path / "second")
    manifest, repeated = [json.loads((path / "split-manifest.json").read_text()) for path in (first, second)]
    assert manifest["seed"] == 0
    assert manifest["source_paths"] == repeated["source_paths"]
    all_hashes = set()
    for split, per_class in (("train", 100), ("val", 20), ("test", 20)):
        frozen = manifest["splits"][split]
        samples = frozen["samples"]
        assert frozen["sha256"] == repeated["splits"][split]["sha256"]
        assert Counter(sample["class_index"] for sample in samples) == dict.fromkeys(range(10), per_class)
        hashes = {sample["sha256"] for sample in samples}
        assert not all_hashes.intersection(hashes)
        all_hashes.update(hashes)
    with pytest.raises(FileExistsError):
        subset.prepare(imagenette, first)


def test_rejects_cross_split_content_overlap_without_publishing(imagenette, tmp_path):
    for image in (imagenette / "val/0").glob("*.jpg"):
        image.write_bytes(b"repeated bytes")
    output = tmp_path / "subset"
    with pytest.raises(ValueError, match="overlapping sample content"):
        subset.prepare(imagenette, output)
    assert not output.exists()


def test_preserves_destination_created_during_preparation(imagenette, tmp_path, monkeypatch):
    output = tmp_path / "subset"
    write = subset.write_json

    def competing_write(path, value):
        write(path, value)
        output.mkdir()

    monkeypatch.setattr(subset, "write_json", competing_write)
    with pytest.raises(FileExistsError):
        subset.prepare(imagenette, output)
    assert not list(output.iterdir())
