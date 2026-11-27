import json

import pytest

from holocron.experiments import sha256
from references.classification import compare_workflow as comparison


def test_freezing_requires_both_completed_trials_on_the_frozen_data(tmp_path):
    hashes = {"train": "a", "val": "b", "test": "c"}
    selection = tmp_path / "selections.json"
    for name in comparison.MODELS:
        directory = tmp_path / name
        directory.mkdir()
        checkpoint = directory / "checkpoint.pth"
        checkpoint.write_bytes(name.encode())
        path = directory / "result.json"
        record = {
            "state": "completed",
            "exit_code": 0,
            "epoch": 3,
            "dataset": {"train": "a", "validation": "b", "test": "c"},
            "models": [{"checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint), "selected_epoch": 1}],
        }
        path.write_text(json.dumps(record))
    record["state"] = "running"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="all three epochs"):
        comparison.freeze_selections(tmp_path, hashes)
    record["state"] = "completed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="did not use the frozen splits"):
        comparison.freeze_selections(tmp_path, {**hashes, "test": "changed"})
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        comparison.freeze_selections(tmp_path, hashes)
    assert not selection.exists()
    checkpoint.write_bytes(original)
    comparison.freeze_selections(tmp_path, hashes)
    assert json.loads(selection.read_text())["models"].keys() == comparison.MODELS.keys()
