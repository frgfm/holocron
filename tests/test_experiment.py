import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from holocron import trainer
from holocron.trainer import experiment


class _TinyTrainer(trainer.Trainer):
    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        losses = [self.criterion(self.model(x), target).item() for x, target in self.val_loader]
        self.model.train()
        return {"val_loss": sum(losses) / len(losses)}

    @staticmethod
    def _eval_metrics_str(eval_metrics: dict[str, float]) -> str:
        return f"Validation loss: {eval_metrics['val_loss']:.4f}"


def _make_trainer(output_file: Path, on_epoch_end=None) -> _TinyTrainer:
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    target = x.sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(x, target), batch_size=2, shuffle=True)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.2), nn.Linear(4, 1))
    return _TinyTrainer(
        model,
        loader,
        loader,
        nn.MSELoss(),
        torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
        output_file=str(output_file),
        on_epoch_end=on_epoch_end,
    )


def _assert_optimizer_equal(left: torch.optim.Optimizer, right: torch.optim.Optimizer) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter, state in left_state["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right_state["state"][parameter][key], rtol=0, atol=0)


@pytest.mark.parametrize("sched_type", ["cosine", "onecycle"])
def test_interrupted_run_resumes_exactly(tmp_path: Path, sched_type: str) -> None:
    def consume_rng(_metrics):
        torch.rand(())

    torch.manual_seed(0)
    baseline = _make_trainer(tmp_path / "baseline.pth", consume_rng)
    initial_state = deepcopy(baseline.model.state_dict())
    training_rng = torch.get_rng_state()
    baseline_result = baseline.fit_n_epochs(2, 0.05, sched_type=sched_type)

    def stop_after_first_epoch(_metrics):
        torch.rand(())
        raise RuntimeError("stop")

    torch.manual_seed(999)
    interrupted = _make_trainer(tmp_path / "interrupted.pth", stop_after_first_epoch)
    interrupted.model.load_state_dict(initial_state)
    torch.set_rng_state(training_rng)
    with pytest.raises(RuntimeError, match="stop"):
        interrupted.fit_n_epochs(2, 0.05, sched_type=sched_type)

    resumed = _make_trainer(tmp_path / "resumed.pth")
    resumed.load(torch.load(interrupted.output_file, map_location="cpu", weights_only=True))
    resumed.output_file = str(tmp_path / "resumed.pth")
    run_dir = tmp_path / "run"
    resumed_result = resumed.fit_n_epochs(1, 0.05, sched_type=sched_type, run_dir=str(run_dir))

    for key, parameter in baseline.model.state_dict().items():
        torch.testing.assert_close(parameter, resumed.model.state_dict()[key], rtol=0, atol=0)
    _assert_optimizer_equal(baseline.optimizer, resumed.optimizer)
    assert baseline.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert (baseline_result.epoch, baseline_result.step) == (resumed_result.epoch, resumed_result.step)
    assert baseline_result.metrics == resumed_result.metrics
    assert (run_dir / "manifest.json").is_file()


def test_checkpoint_v2_is_atomic_and_legacy_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    learner = _make_trainer(tmp_path / "checkpoint.pth")
    learner.save(learner.output_file)
    state = torch.load(learner.output_file, map_location="cpu", weights_only=True)
    assert state["schema_version"] == 2
    assert {"model", "optimizer", "scheduler", "scaler", "rng_state", "config"} <= state.keys()

    legacy = {"epoch": 3, "step": 7, "min_loss": 0.4, "model": deepcopy(learner.model.state_dict())}
    learner.load(legacy)
    assert (learner.start_epoch, learner.epoch, learner.step, learner.min_loss) == (3, 3, 7, 0.4)

    output = Path(learner.output_file)
    previous = output.read_bytes()

    def fail_save(_state, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="write failed"):
        learner.save(learner.output_file)
    assert output.read_bytes() == previous
    assert not list(tmp_path.glob(".checkpoint.pth.*.tmp"))


def test_run_bundle_writes_relative_hashes_and_manifest_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "source.pth"
    checkpoint.write_bytes(b"checkpoint")
    artifact = tmp_path / "predictions.json"
    artifact.write_text("{}", encoding="utf-8")
    result = trainer.RunResult(
        epoch=2,
        step=4,
        best_metric=0.25,
        metrics=({"epoch": 1, "step": 2, "val_loss": 0.5}, {"epoch": 2, "step": 4, "val_loss": 0.25}),
        config={"run": {"lr": 0.01}},
        checkpoint=str(checkpoint),
    )
    bundle = tmp_path / "bundle"
    writes: list[str] = []
    original_write = experiment._atomic_write

    def record_write(path: Path, content: str) -> None:
        if path.name == "manifest.json":
            assert (bundle / "metrics.jsonl").is_file()
            assert (bundle / "checkpoints" / checkpoint.name).is_file()
            assert (bundle / "artifacts" / artifact.name).is_file()
        writes.append(path.name)
        original_write(path, content)

    monkeypatch.setattr(experiment, "_atomic_write", record_write)
    manifest_path = trainer.write_run_bundle(bundle, result, [artifact])
    assert writes[-1] == "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    for bundled_artifact in manifest["artifacts"]:
        path = bundle / bundled_artifact["path"]
        assert not Path(bundled_artifact["path"]).is_absolute()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == bundled_artifact["sha256"]

    with pytest.raises(FileNotFoundError):
        trainer.write_run_bundle(bundle, result, [tmp_path / "missing.json"])
    assert not manifest_path.exists()
