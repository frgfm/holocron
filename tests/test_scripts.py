from collections import OrderedDict

import pytest
import torch

from scripts import eval_latency, export_to_onnx


@pytest.mark.parametrize("bundle", [False, True])
def test_load_model_checkpoint(monkeypatch, tmp_path, bundle):
    reference = torch.nn.Linear(2, 1)
    state_dict = OrderedDict((name, value.clone()) for name, value in reference.state_dict().items())
    checkpoint = {"model": state_dict, "epoch": 3} if bundle else state_dict
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    def tiny_factory(*, pretrained):
        assert pretrained is False
        return torch.nn.Linear(2, 1)

    monkeypatch.setattr(export_to_onnx.models, "tiny", tiny_factory, raising=False)
    restored = export_to_onnx.load_model("tiny", checkpoint=checkpoint_path)

    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, state_dict[name])


def test_export_model_uses_dynamo_and_dynamic_batch(monkeypatch, tmp_path):
    exported = object()
    call = {}

    def fake_export(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return exported

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    output = export_to_onnx.export_model(
        torch.nn.Identity(),
        torch.rand(1, 3, 4, 4),
        tmp_path / "model.onnx",
        verify=True,
        report=True,
    )

    assert output is exported
    assert call["kwargs"]["dynamo"] is True
    assert call["kwargs"]["verify"] is True
    assert call["kwargs"]["report"] is True
    assert call["kwargs"]["dynamic_shapes"][0][0].__name__ == "batch"


def test_run_evaluation_synchronizes_cuda(monkeypatch):
    class Measurement:
        mean = 0.002
        median = 0.0015
        iqr = 0.0005
        raw_times = (0.01, 0.02)
        number_per_run = 10

    class Timer:
        def __init__(self, **kwargs):
            self.num_threads = kwargs["num_threads"]

        def blocked_autorange(self, *, min_run_time):
            assert self.num_threads == 2
            assert min_run_time == 0.1
            return Measurement()

    calls = []
    monkeypatch.setattr(eval_latency.benchmark, "Timer", Timer)
    monkeypatch.setattr(torch.cuda, "synchronize", calls.append)

    result = eval_latency.run_evaluation(
        lambda: None,
        device=torch.device("cuda:0"),
        min_run_time=0.1,
        warmup_it=2,
        num_threads=2,
    )

    assert calls == [torch.device("cuda:0")]
    assert result == {
        "mean_ms": 2.0,
        "median_ms": 1.5,
        "iqr_ms": 0.5,
        "measurements": 2,
        "runs_per_measurement": 10,
        "threads": 2,
    }
