import io
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from torchvision.transforms.v2.functional import InterpolationMode

from holocron.experiments import Trial, evaluation_preprocessing, imagefolder_manifest, sha256
from holocron.trainer import ClassificationTrainer
from references.classification import experiment, train
from scripts.eval_deployment import preprocess
from scripts.export_trial import export_trial


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    for offset, split in enumerate(("train", "val", "test")):
        for index, label in enumerate(("a", "b")):
            directory = tmp_path / split / label
            directory.mkdir(parents=True)
            for sample in range(2):
                Image.new("RGB", (20, 20), (offset * 60 + index * 20 + sample, 30, 90)).save(
                    directory / f"{sample}.png"
                )
    config = {
        "schema_version": 1,
        "initialization": None,
        "test_split": "test",
        "deployment_target": "owner's CPU",
        "recipe": {
            "data_path": ".",
            "device": "cpu",
            "epochs": 2,
            "batch_size": 2,
            "workers": 0,
            "sched": "cosine",
            "mixup_alpha": 0,
            "train_crop_size": 16,
            "val_crop_size": 16,
            "val_resize_size": 20,
        },
    }
    monkeypatch.setattr(
        train,
        "get_model",
        lambda *_args, **_kwargs: nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)),
    )
    return tmp_path / "experiment.json", config


def submit(path, config, trial="trial"):
    path.write_text(json.dumps(config), encoding="utf-8")
    experiment.run_experiment(path, path.parent / trial)
    return json.loads((path.parent / trial / "result.json").read_text())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "schema_version", True),
        (None, "schema_version", 2),
        (None, "wall_time_seconds", 10),
        ("recipe", "device", "meta"),
        ("recipe", "device", 0),
        (None, "initialization", "missing.pth"),
        (None, "deployment_target", ""),
        ("recipe", "seed", -1),
        ("recipe", "seed", 1.5),
        ("recipe", "epochs", 0),
        ("recipe", "epochs", True),
        ("recipe", "lr", float("nan")),
        ("recipe", "opt", "unknown"),
        ("recipe", "sched", "unknown"),
        ("recipe", "workers", -1),
        ("recipe", "amp", "yes"),
        ("recipe", "resume", "old.pth"),
        ("recipe", "random_erase", 2),
        ("recipe", "mean", [0, 0, 0]),
        ("recipe", "arch", "unknown"),
        (None, "test_split", ""),
    ],
)
def test_configuration_rejection(configuration, section, key, value):
    path, config = configuration
    (config if section is None else config[section])[key] = value
    with pytest.raises(ValueError):
        submit(path, config)
    assert not (path.parent / "trial").exists()


@pytest.mark.parametrize("scheduler", ["cosine", "onecycle"])
def test_success_and_independent_reader(configuration, monkeypatch, scheduler):
    path, config = configuration
    config["recipe"]["sched"] = scheduler
    original_getitem = ImageFolder.__getitem__

    def no_test_reads(dataset, index):
        assert dataset.root != str(path.parent / "test")
        return original_getitem(dataset, index)

    def worsening_metrics(learner):
        running = json.loads(Path(learner.output_file).with_name("result.json").read_text())
        assert (running["state"], running["actual_device"]) == ("running", "cpu")
        return {"val_loss": float(learner.epoch), "acc1": 0.5, "acc5": 0.0}

    monkeypatch.setattr(ImageFolder, "__getitem__", no_test_reads)
    monkeypatch.setattr(ClassificationTrainer, "evaluate", worsening_metrics)
    result = submit(path, config)
    directory = path.parent / "trial"
    resolved, provenance, data = [
        json.loads((directory / f"{name}.json").read_text()) for name in ("config", "provenance", "data")
    ]
    events = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    model = result["models"][0]
    assert resolved["recipe"]["lr"] == train.get_parser().get_default("lr")
    assert resolved["recipe"]["data_path"] == str(path.parent)
    assert resolved["deployment_target"] == config["deployment_target"]
    assert data["class_to_idx"] == {"a": 0, "b": 1}
    assert all(len(split["samples"]) == 4 for split in data["splits"].values())
    assert provenance["data"]["train"] == data["splits"]["train"]["sha256"]
    assert provenance["source"]["revision"]
    assert type(provenance["source"]["dirty"]) is bool
    assert provenance["packages"]["torch"]
    assert "RandomResizedCrop" in provenance["preprocessing"]["train"]
    assert provenance["evaluation_preprocessing"] == {
        "input_shape": [1, 3, 16, 16],
        "resize": 20,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    assert len(provenance["initialization"]["state_dict_sha256"]) == 64
    assert (result["state"], result["exit_code"], result["actual_device"]) == ("completed", 0, "cpu")
    assert result["finished_at"]
    assert result["elapsed_seconds"] > 0
    assert [(event["epoch"], event["split"]) for event in events] == [(1, "validation"), (2, "validation")]
    assert result["metrics_split"] == "validation"
    assert result["metric_directions"] == {"val_loss": "minimize", "acc1": "maximize"}
    assert (model["final_epoch"], model["selected_epoch"]) == (2, 1)
    for field, epoch in (("final_metrics", -1), ("selected_metrics", 0)):
        assert model[field] == {key: events[epoch][key] for key in result["metric_directions"]}
    assert model["checkpoint_sha256"] == sha256(directory / model["checkpoint"])
    assert torch.load(directory / model["checkpoint"], map_location="cpu", weights_only=True)["epoch"] == 1
    assert model["fully_resumable"] is False
    config["initialization"] = str(directory / model["checkpoint"])
    submit(path, config, "warm-start")
    assert (
        json.loads((path.parent / "warm-start/provenance.json").read_text())["initialization"]["sha256"]
        == model["checkpoint_sha256"]
    )


@pytest.mark.parametrize("change", ["dtype", "interpolation", "lambda", "order", "fixed_resize", "normalization_order"])
def test_evaluation_preprocessing_rejects_ambiguous_transforms(change):
    transform = train.evaluation_transform(20, 16)
    if change == "dtype":
        transform.transforms[3].dtype = torch.float64
    elif change == "interpolation":
        transform.transforms[0].interpolation = InterpolationMode.NEAREST
    elif change == "lambda":
        transform.transforms[4].transforms[0] = T.Lambda(lambda image: image.contiguous())
    elif change == "fixed_resize":
        transform.transforms[0] = T.Resize((20, 20))
    elif change == "normalization_order":
        transform.transforms[4].transforms.reverse()
    else:
        transform.transforms[1], transform.transforms[2] = transform.transforms[2], transform.transforms[1]
    assert evaluation_preprocessing(transform) is None


@pytest.mark.parametrize("size", [(17, 30), (30, 17)])
def test_evaluation_transform_matches_deployment_preprocessing(size):
    image = Image.frombytes("RGB", size, bytes(value % 256 for value in range(size[0] * size[1] * 3)))
    transform = train.evaluation_transform(24, 16)
    prep = evaluation_preprocessing(transform)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    torch.testing.assert_close(
        transform(image), torch.from_numpy(preprocess(stream.getvalue(), prep)[0]), rtol=0, atol=0
    )


def _tiny_classifier():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))


@pytest.fixture
def completed_trial(configuration, monkeypatch):
    path, config = configuration

    def worsening_metrics(learner):
        return {"val_loss": float(learner.epoch), "acc1": 0.5, "acc5": 0.0}

    final_weights = {}
    fit = ClassificationTrainer.fit_n_epochs

    def fit_and_record(learner, *args, **kwargs):
        result = fit(learner, *args, **kwargs)
        final_weights.update({
            name: tensor.detach().cpu().clone() for name, tensor in learner.model.state_dict().items()
        })
        return result

    monkeypatch.setattr(ClassificationTrainer, "evaluate", worsening_metrics)
    monkeypatch.setattr(ClassificationTrainer, "fit_n_epochs", fit_and_record)
    result = submit(path, config)
    return path, path.parent / "trial", result, final_weights


def _assert_selected_bundle(root, trial, record, final_weights, model):
    bundle = json.loads((root / "bundle.json").read_text())
    data = json.loads((trial / "data.json").read_text())
    validation_split = data["splits"]["validation"]
    validation = validation_split["samples"]
    assert bundle["categories"] == ["a", "b"]
    assert [(sample["id"], sample["label"], sample["sha256"]) for sample in bundle["samples"]] == [
        (sample["path"], sample["class_index"], sample["sha256"]) for sample in validation
    ]
    bundled_checkpoint = bundle["provenance"]["checkpoint"]["path"]
    selected = torch.load(root / bundled_checkpoint, map_location="cpu", weights_only=True)
    assert selected["epoch"] == record["selected_epoch"] < record["final_epoch"]
    assert len(bundle["samples"]) == len(validation) == 4
    assert all(torch.equal(model.state_dict()[name], selected["model"][name]) for name in final_weights)

    selected_model = _tiny_classifier().eval()
    selected_model.load_state_dict(selected["model"])
    final_model = _tiny_classifier().eval()
    final_model.load_state_dict(final_weights)
    prep = json.loads((trial / "provenance.json").read_text())["evaluation_preprocessing"]
    transform = train.evaluation_transform(prep["resize"], prep["input_shape"][2])
    reference = np.load(root / bundle["reference"], allow_pickle=False)
    final_logits = []
    for index, sample in enumerate(validation):
        source = Path(validation_split["root"]) / sample["path"]
        with Image.open(source) as image:
            images = transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            torch.testing.assert_close(torch.from_numpy(reference[index]), selected_model(images)[0], rtol=0, atol=0)
            final_logits.append(final_model(images)[0].numpy())
    assert any(not np.array_equal(reference[index], logits) for index, logits in enumerate(final_logits))
    return bundle, validation


def _replay_bundle(root, output):
    bundle_path = root / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["protocol"] = {"warmup": 1, "iterations": 1, "repeats": 1, "chunk_size": 2}
    bundle_path.write_text(json.dumps(bundle))
    replay = subprocess.run(  # noqa: S603 -- use the frozen, locally built bundle runner
        [sys.executable, str(root / "scripts/eval_deployment.py"), str(root), str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    report = json.loads(next(output.glob("*/report.json")).read_text())
    accuracy = report["cases"]["onnx-fp32"]["accuracy"]
    assert report["complete"]
    assert accuracy["status"] == "completed"
    assert accuracy["source_agreement"] == 1
    assert accuracy["max_logit_difference"] < 1e-3


def test_completed_trial_exports_selected_checkpoint_and_replays(completed_trial):
    path, trial, result, final_weights = completed_trial
    test_images = {item: sha256(item) for item in (path.parent / "test").rglob("*.png")}
    record = result["models"][0]
    assert (record["selected_epoch"], record["final_epoch"]) == (1, 2)
    selected = torch.load(trial / record["checkpoint"], map_location="cpu", weights_only=True)
    assert any(not torch.equal(selected["model"][name], final_weights[name]) for name in final_weights)

    Image.new("RGB", (20, 20), (10, 20, 30)).save(path.parent / "val/a/late.png")
    model = _tiny_classifier()
    root = export_trial(trial, path.parent / "deployment-bundle", model=model)
    bundle, _ = _assert_selected_bundle(root, trial, record, final_weights, model)
    assert len(bundle["samples"]) == 4
    _replay_bundle(root, path.parent / "replay")
    assert {item: sha256(item) for item in test_images} == test_images


def test_trial_cli_exports_and_replays(configuration):
    path, config = configuration
    config["recipe"].update(arch="convnext_atto", epochs=1, train_crop_size=32, val_resize_size=36, val_crop_size=32)
    path.write_text(json.dumps(config))
    repository = Path(__file__).resolve().parents[1]
    trial, bundle = path.parent / "cli-trial", path.parent / "cli-bundle"
    output = path.parent / "cli-replay"
    for command in (
        [sys.executable, "-m", "references.classification.experiment", str(path), str(trial)],
        [sys.executable, str(repository / "scripts/export_trial.py"), str(trial), str(bundle)],
        [sys.executable, str(bundle / "scripts/eval_deployment.py"), str(bundle), str(output)],
    ):
        result = subprocess.run(  # noqa: S603 -- exercise the documented CLI with generated local data
            command,
            cwd=repository,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(next(output.glob("*/report.json")).read_text())
    assert report["complete"]
    accuracy = report["cases"]["onnx-fp32"]["accuracy"]
    assert accuracy["source_agreement"] == 1
    assert accuracy["max_logit_difference"] < 1e-3


@pytest.mark.parametrize("mutation", ["checkpoint", "image", "preprocessing", "changed-preprocessing", "running"])
def test_export_rejects_changed_inputs_and_ambiguous_transforms(completed_trial, mutation, monkeypatch, tmp_path):
    _, trial, result, _ = completed_trial
    selected = result["models"][0]
    model = _tiny_classifier()
    if mutation == "checkpoint":
        checkpoint = trial / selected["checkpoint"]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("loaded an unverified checkpoint"))
        error = "Selected checkpoint hash mismatch"
    elif mutation == "image":
        data = json.loads((trial / "data.json").read_text())
        validation = data["splits"]["validation"]
        source = Path(validation["root"]) / validation["samples"][0]["path"]
        source.write_bytes(source.read_bytes() + b"changed")
        error = "Validation image hash mismatch"
        monkeypatch.setattr(model, "forward", lambda *_args: pytest.fail("ran inference on an unverified image"))
    elif mutation in {"preprocessing", "changed-preprocessing"}:
        provenance_path = trial / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        if mutation == "preprocessing":
            provenance["evaluation_preprocessing"] = None
            error = "Unsupported or ambiguous evaluation preprocessing"
        else:
            provenance["evaluation_preprocessing"]["resize"] += 1
            error = "Unsupported or inconsistent evaluation preprocessing"
        provenance_path.write_text(json.dumps(provenance))
    else:
        result_path = trial / "result.json"
        record = json.loads(result_path.read_text())
        record["state"] = "running"
        result_path.write_text(json.dumps(record))
        error = "Export requires a completed trial"

    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match=error):
        export_trial(trial, destination, model=model)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("error", "state", "code", "after_epoch"),
    [
        (RuntimeError("training broke"), "failed", 1, False),
        (RuntimeError("training broke"), "failed", 1, True),
        (KeyboardInterrupt(), "interrupted", 130, False),
        (SystemExit(143), "interrupted", 143, False),
        (SystemExit(0), "interrupted", 0, False),
        (SystemExit(None), "interrupted", 0, False),
        (SystemExit("exit message"), "interrupted", 1, False),
    ],
)
def test_failure_and_interruption(configuration, monkeypatch, error, state, code, after_epoch):
    path, config = configuration
    fit = ClassificationTrainer.fit_n_epochs

    def fail(learner, _epochs, *args, **kwargs):
        learners.append(learner)
        if after_epoch:
            fit(learner, 1, *args, **kwargs)
        raise error

    learners = []
    monkeypatch.setattr(ClassificationTrainer, "fit_n_epochs", fail)
    with pytest.raises(type(error)):
        submit(path, config)
    result = json.loads((path.parent / "trial/result.json").read_text())
    assert (result["state"], result["exit_code"]) == (state, code)
    assert result["error"]["type"] == type(error).__name__
    assert result["error"]["message"] == str(error)
    assert "fail" in result["error"]["traceback"]
    assert result["finished_at"]
    assert learners[0].on_epoch_end is None
    if after_epoch:
        model = result["models"][0]
        assert model["selected_epoch"] == result["epoch"] == 1
        assert model["checkpoint_sha256"] == sha256(path.parent / "trial" / model["checkpoint"])
    else:
        assert "checkpoint" not in result["models"][0]
        assert not (path.parent / "trial/progress.jsonl").read_text()


@pytest.mark.parametrize("symlink", [False, True])
def test_overwrite_protection(configuration, symlink):
    path, config = configuration
    directory = path.parent / "trial"
    if symlink:
        directory.symlink_to(path.parent / "missing", target_is_directory=True)
    else:
        directory.mkdir()
        (directory / "result.json").write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError):
        submit(path, config)
    assert directory.is_symlink() if symlink else (directory / "result.json").read_text() == "do not replace"


@pytest.mark.parametrize("problem", ["overlap", "mapping", "content"])
def test_data_identity(configuration, problem):
    path, config = configuration
    splits = {name: ImageFolder(path.parent / split) for name, split in (("train", "train"), ("validation", "val"))}
    before = imagefolder_manifest(splits)
    if problem == "content":
        Image.new("RGB", (20, 20), (255, 0, 0)).save(path.parent / "train/a/0.png")
        assert imagefolder_manifest(splits)["splits"]["train"]["sha256"] != before["splits"]["train"]["sha256"]
    else:
        if problem == "overlap":
            shutil.copyfile(path.parent / "train/a/0.png", path.parent / "val/b/renamed.png")
        else:
            (path.parent / "val/b").rename(path.parent / "val/c")
        with pytest.raises(ValueError, match=r"overlapping|inconsistent"):
            submit(path, config)
        assert json.loads((path.parent / "trial/result.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("dirty", [False, True])
def test_source_identity_precedes_trial_artifacts(tmp_path, monkeypatch, dirty):
    directory = tmp_path / "unignored-trial"

    def git_read(command, **_kwargs):
        assert not directory.exists()
        return "a" * 40 if command[1] == "rev-parse" else "?? owner.py" if dirty else ""

    monkeypatch.setattr("holocron.experiments.subprocess.check_output", git_read)
    Trial(directory, {"schema_version": 1})
    source = json.loads((directory / "provenance.json").read_text())["source"]
    assert source["dirty"] is dirty
    assert source["revision"] == "a" * 40
