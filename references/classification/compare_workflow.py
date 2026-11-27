# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Train, select and replay two classifiers on the frozen Imagenette subsets."""

import argparse
import json
import os
import random
import shlex
import statistics
import subprocess  # noqa: S404
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small

from holocron.experiments import Trial, imagefolder_manifest, sha256, write_json
from holocron.models.classification import resnet18
from references.classification._benchmark import run_benchmark
from references.classification.experiment import RECIPE_KEYS
from references.classification.train import _train, get_parser

MODELS = {
    "resnet18": partial(resnet18, pretrained=False, num_classes=10),
    "mobilenet_v3_small": partial(mobilenet_v3_small, weights=None, num_classes=10),
}
RECIPE = {
    "device": "cpu",
    "seed": 0,
    "epochs": 3,
    "batch_size": 20,
    "workers": 0,
    "lr": 0.001,
    "opt": "adamw",
    "sched": "cosine",
    "weight_decay": 0.01,
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,
    "train_crop_size": 128,
    "val_resize_size": 144,
    "val_crop_size": 128,
}
SPLITS = {"train": 1000, "val": 200, "test": 200}


def read(path):
    return json.loads(path.read_text())


def verify_data(data):
    frozen = read(data / "split-manifest.json")
    datasets = {split: ImageFolder(data / split) for split in SPLITS}
    actual = imagefolder_manifest(datasets)
    if frozen["seed"] != 0 or len(actual["class_to_idx"]) != 10 or actual["class_to_idx"] != frozen["class_to_idx"]:
        raise ValueError("Expected the frozen seed-0 subsets and class mapping")
    for split, count in SPLITS.items():
        targets = datasets[split].targets
        if (
            actual["splits"][split]["sha256"] != frozen["splits"][split]["sha256"]
            or len(targets) != count
            or any(targets.count(label) != count // 10 for label in range(10))
        ):
            raise ValueError(f"Frozen {split} images changed or are not class balanced")
    return {split: value["sha256"] for split, value in actual["splits"].items()}


def train(data, output, name):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002
    torch.manual_seed(0)
    args = get_parser().parse_args([str(data)])
    vars(args).update(RECIPE, arch=name, threads=2)
    config = {
        "schema_version": 1,
        "recipe": {key: getattr(args, key) for key in RECIPE_KEYS},
        "initialization": None,
        "test_split": str(data / "test"),
        "deployment_target": "ONNX Runtime CPU, FP32, batch one, one thread",
    }
    with Trial(output / name, config) as trial:
        args.output_file = str(trial.directory / "checkpoint.pth")
        run_benchmark(args, partial(_train, model=MODELS[name]()), trial=trial)


def freeze_selections(output, hashes):
    selected = {}
    for name in MODELS:
        trial = output / name
        result_path = trial / "result.json"
        result = read(result_path)
        if (result["state"], result["exit_code"], result["epoch"]) != ("completed", 0, RECIPE["epochs"]):
            raise ValueError(f"{name}: all three epochs must complete before test scoring")
        recorded = result["dataset"]
        if (recorded["train"], recorded["validation"], recorded["test"]) != tuple(hashes[key] for key in SPLITS):
            raise ValueError(f"{name}: trial did not use the frozen splits")
        record = result["models"][0]
        if sha256(trial / record["checkpoint"]) != record["checkpoint_sha256"]:
            raise ValueError(f"{name}: selected checkpoint changed")
        selected[name] = {**record, "result_sha256": sha256(result_path)}
    write_json(output / "selections.json", {"frozen_at": datetime.now(UTC).isoformat(), "models": selected})
    return selected


@torch.inference_mode()
def correct(model, batches):
    return sum(int((model(images).argmax(1) == labels).sum()) for images, labels in batches)


def command(*argv, threads=1):
    argv = list(map(str, argv))
    limits = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), str(threads))
    env = {**os.environ, **limits}
    env.pop("PYTHONPATH", None)
    print(shlex.join([*(f"{key}={value}" for key, value in limits.items()), *argv]), flush=True)
    subprocess.run(argv, env=env, check=True)  # noqa: S603


def replay(output, name, bundle, runtime):
    destination = output / "replay" / name
    command(runtime, bundle / "scripts/eval_deployment.py", bundle, destination)
    (path,) = destination.glob("*/report.json")
    report = read(path)
    case = report["cases"]["onnx-fp32"]
    accuracy = case["accuracy"]
    timing = [case["timing-1"], case["timing-2"]]
    if (
        not report["complete"]
        or any(step["status"] != "completed" for step in (case["probe"], accuracy, *timing))
        or accuracy["samples"] != SPLITS["val"]
        or accuracy["source_agreement"] < 1.0
    ):
        raise ValueError(f"{name}: incomplete replay or top-1 predictions differ from PyTorch")
    manifest = read(bundle / "bundle.json")
    return {
        "bundle_sha256": sha256(bundle / "bundle.json"),
        "onnx_sha256": manifest["files"][manifest["cases"]["onnx-fp32"]["artifact"]],
        "replay_report_sha256": sha256(path),
        "replay_validation": {
            key: accuracy[key]
            for key in ("correct", "samples", "reference_correct", "max_logit_difference", "source_agreement")
        },
        "latency_ms": {
            key: statistics.median(run[key]["median_ms"] for run in timing) for key in ("inference", "image_to_logits")
        },
        "peak_rss_mib": max(run["peak_rss_mib"] for run in timing),
    }


def run(data, output, summary):
    from references.classification.train import evaluation_transform  # noqa: PLC0415
    from scripts.export_trial import export_trial  # noqa: PLC0415

    hashes = verify_data(data)
    summary.update(split_hashes=hashes, split_manifest_sha256=sha256(data / "split-manifest.json"))
    for name in MODELS:
        command(sys.executable, "-m", __spec__.name, data, output, "--train", name, threads=2)
    selections = freeze_selections(output, hashes)
    if verify_data(data) != hashes:
        raise ValueError("Frozen data changed after training")
    summary["selections_sha256"] = sha256(output / "selections.json")

    torch.set_num_threads(2)
    transform = evaluation_transform(RECIPE["val_resize_size"], RECIPE["val_crop_size"])
    test = ImageFolder(data / "test", transform)
    bundles = {}
    for name, selected in selections.items():
        # Four images speed up MobileNet scoring while keeping memory close to batch one.
        batch_size = 1 if name == "resnet18" else 4
        batches = DataLoader(test, batch_size=batch_size)
        model = MODELS[name]().eval()
        trial = output / name
        weights = torch.load(trial / selected["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(weights["model"], strict=True)
        summary["models"][name] = {
            "selected_epoch": selected["selected_epoch"],
            "selected_validation": {"samples": SPLITS["val"], **selected["selected_metrics"]},
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "training_seconds": selected["training_elapsed_seconds"],
            "training_peak_rss_mib": selected["memory"]["process_lifetime_peak_rss_mib"],
            "test": {"samples": len(test), "correct": correct(model, batches), "batch_size": batch_size},
        }
        bundle = output / "bundles" / name
        print(f"export_trial({str(trial)!r}, {str(bundle)!r}, model={name})", flush=True)
        bundles[name] = export_trial(trial, bundle, model=model)

    runtime = output / "runtime/bin/python"
    command("uv", "venv", "--python", sys.executable, output / "runtime")
    command("uv", "pip", "install", "--python", runtime, "-r", next(iter(bundles.values())) / "requirements-onnx.txt")
    for name, bundle in bundles.items():
        summary["models"][name].update(replay(output, name, bundle, runtime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Frozen Imagenette subsets")
    parser.add_argument("output", type=Path, help="New run directory outside Git")
    parser.add_argument("--train", choices=MODELS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    data, output = args.data.resolve(strict=True), args.output.resolve()
    if args.train:
        train(data, output, args.train)
    else:
        output.mkdir(parents=True, exist_ok=False)
        summary = {"decision": "stop", "manually_edited_metadata_files": [], "models": {}}
        print(shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]), flush=True)
        try:
            run(data, output, summary)
            summary["decision"] = "continue"
        except Exception as error:
            summary["error"] = str(error)
            raise
        finally:
            write_json(output / "results.json", summary)
