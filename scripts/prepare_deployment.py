# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Freeze exported classifiers, image manifests and reference logits for device evaluation."""

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    from .eval_deployment import PACKAGES, digest, safe_path, validate_manifest
    from .eval_latency import environment, save_report
else:
    from eval_deployment import PACKAGES, digest, safe_path, validate_manifest
    from eval_latency import environment, save_report


def validate_embedded_onnx(path):
    import onnx  # noqa: PLC0415 -- preparation uses the existing export dependencies

    def visit(message):
        if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("ONNX artifact must be self-contained; embed external weights")
        for field, value in message.ListFields():
            if field.message_type is not None:
                for child in value if field.is_repeated else (value,):
                    visit(child)

    visit(onnx.load(path, load_external_data=False))


def prepare(config_path, destination):  # noqa: PLR0912, PLR0915 -- assemble one atomic, self-contained bundle
    config_path, destination = Path(config_path).resolve(), Path(destination).absolute()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.pop("schema_version") != 1:
        raise ValueError("Unsupported configuration version")
    unknown = set(config) - {
        "manifest",
        "accuracy_manifest",
        "reference",
        "cases",
        "calibration",
        "checkpoint",
        "provenance",
    }
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    def source(name):
        return config_path.parent / config[name]

    manifest_path = source("manifest")
    manifest = json.loads(manifest_path.read_text())
    accuracy = json.loads(source("accuracy_manifest").read_text()) if "accuracy_manifest" in config else manifest
    for key in ("input_shape", "resize", "mean", "std", "categories"):
        if accuracy[key] != manifest[key]:
            raise ValueError(f"Accuracy and timing preprocessing differ: {key}")
    cases_path = source("cases")
    cases = json.loads(cases_path.read_text())
    reference = np.asarray(np.load(source("reference"), allow_pickle=False), dtype=np.float32)
    expected = (len(accuracy["samples"]), len(manifest["categories"]))
    if reference.shape != expected or not np.isfinite(reference).all():
        raise ValueError(f"Reference logits must be finite with shape {expected}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="deployment-", dir=destination.parent) as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        copied = {}

        def copy(path, group):
            path = path.resolve(strict=True)
            if path not in copied:
                target = root / group / f"{len(copied):04d}-{path.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.is_dir():
                    shutil.copytree(path, target)
                else:
                    shutil.copyfile(path, target)
                copied[path] = target.relative_to(root).as_posix()
            return copied[path]

        samples, indices, seen = [], {}, set()
        for sample in manifest["samples"]:
            path = safe_path(manifest_path.parent, sample["path"])
            stored = copy(path, "validation")
            if digest(root / stored) != sample["sha256"]:
                raise ValueError(f"Sample hash mismatch: {path}")
            if sample["path"] in indices:
                raise ValueError(f"Duplicate sample ID: {sample['path']}")
            indices[sample["path"]] = len(samples)
            samples.append({**sample, "id": sample["path"], "path": stored})
            seen.add(sample["sha256"])
        accuracy_indices = []
        for sample in accuracy["samples"]:
            index = indices[sample["path"]]
            if any(sample[key] != manifest["samples"][index][key] for key in ("label", "sha256")):
                raise ValueError("Accuracy subset labels or hashes differ")
            accuracy_indices.append(index)

        calibration = []
        if "calibration" in config:
            calibration_path = source("calibration")
            for sample in json.loads(calibration_path.read_text())["samples"]:
                path = safe_path(calibration_path.parent, sample["path"])
                stored = copy(path, "calibration")
                if digest(root / stored) != sample["sha256"] or sample["sha256"] in seen:
                    raise ValueError(f"Calibration hash mismatch or validation overlap: {path}")
                calibration.append({**sample, "path": stored})
        for case in cases.values():
            for field in ("artifact", "param"):
                if field in case:
                    case[field] = copy(cases_path.parent / case[field], "artifacts")
            if case["backend"] == "onnx":
                validate_embedded_onnx(root / case["artifact"])

        (root / "scripts").mkdir()
        for name in ("eval_deployment.py", "eval_latency.py"):
            shutil.copyfile(Path(__file__).with_name(name), root / "scripts" / name)
        np.save(root / "reference.npy", reference)
        requirements = {}
        for backend in sorted({case["backend"] for case in cases.values()}):
            requirements[backend] = {package: version(package) for package in PACKAGES[backend]}
            (root / f"requirements-{backend}.txt").write_text(
                "".join(f"{package}=={value}\n" for package, value in requirements[backend].items())
            )
        provenance = {
            "builder": environment(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "preparer_sha256": digest(__file__),
        }
        for key in ("checkpoint", "provenance"):
            if key in config:
                path = source(key)
                stored = copy(path, "provenance")
                provenance[key] = {"sha256": digest(root / stored), "path": stored}
        bundle = {
            "schema_version": 1,
            "preprocessing": {key: manifest[key] for key in ("input_shape", "resize", "mean", "std")},
            "categories": manifest["categories"],
            "samples": samples,
            "accuracy_indices": accuracy_indices,
            "reference": "reference.npy",
            "calibration": calibration,
            "cases": cases,
            "requirements": requirements,
            "provenance": provenance,
            "protocol": {"warmup": 10, "iterations": 100, "repeats": 2, "chunk_size": 100},
            "files": {
                path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob("*")) if path.is_file()
            },
        }
        validate_manifest(bundle)
        save_report(root / "bundle.json", bundle)
        root.rename(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, help="New bundle directory; never overwritten")
    args = parser.parse_args()
    print(prepare(args.config, args.output))
