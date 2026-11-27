# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Export a completed classification trial into a frozen ONNX deployment bundle."""

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from holocron.experiments import evaluation_preprocessing, evaluation_transform, sha256, write_json
from holocron.models import get_model

if __package__:
    from .eval_deployment import preprocess, safe_path
    from .export_to_onnx import export_model
    from .prepare_deployment import freeze_bundle, stage_bundle
else:
    from eval_deployment import preprocess, safe_path
    from export_to_onnx import export_model
    from prepare_deployment import freeze_bundle, stage_bundle


@torch.inference_mode()
def export_trial(trial, destination, *, model=None):  # noqa: PLR0912, PLR0915 -- one verified handoff
    """Freeze the selected weights and validation membership; optionally supply an ordinary nn.Module.

    Returns:
        The completed deployment bundle directory.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If trial records, hashes or evaluation preprocessing are unsupported or inconsistent.
    """
    trial, destination = Path(trial).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    records = {
        name: json.loads((trial / f"{name}.json").read_text()) for name in ("config", "data", "provenance", "result")
    }
    if any(record.get("schema_version") != 1 for record in records.values()):
        raise ValueError("Unsupported trial record version")
    config, data, provenance, result = (records[name] for name in ("config", "data", "provenance", "result"))
    if result["state"] != "completed" or result["exit_code"] != 0 or len(result["models"]) != 1:
        raise ValueError("Export requires a completed trial with one selected model")
    selected = result["models"][0]
    if selected["architecture"] != config["recipe"]["arch"]:
        raise ValueError("Trial architecture records differ")
    mapping = data["class_to_idx"]
    categories = sorted(mapping, key=mapping.get)
    if [mapping[name] for name in categories] != list(range(len(categories))) or mapping != result["dataset"][
        "class_to_index"
    ]:
        raise ValueError("Trial class mappings differ or are not contiguous")
    validation = data["splits"]["validation"]
    identity = hashlib.sha256(
        json.dumps(validation["samples"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not validation["samples"] or any(
        expected != identity
        for expected in (validation["sha256"], provenance["data"]["validation"], result["dataset"]["validation"])
    ):
        raise ValueError("Validation membership hash mismatch or empty split")
    prep = provenance.get("evaluation_preprocessing")
    if prep is None:
        raise ValueError(
            "Unsupported or ambiguous evaluation preprocessing; record a new trial with supported transforms"
        )
    transform = evaluation_transform(config["recipe"]["val_resize_size"], config["recipe"]["val_crop_size"])
    if evaluation_preprocessing(transform) != prep or selected["input_shape"] != prep["input_shape"]:
        raise ValueError("Unsupported or inconsistent evaluation preprocessing")

    with stage_bundle(destination) as root:
        (root / "validation").mkdir()
        checkpoint = root / "checkpoint.pth"
        shutil.copyfile(safe_path(trial, selected["checkpoint"]), checkpoint)
        if sha256(checkpoint) != selected["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["epoch"] != selected["selected_epoch"]:
            raise ValueError("Selected checkpoint epoch mismatch")
        if model is None:
            model = get_model(selected["architecture"], pretrained=False, num_classes=len(categories))
        model = model.cpu().float().eval()
        model.load_state_dict(state["model"], strict=True)
        del state

        samples, seen = [], set()
        reference = np.empty((len(validation["samples"]), len(categories)), dtype=np.float32)
        validation_root = Path(validation["root"])
        for index, sample in enumerate(validation["samples"]):
            name = sample["path"]
            if name in seen:
                raise ValueError(f"Duplicate sample ID: {name}")
            seen.add(name)
            source = safe_path(validation_root, name)
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sample["sha256"]:
                raise ValueError(f"Validation image hash mismatch: {source}")
            stored = f"validation/{index:04d}-{source.name}"
            (root / stored).write_bytes(raw)
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert("RGB")
            images = transform(rgb).unsqueeze(0)
            if not np.array_equal(images.numpy(), preprocess(rgb, prep)):
                raise ValueError(f"Evaluation preprocessing differs from bundle replay: {source}")
            logits = model(images)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.shape != (1, len(categories))
                or logits.dtype != torch.float32
            ):
                raise ValueError("Expected FP32 classification logits with shape (1, classes)")
            reference[index] = logits.numpy()[0]
            samples.append({"id": name, "path": stored, "label": sample["class_index"], "sha256": sample["sha256"]})
            if index == 0:
                example = images

        if not np.isfinite(reference).all():
            raise ValueError("Reference logits must be finite")
        export_model(model, example, root / "model.onnx")
        write_json(root / "trial.json", {"trial": str(trial), **records})
        cases = {
            "onnx-fp32": {"backend": "onnx", "artifact": "model.onnx", "requested_precision": "fp32", "threads": 1}
        }
        freeze_bundle(
            root,
            {**prep, "categories": categories, "samples": samples},
            cases,
            reference,
            {"checkpoint": checkpoint.name, "provenance": "trial.json"},
        )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="Completed classification trial directory")
    parser.add_argument("output", type=Path, help="New bundle directory; never overwritten")
    args = parser.parse_args()
    print(export_trial(args.trial, args.output))
