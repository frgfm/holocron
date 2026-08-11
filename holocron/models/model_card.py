# Copyright (C) 2026, François-Guillaume Fernandez.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Generate deterministic Hugging Face-compatible model cards."""

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from holocron import __version__

from .catalog import ModelInfo, get_model_info, list_checkpoints, list_models
from .checkpoints import Checkpoint

__all__ = ["generate_model_card", "write_model_cards"]

_UNKNOWN = "unknown / not reported"
_PIPELINE_TAGS = {
    "classification": "image-classification",
    "detection": "object-detection",
    "segmentation": "image-segmentation",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _known(value: Any) -> str:
    if value is None or not str(value) or str(value).lower() in {"n/a", "none", "unknown"}:
        return _UNKNOWN
    return str(value).replace("|", "\\|").replace("\n", " ")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml(info: ModelInfo, checkpoint: Checkpoint | None) -> str:
    lines = [
        "---",
        "library_name: holocron",
        f"pipeline_tag: {_PIPELINE_TAGS[info.task]}",
        "license: apache-2.0",
        "tags:",
        "  - pytorch",
        "  - holocron",
        "  - computer-vision",
        f"  - {info.maturity.value}",
    ]
    if checkpoint is not None:
        lines.extend(["datasets:", f"  - {checkpoint.evaluation.dataset.value}"])
        if checkpoint.evaluation.results:
            lines.append("metrics:")
            lines.extend(f"  - {metric.value}" for metric in sorted(checkpoint.evaluation.results, key=str))
    lines.extend(["---", ""])
    return "\n".join(lines)


def _validate_artifact(root: Path, artifact: Any) -> dict[str, str | int]:
    if not isinstance(artifact, dict):
        raise TypeError("malformed run manifest: artifact must be an object")
    path, sha256, size = artifact.get("path"), artifact.get("sha256"), artifact.get("size")
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("malformed run manifest: artifact path must be relative")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ValueError("malformed run manifest: artifact sha256 must be a lowercase SHA-256")
    if not isinstance(size, int) or size < 0:
        raise ValueError("malformed run manifest: artifact size must be a non-negative integer")

    artifact_path = root / path
    if not artifact_path.is_file():
        raise ValueError(f"unfinished run bundle: missing artifact {path}")
    if artifact_path.stat().st_size != size:
        raise ValueError(f"malformed run manifest: size mismatch for {path}")
    if _file_sha256(artifact_path) != sha256:
        raise ValueError(f"malformed run manifest: SHA-256 mismatch for {path}")
    return {"path": path, "sha256": sha256, "size": size}


def _load_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"unfinished run bundle: {manifest_path} is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed run manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise TypeError("malformed run manifest: root must be an object")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported run manifest schema_version: {manifest.get('schema_version')!r}")
    if not isinstance(manifest.get("run"), dict) or not isinstance(manifest.get("artifacts"), list):
        raise TypeError("malformed run manifest: run and artifacts are required")
    manifest["artifacts"] = [_validate_artifact(root, artifact) for artifact in manifest["artifacts"]]
    return manifest


def _checkpoint_sections(checkpoint: Checkpoint | None) -> list[str]:
    if checkpoint is None:
        return [
            "## Checkpoint",
            "",
            f"Checkpoint metadata: **{_UNKNOWN}**",
            "",
            "## Preprocessing",
            "",
            f"Preprocessing metadata: **{_UNKNOWN}**",
            "",
            "## Training recipe and evaluation",
            "",
            f"Training recipe and metrics: **{_UNKNOWN}**",
        ]

    preprocessing = checkpoint.pre_processing
    recipe = checkpoint.recipe
    lines = [
        "## Checkpoint",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | {_known(checkpoint.evaluation.dataset.value)} |",
        f"| Download URL | {_known(checkpoint.meta.url)} |",
        f"| SHA-256 | `{_known(checkpoint.meta.sha256)}` |",
        f"| Size (bytes) | {_known(checkpoint.meta.size)} |",
        f"| Parameters | {_known(checkpoint.meta.num_params)} |",
        "",
        "## Preprocessing",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Input shape | `{preprocessing.input_shape}` |",
        f"| Mean | `{preprocessing.mean}` |",
        f"| Standard deviation | `{preprocessing.std}` |",
        f"| Interpolation | {_known(preprocessing.interpolation.value)} |",
        "",
        "## Training recipe and evaluation",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Source commit | `{_known(recipe.commit)}` |",
        f"| Script | {_known(recipe.script)} |",
        f"| Arguments | `{_known(recipe.args)}` |",
        "",
        "| Metric | Dataset | Value |",
        "|---|---|---:|",
    ]
    if checkpoint.evaluation.results:
        for metric, value in sorted(checkpoint.evaluation.results.items(), key=lambda item: str(item[0])):
            lines.append(f"| {metric.value} | {checkpoint.evaluation.dataset.value} | {value:.6g} |")
    else:
        lines.append(f"| {_UNKNOWN} | {_known(checkpoint.evaluation.dataset.value)} | {_UNKNOWN} |")
    return lines


def _run_sections(manifest: dict[str, Any] | None) -> list[str]:
    if manifest is None:
        return ["## Run provenance", "", f"Run bundle: **{_UNKNOWN}**"]

    run = manifest["run"]
    lines = [
        "## Run provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Schema version | {manifest['schema_version']} |",
        f"| Epoch | {_known(run.get('epoch'))} |",
        f"| Step | {_known(run.get('step'))} |",
        f"| Best metric | {_known(run.get('best_metric'))} |",
        "",
        "### Resolved configuration",
        "",
        "```json",
        json.dumps(run.get("config", {}), indent=2, sort_keys=True),
        "```",
        "",
        "### Run artifacts",
        "",
        "| Path | Size (bytes) | SHA-256 |",
        "|---|---:|---|",
    ]
    artifacts = manifest["artifacts"]
    if artifacts:
        lines.extend(f"| {item['path']} | {item['size']} | `{item['sha256']}` |" for item in artifacts)
    else:
        lines.append(f"| {_UNKNOWN} | {_UNKNOWN} | {_UNKNOWN} |")
    return lines


def generate_model_card(
    name: str,
    checkpoint: Checkpoint | None = None,
    run_dir: str | Path | None = None,
) -> str:
    """Render a deterministic model card from Holocron metadata.

    Args:
        name: public model factory name.
        checkpoint: optional typed checkpoint for the model.
        run_dir: optional completed schema-v1 run bundle.

    Returns:
        Hugging Face-compatible Markdown with YAML metadata.

    Raises:
        ValueError: if the model, checkpoint, or run bundle is invalid.
    """
    info = get_model_info(name)
    if checkpoint is not None and checkpoint.meta.arch != name:
        raise ValueError(f"checkpoint architecture {checkpoint.meta.arch!r} does not match {name!r}")
    manifest = _load_run_manifest(run_dir) if run_dir is not None else None
    availability = "available" if info.pretrained else "not reported"
    lines = [
        _yaml(info, checkpoint),
        f"# {name}",
        "",
        "Generated from Holocron's typed catalog. Missing values are shown as unknown rather than inferred.",
        "",
        "## Model details",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model | `{name}` |",
        f"| Task | {info.task} |",
        f"| Maturity | {info.maturity.value} |",
        f"| Pretrained weights | {availability} |",
        "| Library | Holocron (`pylocron`) |",
        f"| Library version | `{__version__}` |",
        "| Implementation license | Apache-2.0 |",
        f"| Dataset/checkpoint-specific license | {_UNKNOWN} |",
        "",
        "## Intended use",
        "",
        f"Use this model through Holocron for {info.task} with the recorded checkpoint and preprocessing metadata.",
        f"Deployment suitability and domain-specific uses are **{_UNKNOWN}**.",
        "",
        "## Limitations",
        "",
        f"Model-specific limitations are **{_UNKNOWN}**. The `{info.maturity.value}` maturity label describes available",
        "evidence; it is not a fitness-for-purpose claim.",
        "",
        *_checkpoint_sections(checkpoint),
        "",
        *_run_sections(manifest),
        "",
    ]
    return "\n".join(lines)


def _card_filename(name: str, checkpoint: Checkpoint | None, index: int) -> str:
    if checkpoint is None:
        return f"{name}.md"
    dataset = checkpoint.evaluation.dataset.value
    digest = checkpoint.meta.sha256[:8] if _SHA256.fullmatch(checkpoint.meta.sha256) else str(index)
    return f"{name}-{dataset}-{digest}.md"


def write_model_cards(output_dir: str | Path, names: Sequence[str] | None = None) -> tuple[Path, ...]:
    """Write every typed checkpoint card for selected models to a directory.

    Returns:
        paths of the generated cards
    """
    root = Path(output_dir)
    selected_names = sorted(names) if names is not None else list_models()
    written: list[Path] = []
    for name in selected_names:
        checkpoints = list_checkpoints(name)
        selections: tuple[Checkpoint | None, ...] = checkpoints or (None,)
        for index, checkpoint in enumerate(selections):
            path = root / _card_filename(name, checkpoint, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(generate_model_card(name, checkpoint), encoding="utf-8")
            written.append(path)
    return tuple(written)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="public model names; all models when omitted with --output-dir")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="write one model/checkpoint card")
    destination.add_argument("--output-dir", type=Path, help="write every selected model/checkpoint card")
    parser.add_argument("--checkpoint", type=int, default=0, help="checkpoint index used with --output")
    parser.add_argument("--run-dir", type=Path, help="completed schema-v1 run bundle used with --output")
    args = parser.parse_args(argv)

    if args.output is not None:
        if len(args.models) != 1:
            parser.error("--output requires exactly one model")
        if args.checkpoint < 0:
            parser.error("checkpoint index must be non-negative")
        checkpoints = list_checkpoints(args.models[0])
        if not checkpoints:
            if args.checkpoint != 0:
                parser.error("model has no typed checkpoints")
            checkpoint = None
        else:
            try:
                checkpoint = checkpoints[args.checkpoint]
            except IndexError:
                parser.error(f"checkpoint index must be between 0 and {len(checkpoints) - 1}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_model_card(args.models[0], checkpoint, args.run_dir), encoding="utf-8")
    else:
        if args.run_dir is not None or args.checkpoint != 0:
            parser.error("--run-dir and --checkpoint are only valid with --output")
        write_model_cards(args.output_dir, args.models or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
