# Copyright (C) 2019-2026, François-Guillaume Fernandez.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RunResult", "write_run_bundle"]


@dataclass(frozen=True)
class RunResult:
    """Result of a completed training run."""

    epoch: int
    step: int
    best_metric: float
    metrics: tuple[dict[str, float | int | None], ...]
    config: dict[str, Any]
    checkpoint: str | None = None
    bundle_dir: str | None = None


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        Path(tmp_name).replace(path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, tmp_name)
        Path(tmp_name).replace(destination)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _artifact(path: Path, root: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
    }


def write_run_bundle(
    output_dir: str | Path,
    result: RunResult,
    artifacts: Sequence[str | Path] = (),
) -> Path:
    """Write a portable schema-v1 run bundle and return its manifest path.

    The manifest is removed before updating a bundle and written only after all
    referenced artifacts are durable, so its presence marks a complete bundle.

    Returns:
        path to the completed manifest

    Raises:
        FileNotFoundError: if an artifact does not exist
        ValueError: if multiple artifacts have the same file name
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_path.unlink(missing_ok=True)

    metrics_path = root / "metrics.jsonl"
    metrics = "".join(f"{json.dumps(metric, sort_keys=True)}\n" for metric in result.metrics)
    _atomic_write(metrics_path, metrics)
    bundle_artifacts = [_artifact(metrics_path, root)]

    if result.checkpoint is not None:
        checkpoint = Path(result.checkpoint)
        if checkpoint.is_file():
            bundled_checkpoint = root / "checkpoints" / checkpoint.name
            _atomic_copy(checkpoint, bundled_checkpoint)
            bundle_artifacts.append(_artifact(bundled_checkpoint, root))

    used_names: set[str] = set()
    for artifact in artifacts:
        source = Path(artifact)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.name in used_names:
            raise ValueError(f"duplicate artifact name: {source.name}")
        used_names.add(source.name)
        bundled_artifact = root / "artifacts" / source.name
        _atomic_copy(source, bundled_artifact)
        bundle_artifacts.append(_artifact(bundled_artifact, root))

    manifest = {
        "schema_version": 1,
        "run": {
            "epoch": result.epoch,
            "step": result.step,
            "best_metric": result.best_metric,
            "config": result.config,
        },
        "artifacts": bundle_artifacts,
    }
    _atomic_write(manifest_path, f"{json.dumps(manifest, indent=2, sort_keys=True)}\n")
    return manifest_path
