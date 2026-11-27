# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Freeze a small, class-balanced Imagenette train/validation/test comparison."""

import argparse
import os
import random
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from torchvision.datasets import ImageFolder

from holocron.experiments import imagefolder_manifest, write_json

SEED = 0


def prepare(source, destination):
    """Copy fixed disjoint ImageFolder subsets and record their source membership.

    Returns:
        Published subset directory.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If the source mappings, sizes or selected split contents are invalid.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("output must be outside the source dataset")

    source_sets = {split: ImageFolder(source / split) for split in ("train", "val")}
    if source_sets["train"].class_to_idx != source_sets["val"].class_to_idx:
        raise ValueError("train and val class mappings differ")
    classes = source_sets["train"].class_to_idx
    if len(classes) != 10:
        raise ValueError(f"expected 10 classes; found {len(classes)}")
    rng = random.Random(SEED)  # noqa: S311 -- reproducible subset sampling
    selected = {"train": [], "val": [], "test": []}
    for index in classes.values():
        train, val = (
            rng.sample(sorted(path for path, label in source_sets[split].samples if label == index), count)
            for split, count in (("train", 100), ("val", 40))
        )
        selected["train"].extend(train)
        selected["val"].extend(val[:20])
        selected["test"].extend(val[20:])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="imagenette-comparison-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "subset"
        source_paths = {split: {} for split in selected}
        for split, records in selected.items():
            for path in records:
                relative = Path(path).relative_to(source)
                target = staging / split / Path(*relative.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                source_paths[split][target.relative_to(staging / split).as_posix()] = relative.as_posix()

        data = imagefolder_manifest({split: ImageFolder(staging / split) for split in selected})
        for split, manifest in data["splits"].items():
            manifest["root"] = str((destination / split).resolve())

        record = {
            "schema_version": 1,
            "seed": SEED,
            "created_at": datetime.now(UTC).isoformat(),
            "source_root": str(source),
            "class_to_idx": classes,
            "splits": data["splits"],
            "source_paths": source_paths,
        }
        write_json(staging / "split-manifest.json", record)
        if os.name == "nt":
            staging.rename(destination)
        else:
            destination.mkdir()  # Reserve the path so POSIX rename cannot replace a competing empty directory.
            try:
                staging.rename(destination)
            except BaseException:
                with suppress(OSError):
                    destination.rmdir()
                raise
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Imagenette root with train/ and val/ ImageFolder splits")
    parser.add_argument("output", type=Path, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    print(prepare(args.source, args.output))
