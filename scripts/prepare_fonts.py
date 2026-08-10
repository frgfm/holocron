# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Prepare a verified local font directory from a manifest."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

from torch.hub import download_url_to_file

_DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "references" / "fonts" / "latin-starter.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_fonts(manifest_path, output_dir, progress=True):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported font manifest version")
    output_dir.mkdir(parents=True, exist_ok=True)
    font_paths = []

    for font in manifest["fonts"]:
        source = manifest["sources"][font["source"]]
        filename = font["filename"]
        checksum = font["sha256"].lower()
        if Path(filename).name != filename:
            raise ValueError(f"invalid font filename: {filename}")
        try:
            valid_checksum = len(bytes.fromhex(checksum)) == 32
        except ValueError:
            valid_checksum = False
        if not valid_checksum:
            raise ValueError(f"invalid SHA-256 for {filename}")

        destination = output_dir / filename
        if destination.is_file() and _sha256(destination) == checksum:
            font_paths.append(destination)
            continue

        url = "/".join((source["base_url"].rstrip("/"), source["revision"], quote(font["path"], safe="/")))
        with NamedTemporaryFile(dir=output_dir, prefix=f".{filename}.", delete=False) as file:
            temporary = Path(file.name)
        try:
            download_url_to_file(url, str(temporary), hash_prefix=checksum, progress=progress)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        font_paths.append(destination)

    return tuple(font_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, help="Font manifest to prepare")
    parser.add_argument("--output", type=Path, required=True, help="Destination font directory")
    parser.add_argument("--quiet", action="store_true", help="Disable download progress")
    args = parser.parse_args()

    for font_path in prepare_fonts(args.manifest, args.output, progress=not args.quiet):
        print(font_path)
