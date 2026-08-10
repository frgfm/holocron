# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import hashlib
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from PIL import Image, ImageFont
from torch.hub import download_url_to_file, get_dir

__all__ = ["download_fonts", "find_fonts", "render_text"]


_GOOGLE_FONTS_REVISION = "038b637da7b3fd956a4ed93ffc607c3d5e4ce172"
_GOOGLE_FONTS_URL = f"https://raw.githubusercontent.com/google/fonts/{_GOOGLE_FONTS_REVISION}"
_FONT_MANIFEST = (
    (
        "Montserrat.ttf",
        "ofl/montserrat/Montserrat[wght].ttf",
        "0f7b311b2f3279e4eef9b2f968bcdbab6e28f4daeb1f049f4f278a902bcd82f7",
    ),
    (
        "Lora.ttf",
        "ofl/lora/Lora[wght].ttf",
        "822a6621ccbe8d97d20ac88c1c41f5615c9c2c202eaa75f272cd452aac6475a7",
    ),
    (
        "RobotoMono.ttf",
        "ofl/robotomono/RobotoMono[wght].ttf",
        "66a80e79d17e4c7cabd162e2916578a4cc08fd19eef6e2a643305eae9c567b2b",
    ),
    (
        "Caveat.ttf",
        "ofl/caveat/Caveat[wght].ttf",
        "0bdb6b660482d31531b3945849fba5916b3ef8695da7024a9e6b9ee3c4157988",
    ),
    (
        "BebasNeue.ttf",
        "ofl/bebasneue/BebasNeue-Regular.ttf",
        "08e4623805102d819f58601e46e345648846075e363b2ceb23313c2d1c83ec73",
    ),
)


def _validate_color(name: str, color: int) -> None:
    if isinstance(color, bool) or not isinstance(color, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= color <= 255:
        raise ValueError(f"{name} must be an integer between 0 and 255")


def render_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    padding: int = 0,
    background_color: int = 0,
    text_color: int = 255,
) -> Image.Image:
    """Render text as a tight grayscale image.

    Args:
        text: non-empty Unicode text to render
        font: preloaded font used for rasterization
        padding: number of background pixels added on every side
        background_color: grayscale background value
        text_color: grayscale text value

    Returns:
        rendered grayscale image

    Raises:
        TypeError: if `text`, `font`, or `padding` has an invalid type
        ValueError: if the arguments cannot produce a visible image
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        raise ValueError("text must not be empty")
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise TypeError("font must be a PIL.ImageFont.FreeTypeFont")
    if isinstance(padding, bool) or not isinstance(padding, int):
        raise TypeError("padding must be an integer")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    _validate_color("background_color", background_color)
    _validate_color("text_color", text_color)
    if background_color == text_color:
        raise ValueError("background_color and text_color must differ")

    mask, _ = font.getmask2(text, mode="L")
    # Pillow has no public zero-copy wrapper for its rasterized ImagingCore mask.
    image = Image.Image()._new(mask)  # noqa: SLF001
    if image.getbbox() is None:
        raise ValueError("text does not contain a visible glyph")

    if padding == 0 and background_color == 0 and text_color == 255:
        return image

    output = Image.new("L", (image.width + 2 * padding, image.height + 2 * padding), background_color)
    output.paste(text_color, (padding, padding), image)
    return output


@lru_cache(maxsize=128)
def find_fonts(text: str | None = None) -> tuple[str, ...]:
    """Find loadable system fonts that cover the requested text.

    Matplotlib-bundled fonts are included so discovery does not depend on the
    host having user-installed fonts.

    Args:
        text: optional text whose non-whitespace code points must be supported

    Returns:
        sorted, deduplicated font paths

    Raises:
        TypeError: if `text` is neither a string nor `None`
    """
    if text is not None and not isinstance(text, str):
        raise TypeError("text must be a string or None")

    required_codepoints = {ord(char) for char in text or "" if not char.isspace()}
    font_paths = {str(entry.fname) for entry in font_manager.fontManager.ttflist}
    font_paths.update(font_manager.findSystemFonts())

    available = []
    for font_path in sorted(font_paths):
        if not Path(font_path).is_file():
            continue
        try:
            ImageFont.truetype(font_path, 10)
            if required_codepoints and not required_codepoints.issubset(FT2Font(font_path).get_charmap()):
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        available.append(font_path)
    return tuple(available)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_fonts(cache_dir: str | Path | None = None, *, progress: bool = True) -> tuple[str, ...]:
    """Download a small SHA-256-verified Latin OCR font collection.

    Args:
        cache_dir: destination directory, or the Holocron Torch Hub cache by default
        progress: whether to display download progress

    Returns:
        downloaded font paths in deterministic manifest order
    """
    root = Path(cache_dir) if cache_dir is not None else Path(get_dir()) / "holocron" / "fonts"
    root.mkdir(parents=True, exist_ok=True)

    font_paths = []
    for filename, source_path, expected_sha256 in _FONT_MANIFEST:
        destination = root / filename
        if destination.is_file() and _sha256(destination) == expected_sha256:
            font_paths.append(str(destination))
            continue

        url = f"{_GOOGLE_FONTS_URL}/{quote(source_path, safe='/')}"
        with NamedTemporaryFile(dir=root, prefix=f".{filename}.", delete=False) as file:
            temporary = Path(file.name)
        try:
            download_url_to_file(url, str(temporary), hash_prefix=expected_sha256, progress=progress)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        font_paths.append(str(destination))

    return tuple(font_paths)
