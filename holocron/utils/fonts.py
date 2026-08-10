# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from functools import lru_cache
from pathlib import Path

from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from PIL import Image, ImageFont

__all__ = ["find_fonts", "render_text"]


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
