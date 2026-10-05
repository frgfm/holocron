import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from matplotlib import font_manager
from PIL import Image, ImageFont

from holocron import utils
from holocron.utils import fonts as font_utils
from scripts import prepare_fonts as font_prep


@pytest.fixture(scope="module")
def dejavu_sans():
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 32)


def test_mixup():
    batch_size = 8
    num_classes = 10
    shape = (3, 32, 32)
    with pytest.raises(ValueError):
        utils.data.Mixup(num_classes, alpha=-1.0)
    # Generate all dependencies
    mix = utils.data.Mixup(num_classes, alpha=0.2)
    img, target = torch.rand((batch_size, *shape)), torch.arange(num_classes)[:batch_size]
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert img.shape == (batch_size, *shape)
    assert not torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, num_classes)
    assert torch.all(mix_target.sum(dim=1) == 1.0)
    count = (mix_target > 0).sum(dim=1)
    assert torch.all((count == 2.0) | (count == 1.0))

    # Alpha = 0 case
    mix = utils.data.Mixup(num_classes, alpha=0.0)
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, num_classes)
    assert torch.all(mix_target.sum(dim=1) == 1.0)
    assert torch.all((mix_target > 0).sum(dim=1) == 1.0)

    # Binary target
    mix = utils.data.Mixup(1, alpha=0.5)
    img = torch.rand((batch_size, *shape))
    target = torch.concat((torch.zeros(batch_size // 2), torch.ones(batch_size - batch_size // 2)))
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert img.shape == (batch_size, *shape)
    assert not torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, 1)

    # Already in one-hot
    mix = utils.data.Mixup(num_classes, alpha=0.2)
    img, target = torch.rand((batch_size, *shape)), torch.rand((batch_size, num_classes))
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert img.shape == (batch_size, *shape)
    assert not torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    ("arr", "fn", "expected", "progress", "num_threads"),
    [
        ([1, 2, 3], lambda x: x**2, [1, 4, 9], False, 3),
        ([1, 2, 3], lambda x: x**2, [1, 4, 9], True, 1),
        ("hello", lambda x: x.upper(), list("HELLO"), True, None),
        ("hello", lambda x: x.upper(), list("HELLO"), False, None),
    ],
)
def test_parallel(arr, fn, expected, progress, num_threads):
    assert utils.parallel(fn, arr, progress=progress, num_threads=num_threads) == expected


def test_find_image_size():
    ds = [(Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)), 0) for _ in range(100)]
    utils.find_image_size(ds, block=False)


@pytest.mark.parametrize("text", ["A", "é", "OCR-42", " A ", "A A"])
def test_render_text(dejavu_sans, text):
    image = utils.render_text(text, dejavu_sans)
    assert image.mode == "L"
    assert image.width > 0
    assert image.height > 0
    assert image.getbbox() is not None
    assert image.tobytes() == utils.render_text(text, dejavu_sans).tobytes()


def test_render_text_layout(dejavu_sans):
    character = utils.render_text("A", dejavu_sans)
    sequence = utils.render_text("OCR-42", dejavu_sans)
    padded = utils.render_text("A", dejavu_sans, padding=3)
    inverted = utils.render_text("é", dejavu_sans, padding=1, background_color=255, text_color=0)

    assert sequence.width > character.width
    assert padded.size == (character.width + 6, character.height + 6)
    assert padded.crop((3, 3, padded.width - 3, padded.height - 3)).tobytes() == character.tobytes()
    assert inverted.getpixel((0, 0)) == 255
    assert inverted.getextrema()[0] < 255


@pytest.mark.parametrize(
    ("args", "error", "match"),
    [
        (("",), ValueError, "must not be empty"),
        (("   ",), ValueError, "visible glyph"),
        ((123,), TypeError, "must be a string"),
    ],
)
def test_render_text_invalid_text(dejavu_sans, args, error, match):
    with pytest.raises(error, match=match):
        utils.render_text(*args, dejavu_sans)


def test_render_text_invalid_options(dejavu_sans):
    with pytest.raises(TypeError, match="padding must be an integer"):
        utils.render_text("A", dejavu_sans, padding=1.5)
    with pytest.raises(ValueError, match="padding must be non-negative"):
        utils.render_text("A", dejavu_sans, padding=-1)
    with pytest.raises(ValueError, match="between 0 and 255"):
        utils.render_text("A", dejavu_sans, text_color=256)
    with pytest.raises(TypeError, match="must be an integer"):
        utils.render_text("A", dejavu_sans, text_color=True)
    with pytest.raises(ValueError, match="must differ"):
        utils.render_text("A", dejavu_sans, background_color=0, text_color=0)
    with pytest.raises(TypeError, match="FreeTypeFont"):
        utils.render_text("A", ImageFont.load_default_imagefont())


def test_find_fonts_uses_bundled_fonts(monkeypatch):
    bundled_font = font_manager.findfont("DejaVu Sans")
    monkeypatch.setattr(font_manager, "findSystemFonts", list)
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [SimpleNamespace(fname=bundled_font)])
    font_utils.find_fonts.cache_clear()

    assert font_utils.find_fonts("é") == (bundled_font,)
    assert font_utils.find_fonts(chr(0x10FFFF)) == ()
    font_utils.find_fonts.cache_clear()


@pytest.fixture
def font_manifest(tmp_path):
    payload = b"font payload"
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = {
        "version": 1,
        "sources": {"test": {"base_url": "https://example.com/fonts", "revision": "abc123"}},
        "fonts": [
            {
                "family": "Test",
                "style": "Regular",
                "source": "test",
                "path": "Test.ttf",
                "filename": "Test.ttf",
                "sha256": checksum,
                "license": "OFL-1.1",
            }
        ],
    }
    manifest_path = tmp_path / "fonts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, payload


def test_prepare_fonts_cache(font_manifest, monkeypatch, tmp_path):
    manifest_path, payload = font_manifest
    calls = []

    def _download(url, destination, hash_prefix, progress):
        calls.append((url, Path(destination), hash_prefix, progress))
        Path(destination).write_bytes(payload)

    monkeypatch.setattr(font_prep, "download_url_to_file", _download)
    output_dir = tmp_path / "output"
    expected = (output_dir / "Test.ttf",)

    assert font_prep.prepare_fonts(manifest_path, output_dir, progress=False) == expected
    assert font_prep.prepare_fonts(manifest_path, output_dir, progress=False) == expected
    assert len(calls) == 1
    assert calls[0][1].name.startswith(".Test.ttf.")

    expected[0].write_bytes(b"corrupt")
    assert font_prep.prepare_fonts(manifest_path, output_dir, progress=False) == expected
    assert len(calls) == 2
    assert expected[0].read_bytes() == payload


def test_prepare_fonts_rejects_bad_checksum(font_manifest, monkeypatch, tmp_path):
    manifest_path, _ = font_manifest

    def _download(url, destination, hash_prefix, progress):
        del url, destination, hash_prefix, progress
        raise RuntimeError("invalid hash value")

    monkeypatch.setattr(font_prep, "download_url_to_file", _download)
    output_dir = tmp_path / "output"
    with pytest.raises(RuntimeError, match="invalid hash value"):
        font_prep.prepare_fonts(manifest_path, output_dir)
    assert not (output_dir / "Test.ttf").exists()


def test_prepare_fonts_rejects_unknown_version(font_manifest, tmp_path):
    manifest_path, _ = font_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported font manifest version"):
        font_prep.prepare_fonts(manifest_path, tmp_path / "output")
