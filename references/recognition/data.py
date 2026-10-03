# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Seeded synthetic characters, lines, and single-column document pages."""

import random
import string
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from references.classification.train_characters import _inspect_fonts

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase + " .,;:!?-+/=()@%$#'\"_&"
HEIGHT = 32
FIELD_NAMES = ("Invoice", "Total", "Date", "Account", "Reference", "Amount", "Email", "Balance")


class Codec:
    """An ordered Unicode alphabet with index zero reserved for CTC blank."""

    def __init__(self, alphabet=ALPHABET):
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must be nonempty with unique characters")
        if any(not char.isprintable() or (char.isspace() and char != " ") for char in alphabet):
            raise ValueError("only printable characters and ordinary spaces are supported")
        if not alphabet.replace(" ", ""):
            raise ValueError("alphabet must contain a visible character")
        self.alphabet = alphabet
        self.indices = {char: index + 1 for index, char in enumerate(alphabet)}

    def encode(self, text):
        return torch.tensor([self.indices[char] for char in text], dtype=torch.long)

    def decode(self, indices):
        result = []
        previous = 0
        for index in indices:
            if index and index != previous:
                result.append(self.alphabet[index - 1])
            previous = index
        return "".join(result)


def inspect_fonts(records, alphabet):
    """Reject missing glyphs and case-collapsing fonts instead of learning tofu labels.

    Returns:
        inspected records supporting the complete alphabet

    Raises:
        ValueError: if a font lacks glyphs or renders lowercase as uppercase
    """
    visible = [char for char in alphabet if char != " "]
    inspected, supported = _inspect_fonts(visible, records, 32)
    complete = set.intersection(*({record.path for record in group} for group in supported))
    result = []
    for record in inspected:
        if record.path not in complete:
            raise ValueError(f"font does not support the complete alphabet: {record.path}")
        font = load_font(str(record.path), 28)
        for char in string.ascii_lowercase:
            if char in alphabet and char.upper() in alphabet:
                lower, upper = font.getmask(char), font.getmask(char.upper())
                if lower.size == upper.size and bytes(lower) == bytes(upper):
                    raise ValueError(f"font collapses case for {char!r}: {record.path}")
        result.append(record)
    return tuple(result)


def split_fonts(records, held_out):
    """Hold out complete font families, including all of their styles.

    Returns:
        disjoint training and held-out font records

    Raises:
        ValueError: if the split is empty or requests an unknown family
    """
    held_out = set(held_out)
    available = {record.family for record in records}
    if held_out - available:
        raise ValueError(f"unknown held-out families: {sorted(held_out - available)}")
    train = tuple(record for record in records if record.family not in held_out)
    test = tuple(record for record in records if record.family in held_out)
    if not train or not test:
        raise ValueError("both training and held-out font families are required")
    return train, test


@lru_cache(maxsize=256)
def load_font(path, size):
    return ImageFont.truetype(path, size)


def render_line(text, record, rng, *, augment=False):
    """Render kerning and baseline-preserving text without character boxes.

    Returns:
        grayscale line image with optional geometric and photometric degradation
    """
    size = rng.randint(23, 32) if augment else 28
    font = load_font(str(record.path), size)
    ascent, descent = font.getmetrics()
    left, _, right, _ = font.getbbox(text)
    width = max(1, int(max(font.getlength(text), right) - min(0, left))) + 12
    image = Image.new("L", (width, ascent + descent + 8), 255)
    ImageDraw.Draw(image).text((6 - min(0, left), 4), text, font=font, fill=0, anchor="la")
    if augment:
        image = image.rotate(rng.uniform(-1.5, 1.5), resample=Image.Resampling.BILINEAR, fillcolor=255)
        if rng.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.6)))
    width = max(4, round(image.width * HEIGHT / image.height))
    if augment:
        width = max(4, round(width * rng.uniform(0.85, 1.15)))
    image = image.resize((width, HEIGHT), Image.Resampling.LANCZOS)
    if augment:
        array = np.asarray(image, dtype=np.float32)
        contrast = rng.uniform(0.55, 1.0)
        array = 255 - (255 - array) * contrast
        noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(0, 4), array.shape)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    return image


def deskew_line(image):
    """Correct small rotations by maximizing horizontal ink concentration.

    Returns:
        the original image or a rotation improving row concentration by at least 3%
    """
    if image.width < 6 * image.height:
        return image
    counts = (np.asarray(image) < 180).sum(1).astype(np.float64)
    original_score = float((counts**2).sum() / max(1, counts.sum()))
    best, best_score = image, original_score
    for angle in (-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2):
        rotated = image.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=255)
        counts = (np.asarray(rotated) < 180).sum(1).astype(np.float64)
        score = float((counts**2).sum() / max(1, counts.sum()))
        if score > best_score:
            best, best_score = rotated, score
    return best if best_score >= 1.03 * original_score else image


def image_tensor(image, *, character=False, deskew=False):
    """Normalize polarity and height; pad width without distorting glyph proportions.

    Returns:
        float32 image tensor with width divisible by four and values in [-1, 1]
    """
    image = image.convert("L")
    if float(np.asarray(image).mean()) < 127:
        image = Image.fromarray(255 - np.asarray(image))
    if deskew and not character:
        image = deskew_line(image)
    if not character:
        mask = Image.fromarray((np.asarray(image) < 200).astype(np.uint8) * 255)
        bounds = mask.getbbox()
        if bounds is not None:
            image = image.crop(bounds)
        margin = max(2, round(image.height * 0.15))
        canvas = Image.new("L", (image.width + 2 * margin, image.height + 2 * margin), 255)
        canvas.paste(image, (margin, margin))
        image = canvas
    width = max(4, round(image.width * HEIGHT / image.height))
    image = image.resize((width, HEIGHT), Image.Resampling.LANCZOS)
    padded_width = max(32 if character else 4, (width + 3) // 4 * 4)
    canvas = Image.new("L", (padded_width, HEIGHT), 255)
    canvas.paste(image, ((padded_width - width) // 2 if character else 0, 0))
    return torch.from_numpy(np.array(canvas, dtype=np.float32)).unsqueeze(0) / 127.5 - 1


def sample_text(rng, alphabet, max_length):
    """Mix arbitrary strings with document fields; never require a dictionary to decode.

    Returns:
        a nonempty text value drawn entirely from the alphabet
    """
    if rng.random() < 0.5:
        fields = [
            f"{rng.randint(1, 9999)}.{rng.randrange(100):02d}",
            f"{rng.randint(2020, 2030)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(3, 12))),
        ]
        text = f"{rng.choice(FIELD_NAMES)}: {rng.choice(fields)}"
        if all(char in alphabet for char in text) and len(text) <= max_length:
            return text
    length = rng.randint(1, max_length)
    text = "".join(rng.choices(alphabet, k=length)).strip()
    return text or rng.choice(alphabet.replace(" ", ""))


class SyntheticTextDataset(Dataset):
    """Index/epoch-seeded samples reproducible across access order and worker counts."""

    def __init__(self, records, codec, samples, *, seed=0, augment=False, task="lines", max_length=24, deskew=False):
        if samples <= 0 or max_length <= 0 or not records:
            raise ValueError("samples, max_length, and fonts must be nonempty")
        if task not in {"characters", "lines"}:
            raise ValueError("task must be characters or lines")
        self.records, self.codec, self.samples = records, codec, samples
        self.seed, self.augment, self.task, self.max_length = seed, augment, task, max_length
        self.deskew = deskew
        self.epoch = 0
        self.visible = codec.alphabet.replace(" ", "")
        self.families = {}
        for record in records:
            self.families.setdefault(record.family, []).append(record)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = random.Random(self.seed + index * 1_000_003 + self.epoch * 1_000_000_007)  # noqa: S311
        family = rng.choice(list(self.families.values()))
        record = rng.choice(family)
        text = (
            self.visible[index % len(self.visible)]
            if self.task == "characters"
            else sample_text(rng, self.codec.alphabet, self.max_length)
        )
        image = image_tensor(
            render_line(text, record, rng, augment=self.augment),
            character=self.task == "characters",
            deskew=self.deskew,
        )
        return image, text


def collate_lines(samples):
    """Pad on white; preserve exact per-example time lengths for CTC and decoding.

    Returns:
        image batch, unpadded time lengths, and ordered ground-truth strings
    """
    images, texts = zip(*samples, strict=True)
    widths = torch.tensor([image.shape[-1] for image in images], dtype=torch.long)
    batch = torch.ones(len(images), 1, HEIGHT, int(widths.max()))
    for index, image in enumerate(images):
        batch[index, :, :, : image.shape[-1]] = image
    return batch, widths // 4, list(texts)


@dataclass
class Page:
    """A generated page and its ground truth, used only for scoring."""

    image: Image.Image
    text: str
    boxes: list[tuple[int, int, int, int]]


def render_page(records, codec, seed, *, augment=False, max_length=24):
    rng = random.Random(seed)  # noqa: S311
    images, texts = [], []
    for index in range(rng.randint(3, 6)):
        text = sample_text(rng, codec.alphabet, max_length)
        render_rng = random.Random(seed + 10_000_019 * (index + 1))  # noqa: S311
        images.append(render_line(text, rng.choice(records), render_rng, augment=augment))
        texts.append(text)
    width = max(image.width for image in images) + 48
    canvas = Image.new("L", (width, len(images) * 48 + 32), 255)
    boxes = []
    for index, image in enumerate(images):
        x, y = rng.randint(16, 28), 16 + index * 48
        canvas.paste(image, (x, y))
        boxes.append((x, y, x + image.width, y + image.height))
    return Page(canvas, "\n".join(texts), boxes)
