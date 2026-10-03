# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Evaluate character, line, and full-page transcripts on independently seeded data."""

import argparse
import json
import platform
from collections import Counter
from itertools import starmap
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader

from holocron.models.recognition import CharacterClassifier, CTCRecognizer
from holocron.utils import CTCCodec, prefix_beam_decode
from references.classification.train_characters import _sha256, resolve_font_records
from references.recognition.data import (
    FIELD_NAMES,
    SyntheticTextDataset,
    collate_lines,
    image_tensor,
    inspect_fonts,
    render_page,
)


def edit_distance(reference, prediction):
    """Levenshtein distance for either Unicode characters or whitespace-separated words.

    Returns:
        minimum insertion, deletion, and substitution count
    """
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def transcript_metrics(references, predictions):
    """Micro-averaged CER/WER, exact matches, and inspectable error examples.

    Returns:
        metrics and up to twenty transcript errors

    Raises:
        ValueError: if the lists have different lengths or are empty
    """
    if len(references) != len(predictions) or not references:
        raise ValueError("matching nonempty reference and prediction lists are required")
    errors = sum(starmap(edit_distance, zip(references, predictions, strict=True)))
    characters = sum(map(len, references))
    word_errors = sum(
        edit_distance(ref.split(), pred.split()) for ref, pred in zip(references, predictions, strict=True)
    )
    words = sum(len(ref.split()) for ref in references)
    exact = sum(ref == pred for ref, pred in zip(references, predictions, strict=True))
    return {
        "samples": len(references),
        "characters": characters,
        "character_errors": errors,
        "cer": errors / max(1, characters),
        "words": words,
        "word_errors": word_errors,
        "wer": word_errors / max(1, words),
        "exact_match": exact / len(references),
        "errors": [
            {"reference": ref, "prediction": pred}
            for ref, pred in zip(references, predictions, strict=True)
            if ref != pred
        ][:20],
    }


@torch.inference_mode()
def predict_lines(model, codec, images, lengths, device, beam_width=0):
    probabilities = model(images.to(device), lengths)
    if beam_width:
        probabilities = probabilities.transpose(0, 1).cpu().numpy()
        return [
            prefix_beam_decode(row[:length], codec, beam_width)
            for row, length in zip(probabilities, lengths.tolist(), strict=True)
        ]
    indices = probabilities.argmax(-1).transpose(0, 1).cpu().tolist()
    return [codec.decode(row[:length]) for row, length in zip(indices, lengths.tolist(), strict=True)]


@torch.inference_mode()
def evaluate_lines(model, codec, loader, device, beam_width=0):
    model.eval()
    references, predictions = [], []
    for images, lengths, texts in loader:
        references.extend(texts)
        predictions.extend(predict_lines(model, codec, images, lengths, device, beam_width))
    result = transcript_metrics(references, predictions)
    result["by_content"] = {}
    for name, is_field in (("fields", True), ("random", False)):
        pairs = [
            (ref, pred)
            for ref, pred in zip(references, predictions, strict=True)
            if (ref.split(":")[0] in FIELD_NAMES) == is_field
        ]
        if pairs:
            refs, preds = zip(*pairs, strict=True)
            result["by_content"][name] = transcript_metrics(refs, preds)
    return result


@torch.inference_mode()
def evaluate_characters(model, codec, loader, device):
    model.eval()
    references, predictions = [], []
    for images, lengths, texts in loader:
        indices = model(images.to(device), lengths).argmax(-1).cpu().tolist()
        references.extend(texts)
        predictions.extend(codec.alphabet[index] for index in indices)
    metrics = transcript_metrics(references, predictions)
    metrics["accuracy"] = metrics["exact_match"]
    metrics["confusions"] = [
        {"reference": ref, "prediction": pred, "count": count}
        for (ref, pred), count in Counter(
            (ref, pred) for ref, pred in zip(references, predictions, strict=True) if ref != pred
        ).most_common(20)
    ]
    return metrics


def detect_lines(image):
    """Projection-based line detection for separated, horizontal, single-column text.

    Returns:
        inferred bounding boxes in top-to-bottom reading order; no ground-truth boxes are used
    """
    array = np.asarray(image.convert("L"))
    if array.mean() < 127:
        array = 255 - array
    ink = array < 200
    active = np.flatnonzero(ink.any(1))
    if not len(active):
        return []
    splits = np.flatnonzero(np.diff(active) > 4) + 1
    groups = list(np.split(active, splits))
    heights = [int(rows[-1] - rows[0]) + 1 for rows in groups]
    typical_height = float(np.median([height for height in heights if height >= 8])) if max(heights) >= 8 else 20.0
    max_gap = max(12, round(typical_height * 0.45))
    merged = [groups[0]]
    for rows in groups[1:]:
        previous = merged[-1]
        small_fragment = min(int(previous[-1] - previous[0]) + 1, int(rows[-1] - rows[0]) + 1) < typical_height * 0.65
        if rows[0] - previous[-1] <= max_gap and small_fragment:
            merged[-1] = np.concatenate((previous, rows))
        else:
            merged.append(rows)
    boxes = []
    for rows in merged:
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        columns = np.flatnonzero(ink[top:bottom].any(0))
        if len(columns):
            boxes.append((
                max(0, int(columns[0]) - 2),
                max(0, top - 2),
                min(image.width, int(columns[-1]) + 3),
                min(image.height, bottom + 2),
            ))
    return boxes


def transcribe_page(model, codec, image, device="cpu", batch_size=32, beam_width=0, deskew=None):
    """Detect lines and recognize them in reading order using only page pixels.

    Returns:
        full newline-separated transcript and inferred line boxes
    """
    model.eval()
    if deskew is None:
        deskew = getattr(model, "deskew", False)
    boxes = detect_lines(image)
    predictions = []
    for start in range(0, len(boxes), batch_size):
        samples = [(image_tensor(image.crop(box), deskew=deskew), "") for box in boxes[start : start + batch_size]]
        images, lengths, _ = collate_lines(samples)
        predictions.extend(predict_lines(model, codec, images, lengths, device, beam_width))
    return "\n".join(predictions), boxes


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    codec = CTCCodec(checkpoint["alphabet"])
    model = (
        CharacterClassifier(len(codec.alphabet))
        if checkpoint["task"] == "characters"
        else CTCRecognizer(len(codec.alphabet))
    )
    model.load_state_dict(checkpoint["model"])
    model.deskew = checkpoint["configuration"].get("deskew", False)
    return model.to(device).eval(), codec, checkpoint


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--pages", type=int, default=100)
    parser.add_argument("--seed", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=0, help="prefix beam search width; zero uses greedy decoding")
    parser.add_argument(
        "--deskew", action=argparse.BooleanOptionalAction, default=None, help="override checkpoint line normalization"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="transcribe a real single-column page instead of benchmarking")
    return parser


def main(args):
    if min(args.samples, args.batch_size, args.threads, args.max_length) <= 0 or min(args.pages, args.beam_width) < 0:
        raise ValueError("counts must be positive; pages can be zero")
    torch.set_num_threads(args.threads)
    model, codec, checkpoint = load_checkpoint(args.checkpoint, args.device)
    if args.deskew is None:
        args.deskew = model.deskew
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.image is not None:
        if checkpoint["task"] != "lines":
            raise ValueError("page transcription requires a line model")
        with Image.open(args.image) as image:
            text, boxes = transcribe_page(
                model, codec, image, args.device, args.batch_size, args.beam_width, args.deskew
            )
        args.output.write_text(json.dumps({"text": text, "boxes": boxes}, indent=2) + "\n", encoding="utf-8")
        print(text)
        return
    records, _ = resolve_font_records(codec.alphabet.replace(" ", ""), args.font_dir, args.manifest)
    records = inspect_fonts(records, codec.alphabet)
    train_families = {record["family"] for record in checkpoint["fonts"]}
    groups = {
        "seen": tuple(record for record in records if record.family in train_families),
        "unseen": tuple(record for record in records if record.family not in train_families),
    }
    result = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "task": checkpoint["task"],
        "seed": args.seed,
        "max_length": args.max_length,
        "alphabet": codec.alphabet,
        "beam_width": args.beam_width,
        "deskew": args.deskew,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "pillow": pillow_version,
            "numpy": np.__version__,
            "device": args.device,
            "threads": args.threads,
            "batch_size": args.batch_size,
        },
        "benchmarks": {},
    }
    for name, group in groups.items():
        if not group:
            continue
        for augment in (False, True):
            key = f"{name}_{'degraded' if augment else 'clean'}"
            dataset = SyntheticTextDataset(
                group,
                codec,
                args.samples,
                seed=args.seed,
                augment=augment,
                task=checkpoint["task"],
                max_length=args.max_length,
                deskew=args.deskew,
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_lines)
            evaluate = evaluate_characters if checkpoint["task"] == "characters" else evaluate_lines
            metrics = (
                evaluate(model, codec, loader, args.device, args.beam_width)
                if checkpoint["task"] == "lines"
                else evaluate(model, codec, loader, args.device)
            )
            result["benchmarks"][key] = {
                "families": sorted({record.family for record in group}),
                "fonts": [{"file": record.path.name, "sha256": _sha256(record.path)} for record in group],
                **metrics,
            }
            print(f"{key}: CER={metrics['cer']:.4%}, exact={metrics['exact_match']:.2%}", flush=True)
            if checkpoint["task"] == "lines" and args.pages:
                references, predictions, line_errors = [], [], []
                for index in range(args.pages):
                    page = render_page(
                        group, codec, args.seed + 1_000_000 + index, augment=augment, max_length=args.max_length
                    )
                    text, boxes = transcribe_page(
                        model, codec, page.image, args.device, args.batch_size, args.beam_width, args.deskew
                    )
                    references.append(page.text)
                    predictions.append(text)
                    line_errors.append(abs(len(boxes) - len(page.boxes)))
                    if index == 0:
                        page.image.save(args.output.with_name(f"{args.output.stem}-{key}.png"))
                page_metrics = transcript_metrics(references, predictions)
                page_metrics["line_count_errors"] = sum(line_errors)
                result["benchmarks"][key + "_pages"] = page_metrics
                print(
                    f"{key} pages: CER={page_metrics['cer']:.4%}, exact={page_metrics['exact_match']:.2%}, line errors={sum(line_errors)}",
                    flush=True,
                )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(get_parser().parse_args())
