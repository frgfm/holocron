# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Regenerate the experiment's learning curves from its measured training histories."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(args):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    character = json.loads((args.results / "characters-training.json").read_text())["history"]
    axes[0].plot(
        [row["epoch"] for row in character],
        [100 * row["validation_exact_match"] for row in character],
        marker="o",
        markersize=3,
    )
    axes[0].set(
        xlabel="Character epoch (8192 glyphs)",
        ylabel="Validation accuracy (%)",
        title="Baseline-preserving character pretraining",
        ylim=(95, 100.1),
    )
    for label, filename, offset in [
        ("Scratch, clean training", "baseline-training.json", 0),
        ("Character transfer + augmentation", "transfer-training.json", 0),
        ("Longer-line refinement", "long-lines-training.json", 40_960),
        ("Deskew-aware refinement", "deskew-training.json", 67_584),
    ]:
        path = args.results / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        rows = payload["history"]
        samples_per_epoch = payload["configuration"]["samples_per_epoch"]
        axes[1].plot(
            [(row["epoch"] * samples_per_epoch + offset) / 1000 for row in rows],
            [100 * row["validation_cer"] for row in rows],
            label=label,
            marker="o",
            markersize=3,
        )
    axes[1].set(
        xlabel="Sequence training examples (thousands)",
        ylabel="Validation CER (%)",
        title="Sequence recognition (stage-specific validation)",
        yscale="log",
    )
    axes[1].axvline(40.96, linestyle="--", color="grey", alpha=0.5)
    axes[1].axvline(67.584, linestyle="--", color="grey", alpha=0.5)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "learning-curves.png")
    main(parser.parse_args())
