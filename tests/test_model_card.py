import hashlib
import json
from pathlib import Path

import pytest

from holocron.models import list_checkpoints
from holocron.models.model_card import generate_model_card, main, write_model_cards


def _write_manifest(root: Path, schema_version: int = 1) -> None:
    artifact = root / "metrics.jsonl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"epoch": 2, "val_loss": 0.25}\n', encoding="utf-8")
    manifest = {
        "schema_version": schema_version,
        "run": {
            "epoch": 2,
            "step": 4,
            "best_metric": 0.25,
            "config": {"optimizer": {"lr": 0.01}, "seed": 7},
        },
        "artifacts": [
            {
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "size": artifact.stat().st_size,
            }
        ],
    }
    (root / "manifest.json").write_text(f"{json.dumps(manifest)}\n", encoding="utf-8")


def test_generate_model_card_is_deterministic() -> None:
    checkpoint = list_checkpoints("convnext_atto")[0]

    card = generate_model_card("convnext_atto", checkpoint)

    assert card == generate_model_card("convnext_atto", checkpoint)
    for fragment in (
        "library_name: holocron",
        "pipeline_tag: image-classification",
        "license: apache-2.0",
        "datasets:\n  - imagenette",
        "| Maturity | preview |",
        f"`{checkpoint.meta.sha256}`",
        "## Intended use",
        "## Limitations",
        "Run bundle: **unknown / not reported**",
    ):
        assert fragment in card


def test_generate_model_card_with_run_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir)

    card = generate_model_card("convnext_atto", list_checkpoints("convnext_atto")[0], run_dir)

    assert card == generate_model_card("convnext_atto", list_checkpoints("convnext_atto")[0], run_dir)
    assert '"seed": 7' in card
    assert "| metrics.jsonl |" in card
    assert hashlib.sha256((run_dir / "metrics.jsonl").read_bytes()).hexdigest() in card

    (run_dir / "metrics.jsonl").write_text('{"epoch": 2, "val_loss": 0.26}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generate_model_card("convnext_atto", list_checkpoints("convnext_atto")[0], run_dir)


def test_generate_model_card_rejects_unfinished_or_invalid_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unfinished run bundle"):
        generate_model_card("convnext_atto", run_dir=tmp_path)

    _write_manifest(tmp_path, schema_version=2)
    with pytest.raises(ValueError, match="unsupported run manifest schema_version"):
        generate_model_card("convnext_atto", run_dir=tmp_path)


def test_write_model_cards_and_cli(tmp_path: Path) -> None:
    written = write_model_cards(tmp_path / "cards", ["convnext_atto"])
    assert len(written) == 1
    assert written[0].name.startswith("convnext_atto-imagenette-")

    output = tmp_path / "README.md"
    assert main(["convnext_atto", "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith("---\n")

    with pytest.raises(SystemExit):
        main(["convnext_atto", "--checkpoint", "-1", "--output", str(output)])
