# Generate model cards

Holocron can generate deterministic, Hugging Face-compatible Markdown from its typed model catalog. Cards include the
model task and maturity, checkpoint provenance, preprocessing, training recipe, evaluation metrics, and explicit
`unknown / not reported` markers for metadata Holocron does not have.

Generate one checkpoint card:

```shell
python -m holocron.models.model_card convnext_atto --checkpoint 0 --output README.md
```

Attach provenance from a completed schema-v1 run bundle:

```shell
python -m holocron.models.model_card convnext_atto --output README.md --run-dir runs/experiment-1
```

The run directory must contain `manifest.json` and every artifact declared by the manifest. The generator verifies the
schema version, file sizes, and SHA-256 values before rendering the card. A missing manifest is treated as an unfinished
run and rejected.

Generate one card per typed checkpoint of selected models (or one model-only card when typed checkpoint metadata is
unavailable):

```shell
python -m holocron.models.model_card convnext_atto resnet18 --output-dir model-cards
```

Omit model names to generate the complete catalog. Generated cards are release artifacts; they are not maintained as a
second metadata store in the repository.

The Python API is available when a release workflow needs the Markdown directly:

```python
from holocron.models import list_checkpoints
from holocron.models.model_card import generate_model_card

checkpoint = list_checkpoints("convnext_atto")[0]
markdown = generate_model_card("convnext_atto", checkpoint, run_dir="runs/experiment-1")
```
