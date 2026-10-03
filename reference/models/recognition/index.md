# Text recognition

Holocron provides an experimental shared character backbone, glyph classifier, and CNN/BiGRU recognizer trained with connectionist temporal classification (CTC). Import them from `holocron.models.recognition`; training and synthetic benchmark generation are provided by the [recognition reference](https://github.com/frgfm/holocron/tree/main/references/recognition).

The models accept grayscale tensors of shape `(N, 1, 32, W)`, normalized to `[-1, 1]`, with white padding equal to `+1` and width divisible by four. `lengths` contains each image's unpadded width divided by four. The recognizer returns log probabilities of shape `(W // 4, N, alphabet_size + 1)` with CTC blank at index zero. Loss and decoding must use the true lengths.

```python
import torch
from holocron.models.recognition import CharacterClassifier, CTCRecognizer
from holocron.utils import CTCCodec

codec = CTCCodec("AB12 .")
classifier = CharacterClassifier(num_classes=len(codec.alphabet))
reader = CTCRecognizer(num_classes=len(codec.alphabet))
# After glyph pretraining, transfer the visual encoder into the line model.
reader.backbone.load_state_dict(classifier.backbone.state_dict())

# Example input contract; train or load weights before using predictions.
images = torch.ones(2, 1, 32, 80)
lengths = torch.tensor([20, 12])
reader.eval()
with torch.inference_mode():
    log_probabilities = reader(images, lengths)
    labels = log_probabilities.argmax(-1).transpose(0, 1)
    transcripts = [codec.decode(row[:length].tolist()) for row, length in zip(labels, lengths)]
```

`CTCCodec` accepts a custom printable Unicode alphabet, including ordinary space. It preserves predicted punctuation and repeats separated by blanks. `holocron.utils.prefix_beam_decode` optionally sums alternative alignments; it uses no dictionary or language model. Greedy decoding remains the measured default.

The synthetic curriculum benchmarks 83 symbols across ten font families, holding two families out of every training stage. Its final degraded-page CER is 1.06% on both seen and held-out families, with 64% and 53% exact pages. The [full report](https://github.com/frgfm/holocron/blob/main/references/recognition/RESULTS.md) includes the control, training budgets, seeds, and residual errors. No public pretrained checkpoint is downloaded. These measurements cover separated, horizontal, single-column synthetic pages; real scans and general layouts remain unvalidated. Page detection and synthetic rendering remain experimental reference code.

## CharacterBackbone

```python
CharacterBackbone()
```

Shared grayscale encoder for glyph classification and line recognition.

Input images have shape `(N, 1, 32, W)` with widths divisible by four. The output has shape `(N, 96, 4, W // 4)`. Normalize grayscale images to [-1, 1], using +1 for white padding.

Source code in `holocron/models/recognition/crnn.py`

```python
def __init__(self) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    channels = 1
    for output, pool in [(24, (2, 2)), (48, (2, 2)), (96, (2, 1)), (96, None)]:
        layers.extend([nn.Conv2d(channels, output, 3, padding=1, bias=False), nn.BatchNorm2d(output), nn.ReLU()])
        if pool is not None:
            layers.append(nn.MaxPool2d(pool))
        channels = output
    self.features = nn.Sequential(*layers)
```

## CharacterClassifier

```python
CharacterClassifier(num_classes: int)
```

Classify glyphs while retaining their vertical baseline position.

| PARAMETER     | DESCRIPTION                                        |
| ------------- | -------------------------------------------------- |
| `num_classes` | number of output character classes **TYPE:** `int` |

The input contract matches :class:`CharacterBackbone`. `lengths` is a one-dimensional integer tensor containing each image's unpadded width divided by four. Returns logits of shape `(N, num_classes)`. Transfer `backbone.state_dict()` directly to :class:`CTCRecognizer`.

Source code in `holocron/models/recognition/crnn.py`

```python
def __init__(self, num_classes: int) -> None:
    super().__init__()
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    self.backbone = CharacterBackbone()
    self.head = nn.Linear(384, num_classes)
```

## CTCRecognizer

```python
CTCRecognizer(num_classes: int)
```

Compact CNN/BiGRU recognizer for variable-width text lines.

| PARAMETER     | DESCRIPTION                                           |
| ------------- | ----------------------------------------------------- |
| `num_classes` | alphabet size excluding the CTC blank **TYPE:** `int` |

The input contract matches :class:`CharacterBackbone`. `lengths` is a one-dimensional integer tensor of unpadded widths divided by four. Packed recurrence excludes padding from bidirectional context. Returns log probabilities of shape `(W // 4, N, num_classes + 1)` with blank at index zero. Ignore frames beyond each length during loss and decoding. The architecture is experimental; no pretrained weights are downloaded.

Source code in `holocron/models/recognition/crnn.py`

```python
def __init__(self, num_classes: int) -> None:
    super().__init__()
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    self.backbone = CharacterBackbone()
    self.projection = nn.Sequential(nn.Linear(384, 128), nn.ReLU())
    self.context = nn.GRU(128, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
    self.head = nn.Linear(192, num_classes + 1)
```
