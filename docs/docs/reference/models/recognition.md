# Text recognition

Holocron provides an experimental shared character backbone, glyph classifier,
and CNN/BiGRU recognizer trained with connectionist temporal classification
(CTC). Import them from `holocron.models.recognition`; training and synthetic
benchmark generation are provided by the
[recognition reference](https://github.com/frgfm/holocron/tree/main/references/recognition).

The models accept grayscale tensors of shape `(N, 1, 32, W)`, normalized to
`[-1, 1]`, with white padding equal to `+1` and width divisible by four.
`lengths` contains each image's unpadded width divided by four. The recognizer
returns log probabilities of shape `(W // 4, N, alphabet_size + 1)` with CTC
blank at index zero. Loss and decoding must use the true lengths.

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

`CTCCodec` accepts a custom printable Unicode alphabet, including ordinary
space. It preserves predicted punctuation and repeats separated by blanks.
`holocron.utils.prefix_beam_decode` optionally sums alternative alignments;
it uses no dictionary or language model. Greedy decoding remains the measured
default.

The synthetic curriculum benchmarks 83 symbols across ten font families,
holding two families out of every training stage. Its final degraded-page CER
is 1.06% on both seen and held-out families, with 64% and 53% exact pages.
The [full report](https://github.com/frgfm/holocron/blob/main/references/recognition/RESULTS.md)
includes the control, training budgets, seeds, and residual errors. No public
pretrained checkpoint is downloaded. These measurements cover separated,
horizontal, single-column synthetic pages; real scans and general layouts
remain unvalidated. Page detection and synthetic rendering remain experimental
reference code.

::: holocron.models.recognition.crnn.CharacterBackbone
    options:
        members: no
        show_bases: false

::: holocron.models.recognition.crnn.CharacterClassifier
    options:
        members: no
        show_bases: false

::: holocron.models.recognition.crnn.CTCRecognizer
    options:
        members: no
        show_bases: false
