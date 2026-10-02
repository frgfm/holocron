# holocron.utils

`holocron.utils` provides some utilities for general usage.

## Synthetic text rendering

Load fonts once, then render tight grayscale masks for characters or complete
sequences. Font discovery and corpus preparation happen before the sampling
loop; resizing and stochastic effects remain regular TorchVision transforms.

The library performs no network access. To prepare the pinned starter manifest
from the repository, run:

```console
uv run --python 3.12 scripts/prepare_fonts.py --output /tmp/holocron-fonts
```

Use `find_fonts(text)` instead when only installed system fonts are needed.

```python
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import ImageFont
from torchvision.transforms import v2

from holocron.utils import render_text

font_paths = tuple(str(path) for path in sorted(Path("/tmp/holocron-fonts").glob("*.ttf")))[:3]
fonts = [ImageFont.truetype(path, 48) for path in font_paths]

augment = v2.Compose([
    v2.Pad(4, fill=0),
    v2.Resize(48),
    v2.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    v2.RandomInvert(p=0.2),
])

fig, axes = plt.subplots(2, 3, constrained_layout=True)
for ax, text, font in zip(axes.flat, ["A", "OCR-42"] * 3, cycle(fonts)):
    ax.imshow(augment(render_text(text, font)), cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
plt.show()
```

The downloadable fonts come from the
[Google Fonts repository](https://github.com/google/fonts) under the SIL Open
Font License.

::: holocron.utils
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - render_text
            - find_fonts

## Miscellaneous

::: holocron.utils
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - parallel
            - find_image_size
