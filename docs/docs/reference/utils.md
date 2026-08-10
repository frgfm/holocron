# holocron.utils

`holocron.utils` provides some utilities for general usage.

## Synthetic text rendering

Load fonts once, then render tight grayscale masks for characters or complete
sequences. Font discovery and the optional download happen before the sampling
loop; resizing and stochastic effects remain regular TorchVision transforms.

```python
from itertools import cycle

import matplotlib.pyplot as plt
from PIL import ImageFont
from torchvision.transforms import v2

from holocron.utils import find_fonts, render_text

font_paths = find_fonts("OCR-42é")[:3]
# For a reproducible five-family OFL pack instead:
# from holocron.utils import download_fonts
# font_paths = download_fonts()
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
            - download_fonts

## Miscellaneous

::: holocron.utils
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - parallel
            - find_image_size
