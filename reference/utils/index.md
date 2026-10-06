# holocron.utils

`holocron.utils` provides some utilities for general usage.

## Synthetic text rendering

Load fonts once, then render tight grayscale masks for characters or complete sequences. Font discovery and corpus preparation happen before the sampling loop; resizing and stochastic effects remain regular TorchVision transforms.

The library performs no network access. To prepare the pinned starter manifest from the repository, run:

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

The downloadable fonts come from the [Google Fonts repository](https://github.com/google/fonts) under the SIL Open Font License.

### render_text

```python
render_text(text: str, font: FreeTypeFont, *, padding: int = 0, background_color: int = 0, text_color: int = 255) -> Image
```

Render text as a tight grayscale image.

| PARAMETER          | DESCRIPTION                                                                      |
| ------------------ | -------------------------------------------------------------------------------- |
| `text`             | non-empty Unicode text to render **TYPE:** `str`                                 |
| `font`             | preloaded font used for rasterization **TYPE:** `FreeTypeFont`                   |
| `padding`          | number of background pixels added on every side **TYPE:** `int` **DEFAULT:** `0` |
| `background_color` | grayscale background value **TYPE:** `int` **DEFAULT:** `0`                      |
| `text_color`       | grayscale text value **TYPE:** `int` **DEFAULT:** `255`                          |

| RETURNS | DESCRIPTION              |
| ------- | ------------------------ |
| `Image` | rendered grayscale image |

| RAISES       | DESCRIPTION                                     |
| ------------ | ----------------------------------------------- |
| `TypeError`  | if text, font, or padding has an invalid type   |
| `ValueError` | if the arguments cannot produce a visible image |

Source code in `holocron/utils/fonts.py`

```python
def render_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    padding: int = 0,
    background_color: int = 0,
    text_color: int = 255,
) -> Image.Image:
    """Render text as a tight grayscale image.

    Args:
        text: non-empty Unicode text to render
        font: preloaded font used for rasterization
        padding: number of background pixels added on every side
        background_color: grayscale background value
        text_color: grayscale text value

    Returns:
        rendered grayscale image

    Raises:
        TypeError: if `text`, `font`, or `padding` has an invalid type
        ValueError: if the arguments cannot produce a visible image
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        raise ValueError("text must not be empty")
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise TypeError("font must be a PIL.ImageFont.FreeTypeFont")
    if isinstance(padding, bool) or not isinstance(padding, int):
        raise TypeError("padding must be an integer")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    _validate_color("background_color", background_color)
    _validate_color("text_color", text_color)
    if background_color == text_color:
        raise ValueError("background_color and text_color must differ")

    mask, _ = font.getmask2(text, mode="L")
    # Pillow has no public zero-copy wrapper for its rasterized ImagingCore mask.
    image = Image.Image()._new(mask)  # noqa: SLF001
    if image.getbbox() is None:
        raise ValueError("text does not contain a visible glyph")

    if padding == 0 and background_color == 0 and text_color == 255:
        return image

    output = Image.new("L", (image.width + 2 * padding, image.height + 2 * padding), background_color)
    output.paste(text_color, (padding, padding), image)
    return output
```

### find_fonts

```python
find_fonts(text: str | None = None) -> tuple[str, ...]
```

Find loadable system fonts that cover the requested text.

Matplotlib-bundled fonts are included so discovery does not depend on the host having user-installed fonts.

| PARAMETER | DESCRIPTION                                                                      |
| --------- | -------------------------------------------------------------------------------- |
| `text`    | optional text whose non-whitespace code points must be supported **TYPE:** \`str |

| RETURNS           | DESCRIPTION                     |
| ----------------- | ------------------------------- |
| `tuple[str, ...]` | sorted, deduplicated font paths |

| RAISES      | DESCRIPTION                          |
| ----------- | ------------------------------------ |
| `TypeError` | if text is neither a string nor None |

Source code in `holocron/utils/fonts.py`

```python
@lru_cache(maxsize=128)
def find_fonts(text: str | None = None) -> tuple[str, ...]:
    """Find loadable system fonts that cover the requested text.

    Matplotlib-bundled fonts are included so discovery does not depend on the
    host having user-installed fonts.

    Args:
        text: optional text whose non-whitespace code points must be supported

    Returns:
        sorted, deduplicated font paths

    Raises:
        TypeError: if `text` is neither a string nor `None`
    """
    if text is not None and not isinstance(text, str):
        raise TypeError("text must be a string or None")

    required_codepoints = {ord(char) for char in text or "" if not char.isspace()}
    font_paths = {str(entry.fname) for entry in font_manager.fontManager.ttflist}
    font_paths.update(font_manager.findSystemFonts())

    available = []
    for font_path in sorted(font_paths):
        if not Path(font_path).is_file():
            continue
        try:
            ImageFont.truetype(font_path, 10)
            if required_codepoints and not required_codepoints.issubset(FT2Font(font_path).get_charmap()):
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        available.append(font_path)
    return tuple(available)
```

## CTC decoding

Encode target characters and decode line-model predictions using an explicit alphabet. See the [recognition models](../models/recognition/) for the image and sequence-length contract.

### recognition

Alphabet encoding and lexicon-free CTC transcript decoding.

#### CTCCodec

```python
CTCCodec(alphabet: str)
```

Map an ordered alphabet to CTC labels, reserving index zero for blank.

| PARAMETER  | DESCRIPTION                                                                    |
| ---------- | ------------------------------------------------------------------------------ |
| `alphabet` | unique printable Unicode characters; ordinary space is allowed **TYPE:** `str` |

| RAISES       | DESCRIPTION                                                              |
| ------------ | ------------------------------------------------------------------------ |
| `ValueError` | if the alphabet is empty, duplicated, or contains unsupported whitespace |

Source code in `holocron/utils/recognition.py`

```python
def __init__(self, alphabet: str) -> None:
    if not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must be nonempty with unique characters")
    if any(not char.isprintable() or (char.isspace() and char != " ") for char in alphabet):
        raise ValueError("only printable characters and ordinary spaces are supported")
    if not alphabet.replace(" ", ""):
        raise ValueError("alphabet must contain a visible character")
    self.alphabet = alphabet
    self.indices = {char: index + 1 for index, char in enumerate(alphabet)}
```

##### encode

```python
encode(text: str) -> Tensor
```

Encode text without inserting CTC blanks.

| PARAMETER | DESCRIPTION                                                 |
| --------- | ----------------------------------------------------------- |
| `text`    | characters drawn from this codec's alphabet **TYPE:** `str` |

| RETURNS  | DESCRIPTION                                       |
| -------- | ------------------------------------------------- |
| `Tensor` | one-dimensional CPU int64 tensor of target labels |

| RAISES       | DESCRIPTION                                |
| ------------ | ------------------------------------------ |
| `ValueError` | if a character is absent from the alphabet |

Source code in `holocron/utils/recognition.py`

```python
def encode(self, text: str) -> Tensor:
    """Encode text without inserting CTC blanks.

    Args:
        text: characters drawn from this codec's alphabet

    Returns:
        one-dimensional CPU int64 tensor of target labels

    Raises:
        ValueError: if a character is absent from the alphabet
    """
    if any(char not in self.indices for char in text):
        raise ValueError("text contains a character outside the alphabet")
    return torch.tensor([self.indices[char] for char in text], dtype=torch.long)
```

##### decode

```python
decode(indices: Iterable[int]) -> str
```

Collapse a greedy CTC alignment, preserving repeats separated by blanks.

| PARAMETER | DESCRIPTION                                                                               |
| --------- | ----------------------------------------------------------------------------------------- |
| `indices` | consecutive frame labels, truncated to the true sequence length **TYPE:** `Iterable[int]` |

| RETURNS | DESCRIPTION                                                    |
| ------- | -------------------------------------------------------------- |
| `str`   | decoded transcript, including predicted spaces and punctuation |

| RAISES       | DESCRIPTION                                           |
| ------------ | ----------------------------------------------------- |
| `ValueError` | if an index is outside the blank/alphabet label range |

Source code in `holocron/utils/recognition.py`

```python
def decode(self, indices: Iterable[int]) -> str:
    """Collapse a greedy CTC alignment, preserving repeats separated by blanks.

    Args:
        indices: consecutive frame labels, truncated to the true sequence length

    Returns:
        decoded transcript, including predicted spaces and punctuation

    Raises:
        ValueError: if an index is outside the blank/alphabet label range
    """
    result = []
    previous = 0
    for index in indices:
        if not 0 <= index <= len(self.alphabet):
            raise ValueError("CTC indices must lie between zero and the alphabet size")
        if index and index != previous:
            result.append(self.alphabet[index - 1])
        previous = index
    return "".join(result)
```

#### prefix_beam_decode

```python
prefix_beam_decode(log_probabilities: ndarray, codec: CTCCodec, beam_width: int = 5, token_topk: int = 8) -> str
```

Approximate the most probable transcript, respecting blanks and repeated letters.

| PARAMETER           | DESCRIPTION                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `log_probabilities` | frame log probabilities of shape (T, alphabet size + 1) **TYPE:** `ndarray`                         |
| `codec`             | alphabet whose blank index is zero **TYPE:** `CTCCodec`                                             |
| `beam_width`        | number of transcript prefixes retained per frame **TYPE:** `int` **DEFAULT:** `5`                   |
| `token_topk`        | maximum nonblank frame labels considered; blank is always included **TYPE:** `int` **DEFAULT:** `8` |

| RETURNS | DESCRIPTION                                                          |
| ------- | -------------------------------------------------------------------- |
| `str`   | the best prefix after summing blank/nonblank alignment probabilities |

| RAISES       | DESCRIPTION                                                 |
| ------------ | ----------------------------------------------------------- |
| `ValueError` | if beam width, token count, or probability shape is invalid |

Source code in `holocron/utils/recognition.py`

```python
def prefix_beam_decode(log_probabilities: np.ndarray, codec: CTCCodec, beam_width: int = 5, token_topk: int = 8) -> str:
    """Approximate the most probable transcript, respecting blanks and repeated letters.

    Args:
        log_probabilities: frame log probabilities of shape (T, alphabet size + 1)
        codec: alphabet whose blank index is zero
        beam_width: number of transcript prefixes retained per frame
        token_topk: maximum nonblank frame labels considered; blank is always included

    Returns:
        the best prefix after summing blank/nonblank alignment probabilities

    Raises:
        ValueError: if beam width, token count, or probability shape is invalid
    """
    if beam_width <= 0 or token_topk <= 0:
        raise ValueError("beam_width and token_topk must be positive")
    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != len(codec.alphabet) + 1:
        raise ValueError("log_probabilities must have shape (T, alphabet size + 1)")
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for frame in log_probabilities:
        top = np.argsort(frame[1:])[-token_topk:] + 1
        candidates: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (blank, nonblank) in beams.items():
            total = _logadd(blank, nonblank)
            old_blank, old_nonblank = candidates.get(prefix, (-math.inf, -math.inf))
            candidates[prefix] = (_logadd(old_blank, total + float(frame[0])), old_nonblank)
            for raw_token in top:
                token = int(raw_token)
                score = float(frame[token])
                repeated = bool(prefix) and prefix[-1] == token
                if repeated:
                    old_blank, old_nonblank = candidates[prefix]
                    candidates[prefix] = (old_blank, _logadd(old_nonblank, nonblank + score))
                extended = (*prefix, token)
                old_blank, old_nonblank = candidates.get(extended, (-math.inf, -math.inf))
                candidates[extended] = (old_blank, _logadd(old_nonblank, (blank if repeated else total) + score))
        beams = dict(sorted(candidates.items(), key=lambda pair: _logadd(*pair[1]), reverse=True)[:beam_width])
    prefix = max(beams, key=lambda prefix: _logadd(*beams[prefix]))
    return "".join(codec.alphabet[index - 1] for index in prefix)
```

## Miscellaneous

### parallel

```python
parallel(func: Callable[[Inp], Out], arr: Sequence[Inp], num_threads: int | None = None, progress: bool = False, **kwargs: Any) -> Iterable[Out]
```

Performs parallel tasks by leveraging multi-threading.

Example

```python
from holocron.utils.misc import parallel
parallel(lambda x: x ** 2, list(range(10)))
```

| PARAMETER     | DESCRIPTION                                                                        |
| ------------- | ---------------------------------------------------------------------------------- |
| `func`        | function to be executed on multiple workers **TYPE:** `Callable[[Inp], Out]`       |
| `arr`         | function argument's values **TYPE:** `Sequence[Inp]`                               |
| `num_threads` | number of workers to be used for multiprocessing **TYPE:** \`int                   |
| `progress`    | whether the progress bar should be displayed **TYPE:** `bool` **DEFAULT:** `False` |
| `kwargs`      | keyword arguments of tqdm.auto.tqdm **TYPE:** `Any` **DEFAULT:** `{}`              |

| RETURNS         | DESCRIPTION                |
| --------------- | -------------------------- |
| `Iterable[Out]` | list of function's results |

Source code in `holocron/utils/misc.py`

````python
def parallel(
    func: Callable[[Inp], Out],
    arr: Sequence[Inp],
    num_threads: int | None = None,
    progress: bool = False,
    **kwargs: Any,
) -> Iterable[Out]:
    """Performs parallel tasks by leveraging multi-threading.

    Example:
        ```python
        from holocron.utils.misc import parallel
        parallel(lambda x: x ** 2, list(range(10)))
        ```

    Args:
        func: function to be executed on multiple workers
        arr: function argument's values
        num_threads: number of workers to be used for multiprocessing
        progress: whether the progress bar should be displayed
        kwargs: keyword arguments of [`tqdm.auto.tqdm`][tqdm.auto.tqdm]

    Returns:
        list of function's results
    """
    num_threads = num_threads if isinstance(num_threads, int) else min(16, mp.cpu_count())
    if num_threads < 2:
        results = list(map(func, tqdm(arr, total=len(arr), **kwargs))) if progress else map(func, arr)
    else:
        with ThreadPool(num_threads) as tp:
            results = list(tqdm(tp.imap(func, arr), total=len(arr), **kwargs)) if progress else tp.map(func, arr)

    return results
````

### find_image_size

```python
find_image_size(dataset: Sequence[tuple[Image, Any]], **kwargs: Any) -> None
```

Computes the best image size target for a given set of images

| PARAMETER  | DESCRIPTION                                                                                        |
| ---------- | -------------------------------------------------------------------------------------------------- |
| `dataset`  | an iterator yielding a PIL.Image.Image and a target object **TYPE:** `Sequence[tuple[Image, Any]]` |
| `**kwargs` | keyword args of matplotlib.pyplot.show **TYPE:** `Any` **DEFAULT:** `{}`                           |

Source code in `holocron/utils/misc.py`

```python
def find_image_size(dataset: Sequence[tuple[Image.Image, Any]], **kwargs: Any) -> None:
    """Computes the best image size target for a given set of images

    Args:
        dataset: an iterator yielding a [`PIL.Image.Image`][PIL.Image.Image] and a target object
        **kwargs: keyword args of [`matplotlib.pyplot.show`][matplotlib.pyplot.show]
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    # Record height & width
    shapes_ = parallel(lambda x: x[0].size, dataset, progress=True)

    shapes = np.asarray(shapes_)[:, ::-1]
    ratios = shapes[:, 0] / shapes[:, 1]
    sides = np.sqrt(shapes[:, 0] * shapes[:, 1])

    # Compute median aspect ratio & side
    median_ratio = np.median(ratios)
    median_side = np.median(sides)

    height = round(median_side * sqrt(median_ratio))
    width = round(median_side / sqrt(median_ratio))

    # Double histogram
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(ratios, bins=30, alpha=0.7)
    axes[0].title.set_text(f"Aspect ratio (median: {median_ratio:.2})")
    axes[0].grid(True, linestyle="--", axis="x")
    axes[0].axvline(median_ratio, color="r")
    axes[1].hist(sides, bins=30, alpha=0.7)
    axes[1].title.set_text(f"Side (median: {int(median_side)})")
    axes[1].grid(True, linestyle="--", axis="x")
    axes[1].axvline(median_side, color="r")
    fig.suptitle(f"Median image size: ({height}, {width})")
    plt.show(**kwargs)
```
