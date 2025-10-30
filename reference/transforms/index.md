# holocron.transforms

`holocron.transforms` provides PIL and PyTorch tensor transformations.

### ResizeMethod

Bases: `StrEnum`

Resize methods Available methods are `squish`, `pad`.

#### SQUISH

```python
SQUISH = 'squish'
```

#### PAD

```python
PAD = 'pad'
```

### Resize

```python
Resize(size: tuple[int, int], mode: ResizeMethod = SQUISH, pad_mode: str = 'constant', **kwargs: Any)
```

Bases: `Resize`

Implements a more flexible resizing scheme.

Example

```python
from holocron.transforms import Resize, ResizeMethod
pil_img = ...
tf = Resize((224, 224), mode=ResizeMethod.PAD)
resized_img = tf(pil_img)
```

| PARAMETER  | DESCRIPTION                                                                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `size`     | the desired height and width of the image in pixels **TYPE:** `tuple[int, int]`                                                                   |
| `mode`     | the resizing scheme ("squish" is similar to PyTorch, "pad" will preserve the aspect ratio and pad) **TYPE:** `ResizeMethod` **DEFAULT:** `SQUISH` |
| `pad_mode` | padding mode when mode is "pad" **TYPE:** `str` **DEFAULT:** `'constant'`                                                                         |
| `kwargs`   | the keyword arguments of torchvision.transforms.v2.Resize **TYPE:** `Any` **DEFAULT:** `{}`                                                       |

Source code in `holocron/transforms/interpolation.py`

```python
def __init__(
    self,
    size: tuple[int, int],
    mode: ResizeMethod = ResizeMethod.SQUISH,
    pad_mode: str = "constant",
    **kwargs: Any,
) -> None:
    if not isinstance(mode, ResizeMethod):
        raise TypeError("mode is expected to be a ResizeMethod")
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(s <= 0 for s in size):
        raise ValueError("size is expected to be a sequence of 2 positive integers")
    super().__init__(size, **kwargs)
    self.mode: ResizeMethod = mode
    self.pad_mode: str = pad_mode
    self.size: tuple[int, int]
```

### RandomZoomOut

```python
RandomZoomOut(size: tuple[int, int], scale: tuple[float, float] = (0.5, 1.0), **kwargs: Any)
```

Bases: `Module`

Implements a size reduction of the orignal image to provide a zoom out effect.

Example

```python
from holocron.transforms import RandomZoomOut
pil_img = ...
tf = RandomZoomOut((224, 224), scale=(0.3, 1.))
resized_img = tf(pil_img)
```

| PARAMETER | DESCRIPTION                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `size`    | the desired height and width of the image in pixels **TYPE:** `tuple[int, int]`                                                 |
| `scale`   | the range of relative area of the projected image to the desired size **TYPE:** `tuple[float, float]` **DEFAULT:** `(0.5, 1.0)` |
| `kwargs`  | the keyword arguments of torchvision.transforms.functional.resize **TYPE:** `Any` **DEFAULT:** `{}`                             |

Source code in `holocron/transforms/interpolation.py`

```python
def __init__(self, size: tuple[int, int], scale: tuple[float, float] = (0.5, 1.0), **kwargs: Any) -> None:
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(s <= 0 for s in size):
        raise ValueError("size is expected to be a sequence of 2 positive integers")
    if len(scale) != 2 or scale[0] > scale[1]:
        raise ValueError("scale is expected to be a couple of floats, the first one being small than the second")
    super().__init__()
    self.size: tuple[int, int] = size
    self.scale: tuple[float, float] = scale
    self._kwargs: dict[str, Any] = kwargs
```
