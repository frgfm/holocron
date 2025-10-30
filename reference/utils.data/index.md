# holocron.utils.data

## Batch collate

### Mixup

```python
Mixup(num_classes: int, alpha: float = 0.2)
```

Bases: `Module`

Implements a batch collate function with MixUp strategy from ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/pdf/1710.09412.pdf).

Example

```python
import torch
from torch.utils.data._utils.collate import default_collate
from holocron.utils.data import Mixup
mix = Mixup(num_classes=10, alpha=0.4)
loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=lambda b: mix(*default_collate(b)))
```

| PARAMETER     | DESCRIPTION                                       |
| ------------- | ------------------------------------------------- |
| `num_classes` | number of expected classes **TYPE:** `int`        |
| `alpha`       | mixup factor **TYPE:** `float` **DEFAULT:** `0.2` |

Source code in `holocron/utils/data/collate.py`

```python
def __init__(self, num_classes: int, alpha: float = 0.2) -> None:
    super().__init__()
    self.num_classes: int = num_classes
    if alpha < 0:
        raise ValueError("`alpha` only takes positive values")
    self.alpha: float = alpha
```
