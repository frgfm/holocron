# holocron.utils

`holocron.utils` provides some utilities for general usage.

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
