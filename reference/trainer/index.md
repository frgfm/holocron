# holocron.trainer

`holocron.trainer` provides some basic objects for training purposes.

### Trainer

```python
Trainer(model: Module, train_loader: DataLoader, val_loader: DataLoader, criterion: Module, optimizer: Optimizer, gpu: int | None = None, output_file: str = './checkpoint.pth', amp: bool = False, skip_nan_loss: bool = False, nan_tolerance: int = 5, gradient_acc: int = 1, gradient_clip: float | None = None, on_epoch_end: Callable[[dict[str, float]], Any] | None = None)
```

Baseline trainer class.

| PARAMETER       | DESCRIPTION                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| `model`         | model to train **TYPE:** `Module`                                                                                  |
| `train_loader`  | training loader **TYPE:** `DataLoader`                                                                             |
| `val_loader`    | validation loader **TYPE:** `DataLoader`                                                                           |
| `criterion`     | loss criterion **TYPE:** `Module`                                                                                  |
| `optimizer`     | parameter optimizer **TYPE:** `Optimizer`                                                                          |
| `gpu`           | index of the GPU to use **TYPE:** \`int                                                                            |
| `output_file`   | path where checkpoints will be saved **TYPE:** `str` **DEFAULT:** `'./checkpoint.pth'`                             |
| `amp`           | whether to use automatic mixed precision **TYPE:** `bool` **DEFAULT:** `False`                                     |
| `skip_nan_loss` | whether the optimizer step should be skipped when the loss is NaN **TYPE:** `bool` **DEFAULT:** `False`            |
| `nan_tolerance` | number of consecutive batches with NaN loss before stopping the training **TYPE:** `int` **DEFAULT:** `5`          |
| `gradient_acc`  | number of batches to accumulate the gradient of before performing the update step **TYPE:** `int` **DEFAULT:** `1` |
| `gradient_clip` | the gradient clip value **TYPE:** \`float                                                                          |
| `on_epoch_end`  | callback triggered at the end of an epoch **TYPE:** \`Callable\[\[dict[str, float]\], Any\]                        |

Source code in `holocron/trainer/core.py`

```python
def __init__(
    self,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    gpu: int | None = None,
    output_file: str = "./checkpoint.pth",
    amp: bool = False,
    skip_nan_loss: bool = False,
    nan_tolerance: int = 5,
    gradient_acc: int = 1,
    gradient_clip: float | None = None,
    on_epoch_end: Callable[[dict[str, float]], Any] | None = None,
) -> None:
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.amp = amp
    self.scaler: GradScaler
    self.on_epoch_end = on_epoch_end
    self.skip_nan_loss = skip_nan_loss
    self.nan_tolerance = nan_tolerance
    self.gradient_acc = gradient_acc
    self.grad_clip = gradient_clip

    # Output file
    self.output_file = output_file

    # Initialize
    self.step = 0
    self.start_epoch = 0
    self.epoch = 0
    self._grad_count = 0
    self.min_loss = math.inf
    self.gpu = gpu
    self._params: tuple[ParamSeq, ParamSeq] = ([], [])
    self.lr_recorder: list[float] = []
    self.loss_recorder: list[float] = []
    self.set_device(gpu)
    self._reset_opt(self.optimizer.defaults["lr"])
```

#### set_device

```python
set_device(gpu: int | None = None) -> None
```

Move tensor objects to the target GPU

| PARAMETER | DESCRIPTION                                    |
| --------- | ---------------------------------------------- |
| `gpu`     | index of the target GPU device **TYPE:** \`int |

| RAISES           | DESCRIPTION                      |
| ---------------- | -------------------------------- |
| `AssertionError` | if PyTorch cannot access the GPU |
| `ValueError`     | if the device index is invalid   |

Source code in `holocron/trainer/core.py`

```python
def set_device(self, gpu: int | None = None) -> None:
    """Move tensor objects to the target GPU

    Args:
        gpu: index of the target GPU device

    Raises:
        AssertionError: if PyTorch cannot access the GPU
        ValueError: if the device index is invalid
    """
    if isinstance(gpu, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if gpu >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
        torch.cuda.set_device(gpu)
        self.model = self.model.cuda()
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion = self.criterion.cuda()
```

#### to_cuda

```python
to_cuda(x: Tensor, target: Tensor | list[dict[str, Tensor]]) -> tuple[Tensor, Tensor | list[dict[str, Tensor]]]
```

Move input and target to GPU

| PARAMETER | DESCRIPTION                                                     |
| --------- | --------------------------------------------------------------- |
| `x`       | input tensor **TYPE:** `Tensor`                                 |
| `target`  | target tensor or list of target dictionaries **TYPE:** \`Tensor |

| RETURNS                 | DESCRIPTION                   |
| ----------------------- | ----------------------------- |
| \`tuple\[Tensor, Tensor | list\[dict[str, Tensor]\]\]\` |

| RAISES       | DESCRIPTION                    |
| ------------ | ------------------------------ |
| `ValueError` | if the device index is invalid |

Source code in `holocron/trainer/core.py`

```python
def to_cuda(
    self, x: Tensor, target: Tensor | list[dict[str, Tensor]]
) -> tuple[Tensor, Tensor | list[dict[str, Tensor]]]:
    """Move input and target to GPU

    Args:
        x: input tensor
        target: target tensor or list of target dictionaries

    Returns:
        tuple of input and target tensors

    Raises:
        ValueError: if the device index is invalid
    """
    if isinstance(self.gpu, int):
        if self.gpu >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
        return self._to_cuda(x, target)  # type: ignore[arg-type]
    return x, target
```

#### save

```python
save(output_file: str) -> None
```

Save a trainer checkpoint

| PARAMETER     | DESCRIPTION                           |
| ------------- | ------------------------------------- |
| `output_file` | destination file path **TYPE:** `str` |

Source code in `holocron/trainer/core.py`

```python
def save(self, output_file: str) -> None:
    """Save a trainer checkpoint

    Args:
        output_file: destination file path
    """
    torch.save(
        {
            "epoch": self.epoch,
            "step": self.step,
            "min_loss": self.min_loss,
            "model": self.model.state_dict(),
        },
        output_file,
        _use_new_zipfile_serialization=False,
    )
```

#### load

```python
load(state: dict[str, Any]) -> None
```

Resume from a trainer state

| PARAMETER | DESCRIPTION                                      |
| --------- | ------------------------------------------------ |
| `state`   | checkpoint dictionary **TYPE:** `dict[str, Any]` |

Source code in `holocron/trainer/core.py`

```python
def load(self, state: dict[str, Any]) -> None:
    """Resume from a trainer state

    Args:
        state: checkpoint dictionary
    """
    self.start_epoch = state["epoch"]
    self.epoch = self.start_epoch
    self.step = state["step"]
    self.min_loss = state["min_loss"]
    self.model.load_state_dict(state["model"])
```

#### fit_n_epochs

```python
fit_n_epochs(num_epochs: int, lr: float, freeze_until: str | None = None, sched_type: str = 'onecycle', norm_weight_decay: float | None = None, **kwargs: Any) -> None
```

Train the model for a given number of epochs.

| PARAMETER           | DESCRIPTION                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| `num_epochs`        | number of epochs to train **TYPE:** `int`                                |
| `lr`                | learning rate to be used by the scheduler **TYPE:** `float`              |
| `freeze_until`      | last layer to freeze **TYPE:** \`str                                     |
| `sched_type`        | type of scheduler to use **TYPE:** `str` **DEFAULT:** `'onecycle'`       |
| `norm_weight_decay` | weight decay to apply to normalization parameters **TYPE:** \`float      |
| `**kwargs`          | keyword args passed to the LRScheduler **TYPE:** `Any` **DEFAULT:** `{}` |

Source code in `holocron/trainer/core.py`

```python
def fit_n_epochs(
    self,
    num_epochs: int,
    lr: float,
    freeze_until: str | None = None,
    sched_type: str = "onecycle",
    norm_weight_decay: float | None = None,
    **kwargs: Any,
) -> None:
    """Train the model for a given number of epochs.

    Args:
        num_epochs: number of epochs to train
        lr: learning rate to be used by the scheduler
        freeze_until: last layer to freeze
        sched_type: type of scheduler to use
        norm_weight_decay: weight decay to apply to normalization parameters
        **kwargs: keyword args passed to the [`LRScheduler`][torch.optim.lr_scheduler.LRScheduler]
    """
    freeze_model(self.model.train(), freeze_until)
    # Update param groups & LR
    self._reset_opt(lr, norm_weight_decay)
    # Scheduler
    self._reset_scheduler(lr, num_epochs, sched_type, **kwargs)

    if self.amp:
        self.scaler = GradScaler("cuda")

    mb = master_bar(range(num_epochs))
    for _ in mb:
        self._fit_epoch(mb)
        eval_metrics = self.evaluate()

        # master bar
        mb.main_bar.comment = f"Epoch {self.epoch}/{self.start_epoch + num_epochs}"
        mb.write(f"Epoch {self.epoch}/{self.start_epoch + num_epochs} - {self._eval_metrics_str(eval_metrics)}")

        if eval_metrics["val_loss"] < self.min_loss:
            print(  # noqa: T201
                f"Validation loss decreased {self.min_loss:.4} --> {eval_metrics['val_loss']:.4}: saving state..."
            )
            self.min_loss = eval_metrics["val_loss"]
            self.save(self.output_file)

        if self.on_epoch_end is not None:
            self.on_epoch_end(eval_metrics)
```

#### find_lr

```python
find_lr(freeze_until: str | None = None, start_lr: float = 1e-07, end_lr: float = 1, norm_weight_decay: float | None = None, num_it: int = 100) -> None
```

Gridsearch the optimal learning rate for the training as described in ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/pdf/1506.01186.pdf).

| PARAMETER           | DESCRIPTION                                                         |
| ------------------- | ------------------------------------------------------------------- |
| `freeze_until`      | last layer to freeze **TYPE:** \`str                                |
| `start_lr`          | initial learning rate **TYPE:** `float` **DEFAULT:** `1e-07`        |
| `end_lr`            | final learning rate **TYPE:** `float` **DEFAULT:** `1`              |
| `norm_weight_decay` | weight decay to apply to normalization parameters **TYPE:** \`float |
| `num_it`            | number of iterations to perform **TYPE:** `int` **DEFAULT:** `100`  |

| RAISES       | DESCRIPTION                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| `ValueError` | if the number of iterations is greater than the number of available batches |

Source code in `holocron/trainer/core.py`

```python
def find_lr(
    self,
    freeze_until: str | None = None,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    norm_weight_decay: float | None = None,
    num_it: int = 100,
) -> None:
    """Gridsearch the optimal learning rate for the training as described in
    ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/pdf/1506.01186.pdf).

    Args:
       freeze_until: last layer to freeze
       start_lr: initial learning rate
       end_lr: final learning rate
       norm_weight_decay: weight decay to apply to normalization parameters
       num_it: number of iterations to perform

    Raises:
        ValueError: if the number of iterations is greater than the number of available batches
    """
    if num_it > len(self.train_loader):
        raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

    freeze_model(self.model.train(), freeze_until)
    # Update param groups & LR
    self._reset_opt(start_lr, norm_weight_decay)
    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(self.optimizer, lambda step: gamma)

    self.lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    self.loss_recorder = []

    if self.amp:
        self.scaler = GradScaler("cuda")

    for batch_idx, (x, target) in enumerate(self.train_loader):
        x, target = self.to_cuda(x, target)

        # Forward
        batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]
        self._backprop_step(batch_loss)
        # Update LR
        scheduler.step()

        # Record
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            break
        self.loss_recorder.append(batch_loss.item())
        # Stop after the number of iterations
        if batch_idx + 1 == num_it:
            break

    self.lr_recorder = self.lr_recorder[: len(self.loss_recorder)]
```

#### plot_recorder

```python
plot_recorder(beta: float = 0.95, **kwargs: Any) -> None
```

Display the results of the LR grid search

| PARAMETER  | DESCRIPTION                                                              |
| ---------- | ------------------------------------------------------------------------ |
| `beta`     | smoothing factor **TYPE:** `float` **DEFAULT:** `0.95`                   |
| `**kwargs` | keyword args of matplotlib.pyplot.show **TYPE:** `Any` **DEFAULT:** `{}` |

| RAISES           | DESCRIPTION                                                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `AssertionError` | if the number of learning rate recorder and loss recorder are not the same or if the number of learning rate recorder is 0 |

Source code in `holocron/trainer/core.py`

```python
def plot_recorder(self, beta: float = 0.95, **kwargs: Any) -> None:
    """Display the results of the LR grid search

    Args:
        beta: smoothing factor
        **kwargs: keyword args of [`matplotlib.pyplot.show`][matplotlib.pyplot.show]

    Raises:
        AssertionError: if the number of learning rate recorder and loss recorder are not the same or if the number of learning rate recorder is 0
    """
    if len(self.lr_recorder) != len(self.loss_recorder) or len(self.lr_recorder) == 0:
        raise AssertionError("Please run the `lr_find` method first")

    # Exp moving average of loss
    smoothed_losses = []
    avg_loss = 0.0
    for idx, loss in enumerate(self.loss_recorder):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

    # Properly rescale Y-axis
    data_slice = slice(
        min(len(self.loss_recorder) // 10, 10),
        -min(len(self.loss_recorder) // 20, 5) if len(self.loss_recorder) >= 20 else len(self.loss_recorder),
    )
    vals: np.ndarray = np.array(smoothed_losses[data_slice])
    min_idx = vals.argmin()
    max_val = vals.max() if min_idx is None else vals[: min_idx + 1].max()
    delta = max_val - vals[min_idx]

    plt.plot(self.lr_recorder[data_slice], smoothed_losses[data_slice])
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Training loss")
    plt.ylim(vals[min_idx] - 0.1 * delta, max_val + 0.2 * delta)
    plt.grid(True, linestyle="--", axis="x")
    plt.show(**kwargs)
```

#### check_setup

```python
check_setup(freeze_until: str | None = None, lr: float = 0.0003, norm_weight_decay: float | None = None, num_it: int = 100, **kwargs: Any) -> None
```

Check whether you can overfit one batch

| PARAMETER           | DESCRIPTION                                                                   |
| ------------------- | ----------------------------------------------------------------------------- |
| `freeze_until`      | last layer to freeze **TYPE:** \`str                                          |
| `lr`                | learning rate to be used for training **TYPE:** `float` **DEFAULT:** `0.0003` |
| `norm_weight_decay` | weight decay to apply to normalization parameters **TYPE:** \`float           |
| `num_it`            | number of iterations to perform **TYPE:** `int` **DEFAULT:** `100`            |
| `**kwargs`          | keyword args of matplotlib.pyplot.show **TYPE:** `Any` **DEFAULT:** `{}`      |

| RAISES       | DESCRIPTION                     |
| ------------ | ------------------------------- |
| `ValueError` | if the loss value is NaN or inf |

Source code in `holocron/trainer/core.py`

```python
def check_setup(
    self,
    freeze_until: str | None = None,
    lr: float = 3e-4,
    norm_weight_decay: float | None = None,
    num_it: int = 100,
    **kwargs: Any,
) -> None:
    """Check whether you can overfit one batch

    Args:
        freeze_until: last layer to freeze
        lr: learning rate to be used for training
        norm_weight_decay: weight decay to apply to normalization parameters
        num_it: number of iterations to perform
        **kwargs: keyword args of [`matplotlib.pyplot.show`][matplotlib.pyplot.show]

    Raises:
        ValueError: if the loss value is NaN or inf
    """
    freeze_model(self.model.train(), freeze_until)
    # Update param groups & LR
    self._reset_opt(lr, norm_weight_decay)

    x, target = next(iter(self.train_loader))
    x, target = self.to_cuda(x, target)

    losses = []

    if self.amp:
        self.scaler = GradScaler("cuda")

    for _ in range(num_it):
        # Forward
        batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]
        # Backprop
        self._backprop_step(batch_loss)

        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            raise ValueError("loss value is NaN or inf.")

        losses.append(batch_loss.item())

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Optimization steps")
    plt.ylabel("Training loss")
    plt.grid(True, linestyle="--", axis="x")
    plt.show(**kwargs)
```

## Image classification

### ClassificationTrainer

```python
ClassificationTrainer(model: Module, train_loader: DataLoader, val_loader: DataLoader, criterion: Module, optimizer: Optimizer, gpu: int | None = None, output_file: str = './checkpoint.pth', amp: bool = False, skip_nan_loss: bool = False, nan_tolerance: int = 5, gradient_acc: int = 1, gradient_clip: float | None = None, on_epoch_end: Callable[[dict[str, float]], Any] | None = None)
```

Bases: `Trainer`

Image classification trainer class.

| PARAMETER       | DESCRIPTION                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| `model`         | model to train **TYPE:** `Module`                                                                                  |
| `train_loader`  | training loader **TYPE:** `DataLoader`                                                                             |
| `val_loader`    | validation loader **TYPE:** `DataLoader`                                                                           |
| `criterion`     | loss criterion **TYPE:** `Module`                                                                                  |
| `optimizer`     | parameter optimizer **TYPE:** `Optimizer`                                                                          |
| `gpu`           | index of the GPU to use **TYPE:** \`int                                                                            |
| `output_file`   | path where checkpoints will be saved **TYPE:** `str` **DEFAULT:** `'./checkpoint.pth'`                             |
| `amp`           | whether to use automatic mixed precision **TYPE:** `bool` **DEFAULT:** `False`                                     |
| `skip_nan_loss` | whether the optimizer step should be skipped when the loss is NaN **TYPE:** `bool` **DEFAULT:** `False`            |
| `nan_tolerance` | number of consecutive batches with NaN loss before stopping the training **TYPE:** `int` **DEFAULT:** `5`          |
| `gradient_acc`  | number of batches to accumulate the gradient of before performing the update step **TYPE:** `int` **DEFAULT:** `1` |
| `gradient_clip` | the gradient clip value **TYPE:** \`float                                                                          |
| `on_epoch_end`  | callback triggered at the end of an epoch **TYPE:** \`Callable\[\[dict[str, float]\], Any\]                        |

Source code in `holocron/trainer/core.py`

```python
def __init__(
    self,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    gpu: int | None = None,
    output_file: str = "./checkpoint.pth",
    amp: bool = False,
    skip_nan_loss: bool = False,
    nan_tolerance: int = 5,
    gradient_acc: int = 1,
    gradient_clip: float | None = None,
    on_epoch_end: Callable[[dict[str, float]], Any] | None = None,
) -> None:
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.amp = amp
    self.scaler: GradScaler
    self.on_epoch_end = on_epoch_end
    self.skip_nan_loss = skip_nan_loss
    self.nan_tolerance = nan_tolerance
    self.gradient_acc = gradient_acc
    self.grad_clip = gradient_clip

    # Output file
    self.output_file = output_file

    # Initialize
    self.step = 0
    self.start_epoch = 0
    self.epoch = 0
    self._grad_count = 0
    self.min_loss = math.inf
    self.gpu = gpu
    self._params: tuple[ParamSeq, ParamSeq] = ([], [])
    self.lr_recorder: list[float] = []
    self.loss_recorder: list[float] = []
    self.set_device(gpu)
    self._reset_opt(self.optimizer.defaults["lr"])
```

#### evaluate

```python
evaluate() -> dict[str, float]
```

Evaluate the model on the validation set

| RETURNS            | DESCRIPTION                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `dict[str, float]` | evaluation metrics (validation loss, top1 accuracy, top5 accuracy) |

Source code in `holocron/trainer/classification.py`

```python
@torch.inference_mode()
def evaluate(self) -> dict[str, float]:
    """Evaluate the model on the validation set

    Returns:
        evaluation metrics (validation loss, top1 accuracy, top5 accuracy)
    """
    self.model.eval()

    val_loss, top1, top5, num_samples, num_valid_batches = 0.0, 0, 0, 0, 0
    for x, target in self.val_loader:
        x, target = self.to_cuda(x, target)

        loss, out = self._get_loss(x, target, return_logits=True)  # ty: ignore[invalid-argument-type]

        # Safeguard for NaN loss
        if not torch.isnan(loss) and not torch.isinf(loss):
            val_loss += loss.item()
            num_valid_batches += 1

        pred = out.topk(5, dim=1)[1] if out.shape[1] >= 5 else out.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))  # ty: ignore[possibly-missing-attribute]
        top1 += cast(int, correct[:, 0].sum().item())
        if out.shape[1] >= 5:
            top5 += cast(int, correct.any(dim=1).sum().item())

        num_samples += x.shape[0]

    val_loss /= num_valid_batches

    return {"val_loss": val_loss, "acc1": top1 / num_samples, "acc5": top5 / num_samples}
```

#### plot_top_losses

```python
plot_top_losses(mean: tuple[float, float, float], std: tuple[float, float, float], classes: Sequence[str] | None = None, num_samples: int = 12, **kwargs: Any) -> None
```

Plot the top losses

| PARAMETER     | DESCRIPTION                                                              |
| ------------- | ------------------------------------------------------------------------ |
| `mean`        | mean of the dataset **TYPE:** `tuple[float, float, float]`               |
| `std`         | standard deviation of the dataset **TYPE:** `tuple[float, float, float]` |
| `classes`     | list of classes **TYPE:** \`Sequence[str]                                |
| `num_samples` | number of samples to plot **TYPE:** `int` **DEFAULT:** `12`              |
| `**kwargs`    | keyword args of matplotlib.pyplot.show **TYPE:** `Any` **DEFAULT:** `{}` |

| RAISES           | DESCRIPTION                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `AssertionError` | if the argument 'classes' is not specified for multi-class classification |

Source code in `holocron/trainer/classification.py`

```python
@torch.inference_mode()
def plot_top_losses(
    self,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    classes: Sequence[str] | None = None,
    num_samples: int = 12,
    **kwargs: Any,
) -> None:
    """Plot the top losses

    Args:
        mean: mean of the dataset
        std: standard deviation of the dataset
        classes: list of classes
        num_samples: number of samples to plot
        **kwargs: keyword args of [`matplotlib.pyplot.show`][matplotlib.pyplot.show]

    Raises:
        AssertionError: if the argument 'classes' is not specified for multi-class classification
    """
    # Record loss, prob, target, image
    losses = np.zeros(num_samples, dtype=np.float32)
    preds = np.zeros(num_samples, dtype=int)
    probs = np.zeros(num_samples, dtype=np.float32)
    targets = np.zeros(num_samples, dtype=np.float32 if self.is_binary else int)
    images = [None] * num_samples

    # Switch to unreduced loss
    reduction = self.criterion.reduction
    self.criterion.reduction = "none"  # type: ignore[assignment]
    self.model.eval()

    train_iter = iter(self.train_loader)

    for x, target in tqdm(train_iter):
        x, target = self.to_cuda(x, target)

        # Forward
        batch_loss, logits = self._get_loss(x, target, return_logits=True)  # ty: ignore[invalid-argument-type]

        # Binary
        if self.is_binary:
            batch_loss = batch_loss.squeeze(1)
            probs_ = torch.sigmoid(logits.squeeze(1))
        else:
            probs_ = torch.softmax(logits, 1).max(dim=1).values

        if torch.any(batch_loss > losses.min()):
            idcs = np.concatenate((losses, batch_loss.cpu().numpy())).argsort()[-num_samples:]
            kept_idcs = [idx for idx in idcs if idx < num_samples]
            added_idcs = [idx - num_samples for idx in idcs if idx >= num_samples]
            # Update
            losses = np.concatenate((losses[kept_idcs], batch_loss.cpu().numpy()[added_idcs]))
            probs = np.concatenate((probs[kept_idcs], probs_.cpu().numpy()))
            if not self.is_binary:
                preds = np.concatenate((preds[kept_idcs], logits[added_idcs].argmax(dim=1).cpu().numpy()))
            targets = np.concatenate((targets[kept_idcs], target[added_idcs].cpu().numpy()))  # ty: ignore[invalid-argument-type]
            imgs = x[added_idcs].cpu() * torch.tensor(std).view(-1, 1, 1)
            imgs += torch.tensor(mean).view(-1, 1, 1)
            images = [images[idx] for idx in kept_idcs] + [to_pil_image(img) for img in imgs]

    self.criterion.reduction = reduction

    if not self.is_binary and classes is None:
        raise AssertionError("arg 'classes' must be specified for multi-class classification")

    # Final sort
    idcs_ = losses.argsort()[::-1]
    losses, preds, probs, targets = losses[idcs_], preds[idcs_], probs[idcs_], targets[idcs_]
    images = [images[idx] for idx in idcs_]

    # Plot it
    num_cols = 4
    num_rows = math.ceil(num_samples / num_cols)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    for idx, (img, pred, prob, target, loss) in enumerate(zip(images, preds, probs, targets, losses, strict=True)):
        row = int(idx / num_cols)
        col = idx - num_cols * row
        axes[row][col].imshow(img)
        # Loss, prob, target
        if self.is_binary:
            axes[row][col].title.set_text(f"{loss:.3} / {prob:.2} / {target:.2}")
        # Loss, pred (prob), target
        else:
            axes[row][col].title.set_text(
                f"{loss:.3} / {classes[pred]} ({prob:.1%}) / {classes[target]}"  # type: ignore[index]
            )
        axes[row][col].axis("off")

    plt.show(**kwargs)
```

### BinaryClassificationTrainer

```python
BinaryClassificationTrainer(model: Module, train_loader: DataLoader, val_loader: DataLoader, criterion: Module, optimizer: Optimizer, gpu: int | None = None, output_file: str = './checkpoint.pth', amp: bool = False, skip_nan_loss: bool = False, nan_tolerance: int = 5, gradient_acc: int = 1, gradient_clip: float | None = None, on_epoch_end: Callable[[dict[str, float]], Any] | None = None)
```

Bases: `ClassificationTrainer`

Image binary classification trainer class.

| PARAMETER      | DESCRIPTION                                                                            |
| -------------- | -------------------------------------------------------------------------------------- |
| `model`        | model to train **TYPE:** `Module`                                                      |
| `train_loader` | training loader **TYPE:** `DataLoader`                                                 |
| `val_loader`   | validation loader **TYPE:** `DataLoader`                                               |
| `criterion`    | loss criterion **TYPE:** `Module`                                                      |
| `optimizer`    | parameter optimizer **TYPE:** `Optimizer`                                              |
| `gpu`          | index of the GPU to use **TYPE:** \`int                                                |
| `output_file`  | path where checkpoints will be saved **TYPE:** `str` **DEFAULT:** `'./checkpoint.pth'` |
| `amp`          | whether to use automatic mixed precision **TYPE:** `bool` **DEFAULT:** `False`         |

Source code in `holocron/trainer/core.py`

```python
def __init__(
    self,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    gpu: int | None = None,
    output_file: str = "./checkpoint.pth",
    amp: bool = False,
    skip_nan_loss: bool = False,
    nan_tolerance: int = 5,
    gradient_acc: int = 1,
    gradient_clip: float | None = None,
    on_epoch_end: Callable[[dict[str, float]], Any] | None = None,
) -> None:
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.amp = amp
    self.scaler: GradScaler
    self.on_epoch_end = on_epoch_end
    self.skip_nan_loss = skip_nan_loss
    self.nan_tolerance = nan_tolerance
    self.gradient_acc = gradient_acc
    self.grad_clip = gradient_clip

    # Output file
    self.output_file = output_file

    # Initialize
    self.step = 0
    self.start_epoch = 0
    self.epoch = 0
    self._grad_count = 0
    self.min_loss = math.inf
    self.gpu = gpu
    self._params: tuple[ParamSeq, ParamSeq] = ([], [])
    self.lr_recorder: list[float] = []
    self.loss_recorder: list[float] = []
    self.set_device(gpu)
    self._reset_opt(self.optimizer.defaults["lr"])
```

#### evaluate

```python
evaluate() -> dict[str, float]
```

Evaluate the model on the validation set

| RETURNS            | DESCRIPTION                                    |
| ------------------ | ---------------------------------------------- |
| `dict[str, float]` | evaluation metrics (validation loss, accuracy) |

Source code in `holocron/trainer/classification.py`

```python
@torch.inference_mode()
def evaluate(self) -> dict[str, float]:
    """Evaluate the model on the validation set

    Returns:
        evaluation metrics (validation loss, accuracy)
    """
    self.model.eval()

    val_loss, top1, num_samples, num_valid_batches = 0.0, 0.0, 0, 0
    for x, target in self.val_loader:
        x, target = self.to_cuda(x, target)

        loss, out = self._get_loss(x, target, return_logits=True)  # ty: ignore[invalid-argument-type]

        # Safeguard for NaN loss
        if not torch.isnan(loss) and not torch.isinf(loss):
            val_loss += loss.item()
            num_valid_batches += 1

        top1 += torch.sum((target.view_as(out) >= 0.5) == (torch.sigmoid(out) >= 0.5)).item() / out[0].numel()  # ty: ignore[possibly-missing-attribute]

        num_samples += x.shape[0]

    val_loss /= num_valid_batches

    return {"val_loss": val_loss, "acc": top1 / num_samples}
```

## Semantic segmentation

### SegmentationTrainer

```python
SegmentationTrainer(*args: Any, num_classes: int = 10, **kwargs: Any)
```

Bases: `Trainer`

Semantic segmentation trainer class.

| PARAMETER     | DESCRIPTION                                                |
| ------------- | ---------------------------------------------------------- |
| `*args`       | args of Trainer **TYPE:** `Any` **DEFAULT:** `()`          |
| `num_classes` | number of output classes **TYPE:** `int` **DEFAULT:** `10` |
| `**kwargs`    | keyword args of Trainer **TYPE:** `Any` **DEFAULT:** `{}`  |

Source code in `holocron/trainer/segmentation.py`

```python
def __init__(self, *args: Any, num_classes: int = 10, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self.num_classes = num_classes
```

#### evaluate

```python
evaluate(ignore_index: int = 255) -> dict[str, float]
```

Evaluate the model on the validation set

| PARAMETER      | DESCRIPTION                                                                   |
| -------------- | ----------------------------------------------------------------------------- |
| `ignore_index` | index of the class to ignore in evaluation **TYPE:** `int` **DEFAULT:** `255` |

| RETURNS            | DESCRIPTION                                                     |
| ------------------ | --------------------------------------------------------------- |
| `dict[str, float]` | evaluation metrics (validation loss, global accuracy, mean IoU) |

Source code in `holocron/trainer/segmentation.py`

```python
@torch.inference_mode()
def evaluate(self, ignore_index: int = 255) -> dict[str, float]:
    """Evaluate the model on the validation set

    Args:
        ignore_index: index of the class to ignore in evaluation

    Returns:
        evaluation metrics (validation loss, global accuracy, mean IoU)
    """
    self.model.eval()

    val_loss, mean_iou, num_valid_batches = 0.0, 0.0, 0
    conf_mat = torch.zeros(
        (self.num_classes, self.num_classes), dtype=torch.int64, device=next(self.model.parameters()).device
    )
    for x, target in self.val_loader:
        x, target = self.to_cuda(x, target)

        loss, out = self._get_loss(x, target, return_logits=True)  # ty: ignore[invalid-argument-type]

        # Safeguard for NaN loss
        if not torch.isnan(loss) and not torch.isinf(loss):
            val_loss += loss.item()
            num_valid_batches += 1

        # borrowed from https://github.com/pytorch/vision/blob/master/references/segmentation/train.py
        pred = out.argmax(dim=1).flatten()
        target = target.flatten()  # ty: ignore[possibly-missing-attribute]
        k = (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[k].to(torch.int64) + pred[k]
        nc = self.num_classes
        conf_mat += torch.bincount(inds, minlength=nc**2).reshape(nc, nc)

    val_loss /= num_valid_batches
    acc_global = (torch.diag(conf_mat).sum() / conf_mat.sum()).item()
    mean_iou = (torch.diag(conf_mat) / (conf_mat.sum(1) + conf_mat.sum(0) - torch.diag(conf_mat))).mean().item()

    return {"val_loss": val_loss, "acc_global": acc_global, "mean_iou": mean_iou}
```

## Object detection

### DetectionTrainer

```python
DetectionTrainer(model: Module, train_loader: DataLoader, val_loader: DataLoader, criterion: Module, optimizer: Optimizer, gpu: int | None = None, output_file: str = './checkpoint.pth', amp: bool = False, skip_nan_loss: bool = False, nan_tolerance: int = 5, gradient_acc: int = 1, gradient_clip: float | None = None, on_epoch_end: Callable[[dict[str, float]], Any] | None = None)
```

Bases: `Trainer`

Object detection trainer class.

| PARAMETER       | DESCRIPTION                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| `model`         | model to train **TYPE:** `Module`                                                                                  |
| `train_loader`  | training loader **TYPE:** `DataLoader`                                                                             |
| `val_loader`    | validation loader **TYPE:** `DataLoader`                                                                           |
| `criterion`     | loss criterion **TYPE:** `Module`                                                                                  |
| `optimizer`     | parameter optimizer **TYPE:** `Optimizer`                                                                          |
| `gpu`           | index of the GPU to use **TYPE:** \`int                                                                            |
| `output_file`   | path where checkpoints will be saved **TYPE:** `str` **DEFAULT:** `'./checkpoint.pth'`                             |
| `amp`           | whether to use automatic mixed precision **TYPE:** `bool` **DEFAULT:** `False`                                     |
| `skip_nan_loss` | whether the optimizer step should be skipped when the loss is NaN **TYPE:** `bool` **DEFAULT:** `False`            |
| `nan_tolerance` | number of consecutive batches with NaN loss before stopping the training **TYPE:** `int` **DEFAULT:** `5`          |
| `gradient_acc`  | number of batches to accumulate the gradient of before performing the update step **TYPE:** `int` **DEFAULT:** `1` |
| `gradient_clip` | the gradient clip value **TYPE:** \`float                                                                          |
| `on_epoch_end`  | callback triggered at the end of an epoch **TYPE:** \`Callable\[\[dict[str, float]\], Any\]                        |

Source code in `holocron/trainer/core.py`

```python
def __init__(
    self,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    gpu: int | None = None,
    output_file: str = "./checkpoint.pth",
    amp: bool = False,
    skip_nan_loss: bool = False,
    nan_tolerance: int = 5,
    gradient_acc: int = 1,
    gradient_clip: float | None = None,
    on_epoch_end: Callable[[dict[str, float]], Any] | None = None,
) -> None:
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.amp = amp
    self.scaler: GradScaler
    self.on_epoch_end = on_epoch_end
    self.skip_nan_loss = skip_nan_loss
    self.nan_tolerance = nan_tolerance
    self.gradient_acc = gradient_acc
    self.grad_clip = gradient_clip

    # Output file
    self.output_file = output_file

    # Initialize
    self.step = 0
    self.start_epoch = 0
    self.epoch = 0
    self._grad_count = 0
    self.min_loss = math.inf
    self.gpu = gpu
    self._params: tuple[ParamSeq, ParamSeq] = ([], [])
    self.lr_recorder: list[float] = []
    self.loss_recorder: list[float] = []
    self.set_device(gpu)
    self._reset_opt(self.optimizer.defaults["lr"])
```

#### evaluate

```python
evaluate(iou_threshold: float = 0.5) -> dict[str, float | None]
```

Evaluate the model on the validation set.

| PARAMETER       | DESCRIPTION                                                            |
| --------------- | ---------------------------------------------------------------------- |
| `iou_threshold` | IoU threshold for pair assignment **TYPE:** `float` **DEFAULT:** `0.5` |

| RETURNS            | DESCRIPTION |
| ------------------ | ----------- |
| \`dict\[str, float | None\]\`    |

Source code in `holocron/trainer/detection.py`

```python
@torch.inference_mode()
def evaluate(self, iou_threshold: float = 0.5) -> dict[str, float | None]:
    """Evaluate the model on the validation set.

    Args:
        iou_threshold: IoU threshold for pair assignment

    Returns:
        evaluation metrics (validation loss, localization error rate, classification error rate, detection error rate)
    """
    self.model.eval()

    loc_assigns = 0
    correct, clf_error, loc_fn, loc_fp, num_samples = 0, 0, 0, 0, 0

    for x, target in self.val_loader:
        x, target = self.to_cuda(x, target)

        if self.amp:
            with torch.amp.autocast("cuda"):
                detections = self.model(x)
        else:
            detections = self.model(x)

        for dets, t in zip(detections, target, strict=True):
            if t["boxes"].shape[0] > 0 and dets["boxes"].shape[0] > 0:
                gt_indices, pred_indices = assign_iou(t["boxes"], dets["boxes"], iou_threshold)
                loc_assigns += len(gt_indices)
                correct_ = (t["labels"][gt_indices] == dets["labels"][pred_indices]).sum().item()
            else:
                gt_indices, pred_indices = [], []
                correct_ = 0
            correct += correct_
            clf_error += len(gt_indices) - correct_
            loc_fn += t["boxes"].shape[0] - len(gt_indices)
            loc_fp += dets["boxes"].shape[0] - len(pred_indices)
        num_samples += sum(t["boxes"].shape[0] for t in target)

    nb_preds = num_samples - loc_fn + loc_fp
    # Localization
    loc_err = 1 - 2 * loc_assigns / (nb_preds + num_samples) if nb_preds + num_samples > 0 else None
    # Classification
    clf_err = 1 - correct / loc_assigns if loc_assigns > 0 else None
    # End-to-end
    det_err = 1 - 2 * correct / (nb_preds + num_samples) if nb_preds + num_samples > 0 else None
    return {"loc_err": loc_err, "clf_err": clf_err, "det_err": det_err, "val_loss": loc_err}
```

## Miscellaneous

### freeze_bn

```python
freeze_bn(mod: Module) -> None
```

Prevents parameter and stats from updating in Batchnorm layers that are frozen

Example

```python
from holocron.models import rexnet1_0x
from holocron.trainer.utils import freeze_bn
model = rexnet1_0x()
freeze_bn(model)
```

| PARAMETER | DESCRIPTION                       |
| --------- | --------------------------------- |
| `mod`     | model to train **TYPE:** `Module` |

Source code in `holocron/trainer/utils.py`

````python
def freeze_bn(mod: nn.Module) -> None:
    """Prevents parameter and stats from updating in Batchnorm layers that are frozen

    Example:
        ```python
        from holocron.models import rexnet1_0x
        from holocron.trainer.utils import freeze_bn
        model = rexnet1_0x()
        freeze_bn(model)
        ```

    Args:
        mod: model to train
    """
    # Loop on modules
    for m in mod.modules():
        if isinstance(m, _BatchNorm) and m.affine and all(not p.requires_grad for p in m.parameters()):
            # Switch back to commented code when https://github.com/pytorch/pytorch/issues/37823 is resolved
            m.track_running_stats = False  # ty: ignore[unresolved-attribute]
            m.eval()
````

### freeze_model

```python
freeze_model(model: Module, last_frozen_layer: str | None = None, frozen_bn_stat_update: bool = False) -> None
```

Freeze a specific range of model layers.

Example

```python
from holocron.models import rexnet1_0x
from holocron.trainer.utils import freeze_model
model = rexnet1_0x()
freeze_model(model)
```

| PARAMETER               | DESCRIPTION                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `model`                 | model to train **TYPE:** `Module`                                                          |
| `last_frozen_layer`     | last layer to freeze. Assumes layers have been registered in forward order **TYPE:** \`str |
| `frozen_bn_stat_update` | force stats update in BN layers that are frozen **TYPE:** `bool` **DEFAULT:** `False`      |

| RAISES       | DESCRIPTION                           |
| ------------ | ------------------------------------- |
| `ValueError` | if the last frozen layer is not found |

Source code in `holocron/trainer/utils.py`

````python
def freeze_model(
    model: nn.Module,
    last_frozen_layer: str | None = None,
    frozen_bn_stat_update: bool = False,
) -> None:
    """Freeze a specific range of model layers.

    Example:
        ```python
        from holocron.models import rexnet1_0x
        from holocron.trainer.utils import freeze_model
        model = rexnet1_0x()
        freeze_model(model)
        ```

    Args:
        model: model to train
        last_frozen_layer: last layer to freeze. Assumes layers have been registered in forward order
        frozen_bn_stat_update: force stats update in BN layers that are frozen

    Raises:
        ValueError: if the last frozen layer is not found
    """
    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad_(True)

    # Loop on parameters
    if isinstance(last_frozen_layer, str):
        layer_reached = False
        for n, p in model.named_parameters():
            if not layer_reached or n.startswith(last_frozen_layer):
                p.requires_grad_(False)
            if n.startswith(last_frozen_layer):
                layer_reached = True
            # Once the last param of the layer is frozen, we break
            elif layer_reached:
                break
        if not layer_reached:
            raise ValueError(f"Unable to locate child module {last_frozen_layer}")

    # Loop on modules
    if not frozen_bn_stat_update:
        freeze_bn(model)
````
