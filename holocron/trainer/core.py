# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import ConsoleMasterBar, NBMasterBar
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, MultiplicativeLR, OneCycleLR
from torch.utils.data import DataLoader

from .experiment import RunResult, write_run_bundle
from .utils import freeze_bn, freeze_model, split_normalization_params

ParamSeq = Sequence[torch.nn.Parameter]

__all__ = ["Trainer"]


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _primitive(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class Trainer:
    """Baseline trainer class.

    Args:
        model: model to train
        train_loader: training loader
        val_loader: validation loader
        criterion: loss criterion
        optimizer: parameter optimizer
        gpu: index of the GPU to use
        output_file: path where checkpoints will be saved
        amp: whether to use automatic mixed precision
        skip_nan_loss: whether the optimizer step should be skipped when the loss is NaN
        nan_tolerance: number of consecutive batches with NaN loss before stopping the training
        gradient_acc: number of batches to accumulate the gradient of before performing the update step
        gradient_clip: the gradient clip value
        on_epoch_end: callback triggered at the end of an epoch
    """

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
        self._run_config: dict[str, Any] = {}
        self._metrics_history: list[dict[str, float | int | None]] = []
        self._resume_scheduler_state: dict[str, Any] | None = None
        self._resume_optimizer_state: dict[str, Any] | None = None
        self._resume_scaler_state: dict[str, Any] | None = None
        self._is_resuming = False
        self.set_device(gpu)
        self._reset_opt(self.optimizer.defaults["lr"])

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

    def save(self, output_file: str) -> None:
        """Save a trainer checkpoint

        Args:
            output_file: destination file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rng_state: dict[str, Tensor | list[Tensor]] = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        parameter_names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        state = {
            "schema_version": 2,
            "epoch": self.epoch,
            "step": self.step,
            "min_loss": self.min_loss,
            "best_metric": self.min_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
            "scaler": self.scaler.state_dict() if hasattr(self, "scaler") else None,
            "rng_state": rng_state,
            "metrics": self._metrics_history,
            "config": self._checkpoint_config(),
            "optimizer_param_names": [
                [parameter_names[id(parameter)] for parameter in group["params"]]
                for group in self.optimizer.param_groups
            ],
            "parameter_requires_grad": {
                name: parameter.requires_grad for name, parameter in self.model.named_parameters()
            },
        }
        fd, tmp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
        os.close(fd)
        try:
            torch.save(state, tmp_name)
            Path(tmp_name).replace(output_path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            "trainer": {
                "class": _type_name(self),
                "model": _type_name(self.model),
                "criterion": _type_name(self.criterion),
                "optimizer": _type_name(self.optimizer),
                "amp": self.amp,
                "gradient_acc": self.gradient_acc,
                "gradient_clip": self.grad_clip,
                "skip_nan_loss": self.skip_nan_loss,
                "nan_tolerance": self.nan_tolerance,
            },
            "run": self._run_config.copy(),
        }

    def load(self, state: dict[str, Any]) -> None:
        """Resume from a trainer state

        Args:
            state: checkpoint dictionary
        """
        self.start_epoch = state.get("epoch", 0)
        self.epoch = self.start_epoch
        self.step = state.get("step", 0)
        self.min_loss = state.get("best_metric", state.get("min_loss", math.inf))
        self.model.load_state_dict(state["model"])
        if state.get("schema_version") != 2:
            self._run_config = {}
            self._metrics_history = []
            self._is_resuming = False
            return

        config = state.get("config", {})
        self._run_config = dict(config.get("run", {}))
        self._metrics_history = [dict(metric) for metric in state.get("metrics", [])]
        self._resume_optimizer_state = state.get("optimizer")
        self._resume_scheduler_state = state.get("scheduler")
        self._resume_scaler_state = state.get("scaler")
        self._is_resuming = bool(self._run_config and self._resume_scheduler_state is not None)

        if self._resume_optimizer_state is not None:
            named_parameters = dict(self.model.named_parameters())
            for name, requires_grad in state.get("parameter_requires_grad", {}).items():
                named_parameters[name].requires_grad_(requires_grad)
            parameter_groups = state.get("optimizer_param_names", [])
            if parameter_groups:
                self.optimizer.state = defaultdict(dict)
                self.optimizer.param_groups = []
                for group in parameter_groups:
                    self.optimizer.add_param_group({"params": [named_parameters[name] for name in group]})
            self.optimizer.load_state_dict(self._resume_optimizer_state)

        rng_state = state.get("rng_state", {})
        if "cpu" in rng_state:
            torch.set_rng_state(rng_state["cpu"])
        if torch.cuda.is_available() and rng_state.get("cuda"):
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    def _fit_epoch(self, mb: ConsoleMasterBar | NBMasterBar) -> None:
        """Fit a single epoch

        Args:
            mb: primary progress bar

        Raises:
            ValueError: if the loss value is NaN or inf
        """
        freeze_bn(self.model.train())

        nan_cnt = 0

        pb = progress_bar(self.train_loader, parent=mb)
        for x, target in pb:
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]

            # Backprop
            if not self.skip_nan_loss or torch.isfinite(batch_loss):
                nan_cnt = 0
                if self._backprop_step(batch_loss):
                    self.scheduler.step()
            else:
                nan_cnt += 1
                if nan_cnt > self.nan_tolerance:
                    raise ValueError(f"loss value has been NaN or inf for more than {self.nan_tolerance} steps.")
            pb.comment = f"Training loss: {batch_loss.item():.4}"

            self.step += 1
        if self._optimizer_step():
            self.scheduler.step()
        self.epoch += 1

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

    @staticmethod
    def _to_cuda(x: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        return x, target

    def _backprop_step(self, loss: Tensor, force: bool = False) -> bool:
        self._grad_count += 1
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self._grad_count < self.gradient_acc and not force:
            return False
        return self._optimizer_step()

    def _optimizer_step(self) -> bool:
        if self._grad_count == 0:
            return False

        if self.amp:
            self.scaler.unscale_(self.optimizer)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(self._grad_count)
        if isinstance(self.grad_clip, float):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.amp:
            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = self.scaler.get_scale() >= scale
        else:
            self.optimizer.step()
            stepped = True
        self.optimizer.zero_grad()
        self._grad_count = 0
        return stepped

    def _get_loss(self, x: Tensor, target: Tensor, return_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        # AMP
        if self.amp:
            with torch.amp.autocast("cuda"):
                # Forward
                out = self.model(x)
                # Loss computation
                loss = cast(Tensor, self.criterion(out, target))
                if return_logits:
                    return loss, out
                return loss

        # Forward
        out = self.model(x)
        loss = cast(Tensor, self.criterion(out, target))
        if return_logits:
            return loss, out
        return loss

    def _set_params(self, norm_weight_decay: float | None = None) -> None:
        if not any(p.requires_grad for p in self.model.parameters()):
            raise AssertionError("All parameters are frozen")

        if norm_weight_decay is None:
            self._params = [p for p in self.model.parameters() if p.requires_grad], []
        else:
            self._params = split_normalization_params(self.model)

    def _reset_opt(self, lr: float, norm_weight_decay: float | None = None) -> None:
        """Reset the target params of the optimizer"""
        self.optimizer.defaults["lr"] = lr
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups = []
        self._set_params(norm_weight_decay)
        # Split it if norm layers needs custom WD
        if norm_weight_decay is None:
            self.optimizer.add_param_group({"params": self._params[0]})
        else:
            wd_groups = [norm_weight_decay, self.optimizer.defaults.get("weight_decay", 0)]
            for params, wd in zip(self._params, wd_groups, strict=True):
                if len(params) > 0:
                    self.optimizer.add_param_group({"params": params, "weight_decay": wd})
        self.optimizer.zero_grad()
        self._grad_count = 0

    @torch.inference_mode()
    def evaluate(self):  # type: ignore[no-untyped-def]  # noqa: D102, ANN201
        raise NotImplementedError

    @staticmethod
    def _eval_metrics_str(eval_metrics) -> str:  # type: ignore[no-untyped-def]  # noqa: ANN001
        raise NotImplementedError

    def _reset_scheduler(self, lr: float, num_epochs: int, sched_type: str = "onecycle", **kwargs: Any) -> None:
        self.scheduler: LRScheduler
        num_steps = num_epochs * math.ceil(len(self.train_loader) / self.gradient_acc)
        if sched_type == "onecycle":
            self.scheduler = OneCycleLR(self.optimizer, lr, num_steps, **kwargs)
        elif sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, num_steps, **kwargs)
        else:
            raise ValueError(f"The following scheduler type is not supported: {sched_type}")

    def fit_n_epochs(
        self,
        num_epochs: int,
        lr: float,
        freeze_until: str | None = None,
        sched_type: str = "onecycle",
        norm_weight_decay: float | None = None,
        run_dir: str | None = None,
        **kwargs: Any,
    ) -> RunResult:
        """Train the model for a given number of epochs.

        A resumed run reuses its stored optimizer and scheduler configuration;
        ``num_epochs`` is the number of remaining epochs to execute.

        Args:
            num_epochs: number of epochs to train
            lr: learning rate to be used by the scheduler
            freeze_until: last layer to freeze
            sched_type: type of scheduler to use
            norm_weight_decay: weight decay to apply to normalization parameters
            run_dir: optional directory for a schema-v1 run bundle
            **kwargs: keyword args passed to the [`LRScheduler`][torch.optim.lr_scheduler.LRScheduler]

        Returns:
            run result including metrics, resolved configuration, and checkpoint path

        Raises:
            ValueError: if a resumed run would exceed its original schedule
        """
        if self._is_resuming:
            resume_config = self._run_config
            if self.epoch + num_epochs > int(resume_config["end_epoch"]):
                raise ValueError("resumed run exceeds the original scheduler length")
            freeze_until = resume_config.get("freeze_until")
            freeze_model(self.model.train(), freeze_until)
            self._reset_scheduler(
                float(resume_config["lr"]),
                int(resume_config["num_epochs"]),
                str(resume_config["sched_type"]),
                **resume_config.get("scheduler_kwargs", {}),
            )
            if self._resume_scheduler_state is not None:
                self.scheduler.load_state_dict(self._resume_scheduler_state)
            if self._resume_optimizer_state is not None:
                self.optimizer.load_state_dict(self._resume_optimizer_state)
            if self.amp:
                self.scaler = GradScaler("cuda")
                if self._resume_scaler_state is not None:
                    self.scaler.load_state_dict(self._resume_scaler_state)
            self._is_resuming = False
        else:
            self.start_epoch = self.epoch
            self._run_config = {
                "num_epochs": num_epochs,
                "start_epoch": self.start_epoch,
                "end_epoch": self.start_epoch + num_epochs,
                "lr": lr,
                "freeze_until": freeze_until,
                "sched_type": sched_type,
                "norm_weight_decay": norm_weight_decay,
                "scheduler_kwargs": _primitive(kwargs),
            }
            self._metrics_history = []
            freeze_model(self.model.train(), freeze_until)
            self._reset_opt(lr, norm_weight_decay)
            self._reset_scheduler(lr, num_epochs, sched_type, **kwargs)

            if self.amp:
                self.scaler = GradScaler("cuda")

        mb = master_bar(range(num_epochs))
        for _ in mb:
            self._fit_epoch(mb)
            eval_metrics = self.evaluate()
            metric_record: dict[str, float | int | None] = {"epoch": self.epoch, "step": self.step}
            metric_record.update({key: None if value is None else float(value) for key, value in eval_metrics.items()})
            self._metrics_history.append(metric_record)

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

        checkpoint = str(Path(self.output_file)) if Path(self.output_file).is_file() else None
        result = RunResult(
            epoch=self.epoch,
            step=self.step,
            best_metric=self.min_loss,
            metrics=tuple(self._metrics_history),
            config=self._checkpoint_config(),
            checkpoint=checkpoint,
            bundle_dir=run_dir,
        )
        if run_dir is not None:
            write_run_bundle(run_dir, result)
        return result

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
           num_it: maximum number of microbatches to consume

        Raises:
            ValueError: if the number of iterations is greater than the number of available batches
        """
        if num_it > len(self.train_loader):
            raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

        freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(start_lr, norm_weight_decay)
        num_steps = math.ceil(num_it / self.gradient_acc)
        gamma = (end_lr / start_lr) ** (1 / (num_steps - 1)) if num_steps > 1 else 1
        scheduler = MultiplicativeLR(self.optimizer, lambda step: gamma)

        self.lr_recorder = []
        self.loss_recorder = []
        accumulated_loss = 0.0
        accumulated_batches = 0

        if self.amp:
            self.scaler = GradScaler("cuda")

        for batch_idx, (x, target) in enumerate(self.train_loader):
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                if batch_idx == 0:
                    raise ValueError("loss value is NaN or inf.")
                break

            accumulated_loss += batch_loss.item()
            accumulated_batches += 1
            is_last_batch = batch_idx + 1 == num_it
            stepped = self._backprop_step(batch_loss, force=is_last_batch)
            if self._grad_count == 0:
                if stepped:
                    self.lr_recorder.append(float(self.optimizer.param_groups[0]["lr"]))
                    self.loss_recorder.append(accumulated_loss / accumulated_batches)
                    scheduler.step()
                accumulated_loss = 0.0
                accumulated_batches = 0

            if is_last_batch:
                break

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
