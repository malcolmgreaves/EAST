from pathlib import Path

import torch
from torch import nn
from torch.nn.modules import Module
from typing import TypeVar, Callable, Tuple, Optional, Any, Mapping

from model import EAST

Model = TypeVar("Model", bound=Module)

# @dataclass
# class LoadedModel(Generic[Model]):
#     model: Model
#     device: torch.device


def load_east_model(
    serialized_model: Path, pretrained: bool = True, set_eval: bool = True
) -> Tuple[EAST, torch.device]:
    return load_model(
        serialized_model, model_init=lambda: EAST(pretrained), set_eval=set_eval,
    )


def get_torch_device(cuda_device_num: int = 0) -> torch.device:
    return torch.device(
        f"cuda:{cuda_device_num}" if torch.cuda.is_available() else "cpu"
    )


def load_model(
    serialized_model: Path,
    model_init: Callable[[], Model],
    set_eval: bool = True,
    cuda_device_num: int = 0,
) -> Tuple[Model, torch.device]:
    device = torch.device(
        f"cuda:{cuda_device_num}" if torch.cuda.is_available() else "cpu"
    )
    model = model_init().to(device)
    model.load_state_dict(
        torch.load(str(serialized_model.absolute()), map_location=device)
    )
    if set_eval:
        model.eval()
    return model, device


class EarlyStopping:
    """Early stopping regularization. Use :func:`observe_step` on each model training epoch.

    Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self,
        model_name_prefix: str,
        lower_is_better: bool,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
    ) -> None:
        """Performs field assignment with the supplied parameters and initializes internal state.

        Args:
            model_name_prefix: Name for model
            lower_is_better: If `True`, lower values of the validation metric are better.
                             Otherwise, larger values are considered an improvement.
            patience: How long to wait after last time validation metric improved.
            verbose: If True, prints a message for each validation metric improvement.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.model_name_prefix = model_name_prefix
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.reset()

    def reset(self) -> None:
        """Sets all mutable state to initial conditions.

        NOTE: MUTATION: Initializes `counter`, `early_stop`, `best_val_metric`, `checkpoint_num`,
                        `best_name`.
        """
        self.counter = 0
        self.early_stop = False
        self.best_val_metric: Optional[float] = None
        self.checkpoint_num = 0
        self.best_name = ""

    def __call__(self, *args) -> bool:
        """Alias for :func:`observe_step` and then returns whether or not the
        early stopping criterion was hit.
        """
        self.observe_step(*args)
        return self.early_stop

    def observe_step(self, val_metric: float, model: nn.Module) -> None:
        """Observe the validation metric on the `model` for a discrete training step.

        NOTE: MUTATION: Potentially updates `counter`, `best_score`, `early_stop`,
                        `best_val_metric`, `checkpoint_num`.
        """
        if self.early_stop:
            if self.verbose:
                print(
                    f"Cannot observe step. Already stopped early.\n{self.saved_info()}"
                )
        elif self.loss_improvement(val_metric):
            self.save_checkpoint(val_metric, model)
        else:
            self.increment()

    def loss_improvement(self, val_metric: float) -> bool:
        """Evaluates to `True` iff `val_metric` is an improvement on the best observed validation metric.
        `False` otherwise.
        """
        return self.best_val_metric is None or (
            # e.g. new loss is lower than the best & the improvement threshold
            (val_metric < self.best_val_metric - self.delta)
            if self.lower_is_better
            else (val_metric > self.best_val_metric + self.delta)
            # e.g. new accuracy is higher than the best & the improvement threshold
        )

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Checkpoints model. Use when `val_loss` is an improvement.

        NOTE: MUTATION: Sets `best_val_metric`, `best_score` to neg. val loss, resets `counter`,
                        and increments `checkpoint_num`.
        """
        if self.verbose:
            if self.best_val_metric is None:
                print(
                    "Initial observation. "
                    f"Setting best validation metric to '{val_loss:.6f}' "
                    f"for checkpoint '{self.checkpoint_num}'"
                )
            else:
                print(
                    f"Validation metric improvement ({self.best_val_metric:.6f} --> {val_loss:.6f}). "
                    f"Saving model for checkpoint '{self.checkpoint_num}'..."
                )
        filename = self.checkpoint_name()
        torch.save(model.state_dict(), filename)
        self.best_name = filename
        self.best_val_metric = val_loss
        self.counter = 0
        self.checkpoint_num += 1

    def checkpoint_name(self) -> str:
        """Current filename for model when it is checkpointed next.
        """
        return f"{self.model_name_prefix}--{self.checkpoint_num}_checkpoint.pth"

    def increment(self) -> None:
        """Increment internal counters due to observing a training step without an improvement of validation loss.
        Sets `early_stop` to `True` iff the incrementing the `counter` here exceeds the `patience` threshold.

        NOTE: MUTATION: Increments `counter`, potentially sets `early_stop`.
        """
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Stopped early. {self.saved_info()}")

    def saved_info(self) -> str:
        """Human-readable logging string of the current minimum validation loss and checkpoint model filename.
        """
        return f"Best validation metric '{self.best_val_metric:.6f}' saved as '{self.best_name}'"
