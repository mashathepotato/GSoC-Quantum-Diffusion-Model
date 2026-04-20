from __future__ import annotations

"""
Classical baseline models (diffusion + flow matching) for apples-to-apples comparisons.

Design goals:
  - Self-contained PyTorch implementations (no network downloads).
  - Optional hooks for off-the-shelf libraries (diffusers / etc.) behind try/except.
  - A small registry + consistent train/sample/save/load surface so it's easy to add
    new baselines later.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "utils/baselines.py requires PyTorch. Activate the project's environment (e.g. .venv)."
    ) from e


# Optional, off-the-shelf implementations (not required for this repo to run).
try:  # pragma: no cover
    import diffusers as _diffusers  # noqa: F401
except Exception:  # pragma: no cover
    _diffusers = None

try:  # pragma: no cover
    import denoising_diffusion_pytorch as _ddp  # noqa: F401
except Exception:  # pragma: no cover
    _ddp = None

try:  # pragma: no cover
    import torchcfm as _torchcfm  # noqa: F401
except Exception:  # pragma: no cover
    _torchcfm = None


class BaselineError(RuntimeError):
    pass


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 2e-4
    grad_clip: float | None = 1.0
    amp: bool = False


@dataclass(frozen=True)
class SampleConfig:
    n: int = 64
    steps: int | None = None
    seed: int = 123


class Baseline:
    """
    Minimal interface for baseline models.
    Subclasses should implement: fit(), sample(), save(), load().
    """

    name: str

    def __init__(self, *, device: torch.device | None = None):
        self.device = device or get_device()

    def fit(self, x_train: torch.Tensor, *, cfg: TrainConfig) -> dict[str, float]:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, *, cfg: SampleConfig) -> torch.Tensor:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | None = None) -> "Baseline":
        raise NotImplementedError


_REGISTRY: dict[str, type[Baseline]] = {}


def register(name: str) -> Callable[[type[Baseline]], type[Baseline]]:
    def _wrap(cls: type[Baseline]) -> type[Baseline]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise BaselineError(f"Baseline '{name}' already registered to {_REGISTRY[name].__name__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def list_baselines() -> list[str]:
    return sorted(_REGISTRY.keys())


def create_baseline(name: str, **kwargs: Any) -> Baseline:
    if name not in _REGISTRY:
        raise BaselineError(f"Unknown baseline '{name}'. Available: {list_baselines()}")
    return _REGISTRY[name](**kwargs)


def _to_model_range01_to_pm1(x01: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1]
    return x01 * 2.0 - 1.0


def _from_model_range_pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x + 1.0) * 0.5


def _check_image_tensor(x: torch.Tensor) -> None:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected x shape (N,1,H,W); got {tuple(x.shape)}")

