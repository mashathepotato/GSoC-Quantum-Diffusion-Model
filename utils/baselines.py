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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
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


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64
    time_dim: int = 128


def _timestep_embedding(t: torch.Tensor, dim: int, *, max_period: int = 10_000) -> torch.Tensor:
    """
    Standard sinusoidal embedding.

    Args:
        t: (B,) float or int tensor. Treats t as continuous.
        dim: embedding dimension
    """
    if t.ndim != 1:
        raise ValueError(f"Expected t shape (B,), got {tuple(t.shape)}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.to(torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=-1)
    return emb


class _TimeMLP(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        return h, self.pool(h)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.conv = _DoubleConv(out_ch * 2, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = x + self.time_proj(t_emb)[:, :, None, None]
        return x


class UNet(nn.Module):
    """
    Lightweight U-Net used for both DDPM and Rectified Flow baselines.
    Time conditioning uses a sinusoidal embedding + MLP, allowing float or int timesteps.
    """

    def __init__(self, cfg: UNetConfig, *, max_period: int = 10_000):
        super().__init__()
        self.cfg = cfg
        self.max_period = int(max_period)

        time_dim = int(cfg.time_dim)
        base = int(cfg.base_channels)

        self.time_mlp = _TimeMLP(time_dim)
        self.down1 = _Down(cfg.in_channels, base, time_dim)
        self.down2 = _Down(base, base * 2, time_dim)
        self.bot = _DoubleConv(base * 2, base * 4)
        self.up2 = _Up(base * 4, base * 2, time_dim)
        self.up1 = _Up(base * 2, base, time_dim)
        self.out = nn.Conv2d(base, cfg.out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected t shape (B,) matching x batch; got t={tuple(t.shape)} x={tuple(x.shape)}")
        t_emb = _timestep_embedding(t, self.cfg.time_dim, max_period=self.max_period)
        t_emb = self.time_mlp(t_emb)

        h1, x = self.down1(x, t_emb)
        h2, x = self.down2(x, t_emb)
        x = self.bot(x)
        x = self.up2(x, h2, t_emb)
        x = self.up1(x, h1, t_emb)
        return self.out(x)


def _cosine_beta_schedule(timesteps: int, *, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-6, 0.999)


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t).float()
    return out.view(-1, 1, 1, 1)


@dataclass(frozen=True)
class DDPMConfig:
    image_size: int = 64
    timesteps: int = 200
    base_channels: int = 64
    time_dim: int = 128
    data_range: str = "01"  # "01" or "pm1"


def _to_model_range(x01: torch.Tensor, data_range: str) -> torch.Tensor:
    if data_range == "01":
        return x01
    if data_range == "pm1":
        return _to_model_range01_to_pm1(x01)
    raise ValueError("data_range must be '01' or 'pm1'")


def _from_model_range(x: torch.Tensor, data_range: str) -> torch.Tensor:
    if data_range == "01":
        return x
    if data_range == "pm1":
        return _from_model_range_pm1_to_01(x)
    raise ValueError("data_range must be '01' or 'pm1'")


@register("ddpm")
class DDPMBaseline(Baseline):
    def __init__(
        self,
        *,
        cfg: DDPMConfig | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(device=device)
        self.cfg = cfg or DDPMConfig()

        model_cfg = UNetConfig(
            in_channels=1,
            out_channels=1,  # eps prediction
            base_channels=self.cfg.base_channels,
            time_dim=self.cfg.time_dim,
        )
        self.model = UNet(model_cfg).to(self.device)

        self.T = int(self.cfg.timesteps)
        betas = _cosine_beta_schedule(self.T).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def fit(self, x_train: torch.Tensor, *, cfg: TrainConfig) -> dict[str, float]:
        _check_image_tensor(x_train)
        x_train = x_train.to(self.device, dtype=torch.float32)

        if x_train.shape[2] != self.cfg.image_size or x_train.shape[3] != self.cfg.image_size:
            raise ValueError(
                f"Expected images {self.cfg.image_size}x{self.cfg.image_size}; got {tuple(x_train.shape[2:])}"
            )

        x_train_m = _to_model_range(x_train, self.cfg.data_range)
        loader = DataLoader(TensorDataset(x_train_m), batch_size=cfg.batch_size, shuffle=True, drop_last=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.model.train()
        last_loss = float("nan")

        use_amp = bool(cfg.amp) and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for _epoch in range(cfg.epochs):
            losses = []
            for (x0,) in loader:
                x0 = x0.to(self.device)
                t = torch.randint(0, self.T, (x0.shape[0],), device=self.device, dtype=torch.long)
                noise = torch.randn_like(x0)

                sqrt_ac = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
                sqrt_om = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
                x_t = sqrt_ac * x0 + sqrt_om * noise

                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(x_t, t.to(torch.float32))
                        loss = F.mse_loss(pred, noise)
                    scaler.scale(loss).backward()
                    if cfg.grad_clip is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = self.model(x_t, t.to(torch.float32))
                    loss = F.mse_loss(pred, noise)
                    loss.backward()
                    if cfg.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    opt.step()

                losses.append(float(loss.detach().cpu().item()))

            last_loss = float(np.mean(losses)) if losses else last_loss

        return {"train_loss": last_loss}

    @torch.no_grad()
    def sample(self, *, cfg: SampleConfig) -> torch.Tensor:
        set_seed(int(cfg.seed))
        self.model.eval()

        n = int(cfg.n)
        x = torch.randn((n, 1, self.cfg.image_size, self.cfg.image_size), device=self.device)

        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            betas_t = _extract(self.betas, t, x.shape)
            sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x.shape)
            sqrt_om = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

            eps_pred = self.model(x, t.to(torch.float32))

            # DDPM mean (predict eps, derive posterior mean)
            model_mean = sqrt_recip_alphas_t * (x - betas_t * eps_pred / sqrt_om)

            if i == 0:
                x = model_mean
                break

            var = _extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(var) * noise

        # Convert to [0,1] for downstream jet metrics
        x01 = _from_model_range(x, self.cfg.data_range)
        return torch.clamp(x01, 0.0, 1.0).detach().cpu()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "cfg": self.cfg,
            "model_cfg": self.model.cfg,
            "model_state": self.model.state_dict(),
        }
        torch.save(payload, p)

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | None = None) -> "DDPMBaseline":
        payload = torch.load(Path(path), map_location="cpu")
        cfg = payload.get("cfg")
        if cfg is None:
            raise BaselineError(f"Missing 'cfg' in checkpoint: {path}")
        obj = cls(cfg=cfg, device=device)
        obj.model.load_state_dict(payload["model_state"])
        obj.model.to(obj.device)
        return obj

