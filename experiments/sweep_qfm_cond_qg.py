from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils.jet_metrics import compute_jet_metrics


def get_device(name: str | None) -> torch.device:
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
class DataBundle:
    x_all: torch.Tensor  # (N,1,64,64) in [0,1]
    q_enc_all: torch.Tensor  # (N,4,32,32)


def load_qg_pixels(
    *,
    channel: int,
    preprocess: bool = True,
    center: bool = True,
) -> torch.Tensor:
    filename = Path(f"data/QG{channel}_64x64_1k")
    if not filename.exists():
        raise FileNotFoundError(str(filename))

    with h5py.File(filename, "r") as f:
        data_x = np.array(f["X"])

    if preprocess:
        data_x = data_x.astype(np.float32)
        data_x = np.log1p(data_x)
        data_x = data_x / (data_x.max() + 1e-8)

    x = torch.tensor(data_x, dtype=torch.float32).unsqueeze(1)  # (N,1,64,64)

    if center:
        # Keep centering consistent with the notebook's preprocessing.
        from utils.jet_metrics import center_by_energy_centroid_zeropad

        x = center_by_energy_centroid_zeropad(x)

    return x


def load_q_enc(
    *,
    channel: int,
    expected_n: int,
) -> torch.Tensor:
    # Precomputed patch-based QFM encoding (NHWC): (N,32,32,4)
    enc_path = Path(f"data/QG{channel}_64x64_{expected_n}_encoded.pt")
    if not enc_path.exists():
        raise FileNotFoundError(
            f"Missing precomputed q_enc at {enc_path}. "
            "Generate it once via the notebook or add a generator script."
        )

    q = torch.load(enc_path, map_location="cpu")
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {enc_path}, got {type(q)}")
    if q.ndim != 4 or q.shape[-1] != 4:
        raise ValueError(f"Expected q_enc NHWC (N,32,32,4); got {tuple(q.shape)}")
    if q.shape[0] != expected_n:
        raise ValueError(f"Expected q_enc N={expected_n}; got {q.shape[0]}")

    # Convert NHWC -> NCHW
    q = q.permute(0, 3, 1, 2).contiguous().to(torch.float32)
    return q


def build_permutation(n: int, *, seed: int, mode: str) -> np.ndarray:
    if mode == "prefix":
        return np.arange(n, dtype=np.int64)
    if mode == "random":
        rng = np.random.RandomState(seed)
        return rng.permutation(n).astype(np.int64)
    raise ValueError("subset_mode must be 'prefix' or 'random'")


#########################################
# Diffusion utilities + model (from nb)
#########################################


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-6, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t).float()
    return out.view(-1, 1, 1, 1)


def q_sample(
    *,
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ac = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_om = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_ac * x0 + sqrt_om * noise


def predict_eps_from_x0(
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x0: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    sqrt_ac = extract(sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_om = extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    return (x_t - sqrt_ac * x0) / sqrt_om


class DoubleConv(nn.Module):
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


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        return h, self.pool(h)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.conv = DoubleConv(out_ch * 2, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = x + self.time_proj(t_emb)[:, :, None, None]
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, n_steps: int, time_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_steps, time_dim)
        self.mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(t))


class SimpleUNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base: int = 64,
        time_dim: int = 128,
        n_steps: int = 200,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(n_steps, time_dim)
        self.down1 = Down(in_channels, base, time_dim)
        self.down2 = Down(base, base * 2, time_dim)
        self.bot = DoubleConv(base * 2, base * 4)
        self.up2 = Up(base * 4, base * 2, time_dim)
        self.up1 = Up(base * 2, base, time_dim)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h1, x = self.down1(x, t_emb)
        h2, x = self.down2(x, t_emb)
        x = self.bot(x)
        x = self.up2(x, h2, t_emb)
        x = self.up1(x, h1, t_emb)
        return self.out(x)


#########################################
# Post-encoding scrambling augmentation
#########################################


def _make_cnot_perm(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    dim = 2**n_qubits
    idx = torch.arange(dim, dtype=torch.long, device=device)
    mask = (idx >> control) & 1
    perm = idx.clone()
    perm[mask == 1] = perm[mask == 1] ^ (1 << target)
    return perm


@torch.no_grad()
def _apply_ry(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    # state: (B, 2**n), theta: (B,)
    B = state.shape[0]
    dim_hi = 2 ** (n_qubits - qubit - 1)
    dim_lo = 2**qubit

    psi = state.view(B, dim_hi, 2, dim_lo)
    a0 = psi[:, :, 0, :]
    a1 = psi[:, :, 1, :]

    half = 0.5 * theta
    c = torch.cos(half).view(B, 1, 1)
    s = torch.sin(half).view(B, 1, 1)

    b0 = c * a0 - s * a1
    b1 = s * a0 + c * a1
    return torch.stack([b0, b1], dim=2).reshape(B, -1)


@torch.no_grad()
def scramble_qenc_batch(
    q: torch.Tensor,
    *,
    depth: int,
    seed: int,
    shared_unitary: bool,
) -> torch.Tensor:
    # q: (B,C,H,W) real tensor; treated as a 2**n vector per sample
    B = q.shape[0]
    flat = q.view(B, -1)
    q_dim = int(flat.shape[1])
    n_qubits = int(np.log2(q_dim))
    if 2**n_qubits != q_dim:
        raise ValueError(f"Encoded conditioning dimension must be a power of two; got {q_dim}")

    # Normalize to a unit vector (valid state amplitudes), but rescale back after scrambling
    norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    state = flat / norm

    # Generate angles on CPU for broad device compatibility, then move to device
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if shared_unitary:
        thetas = 2 * math.pi * torch.rand((depth, n_qubits), generator=gen)
        thetas = thetas.unsqueeze(0).expand(B, -1, -1)
    else:
        thetas = 2 * math.pi * torch.rand((B, depth, n_qubits), generator=gen)
    thetas = thetas.to(state.device)

    # CNOT chain perms (control=i, target=i+1)
    perms = [_make_cnot_perm(n_qubits, i, i + 1, state.device) for i in range(n_qubits - 1)]

    for d in range(depth):
        for qb in range(n_qubits):
            state = _apply_ry(state, thetas[:, d, qb], qb, n_qubits)
        for p in perms:
            state = state.index_select(1, p)

    flat_scr = state * norm
    return flat_scr.view_as(q)


def upsample_cond(q: torch.Tensor) -> torch.Tensor:
    return F.interpolate(q, size=(64, 64), mode="bilinear", align_corners=False)


def decode_x0_from_heads(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # out: (N,2,H,W)
    occ_logit = out[:, :1]
    int_logit = out[:, 1:2]
    occ_prob = torch.sigmoid(occ_logit)
    intensity = torch.sigmoid(int_logit)
    x0_hat = occ_prob * intensity
    return x0_hat, occ_logit, intensity, occ_prob


@torch.no_grad()
def one_step_recon_x0(
    *,
    model: nn.Module,
    x0: torch.Tensor,
    t_mid: int,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    gen: torch.Generator,
    cond: torch.Tensor | None = None,
) -> torch.Tensor:
    t = torch.full((x0.shape[0],), t_mid, device=x0.device, dtype=torch.long)
    eps = torch.randn(x0.shape, device=x0.device, generator=gen)
    x_t = q_sample(
        x0=x0,
        t=t,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        noise=eps,
    )
    if cond is None:
        out = model(x_t, t)
    else:
        out = model(torch.cat([x_t, cond], dim=1), t)
    x0_hat, _, _, _ = decode_x0_from_heads(out)
    return x0_hat


def train_epoch_px(
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    T: int,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    occ_thr: float,
    w_occ: float,
    w_int: float,
    w_bg: float,
) -> float:
    model.train()
    totals = []
    for x0, _dummy in loader:
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        x_t = q_sample(
            x0=x0,
            t=t,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            noise=torch.randn_like(x0),
        )
        out = model(x_t, t)
        x0_hat, occ_logit, intensity, _occ_prob = decode_x0_from_heads(out)

        occ_true = (x0 > occ_thr).float()
        loss_occ = F.binary_cross_entropy_with_logits(occ_logit, occ_true)
        loss_int = (((intensity - x0) ** 2) * occ_true).sum() / (occ_true.sum() + 1e-8)
        loss_bg = ((intensity**2) * (1.0 - occ_true)).mean()
        loss = w_occ * loss_occ + w_int * loss_int + w_bg * loss_bg

        opt.zero_grad()
        loss.backward()
        opt.step()
        totals.append(float(loss.item()))
    return float(np.mean(totals))


def train_epoch_qfm(
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    T: int,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    occ_thr: float,
    w_occ: float,
    w_int: float,
    w_bg: float,
    cond_drop_p: float,
) -> float:
    model.train()
    totals = []
    for x0, q0 in loader:
        x0 = x0.to(device)
        q0 = upsample_cond(q0.to(device))
        if cond_drop_p > 0:
            drop_mask = (torch.rand((x0.shape[0], 1, 1, 1), device=device) < cond_drop_p).float()
            q0 = q0 * (1.0 - drop_mask)

        t = torch.randint(0, T, (x0.shape[0],), device=device)
        x_t = q_sample(
            x0=x0,
            t=t,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            noise=torch.randn_like(x0),
        )

        out = model(torch.cat([x_t, q0], dim=1), t)
        x0_hat, occ_logit, intensity, _occ_prob = decode_x0_from_heads(out)

        occ_true = (x0 > occ_thr).float()
        loss_occ = F.binary_cross_entropy_with_logits(occ_logit, occ_true)
        loss_int = (((intensity - x0) ** 2) * occ_true).sum() / (occ_true.sum() + 1e-8)
        loss_bg = ((intensity**2) * (1.0 - occ_true)).mean()
        loss = w_occ * loss_occ + w_int * loss_int + w_bg * loss_bg

        opt.zero_grad()
        loss.backward()
        opt.step()
        totals.append(float(loss.item()))
    return float(np.mean(totals))


@torch.no_grad()
def p_sample(
    *,
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    posterior_variance: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    cond: torch.Tensor | None,
    clip_x0: bool = True,
) -> torch.Tensor:
    betas_t = extract(betas, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    sqrt_om = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)

    if cond is None:
        out = model(x, t)
    else:
        out = model(torch.cat([x, cond], dim=1), t)

    x0_pred, _, _, _ = decode_x0_from_heads(out)
    if clip_x0:
        x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

    eps = predict_eps_from_x0(
        x_t=x,
        t=t,
        x0=x0_pred,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
    )

    model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_om)
    if (t == 0).all():
        return model_mean

    noise = torch.randn_like(x)
    var = extract(posterior_variance, t, x.shape)
    return model_mean + torch.sqrt(var) * noise


@torch.no_grad()
def p_sample_loop(
    *,
    model: nn.Module,
    n: int,
    T: int,
    device: torch.device,
    betas: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    posterior_variance: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    cond: torch.Tensor | None = None,
    clip_x0: bool = True,
) -> torch.Tensor:
    x = torch.randn((n, 1, 64, 64), device=device)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        x = p_sample(
            model=model,
            x=x,
            t=t,
            betas=betas,
            sqrt_recip_alphas=sqrt_recip_alphas,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            posterior_variance=posterior_variance,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            cond=cond,
            clip_x0=clip_x0,
        )
    return x


def _as_list_int(values: Iterable[str]) -> list[int]:
    out = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        out.append(int(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--subset_mode", choices=("prefix", "random"), default="prefix")
    ap.add_argument("--raw_q_counts", type=str, default="125,250,500")
    ap.add_argument("--seeds", type=str, default="123,456,789")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--device", type=str, default=None)

    # Augmentation (post-encoding scrambling)
    ap.add_argument("--aug_depth", type=int, default=2)
    ap.add_argument("--aug_shared_unitary", action="store_true", default=True)
    ap.add_argument("--aug_seed", type=int, default=123)

    # Optional sample-based eval
    ap.add_argument("--eval_samples", type=int, default=0, help="If >0, generate this many samples for metrics.")

    ap.add_argument("--out_csv", type=str, default="plots/qfm_cond_qg_learning_curve.csv")
    args = ap.parse_args()

    device = get_device(args.device)
    raw_q_counts = _as_list_int(args.raw_q_counts.split(","))
    seeds = _as_list_int(args.seeds.split(","))
    if not raw_q_counts:
        raise ValueError("--raw_q_counts must be non-empty")
    if not seeds:
        raise ValueError("--seeds must be non-empty")

    # Load full 1k dataset once.
    x_all = load_qg_pixels(channel=args.channel, preprocess=True, center=True)
    q_enc_all = load_q_enc(channel=args.channel, expected_n=int(x_all.shape[0]))
    data = DataBundle(x_all=x_all, q_enc_all=q_enc_all)

    # Diffusion schedule tensors
    T = int(args.T)
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "raw_q",
        "raw_px",
        "train_views_q",
        "epochs",
        "metric_kind",
        "model",
        "val_x0_mse",
        "E_w1",
        "active_frac_w1",
        "radial_l2",
        "radial_l2_log",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # Fixed reference set for jet-metric comparisons (constant across runs).
        real_ref_full = data.x_all.to(device).clamp(0.0, 1.0)
        n_fix = min(64, int(data.x_all.shape[0]))
        eval_x0 = data.x_all[:n_fix].to(device).clamp(0.0, 1.0)
        eval_q0 = upsample_cond(data.q_enc_all[:n_fix].to(device))

        for seed in seeds:
            set_seed(seed)
            perm = build_permutation(int(data.x_all.shape[0]), seed=seed, mode=args.subset_mode)

            for raw_q in raw_q_counts:
                raw_px = int(2 * raw_q)
                if raw_px > data.x_all.shape[0]:
                    raise ValueError(f"raw_px={raw_px} exceeds dataset size {data.x_all.shape[0]}")

                idx_q = perm[:raw_q]
                idx_px = perm[:raw_px]

                x_px = data.x_all[idx_px].clone()
                x_q = data.x_all[idx_q].clone()
                q_enc = data.q_enc_all[idx_q].clone()

                # Splits: baseline uses its own raw set; qfm uses paired (x_q, q_enc)
                train_px_full, val_px_full = train_test_split(
                    x_px, test_size=0.2, random_state=seed, shuffle=True
                )
                train_px, val_px, train_q, val_q = train_test_split(
                    x_q, q_enc, test_size=0.2, random_state=seed, shuffle=True
                )

                # Dataloaders
                dummy_px = torch.zeros((train_px_full.shape[0], 1), dtype=torch.float32)
                train_loader_px = DataLoader(
                    TensorDataset(train_px_full, dummy_px),
                    batch_size=args.batch_size,
                    shuffle=True,
                )

                train_q_aug = scramble_qenc_batch(
                    train_q.to(device),
                    depth=int(args.aug_depth),
                    seed=int(args.aug_seed + seed),
                    shared_unitary=bool(args.aug_shared_unitary),
                ).cpu()
                train_px_q = torch.cat([train_px, train_px], dim=0)
                train_q_q = torch.cat([train_q, train_q_aug], dim=0)
                train_loader_q = DataLoader(
                    TensorDataset(train_px_q, train_q_q),
                    batch_size=args.batch_size,
                    shuffle=True,
                )

                # Models
                q_channels = int(train_q.shape[1])
                model_px = SimpleUNet(in_channels=1, out_channels=2, base=64, time_dim=128, n_steps=T).to(device)
                model_q = SimpleUNet(
                    in_channels=1 + q_channels, out_channels=2, base=64, time_dim=128, n_steps=T
                ).to(device)

                opt_px = torch.optim.Adam(model_px.parameters(), lr=2e-4)
                opt_q = torch.optim.Adam(model_q.parameters(), lr=2e-4)

                # Training hyperparams consistent with notebook
                occ_thr = 0.01
                w_occ, w_int, w_bg = 1.0, 5.0, 0.1
                cond_drop_p = 0.1

                for _e in range(int(args.epochs)):
                    train_epoch_px(
                        model=model_px,
                        opt=opt_px,
                        loader=train_loader_px,
                        device=device,
                        T=T,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        occ_thr=occ_thr,
                        w_occ=w_occ,
                        w_int=w_int,
                        w_bg=w_bg,
                    )
                    train_epoch_qfm(
                        model=model_q,
                        opt=opt_q,
                        loader=train_loader_q,
                        device=device,
                        T=T,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        occ_thr=occ_thr,
                        w_occ=w_occ,
                        w_int=w_int,
                        w_bg=w_bg,
                        cond_drop_p=cond_drop_p,
                    )

                eval_gen = torch.Generator(device=device).manual_seed(int(seed + 1))
                t_mid = T // 2

                rec_px = one_step_recon_x0(
                    model=model_px,
                    x0=eval_x0,
                    t_mid=t_mid,
                    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    gen=eval_gen,
                    cond=None,
                )
                rec_q = one_step_recon_x0(
                    model=model_q,
                    x0=eval_x0,
                    t_mid=t_mid,
                    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    gen=eval_gen,
                    cond=eval_q0,
                )

                def _log_row(
                    model_name: str,
                    kind: str,
                    *,
                    real_ref: torch.Tensor,
                    x_hat: torch.Tensor,
                    val_mse: float | None,
                ) -> None:
                    real_ref = real_ref.clamp(0.0, 1.0)
                    x_hat = x_hat.clamp(0.0, 1.0)
                    jet = compute_jet_metrics(real_ref, x_hat)
                    w.writerow(
                        {
                            "seed": seed,
                            "raw_q": raw_q,
                            "raw_px": raw_px,
                            "train_views_q": int(train_px.shape[0] * 2),
                            "epochs": int(args.epochs),
                            "metric_kind": kind,
                            "model": model_name,
                            "val_x0_mse": float("nan") if val_mse is None else float(val_mse),
                            "E_w1": jet["E_w1"],
                            "active_frac_w1": jet["active_frac_w1"],
                            "radial_l2": jet["radial_l2"],
                            "radial_l2_log": jet["radial_l2_log"],
                        }
                    )

                _log_row(
                    "px",
                    "recon",
                    real_ref=eval_x0,
                    x_hat=rec_px,
                    val_mse=float(F.mse_loss(rec_px.clamp(0.0, 1.0), eval_x0).item()),
                )
                _log_row(
                    "qfm",
                    "recon",
                    real_ref=eval_x0,
                    x_hat=rec_q,
                    val_mse=float(F.mse_loss(rec_q.clamp(0.0, 1.0), eval_x0).item()),
                )

                if int(args.eval_samples) > 0:
                    n_samp = int(args.eval_samples)
                    samples_px = p_sample_loop(
                        model=model_px,
                        n=n_samp,
                        T=T,
                        device=device,
                        betas=betas,
                        sqrt_recip_alphas=sqrt_recip_alphas,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        posterior_variance=posterior_variance,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        cond=None,
                        clip_x0=True,
                    )

                    # For conditional sampling, reuse fixed eval conditions (wrap if needed).
                    if n_samp <= n_fix:
                        cond = eval_q0[:n_samp]
                    else:
                        reps = int(math.ceil(n_samp / n_fix))
                        cond = eval_q0.repeat(reps, 1, 1, 1)[:n_samp]

                    samples_q = p_sample_loop(
                        model=model_q,
                        n=n_samp,
                        T=T,
                        device=device,
                        betas=betas,
                        sqrt_recip_alphas=sqrt_recip_alphas,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        posterior_variance=posterior_variance,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        cond=cond,
                        clip_x0=True,
                    )

                    _log_row("px", "sample", real_ref=real_ref_full, x_hat=samples_px, val_mse=None)
                    _log_row("qfm", "sample", real_ref=real_ref_full, x_hat=samples_q, val_mse=None)

    print(f"Wrote learning-curve metrics to: {out_csv}")


if __name__ == "__main__":
    main()
