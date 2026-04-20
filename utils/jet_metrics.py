from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@torch.no_grad()
def center_by_energy_centroid_zeropad(x01: torch.Tensor) -> torch.Tensor:
    """
    Center jet images by (energy-weighted) centroid using zero-padding (no wrap-around).

    Args:
        x01: (N,1,H,W) tensor in [0, 1].

    Returns:
        (N,1,H,W) centered tensor in [0, 1].
    """
    if x01.ndim != 4 or x01.shape[1] != 1:
        raise ValueError(f"Expected x01 shape (N,1,H,W); got {tuple(x01.shape)}")

    N, _, H, W = x01.shape
    w = x01
    w_sum = w.sum(dim=(2, 3), keepdim=True) + 1e-8

    yy = torch.arange(H, device=x01.device).view(1, 1, H, 1).float()
    xx = torch.arange(W, device=x01.device).view(1, 1, 1, W).float()

    cy = (w * yy).sum(dim=(2, 3), keepdim=True) / w_sum
    cx = (w * xx).sum(dim=(2, 3), keepdim=True) / w_sum

    ty = torch.round((H // 2) - cy.squeeze(-1).squeeze(-1).squeeze(1)).to(torch.int64)
    tx = torch.round((W // 2) - cx.squeeze(-1).squeeze(-1).squeeze(1)).to(torch.int64)

    out = torch.zeros_like(x01)
    for i in range(N):
        dy = int(ty[i].item())
        dx = int(tx[i].item())
        src_y0 = max(0, -dy)
        dst_y0 = max(0, dy)
        h = H - max(0, dy) - max(0, -dy)
        src_x0 = max(0, -dx)
        dst_x0 = max(0, dx)
        w_ = W - max(0, dx) - max(0, -dx)
        if h > 0 and w_ > 0:
            out[i, :, dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w_] = x01[
                i, :, src_y0 : src_y0 + h, src_x0 : src_x0 + w_
            ]
    return out


@torch.no_grad()
def center_by_energy_centroid_roll(x01: torch.Tensor) -> torch.Tensor:
    """
    Center jet images by (energy-weighted) centroid using wrap-around (torch.roll).

    This matches the centering used by the metric helper in `notebooks/quantum/qfm_cond_qg.ipynb`.

    Args:
        x01: (N,1,H,W) tensor in [0, 1].

    Returns:
        (N,1,H,W) centered tensor in [0, 1].
    """
    if x01.ndim != 4 or x01.shape[1] != 1:
        raise ValueError(f"Expected x01 shape (N,1,H,W); got {tuple(x01.shape)}")

    x01 = x01.clone()
    N, _, H, W = x01.shape

    w = x01
    w_sum = w.sum(dim=(2, 3), keepdim=True) + 1e-8

    yy = torch.arange(H, device=x01.device).view(1, 1, H, 1).float()
    xx = torch.arange(W, device=x01.device).view(1, 1, 1, W).float()

    cy = (w * yy).sum(dim=(2, 3), keepdim=True) / w_sum
    cx = (w * xx).sum(dim=(2, 3), keepdim=True) / w_sum

    ty = torch.round((H // 2) - cy.squeeze(-1).squeeze(-1).squeeze(1)).to(torch.int64)
    tx = torch.round((W // 2) - cx.squeeze(-1).squeeze(-1).squeeze(1)).to(torch.int64)

    out = x01
    for i in range(N):
        out[i] = torch.roll(out[i], shifts=(int(ty[i].item()), int(tx[i].item())), dims=(1, 2))
    return out


@dataclass(frozen=True)
class RadialBins:
    bin_w_flat: torch.Tensor  # (B, H*W)
    bin_norm: torch.Tensor  # (B,)
    bin_edges: torch.Tensor  # (B+1,)


def make_radial_bins(height: int, width: int, nbins: int, device: torch.device) -> RadialBins:
    if nbins <= 0:
        raise ValueError("nbins must be positive")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    rr = torch.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2).float()
    rmax = float(rr.max())
    edges = torch.linspace(0.0, rmax + 1e-6, nbins + 1, device=device)

    bin_w = []
    for b in range(nbins):
        m = ((rr >= edges[b]) & (rr < edges[b + 1])).float()
        bin_w.append(m)

    bin_w = torch.stack(bin_w, dim=0)  # (B,H,W)
    bin_w_flat = bin_w.view(nbins, -1)  # (B,P)
    bin_norm = bin_w_flat.sum(dim=1).clamp_min(1.0)  # (B,)
    return RadialBins(bin_w_flat=bin_w_flat, bin_norm=bin_norm, bin_edges=edges)


@torch.no_grad()
def radial_profile_batch(x01: torch.Tensor, bins: RadialBins) -> torch.Tensor:
    """
    Compute per-image mean radial profiles.

    Args:
        x01: (N,1,H,W) in [0,1]
        bins: precomputed RadialBins for (H,W)

    Returns:
        profiles: (N,B) where B=nbins
    """
    if x01.ndim != 4 or x01.shape[1] != 1:
        raise ValueError(f"Expected x01 shape (N,1,H,W); got {tuple(x01.shape)}")

    x_flat = x01.view(x01.shape[0], -1)
    prof = (x_flat @ bins.bin_w_flat.T) / bins.bin_norm
    return prof


def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(a.reshape(-1), b.reshape(-1)))


@torch.no_grad()
def compute_jet_metrics(
    real01: torch.Tensor,
    gen01: torch.Tensor,
    *,
    nbins: int = 32,
    thr_percentile: float = 99.5,
    centering: str = "zeropad",
) -> dict:
    """
    Domain-aware jet-image metrics intended for learning curves:
    - total energy distribution distance
    - active pixel fraction distribution distance
    - mean radial profile distance (linear + log)
    - tail fraction above a high percentile of real intensities

    Args:
        real01: (N,1,64,64) in [0,1]
        gen01:  (M,1,64,64) in [0,1]
    """
    if real01.ndim != 4 or real01.shape[1] != 1:
        raise ValueError(f"Expected real01 shape (N,1,H,W); got {tuple(real01.shape)}")
    if gen01.ndim != 4 or gen01.shape[1] != 1:
        raise ValueError(f"Expected gen01 shape (M,1,H,W); got {tuple(gen01.shape)}")
    if real01.shape[2:] != gen01.shape[2:]:
        raise ValueError(f"Spatial dims mismatch: real={tuple(real01.shape)} gen={tuple(gen01.shape)}")

    device = real01.device
    H, W = int(real01.shape[2]), int(real01.shape[3])
    bins = make_radial_bins(H, W, nbins, device=device)

    if centering == "zeropad":
        center_fn = center_by_energy_centroid_zeropad
    elif centering == "roll":
        center_fn = center_by_energy_centroid_roll
    else:
        raise ValueError("centering must be 'zeropad' or 'roll'")

    real_c = center_fn(real01.clamp(0.0, 1.0))
    gen_c = center_fn(gen01.clamp(0.0, 1.0))

    real_pix = _to_numpy_1d(real_c)
    thr = float(np.percentile(real_pix, thr_percentile))

    def frac_above(x_np: np.ndarray, t: float) -> float:
        return float(np.mean(x_np > t))

    gen_pix = _to_numpy_1d(gen_c)

    # Scalars / 1D observables per image
    E_real = _to_numpy_1d(real_c.sum(dim=(2, 3)))
    E_gen = _to_numpy_1d(gen_c.sum(dim=(2, 3)))

    active_real = _to_numpy_1d((real_c > thr).float().mean(dim=(2, 3)))
    active_gen = _to_numpy_1d((gen_c > thr).float().mean(dim=(2, 3)))

    # Radial profiles
    prof_real = radial_profile_batch(real_c, bins)  # (N,B)
    prof_gen = radial_profile_batch(gen_c, bins)  # (M,B)
    mean_prof_real = prof_real.mean(dim=0).detach().cpu().numpy()
    mean_prof_gen = prof_gen.mean(dim=0).detach().cpu().numpy()

    eps = 1e-8
    radial_l2 = float(np.sqrt(np.mean((mean_prof_gen - mean_prof_real) ** 2)))
    radial_l2_log = float(
        np.sqrt(np.mean((np.log10(mean_prof_gen + eps) - np.log10(mean_prof_real + eps)) ** 2))
    )

    metrics = {
        "thr_pctl": float(thr_percentile),
        "thr": thr,
        "frac_pix_above_thr_real": frac_above(real_pix, thr),
        "frac_pix_above_thr_gen": frac_above(gen_pix, thr),
        "E_w1": _wasserstein_1d(E_real, E_gen),
        "active_frac_w1": _wasserstein_1d(active_real, active_gen),
        "radial_l2": radial_l2,
        "radial_l2_log": radial_l2_log,
    }
    return metrics
