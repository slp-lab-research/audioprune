"""Foote checkerboard kernels and diagonal novelty on the SSM."""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F

KernelKind = Literal["uniform", "gaussian", "gaussian_centered"]
CorrMode = Literal["constant", "valid"]

DEFAULT_GAUSSIAN_VAR = 0.5


def _even_kernel_size(size: int) -> int:
    if size < 2:
        raise ValueError("checkerboard kernel size must be >= 2")
    if size % 2 != 0:
        size += 1
    return size


def checkerboard_kernel(size: int) -> np.ndarray:
    """Uniform Foote checkerboard (even size): +1 / -1 quadrants, L1-normalized."""
    size = _even_kernel_size(size)
    half = size // 2
    kernel = np.ones((size, size), dtype=np.float64)
    kernel[:half, half:] = -1.0
    kernel[half:, :half] = -1.0
    kernel /= float(np.sum(np.abs(kernel)))
    return kernel.astype(np.float32)


def gaussian_checkerboard_kernel(size: int, *, var: float = DEFAULT_GAUSSIAN_VAR) -> np.ndarray:
    """Foote checkerboard × FMP Gaussian envelope (even size)."""
    size = _even_kernel_size(size)
    half = size // 2
    if float(var) <= 0.0:
        raise ValueError("gaussian var must be > 0")
    taper = float(np.sqrt(0.5) / (float(half) * float(var)))
    yy, xx = np.mgrid[-half:half, -half:half]
    gauss = np.exp(-(taper**2) * (xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2))
    kernel = np.ones((size, size), dtype=np.float64)
    kernel[:half, half:] = -1.0
    kernel[half:, :half] = -1.0
    kernel *= gauss
    denom = float(np.sum(np.abs(kernel)))
    if denom <= 0.0:
        raise RuntimeError("gaussian checkerboard kernel has zero L1 mass")
    kernel /= denom
    return kernel.astype(np.float32)


def gaussian_centered_checkerboard_kernel(
    size: int, *, var: float = DEFAULT_GAUSSIAN_VAR
) -> np.ndarray:
    """Odd FMP-style Gaussian checkerboard over coordinates ``[-L, L]``."""
    size = int(size)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    L = size // 2
    if float(var) <= 0.0:
        raise ValueError("gaussian var must be > 0")
    taper = float(np.sqrt(0.5) / (float(L) * float(var)))
    axis = np.arange(-L, L + 1, dtype=np.float64)
    gaussian_1d = np.exp(-(taper**2) * (axis**2))
    kernel = np.outer(np.sign(axis), np.sign(axis)) * np.outer(gaussian_1d, gaussian_1d)
    denom = float(np.sum(np.abs(kernel)))
    if denom <= 0.0:
        raise RuntimeError("centered gaussian checkerboard kernel has zero L1 mass")
    kernel /= denom
    return kernel.astype(np.float32)


def resolve_kernel(
    size: int,
    *,
    kernel_kind: KernelKind = "uniform",
    gaussian_var: float = DEFAULT_GAUSSIAN_VAR,
) -> np.ndarray:
    if kernel_kind == "uniform":
        return checkerboard_kernel(size)
    if kernel_kind == "gaussian":
        return gaussian_checkerboard_kernel(size, var=gaussian_var)
    if kernel_kind == "gaussian_centered":
        return gaussian_centered_checkerboard_kernel(size, var=gaussian_var)
    raise ValueError(f"Unknown kernel_kind: {kernel_kind!r}")


def effective_kernel_size(
    n_tokens: int,
    kernel_size: int,
    kernel_kind: KernelKind = "uniform",
) -> int:
    minimum = 3 if kernel_kind == "gaussian_centered" else 2
    eff = min(int(kernel_size), max(int(n_tokens) - 1, minimum))
    if kernel_kind == "gaussian_centered":
        if eff % 2 == 0:
            eff -= 1
    elif eff % 2 != 0:
        eff -= 1
    return max(eff, minimum)


def scale_L_to_kernel_size(half_width_L: int, kernel_kind: KernelKind = "uniform") -> int:
    """Map Foote half-width ``L`` to full kernel size (``2L`` or ``2L+1``)."""
    L = int(half_width_L)
    if L < 1:
        raise ValueError(f"scale L must be >= 1, got {half_width_L}")
    return 2 * L + 1 if kernel_kind == "gaussian_centered" else 2 * L


def compute_novelty_diag(
    ssm: torch.Tensor,
    kernel_size: int,
    *,
    kernel_kind: KernelKind = "uniform",
    gaussian_var: float = DEFAULT_GAUSSIAN_VAR,
    corr_mode: CorrMode = "constant",
    apply_relu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Diagonal-only Foote novelty: ``n(t) = <SSM[t-·, t-·], K>``.

    Avoids full N×N convolution. ``corr_mode='constant'`` zero-pads edges;
    ``'valid'`` only fills interior positions with a full K×K window.
    """
    if ssm.ndim != 2 or int(ssm.shape[0]) != int(ssm.shape[1]):
        raise ValueError(f"ssm must be square [N,N], got {tuple(ssm.shape)}")
    if corr_mode not in ("constant", "valid"):
        raise ValueError(f"corr_mode must be 'constant' or 'valid', got {corr_mode!r}")

    n = int(ssm.shape[0])
    k_np = resolve_kernel(
        max(int(kernel_size), 2), kernel_kind=kernel_kind, gaussian_var=gaussian_var
    )
    if n == 0:
        k0 = torch.as_tensor(k_np, device=ssm.device, dtype=torch.float32)
        return ssm.new_zeros((0,), dtype=torch.float32), k0

    k_size = effective_kernel_size(n, kernel_size, kernel_kind)
    ssm_f = ssm.float()
    kernel = torch.as_tensor(
        resolve_kernel(k_size, kernel_kind=kernel_kind, gaussian_var=gaussian_var),
        device=ssm_f.device,
        dtype=ssm_f.dtype,
    )
    pad_l = k_size // 2
    pad_r = k_size - 1 - pad_l
    offs = torch.arange(k_size, device=ssm_f.device)

    if corr_mode == "valid":
        n_valid = n - k_size + 1
        novelty = ssm_f.new_zeros((n,))
        if n_valid > 0:
            starts = torch.arange(n_valid, device=ssm_f.device)
            rows = starts[:, None, None] + offs[None, :, None]
            cols = starts[:, None, None] + offs[None, None, :]
            patches = ssm_f[rows, cols]
            novelty[pad_l : pad_l + n_valid] = (patches * kernel).sum(dim=(-1, -2))
    else:
        padded = F.pad(ssm_f, (pad_l, pad_r, pad_l, pad_r), mode="constant", value=0.0)
        t_idx = torch.arange(n, device=ssm_f.device)
        rows = t_idx[:, None, None] + offs[None, :, None]
        cols = t_idx[:, None, None] + offs[None, None, :]
        patches = padded[rows, cols]
        novelty = (patches * kernel).sum(dim=(-1, -2))

    if apply_relu:
        novelty = torch.clamp_min(novelty, 0.0)
    return novelty, kernel


def compute_novelty(
    ssm: np.ndarray | torch.Tensor,
    kernel_size: int,
    *,
    kernel_kind: KernelKind = "uniform",
    gaussian_var: float = DEFAULT_GAUSSIAN_VAR,
    corr_mode: CorrMode = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning NumPy ``(novelty[T], kernel[K,K])``."""
    empty_k = resolve_kernel(
        max(int(kernel_size), 2), kernel_kind=kernel_kind, gaussian_var=gaussian_var
    )
    if isinstance(ssm, np.ndarray):
        if ssm.size == 0:
            return np.zeros(0, dtype=np.float32), empty_k
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ssm_t = torch.from_numpy(np.asarray(ssm, dtype=np.float32)).to(device)
    else:
        if int(ssm.numel()) == 0:
            return np.zeros(0, dtype=np.float32), empty_k
        ssm_t = ssm
    nov_t, k_t = compute_novelty_diag(
        ssm_t,
        kernel_size,
        kernel_kind=kernel_kind,
        gaussian_var=gaussian_var,
        corr_mode=corr_mode,
    )
    return (
        nov_t.detach().float().cpu().numpy().astype(np.float32, copy=False),
        k_t.detach().float().cpu().numpy().astype(np.float32, copy=False),
    )


def mad_normalize(curve: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """``tilde_n = (n - median(n)) / (MAD(n) + eps)``."""
    if curve.size == 0:
        return curve.astype(np.float32, copy=False)
    x = curve.astype(np.float64, copy=False)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return ((x - med) / (mad + float(eps))).astype(np.float32)


def multiscale_novelty(
    ssm: np.ndarray | torch.Tensor,
    scales_L: Tuple[int, ...] | list[int],
    *,
    scale_weights: Tuple[float, ...] | list[float] | None = None,
    kernel_kind: KernelKind = "uniform",
    gaussian_var: float = DEFAULT_GAUSSIAN_VAR,
    corr_mode: CorrMode = "constant",
    mad: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    MAD-normalize per-scale Foote novelty (kernel ``2L``) and integrate.

    ``n_multi(t) = sum_L w_L * tilde_n_L(t)``.
    """
    scales = [int(L) for L in scales_L]
    if not scales:
        raise ValueError("scales_L must be non-empty")
    if scale_weights is None:
        weights = [1.0 / len(scales)] * len(scales)
    else:
        if len(scale_weights) != len(scales):
            raise ValueError("scale_weights length must match scales_L")
        w_sum = float(sum(scale_weights))
        weights = [float(w) / w_sum for w in scale_weights]

    curves: list[np.ndarray] = []
    for L in scales:
        k = scale_L_to_kernel_size(L, kernel_kind)
        raw, _ = compute_novelty(
            ssm, k, kernel_kind=kernel_kind, gaussian_var=gaussian_var, corr_mode=corr_mode
        )
        curves.append(mad_normalize(raw, eps=eps) if mad else raw.astype(np.float32))

    out = np.zeros_like(curves[0], dtype=np.float64)
    for w, c in zip(weights, curves):
        out += float(w) * c.astype(np.float64)
    return out.astype(np.float32)
