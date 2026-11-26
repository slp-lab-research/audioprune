"""
AudioPrune: checkerboard events + free drop + global keep-budget reallocation.

Pipeline
--------
1. Embeddings → cosine SSM → Foote novelty (single- or multi-scale)
2. Peak-pick → event spans
3. Phase-1: per-event drop ``floor(L_e * R)`` lowest-attention tokens (min_keep=0)
4. Phase-2: restore/trim to ``K = floor(N * (1-R))`` with
   ``score = (1-λ)*attn + λ*length``
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .allocation import token_realloc_scores
from .novelty import (
    DEFAULT_GAUSSIAN_VAR,
    CorrMode,
    KernelKind,
    compute_novelty,
    effective_kernel_size,
    multiscale_novelty,
)
from .peak_detection import segment_embeddings
from .ssm import compute_ssm


def phase1_drop_count(seg_len: int, prune_rate: float) -> int:
    """Allow emptying a segment (including length-1 when ``R > 0``)."""
    if seg_len <= 0 or prune_rate <= 0.0:
        return 0
    if seg_len == 1:
        return 1
    k_drop = int(math.floor(seg_len * prune_rate + 1e-9))
    return min(max(k_drop, 0), seg_len)


def compute_segment_spans(
    embeddings: torch.Tensor,
    *,
    kernel_size: int = 4,
    scales_L: Sequence[int] | None = None,
    scale_weights: Sequence[float] | None = None,
    kernel_kind: KernelKind = "uniform",
    gaussian_var: float = DEFAULT_GAUSSIAN_VAR,
    corr_mode: CorrMode = "constant",
    mad_normalize: bool = True,
    min_boundary_distance: int = 2,
    boundary_prominence: float | None = None,
    boundary_height: float | None = None,
) -> Tuple[List[Tuple[int, int]], List[int], Dict[str, Any]]:
    """Build event spans from encoder embeddings via checkerboard novelty."""
    n = int(embeddings.shape[0])
    if n <= 1:
        return (
            [(0, max(n - 1, 0))],
            [],
            {"num_segments": 1 if n else 0, "kernel_size": kernel_size},
        )

    ssm = compute_ssm(embeddings, as_numpy=False)
    if scales_L is not None:
        novelty = multiscale_novelty(
            ssm,
            tuple(int(L) for L in scales_L),
            scale_weights=list(scale_weights) if scale_weights is not None else None,
            kernel_kind=kernel_kind,
            gaussian_var=gaussian_var,
            corr_mode=corr_mode,
            mad=mad_normalize,
        )
        meta_extra: Dict[str, Any] = {
            "scales_L": list(int(L) for L in scales_L),
            "multiscale": True,
            "kernel_size": 0,
        }
    else:
        eff = effective_kernel_size(n, kernel_size, kernel_kind)
        novelty, _ = compute_novelty(
            ssm,
            eff,
            kernel_kind=kernel_kind,
            gaussian_var=gaussian_var,
            corr_mode=corr_mode,
        )
        meta_extra = {"kernel_size": eff, "multiscale": False}

    spans, boundaries, meta = segment_embeddings(
        novelty,
        min_distance=min_boundary_distance,
        prominence=boundary_prominence,
        height=boundary_height,
    )
    meta = dict(meta)
    meta.update(meta_extra)
    meta["kernel_kind"] = kernel_kind
    meta["corr_mode"] = corr_mode
    return spans, boundaries, meta


def budget_realloc_drop_ranks(
    scores: torch.Tensor,
    segment_spans: Sequence[Tuple[int, int]],
    prune_rate: float,
    *,
    length_lambda: float = 0.5,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Phase-1 free per-segment attention drop + phase-2 global keep match.

    Returns sorted drop ranks and metadata.
    """
    n = int(scores.numel())
    meta: Dict[str, Any] = {
        "selection_mode": "budget_realloc_attention",
        "length_lambda": float(length_lambda),
    }
    if n <= 0 or prune_rate <= 0.0:
        meta.update({"total_keep": n, "phase1_keep": n, "restored": 0, "trimmed": 0})
        return [], meta
    if n == 1:
        meta.update({"total_keep": 1, "phase1_keep": 1, "restored": 0, "trimmed": 0})
        return [], meta

    k_target = int(math.floor(n * (1.0 - prune_rate) + 1e-6))
    k_target = max(1, min(k_target, n))
    meta["total_keep"] = k_target

    drop: set[int] = set()
    phase1_per_seg: List[int] = []
    for start, end in segment_spans:
        start_i, end_i = int(start), int(end)
        if start_i > end_i:
            phase1_per_seg.append(0)
            continue
        seg_len = end_i - start_i + 1
        k_drop = phase1_drop_count(seg_len, prune_rate)
        phase1_per_seg.append(k_drop)
        if k_drop <= 0:
            continue
        if k_drop >= seg_len:
            drop.update(range(start_i, end_i + 1))
            continue
        seg_scores = scores[start_i : end_i + 1]
        worst = torch.topk(seg_scores, k_drop, largest=False).indices
        for j in worst.tolist():
            drop.add(start_i + int(j))

    if len(drop) >= n:
        drop = set(range(n))

    phase1_keep = n - len(drop)
    meta["phase1_keep"] = phase1_keep
    meta["phase1_drop_per_segment"] = phase1_per_seg

    realloc_scores = token_realloc_scores(
        scores, segment_spans, length_lambda=length_lambda
    )
    restored = 0
    trimmed = 0

    if phase1_keep < k_target:
        need = k_target - phase1_keep
        candidates = sorted(
            (i for i in drop),
            key=lambda i: (realloc_scores[i], float(scores[i].item()), -i),
            reverse=True,
        )
        for i in candidates[:need]:
            drop.discard(i)
            restored += 1
    elif phase1_keep > k_target:
        need = phase1_keep - k_target
        kept = [i for i in range(n) if i not in drop]
        kept_sorted = sorted(
            kept,
            key=lambda i: (realloc_scores[i], float(scores[i].item()), -i),
        )
        for i in kept_sorted[:need]:
            drop.add(i)
            trimmed += 1

    if len(drop) >= n:
        keep_idx = max(range(n), key=lambda i: (realloc_scores[i], float(scores[i].item())))
        drop.discard(keep_idx)
        meta["forced_global_min_keep"] = True

    meta.update({"restored": restored, "trimmed": trimmed, "final_keep": n - len(drop)})
    return sorted(drop), meta


def audioprune_drop_ranks(
    embeddings: torch.Tensor,
    attention_scores: torch.Tensor,
    prune_rate: float,
    *,
    kernel_size: int = 4,
    scales_L: Sequence[int] | None = None,
    scale_weights: Sequence[float] | None = None,
    kernel_kind: KernelKind = "uniform",
    gaussian_var: float = DEFAULT_GAUSSIAN_VAR,
    length_lambda: float = 0.5,
    mad_normalize: bool = True,
    min_boundary_distance: int = 2,
    boundary_prominence: float | None = None,
    boundary_height: float | None = None,
) -> Tuple[List[int], List[Tuple[int, int]], Dict[str, Any]]:
    """
    End-to-end AudioPrune on one sample.

    Returns ``(drop_ranks, segment_spans, meta)``.
    """
    if int(embeddings.shape[0]) != int(attention_scores.numel()):
        raise ValueError(
            f"embeddings T={embeddings.shape[0]} != scores N={attention_scores.numel()}"
        )
    spans, _boundaries, seg_meta = compute_segment_spans(
        embeddings,
        kernel_size=kernel_size,
        scales_L=scales_L,
        scale_weights=scale_weights,
        kernel_kind=kernel_kind,
        gaussian_var=gaussian_var,
        mad_normalize=mad_normalize,
        min_boundary_distance=min_boundary_distance,
        boundary_prominence=boundary_prominence,
        boundary_height=boundary_height,
    )
    drops, drop_meta = budget_realloc_drop_ranks(
        attention_scores,
        spans,
        prune_rate,
        length_lambda=length_lambda,
    )
    meta = dict(seg_meta)
    meta.update(drop_meta)
    return drops, spans, meta


def kept_ranks_from_drop(n_tokens: int, drop_ranks: Sequence[int]) -> List[int]:
    drop = set(int(i) for i in drop_ranks)
    return [i for i in range(int(n_tokens)) if i not in drop]
