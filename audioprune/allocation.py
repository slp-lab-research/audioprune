"""Length / attention budget mixing for AudioPrune keep–drop allocation."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch


def minmax_norm(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def mix_proportions(
    lengths: Sequence[int],
    attn_masses: Sequence[float],
    length_lambda: float,
) -> Tuple[List[float], List[float], List[float], float]:
    """
    ``p_mix = λ p_length + (1-λ) p_attn``.

    Falls back to length-only when attention masses are all ≤ 0.
    """
    lam = float(length_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"length_lambda must be in [0, 1], got {length_lambda}")
    m = len(lengths)
    len_sum = float(sum(lengths))
    p_length = [(float(L) / len_sum) if len_sum > 0.0 else 0.0 for L in lengths]

    masses = [max(0.0, float(a)) for a in attn_masses]
    if len(masses) != m:
        raise ValueError("attn_masses length must match lengths.")
    mass_sum = float(sum(masses))
    p_attn = list(p_length) if mass_sum <= 0.0 else [a / mass_sum for a in masses]

    p_mix = [lam * p_length[i] + (1.0 - lam) * p_attn[i] for i in range(m)]
    mix_sum = float(sum(p_mix))
    if mix_sum <= 0.0:
        p_mix = list(p_length)
        mix_sum = float(sum(p_mix))
    return p_length, p_attn, p_mix, mix_sum


def allocate_keep_by_length_attn(
    segment_spans: Sequence[Tuple[int, int]],
    attn_masses: Sequence[float],
    total_keep: int,
    *,
    length_lambda: float,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Integer keep counts via mixed proportions (floor + largest remainder),
    capped by segment length.
    """
    m = len(segment_spans)
    if m == 0:
        return [], {
            "p_length": [],
            "p_attn": [],
            "p_mix": [],
            "length_lambda": float(length_lambda),
        }

    lengths = [max(0, int(end) - int(start) + 1) for start, end in segment_spans]
    total_capacity = sum(lengths)
    k = max(0, min(int(total_keep), total_capacity))
    lam = float(length_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"length_lambda must be in [0, 1], got {length_lambda}")

    if k <= 0:
        zeros = [0] * m
        return zeros, {
            "p_length": [0.0] * m,
            "p_attn": [0.0] * m,
            "p_mix": [0.0] * m,
            "length_lambda": lam,
        }

    p_length, p_attn, p_mix, mix_sum = mix_proportions(lengths, attn_masses, lam)
    raw = [k * (p_mix[i] / mix_sum) for i in range(m)] if mix_sum > 0.0 else [0.0] * m
    keep = [min(int(math.floor(raw[i])), lengths[i]) for i in range(m)]

    rem = k - sum(keep)
    order = sorted(
        range(m),
        key=lambda i: (raw[i] - math.floor(raw[i]), p_mix[i], lengths[i]),
        reverse=True,
    )
    while rem > 0:
        progressed = False
        for i in order:
            if rem <= 0:
                break
            if keep[i] < lengths[i]:
                keep[i] += 1
                rem -= 1
                progressed = True
        if not progressed:
            break

    return keep, {
        "p_length": p_length,
        "p_attn": p_attn,
        "p_mix": p_mix,
        "length_lambda": lam,
        "raw_keep": raw,
    }


def token_realloc_scores(
    scores: torch.Tensor,
    segment_spans: Sequence[Tuple[int, int]],
    *,
    length_lambda: float,
) -> List[float]:
    """
    Per-token restore/trim priority for AudioPrune phase-2.

    Higher is preferred to keep / restore::

        score = (1 - λ) * attn_norm + λ * length_norm(event)
    """
    n = int(scores.numel())
    lam = float(length_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"length_lambda must be in [0, 1], got {length_lambda}")

    attn_list = [float(x) for x in scores[:n].detach().float().cpu().tolist()]
    attn_n = minmax_norm(attn_list)

    lengths = [max(0, int(e) - int(s) + 1) for s, e in segment_spans]
    length_n = minmax_norm([math.log1p(float(L)) for L in lengths])
    seg_of = [0] * n
    for si, (start, end) in enumerate(segment_spans):
        for i in range(int(start), int(end) + 1):
            if 0 <= i < n:
                seg_of[i] = si

    out: List[float] = []
    for i in range(n):
        ln = length_n[seg_of[i]] if segment_spans else 0.0
        out.append((1.0 - lam) * attn_n[i] + lam * ln)
    return out
