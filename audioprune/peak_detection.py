"""Peak picking on novelty curves and conversion to event spans."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks


def detect_peaks(
    novelty: np.ndarray,
    *,
    min_distance: int = 2,
    prominence: float | None = None,
    height: float | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Peak-pick boundaries on a 1-D novelty curve.

    A peak at rank ``i`` marks a boundary between tokens ``i-1`` and ``i``
    (new segment starts at ``i``).
    """
    n = int(novelty.shape[0])
    if n <= 1:
        return [], {"peak_indices": [], "peak_values": [], "threshold": None}

    x = np.asarray(novelty, dtype=np.float64).reshape(-1)
    if prominence is None:
        prominence = max(float(np.std(x)) * 0.5, 1e-4)
    if height is None:
        height = float(np.median(x))

    peaks, _props = find_peaks(
        x,
        distance=max(int(min_distance), 1),
        prominence=float(prominence),
        height=float(height),
    )
    peak_list = [int(p) for p in peaks.tolist()]
    meta = {
        "peak_indices": peak_list,
        "peak_values": [float(x[p]) for p in peak_list],
        "prominence": float(prominence),
        "height": float(height),
        "min_distance": int(min_distance),
    }
    return peak_list, meta


def boundaries_to_spans(
    boundary_ranks: Sequence[int],
    n_tokens: int,
) -> List[Tuple[int, int]]:
    """Inclusive ``(start, end)`` token-rank spans between boundaries."""
    if n_tokens <= 0:
        return []
    starts = [0] + [int(b) for b in boundary_ranks if 0 < int(b) < n_tokens]
    starts = sorted(set(starts))
    spans: List[Tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else (n_tokens - 1)
        spans.append((start, end))
    return spans


def segment_embeddings(
    novelty: np.ndarray,
    *,
    min_distance: int = 2,
    prominence: float | None = None,
    height: float | None = None,
) -> Tuple[List[Tuple[int, int]], List[int], Dict[str, Any]]:
    """Peak-pick ``novelty`` and return ``(spans, boundaries, meta)``."""
    n = int(novelty.shape[0])
    boundaries, meta = detect_peaks(
        novelty,
        min_distance=min_distance,
        prominence=prominence,
        height=height,
    )
    spans = boundaries_to_spans(boundaries, n)
    meta = dict(meta)
    meta["num_segments"] = len(spans)
    return spans, boundaries, meta
