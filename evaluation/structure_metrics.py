"""Structure / event-alignment metrics for AudioPrune segmentations."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


Span = Tuple[int, int]


def span_length(span: Span) -> int:
    s, e = int(span[0]), int(span[1])
    return max(0, e - s + 1)


def span_intersection(a: Span, b: Span) -> int:
    lo = max(int(a[0]), int(b[0]))
    hi = min(int(a[1]), int(b[1]))
    return max(0, hi - lo + 1)


def span_iou(a: Span, b: Span) -> float:
    inter = span_intersection(a, b)
    if inter <= 0:
        return 0.0
    union = span_length(a) + span_length(b) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def coverage_of_gt(
    pred_kept_ranks: Sequence[int],
    gt_spans: Sequence[Span],
) -> Dict[str, float]:
    """
    Fraction of GT event tokens that intersect at least one kept rank,
    plus mean per-event coverage.
    """
    kept = set(int(i) for i in pred_kept_ranks)
    if not gt_spans:
        return {"token_coverage": 0.0, "mean_event_coverage": 0.0, "num_events": 0.0}

    covered_tokens = 0
    total_tokens = 0
    per_event: List[float] = []
    for span in gt_spans:
        L = span_length(span)
        total_tokens += L
        hit = sum(1 for i in range(int(span[0]), int(span[1]) + 1) if i in kept)
        covered_tokens += hit
        per_event.append(float(hit) / float(L) if L > 0 else 0.0)

    return {
        "token_coverage": float(covered_tokens) / float(total_tokens) if total_tokens else 0.0,
        "mean_event_coverage": float(sum(per_event) / len(per_event)),
        "num_events": float(len(gt_spans)),
    }


def boundary_f1(
    pred_boundaries: Sequence[int],
    gt_boundaries: Sequence[int],
    *,
    tolerance: int = 1,
) -> Dict[str, float]:
    """
    Boundary detection F1 with ±``tolerance`` token matching (greedy 1-1).
    """
    pred = sorted(set(int(b) for b in pred_boundaries))
    gt = sorted(set(int(b) for b in gt_boundaries))
    if not pred and not gt:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not gt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gt: set[int] = set()
    tp = 0
    for p in pred:
        for g in gt:
            if g in matched_gt:
                continue
            if abs(p - g) <= int(tolerance):
                matched_gt.add(g)
                tp += 1
                break
    precision = tp / len(pred)
    recall = tp / len(gt)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
