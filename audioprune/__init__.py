"""AudioPrune: event-aware audio token pruning via checkerboard novelty."""

from .allocation import allocate_keep_by_length_attn, token_realloc_scores
from .novelty import (
    DEFAULT_GAUSSIAN_VAR,
    compute_novelty,
    multiscale_novelty,
    resolve_kernel,
    scale_L_to_kernel_size,
)
from .peak_detection import boundaries_to_spans, detect_peaks, segment_embeddings
from .pruning import (
    audioprune_drop_ranks,
    budget_realloc_drop_ranks,
    compute_segment_spans,
    kept_ranks_from_drop,
)
from .ssm import compute_ssm

__all__ = [
    "DEFAULT_GAUSSIAN_VAR",
    "allocate_keep_by_length_attn",
    "audioprune_drop_ranks",
    "boundaries_to_spans",
    "budget_realloc_drop_ranks",
    "compute_novelty",
    "compute_segment_spans",
    "compute_ssm",
    "detect_peaks",
    "kept_ranks_from_drop",
    "multiscale_novelty",
    "resolve_kernel",
    "scale_L_to_kernel_size",
    "segment_embeddings",
    "token_realloc_scores",
]
