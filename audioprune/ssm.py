"""Cosine self-similarity matrix (SSM) over audio-token embeddings."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_ssm(
    embeddings: torch.Tensor,
    *,
    as_numpy: bool = False,
) -> np.ndarray | torch.Tensor:
    """
    Pairwise cosine similarity matrix ``S ∈ R^{T×T}``.

    Args:
        embeddings: ``[T, D]`` token embeddings.
        as_numpy: If True, return a CPU float32 NumPy array.

    Returns:
        Square similarity matrix on the embedding device (or NumPy).
    """
    n = int(embeddings.shape[0])
    if n == 0:
        empty = embeddings.new_zeros((0, 0), dtype=torch.float32)
        return empty.detach().cpu().numpy() if as_numpy else empty

    normed = F.normalize(embeddings.float(), dim=-1)
    sim = normed @ normed.T
    if as_numpy:
        return sim.detach().cpu().numpy().astype(np.float32)
    return sim


def compute_self_similarity_matrix(
    embeddings: torch.Tensor,
    *,
    as_numpy: bool = True,
) -> np.ndarray | torch.Tensor:
    """Alias matching the repo helper name."""
    return compute_ssm(embeddings, as_numpy=as_numpy)
