"""Shared model specification for AudioPrune experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple

ModelName = Literal["af3", "qwen2_audio", "salmonn", "kimi_audio"]

# Paper AudioPrune variants → per-backend method / CLI flags.
Variant = Literal["k4", "ms24", "ms248"]


@dataclass(frozen=True)
class ModelSpec:
    """Describe how to load / invoke a backbone for AudioPrune."""

    name: ModelName
    display_name: str
    # HuggingFace id and/or local snapshot path (resolved at runtime).
    model_id: str
    # Repo-root Python entry (relative to compression0/).
    runner: str
    # Extra local asset paths (SALMONN ckpt / whisper / …).
    local_paths: Dict[str, str] = field(default_factory=dict)
    # Whether the AF3/Qwen-style HF loader needs eager attention.
    eager_attn: bool = True
    # Default max_new_tokens when the dataset does not override.
    max_new_tokens_default: int = 16
    aliases: Tuple[str, ...] = ()

    def matches(self, key: str) -> bool:
        k = key.strip().lower().replace("-", "_").replace(" ", "_")
        cand = {
            self.name,
            self.display_name.lower().replace("-", "_").replace(" ", "_"),
            self.model_id.lower(),
            *self.aliases,
        }
        # also accept compact forms
        cand |= {a.replace("_", "") for a in cand}
        return k in cand or k.replace("_", "") in cand


def variant_scales(variant: Variant) -> Tuple[int, ...] | None:
    if variant == "k4":
        return None
    if variant == "ms24":
        return (2, 4)
    if variant == "ms248":
        return (2, 4, 8)
    raise ValueError(f"Unknown variant: {variant}")


def variant_kernel_size(variant: Variant) -> int:
    return 4 if variant == "k4" else 4
