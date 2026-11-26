#!/usr/bin/env python3
"""Smoke-test core AudioPrune math (SSM → novelty → peaks → drop ranks)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
# Prefer importing ``audioprune`` from ``code/`` (avoids clashing with stdlib ``code``).
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from audioprune import audioprune_drop_ranks  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    t, d = 64, 32
    emb = torch.randn(t, d)
    # Make two rough clusters so novelty has structure.
    emb[32:] = emb[32:] + 3.0
    scores = torch.rand(t)
    drops, spans, meta = audioprune_drop_ranks(
        emb,
        scores,
        prune_rate=0.5,
        kernel_size=4,
        length_lambda=0.5,
    )
    print(f"T={t} spans={len(spans)} drop={len(drops)} keep={meta.get('final_keep')} λ={meta.get('length_lambda')}")
    print(f"meta keys: {sorted(meta.keys())[:12]} ...")
    assert meta["final_keep"] == t - len(drops)
    assert meta["final_keep"] >= 1
    print("ok")


if __name__ == "__main__":
    main()
