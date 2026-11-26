"""Baseline token-pruning methods compared in the paper main table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    repo_entry: str
    notes: str = ""


# Names match ``paper_results/main_result_new.tex`` Method column.
BASELINES: Tuple[BaselineSpec, ...] = (
    BaselineSpec("FastV", "flamingo3_attention_hf.py / eval_mmau_fastv_pruning.sh"),
    BaselineSpec("VisionZip", "flamingo3_attention_visionzip.py"),
    BaselineSpec("DivPrune", "flamingo3_attention_divprune.py / divprune/"),
    BaselineSpec("SpeechPrune", "flamingo3_attention_speechprune.py"),
    BaselineSpec("SegTop-K", "flamingo3_attention_*_segment_topk_prune.py"),
    BaselineSpec("DASH", "flamingo3_attention_dash_prune.py / dash_audio_prune_core.py"),
    BaselineSpec("Zoo-Prune", "flamingo3_attention_*_zooprune.py"),
    BaselineSpec("AgilePruner", "agilepruner/"),
)


def list_baselines() -> Tuple[str, ...]:
    return tuple(b.name for b in BASELINES)
