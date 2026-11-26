"""Dataset registry for AudioPrune experiments (model-agnostic names)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Canonical dataset keys used across launchers / configs.
DatasetName = Literal[
    "mmau_test_mini",
    "mmau_test",
    "clotho",
    "clotho_aqa",
    "cremad",
    "meld",
    "librispeech",
]

MetricKind = Literal["mmau", "cider", "accuracy", "wer", "aqa"]


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    display_name: str
    default_root: str
    # Which CLI flag carries the primary data root for multi-dataset runners.
    root_flag: str
    metric: MetricKind
    expected_rows: int
    # Name accepted by qwen2_audio_attention_eval --dataset (if different).
    qwen_name: Optional[str] = None
    # Default max_new_tokens override (None → model default).
    max_new_tokens: Optional[int] = None
    aliases: Tuple[str, ...] = ()
    extra_defaults: Dict[str, str] = field(default_factory=dict)

    def matches(self, key: str) -> bool:
        k = key.strip().lower().replace("-", "_").replace(" ", "_")
        cand = {self.name, self.display_name.lower().replace("-", "_").replace(" ", "_"), *self.aliases}
        cand |= {c.replace("_", "") for c in cand}
        return k in cand or k.replace("_", "") in cand


REPO_REL = Path("data")

DATASETS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="mmau_test_mini",
        display_name="MMAU-test-mini",
        default_root="data/mmau_test_mini",
        root_flag="mmau_root",
        metric="mmau",
        expected_rows=1000,
        qwen_name="mmau",
        max_new_tokens=16,
        aliases=("mmau", "mmau_mini", "mmau-test-mini"),
    ),
    DatasetSpec(
        name="mmau_test",
        display_name="MMAU-test",
        default_root="data/mmau_test",
        root_flag="mmau_root",
        metric="mmau",
        expected_rows=10000,
        qwen_name="mmau",
        max_new_tokens=16,
        aliases=("mmau_full",),
    ),
    DatasetSpec(
        name="clotho",
        display_name="Clotho-v2",
        default_root="data/clotho_v2/evaluation",
        root_flag="audio_root",
        metric="cider",
        expected_rows=1045,
        qwen_name="clotho",
        max_new_tokens=64,
        aliases=("clotho_v2", "clotho-v2"),
        extra_defaults={
            "prompt": "Provide a caption describing the audio.",
        },
    ),
    DatasetSpec(
        name="clotho_aqa",
        display_name="ClothoAQA",
        default_root="data/clotho_aqa/audio_files",
        root_flag="audio_root",
        metric="aqa",
        expected_rows=2055,
        qwen_name="clotho_aqa",
        max_new_tokens=32,
        aliases=("aqa", "clotho-aqa"),
        extra_defaults={
            "split_jsons": (
                "data/clotho_aqa/splits/clotho_aqa_test_unanimous.json,"
                "data/clotho_aqa/splits/clotho_aqa_test_non_binary.json"
            ),
        },
    ),
    DatasetSpec(
        name="cremad",
        display_name="CREMA-D",
        default_root="data/crema_d/AudioWAV",
        root_flag="ser_audio_root",
        metric="accuracy",
        expected_rows=7442,
        qwen_name=None,  # not in qwen2_audio_attention_eval choices
        max_new_tokens=32,
        aliases=("crema_d", "crema-d", "crema"),
        extra_defaults={
            "ser_manifest": "data/crema_d/manifest.csv",
        },
    ),
    DatasetSpec(
        name="meld",
        display_name="MELD",
        default_root="data/meld/raw/MELD.Raw/output_repeated_splits_test",
        root_flag="meld_audio_root",
        metric="accuracy",
        expected_rows=2610,
        qwen_name="meld",
        max_new_tokens=32,
        aliases=(),
        extra_defaults={
            "meld_labels_csv": "data/meld/labels/test_sent_emo.csv",
        },
    ),
    DatasetSpec(
        name="librispeech",
        display_name="LibriSpeech",
        default_root="data/librispeech",
        root_flag="librispeech_root",
        metric="wer",
        expected_rows=0,  # depends on split
        qwen_name=None,
        max_new_tokens=256,
        aliases=("libri", "asr"),
    ),
)


def list_datasets() -> List[str]:
    return [d.name for d in DATASETS]


def resolve_dataset(name: str) -> DatasetSpec:
    key = name.strip()
    for spec in DATASETS:
        if spec.matches(key):
            return spec
    raise ValueError(
        f"Unknown dataset {name!r}; expected one of {list_datasets()}"
    )


def default_root(name: str, repo_root: Path | str | None = None) -> Path:
    spec = resolve_dataset(name)
    root = Path(spec.default_root)
    if repo_root is not None and not root.is_absolute():
        return (Path(repo_root) / root).resolve()
    return root


def describe_all() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in DATASETS:
        out[d.name] = {
            "display_name": d.display_name,
            "default_root": d.default_root,
            "root_flag": d.root_flag,
            "metric": d.metric,
            "expected_rows": d.expected_rows,
            "qwen_name": d.qwen_name,
            "aliases": list(d.aliases),
        }
    return out


def eval_script_for(metric: MetricKind) -> str:
    """Repo-relative metric script path."""
    return {
        "mmau": "metric/calc_mmau_accuracy.py",
        "cider": "metric/calc_cider.py",
        "accuracy": "metric/calc_meld_accuracy.py",
        "aqa": "metric/calc_clotho_aqa_accuracy.py",
        "wer": "metric/calc_librispeech_wer.py",
    }[metric]


__all__ = [
    "DATASETS",
    "DatasetName",
    "DatasetSpec",
    "MetricKind",
    "default_root",
    "describe_all",
    "eval_script_for",
    "list_datasets",
    "resolve_dataset",
]
