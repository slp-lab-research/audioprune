"""Kimi-Audio-7B-Instruct adapter for AudioPrune."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from datasets import resolve_dataset
from .base import ModelSpec, Variant
from .cmd_utils import (
    append_common_gen_args,
    append_dataset_path_args,
    append_variant_kernel_args,
    extend,
    resolve_data_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SNAPSHOT = REPO_ROOT / "models" / "kimi_audio_7b_instruct"
HF_ID = "moonshotai/Kimi-Audio-7B-Instruct"

SPEC = ModelSpec(
    name="kimi_audio",
    display_name="Kimi-Audio-7B-Instruct",
    model_id=str(LOCAL_SNAPSHOT if LOCAL_SNAPSHOT.is_dir() else HF_ID),
    runner="kimi_attention_eval.py",
    local_paths={
        "local_snapshot": str(LOCAL_SNAPSHOT),
        "hf_id": HF_ID,
        "kimia_root": str(REPO_ROOT / "third_party" / "Kimi-Audio"),
    },
    eager_attn=False,
    max_new_tokens_default=32,
    aliases=(
        "kimi",
        "kimiaudio",
        "kimi-audio",
        "kimi_audio_7b",
        "kimi_audio_7b_instruct",
        "moonshotai/kimi-audio-7b-instruct",
    ),
)

DEFAULT_MODEL_ID = SPEC.model_id

_KIMI_DATASETS = frozenset(
    {"mmau_test_mini", "clotho", "clotho_aqa", "meld", "cremad"}
)


def resolve_model_path(model_id: str | None = None) -> str:
    if model_id:
        p = Path(model_id)
        if p.is_dir():
            return str(p.resolve())
        return model_id
    if LOCAL_SNAPSHOT.is_dir() and (LOCAL_SNAPSHOT / "config.json").is_file():
        return str(LOCAL_SNAPSHOT.resolve())
    return HF_ID


def _method_for_variant(variant: Variant) -> str:
    if variant == "k4":
        return "checkboard"
    return "checkboard_multiscale"


def build_audioprune_cmd(
    *,
    dataset: str = "mmau_test_mini",
    output_csv: Path | str,
    data_root: Path | str | None = None,
    variant: Variant = "k4",
    prune_rate: float = 0.5,
    length_lambda: float = 0.5,
    kernel_kind: str = "uniform",
    gaussian_var: float = 0.5,
    device: str = "cuda:0",
    max_new_tokens: int | None = None,
    batch_size: int = 1,
    seed: int = 42,
    max_samples: int | None = None,
    model_id: str | None = None,
    python: str = "python",
    extra_args: Sequence[str] = (),
    mmau_root: Path | str | None = None,
) -> List[str]:
    del length_lambda, kernel_kind, gaussian_var
    if mmau_root is not None and data_root is None:
        data_root = mmau_root
    dspec = resolve_dataset(dataset)
    if dspec.name not in _KIMI_DATASETS:
        raise ValueError(
            f"Kimi-Audio supports {sorted(_KIMI_DATASETS)}; got {dspec.name!r}"
        )

    runner = REPO_ROOT / SPEC.runner
    root = resolve_data_root(dspec, data_root, REPO_ROOT)
    tok = (
        max_new_tokens
        if max_new_tokens is not None
        else (dspec.max_new_tokens or SPEC.max_new_tokens_default)
    )
    cmd: List[str] = [
        python,
        str(runner),
        "--dataset",
        dspec.name,
        "--method",
        _method_for_variant(variant),
        "--output-csv",
        str(output_csv),
        "--model-id",
        resolve_model_path(model_id),
    ]
    append_dataset_path_args(cmd, dspec, root, repo_root=REPO_ROOT)
    append_common_gen_args(
        cmd,
        prune_rate=prune_rate,
        max_new_tokens=tok,
        batch_size=batch_size,
        seed=seed,
        device=device,
        max_samples=max_samples,
    )
    append_variant_kernel_args(cmd, variant)
    return extend(cmd, extra_args)


build_mmau_audioprune_cmd = build_audioprune_cmd


def describe() -> Dict[str, Any]:
    return {
        "name": SPEC.name,
        "display_name": SPEC.display_name,
        "model_id": resolve_model_path(),
        "runner": SPEC.runner,
        "local_paths": dict(SPEC.local_paths),
        "pythonpath_extra": SPEC.local_paths.get("kimia_root"),
        "datasets": sorted(_KIMI_DATASETS),
        "audioprune": {
            "k4": "checkboard kernel 4",
            "ms24": "checkboard_multiscale L=2,4",
            "ms248": "checkboard_multiscale L=2,4,8",
        },
    }
