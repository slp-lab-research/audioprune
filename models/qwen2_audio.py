"""Qwen2-Audio-7B-Instruct adapter for AudioPrune."""
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

SPEC = ModelSpec(
    name="qwen2_audio",
    display_name="Qwen2-Audio-7B-Instruct",
    model_id="Qwen/Qwen2-Audio-7B-Instruct",
    runner="qwen2_audio_checkboard_budget_realloc_infer.py",
    eager_attn=True,
    max_new_tokens_default=16,
    aliases=(
        "qwen2",
        "qwen2audio",
        "qwen2_audio_instruct",
        "qwen2-audio",
        "qwen2-audio-instruct",
        "qwen2_audio_7b_instruct",
    ),
)

DEFAULT_MODEL_ID = SPEC.model_id

# Datasets accepted by qwen2_audio_attention_eval.
_QWEN_DATASETS = frozenset({"mmau_test_mini", "mmau_test", "clotho", "clotho_aqa", "meld"})


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
    if mmau_root is not None and data_root is None:
        data_root = mmau_root
    dspec = resolve_dataset(dataset)
    if dspec.name not in _QWEN_DATASETS or dspec.qwen_name is None:
        raise ValueError(
            f"Qwen2-Audio AudioPrune supports {sorted(_QWEN_DATASETS)}; got {dspec.name!r}"
        )

    runner = REPO_ROOT / SPEC.runner
    root = resolve_data_root(dspec, data_root, REPO_ROOT)
    budget_mode = "single" if variant == "k4" else "multiscale"
    tok = (
        max_new_tokens
        if max_new_tokens is not None
        else (dspec.max_new_tokens or SPEC.max_new_tokens_default)
    )
    cmd: List[str] = [
        python,
        str(runner),
        "--budget-mode",
        budget_mode,
        "--length-lambda",
        str(length_lambda),
        "--kernel-kind",
        kernel_kind,
        "--gaussian-var",
        str(gaussian_var),
        "--dataset",
        dspec.qwen_name,
        "--method",
        "checkboard" if budget_mode == "single" else "checkboard_multiscale",
        "--output-csv",
        str(output_csv),
        "--model-id",
        str(model_id or SPEC.model_id),
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
        "model_id": SPEC.model_id,
        "runner": SPEC.runner,
        "datasets": sorted(_QWEN_DATASETS),
        "audioprune": "budget_realloc via qwen2_audio_checkboard_budget_realloc_infer.py",
    }
