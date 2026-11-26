"""AudioFlamingo3 (AF3) adapter for AudioPrune."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from datasets import resolve_dataset
from .base import ModelSpec, Variant
from .cmd_utils import (
    append_common_gen_args,
    append_variant_kernel_args,
    extend,
    resolve_data_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SPEC = ModelSpec(
    name="af3",
    display_name="AudioFlamingo3",
    model_id="nvidia/audio-flamingo-3-hf",
    runner="flamingo3_audioprune_eval.py",
    eager_attn=True,
    max_new_tokens_default=16,
    aliases=(
        "audioflamingo3",
        "audio_flamingo3",
        "audio_flamingo_3",
        "flamingo3",
        "af-3",
        "af3",
    ),
)

DEFAULT_MODEL_ID = SPEC.model_id
# Back-compat alias.
DEFAULT_MAX_NEW_TOKENS_MMAU = SPEC.max_new_tokens_default

_AF3_RUNNERS = {
    "mmau_test_mini": "flamingo3_audioprune_eval.py",
    "mmau_test": "flamingo3_audioprune_eval.py",
    "librispeech": "flamingo3_librispeech_audioprune_asr_eval.py",
}


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
    # legacy kw
    mmau_root: Path | str | None = None,
) -> List[str]:
    """Build AudioPrune CLI for AF3 on the requested dataset."""
    if mmau_root is not None and data_root is None:
        data_root = mmau_root
    dspec = resolve_dataset(dataset)
    if dspec.name not in _AF3_RUNNERS:
        raise ValueError(
            f"AF3 AudioPrune runner not wired for dataset={dspec.name!r}. "
            f"Supported: {sorted(_AF3_RUNNERS)}"
        )
    runner = REPO_ROOT / _AF3_RUNNERS[dspec.name]
    root = resolve_data_root(dspec, data_root, REPO_ROOT)
    tok = (
        max_new_tokens
        if max_new_tokens is not None
        else (dspec.max_new_tokens or SPEC.max_new_tokens_default)
    )

    if dspec.name == "librispeech":
        cmd: List[str] = [
            python,
            str(runner),
            "--output-csv",
            str(output_csv),
            "--librispeech-root",
            str(root),
            "--variant",
            variant,
            "--model-id",
            str(model_id or SPEC.model_id),
            "--length-lambda",
            str(length_lambda),
            "--split",
            "test_clean",
        ]
        append_common_gen_args(
            cmd,
            prune_rate=prune_rate,
            max_new_tokens=tok,
            batch_size=batch_size,
            seed=seed,
            device=device,
            max_samples=max_samples,
        )
        return extend(cmd, extra_args)

    cmd = [
        python,
        str(runner),
        "--output-csv",
        str(output_csv),
        "--data-root",
        str(root),
        "--model-id",
        str(model_id or SPEC.model_id),
        "--variant",
        variant,
        "--length-lambda",
        str(length_lambda),
        "--kernel-kind",
        kernel_kind,
        "--gaussian-var",
        str(gaussian_var),
    ]
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


# Back-compat
build_mmau_audioprune_cmd = build_audioprune_cmd


def describe() -> Dict[str, Any]:
    return {
        "name": SPEC.name,
        "display_name": SPEC.display_name,
        "model_id": SPEC.model_id,
        "datasets": sorted(_AF3_RUNNERS),
        "runners": dict(_AF3_RUNNERS),
        "audioprune": "budget_realloc",
    }
