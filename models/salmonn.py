"""SALMONN-7B adapter for AudioPrune."""
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
SALMONN_ROOT = REPO_ROOT / "models" / "salmonn7b"

_DEFAULT_PATHS = {
    "ckpt": str(SALMONN_ROOT / "ckpt" / "salmonn_7b_v0.pth"),
    "whisper_path": str(SALMONN_ROOT / "whisper-large-v2"),
    "beats_path": str(
        SALMONN_ROOT
        / "beats"
        / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    ),
    "vicuna_path": str(SALMONN_ROOT / "vicuna-7b-v1.5"),
}

SPEC = ModelSpec(
    name="salmonn",
    display_name="SALMONN-7B",
    model_id=str(SALMONN_ROOT),
    runner="salmonn_attention_eval.py",
    local_paths=dict(_DEFAULT_PATHS),
    eager_attn=True,
    max_new_tokens_default=16,
    aliases=("salmonn7b", "salmonn_7b", "salmonn-7b"),
)

_SALMONN_DATASETS = frozenset(
    {"mmau_test_mini", "mmau_test", "clotho", "clotho_aqa", "cremad"}
)


def default_paths() -> Dict[str, str]:
    return dict(SPEC.local_paths)


def _method_for_variant(variant: Variant) -> str:
    if variant == "k4":
        return "checkboard_attdiff"
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
    paths: Dict[str, str] | None = None,
    extra_args: Sequence[str] = (),
    mmau_root: Path | str | None = None,
) -> List[str]:
    del model_id, length_lambda, kernel_kind, gaussian_var
    if mmau_root is not None and data_root is None:
        data_root = mmau_root
    dspec = resolve_dataset(dataset)
    if dspec.name not in _SALMONN_DATASETS:
        raise ValueError(
            f"SALMONN supports {sorted(_SALMONN_DATASETS)}; got {dspec.name!r}"
        )

    p = default_paths() if paths is None else {**default_paths(), **paths}
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
        "--ckpt",
        p["ckpt"],
        "--whisper-path",
        p["whisper_path"],
        "--beats-path",
        p["beats_path"],
        "--vicuna-path",
        p["vicuna_path"],
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
        "local_paths": default_paths(),
        "datasets": sorted(_SALMONN_DATASETS),
        "audioprune": {
            "k4": "checkboard_attdiff (kernel 4)",
            "ms24": "checkboard_multiscale L=2,4",
            "ms248": "checkboard_multiscale L=2,4,8",
        },
    }
