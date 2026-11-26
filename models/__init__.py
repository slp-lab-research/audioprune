"""
Model registry for AudioPrune experiments.

Supported backbones:
  - af3           AudioFlamingo3
  - qwen2_audio   Qwen2-Audio-7B-Instruct
  - salmonn       SALMONN-7B
  - kimi_audio    Kimi-Audio-7B-Instruct
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from . import af3, kimi_audio, qwen2_audio, salmonn
from .base import ModelName, ModelSpec, Variant, variant_scales

AF3 = af3.SPEC
QWEN2_AUDIO = qwen2_audio.SPEC
SALMONN = salmonn.SPEC
KIMI_AUDIO = kimi_audio.SPEC

PAPER_MODELS: tuple[ModelSpec, ...] = (AF3, QWEN2_AUDIO, SALMONN, KIMI_AUDIO)
MODEL_MODULES = {
    "af3": af3,
    "qwen2_audio": qwen2_audio,
    "salmonn": salmonn,
    "kimi_audio": kimi_audio,
}


def list_models() -> List[str]:
    return [m.name for m in PAPER_MODELS]


def resolve_model_spec(name: str) -> ModelSpec:
    key = name.strip()
    for spec in PAPER_MODELS:
        if spec.matches(key):
            return spec
    raise ValueError(
        f"Unknown model {name!r}; expected one of {list_models()} "
        f"(aliases: af3, qwen2_audio, salmonn, kimi_audio)."
    )


def describe_all() -> Dict[str, Any]:
    return {name: MODEL_MODULES[name].describe() for name in list_models()}


def build_audioprune_cmd(
    model: str,
    *,
    dataset: str = "mmau_test_mini",
    output_csv: str,
    data_root: str | None = None,
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
    # legacy
    mmau_root: str | None = None,
) -> List[str]:
    """Dispatch AudioPrune CLI construction for ``model`` × ``dataset``."""
    if mmau_root is not None and data_root is None:
        data_root = mmau_root
    spec = resolve_model_spec(model)
    mod = MODEL_MODULES[spec.name]
    return mod.build_audioprune_cmd(
        dataset=dataset,
        output_csv=output_csv,
        data_root=data_root,
        variant=variant,
        prune_rate=prune_rate,
        length_lambda=length_lambda,
        kernel_kind=kernel_kind,
        gaussian_var=gaussian_var,
        device=device,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        seed=seed,
        max_samples=max_samples,
        model_id=model_id,
        python=python,
        extra_args=extra_args,
    )


# Back-compat alias
def build_mmau_audioprune_cmd(*args: Any, **kwargs: Any) -> List[str]:
    if args:
        # old signature: (model, output_csv=..., mmau_root=...)
        kwargs.setdefault("dataset", "mmau_test_mini")
        return build_audioprune_cmd(args[0], **kwargs)
    return build_audioprune_cmd(**kwargs)


__all__ = [
    "AF3",
    "QWEN2_AUDIO",
    "SALMONN",
    "KIMI_AUDIO",
    "PAPER_MODELS",
    "ModelName",
    "ModelSpec",
    "Variant",
    "build_audioprune_cmd",
    "build_mmau_audioprune_cmd",
    "describe_all",
    "list_models",
    "resolve_model_spec",
    "variant_scales",
    "af3",
    "qwen2_audio",
    "salmonn",
    "kimi_audio",
]
