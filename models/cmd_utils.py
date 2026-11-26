"""CLI helpers shared by model adapters."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from datasets import DatasetSpec, default_root, resolve_dataset
from .base import Variant, variant_kernel_size, variant_scales


def flag_name(root_flag: str) -> str:
    """``mmau_root`` → ``--mmau-root``."""
    return "--" + root_flag.replace("_", "-")


def resolve_data_root(
    dataset: str | DatasetSpec,
    data_root: Path | str | None,
    repo_root: Path,
) -> Path:
    spec = dataset if isinstance(dataset, DatasetSpec) else resolve_dataset(dataset)
    if data_root is not None:
        return Path(data_root).resolve()
    return default_root(spec.name, repo_root)


def append_variant_kernel_args(cmd: List[str], variant: Variant) -> None:
    if variant == "k4":
        cmd.extend(["--kernel-size", str(variant_kernel_size(variant))])
    else:
        scales = variant_scales(variant) or (2, 4)
        cmd.append("--scales-L")
        cmd.extend(str(x) for x in scales)


def append_common_gen_args(
    cmd: List[str],
    *,
    prune_rate: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    device: str,
    max_samples: int | None,
) -> None:
    cmd.extend(
        [
            "--R",
            str(prune_rate),
            "--max-new-tokens",
            str(max_new_tokens),
            "--batch-size",
            str(batch_size),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
    )
    if max_samples is not None:
        cmd.extend(["--max-samples", str(int(max_samples))])


def append_dataset_path_args(
    cmd: List[str],
    spec: DatasetSpec,
    data_root: Path,
    *,
    repo_root: Path,
    include_extras: bool = True,
) -> None:
    cmd.extend([flag_name(spec.root_flag), str(data_root)])
    if not include_extras:
        return
    for key, value in spec.extra_defaults.items():
        if key == "split_jsons":
            # salmonn/kimi take repeated --split-json or a single comma list depending on runner;
            # pass as --split-json for each path when comma-separated.
            for part in str(value).split(","):
                part = part.strip()
                if part:
                    p = Path(part)
                    if not p.is_absolute():
                        p = (repo_root / p).resolve()
                    cmd.extend(["--split-json", str(p)])
            continue
        if key == "prompt":
            cmd.extend(["--prompt", str(value)])
            continue
        path_val = Path(value)
        if not path_val.is_absolute():
            path_val = (repo_root / path_val).resolve()
        cmd.extend([flag_name(key), str(path_val)])


def extend(cmd: List[str], extra_args: Sequence[str]) -> List[str]:
    cmd.extend(list(extra_args))
    return cmd
