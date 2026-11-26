#!/usr/bin/env python3
"""Multi-model / multi-dataset AudioPrune launcher.

Models  : af3 | qwen2_audio | salmonn | kimi_audio
Datasets: mmau_test_mini | mmau_test | clotho | clotho_aqa | cremad | meld | librispeech
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from datasets import describe_all as describe_datasets  # noqa: E402
from datasets import list_datasets, resolve_dataset  # noqa: E402
from models import (  # noqa: E402
    build_audioprune_cmd,
    describe_all as describe_models,
    list_models,
    resolve_model_spec,
)

DEFAULT_CONFIG = CODE_ROOT / "configs" / "default.yaml"


def load_config(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML is required: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser(
        description="AudioPrune launcher (multi-model × multi-dataset)."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", default=None, help=f"One of {list_models()}")
    p.add_argument("--dataset", default=None, help=f"One of {list_datasets()}")
    p.add_argument("--data-root", type=Path, default=None, help="Override dataset root.")
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--variant", default=None, choices=("k4", "ms24", "ms248"))
    p.add_argument("--R", type=float, default=None)
    p.add_argument("--length-lambda", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--model-id", default=None)
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--list-datasets", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    # legacy alias
    p.add_argument("--mmau-root", type=Path, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.list_models:
        print(json.dumps(describe_models(), indent=2))
        return
    if args.list_datasets:
        print(json.dumps(describe_datasets(), indent=2))
        return

    if args.output_csv is None:
        p.error("--output-csv is required unless --list-models/--list-datasets")

    cfg = load_config(args.config.resolve()) if args.config.is_file() else {}
    model_key = args.model or cfg.get("model") or "af3"
    dataset_key = args.dataset or cfg.get("dataset") or "mmau_test_mini"
    spec = resolve_model_spec(str(model_key))
    dspec = resolve_dataset(str(dataset_key))
    variant = args.variant or cfg.get("variant") or "k4"
    r = float(args.R if args.R is not None else cfg.get("R", 0.5))
    lam = float(
        args.length_lambda
        if args.length_lambda is not None
        else cfg.get("length_lambda", 0.5)
    )
    data_root = args.data_root or args.mmau_root or cfg.get("data_root") or cfg.get("mmau_root")
    device = args.device or cfg.get("device", "cuda:0")

    env = os.environ.copy()
    if spec.name == "kimi_audio":
        kimia = str(REPO_ROOT / "third_party" / "Kimi-Audio")
        env["PYTHONPATH"] = kimia + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Only inherit config model_id when it belongs to the selected model.
    model_id = args.model_id
    if model_id is None:
        cfg_model = cfg.get("model")
        if cfg_model and resolve_model_spec(str(cfg_model)).name == spec.name:
            model_id = cfg.get("model_id")

    cmd = build_audioprune_cmd(
        spec.name,
        dataset=dspec.name,
        output_csv=str(args.output_csv.resolve()),
        data_root=str(Path(data_root).resolve()) if data_root else None,
        variant=variant,
        prune_rate=r,
        length_lambda=lam,
        kernel_kind=str(cfg.get("kernel_kind", "uniform")),
        gaussian_var=float(cfg.get("gaussian_var", 0.5)),
        device=str(device),
        max_new_tokens=cfg.get("max_new_tokens"),
        batch_size=int(cfg.get("batch_size", 1)),
        seed=int(cfg.get("seed", 42)),
        max_samples=args.max_samples,
        model_id=model_id,
        python=sys.executable,
    )
    print(f"[model={spec.name} dataset={dspec.name}] {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT), env=env))


if __name__ == "__main__":
    main()
