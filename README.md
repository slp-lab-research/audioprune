# AudioPrune code package

Clean layout for the paper method (**AudioPrune**): Foote checkerboard novelty
on audio-token SSMs, event peak picking, and length/attention keep-budget
reallocation.

```
code/
├── audioprune/          # core algorithm (model-agnostic)
├── models/              # AF3 / Qwen2-Audio / SALMONN / Kimi-Audio
├── datasets/            # dataset registry (paths, metrics, aliases)
├── baselines/
├── evaluation/
├── configs/
├── scripts/
│   ├── run_eval.py      # multi-model × multi-dataset launcher
│   └── eval.sh          # R / λ sweep
└── README.md
```

Repo-root runners (dataset via CLI / env, not filename):

| File | Role |
|------|------|
| `flamingo3_audioprune_eval.py` | AF3 AudioPrune |
| `flamingo3_audioprune_single_infer.py` | AF3 k4 CLI |
| `flamingo3_audioprune_multiscale_infer.py` | AF3 multiscale CLI |
| `qwen2_audio_checkboard_budget_realloc_infer.py` | Qwen2 AudioPrune |
| `eval_audioprune_af3.sh` | thin wrapper → `code/scripts/eval.sh` |

## Algorithm (short)

1. Encoder embeddings → cosine **SSM**
2. **Novelty** via Foote checkerboard (single-scale `4×4` or multi-scale `L∈{2,4}`)
3. **Peak detection** → event spans
4. Phase-1: drop `floor(L_e · R)` lowest-attention tokens per event (`min_keep=0`)
5. Phase-2: restore/trim to `K = floor(N · (1-R))` with  
   `score = (1-λ)·attn_norm + λ·length_norm`

## Models × datasets

| Model (`MODEL`) | Typical datasets |
|-----------------|------------------|
| `af3` | `mmau_test_mini`, `mmau_test`, `librispeech` |
| `qwen2_audio` | `mmau_test_mini`, `clotho`, `clotho_aqa`, `meld` |
| `salmonn` | `mmau_test_mini`, `mmau_test`, `clotho`, `clotho_aqa`, `cremad` |
| `kimi_audio` | `mmau_test_mini`, `clotho`, `clotho_aqa`, `meld`, `cremad` |

```bash
python code/scripts/run_eval.py --list-models --output-csv /tmp/x.csv
python code/scripts/run_eval.py --list-datasets --output-csv /tmp/x.csv

python code/scripts/run_eval.py \
  --model qwen2_audio --dataset clotho --variant ms24 \
  --output-csv results/tmp.csv --R 0.5 --dry-run

MODEL=af3 DATASET=mmau_test_mini VARIANT=k4 bash code/scripts/eval.sh
MODEL=salmonn DATASET=cremad VARIANT=k4 bash code/scripts/eval.sh
MODEL=kimi_audio DATASET=meld VARIANT=ms24 bash code/scripts/eval.sh
```

Override roots with `DATA_ROOT=/path` or `--data-root`.

## Core smoke test

```bash
python code/scripts/smoke_audioprune_core.py
```

## Import

```bash
export PYTHONPATH=/path/to/compression0/code:$PYTHONPATH
```

```python
from audioprune import audioprune_drop_ranks
from datasets import list_datasets, resolve_dataset
from models import list_models, build_audioprune_cmd
```

## License

This project is released under the [MIT License](LICENSE).
