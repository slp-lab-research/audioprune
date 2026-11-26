#!/usr/bin/env bash
# Multi-model / multi-dataset AudioPrune sweep.
#
#   MODEL=af3|qwen2_audio|salmonn|kimi_audio
#   DATASET=mmau_test_mini|mmau_test|clotho|clotho_aqa|cremad|meld|librispeech
#   VARIANT=k4|ms24|ms248
#   DATA_ROOT=...          # optional override
#   LENGTH_LAMBDAS="0.5 0.7 0.8"
#   EXPECTED_ROWS=1000     # optional; defaults from dataset registry when possible
#
# Examples:
#   MODEL=af3 DATASET=mmau_test_mini VARIANT=k4 bash code/scripts/eval.sh
#   MODEL=qwen2_audio DATASET=clotho VARIANT=ms24 bash code/scripts/eval.sh
#   MODEL=salmonn DATASET=cremad VARIANT=k4 bash code/scripts/eval.sh
#   MODEL=kimi_audio DATASET=meld VARIANT=ms24 bash code/scripts/eval.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
LAUNCHER="${CODE_ROOT}/scripts/run_eval.py"

MODEL="${MODEL:-af3}"
DATASET="${DATASET:-mmau_test_mini}"
VARIANT="${VARIANT:-k4}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/default.yaml}"
LENGTH_LAMBDA="${LENGTH_LAMBDA:-0.5}"
LENGTH_LAMBDAS="${LENGTH_LAMBDAS:-${LENGTH_LAMBDA}}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/${DATASET}_audioprune_${VARIANT}_${MODEL}}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${MODEL}" == "kimi_audio" || "${MODEL}" == "kimi" ]]; then
  export PYTHONPATH="${REPO_ROOT}/third_party/Kimi-Audio:${PYTHONPATH:-}"
fi

# Resolve metric script + expected rows from dataset registry.
read -r EVAL_SCRIPT EXPECTED_ROWS < <(
  python - <<PY
import sys
sys.path.insert(0, "${CODE_ROOT}")
from datasets import resolve_dataset, eval_script_for
d = resolve_dataset("${DATASET}")
print(f"${REPO_ROOT}/{eval_script_for(d.metric)}", d.expected_rows)
PY
)
if [[ -n "${EXPECTED_ROWS_OVERRIDE:-}" ]]; then
  EXPECTED_ROWS="${EXPECTED_ROWS_OVERRIDE}"
fi

RS=(0.9 0.75 0.5)
RS_LABELS=(90 75 50)

csv_is_complete() {
  local csv_path="$1"
  [[ -f "${csv_path}" ]] || return 1
  [[ -s "${csv_path}" ]] || return 1
  if [[ "${EXPECTED_ROWS}" -le 0 ]]; then
    return 0
  fi
  local rows
  rows="$(awk 'END {print NR}' "${csv_path}")"
  [[ "${rows}" -ge $((EXPECTED_ROWS + 1)) ]]
}

run_eval() {
  local out_csv="$1"
  local r="${2:-}"
  local out_json="${out_csv%.csv}.json"
  if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "No eval script for this dataset metric; skipping JSON eval."
    return 0
  fi
  if [[ -f "${out_json}" ]] && [[ -s "${out_json}" ]]; then
    echo "Eval JSON already exists, skipping: ${out_json}"
    return 0
  fi
  local -a eval_args=(--csv-path "${out_csv}" --output-json "${out_json}")
  if [[ -n "${r}" ]]; then
    eval_args+=(--R "${r}")
  fi
  # Clotho CIDEr needs ref csv; pass through if present.
  if [[ "${DATASET}" == "clotho" ]]; then
    local ref="${REF_CSV:-${REPO_ROOT}/data/clotho_v2/clotho_captions_evaluation.csv}"
    if [[ -f "${ref}" ]]; then
      eval_args+=(--ref-csv "${ref}")
    fi
  fi
  echo "Evaluating -> ${out_json}"
  python "${EVAL_SCRIPT}" "${eval_args[@]}" || true
}

run_lambda() {
  local length_lambda="$1"
  local lam_pct
  lam_pct="$(python -c "print(int(round(float('${length_lambda}') * 100)))")"

  echo "=== AudioPrune model=${MODEL} dataset=${DATASET} variant=${VARIANT} λ=${length_lambda} ==="
  echo "Config      : ${CONFIG}"
  echo "Output dir  : ${OUTPUT_DIR}"
  echo "Data root   : ${DATA_ROOT:-<default>}"
  echo "Eval script : ${EVAL_SCRIPT}"
  echo

  for i in "${!RS[@]}"; do
    local r="${RS[$i]}"
    local label="${RS_LABELS[$i]}"
    local out_csv="${OUTPUT_DIR}/${DATASET}_audioprune_${MODEL}_${VARIANT}_lam${lam_pct}_R${label}.csv"

    if csv_is_complete "${out_csv}"; then
      echo "Skipping inference (complete): ${out_csv}"
      run_eval "${out_csv}" "${r}"
      continue
    fi

    local -a launch=(
      python "${LAUNCHER}"
      --config "${CONFIG}"
      --model "${MODEL}"
      --dataset "${DATASET}"
      --variant "${VARIANT}"
      --R "${r}"
      --length-lambda "${length_lambda}"
      --device "${DEVICE}"
      --output-csv "${out_csv}"
    )
    if [[ -n "${DATA_ROOT}" ]]; then
      launch+=(--data-root "${DATA_ROOT}")
    fi
    if [[ -n "${MAX_SAMPLES}" ]]; then
      launch+=(--max-samples "${MAX_SAMPLES}")
    fi
    echo "Running -> ${out_csv}"
    "${launch[@]}"
    run_eval "${out_csv}" "${r}"
    echo
  done
}

# shellcheck disable=SC2206
LAM_ARR=(${LENGTH_LAMBDAS})
echo "Models   : af3 qwen2_audio salmonn kimi_audio"
echo "Datasets : mmau_test_mini mmau_test clotho clotho_aqa cremad meld librispeech"
echo "Lambda   : ${LAM_ARR[*]}"
for lam in "${LAM_ARR[@]}"; do
  run_lambda "${lam}"
done

echo "Done. Results under: ${OUTPUT_DIR}"
