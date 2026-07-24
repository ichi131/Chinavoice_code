#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/workspace/FireRedASR2S-fintuning
PYTHON_BIN=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S/bin/python3.10
MODEL_DIR=${MODEL_DIR:-$PROJECT_ROOT/exp/asr_chinavoices_llm_lora_20260720_140508}
INPUT_JSONL=${INPUT_JSONL:-/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/workspace/challenge_full_ft/data/test.jsonl}
OUTPUT_JSONL=${OUTPUT_JSONL:-$MODEL_DIR/pred_test.jsonl}
GPU_IDS=${GPU_IDS:-${GPU_ID:-0}}
BATCH_SIZE=${BATCH_SIZE:-1}

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

PYTHON_ENV_DIR=$(dirname "$(dirname "$PYTHON_BIN")")
TORCH_LIB_DIR=$PYTHON_ENV_DIR/lib/python3.10/site-packages/torch/lib
export PATH="$PYTHON_ENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
unset PYTHONHOME
unset PYTHONSTARTUP
unset LD_PRELOAD
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/fireredasr2s"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$PYTHON_ENV_DIR/lib"

decode_args=(
  --model-dir "$MODEL_DIR"
  --input-jsonl "$INPUT_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --gpu-ids "$GPU_IDS"
  --batch-size "$BATCH_SIZE"
  --beam-size "${BEAM_SIZE:-3}"
  --decode-max-len "${DECODE_MAX_LEN:-0}"
  --decode-min-len "${DECODE_MIN_LEN:-0}"
  --repetition-penalty "${REPETITION_PENALTY:-3.0}"
  --length-penalty "${LENGTH_PENALTY:-1.0}"
  --temperature "${TEMPERATURE:-1.0}"
  --log-interval "${LOG_INTERVAL:-20}"
  --limit "${LIMIT:-0}"
)
if [[ "${USE_HALF:-0}" == "1" ]]; then
  decode_args+=(--use-half)
fi

printf 'model:          %s\ninput:          %s\noutput:         %s\ngpu_ids:        %s\nbatch_per_gpu:  %s\n' \
  "$MODEL_DIR" "$INPUT_JSONL" "$OUTPUT_JSONL" "$GPU_IDS" "$BATCH_SIZE"
"$PYTHON_BIN" \
  examples_train/asr_chinavoices_llm/decode_asr_chinavoices_llm.py \
  "${decode_args[@]}"
