#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/workspace/FireRedASR2S-fintuning
PYTHON_BIN=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S/bin/python3.10
MODEL_DIR=$PROJECT_ROOT/pretrained_models/FireRedASR2-LLM
TRAIN_JSONL=/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_train_vc.jsonl
VAL_JSONL=/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_val_vc.jsonl
RUN_NAME=${RUN_NAME:-asr_chinavoices_llm_lora_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/exp/$RUN_NAME}

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3)
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

"$PYTHON_BIN" -c '
import sys
import torch
if not hasattr(torch._C, "_OutOfMemoryError"):
    extension_path = getattr(torch._C, "__file__", None)
    raise RuntimeError(
        f"PyTorch Python/C extension mismatch: torch={torch.__file__}, "
        f"torch._C={extension_path}"
    )
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__} ({torch.__file__})")
'

required_files=(
  "$MODEL_DIR/model.pth.tar"
  "$MODEL_DIR/asr_encoder.pth.tar"
  "$MODEL_DIR/cmvn.ark"
  "$MODEL_DIR/Qwen2-7B-Instruct/config.json"
  "$MODEL_DIR/Qwen2-7B-Instruct/model.safetensors.index.json"
  "$TRAIN_JSONL"
  "$VAL_JSONL"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "缺少必要文件：$required_file" >&2
    exit 1
  fi
done

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node=$NPROC_PER_NODE
else
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  nproc_per_node=${#visible_gpus[@]}
fi

if [[ "$nproc_per_node" -gt 1 ]]; then
  launcher=("$PYTHON_BIN" -m torch.distributed.run
    --nnodes 1
    --node_rank 0
    --nproc_per_node "$nproc_per_node"
    --master_addr "${MASTER_ADDR:-127.0.0.1}"
    --master_port "${MASTER_PORT:-29501}")
else
  launcher=("$PYTHON_BIN")
fi

train_args=(
  --train_jsonl "$TRAIN_JSONL"
  --val_jsonl "$VAL_JSONL"
  --pretrained_model_dir "$MODEL_DIR"
  --output_dir "$OUTPUT_DIR"
  --train_mode "${TRAIN_MODE:-adapter_lora}"
  --freeze_encoder "${FREEZE_ENCODER:-1}"
  --epochs "${EPOCHS:-10}"
  --batch_size "${BATCH_SIZE:-1}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-8}"
  --num_workers "${NUM_WORKERS:-4}"
  --adapter_lr "${ADAPTER_LR:-1e-4}"
  --lora_lr "${LORA_LR:-1e-4}"
  --encoder_lr "${ENCODER_LR:-5e-6}"
  --weight_decay "${WEIGHT_DECAY:-1e-2}"
  --warmup_steps "${WARMUP_STEPS:-500}"
  --grad_clip "${GRAD_CLIP:-1.0}"
  --max_input_frames "${MAX_INPUT_FRAMES:-4000}"
  --max_text_length "${MAX_TEXT_LENGTH:-256}"
  --use_amp "${USE_AMP:-1}"
  --use_flash_attn "${USE_FLASH_ATTN:-0}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-1}"
  --save_optimizer "${SAVE_OPTIMIZER:-1}"
  --seed "${SEED:-1337}"
  --log_interval "${LOG_INTERVAL:-20}"
)
if [[ -n "${RESUME:-}" ]]; then
  train_args+=(--resume "$RESUME")
fi

echo "FireRedASR2-LLM 微调输出目录：$OUTPUT_DIR"
"${launcher[@]}" \
  examples_train/asr_chinavoices_llm/finetune_asr_chinavoices_llm.py \
  "${train_args[@]}"
