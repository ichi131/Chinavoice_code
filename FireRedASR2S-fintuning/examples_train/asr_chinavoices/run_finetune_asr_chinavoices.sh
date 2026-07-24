#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/workspace/FireRedASR2S-fintuning
PYTHON_BIN=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin/python3.10
MODEL_DIR=$PROJECT_ROOT/pretrained_models/FireRedASR2-AED
TRAIN_JSONL=/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_train_vc.jsonl
VAL_JSONL=/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_val_vc.jsonl
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/exp/asr_chinavoices_vc}

cd "$PROJECT_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3)
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/fireredasr2s:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if [[ ! -f "$MODEL_DIR/model.pth.tar" ]]; then
  cat >&2 <<EOF
缺少 FireRedASR2-AED 预训练模型：$MODEL_DIR/model.pth.tar
请先下载到当前项目目录（约 4.73 GB）：
  hf download FireRedTeam/FireRedASR2-AED --local-dir "$MODEL_DIR"
EOF
  exit 1
fi

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node=$NPROC_PER_NODE
else
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  nproc_per_node=${#visible_gpus[@]}
fi

if [[ "$nproc_per_node" -gt 1 ]]; then
  master_addr=${MASTER_ADDR:-127.0.0.1}
  master_port=${MASTER_PORT:-29500}
  launcher=("$(dirname "$PYTHON_BIN")/torchrun"
    --nnodes 1
    --node_rank 0
    --nproc_per_node "$nproc_per_node"
    --master_addr "$master_addr"
    --master_port "$master_port")
else
  launcher=("$PYTHON_BIN")
fi

"${launcher[@]}" examples_train/asr_chinavoices/finetune_asr_chinavoices.py \
  --train_jsonl "$TRAIN_JSONL" \
  --val_jsonl "$VAL_JSONL" \
  --pretrained_model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-10}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --encoder_lr "${ENCODER_LR:-5e-6}" \
  --decoder_lr "${DECODER_LR:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-1e-2}" \
  --ctc_weight "${CTC_WEIGHT:-0.3}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.1}" \
  --warmup_steps "${WARMUP_STEPS:-1000}" \
  --grad_clip "${GRAD_CLIP:-5.0}" \
  --max_input_frames "${MAX_INPUT_FRAMES:-6000}" \
  --max_target_length "${MAX_TARGET_LENGTH:-256}" \
  --use_amp "${USE_AMP:-1}" \
  --save_optimizer "${SAVE_OPTIMIZER:-1}"
