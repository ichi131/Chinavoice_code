#!/bin/bash
set -euo pipefail

cd /mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning
export PATH=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin:$PATH
export PYTHONPATH=$PWD/fireredasr2s:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node=$NPROC_PER_NODE
else
  nproc_per_node=$(python3.10 - <<'PY'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([x for x in visible.split(",") if x.strip()]) if visible else 1)
PY
)
fi

if [[ "$nproc_per_node" -gt 1 ]]; then
  launcher=(torchrun --standalone --nnodes 1 --nproc_per_node "$nproc_per_node")
else
  launcher=(python3.10)
fi

"${launcher[@]}" examples_train/lid_chinavoices/finetune_lid_chinavoices.py \
  --train_jsonl /mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_train_vc.jsonl \
  --val_jsonl /mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_val_vc.jsonl \
  --pretrained_model_dir ./pretrained_models/FireRedLID \
  --output_dir ./exp/lid_chinavoices_data_speaker_ft_encoder \
  --epochs 10 \
  --batch_size 24 \
    --num_workers 8 \
  --lr 1e-3 \
  --encoder_lr 1e-5 \
  --weight_decay 1e-5 \
  --dropout 0.2 \
  --label_smoothing 0.05 \
  --grad_clip 1.0 \
  --warmup_steps 500 \
  --freeze_encoder 0 \
  --use_amp 1 \
  --patience 3 \
  --min_delta 0.001
