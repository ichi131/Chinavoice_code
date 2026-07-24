#!/bin/bash
set -euo pipefail

cd /mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning
export PATH=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin:$PATH
export PYTHONPATH=$PWD/fireredasr2s:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

# --- 路径配置 ---
exp_dir=${EXP_DIR:-./exp/lid_chinavoices_data_speaker_ft_encoder}
checkpoint=${CHECKPOINT:-$exp_dir/best.pt}
input_jsonl=${INPUT_JSONL:-/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/data/challenge_data_speaker/data_test.jsonl}
output_jsonl=${OUTPUT_JSONL:-$exp_dir/data_test_pred.jsonl}
output_txt=${OUTPUT_TXT:-$exp_dir/data_test_pred.txt}
batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-4}
pretrained_model_dir=${PRETRAINED_MODEL_DIR:-./pretrained_models/FireRedLID}

echo "Using single GPU: physical GPU ${GPU_ID:-0}"

python3.10 examples_train/lid_chinavoices/infer_lid_chinavoices.py \
  --checkpoint "$checkpoint" \
  --input_jsonl "$input_jsonl" \
  --output "$output_jsonl" \
  --pretrained_model_dir "$pretrained_model_dir" \
  --batch_size "$batch_size" \
  --num_workers "$num_workers"

# --- 转成 txt ---
python3.10 - "$output_jsonl" "$output_txt" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        fout.write(f"{obj['key']}\t{obj['accent']}\n")
PY

wc -l "$input_jsonl" "$output_jsonl" "$output_txt"
echo "jsonl: $output_jsonl"
echo "txt:   $output_txt"
