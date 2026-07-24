#!/bin/bash
set -euo pipefail

project_root=/mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning
cd "$project_root"

export PATH=/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin:$PATH
export PYTHONPATH=$PWD/fireredasr2s:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

# --- 路径配置 ---
exp_dir=${EXP_DIR:-./exp/lid_chinavoices_data_speaker_ft_encoder}
checkpoint=${CHECKPOINT:-$exp_dir/best.pt}
wav_scp=${WAV_SCP:-/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/data/chinavoices_challenge/evaluation_set/wav.scp}
input_jsonl=${INPUT_JSONL:-$exp_dir/evaluation_input.jsonl}
output_jsonl=${OUTPUT_JSONL:-$exp_dir/evaluation_pred.jsonl}
batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-4}
pretrained_model_dir=${PRETRAINED_MODEL_DIR:-./pretrained_models/FireRedLID}

mkdir -p "$exp_dir"
mkdir -p "$(dirname "$input_jsonl")"
mkdir -p "$(dirname "$output_jsonl")"

if [[ ! -f "$wav_scp" ]]; then
  echo "ERROR: wav.scp not found: $wav_scp" >&2
  exit 1
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: checkpoint not found: $checkpoint" >&2
  exit 1
fi

echo "Using single GPU: physical GPU ${GPU_ID:-0}"
echo "wav.scp:    $wav_scp"
echo "checkpoint: $checkpoint"
echo "output:     $output_jsonl"

# wav.scp 中的相对路径以 chinavoices_challenge 目录为基准。
python3.10 - "$wav_scp" "$input_jsonl" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2])

# src 为 .../chinavoices_challenge/evaluation_set/wav.scp
# wav.scp 中路径形如 evaluation_set/wav/eval_000001.wav
data_root = src.parent.parent

num_samples = 0
with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line_number, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(
                f"Invalid wav.scp line {line_number}: expected '<key> <wav_path>', "
                f"got {line!r}"
            )

        key, wav_path = fields
        wav_path = Path(wav_path)
        if not wav_path.is_absolute():
            wav_path = data_root / wav_path

        fout.write(
            json.dumps(
                {
                    "key": key,
                    "wav_path": str(wav_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        num_samples += 1

print(f"Converted {num_samples} samples: {src} -> {dst}")
PY

python3.10 examples_train/lid_chinavoices/infer_lid_chinavoices.py \
  --checkpoint "$checkpoint" \
  --input_jsonl "$input_jsonl" \
  --output "$output_jsonl" \
  --pretrained_model_dir "$pretrained_model_dir" \
  --batch_size "$batch_size" \
  --num_workers "$num_workers"

wc -l "$wav_scp" "$input_jsonl" "$output_jsonl"
echo "Decode finished."
echo "jsonl: $output_jsonl"
