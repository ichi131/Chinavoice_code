#!/bin/bash
set -euo pipefail

# 项目根目录：优先用外部注入的 FIREREDASR2S_ROOT（如 Docker 容器场景），
# 否则从脚本自身路径反推（宿主机直接运行场景），避免写死宿主机绝对路径。
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=${FIREREDASR2S_ROOT:-$(cd "$script_dir/../.." && pwd)}
cd "$project_root"

export PYTHONPATH=$PWD/fireredasr2s:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

# --- 路径配置 ---
exp_dir=${EXP_DIR:-./exp/lid_output}
checkpoint=${CHECKPOINT:?ERROR: 请通过环境变量 CHECKPOINT 指定 LID checkpoint 路径（docker run -e CHECKPOINT=<容器内路径>，并用 -v 挂载对应宿主机文件）}
wav_scp=${WAV_SCP:?ERROR: 请通过环境变量 WAV_SCP 指定 wav.scp 路径（docker run -e WAV_SCP=<容器内路径>，并用 -v 挂载对应宿主机文件）}
data_root=${DATA_ROOT:?ERROR: 请通过环境变量 DATA_ROOT 指定 evaluation_set 目录路径（如 .../chinavoices_challenge/evaluation_set，其上层 chinavoices_challenge 路径可能因机器而异，直接指到 evaluation_set 即可）}
input_jsonl=${INPUT_JSONL:-$exp_dir/evaluation_input.jsonl}
output_jsonl=${OUTPUT_JSONL:-$exp_dir/evaluation_pred.jsonl}
batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-4}
# LID 推理只依赖 pretrained_model_dir 下的两个文件：
#   - model.pth.tar：仅用其中的模型结构 args 搭建 encoder 骨架，其权重会被下面加载的
#     checkpoint 以 strict=True 整体覆盖，具体数值不影响推理结果；
#   - cmvn.ark：特征提取的归一化统计量，必须与训练时使用的一致。
# 因此默认指向 pretrained_models/FireRedLID_min ——它是从 pretrained_models/FireRedLID
# 离线抽取出的精简版（model.pth.tar 只保留 args，权重置空，约几 KB），不再依赖体积几 GB
# 的原始 pretrained_model_dir，即使解码环境里没有 pretrained_models/FireRedLID 也能跑。
pretrained_model_dir=${PRETRAINED_MODEL_DIR:-./pretrained_models/FireRedLID_min}

mkdir -p "$exp_dir"
mkdir -p "$(dirname "$input_jsonl")"
mkdir -p "$(dirname "$output_jsonl")"

if [[ ! -f "$wav_scp" ]]; then
  echo "ERROR: wav.scp not found: $wav_scp" >&2
  echo "Hint: 用 -e WAV_SCP=<路径> 指定 wav.scp 位置，用 -e DATA_ROOT=<路径> 指定" >&2
  echo "      其中相对路径的拼接基准目录（容器内需确保二者均已挂载可见）。" >&2
  exit 1
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: checkpoint not found: $checkpoint" >&2
  exit 1
fi

echo "Using single GPU: physical GPU ${GPU_ID:-0}"
echo "wav.scp:    $wav_scp"
echo "data_root:  $data_root"
echo "checkpoint: $checkpoint"
echo "output:     $output_jsonl"

# $DATA_ROOT 指向 evaluation_set 目录本身；但 wav.scp 里的相对路径形如
# evaluation_set/wav/eval_000001.wav（是相对于 evaluation_set 的上层目录写的）。
# 所以这里如果相对路径的第一级目录名和 $DATA_ROOT 自己的目录名同名（都是
# "evaluation_set"），就把这个多余的前缀去掉，再拼到 $DATA_ROOT 上，
# 避免拼出 evaluation_set/evaluation_set/wav/... 这种重复路径。
python3.10 - "$wav_scp" "$input_jsonl" "$data_root" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2])
data_root = Path(sys.argv[3]).resolve()
data_root_name = data_root.name

# wav.scp 中路径形如 evaluation_set/wav/eval_000001.wav

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
            parts = wav_path.parts
            if parts and parts[0] == data_root_name:
                wav_path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
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
