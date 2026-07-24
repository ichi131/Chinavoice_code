#!/bin/bash
set -euo pipefail

# 用 wav.scp 生成 decode_asr_chinavoices.py 需要的 evaluation.jsonl，再跑解码。
#
# 项目根目录：优先用外部注入的 FIREREDASR2S_ROOT（如 Docker 容器场景），
# 否则从脚本自身路径反推（宿主机直接运行场景），避免写死宿主机绝对路径。
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=${FIREREDASR2S_ROOT:-$(cd "$script_dir/../.." && pwd)}
cd "$project_root"

export PYTHONPATH=$PWD/fireredasr2s:${PYTHONPATH:-}

# qwen3asr 这个 docker 镜像是给 challenge_full_ft（Qwen3-ASR）用的，没有装
# FireRedASR2S-fintuning/requirements.txt 里的几个轻量依赖。这里只按需补装缺失的
# 几个包，不动 torch/numpy/transformers，避免跟镜像里已装的版本冲突。
missing_pkgs=()
for pkg in kaldiio kaldi_native_fbank sentencepiece soundfile textgrid; do
  python3 -c "import $pkg" >/dev/null 2>&1 || missing_pkgs+=("$pkg")
done
if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
  echo "Installing missing python packages: ${missing_pkgs[*]}"
  # --user：docker run 用非 root 身份跑容器时没权限写系统 site-packages，
  # 装到 $HOME/.local（脚本里 HOME 一般被设为 /tmp）。
  pip install --no-deps --user "${missing_pkgs[@]}"
fi

# --- 路径配置 ---
data_root=${DATA_ROOT:?ERROR: 请通过环境变量 DATA_ROOT 指定 evaluation_set 目录路径（须包含 wav.scp，如 .../evaluation_set）}
model_dir=${MODEL_DIR:-$project_root/exp/asr_chinavoices_vc}
exp_dir=${EXP_DIR:-$project_root/exp/asr_chinavoices_eval}
input_jsonl=${INPUT_JSONL:-$exp_dir/evaluation.jsonl}
output_jsonl=${OUTPUT_JSONL:-$exp_dir/pred_evaluation.jsonl}
gpu_ids=${GPU_IDS:-all}
batch_size=${BATCH_SIZE:-16}
use_half=${USE_HALF:-1}

wav_scp=$data_root/wav.scp
if [[ ! -f "$wav_scp" ]]; then
  echo "ERROR: wav.scp not found: $wav_scp" >&2
  exit 1
fi

mkdir -p "$exp_dir"
mkdir -p "$(dirname "$input_jsonl")"
mkdir -p "$(dirname "$output_jsonl")"

echo "data_root:    $data_root"
echo "model_dir:    $model_dir"
echo "input_jsonl:  $input_jsonl"
echo "output_jsonl: $output_jsonl"
echo "gpu_ids:      $gpu_ids"
echo "batch_size:   $batch_size"
echo "use_half:     $use_half"

# wav.scp 里的相对路径形如 evaluation_set/wav/eval_000001.wav（是相对于 $data_root
# 的上层目录写的）。所以这里如果相对路径的第一级目录名和 $data_root 自己的目录名同名
# （都是 "evaluation_set"），就把这个多余的前缀去掉，再拼到 $data_root 上，避免拼出
# evaluation_set/evaluation_set/wav/... 这种重复路径。生成的 evaluation.jsonl 字段
# 对齐 decode_asr_chinavoices.py 期望的输入格式（audio/text/prompt/key/accent）。
python3 - "$wav_scp" "$input_jsonl" "$data_root" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2])
data_root = Path(sys.argv[3]).resolve()
data_root_name = data_root.name

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
                    "audio": str(wav_path),
                    "text": "",
                    "prompt": "",
                    "key": key,
                    "accent": "",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        num_samples += 1

print(f"Converted {num_samples} samples: {src} -> {dst}")
PY

decode_args=(
  --model-dir "$model_dir"
  --input-jsonl "$input_jsonl"
  --output-jsonl "$output_jsonl"
  --gpu-ids "$gpu_ids"
  --batch-size "$batch_size"
)
if [[ "$use_half" == "1" ]]; then
  decode_args+=(--use-half)
fi

python3 examples_train/asr_chinavoices/decode_asr_chinavoices.py "${decode_args[@]}"

wc -l "$wav_scp" "$input_jsonl" "$output_jsonl"
echo "Decode finished."
echo "jsonl: $output_jsonl"
