#!/bin/bash
set -euo pipefail

# 在 docker 容器里跑 run_decode_evaluation_asr_chinavoices.sh。
# 所有宿主机路径都通过环境变量传入，脚本本身不写死任何机器/用户相关的绝对路径，
# 换一台机器只需要重新 export 这几个变量即可。
#
# 用法：
#   HOST_DATA_ROOT=/n/work6/yiwang/chinavoices_challenge/evaluation_set \
#     bash examples_train/asr_chinavoices/run_docker_decode_evaluation_asr_chinavoices.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host_project_root=${HOST_PROJECT_ROOT:-$(cd "$script_dir/../.." && pwd)}

# --- 宿主机路径（必填，按你自己机器上的实际位置传入） ---
host_data_root=${HOST_DATA_ROOT:?ERROR: 请设置 HOST_DATA_ROOT（宿主机上 evaluation_set 目录路径，须包含 wav.scp，如 .../evaluation_set）}

# --- 宿主机路径（有默认值，可覆盖） ---
host_model_dir=${HOST_MODEL_DIR:-$host_project_root/exp/asr_chinavoices_vc}
# 输出/中间产物目录默认单独指到一个目录（不是 model_dir 本身），因为 model_dir 只读挂载，
# 避免 mkdir/写文件时 Permission denied。
host_exp_dir=${HOST_EXP_DIR:-$host_project_root/exp/asr_chinavoices_eval}

host_data_root=$(cd "$host_data_root" && pwd)
host_model_dir=$(cd "$host_model_dir" && pwd)
mkdir -p "$host_exp_dir"
host_exp_dir=$(cd "$host_exp_dir" && pwd)

# --- 容器内固定路径（跟宿主机路径无关，不需要改） ---
container_project_root=/workspace/FireRedASR2S-fintuning
container_data_root=/data/evaluation_set
container_model_dir=/models/asr_chinavoices_vc
container_exp_dir=/output/asr_chinavoices_eval

# --- 其他可覆盖参数 ---
image=${DOCKER_IMAGE:-ghcr.io/ichi131/qwen3asr:py312-torch2.6-cu124}
gpu_ids=${GPU_IDS:-all}
batch_size=${BATCH_SIZE:-16}
use_half=${USE_HALF:-1}

echo "image:             $image"
echo "host_project_root: $host_project_root"
echo "host_data_root:    $host_data_root"
echo "host_model_dir:    $host_model_dir"
echo "host_exp_dir:      $host_exp_dir"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  --shm-size=16g \
  -v "$host_project_root:$container_project_root" \
  -v "$host_data_root:$container_data_root:ro" \
  -v "$host_model_dir:$container_model_dir:ro" \
  -v "$host_exp_dir:$container_exp_dir" \
  -e FIREREDASR2S_ROOT="$container_project_root" \
  -e DATA_ROOT="$container_data_root" \
  -e MODEL_DIR="$container_model_dir" \
  -e EXP_DIR="$container_exp_dir" \
  -e GPU_IDS="$gpu_ids" \
  -e BATCH_SIZE="$batch_size" \
  -e USE_HALF="$use_half" \
  "$image" \
  bash "$container_project_root/examples_train/asr_chinavoices/run_decode_evaluation_asr_chinavoices.sh"

echo "Done. Output written under host path: $host_exp_dir"
