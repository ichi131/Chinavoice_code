#!/bin/bash
set -euo pipefail

# =============================================================================
# run_docker_infer_evalset.sh
# -----------------------------------------------------------------------------
# 在 docker 容器里跑 challenge_full_ft/infer_evalset.sh。
# 所有宿主机路径都通过环境变量传入，脚本本身不写死任何机器/用户相关的绝对路径，
# 换一台机器只需要重新 export 这几个变量即可。
#
# 用法：
#   HOST_MODEL_CKPT=/path/to/checkpoint-500 \
#   HOST_EVAL_SCP=/n/work6/yiwang/chinavoices_challenge/evaluation_set/wav.scp \
#     bash challenge_full_ft/run_docker_infer_evalset.sh
# =============================================================================

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host_project_root=${HOST_PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}

# --- 宿主机路径（必填） ---
host_model_ckpt=${HOST_MODEL_CKPT:?ERROR: 请设置 HOST_MODEL_CKPT（宿主机上模型 checkpoint 目录路径，需含 config.json）}
host_eval_scp=${HOST_EVAL_SCP:?ERROR: 请设置 HOST_EVAL_SCP（宿主机上 wav.scp 路径，如 .../evaluation_set/wav.scp）}

host_model_ckpt=$(cd "$host_model_ckpt" && pwd)
host_eval_set_dir=$(cd "$(dirname "$host_eval_scp")" && pwd)
eval_scp_name=$(basename "$host_eval_scp")

# --- 容器内固定路径（跟宿主机路径无关，不需要改） ---
container_project_root=/workspace/Chinavoice_code
container_model_ckpt=/models/checkpoint
# 只挂 evaluation_set 本身；容器内父目录 /data/chinavoices_challenge 由 docker
# 自动创建（不对应任何宿主机目录），仅用来让 wav.scp 里 "evaluation_set/wav/xxx.wav"
# 这种相对路径（约定相对 wav.scp 所在目录的父目录解析）能正确落到挂载点上。
container_eval_set_dir=/data/chinavoices_challenge/evaluation_set
container_eval_scp="$container_eval_set_dir/$eval_scp_name"

# --- 其他可覆盖参数 ---
image=${DOCKER_IMAGE:-ghcr.io/ichi131/qwen3asr:py312-torch2.6-cu124}
num_gpus=${NUM_GPUS:-}
batch_size=${BATCH_SIZE:-32}
max_tokens=${MAX_TOKENS:-512}

echo "image:               $image"
echo "host_project_root:   $host_project_root"
echo "host_model_ckpt:     $host_model_ckpt"
echo "host_eval_set_dir:   $host_eval_set_dir"
echo "container_eval_scp:  $container_eval_scp"

# 输出目录不单独挂载：project_root 已整体挂进容器（读写），infer_evalset.sh
# 自身默认把 OUT_DIR 落在 ${SCRIPT_DIR}/infer_data，也就是宿主机上的
# challenge_full_ft/infer_data，跟着 project_root 一起出现，无需额外指定。
sudo docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  --shm-size=16g \
  -v "$host_project_root:$container_project_root" \
  -v "$host_model_ckpt:$container_model_ckpt:ro" \
  -v "$host_eval_set_dir:$container_eval_set_dir:ro" \
  -e MODEL_CKPT="$container_model_ckpt" \
  -e EVAL_SCP="$container_eval_scp" \
  -e NUM_GPUS="$num_gpus" \
  -e BATCH_SIZE="$batch_size" \
  -e MAX_TOKENS="$max_tokens" \
  "$image" \
  bash "$container_project_root/challenge_full_ft/infer_evalset.sh"

echo "Done. Output written under host path: $host_project_root/challenge_full_ft/infer_data"
