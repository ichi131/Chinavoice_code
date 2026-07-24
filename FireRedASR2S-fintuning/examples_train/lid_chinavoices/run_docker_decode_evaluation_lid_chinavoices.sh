#!/bin/bash
set -euo pipefail

# 在 docker 容器里跑 run_decode_evaluation_lid_chinavoices.sh。
# 所有宿主机路径都通过环境变量传入，脚本本身不写死任何机器/用户相关的绝对路径，
# 换一台机器只需要重新 export 这几个变量即可。

# --- 宿主机路径（必填，按你自己机器上的实际位置传入） ---
host_data_root=${HOST_DATA_ROOT:?ERROR: 请设置 HOST_DATA_ROOT（宿主机上 evaluation_set 目录路径，如 .../chinavoices_challenge/evaluation_set；上层 chinavoices_challenge 路径因机器而异，直接指到 evaluation_set 即可）}
host_checkpoint=${HOST_CHECKPOINT:?ERROR: 请设置 HOST_CHECKPOINT（宿主机上 LID checkpoint 文件路径，如 best.pt）}

# --- 宿主机路径（有默认值，可覆盖） ---
# 项目根目录：默认按本脚本所在位置反推，不写死用户名/机器路径。
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host_project_root=${HOST_PROJECT_ROOT:-$(cd "$script_dir/../.." && pwd)}
host_wav_scp=${HOST_WAV_SCP:-$host_project_root/wav.scp}
# 输出目录默认单独指到宿主机的一个目录（不强制在项目目录下），避免项目目录所在的
# 共享/网络存储（如 NFS）对容器内用户没有写权限导致 mkdir Permission denied。
host_exp_dir=${HOST_EXP_DIR:-$host_project_root/exp/lid_output}

# --- 容器内固定路径（跟宿主机路径无关，不需要改） ---
container_project_root=/workspace/FireRedASR2S-fintuning
container_data_root=/data/chinavoices_challenge
container_checkpoint=/models/checkpoint.pt
container_wav_scp=/data/wav.scp
container_exp_dir=/output/lid_output

# --- 其他可覆盖参数 ---
image=${DOCKER_IMAGE:-ghcr.io/ichi131/fireredasr2s:h20-cu124}
gpu_id=${GPU_ID:-0}
batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-4}
# DataLoader num_workers>0 时，worker 进程间通过 /dev/shm 传递 tensor；docker 默认
# /dev/shm 只有 64MB，装不下就会报 "No space left on device" / bus error，因此加大它。
shm_size=${SHM_SIZE:-8g}

mkdir -p "$host_exp_dir"

echo "image:            $image"
echo "host_project_root: $host_project_root"
echo "host_data_root:    $host_data_root"
echo "host_checkpoint:   $host_checkpoint"
echo "host_wav_scp:      $host_wav_scp"
echo "host_exp_dir:      $host_exp_dir"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --shm-size="$shm_size" \
  -e HOME=/tmp \
  -v "$host_project_root:$container_project_root" \
  -v "$host_data_root:$container_data_root:ro" \
  -v "$host_checkpoint:$container_checkpoint:ro" \
  -v "$host_wav_scp:$container_wav_scp:ro" \
  -v "$host_exp_dir:$container_exp_dir" \
  -e FIREREDASR2S_ROOT="$container_project_root" \
  -e GPU_ID="$gpu_id" \
  -e CHECKPOINT="$container_checkpoint" \
  -e WAV_SCP="$container_wav_scp" \
  -e DATA_ROOT="$container_data_root" \
  -e EXP_DIR="$container_exp_dir" \
  -e BATCH_SIZE="$batch_size" \
  -e NUM_WORKERS="$num_workers" \
  "$image" \
  bash "$container_project_root/examples_train/lid_chinavoices/run_decode_evaluation_lid_chinavoices.sh"

echo "Done. Output written under host path: $host_exp_dir"
