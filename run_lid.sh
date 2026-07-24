#!/bin/bash
set -euo pipefail

# 一键运行 LID（方言识别）流水线：拉镜像 -> 拉代码 -> 下模型 -> 跑 FireRedLID
# -> 跑 Qwen3ASR-LID -> 融合两者结果。只需要改下面两行，其余全部自动推导。
#
# 用法：
#   WORKDIR=/n/work6/yiwang/docker_test \
#   DATA_ROOT=/n/work6/yiwang/chinavoices_challenge/evaluation_set \
#     bash run_lid.sh

# ============ 只需要改这两行 ============
workdir=${WORKDIR:?ERROR: 请设置 WORKDIR（本次运行用的工作目录，代码和模型都会下载到这里面，如 /n/work6/yiwang/docker_test）}
data_root=${DATA_ROOT:?ERROR: 请设置 DATA_ROOT（evaluation_set 目录路径，如 .../chinavoices_challenge/evaluation_set；其上层路径因机器而异，直接指到 evaluation_set 即可）}
# ========================================

repo_url=${REPO_URL:-https://github.com/ichi131/Chinavoice_code}
fr_lid_image=${FR_LID_IMAGE:-ghcr.io/ichi131/fireredasr2s:h20-cu124}
qwen_image=${QWEN_IMAGE:-ghcr.io/ichi131/qwen3asr:py312-torch2.6-cu124}
fr_lid_hf_repo=${FR_LID_HF_REPO:-SerenaWhite/firered-lid-chinavoices}
qwen_hf_repo=${QWEN_HF_REPO:-SerenaWhite/qwen3asr-chinavoices-vc}
gpu_id=${GPU_ID:-0}

repo_dir=$workdir/Chinavoice_code
models_dir=$workdir/models
fr_lid_model_dir=$models_dir/firered-lid-chinavoices
qwen_model_dir=$models_dir/qwen3asr-chinavoices-vc

mkdir -p "$workdir" "$models_dir"

echo "=================================================="
echo "[0/5] 拉取 docker 镜像"
echo "=================================================="
sudo docker pull "$fr_lid_image"
sudo docker pull "$qwen_image"

echo "=================================================="
echo "[1/5] 拉取/更新代码仓库 -> $repo_dir"
echo "=================================================="
if [[ -d "$repo_dir/.git" ]]; then
  git -C "$repo_dir" pull --ff-only
else
  git clone "$repo_url" "$repo_dir"
fi

echo "=================================================="
echo "[2/5] 下载 FireRedLID checkpoint -> $fr_lid_model_dir"
echo "=================================================="
if [[ -f "$fr_lid_model_dir/best.pt" ]]; then
  echo "已存在，跳过：$fr_lid_model_dir/best.pt"
else
  HF_HUB_DISABLE_XET=1 hf download "$fr_lid_hf_repo" --local-dir "$fr_lid_model_dir"
fi

echo "=================================================="
echo "[3/5] 下载 Qwen3ASR checkpoint -> $qwen_model_dir"
echo "=================================================="
if [[ -f "$qwen_model_dir/config.json" ]]; then
  echo "已存在，跳过：$qwen_model_dir/config.json"
else
  HF_HUB_DISABLE_XET=1 hf download "$qwen_hf_repo" --local-dir "$qwen_model_dir"
fi

echo "=================================================="
echo "[4/5] step1 FireRedLID 解码"
echo "=================================================="
sudo docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$repo_dir/FireRedASR2S-fintuning:/workspace/FireRedASR2S-fintuning" \
  -v "$data_root:/data/evaluation_set:ro" \
  -v "$fr_lid_model_dir/best.pt:/models/checkpoint.pt:ro" \
  -e FIREREDASR2S_ROOT=/workspace/FireRedASR2S-fintuning \
  -e GPU_ID="$gpu_id" \
  -e CHECKPOINT=/models/checkpoint.pt \
  -e WAV_SCP=/data/evaluation_set/wav.scp \
  -e DATA_ROOT=/data/evaluation_set \
  -e EXP_DIR=/workspace/FireRedASR2S-fintuning/exp/lid_output \
  "$fr_lid_image" \
  bash /workspace/FireRedASR2S-fintuning/examples_train/lid_chinavoices/run_decode_evaluation_lid_chinavoices.sh

echo "=================================================="
echo "[5/5] step2 Qwen3ASR-LID 解码"
echo "=================================================="
sudo docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  --shm-size=16g \
  -v "$repo_dir:/workspace/Chinavoice_code" \
  -v "$qwen_model_dir:/models/checkpoint:ro" \
  -v "$data_root:/data/chinavoices_challenge/evaluation_set:ro" \
  -e MODEL_CKPT=/models/checkpoint \
  -e EVAL_SCP=/data/chinavoices_challenge/evaluation_set/wav.scp \
  "$qwen_image" \
  bash /workspace/Chinavoice_code/challenge_full_ft/infer_evalset.sh

echo "=================================================="
echo "融合 step1(FireRedLID) / step2(Qwen3ASR-LID) 两个解码 jsonl 的结果"
echo "=================================================="
# apply_fuse_c_v5.py 按自身文件位置推导 REPO_ROOT=$repo_dir，默认 FR_INFER/QW_INFER
# 正好对应 step1 的 evaluation_pred.jsonl 和 step2 的 result.jsonl，无需额外传参；
# 融合结果写到 $repo_dir/infer_data/lid.jsonl。
python3 "$repo_dir/analyse/apply_fuse_c_v5.py"

echo "=================================================="
echo "完成！融合结果：$repo_dir/infer_data/lid.jsonl"
echo "=================================================="
