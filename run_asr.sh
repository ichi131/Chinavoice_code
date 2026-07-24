#!/usr/bin/env bash
# =============================================================================
# run_asr.sh
# -----------------------------------------------------------------------------
# 一键跑通 ASR 提交流程：拉镜像 -> 拉代码 -> 从 Hugging Face 拉模型 -> docker 里跑
# FireRedASR2S 解码 -> 转换成竞赛提交格式 asr.jsonl。
#
# 用到的模型 SerenaWhite/firered-asr-chinavoices 是 FireRedASR2S 格式的 checkpoint
# （model.pth.tar / cmvn.ark / dict.txt / train_bpe1000.model / train_config.json），
# 对应的解码脚本是 FireRedASR2S-fintuning/examples_train/asr_chinavoices 下的
# run_decode_evaluation_asr_chinavoices.sh（通过其 docker 包装脚本调用）。
#
# 用法：
#   CLONE_DIR=/n/work6/yiwang/docker_test EVALUATION_SET_DIR=/n/work6/yiwang/chinavoices_challenge/evaluation_set bash run_asr.sh
# =============================================================================
set -euo pipefail

DOCKER_IMAGE=${DOCKER_IMAGE:-ghcr.io/ichi131/fireredasr2s:h20-cu124}
REPO_URL=${REPO_URL:-https://github.com/ichi131/Chinavoice_code.git}
CLONE_DIR=${CLONE_DIR:?"请通过环境变量 CLONE_DIR 指定代码克隆到的目录，如 /n/work6/yiwang/docker_test"}

sudo docker pull "$DOCKER_IMAGE"

# ---- 拉代码：固定 clone 到 CLONE_DIR 下 ----
PROJECT_ROOT="$CLONE_DIR/Chinavoice_code"
if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
    mkdir -p "$CLONE_DIR"
    git clone "$REPO_URL" "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"

# ---- 拉模型：下载到项目自己的 models/ 目录下 ----
MODEL_DIR=${MODEL_DIR:-"$PROJECT_ROOT/models/firered-asr-chinavoices"}
if [[ ! -f "$MODEL_DIR/model.pth.tar" ]]; then
    HF_HUB_DISABLE_XET=1 hf download SerenaWhite/firered-asr-chinavoices --local-dir "$MODEL_DIR"
fi

# ---- 评测集目录（机器相关，运行时通过环境变量指定）----
EVALUATION_SET_DIR=${EVALUATION_SET_DIR:?"请通过环境变量 EVALUATION_SET_DIR 指定评测集目录，如 /path/to/evaluation_set"}

# ---- docker 里跑 FireRedASR2S 解码，复用仓库自带的 docker 包装脚本 ----
HOST_PROJECT_ROOT="$PROJECT_ROOT/FireRedASR2S-fintuning" \
HOST_DATA_ROOT="$EVALUATION_SET_DIR" \
HOST_MODEL_DIR="$MODEL_DIR" \
DOCKER_IMAGE="$DOCKER_IMAGE" \
bash "$PROJECT_ROOT/FireRedASR2S-fintuning/examples_train/asr_chinavoices/run_docker_decode_evaluation_asr_chinavoices.sh"

# ---- 后处理：pred_evaluation.jsonl -> asr.jsonl ----
python "$PROJECT_ROOT/FireRedASR2S-fintuning/exp/asr_chinavoices_vc/convert_pred_to_asr.py"
