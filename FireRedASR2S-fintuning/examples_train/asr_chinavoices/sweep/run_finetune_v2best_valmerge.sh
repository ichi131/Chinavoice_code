#!/bin/bash
# =============================================================================
# 一键脚本：回到 sweep 最优超参 + VC_data_v2 训练集 + val 合并集 训练 → 推理 → CER 评估
# ---------------------------------------------------------------------------
# 说明：
#   本脚本直接串起 train → decode → format → eval 四步（绕开 sweep 框架），
#   因为 sweep_main.py 在每维度只放 1 个候选时会判为 "skipped=True" 直接跳过训练，
#   导致 sweep_summary 里 test overall CER = None（可参考同目录 run_train_test_mixft_v5_best.sh）。
#
# 目的：
#   1. 复用 sweep_20260721_114230 最优 run（04_label_smoothing__0.05, test overall CER=11.53%）
#      的全部训练超参和训练数据 VC_data_v2/data_train_vc.jsonl；
#   2. 仅将 val_jsonl 替换为 VC_data_v2/data_val_plus_test_vc.jsonl（val+test 合并集），
#      让 best ckpt 的选择更稳定；
#   3. epochs=8（sweep 最优 run best 落在 epoch 7，+1 buffer，节省 ~20% 时间）。
#
# 流程：
#   Step 1. 预检查输入文件是否齐全
#   Step 2. torchrun 调 finetune_asr_chinavoices.py 训练（最优超参 + epochs=8）
#   Step 3. 生成 test 集推理输入（拼 "language Chinese <accent><asr_text><text>" 前缀）
#   Step 4. python decode_asr_chinavoices.py 推理 test 集
#   Step 5. python format_pred_jsonl.py → bash eval_jsonl_with_wer_tools.sh → 抽 Overall CER
#
# 用法：
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash examples_train/asr_chinavoices/sweep/run_finetune_v2best_valmerge.sh
#
# 可选环境变量：
#   OUTPUT_DIR      自定义产物目录（默认 exp/asr_v2best_valmerge_<时间戳>）
#   EPOCHS          覆盖 epochs（默认 8）
#   BATCH_SIZE      覆盖 batch_size（默认 4）
#   SKIP_TRAIN=1    跳过训练（仅当 OUTPUT_DIR/model.pth.tar 已存在时用于补跑评估）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# 数据 / 环境 路径
# ---------------------------------------------------------------------------
TRAIN_JSONL=/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_train_vc.jsonl
VAL_JSONL=/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2/data_val_plus_test_vc.jsonl
TEST_JSONL_SRC=/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/data/challenge_data_speaker/data_test.jsonl

MODEL_DIR=$PROJECT_ROOT/pretrained_models/FireRedASR2-AED
FINETUNE_PY=$PROJECT_ROOT/examples_train/asr_chinavoices/finetune_asr_chinavoices.py
DECODE_PY=$PROJECT_ROOT/examples_train/asr_chinavoices/decode_asr_chinavoices.py
FORMAT_PY=/mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning/exp/asr_chinavoices_vc/format_pred_jsonl.py
EVAL_SH=/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh

PYTHON_BIN=${PYTHON_BIN:-/mnt/geminihzceph/user_ichiwang/envs/FireRedASR2S_H20/bin/python3.10}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3)
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/fireredasr2s:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# 分布式训练稳定性设置（与 run_sweep_asr_chinavoices.sh 保持一致）
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# 超参（默认 = sweep_example.yaml 04_label_smoothing__0.05 winner，仅 epochs 由 10 → 8）
EPOCHS=${EPOCHS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
ENCODER_LR=${ENCODER_LR:-1e-5}
DECODER_LR=${DECODER_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
CTC_WEIGHT=${CTC_WEIGHT:-0.3}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.05}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
GRAD_CLIP=${GRAD_CLIP:-5.0}
MAX_INPUT_FRAMES=${MAX_INPUT_FRAMES:-6000}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-256}
USE_AMP=${USE_AMP:-1}
SAVE_OPTIMIZER=${SAVE_OPTIMIZER:-1}
SEED=${SEED:-1337}
LOG_INTERVAL=${LOG_INTERVAL:-100}

# 推理超参（与 sweep_example 完全一致，保证与 11.53% 同口径对比）
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-16}
BEAM_SIZE=${BEAM_SIZE:-3}
DECODE_MAX_LEN=${DECODE_MAX_LEN:-300}
SOFTMAX_SMOOTHING=${SOFTMAX_SMOOTHING:-1.25}
LENGTH_PENALTY=${LENGTH_PENALTY:-0.6}
EOS_PENALTY=${EOS_PENALTY:-1.0}

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/exp/asr_v2best_valmerge_${TS}}
mkdir -p "$OUTPUT_DIR"

# 每步共享的中间产物
TEST_INPUT_JSONL=$OUTPUT_DIR/test_input_converted.jsonl
PRED_JSONL=$OUTPUT_DIR/pred_test.jsonl
PRED_FMT_JSONL=$OUTPUT_DIR/pred_test_formatted.jsonl
WER_DIR=$OUTPUT_DIR/wer_eval
TRAIN_LOG=$OUTPUT_DIR/train.log
INFER_LOG=$OUTPUT_DIR/infer.log
EVAL_LOG=$OUTPUT_DIR/eval.log

echo "[v2best_valmerge] PROJECT_ROOT   = $PROJECT_ROOT"
echo "[v2best_valmerge] OUTPUT_DIR     = $OUTPUT_DIR"
echo "[v2best_valmerge] TRAIN_JSONL    = $TRAIN_JSONL"
echo "[v2best_valmerge] VAL_JSONL      = $VAL_JSONL"
echo "[v2best_valmerge] TEST_JSONL_SRC = $TEST_JSONL_SRC"
echo "[v2best_valmerge] EPOCHS         = $EPOCHS"
echo "[v2best_valmerge] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# ---------------------------------------------------------------------------
# Step 1: 预检查输入文件
# ---------------------------------------------------------------------------
for f in "$TRAIN_JSONL" "$VAL_JSONL" "$TEST_JSONL_SRC" "$MODEL_DIR" \
         "$FINETUNE_PY" "$DECODE_PY" "$FORMAT_PY" "$EVAL_SH"; do
  if [[ ! -e "$f" ]]; then
    echo "[v2best_valmerge][ERROR] 缺少必需路径：$f" >&2
    exit 1
  fi
done
echo "[v2best_valmerge][Step 1] 输入文件预检查通过："
echo "  train : $(wc -l < "$TRAIN_JSONL") 行"
echo "  val   : $(wc -l < "$VAL_JSONL") 行"
echo "  test  : $(wc -l < "$TEST_JSONL_SRC") 行"

# ---------------------------------------------------------------------------
# Step 2: 训练
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN:-0}" == "1" && -f "$OUTPUT_DIR/model.pth.tar" ]]; then
  echo "[v2best_valmerge][Step 2] SKIP_TRAIN=1，跳过训练，直接进入推理"
else
  echo "[v2best_valmerge][Step 2] 开始训练 → $OUTPUT_DIR"
  cd "$PROJECT_ROOT"

  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  nproc_per_node=${NPROC_PER_NODE:-${#visible_gpus[@]}}

  if [[ "$nproc_per_node" -gt 1 ]]; then
    launcher=("$(dirname "$PYTHON_BIN")/torchrun"
      --nnodes 1
      --node_rank 0
      --nproc_per_node "$nproc_per_node"
      --master_addr "${MASTER_ADDR:-127.0.0.1}"
      --master_port "${MASTER_PORT:-29500}")
  else
    launcher=("$PYTHON_BIN")
  fi

  "${launcher[@]}" "$FINETUNE_PY" \
    --train_jsonl "$TRAIN_JSONL" \
    --val_jsonl "$VAL_JSONL" \
    --pretrained_model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --num_workers "$NUM_WORKERS" \
    --encoder_lr "$ENCODER_LR" \
    --decoder_lr "$DECODER_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --ctc_weight "$CTC_WEIGHT" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --warmup_steps "$WARMUP_STEPS" \
    --grad_clip "$GRAD_CLIP" \
    --max_input_frames "$MAX_INPUT_FRAMES" \
    --max_target_length "$MAX_TARGET_LENGTH" \
    --use_amp "$USE_AMP" \
    --save_optimizer "$SAVE_OPTIMIZER" \
    --seed "$SEED" \
    --log_interval "$LOG_INTERVAL" \
    2>&1 | tee -a "$TRAIN_LOG"

  echo "[v2best_valmerge][Step 2] 训练完成"
fi

# ---------------------------------------------------------------------------
# Step 3: 生成 test_input_converted.jsonl（拼 "language Chinese <accent><asr_text><text>" 前缀）
# ---------------------------------------------------------------------------
echo ""
echo "[v2best_valmerge][Step 3] 生成 test 集推理输入 → $TEST_INPUT_JSONL"
"$PYTHON_BIN" - <<PYCONV
import json, os, sys
src = "$TEST_JSONL_SRC"
dst = "$TEST_INPUT_JSONL"
ASR_MARKER = "<asr_text>"
n = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "audio" in obj:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            continue
        wav_path = obj.get("wav_path")
        if not wav_path:
            raise SystemExit(f"缺少 wav_path/audio: {obj}")
        key = str(obj.get("key") or os.path.splitext(os.path.basename(wav_path))[0])
        text = str(obj.get("text") or "").strip()
        accent = str(obj.get("accent") or "").strip()
        if accent and text:
            full_text = f"language Chinese {accent}{ASR_MARKER}{text}"
        elif text:
            full_text = f"{ASR_MARKER}{text}"
        else:
            full_text = ""
        out = {"audio": wav_path, "text": full_text, "key": key, "accent": accent}
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        n += 1
print(f"[v2best_valmerge][Step 3] 转换 {n} 行 → {dst}")
PYCONV

TEST_LINES=$(wc -l < "$TEST_INPUT_JSONL")
echo "[v2best_valmerge][Step 3] test 集样本数：$TEST_LINES"

# ---------------------------------------------------------------------------
# Step 4: decode（推理 test 集）
# ---------------------------------------------------------------------------
echo ""
echo "[v2best_valmerge][Step 4] 推理 test 集 → $PRED_JSONL"

"$PYTHON_BIN" "$DECODE_PY" \
  --model-dir "$OUTPUT_DIR" \
  --input-jsonl "$TEST_INPUT_JSONL" \
  --output-jsonl "$PRED_JSONL" \
  --gpu-ids all \
  --batch-size "$INFER_BATCH_SIZE" \
  --beam-size "$BEAM_SIZE" \
  --decode-max-len "$DECODE_MAX_LEN" \
  --softmax-smoothing "$SOFTMAX_SMOOTHING" \
  --length-penalty "$LENGTH_PENALTY" \
  --eos-penalty "$EOS_PENALTY" \
  --log-interval 20 \
  --use-half \
  2>&1 | tee -a "$INFER_LOG"

PRED_LINES=$(wc -l < "$PRED_JSONL")
if [[ "$PRED_LINES" -ne "$TEST_LINES" ]]; then
  echo "[v2best_valmerge][Step 4][ERROR] pred 行数=$PRED_LINES 与 test 行数=$TEST_LINES 不一致" >&2
  exit 3
fi
echo "[v2best_valmerge][Step 4] 推理完成，$PRED_LINES 行"

# ---------------------------------------------------------------------------
# Step 5: 格式化 + 官方 CER 评估
# ---------------------------------------------------------------------------
echo ""
echo "[v2best_valmerge][Step 5] 格式化 pred → $PRED_FMT_JSONL"
"$PYTHON_BIN" "$FORMAT_PY" \
  --input  "$PRED_JSONL" \
  --output "$PRED_FMT_JSONL" \
  2>&1 | tee -a "$EVAL_LOG"

echo "[v2best_valmerge][Step 5] 调用官方 CER 评估：$EVAL_SH"
mkdir -p "$WER_DIR"
bash "$EVAL_SH" \
  --pred_jsonl "$PRED_FMT_JSONL" \
  --output_dir "$WER_DIR" \
  --apply_t2s 1 \
  --by_dialect 1 \
  2>&1 | tee -a "$EVAL_LOG"

RESULT_WER=$WER_DIR/result.wer
if [[ ! -f "$RESULT_WER" ]]; then
  echo "[v2best_valmerge][Step 5][ERROR] result.wer 不存在：$RESULT_WER" >&2
  exit 4
fi

# ---------------------------------------------------------------------------
# 结果对比
# ---------------------------------------------------------------------------
NEW_CER=$(grep -Eo "Overall\s*->\s*[0-9.]+" "$RESULT_WER" | tail -1 | awk '{print $NF}')

echo ""
echo "==================== 结果 ===================="
echo "OUTPUT_DIR : $OUTPUT_DIR"
echo "result.wer : $RESULT_WER"
tail -5 "$RESULT_WER" || true
echo "==================== 对比 ===================="
echo "VC_data_v2 sweep 最优 (baseline, epochs=10)   : 11.53 % overall CER"
echo "本次 (相同超参, epochs=8, val=val+test 合并集): ${NEW_CER:-<解析失败>} % overall CER"
echo "=============================================="

echo "[v2best_valmerge] Done."
