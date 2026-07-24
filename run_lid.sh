# step1 firelid解码
host_project_root=/n/work6/yiwang/Chinavoice_code/FireRedASR2S-fintuning
host_data_root=/n/work6/yiwang/chinavoices_challenge/evaluation_set
host_checkpoint=/n/work6/yiwang/Chinavoice_code/FireRedASR2S-fintuning/models/best.pt

sudo docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$host_project_root:/workspace/FireRedASR2S-fintuning" \
  -v "$host_data_root:/data/evaluation_set:ro" \
  -v "$host_checkpoint:/models/checkpoint.pt:ro" \
  -e FIREREDASR2S_ROOT=/workspace/FireRedASR2S-fintuning \
  -e GPU_ID=0 \
  -e CHECKPOINT=/models/checkpoint.pt \
  -e WAV_SCP=/workspace/FireRedASR2S-fintuning/wav.scp \
  -e DATA_ROOT=/data/evaluation_set \
  -e EXP_DIR=/workspace/FireRedASR2S-fintuning/exp/lid_output \
  ghcr.io/ichi131/fireredasr2s:h20-cu124 \
  bash /workspace/FireRedASR2S-fintuning/examples_train/lid_chinavoices/run_decode_evaluation_lid_chinavoices.sh

# step2 qwen3asr-lid解码
PROJECT_ROOT=/n/work6/yiwang/Chinavoice_code
MODEL_CKPT_DIR=/n/work6/yiwang/Chinavoice_code/FireRedASR2S-fintuning/models/pretrained_models/qwen3asr-chinavoices-vc
EVALUATION_SET_DIR=/n/work6/yiwang/chinavoices_challenge/evaluation_set

sudo docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  --shm-size=16g \
  -v "$PROJECT_ROOT:/workspace/Chinavoice_code" \
  -v "$MODEL_CKPT_DIR:/models/checkpoint:ro" \
  -v "$EVALUATION_SET_DIR:/data/chinavoices_challenge/evaluation_set:ro" \
  -e MODEL_CKPT=/models/checkpoint \
  -e EVAL_SCP=/data/chinavoices_challenge/evaluation_set/wav.scp \
  ghcr.io/ichi131/qwen3asr:py312-torch2.6-cu124 \
  bash /workspace/Chinavoice_code/challenge_full_ft/infer_evalset.sh

# step3 融合 step1(FireRedLID) / step2(Qwen3ASR-LID) 两个解码 jsonl 的结果
# apply_fuse_c_v5.py 默认按 REPO_ROOT 推导 FR_INFER/QW_INFER，正好对应
# step1 的 evaluation_pred.jsonl 和 step2 的 result.jsonl，无需额外传参；
# 融合结果写到 $PROJECT_ROOT/infer_data/lid.jsonl。
python $PROJECT_ROOT/analyse/apply_fuse_c_v5.py