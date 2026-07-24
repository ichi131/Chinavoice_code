
# Chinavoice 最终模型说明 (MODELS)

本项目最终提交使用了 **3 个模型 checkpoint**，因为体积较大（合计约 38 GB，仅推理权重约 16 GB），无法直接托管在 GitHub。所有推理权重将上传至 **Hugging Face Hub**，本文档记录每个模型的来源、用途、以及下载/使用方法。

---

## 1. LID (方言识别)

LID 采用 **Qwen3ASR + FireRedLID** 双模型融合的结果，融合策略见后文。

### 1.1 Qwen3ASR (LID 分支)

| 项目 | 值 |
|---|---|
| **原始 ckpt 路径** | `/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/outputs_vc_v2/checkpoint-500` |
| **完整目录大小** | 12 GB |
| **推理必需大小** | ≈ 3.8 GB (`model.safetensors` + 各种 config/tokenizer) |
| **推理脚本** | [`challenge_full_ft/infer_evalset.sh`](challenge_full_ft/infer_evalset.sh) |
| **Hugging Face** | `SerenaWhite/qwen3asr-chinavoices-vc` *(待上传)* |

**运行推理：**
```bash
bash challenge_full_ft/infer_evalset.sh
```

### 1.2 FireRedLID

| 项目 | 值 |
|---|---|
| **原始 ckpt 路径** | `/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/workspace/FireRedASR2S-fintuning/exp/lid_chinavoices_data_speaker_ft_encoder/best.pt` |
| **文件大小** | 8 GB |
| **推理脚本** | [`FireRedASR2S-fintuning/examples_train/lid_chinavoices/run_decode_evaluation_lid_chinavoices.sh`](FireRedASR2S-fintuning/examples_train/lid_chinavoices/run_decode_evaluation_lid_chinavoices.sh) |
| **Hugging Face** | `SerenaWhite/firered-lid-chinavoices` *(待上传)* |

**运行推理：**
```bash
bash FireRedASR2S-fintuning/examples_train/lid_chinavoices/run_decode_evaluation_lid_chinavoices.sh
```

### 1.3 LID 融合（双模型输出合并）

使用 [`analyse/apply_fuse_c_v5.py`](analyse/apply_fuse_c_v5.py) 将 Qwen3ASR 与 FireRedLID 的推理结果按 dev 集上 per-label precision 择优融合：

- 两模型预测一致 → 直接采用
- 不一致 → 各自根据 dev 集上该预测标签的 precision 择优；平局倾向 FR

```bash
python analyse/apply_fuse_c_v5.py
```

输出：`infer_data/mix_v5/lid.jsonl`

---

## 2. ASR

采用 **FireRedASR 微调后的 04_label_smoothing__0.05** 版本。

| 项目 | 值 |
|---|---|
| **原始 ckpt 路径** | `/mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning/exp/asr_sweep_20260721_114230/04_label_smoothing__0.05` |
| **完整目录大小** | 18 GB |
| **推理必需大小** | ≈ 4.4 GB (`model.pth.tar` + config/dict/cmvn 等) |
| **推理脚本** | [`FireRedASR2S-fintuning/examples_train/asr_chinavoices/decode_asr_chinavoices.py`](FireRedASR2S-fintuning/examples_train/asr_chinavoices/decode_asr_chinavoices.py) |
| **后处理脚本** | [`FireRedASR2S-fintuning/exp/asr_chinavoices_vc/convert_pred_to_asr.py`](FireRedASR2S-fintuning/exp/asr_chinavoices_vc/convert_pred_to_asr.py) |
| **Hugging Face** | `SerenaWhite/firered-asr-chinavoices` *(待上传)* |

**运行推理（4 卡半精度）：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python FireRedASR2S-fintuning/examples_train/asr_chinavoices/decode_asr_chinavoices.py \
  --gpu-ids all \
  --batch-size 16 \
  --use-half
```

**推理后处理**（去除 `<sos>/<eos>/<unk>/<pad>` 等特殊 token、按 `eval_XXXXXX` 排序、转成竞赛提交格式 `asr.jsonl`）：
```bash
python FireRedASR2S-fintuning/exp/asr_chinavoices_vc/convert_pred_to_asr.py
```

---

## 3. 完整推理流水线

```
┌──────────────────────────────────────────────────────┐
│                      音频 (evaluation)               │
└──────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  ┌──────────┐      ┌───────────┐     ┌──────────┐
  │ Qwen3ASR │      │ FireRedLID│     │FireRedASR│
  │   LID    │      │           │     │          │
  └──────────┘      └───────────┘     └──────────┘
        │                 │                 │
        └────────┬────────┘                 │
                 ▼                          ▼
         apply_fuse_c_v5.py         convert_pred_to_asr.py
                 │                          │
                 ▼                          ▼
              lid.jsonl                  asr.jsonl
```

---

## 4. Checkpoint 下载 *(Hugging Face 上传后补充)*

上传完成后，本节将提供 `huggingface-cli` 或 `snapshot_download` 的一键下载命令。计划的目录组织如下：

```
Chinavoice_code/
├── challenge_full_ft/outputs_vc_v2/checkpoint-500/     ← 从 SerenaWhite/qwen3asr-chinavoices-vc 下载
├── FireRedASR2S-fintuning/
│   └── exp/
│       ├── lid_chinavoices_data_speaker_ft_encoder/    ← 从 SerenaWhite/firered-lid-chinavoices 下载
│       └── asr_sweep_20260721_114230/04_label_smoothing__0.05/  ← 从 SerenaWhite/firered-asr-chinavoices 下载
```

> 上传策略：**只上传推理必需的权重**，不包含 `optimizer.pt` / `last.pt` / `rng_state_*.pth` 等训练态文件。
