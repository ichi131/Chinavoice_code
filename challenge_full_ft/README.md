# challenge_full_ft

Qwen3-ASR-1.7B 在 `challenge_data_speaker` 全方言数据上的**全参数微调**
（Full-Parameter Fine-tuning，**不使用 LoRA**）工程目录。

- 数据处理与目标文本格式与 [`ChinaVoices-Challenge`](/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/ChinaVoices-Challenge) baseline 对齐：`language Chinese <accent><asr_text><原始 text>`。
- 训练脚本以官方 `Qwen3-ASR/finetuning/qwen3_asr_sft.py` 为基础做**薄封装**，新增两项能力：
  1. `load_best_model_at_end=True`（按 `eval_loss` 挑最优 ckpt 并在训练收尾时加载）。
  2. `EarlyStoppingCallback`：`eval_loss` 连续 `patience` 次未改善即早停，`patience` 默认 **3**（可关闭）。
- 推理与 CER 评估复用 `ChinaVoices-Challenge/infer/infer_batch.py` 的解析规则和 `eval/eval_jsonl_with_wer_tools.sh` 的评测入口。

> ⚠️ 所有 GPU/推理相关命令都仅落地为脚本，**不要在当前 IDE 机器执行**。请把该目录拷到实验机再运行。

---

## 目录结构

```
challenge_full_ft/
├── README.md                # 使用说明（本文件）
├── prepare_data.py          # 步骤 1：数据准备（可配置 src_dir / 文件名）
├── qwen3_asr_sft_full.py    # 步骤 2：官方 finetune 的薄封装，加入 load_best + EarlyStopping
├── train_full_ft.sh         # 步骤 2：训练入口 shell（torchrun / python）
├── pick_best_ckpt.py        # 步骤 3：训练结束后挑最佳 ckpt
├── infer_test.py            # 步骤 3：test 集批量推理
├── run_eval.sh              # 步骤 4：一键 挑 ckpt → 推理 → CER 评估
├── data/                    # prepare_data.py 输出：train.jsonl / val.jsonl / test.jsonl
└── outputs/                 # 训练/推理/评估产物
    ├── checkpoint-*/
    ├── best_ckpt.txt        # 由 qwen3_asr_sft_full.py 写入的最佳 ckpt 绝对路径
    ├── pred_test.jsonl
    └── wer_eval/
        ├── result.wer              # 整体 CER
        ├── by_dialect_summary.txt  # 按方言 CER 汇总
        └── dialect_accuracy.txt    # 方言识别准确率
```

---

## 4 步使用流程

### Step 1 · 数据准备

把 `challenge_data_speaker` 三份 jsonl 转为 SFT 训练格式（`audio / text / prompt / key / accent`）：

```bash
python challenge_full_ft/prepare_data.py \
    --src_dir /mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_ichiwang/data/challenge_data_speaker \
    --out_dir /mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data \
    --check_audio_exists 0
```

数据文件名可通过 `--train_name / --val_name / --test_name` 覆盖。后期若要替换数据，只需改 CLI 参数，脚本内**没有任何硬编码路径**。

### Step 2 · 全参数微调训练（支持多 epoch + EarlyStopping）

```bash
NUM_GPUS=8 \
MODEL_PATH=/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B \
EPOCHS=5 \
BATCH_SIZE=8 GRAD_ACC=4 LR=2e-5 \
SAVE_STEPS=50 SAVE_TOTAL_LIMIT=2 \
EARLY_STOP_PATIENCE=3 \
EARLY_STOP_THRESHOLD=0.0 \
bash challenge_full_ft/train_full_ft.sh
```

要点：
- 默认 `SAVE_STEPS=50 / SAVE_TOTAL_LIMIT=2`：当前训练集约 3.4w 条、8 卡全局 batch=256 时 1 epoch ≈ 134 步，5 epoch 会产生约 13 次 eval；只保留"最佳 + 最近一次"两份 ckpt。换数据/换 batch 后可命令行覆盖。
- `EARLY_STOP_PATIENCE=3` 意味着 `eval_loss` 连续 3 次 eval 未下降即早停；设为 `0` 或负数可关闭早停。
- `load_best_model_at_end=True` + `metric_for_best_model=eval_loss`，训练结束时主模型权重会自动回到最佳 ckpt。
- 训练脚本收尾时会把最佳 ckpt 路径写到 `outputs/best_ckpt.txt`，供后续步骤直接引用。
- **不使用 LoRA/PEFT**，脚本里没有任何 adapter/lora 参数。

### Step 3 · 挑最佳 ckpt & 在 test 集上推理（也可 Step 4 一键做）

单独调试用：

```bash
# 3.1 挑最佳 ckpt（仅打印一行绝对路径到 stdout）
BEST_CKPT=$(python challenge_full_ft/pick_best_ckpt.py \
    --output_dir challenge_full_ft/outputs \
    --metric eval_loss --greater_is_better 0)
echo "best ckpt: ${BEST_CKPT}"

# 3.2 用最佳 ckpt 在 test 集上批量推理
python challenge_full_ft/infer_test.py \
    --model "${BEST_CKPT}" \
    --data  challenge_full_ft/data/test.jsonl \
    --output challenge_full_ft/outputs/pred_test.jsonl \
    --batch-size 32 --max-tokens 512 --device-map auto
```

### Step 4 · 一键：挑 ckpt → 推理 → CER 评估

```bash
OUTPUT_DIR=challenge_full_ft/outputs \
DATA_TEST=challenge_full_ft/data/test.jsonl \
EVAL_TOOL_SH=/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh \
bash challenge_full_ft/run_eval.sh
```

产物：

| 文件 | 内容 |
| --- | --- |
| `outputs/pred_test.jsonl` | 推理 JSONL（含 `utt_id / audio_path / ref_full / ref_text / ref_dialect / pred_full / pred_text / pred_dialect / error`） |
| `outputs/wer_eval/result.wer` | **整体 CER** |
| `outputs/wer_eval/by_dialect_summary.txt` | **按方言 CER 汇总** |
| `outputs/wer_eval/dialect_accuracy.txt` | 方言识别准确率 |

如要手动指定推理用的 ckpt，可以 `MODEL_CKPT=/abs/path/to/checkpoint-xxxx bash challenge_full_ft/run_eval.sh` 跳过挑选步骤。

### 多卡并行推理

`run_eval.sh` 会**自动检测可用 GPU 数**：>=2 张时走 `run_infer_multi_gpu.sh`（多进程数据并行），=1 张时走单卡 `infer_test.py`。也可显式指定：

```bash
# 显式指定 8 卡
NUM_GPUS=8 bash challenge_full_ft/run_eval.sh

# 只用单卡（如调试）
NUM_GPUS=1 bash challenge_full_ft/run_eval.sh
```

也可以脱离 `run_eval.sh`，单独跑多卡推理：

```bash
MODEL_CKPT=/abs/path/to/checkpoint-xxxx \
NUM_GPUS=8 BATCH_SIZE=32 \
bash challenge_full_ft/run_infer_multi_gpu.sh
```

原理：每张 GPU 起一个 `infer_test.py` 子进程，`CUDA_VISIBLE_DEVICES=rank` 独占一张卡 + 进程内 `device_map=cuda:0`，按 `idx % world_size == rank` 做 stride 切分，输出到 `pred_test.rank{N}.jsonl`，最后合并成 `pred_test.jsonl`。日志在 `challenge_full_ft/infer_logs/infer_rank*.log`。

**中间产物清理**（在多卡脚本 `run_infer_multi_gpu.sh` 里）：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `CLEAN_SHARDS` | `1` | 合并成功后自动删除 `pred_test.rank*.jsonl` 分片文件 |
| `CLEAN_LOGS`   | `0` | 保留 `infer_logs/infer_rank*.log`；设为 `1` 才一起删 |

清理仅在"合并结果行数 > 0"时执行，跑挂时**不会**误删分片和日志，方便排错。要保留分片自查：`CLEAN_SHARDS=0 bash challenge_full_ft/run_infer_multi_gpu.sh`。

---

## Troubleshooting

### 推理时报 `OSError: Can't load feature extractor for '.../checkpoint-xxxx'`

**症状**：`AutoProcessor.from_pretrained` 报 `preprocessor_config.json` 缺失。

**原因**：训练时 `MODEL_PATH` 传了 HF Hub ID（如 `Qwen/Qwen3-ASR-1.7B`）而非本地目录，导致 `MakeEveryCheckpointInferableCallback` 在拷贝 processor 相关文件时静默跳过（内部逻辑是 `if os.path.exists(src)`）。

**修复**：用 `fix_ckpt_processor.py` 从 base 模型目录**补齐**缺失文件（只补不覆盖）。

```bash
# 批量修复 outputs/ 下所有 checkpoint-*
python challenge_full_ft/fix_ckpt_processor.py \
    --base_model /mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B \
    --output_dir challenge_full_ft/outputs

# 或只修单个 ckpt
python challenge_full_ft/fix_ckpt_processor.py \
    --base_model /mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B \
    --ckpt       challenge_full_ft/outputs/checkpoint-400
```

**根治**：`train_full_ft.sh` 的默认 `MODEL_PATH` 已经指向本地绝对路径；`qwen3_asr_sft_full.py` 也会在训练时把 HF Hub ID 自动通过 `snapshot_download` 解析成本地路径，因此**新训练不会再出现此问题**。

### 推理时报 `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!`

**原因**：给 `AutoModel.from_pretrained` 传了 `device_map="auto"`。这个模式是**模型并行**——Accelerate 会把 Qwen3-ASR 的层拆到多张卡上，但 audio encoder 分支的 hook 处理不完善，前向时会撞出跨卡张量。它**不是数据并行**，8 卡也帮不上你的推理吞吐。

**修复**：`infer_test.py` / `run_eval.sh` / `run_infer_multi_gpu.sh` 的默认 `DEVICE_MAP` 已改为 `cuda:0`，单进程只用单卡；多卡加速通过 `run_infer_multi_gpu.sh` 起多进程数据并行实现（见上面「多卡并行推理」小节）。

---

## 数据文件后期替换指南

所有脚本都通过 CLI 或 shell 变量暴露数据路径，**没有硬编码**：

- **prepare_data.py**: `--src_dir / --train_name / --val_name / --test_name / --out_dir`。
- **train_full_ft.sh**: 环境变量 `TRAIN_FILE / EVAL_FILE / OUTPUT_DIR`。
- **infer_test.py**: `--data / --output / --model`。
- **run_eval.sh**: 环境变量 `DATA_TEST / PRED_JSONL / OUTPUT_DIR / MODEL_CKPT / EVAL_TOOL_SH`。

只要更新对应参数即可，无需改代码。

---

## 参考资源

- 官方全参微调脚本：`/mnt/geminihzceph/user_johannapeng/challenge_model/Qwen3-ASR/finetuning/qwen3_asr_sft.py`（`qwen3_asr_sft_full.py` 通过 `sys.path` 复用其组件；如需迁移，可用环境变量 `QWEN3_ASR_FT_DIR` 覆盖官方目录路径）。
- ChinaVoices baseline 数据格式与推理脚本：`/mnt/wfs/.../ChinaVoices-Challenge/infer/infer_batch.py`。
- ChinaVoices baseline CER/WER 评测入口：`/mnt/wfs/.../ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh`。
