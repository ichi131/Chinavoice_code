# LID 超参贪心式扫描（sweep）

自动化的 LID（16 类中文方言识别）微调超参寻优工具。按用户配置的维度顺序依次扫描，每个维度扫完后选出 test 集 LID 准确率最高的取值作为固化基线，进入下一维度。

> **本工具与原有训练/推理流程物理隔离。** 所有新增代码位于本目录（`examples_train/lid_chinavoices/sweep/`），所有实验产物位于独立顶层目录 `exp/lid_sweep_<timestamp>/`。删除本目录不会影响任何现有的手动运行入口。

---

## 目录结构

```
examples_train/lid_chinavoices/sweep/
├── README.md                        # 本文件
├── run_sweep_lid_chinavoices.sh     # 用户入口 shell（bash 启动）
├── sweep_main.py                    # 主控循环
├── runner.py                        # 单次实验的训练/推理封装
├── config_loader.py                 # YAML 加载与校验
└── configs/
    ├── sweep_example.yaml           # 完整示例配置（7 维度，每维 3 候选）
    └── sweep_dryrun.yaml            # 轻量 dry-run 配置（每维 1 候选，epochs=1）
```

---

## 快速上手

### 1. 完整扫描（推荐）

```bash
# 使用 0-7 号 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples_train/lid_chinavoices/sweep/run_sweep_lid_chinavoices.sh
```

默认会：

- 读取 `configs/sweep_example.yaml`
- 在 `exp/lid_sweep_<YYYYmmdd_HHMMSS>/` 下新建扫描目录
- 依次扫描 `lr → encoder_lr → batch_size → dropout → weight_decay → label_smoothing → seed`（每维 3 候选，共 21 次实验）
- 每次实验 `epochs=10, patience=2`，加权早停

### 2. 使用自定义配置

```bash
SWEEP_CONFIG=/path/to/my_sweep.yaml \
  bash examples_train/lid_chinavoices/sweep/run_sweep_lid_chinavoices.sh
```

### 3. 指定输出目录

```bash
SWEEP_DIR=./exp/my_experiment_run \
  bash examples_train/lid_chinavoices/sweep/run_sweep_lid_chinavoices.sh
```

### 4. 断点恢复

```bash
RESUME=1 \
SWEEP_DIR=./exp/lid_sweep_20260721_170000 \
  bash examples_train/lid_chinavoices/sweep/run_sweep_lid_chinavoices.sh
```

`--resume` 会跳过所有已产出 `test_accuracy.json (status=success)` 的实验，直接读取其准确率参与贪心决策。

### 5. dry-run 快速验证流水线

```bash
CUDA_VISIBLE_DEVICES=0 \
SWEEP_CONFIG=examples_train/lid_chinavoices/sweep/configs/sweep_dryrun.yaml \
  bash examples_train/lid_chinavoices/sweep/run_sweep_lid_chinavoices.sh
```

每维仅 1 个候选、`epochs=1, patience=1`，用于最小成本地验证流水线是否可跑通。

---

## 配置文件格式

见 `configs/sweep_example.yaml`。核心字段：

| 段 | 用途 |
|---|---|
| `fixed_params` | 所有实验共用的路径参数：`train_jsonl`、`val_jsonl`、`test_jsonl`、`pretrained_model_dir`、以及可选的 `num_workers`、`use_amp`、`log_interval` |
| `baseline` | 所有维度扫描前的"起点"超参；随扫描逐维度被固化更新 |
| `sweep_order` | 扫描维度顺序（列表）；每项必须来自 `sweep_space` 键集合 |
| `sweep_space` | 每个维度的候选值列表；候选数为 1 时会跳过扫描直接固化 |
| `run_options` | 顶层扫描目录前缀、是否清理非最优 checkpoint 等运行时选项 |

**允许扫描的超参白名单**（在 `config_loader.py::ALLOWED_HPARAMS`）：

```
lr, encoder_lr, batch_size, dropout, weight_decay, label_smoothing,
seed, grad_clip, warmup_steps, epochs, patience, min_delta, freeze_encoder
```

引入白名单外的维度会立即报错，不启动任何训练。

---

## 输出目录结构

```
exp/lid_sweep_<timestamp>/
├── config_snapshot.json             # 配置快照（可复现）
├── sweep_progress.json              # 实时进度（每次实验后原子更新）
├── sweep_summary.md                 # 全部完成后生成的表格总览
├── best_hparams.json                # 最终固化的全局最优基线
├── 01_lr__0.0005/                   # 第 1 维第 1 个候选
│   ├── config.json                  # 本次实验实际超参
│   ├── train.log                    # 训练日志（stdout+stderr）
│   ├── best.pt                      # 训练产出的最优 checkpoint（保留策略见下）
│   ├── last.pt                      # 最新 checkpoint（保留策略见下）
│   ├── labels.json                  # 标签映射
│   ├── val_pred_best.jsonl          # 验证集最优预测
│   ├── infer.log                    # 推理日志
│   ├── pred_test.jsonl              # test 集预测输出
│   └── test_accuracy.json           # LID 准确率评估结果（贪心决策依据）
├── 01_lr__0.001/ ...
├── 01_lr__0.002/ ...
├── 02_encoder_lr__5e-06/ ...
└── ...
```

**Checkpoint 保留策略**（受 `run_options.keep_only_best_ckpt` 控制）：

- `true`（默认）：每维扫描完成后仅保留该维度最优实验的 `best.pt`，其余实验的 `best.pt` / `last.pt` 被删除。所有文本产物（日志、预测、准确率）全部保留。
- `false`：所有实验的 checkpoint 全部保留（磁盘占用大，21 次实验约 6-8 GB × 每 ckpt ~300MB）。

---

## 贪心决策规则

- 每个维度扫描完，选出 `overall_acc` **最大**的候选作为该维度最优值，并固化到当前 baseline 中。
- 若该维度全部候选均训练/推理失败（`overall_acc = null`），baseline 保持不变继续下一维度。
- 候选值长度为 1 时不训练，直接固化。
- `overall_acc < min_acc_warning_threshold`（默认 1/16 = 0.0625）时打印 warning，但流程继续。

---

## 常见问题

**Q: 训练进程报 IPv6 网络地址解析失败并卡住？**
A: 与手动运行 `run_finetune_lid_chinavoices.sh` 相同的问题，将当前 hostname 加入 `/etc/hosts`：
```bash
echo "127.0.0.1 $(hostname)" | sudo tee -a /etc/hosts
```

**Q: 某个实验准确率异常低（如 < 0.1）？**
A: 通常是训练不收敛或推理出错。查看该实验目录下的 `train.log` / `infer.log` 排查。若确认异常，直接删除该实验目录，`--resume` 时会自动重跑。

**Q: 中断后如何恢复？**
A: 找到扫描目录路径（`exp/lid_sweep_<ts>/`），用 `SWEEP_DIR=<path> RESUME=1` 重新启动同一 shell 入口即可。已产出 `test_accuracy.json (status=success)` 的实验会被跳过。

**Q: 推理只用 1 卡够吗？**
A: 是的。推理已强制单卡执行（避免误触 DDP 推理引入不必要复杂度），若外层 `CUDA_VISIBLE_DEVICES` 设置了多卡，会自动取第一张卡。

**Q: 如何只测 1-2 个维度？**
A: 修改配置的 `sweep_order`，只保留要测的维度名即可；未列入 `sweep_order` 的维度会保持 baseline 值不参与扫描。

---

## 隔离性说明

**本功能不修改以下现有文件**（应受保护，`git status` 不应看到它们的 diff）：

- `examples_train/lid_chinavoices/finetune_lid_chinavoices.py`
- `examples_train/lid_chinavoices/infer_lid_chinavoices.py`
- `examples_train/lid_chinavoices/run_finetune_lid_chinavoices.sh`
- `examples_train/lid_chinavoices/run_decode_test_lid_chinavoices.sh`
- `examples_train/lid_chinavoices/run_decode_evaluation_lid_chinavoices.sh`

**本功能新增的文件**（`git status` 应仅看到这些）：

```
examples_train/lid_chinavoices/sweep/README.md
examples_train/lid_chinavoices/sweep/run_sweep_lid_chinavoices.sh
examples_train/lid_chinavoices/sweep/sweep_main.py
examples_train/lid_chinavoices/sweep/runner.py
examples_train/lid_chinavoices/sweep/config_loader.py
examples_train/lid_chinavoices/sweep/configs/sweep_example.yaml
examples_train/lid_chinavoices/sweep/configs/sweep_dryrun.yaml
```

**运行时新增目录**：`exp/lid_sweep_<timestamp>/`（可任意删除、不影响原有实验目录）。

如需完全卸载扫描功能：直接删除 `examples_train/lid_chinavoices/sweep/` 与 `exp/lid_sweep_*/`，其余项目文件不受任何影响。
