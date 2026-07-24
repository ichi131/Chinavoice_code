# ASR 超参贪心式扫描（Coordinate Descent Sweep）

## 一、这是什么

针对 `examples_train/asr_chinavoices/finetune_asr_chinavoices.py` 的 ASR 微调任务，构建了一套**贪心式超参扫描**流水线：
- 逐维度扫描：`encoder_lr → decoder_lr → ctc_weight → label_smoothing → warmup_steps`
- 每维度选出**外部 test overall CER 最小**的候选值固化为新 baseline，进入下一维度
- 每次实验会**训练 → 推理 test 集 → 官方 CER 评估**，用 `result.wer` 里的 `Overall ->` 作为选优字段
- 24h 预算下最坏 10 次训练（每维度候选都包含当前 baseline，触发**方案 C 复用**后最好只需 6 次）

**所有产物写入用户目录** `/mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning/exp/asr_sweep_*`，**严禁**写到 `user_ichiwang` 或其他非用户目录。

## 二、目录结构

```
examples_train/asr_chinavoices/sweep/
  ├── __init__.py
  ├── config_loader.py           # YAML 加载 + 预检
  ├── runner.py                  # 训练/推理/评估的 subprocess 封装
  ├── sweep_main.py              # 贪心主控（写 progress / summary / best_hparams）
  ├── run_sweep_asr_chinavoices.sh  # 入口 shell
  ├── configs/
  │     ├── sweep_example.yaml   # 5 维 × 2 候选（最好 6 次 / 最坏 10 次）
  │     └── sweep_dryrun.yaml    # 单维度 × 2 候选 + epochs=1，用于冒烟测试
  └── README.md                  # 本文档
```

扫描过程中会在 `<sweep_dir>/` 下生成：

```
asr_sweep_YYYYmmdd_HHMMSS/
  ├── config_snapshot.json
  ├── test_input_converted.jsonl   # 由 test_jsonl 字段转换而来（推理输入）
  ├── sweep_progress.json          # 进度快照，供 RESUME=1 使用
  ├── sweep_console.log            # 每次实验的一行简报
  ├── sweep_summary.md             # 全部完成后的人类可读汇总
  ├── best_hparams.json            # 最终最佳超参组合
  └── <idx>_<dim>__<value>/
        ├── status.json                # {status, elapsed_s, failed_stage, error, ...}
        ├── train_config.json          # 本次实验的完整 hparams
        ├── train.log / infer.log / eval.log
        ├── model.pth.tar              # best 权重（若被 keep_only_best_ckpt 删除则不存在）
        ├── best_metrics.json          # 训练内 val 指标（含 val macro_cer）
        ├── accents.json               # 训练脚本导出的方言列表
        ├── pred_test.jsonl            # 推理产出
        ├── pred_test_formatted.jsonl  # 9 字段格式化产物
        ├── wer_eval/                  # 官方评估产物
        │     ├── result.wer
        │     ├── by_dialect_summary.txt
        │     ├── dialect_accuracy.txt
        │     └── by_dialect/<dialect>/result.wer
        ├── test_metrics.json          # 从 result.wer 解析出的选优摘要
        └── reused_from.json           # 命中方案 C 复用时的来源记录（可选）
```

## 三、启动方式（tmux 长跑）

```bash
# 1) 新开 tmux 会话（脱离终端后仍继续跑）
tmux new -s asr_sweep

# 2) 在 tmux 内执行扫描
cd /mnt/geminihzceph/user_johannapeng/challenge_model/FireRedASR2S-fintuning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples_train/asr_chinavoices/sweep/run_sweep_asr_chinavoices.sh

# 3) 断开 tmux（不会杀掉扫描进程）
#    按下 Ctrl-B 然后按 D

# 4) 重新连回
tmux attach -t asr_sweep
```

**监控进度**：
```bash
tail -f exp/asr_sweep_*/sweep_console.log
tail -f exp/asr_sweep_*/<idx>_<dim>__<value>/train.log
```

## 四、冒烟测试（先跑 dryrun）

正式开跑前，用 `sweep_dryrun.yaml`（单维度 × 2 候选，epochs=1）验证整条流水线：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SWEEP_CONFIG=examples_train/asr_chinavoices/sweep/configs/sweep_dryrun.yaml \
  bash examples_train/asr_chinavoices/sweep/run_sweep_asr_chinavoices.sh
```

预期用时 30-60 分钟（单 epoch × 2 次训练 + 2 次推理 + 2 次评估）。检查：
- `exp/asr_sweep_dryrun_*/sweep_summary.md` 生成
- `<exp_dir>/test_metrics.json` 里的 `overall_cer` 与 `wer_eval/result.wer` 的 `Overall ->` 一致

## 五、断点恢复

如果扫描因为 OOM / 机器抖动 / Ctrl-C 中断，重新启动只需带上 `RESUME=1` 与之前的目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SWEEP_DIR=exp/asr_sweep_YYYYmmdd_HHMMSS \
RESUME=1 \
  bash examples_train/asr_chinavoices/sweep/run_sweep_asr_chinavoices.sh
```

- 会跳过 `status=success/reused` 的实验
- 若某实验的 `pred_test.jsonl` 存在但 `pred_test_formatted.jsonl` 缺失，会从格式化步骤重启（不重推理）
- 若 `wer_eval/result.wer` 存在但 `test_metrics.json` 缺失，会仅重新解析
- **手动删除某实验目录 → 会被重跑**

## 六、扫描空间说明（`configs/sweep_example.yaml`）

| 维度 | Baseline | 候选值 | 意图 |
|---|---|---|---|
| `encoder_lr` | 5e-6 | `[5e-6, 1e-5]` | encoder 是否需要更快更新 |
| `decoder_lr` | 1e-5 | `[1e-5, 3e-5]` | decoder 是否需要更强适配 |
| `ctc_weight` | 0.3 | `[0.3, 0.5]` | CTC 分支是否应更强 |
| `label_smoothing` | 0.1 | `[0.05, 0.1]` | 更轻的正则是否更利于收敛 |
| `warmup_steps` | 1000 | `[1000, 2000]` | 更长 warmup 是否更稳 |

每维度候选都包含当前 baseline 值 → 从第 2 维度起可复用第 1 维度 winner，**最好 6 次训练 / 最坏 10 次训练**。

**保持不变的超参**：`epochs=10`、`batch_size=4`、`grad_accum_steps=8`、`num_workers=4`、`weight_decay=1e-2`、`grad_clip=5.0`、`max_input_frames=6000`、`max_target_length=256`、`use_amp=1`、`save_optimizer=1`、`seed=1337`。

## 七、常见问题排查

### 1. `dist.destroy_process_group()` 报 `unhandled cuda error / out of memory`
LID 训练也遇到过这个"训练已成功但结束时 NCCL 报错"。sweep runner **不依赖 return code 判定成败**，而是检查 `model.pth.tar` / `best_metrics.json` / `accents.json` 是否齐备，齐备就视为成功。看到这个 warning 不用理会。

### 2. `hostname: node-X: Name or service not known`
新机器缺失 `hosts` 条目：
```bash
echo "127.0.0.1 $(hostname)" | sudo tee -a /etc/hosts
```

### 3. 推理阶段 OOM
减小推理 `batch_size`（改配置 `infer_args.batch_size`）或关闭 `use_half`（虽然会更慢）。

### 4. `pred_test.jsonl 行数与 test 集不一致`
说明推理没跑完；删掉对应实验目录，`RESUME=1` 重启会自动重跑。

### 5. `result.wer 中未找到 'Overall ->' 行`
说明官方评估工具异常退出，查看 `<exp_dir>/eval.log`。常见原因：`opencc` 未安装 / `tools/wer.py` 依赖缺失。

### 6. 主控进程被误杀但训练子进程仍在跑
`ps aux | grep torchrun`，若确认在跑，等它跑完；再 `RESUME=1` 恢复即可（sweep_main 会看到 `model.pth.tar` 已产出，判为 success）。

## 八、24h 预算下的时间预算参考

单次 ASR 全量微调（10 epoch × VC 数据 × 8 卡）约 **90-120 分钟**；
每次外部 test 推理 + 官方评估约 **10-20 分钟**。

- **最好情况**（每维 winner 都保持 baseline，触发 4 次复用）：约 6 × 120 = **12 小时**
- **最坏情况**（每维 winner 都切到新值，无任何复用）：约 10 × 120 = **20 小时**

若接近 24h 上限，及时 tmux 内看进度：
```bash
tail -n 30 exp/asr_sweep_*/sweep_console.log
```

## 九、修改扫描空间

只需编辑 `configs/sweep_example.yaml`，无需改代码。约束：
1. `sweep_order` 每项必须是 `baseline_hparams` 的合法键
2. 每个 `sweep_space[dim]` 必须**包含 `baseline_hparams[dim]` 的取值**（否则贪心链断裂）
3. 若某维度只想固化不扫描，把候选列表设成单元素（例如 `warmup_steps: [1000]`），sweep_main 会自动跳过训练
