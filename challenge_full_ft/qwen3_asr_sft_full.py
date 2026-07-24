#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qwen3_asr_sft_full.py
=====================

`Qwen3-ASR/finetuning/qwen3_asr_sft.py` 的**薄封装**版本，用于在
`challenge_data_speaker` 上进行**全参数微调**（Full-Parameter FT，**不使用 LoRA**）。

与官方脚本的差异（仅在此处新增/覆盖，不改官方文件）：
1. `TrainingArguments` 中强制启用：
   - ``load_best_model_at_end=True``
   - ``metric_for_best_model="eval_loss"``
   - ``greater_is_better=False``
   - ``eval_strategy="steps"`` 且 ``eval_steps == save_steps``（保证按同一步触发）
   - 默认较小的 ``save_total_limit``（默认 2，仅保留"最佳 + 最近一次"）
2. 新增 CLI 参数：
   - ``--early_stopping_patience``（默认 3）
   - ``--early_stopping_threshold``（默认 0.0）
   - ``patience > 0`` 时注册 :class:`transformers.EarlyStoppingCallback`；
     ``patience <= 0`` 显式禁用早停。
3. 显式重申：**不启用任何 PEFT/LoRA 分支**（本文件不引入 peft 依赖，
   不引入任何 adapter/lora 参数）。

保留完全一致的：
- ``patch_outer_forward``、``load_audio``、``build_prefix_messages``、
  ``make_preprocess_fn_prefix_only``、``DataCollatorForQwen3ASRFinetuning``、
  ``CastFloatInputsTrainer``、``MakeEveryCheckpointInferableCallback``、
  ``find_latest_checkpoint``、``copy_required_hf_files_for_qwen_asr``。

这些组件通过将官方 ``finetuning`` 目录追加到 ``sys.path`` 后直接 import 复用，
从而做到"薄封装"、避免代码漂移。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 让本文件能直接找到并复用官方 qwen3_asr_sft.py 中的类/函数。
# 官方脚本位置：/mnt/geminihzceph/user_johannapeng/challenge_model/Qwen3-ASR/finetuning/
# 允许通过环境变量 QWEN3_ASR_FT_DIR 覆盖，方便迁移到其他机器。
# ---------------------------------------------------------------------------
_DEFAULT_OFFICIAL_FT_DIR = os.environ.get(
    "QWEN3_ASR_FT_DIR",
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "Qwen3-ASR",
            "finetuning",
        )
    ),
)
if _DEFAULT_OFFICIAL_FT_DIR not in sys.path:
    sys.path.insert(0, _DEFAULT_OFFICIAL_FT_DIR)

# 从官方脚本模块中复用工具类/函数
from qwen3_asr_sft import (  # noqa: E402  (delayed import by design)
    CastFloatInputsTrainer,
    DataCollatorForQwen3ASRFinetuning,
    MakeEveryCheckpointInferableCallback,
    find_latest_checkpoint,
    make_preprocess_fn_prefix_only,
    patch_outer_forward,
)
from qwen_asr import Qwen3ASRModel  # noqa: E402
from transformers import (  # noqa: E402
    EarlyStoppingCallback,
    GenerationConfig,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Qwen3-ASR Full-Parameter Finetuning (challenge_data_speaker)"
    )

    # Paths
    p.add_argument("--model_path", type=str,
                   default="/mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B",
                   help="Qwen3-ASR base 模型：**建议**用本地绝对路径；"
                        "也可传 HF Hub ID（如 Qwen/Qwen3-ASR-1.7B），"
                        "脚本内部会用 snapshot_download 解析成本地路径。")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file",  type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc",   type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--epochs",     type=float, default=3)
    p.add_argument("--log_steps",  type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio",      type=float, default=0.03)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory",  type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor",    type=int, default=2)

    # Save / Eval
    p.add_argument("--save_steps",       type=int, default=50)
    p.add_argument("--eval_steps",       type=int, default=0,
                   help="0 表示与 save_steps 对齐；>0 则单独设置")
    p.add_argument("--save_total_limit", type=int, default=2)

    # 精度
    p.add_argument("--bf16", type=int, default=1,
                   help="1=开启 bf16（默认，若硬件支持）；0=关闭，此时回退 fp16")

    # 早停
    p.add_argument("--early_stopping_patience",  type=int,   default=3,
                   help="连续 N 次 eval 未改善则早停；<=0 显式禁用")
    p.add_argument("--early_stopping_threshold", type=float, default=0.0,
                   help="eval_loss 至少下降多少才算改善（默认 0.0）")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume",      type=int, default=0)

    # 随机种子（HPO 搜索时用）
    p.add_argument("--seed", type=int, default=42,
                   help="全局随机种子，同时用于 TrainingArguments.seed / data_seed；"
                        "训练入口调用 transformers.set_seed 生效")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _resolve_base_model_local_dir(model_path: str) -> str:
    """把 ``--model_path`` 解析为**本地目录路径**，供 processor 文件拷贝使用。

    规则：
    1) 若已是本地目录且包含 ``preprocessor_config.json`` → 直接返回。
    2) 若是本地目录但缺 ``preprocessor_config.json`` → 报错（很可能路径错）。
    3) 否则视为 HF Hub ID，用 ``huggingface_hub.snapshot_download`` 下载/定位
       本地 snapshot 目录后返回。
    """
    import os

    # Case 1 / 2: 看起来是本地路径（存在或明显含斜杠+存在）
    if os.path.isdir(model_path):
        if os.path.isfile(os.path.join(model_path, "preprocessor_config.json")):
            return os.path.abspath(model_path)
        raise FileNotFoundError(
            f"base model dir exists but missing preprocessor_config.json: "
            f"{model_path}. 请检查路径是否指向 Qwen3-ASR 的完整 base 目录。"
        )

    # Case 3: HF Hub ID → 定位/下载 snapshot
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "model_path 看起来不是本地目录，需要 huggingface_hub 来解析 "
            "HF Hub ID，但未安装。请 `pip install huggingface_hub`，或直接把 "
            "--model_path 改成 base 模型的本地绝对路径。"
        ) from e

    local_dir = snapshot_download(
        repo_id=model_path,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "chat_template.json",
            "merges.txt",
            "vocab.json",
            "added_tokens.json",
        ],
    )
    if not os.path.isfile(os.path.join(local_dir, "preprocessor_config.json")):
        raise FileNotFoundError(
            f"snapshot_download 完成但 {local_dir} 仍缺 preprocessor_config.json。"
        )
    return os.path.abspath(local_dir)


def main() -> None:
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError(
            "--train_file is required (jsonl with fields: audio/text[/prompt])."
        )

    # -------- 随机种子（尽早设置，覆盖 numpy/torch/random/hf） --------
    set_seed(args_cli.seed)
    print(f"[seed] set_seed({args_cli.seed})", flush=True)

    # -------- 模型 / processor --------
    hw_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    )
    use_bf16 = bool(args_cli.bf16) and hw_bf16

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # 显式确认：不启用任何 PEFT/LoRA
    # （本文件不引入 peft，也不构造任何 LoraConfig/AdapterConfig）

    # -------- 数据集 --------
    data_files = {"train": args_cli.train_file}
    if args_cli.eval_file:
        data_files["validation"] = args_cli.eval_file
    raw_ds = load_dataset("json", data_files=data_files)

    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor, sampling_rate=args_cli.sr
    )

    # -------- TrainingArguments --------
    # 强制 save_strategy/eval_strategy 均为 steps，且 eval_steps == save_steps
    # （load_best_model_at_end=True 要求两者一致触发）
    eval_steps = args_cli.eval_steps if args_cli.eval_steps > 0 else args_cli.save_steps
    if eval_steps != args_cli.save_steps:
        # 为满足 HF `load_best_model_at_end` 的约束，强制对齐
        print(
            f"[warn] eval_steps ({eval_steps}) != save_steps ({args_cli.save_steps}), "
            f"align eval_steps to save_steps for load_best_model_at_end.",
            flush=True,
        )
        eval_steps = args_cli.save_steps

    has_eval = bool(args_cli.eval_file)
    if not has_eval:
        raise ValueError(
            "--eval_file is required in full-ft workflow because we use "
            "load_best_model_at_end + EarlyStopping (both need eval_loss)."
        )

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=(
            args_cli.prefetch_factor if args_cli.num_workers > 0 else None
        ),
        save_strategy="steps",
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        do_eval=True,
        # --- 全参微调关键：加载并保留最佳 ckpt ---
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # --- 精度 ---
        bf16=use_bf16,
        fp16=(not use_bf16),
        # --- 其它 ---
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        # --- 随机种子 ---
        seed=args_cli.seed,
        data_seed=args_cli.seed,
    )

    # -------- Callbacks --------
    # ⚠️ 关键：``MakeEveryCheckpointInferableCallback`` 内部通过
    # ``os.path.exists(os.path.join(base_model_path, fn))`` 判断是否拷贝，
    # 若传入的是 HF Hub ID（如 ``Qwen/Qwen3-ASR-1.7B``），本地不存在同名
    # 目录 → 静默跳过 → ckpt 里缺 preprocessor_config.json 等文件 →
    # 推理时 ``AutoProcessor.from_pretrained`` 直接报 OSError。
    # 这里显式把 ``model_path`` 解析成本地目录后再传给 callback。
    base_model_local = _resolve_base_model_local_dir(args_cli.model_path)
    print(f"[callback] base_model_local = {base_model_local}", flush=True)

    callbacks = [
        MakeEveryCheckpointInferableCallback(base_model_path=base_model_local),
    ]
    if args_cli.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args_cli.early_stopping_patience,
                early_stopping_threshold=args_cli.early_stopping_threshold,
            )
        )
        print(
            f"[early-stop] enabled: patience={args_cli.early_stopping_patience}, "
            f"threshold={args_cli.early_stopping_threshold}",
            flush=True,
        )
    else:
        print("[early-stop] disabled (patience<=0)", flush=True)

    # -------- Trainer --------
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    # -------- Resume --------
    resume_from: Optional[str] = (args_cli.resume_from or "").strip() or None
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir)

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}", flush=True)
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # -------- 收尾 --------
    # 由于 load_best_model_at_end=True，此时 model 已被加载为 eval_loss 最优权重。
    # 官方 MakeEveryCheckpointInferableCallback 会在每次 on_save 时把 tokenizer/
    # processor/generation_config 等复制到对应 checkpoint-* 目录。
    # 此外，把最佳 ckpt 路径持久化到 output_dir/best_ckpt.txt，方便 shell 使用。
    if trainer.args.process_index == 0:
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None) or ""
        try:
            with open(
                os.path.join(training_args.output_dir, "best_ckpt.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(best_ckpt.strip() + "\n")
            print(f"[done] best_model_checkpoint = {best_ckpt}", flush=True)
        except OSError as e:
            print(f"[warn] cannot write best_ckpt.txt: {e}", flush=True)


if __name__ == "__main__":
    main()
