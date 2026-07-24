#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_ckpt_processor.py
=====================

**事后修复脚本**：把 Qwen3-ASR base 模型目录里的 processor 相关文件补齐到
Trainer 产出的 ``checkpoint-*`` 目录，使其能被 ``AutoProcessor.from_pretrained``
直接加载。

为什么需要这个脚本？
--------------------
官方 ``qwen3_asr_sft.py`` 中的 ``MakeEveryCheckpointInferableCallback`` 会在
每次 ``on_save`` 时把 base 模型目录下的 processor / tokenizer 相关文件
（``preprocessor_config.json / chat_template.json`` 等）拷贝到当前 ckpt 目录。
但它的拷贝逻辑是 ``if os.path.exists(src): shutil.copy2(src, dst)``——
若训练时传的是 HF Hub ID（如 ``Qwen/Qwen3-ASR-1.7B``）而非本地路径，
``src`` 根本不存在，就会静默跳过，导致 ckpt 目录缺失 processor 文件、
后续无法推理。

本脚本可以在这种情况下把已有 ckpt 修复成可推理状态。

用法
----
1) 修复单个 ckpt::

    python challenge_full_ft/fix_ckpt_processor.py \\
        --base_model /mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B \\
        --ckpt       /mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/outputs/checkpoint-400

2) 批量修复 output_dir 下所有 checkpoint-*::

    python challenge_full_ft/fix_ckpt_processor.py \\
        --base_model /mnt/geminihzceph/user_johannapeng/Qwen3-ASR/Qwen3-ASR-1.7B \\
        --output_dir /mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/outputs

行为
----
- **只补齐缺失文件**，不覆盖已存在文件（避免踩到 ``config.json`` 这种
  已由 Trainer 按训练后的 model.config 写好的文件）。
- 若 base_model 里也没有某个文件（比如 ``processor_config.json`` 本身
  就不是 Qwen3-ASR 的必需文件），则跳过，不算错误。
- 打印每个 ckpt 的修复结果，便于人工核对。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import List

# 与官方 ``copy_required_hf_files_for_qwen_asr`` 一致的 required 列表。
# 允许 base_model 里缺失（silently skipped）。
REQUIRED_FILES = [
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
]

_CKPT_RE = re.compile(r"^checkpoint-\d+$")


def fix_one_ckpt(base_model: str, ckpt_dir: str) -> None:
    if not os.path.isdir(ckpt_dir):
        print(f"[fix][skip] not a dir: {ckpt_dir}", flush=True)
        return

    copied: List[str] = []
    kept: List[str] = []
    missing_in_base: List[str] = []

    for fn in REQUIRED_FILES:
        dst = os.path.join(ckpt_dir, fn)
        src = os.path.join(base_model, fn)

        if os.path.exists(dst):
            kept.append(fn)
            continue

        if not os.path.exists(src):
            missing_in_base.append(fn)
            continue

        shutil.copy2(src, dst)
        copied.append(fn)

    print(f"[fix] ckpt={ckpt_dir}", flush=True)
    if copied:
        print(f"[fix]   copied ({len(copied)}): {copied}", flush=True)
    if kept:
        print(f"[fix]   kept   ({len(kept)}): {kept}", flush=True)
    if missing_in_base:
        print(
            f"[fix]   base-miss ({len(missing_in_base)}): {missing_in_base} "
            f"(base_model 里也没有，跳过)",
            flush=True,
        )


def iter_ckpt_dirs(output_dir: str) -> List[str]:
    if not os.path.isdir(output_dir):
        return []
    out = []
    for name in sorted(os.listdir(output_dir)):
        if not _CKPT_RE.match(name):
            continue
        p = os.path.join(output_dir, name)
        if os.path.isdir(p):
            out.append(p)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fill missing processor/tokenizer files into Qwen3-ASR ckpts."
    )
    p.add_argument(
        "--base_model", type=str, required=True,
        help="Qwen3-ASR base 模型的本地完整目录（含 preprocessor_config.json 等）",
    )
    p.add_argument(
        "--ckpt", type=str, default="",
        help="要修复的单个 ckpt 目录（与 --output_dir 二选一）",
    )
    p.add_argument(
        "--output_dir", type=str, default="",
        help="批量模式：扫描该目录下所有 checkpoint-* 并逐一修复",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.base_model):
        print(f"[fix][error] base_model dir not found: {args.base_model}",
              file=sys.stderr)
        sys.exit(2)

    if not args.ckpt and not args.output_dir:
        print("[fix][error] 必须至少指定 --ckpt 或 --output_dir 之一",
              file=sys.stderr)
        sys.exit(2)

    targets: List[str] = []
    if args.ckpt:
        targets.append(os.path.abspath(args.ckpt))
    if args.output_dir:
        targets.extend(iter_ckpt_dirs(os.path.abspath(args.output_dir)))

    if not targets:
        print(f"[fix] no checkpoint-* found under: {args.output_dir}", flush=True)
        return

    print(f"[fix] base_model = {args.base_model}", flush=True)
    print(f"[fix] targets    = {len(targets)}", flush=True)
    for t in targets:
        fix_one_ckpt(args.base_model, t)

    print("[fix] done.", flush=True)


if __name__ == "__main__":
    main()
