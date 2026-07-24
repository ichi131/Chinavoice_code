#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_mix_data.py
=================

把 VC_data_v2 与 basic_change_data 两份"含原始+增强"的 jsonl 合并成一份混合训练/验证集，
并且**只保留一份原始样本**，避免原始被采样两次。

判定规则（基于 basic_augment.py / vc_augment.py 的输出约定）：
- 原始样本 : ``key`` 中既不含 ``__aug_`` 也不含 ``__to__``；``wav_path`` 指向 reference_set。
- basic 增强样本 : ``key`` 中包含 ``__aug_``。
- VC 增强样本 : ``key`` 中包含 ``__to__``。

合并方式：
    output_train = (VC 的所有 __to__ 行)
                 + (basic 的所有 __aug_ 行)
                 + (原始行，按 key 去重，只保留一份)
    output_val   同上（各自 split 内合并）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterator, Set, Tuple


def is_aug(key: str) -> bool:
    return ("__aug_" in key) or ("__to__" in key)


def iter_jsonl(path: str) -> Iterator[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"jsonl not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[WARN] skip invalid line {path}:{line_no}: {e}", file=sys.stderr)


def merge_split(
    vc_path: str,
    basic_path: str,
    out_path: str,
) -> Tuple[int, int, int, int]:
    """合并单个 split。

    Returns
    -------
    (n_orig_kept, n_vc_aug, n_basic_aug, n_total)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seen_orig_keys: Set[str] = set()
    n_orig_kept = n_vc_aug = n_basic_aug = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        # ---- Step 1: 写 VC 的所有 __to__ 增强 ----
        for o in iter_jsonl(vc_path):
            k = o.get("key", "")
            if "__to__" in k:
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                n_vc_aug += 1

        # ---- Step 2: 写 basic 的所有 __aug_ 增强 ----
        for o in iter_jsonl(basic_path):
            k = o.get("key", "")
            if "__aug_" in k:
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                n_basic_aug += 1

        # ---- Step 3: 写原始样本（跨两个源文件去重，只留一份） ----
        for src in (vc_path, basic_path):
            for o in iter_jsonl(src):
                k = o.get("key", "")
                if is_aug(k):
                    continue
                if not k:
                    # 空 key 无法安全去重，直接跳过（不应出现）
                    continue
                if k in seen_orig_keys:
                    continue
                seen_orig_keys.add(k)
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                n_orig_kept += 1

    n_total = n_orig_kept + n_vc_aug + n_basic_aug
    return n_orig_kept, n_vc_aug, n_basic_aug, n_total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge VC_data_v2 + basic_change_data into a single jsonl per split; "
                    "deduplicate the original (non-aug) samples."
    )
    p.add_argument(
        "--vc_dir", type=str,
        default="/mnt/geminihzceph/user_johannapeng/challenge_model/VC_data_v2",
    )
    p.add_argument(
        "--basic_dir", type=str,
        default="/mnt/geminihzceph/user_johannapeng/challenge_model/basic_change_data",
    )
    p.add_argument(
        "--out_dir", type=str,
        default="/mnt/geminihzceph/user_johannapeng/challenge_model/mix_data",
    )
    p.add_argument("--vc_train_name",    type=str, default="data_train_vc.jsonl")
    p.add_argument("--vc_val_name",      type=str, default="data_val_vc.jsonl")
    p.add_argument("--basic_train_name", type=str, default="data_train_aug.jsonl")
    p.add_argument("--basic_val_name",   type=str, default="data_val_aug.jsonl")
    p.add_argument("--out_train_name",   type=str, default="data_train_mix.jsonl")
    p.add_argument("--out_val_name",     type=str, default="data_val_mix.jsonl")
    # 顺便产出一个空 test 占位，让 prepare_data.py 不报错
    p.add_argument("--out_test_name",    type=str, default="data_test_mix.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plan = [
        ("train",
         os.path.join(args.vc_dir,    args.vc_train_name),
         os.path.join(args.basic_dir, args.basic_train_name),
         os.path.join(args.out_dir,   args.out_train_name)),
        ("val",
         os.path.join(args.vc_dir,    args.vc_val_name),
         os.path.join(args.basic_dir, args.basic_val_name),
         os.path.join(args.out_dir,   args.out_val_name)),
    ]

    print("[merge_mix_data] plan:")
    for split, vc_p, basic_p, out_p in plan:
        print(f"  [{split}] VC   = {vc_p}")
        print(f"  [{split}] BASIC= {basic_p}")
        print(f"  [{split}] OUT  = {out_p}")
    print(f"[merge_mix_data] out_dir = {args.out_dir}")

    for split, vc_p, basic_p, out_p in plan:
        print(f"\n[merge_mix_data] === merging split={split} ===")
        n_orig, n_vc, n_basic, n_total = merge_split(vc_p, basic_p, out_p)
        print(f"[merge_mix_data]   orig kept (dedup) = {n_orig}")
        print(f"[merge_mix_data]   vc __to__         = {n_vc}")
        print(f"[merge_mix_data]   basic __aug_      = {n_basic}")
        print(f"[merge_mix_data]   TOTAL written     = {n_total}  -> {out_p}")

    # 建一个空 test 占位（prepare_data.py 强制要求三份都在）
    test_placeholder = os.path.join(args.out_dir, args.out_test_name)
    if not os.path.isfile(test_placeholder):
        open(test_placeholder, "w").close()
        print(f"\n[merge_mix_data] touched empty test placeholder: {test_placeholder}")

    print("\n[merge_mix_data] done.")


if __name__ == "__main__":
    main()
