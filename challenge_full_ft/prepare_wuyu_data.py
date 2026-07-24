#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_wuyu_data.py
====================

吴语（wuyu）专用 ASR 两阶段微调的数据处理脚本。共提供 4 个子命令：

  1) normalize          规范化任一数据源（wenetspeech50h / vc_v2 / challenge_raw）
                        为 Qwen3-ASR-SFT 格式，并**强制** accent=wuyu、target 前缀
                        为 "language Chinese wuyu<asr_text>"。
  2) split_50h          将规范化后的 50h SFT jsonl 按 seed=42、val=5% 切分为
                        train.jsonl / val.jsonl，输出到 data_wuyu_50h/。
  3) build_stage2       合并 VC_data_v2 wuyu + 挑战集原始 wuyu，按原始 key 去重
                        原始样本，全部保留 __to__ / __aug_ 增广样本，输出到
                        data_wuyu_stage2/{train,val}.jsonl。
  4) extract_test_wuyu  从挑战集官方 test.jsonl 中筛出 accent==wuyu 的行
                        （151 条），写入 data_wuyu_stage2/test_wuyu.jsonl。

所有子命令默认支持"若目标已存在且非空则跳过"；`FORCE_REBUILD=1` 环境变量或
`--force` 参数可强制重算。

统一 SFT 输出格式：
    {"audio": ..., "text": "language Chinese wuyu<asr_text>{原文}",
     "prompt": "", "key": ..., "accent": "wuyu"}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 通用常量
# ---------------------------------------------------------------------------

FORCED_ACCENT = "wuyu"
TARGET_PREFIX = f"language Chinese {FORCED_ACCENT}<asr_text>"
SEED = 42

# 用于识别音色增广样本的 key 后缀 pattern：
# - VC_data_v2 用 __to__ 后缀（例如 xxx__to__spkA）
# - basic_change_data 用 __aug_ 后缀（例如 xxx__aug_1）
AUG_SUFFIX_RE = re.compile(r"(?:__to__|__aug_).*$")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def iter_jsonl(path: str) -> Iterable[Dict]:
    """流式读取一行一个 json 的 jsonl 文件，自动跳过空行与非法行。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"jsonl not found: {path}")
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[WARN] skip invalid json at {path}:{line_no}: {e}",
                      file=sys.stderr)


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fout:
        for obj in rows:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_target(text: str) -> str:
    """强制 accent=wuyu，target 前缀始终为 language Chinese wuyu。"""
    text = (text or "").strip()
    return f"{TARGET_PREFIX}{text}"


def strip_aug_suffix(key: str) -> str:
    """去掉 __to__* / __aug_* 后缀，返回原始 key。"""
    if not key:
        return ""
    return AUG_SUFFIX_RE.sub("", key)


def is_augmented_key(key: str) -> str:
    """
    返回增广类型标签："__to__" / "__aug_" / ""（未增广）。
    """
    if "__to__" in key:
        return "__to__"
    if "__aug_" in key:
        return "__aug_"
    return ""


def is_target_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def should_skip(target_path: str, force: bool, tag: str) -> bool:
    if force:
        return False
    if is_target_nonempty(target_path):
        print(f"[{tag}] target exists and non-empty, SKIP: {target_path}")
        return True
    return False


# ---------------------------------------------------------------------------
# Sub-command: normalize
# ---------------------------------------------------------------------------

def normalize_row_wenetspeech50h(obj: Dict, check_audio_exists: bool
                                 ) -> Tuple[Optional[Dict], str]:
    """
    输入格式：{"audio": abs_wav_path, "text": ..., "label": "Wuyu"}
    输出：SFT 格式 dict 或 None（跳过），第二值是跳过原因 tag。
    """
    audio = (obj.get("audio") or "").strip()
    text = (obj.get("text") or "").strip()
    key = (obj.get("key") or "").strip()
    if not audio:
        return None, "empty_audio"
    if not text:
        return None, "empty_text"
    if check_audio_exists and not os.path.isfile(audio):
        return None, "missing_audio"
    if not key:
        # 用 wav 文件名（去扩展名）作为 key
        key = os.path.splitext(os.path.basename(audio))[0]
    return {
        "audio": audio,
        "text": build_target(text),
        "prompt": "",
        "key": key,
        "accent": FORCED_ACCENT,
    }, ""


def normalize_row_vc_v2(obj: Dict, check_audio_exists: bool
                        ) -> Tuple[Optional[Dict], str]:
    """
    输入格式：{"key": ..., "wav_path": ..., "text": ..., "accent": "wuyu"}
    仅当 accent=="wuyu" 时保留，其它跳过。
    """
    accent = (obj.get("accent") or "").strip()
    if accent != FORCED_ACCENT:
        return None, "not_wuyu"
    audio = (obj.get("wav_path") or "").strip()
    text = (obj.get("text") or "").strip()
    key = (obj.get("key") or "").strip()
    if not audio:
        return None, "empty_audio"
    if not text:
        return None, "empty_text"
    if check_audio_exists and not os.path.isfile(audio):
        return None, "missing_audio"
    if not key:
        key = os.path.splitext(os.path.basename(audio))[0]
    return {
        "audio": audio,
        "text": build_target(text),
        "prompt": "",
        "key": key,
        "accent": FORCED_ACCENT,
    }, ""


def normalize_row_challenge_raw(obj: Dict, check_audio_exists: bool
                                ) -> Tuple[Optional[Dict], str]:
    """
    输入格式（已经是 SFT 格式）：
        {"audio": ..., "text": "language Chinese wuyu<asr_text>{原文}",
         "prompt": "", "key": ..., "accent": "wuyu"}
    仅当 accent=="wuyu" 时保留；同时校验 target 前缀，异常则打印警告并重写。
    """
    accent = (obj.get("accent") or "").strip()
    if accent != FORCED_ACCENT:
        return None, "not_wuyu"
    audio = (obj.get("audio") or "").strip()
    text = (obj.get("text") or "").strip()
    key = (obj.get("key") or "").strip()
    if not audio:
        return None, "empty_audio"
    if not text:
        return None, "empty_text"
    if check_audio_exists and not os.path.isfile(audio):
        return None, "missing_audio"
    # 校验 target 前缀
    if not text.startswith(TARGET_PREFIX):
        # 尝试从 "language Chinese *<asr_text>" 里剥出原文
        m = re.match(r"^language Chinese \S+<asr_text>(.*)$", text, re.DOTALL)
        raw_text = m.group(1) if m else text
        print(f"[WARN] challenge_raw target prefix mismatch, rewriting. key={key}",
              file=sys.stderr)
        text = build_target(raw_text)
    if not key:
        key = os.path.splitext(os.path.basename(audio))[0]
    return {
        "audio": audio,
        "text": text,
        "prompt": "",
        "key": key,
        "accent": FORCED_ACCENT,
    }, ""


NORMALIZERS = {
    "wenetspeech50h": normalize_row_wenetspeech50h,
    "vc_v2": normalize_row_vc_v2,
    "challenge_raw": normalize_row_challenge_raw,
}


def cmd_normalize(args: argparse.Namespace) -> None:
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    src_type = args.source
    check_audio = bool(args.check_audio_exists)
    force = bool(args.force) or os.environ.get("FORCE_REBUILD", "0") == "1"

    if should_skip(dst, force, f"normalize:{src_type}"):
        return

    if src_type not in NORMALIZERS:
        raise ValueError(f"unknown --source: {src_type}")
    normalizer = NORMALIZERS[src_type]

    print(f"[normalize] source          = {src_type}")
    print(f"[normalize] src             = {src}")
    print(f"[normalize] dst             = {dst}")
    print(f"[normalize] check_audio     = {check_audio}")

    n_in = n_out = 0
    reasons: Counter = Counter()
    seen_keys: set = set()

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fout:
        for obj in iter_jsonl(src):
            n_in += 1
            new_obj, reason = normalizer(obj, check_audio)
            if new_obj is None:
                reasons[reason] += 1
                continue
            # key 去重（同一批产物内保证唯一）
            k = new_obj["key"]
            if k in seen_keys:
                reasons["dup_key"] += 1
                continue
            seen_keys.add(k)
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[normalize] in              = {n_in}")
    print(f"[normalize] out             = {n_out}")
    if reasons:
        print(f"[normalize] skipped by reason:")
        for r, c in sorted(reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"[normalize]   - {r:<16s} {c}")
    print("[normalize] done.")


# ---------------------------------------------------------------------------
# Sub-command: split_50h
# ---------------------------------------------------------------------------

def cmd_split_50h(args: argparse.Namespace) -> None:
    src = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out_dir)
    val_ratio = float(args.val_ratio)
    force = bool(args.force) or os.environ.get("FORCE_REBUILD", "0") == "1"

    train_out = os.path.join(out_dir, "train.jsonl")
    val_out = os.path.join(out_dir, "val.jsonl")

    if not force and is_target_nonempty(train_out) and is_target_nonempty(val_out):
        print(f"[split_50h] both target files exist and non-empty, SKIP:")
        print(f"[split_50h]   {train_out}")
        print(f"[split_50h]   {val_out}")
        return

    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict] = list(iter_jsonl(src))
    n_total = len(rows)
    if n_total == 0:
        print(f"[split_50h] ERROR: {src} is empty.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(SEED)
    rng.shuffle(rows)

    n_val = max(1, int(round(n_total * val_ratio)))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    train_keys = {r["key"] for r in train_rows}
    val_keys = {r["key"] for r in val_rows}
    inter = train_keys & val_keys

    print(f"[split_50h] src             = {src}")
    print(f"[split_50h] out_dir         = {out_dir}")
    print(f"[split_50h] seed            = {SEED}")
    print(f"[split_50h] val_ratio       = {val_ratio}")
    print(f"[split_50h] total           = {n_total}")
    print(f"[split_50h] train           = {len(train_rows)}")
    print(f"[split_50h] val             = {len(val_rows)}")
    print(f"[split_50h] key intersect   = {len(inter)}")
    if inter:
        print(f"[split_50h] ERROR: train/val key overlap detected!", file=sys.stderr)
        sys.exit(1)
    print("[split_50h] done.")


# ---------------------------------------------------------------------------
# Sub-command: build_stage2
# ---------------------------------------------------------------------------

def _load_and_filter_wuyu(path: str, source_tag: str, check_audio: bool
                          ) -> List[Dict]:
    """加载 jsonl 并只保留 accent=wuyu 的行；结果全部转换为统一 SFT 格式。"""
    rows: List[Dict] = []
    normalizer = NORMALIZERS[source_tag]
    for obj in iter_jsonl(path):
        new_obj, _reason = normalizer(obj, check_audio)
        if new_obj is not None:
            rows.append(new_obj)
    return rows


def _dedup_and_merge(vc_rows: List[Dict], raw_rows: List[Dict],
                     tag: str) -> Tuple[List[Dict], Dict[str, int]]:
    """
    - 所有 __to__ / __aug_ 增广样本全部保留
    - 原始样本（无增广后缀）以原始 key 为去重键，只保留一份
      （若同一原始 key 在 vc 与 challenge_raw 中同时出现，优先保留 challenge_raw 的
       version，因为它是官方最权威的一份）
    """
    aug_rows: List[Dict] = []
    orig_by_key: Dict[str, Dict] = {}
    stats = {"__to__": 0, "__aug_": 0, "orig_vc": 0, "orig_raw": 0}

    # 先处理 VC 侧（增广多来自这里）
    for r in vc_rows:
        aug_tag = is_augmented_key(r["key"])
        if aug_tag:
            aug_rows.append(r)
            stats[aug_tag] += 1
        else:
            base_key = strip_aug_suffix(r["key"])
            if base_key not in orig_by_key:
                orig_by_key[base_key] = r
                stats["orig_vc"] += 1

    # 再叠加 challenge_raw 侧（同一原始 key 覆盖 vc 侧）
    for r in raw_rows:
        aug_tag = is_augmented_key(r["key"])
        if aug_tag:
            aug_rows.append(r)
            stats[aug_tag] += 1
        else:
            base_key = strip_aug_suffix(r["key"])
            # 覆盖 vc 侧同名 key
            if base_key in orig_by_key and orig_by_key[base_key] is not None \
                    and orig_by_key[base_key].get("__from") != "raw":
                stats["orig_vc"] -= 1
            r["__from"] = "raw"
            orig_by_key[base_key] = r
            stats["orig_raw"] += 1

    orig_rows = list(orig_by_key.values())
    # 移除内部标记字段
    for r in orig_rows:
        r.pop("__from", None)

    merged = aug_rows + orig_rows
    print(f"[build_stage2:{tag}] __to__ aug          = {stats['__to__']}")
    print(f"[build_stage2:{tag}] __aug_ aug          = {stats['__aug_']}")
    print(f"[build_stage2:{tag}] orig from vc_v2     = {stats['orig_vc']}")
    print(f"[build_stage2:{tag}] orig from raw       = {stats['orig_raw']}")
    print(f"[build_stage2:{tag}] merged total        = {len(merged)}")
    return merged, stats


def cmd_build_stage2(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.out_dir)
    force = bool(args.force) or os.environ.get("FORCE_REBUILD", "0") == "1"

    train_out = os.path.join(out_dir, "train.jsonl")
    val_out = os.path.join(out_dir, "val.jsonl")

    if not force and is_target_nonempty(train_out) and is_target_nonempty(val_out):
        print(f"[build_stage2] both target files exist and non-empty, SKIP:")
        print(f"[build_stage2]   {train_out}")
        print(f"[build_stage2]   {val_out}")
        return

    os.makedirs(out_dir, exist_ok=True)
    check_audio = bool(args.check_audio_exists)

    print(f"[build_stage2] vc_train        = {args.vc_train}")
    print(f"[build_stage2] vc_val          = {args.vc_val}")
    print(f"[build_stage2] raw_train       = {args.raw_train}")
    print(f"[build_stage2] raw_val         = {args.raw_val}")
    print(f"[build_stage2] out_dir         = {out_dir}")

    # ---- Train ----
    print(f"\n[build_stage2] === Train ===")
    vc_tr = _load_and_filter_wuyu(args.vc_train, "vc_v2", check_audio)
    raw_tr = _load_and_filter_wuyu(args.raw_train, "challenge_raw", check_audio)
    print(f"[build_stage2:train] vc_v2 wuyu rows       = {len(vc_tr)}")
    print(f"[build_stage2:train] challenge_raw wuyu    = {len(raw_tr)}")
    merged_tr, _ = _dedup_and_merge(vc_tr, raw_tr, "train")

    # ---- Val ----
    print(f"\n[build_stage2] === Val ===")
    vc_va = _load_and_filter_wuyu(args.vc_val, "vc_v2", check_audio)
    raw_va = _load_and_filter_wuyu(args.raw_val, "challenge_raw", check_audio)
    print(f"[build_stage2:val] vc_v2 wuyu rows         = {len(vc_va)}")
    print(f"[build_stage2:val] challenge_raw wuyu      = {len(raw_va)}")
    merged_va, _ = _dedup_and_merge(vc_va, raw_va, "val")

    if not merged_tr or not merged_va:
        print(f"[build_stage2] ERROR: train or val merged empty.", file=sys.stderr)
        sys.exit(1)

    n_tr = write_jsonl(train_out, merged_tr)
    n_va = write_jsonl(val_out, merged_va)

    print(f"\n[build_stage2] written train           = {n_tr}  ({train_out})")
    print(f"[build_stage2] written val             = {n_va}  ({val_out})")
    print("[build_stage2] done.")


# ---------------------------------------------------------------------------
# Sub-command: extract_test_wuyu
# ---------------------------------------------------------------------------

def cmd_extract_test_wuyu(args: argparse.Namespace) -> None:
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    force = bool(args.force)  # test 天然固定，不受 FORCE_REBUILD 影响

    if not force and is_target_nonempty(dst):
        print(f"[extract_test_wuyu] target exists and non-empty, SKIP: {dst}")
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    n_in = n_out = 0
    with open(dst, "w", encoding="utf-8") as fout:
        for obj in iter_jsonl(src):
            n_in += 1
            if (obj.get("accent") or "").strip() != FORCED_ACCENT:
                continue
            # test 保持原样输出（不做 target 重写）以对齐官方评估
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[extract_test_wuyu] src           = {src}")
    print(f"[extract_test_wuyu] dst           = {dst}")
    print(f"[extract_test_wuyu] scanned       = {n_in}")
    print(f"[extract_test_wuyu] kept (wuyu)   = {n_out}")
    if n_out == 0:
        print(f"[extract_test_wuyu] ERROR: no wuyu row extracted!", file=sys.stderr)
        sys.exit(1)
    print("[extract_test_wuyu] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wuyu specialist ASR 数据处理脚本（4 子命令）。"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- normalize ----
    p1 = sub.add_parser("normalize", help="规范化单个数据源为 SFT jsonl")
    p1.add_argument("--source", required=True,
                    choices=list(NORMALIZERS.keys()),
                    help="数据源类型")
    p1.add_argument("--src", required=True, help="源 jsonl 路径")
    p1.add_argument("--dst", required=True, help="输出 jsonl 路径")
    p1.add_argument("--check_audio_exists", type=int, default=0)
    p1.add_argument("--force", action="store_true",
                    help="强制重跑（也可用 FORCE_REBUILD=1）")
    p1.set_defaults(func=cmd_normalize)

    # ---- split_50h ----
    p2 = sub.add_parser("split_50h", help="将规范化后的 50h SFT jsonl 切分 train/val")
    p2.add_argument("--src", required=True, help="规范化后的 50h SFT jsonl")
    p2.add_argument("--out_dir", required=True,
                    help="输出目录（含 train.jsonl / val.jsonl）")
    p2.add_argument("--val_ratio", type=float, default=0.05)
    p2.add_argument("--force", action="store_true")
    p2.set_defaults(func=cmd_split_50h)

    # ---- build_stage2 ----
    p3 = sub.add_parser("build_stage2",
                        help="合并 VC_v2 wuyu + 挑战集原始 wuyu，输出 stage2 数据")
    p3.add_argument("--vc_train", required=True,
                    help="VC_data_v2/data_train_vc.jsonl 路径")
    p3.add_argument("--vc_val", required=True,
                    help="VC_data_v2/data_val_vc.jsonl 路径")
    p3.add_argument("--raw_train", required=True,
                    help="challenge_full_ft/data/train.jsonl（已 SFT 格式）")
    p3.add_argument("--raw_val", required=True,
                    help="challenge_full_ft/data/val.jsonl（已 SFT 格式）")
    p3.add_argument("--out_dir", required=True,
                    help="输出目录（data_wuyu_stage2）")
    p3.add_argument("--check_audio_exists", type=int, default=0)
    p3.add_argument("--force", action="store_true")
    p3.set_defaults(func=cmd_build_stage2)

    # ---- extract_test_wuyu ----
    p4 = sub.add_parser("extract_test_wuyu",
                        help="从官方 test.jsonl 中筛出 accent=wuyu 的行")
    p4.add_argument("--src", required=True,
                    help="challenge_full_ft/data/test.jsonl 路径")
    p4.add_argument("--dst", required=True,
                    help="输出 test_wuyu.jsonl 路径")
    p4.add_argument("--force", action="store_true")
    p4.set_defaults(func=cmd_extract_test_wuyu)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
