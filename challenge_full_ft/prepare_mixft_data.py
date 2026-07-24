#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_mixft_data.py
=====================

混合方言联合微调 v3 数据准备脚本。

目标：把 12 个外部方言 jsonl 数据（含 label 映射与采样截断）与 VC_v2
挑战集 SFT 数据合并，得到统一的 SFT 训练/验证集。

输出：
    - challenge_full_ft/data_mixft_v3/train.jsonl
    - challenge_full_ft/data_mixft_v3/val.jsonl
    - challenge_full_ft/data_mixft_v3/test.jsonl
    - challenge_full_ft/data_mixft_v3/stats.txt

用法：
    python challenge_full_ft/prepare_mixft_data.py \
        --out_dir challenge_full_ft/data_mixft_v3 \
        --per_accent_max 25000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# --------------------------- 配置区 --------------------------- #

EXT_ROOT = "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/user_johannapeng/challenge-data"

# 12 个外部方言数据源
# 每条记录：(源 jsonl 路径, source 名字用于日志)
EXT_SOURCES: List[Tuple[str, str]] = [
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/hefei_anhui.jsonl",                                       "kespeech_anhui"),
    (f"{EXT_ROOT}/wenetspeech-yue/data_50h/meta_50h_converted.jsonl",                              "wenetspeech_yue"),
    (f"{EXT_ROOT}/Changsha_Dialect_Conversational_Speech_Corpus/changsha_segments_manifest.jsonl", "corpus_changsha"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/dongbei.jsonl",                                           "kespeech_dongbei"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/Henan.jsonl",                                             "kespeech_henan"),
    (f"{EXT_ROOT}/Zhengzhou_Dialect_Conversational_Speech_Corpus/Zhengzhou_segments_manifest.jsonl","corpus_zhengzhou"),
    (f"{EXT_ROOT}/Nanchang_Dialect_Conversational_Speech_Corpus/nanchang_segments_manifest.jsonl", "corpus_nanchang"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/Nanjing.jsonl",                                           "kespeech_nanjing"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/shan3xi.jsonl",                                           "kespeech_shan3xi"),
    (f"{EXT_ROOT}/scripts/Kespeech_jsonl/Shandong.jsonl",                                          "kespeech_shandong"),
    (f"{EXT_ROOT}/wenetspeech-chuan/data-50h/meta_50h_clean.jsonl",                                "wenetspeech_chuan"),
    (f"{EXT_ROOT}/Wuhan_Dialect_Scripted_Speech_Corpus/wuhan_dataset.jsonl",                       "corpus_wuhan"),
    (f"{EXT_ROOT}/wenetspeech-wu/audios_50h.jsonl",                                                "wenetspeech_wu"),
]

# label → 挑战集 accent 映射表
# 覆盖 KeSpeech 与其他数据源的所有已知 label 值（大小写敏感，精确匹配）
LABEL_TO_ACCENT: Dict[str, str] = {
    # ==== KeSpeech 系（label 首字母大写） ====
    "Anhui":     "anhui",
    "Harbin":    "dongbei",
    "Shenyang":  "dongbei",
    "Dalian":    "dongbei",
    "Changchun": "dongbei",
    "Chaoyang":  "dongbei",
    "Chifeng":   "dongbei",
    "Dandong":   "dongbei",
    "Yingkou":   "dongbei",
    "Jilin":     "dongbei",   # 保险起见
    "Liaoning":  "dongbei",   # 保险起见
    "Zhengzhou": "henan",
    "Nanjing":   "nanjing",
    "XiAn":      "shan3xi",
    "Xian":      "shan3xi",   # 兼容
    "Jinan":     "shandong",
    "Qingdao":   "shandong",
    "Weihai":    "shandong",
    "Yantai":    "shandong",
    # ==== 其他 corpus / wenetspeech 数据源 ====
    "cantonese": "cantonese",
    "Cantonese": "cantonese",
    "Changsha":  "changsha",
    "Henan":     "henan",     # Zhengzhou_segments_manifest.jsonl 使用
    "Nanchang":  "nanchang",
    "Sichuan":   "sichuan",
    "wuhan":     "wuhan",
    "Wuhan":     "wuhan",
    "Wuyu":      "wuyu",
    "wuyu":      "wuyu",
}

# VC_v2 SFT 数据（现有格式，直接合并）
VC_V2_TRAIN = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_vc_v2/train.jsonl"
VC_V2_VAL   = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_vc_v2/val.jsonl"
VC_V2_TEST  = "/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_vc_v2/test.jsonl"


# --------------------------- 工具函数 --------------------------- #

def build_target_text(accent: str, text: str) -> str:
    """构造 SFT 目标文本，与 prepare_data.py 保持一致。"""
    accent = (accent or "").strip()
    text = (text or "").strip()
    return f"language Chinese {accent}<asr_text>{text}"


def load_external_jsonl(path: str, source_tag: str,
                        unknown_label_counter: Counter) -> List[Dict]:
    """加载单个外部 jsonl，做 label 映射，返回标准化字典列表。

    每条输出为：
        {
          "audio":  <绝对路径>,
          "text":   "language Chinese {accent}<asr_text>{原文}",
          "prompt": "",
          "key":    "ext_{source_tag}_{n}",
          "accent": "{挑战集 accent}",
          "src":    source_tag,   # 辅助字段，便于调试
        }
    对于无法映射的 label，会跳过并累加计数。
    """
    if not os.path.isfile(path):
        print(f"[ERR] not found: {path}", file=sys.stderr)
        return []

    kept: List[Dict] = []
    n_line = 0
    n_skip_json = 0
    n_skip_label = 0
    n_skip_field = 0

    with open(path, "r", encoding="utf-8") as fin:
        for line_no, raw in enumerate(fin, start=1):
            n_line += 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                n_skip_json += 1
                continue

            audio = obj.get("audio") or obj.get("wav_path") or ""
            text = obj.get("text") or ""
            label = obj.get("label") or obj.get("accent") or ""

            if not audio or not text:
                n_skip_field += 1
                continue

            accent = LABEL_TO_ACCENT.get(label)
            if accent is None:
                unknown_label_counter[f"{source_tag}::{label}"] += 1
                n_skip_label += 1
                continue

            key = f"ext_{source_tag}_{len(kept):07d}"
            kept.append({
                "audio":  audio,
                "text":   build_target_text(accent, text),
                "prompt": "",
                "key":    key,
                "accent": accent,
                "src":    source_tag,
            })

    print(f"[EXT] {source_tag:<20s} lines={n_line:>7d}  kept={len(kept):>7d}  "
          f"skip_json={n_skip_json}  skip_label={n_skip_label}  skip_field={n_skip_field}")
    return kept


def truncate_per_accent(samples: List[Dict], per_accent_max: int,
                        rng: random.Random) -> List[Dict]:
    """对每个 accent 做随机截断到 per_accent_max 条。"""
    by_accent: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_accent[s["accent"]].append(s)

    result: List[Dict] = []
    for accent, lst in sorted(by_accent.items()):
        if len(lst) > per_accent_max:
            rng.shuffle(lst)
            kept = lst[:per_accent_max]
            print(f"[TRUNC] {accent:<12s} {len(lst):>7d} -> {len(kept):>7d}")
        else:
            kept = lst
            print(f"[TRUNC] {accent:<12s} {len(lst):>7d} -> {len(kept):>7d}  (kept all)")
        result.extend(kept)
    return result


def load_vc_v2_jsonl(path: str, tag: str) -> List[Dict]:
    """VC_v2 数据已经是标准 SFT 格式，直接加载即可。"""
    if not os.path.isfile(path):
        print(f"[ERR] VC_v2 file not found: {path}", file=sys.stderr)
        return []

    kept: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fin:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # 补上 src 字段方便调试
            obj["src"] = tag
            kept.append(obj)
    print(f"[VC ] {tag:<20s} loaded={len(kept):>7d}")
    return kept


def dump_jsonl(samples: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for obj in samples:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def accent_stats(samples: List[Dict]) -> Counter:
    return Counter(s["accent"] for s in samples)


def src_stats(samples: List[Dict]) -> Counter:
    return Counter(s.get("src", "?") for s in samples)


def format_stats(name: str, samples: List[Dict]) -> str:
    lines = [f"===== {name} (total={len(samples)}) ====="]
    lines.append("-- by accent --")
    for a, c in sorted(accent_stats(samples).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {a:<12s} {c}")
    lines.append("-- by source --")
    for s, c in sorted(src_stats(samples).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {s:<20s} {c}")
    return "\n".join(lines) + "\n"


# --------------------------- 主流程 --------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str,
                   default="/mnt/geminihzceph/user_johannapeng/challenge_model/challenge_full_ft/data_mixft_v3")
    p.add_argument("--per_accent_max", type=int, default=25000,
                   help="每个 accent 从外部数据中最多采样多少条（超过才截断）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. 加载全部外部方言 ---------- #
    print("\n[STEP 1] Loading external dialect jsonls ...")
    unknown_label_counter: Counter = Counter()
    ext_all: List[Dict] = []
    for path, tag in EXT_SOURCES:
        ext_all.extend(load_external_jsonl(path, tag, unknown_label_counter))

    if unknown_label_counter:
        print("\n[WARN] Unknown labels (skipped):")
        for k, v in unknown_label_counter.most_common():
            print(f"       - {k}: {v}")

    print(f"\n[STEP 1] external total loaded = {len(ext_all)}")

    # ---------- 2. 按 accent 采样截断 ---------- #
    print("\n[STEP 2] Truncating per-accent to <= {} ...".format(args.per_accent_max))
    ext_trunc = truncate_per_accent(ext_all, args.per_accent_max, rng)
    print(f"[STEP 2] external after truncation = {len(ext_trunc)}")

    # ---------- 3. 加载 VC_v2 挑战集 SFT 数据 ---------- #
    print("\n[STEP 3] Loading VC_v2 challenge SFT data ...")
    vc_train = load_vc_v2_jsonl(VC_V2_TRAIN, "vc_v2_train")
    vc_val   = load_vc_v2_jsonl(VC_V2_VAL,   "vc_v2_val")

    # ---------- 4. 合并并 shuffle ---------- #
    print("\n[STEP 4] Merging & shuffling ...")
    train_all = ext_trunc + vc_train
    rng.shuffle(train_all)
    print(f"[STEP 4] final train samples = {len(train_all)}")
    print(f"[STEP 4] final val   samples = {len(vc_val)}  (VC_v2 val, unchanged)")

    # ---------- 5. 写出 ---------- #
    print("\n[STEP 5] Writing jsonl ...")
    train_out = os.path.join(out_dir, "train.jsonl")
    val_out   = os.path.join(out_dir, "val.jsonl")
    test_out  = os.path.join(out_dir, "test.jsonl")
    stats_out = os.path.join(out_dir, "stats.txt")

    dump_jsonl(train_all, train_out)
    dump_jsonl(vc_val,    val_out)

    # test 集用 VC_v2 test 直接软链接/拷贝；训练管线里其实不用 test
    if os.path.isfile(VC_V2_TEST):
        shutil.copyfile(VC_V2_TEST, test_out)
        print(f"[STEP 5] test.jsonl copied from {VC_V2_TEST}")
    else:
        with open(test_out, "w") as _:
            pass
        print("[STEP 5] test.jsonl written as empty placeholder")

    # ---------- 6. 打印/写出统计 ---------- #
    stats_lines: List[str] = []
    stats_lines.append(format_stats("TRAIN", train_all))
    stats_lines.append(format_stats("VAL",   vc_val))
    if unknown_label_counter:
        stats_lines.append("===== Unknown labels (skipped) =====")
        for k, v in unknown_label_counter.most_common():
            stats_lines.append(f"  {k}: {v}")
        stats_lines.append("")
    stats_lines.append(f"per_accent_max = {args.per_accent_max}")
    stats_lines.append(f"seed           = {args.seed}")

    stats_text = "\n".join(stats_lines)
    with open(stats_out, "w", encoding="utf-8") as fout:
        fout.write(stats_text)

    print("\n" + stats_text)
    print(f"\n[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
