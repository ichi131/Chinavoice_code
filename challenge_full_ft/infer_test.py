#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_test.py
==============

使用挑选出的最佳 ckpt 在 test 集上进行**批量推理**，输出的 JSONL 直接兼容
`ChinaVoices-Challenge/eval/eval_jsonl_with_wer_tools.sh` 需要的字段。

输入 JSONL 每行（由 `prepare_data.py` 产出）：
    {"audio": "...wav", "text": "language Chinese chaoshan<asr_text>xxx",
     "prompt": "", "key": "chaoshan_000459", "accent": "chaoshan"}

输出 JSONL 每行：
    {"utt_id":       "chaoshan_000459",
     "audio_path":   "...wav",
     "ref_full":     "language Chinese chaoshan<asr_text>xxx",
     "ref_text":     "xxx",
     "ref_dialect":  "chaoshan",
     "pred_full":    "language Chinese chaoshan<asr_text>yyy",
     "pred_text":    "yyy",
     "pred_dialect": "chaoshan",
     "error":        ""}

推理调用方式与 baseline `infer_batch.py` 一致：
`Qwen3ASRModel._infer_asr(contexts, wavs, languages)`。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch


# =============================================================================
# 从 baseline 的 split_asr_content 复刻的解析函数（与之保持一致）
# =============================================================================
ASR_MARKER = "<asr_text>"
DIALECT_RE = re.compile(r"language\s+Chinese\s+([^\s<]+)", re.IGNORECASE)


def split_asr_content(content: str) -> Dict[str, str]:
    """解析 ``language Chinese anhui<asr_text>...`` 结构。"""
    content = (content or "").strip()
    if ASR_MARKER in content:
        prefix, text = content.split(ASR_MARKER, 1)
    else:
        prefix, text = "", content
    dialect = ""
    m = DIALECT_RE.search(prefix)
    if m:
        dialect = m.group(1).strip()
    return {
        "full": content,
        "prefix": prefix.strip(),
        "dialect": dialect,
        "text": text.strip(),
    }


# =============================================================================
# 数据读取：直接读 prepare_data.py 产出的 SFT JSONL
# =============================================================================
def load_test_jsonl(path: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[warn] skip invalid json at {path}:{line_no}: {e}",
                      file=sys.stderr, flush=True)
                continue

            audio_path = obj.get("audio", "")
            if not audio_path:
                print(f"[warn] line {line_no}: missing 'audio', skipped",
                      file=sys.stderr, flush=True)
                continue

            ref_full = obj.get("text", "")
            ref = split_asr_content(ref_full)

            utt_id = (
                obj.get("key")
                or obj.get("id")
                or Path(audio_path).stem
                or str(line_no)
            )
            accent_field = obj.get("accent", "")
            ref_dialect = accent_field if accent_field else ref["dialect"]

            samples.append({
                "utt_id":       utt_id,
                "audio_path":   audio_path,
                "ref_full":     ref["full"],
                "ref_text":     ref["text"],
                "ref_dialect":  ref_dialect,
            })
    return samples


# =============================================================================
# 推理
# =============================================================================
def infer_raw(asr, audio_paths: List[str]) -> List[str]:
    """直接调用 Qwen3-ASR 原始生成，与 baseline 一致。"""
    from qwen_asr.inference.utils import normalize_audios
    wavs = normalize_audios(audio_paths)
    contexts = [""] * len(wavs)
    languages = [None] * len(wavs)
    return asr._infer_asr(contexts, wavs, languages)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch inference on test.jsonl with a Qwen3-ASR ckpt."
    )
    p.add_argument("--model", type=str, required=True,
                   help="模型或 ckpt 目录（必须含 config.json 等完整 HF 文件）")
    p.add_argument("--data", type=str, required=True,
                   help="prepare_data.py 产出的 test.jsonl")
    p.add_argument("--output", type=str, required=True,
                   help="预测 JSONL 保存路径（默认建议 outputs/pred_test.jsonl）")
    p.add_argument("--batch-size", type=int, default=32,
                   help="推理 batch，显存不够就调小")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="每条音频最多生成 token 数")
    p.add_argument("--device-map", type=str, default="cuda:0",
                   help="transformers device_map。**强烈建议保持 cuda:0**。"
                        "不要用 'auto'——它会把模型拆到多卡引发 "
                        "'Expected all tensors on same device' 报错；"
                        "真正的多卡并行请用 run_infer_multi_gpu.sh（多进程方式）。")
    # --- 多卡并行相关（与官方 inference_jsonl.py 保持一致的接口）---
    p.add_argument("--rank", type=int, default=0,
                   help="当前进程的 rank（从 0 开始），仅处理 idx %% world_size == rank 的样本")
    p.add_argument("--world-size", type=int, default=1,
                   help="总进程数（常为 GPU 数）；为 1 时等价于单卡推理")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.world_size < 1:
        raise ValueError(f"--world-size 必须 >= 1，当前: {args.world_size}")
    if not (0 <= args.rank < args.world_size):
        raise ValueError(
            f"--rank 必须在 [0, world_size) 范围内，当前 rank={args.rank}, world_size={args.world_size}"
        )

    model_dir = args.model.strip()
    if not (Path(model_dir) / "config.json").exists():
        raise FileNotFoundError(
            f"--model 需要指向完整模型目录（应含 config.json）；当前是: {model_dir}"
        )

    # 多卡并行时：输出自动加 .rank{N} 后缀，防止进程互相覆写。
    # 单卡（world_size==1）时保持原名，与旧行为兼容。
    output_path = Path(args.output)
    if args.world_size > 1:
        stem = output_path.stem
        suffix = output_path.suffix  # 包含前导点，如 ".jsonl"
        output_path = output_path.with_name(f"{stem}.rank{args.rank}{suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[infer_test] data       = {args.data}", flush=True)
    print(f"[infer_test] model      = {model_dir}", flush=True)
    print(f"[infer_test] output     = {output_path}", flush=True)
    print(f"[infer_test] batch_size = {args.batch_size}", flush=True)
    print(f"[infer_test] max_tokens = {args.max_tokens}", flush=True)
    print(f"[infer_test] device_map = {args.device_map}", flush=True)
    print(f"[infer_test] rank/world = {args.rank}/{args.world_size}", flush=True)

    all_samples = load_test_jsonl(args.data)
    # stride 切分：与官方 run_infer_multi_gpu.sh 完全一致（idx %% world_size == rank）
    if args.world_size > 1:
        samples = [s for idx, s in enumerate(all_samples) if idx % args.world_size == args.rank]
        print(
            f"[infer_test] shard: total={len(all_samples)}, this rank={len(samples)}",
            flush=True,
        )
    else:
        samples = all_samples
        print(f"[infer_test] loaded {len(samples)} samples", flush=True)

    # 延迟 import，避免无 GPU/无依赖时脚本一加载就报错
    from qwen_asr import Qwen3ASRModel  # noqa: F401

    print(f"[infer_test] loading Qwen3-ASR from {model_dir} ...", flush=True)
    asr = Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
    )

    done = 0
    total = len(samples)
    with output_path.open("w", encoding="utf-8") as fout:
        for start in range(0, total, args.batch_size):
            batch = samples[start:start + args.batch_size]
            audio_paths = [s["audio_path"] for s in batch]

            try:
                pred_full_list = infer_raw(asr, audio_paths)
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                print(
                    f"[error] batch {start}-{start + len(batch) - 1} failed: {err_msg}",
                    file=sys.stderr, flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                # 该 batch 全部标记 error 后继续
                for sample in batch:
                    result = dict(sample)
                    result.update({
                        "pred_full":    "",
                        "pred_text":    "",
                        "pred_dialect": "",
                        "error":        err_msg,
                    })
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                done += len(batch)
                continue

            for sample, pred_full in zip(batch, pred_full_list):
                pred = split_asr_content(pred_full)
                result = dict(sample)
                result.update({
                    "pred_full":    (pred_full or "").strip(),
                    "pred_text":    pred["text"],
                    "pred_dialect": pred["dialect"],
                    "error":        "",
                })
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[infer_test] progress {done}/{total}", flush=True)
            fout.flush()

    print(f"[infer_test] done. saved -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
