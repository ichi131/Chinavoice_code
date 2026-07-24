#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_test_with_confidence.py
==============================

**方案 A（零成本置信度）** 的推理脚本，与原 `infer_test.py` 完全解耦、平行存在，
**不会影响任何原有代码**。它做两件事：

1. 复用 Qwen3ASRModel 的模型 / processor 加载，输入输出跟原脚本一致；
2. 自己实现 batch 生成循环，打开 `output_scores=True` + `return_dict_in_generate=True`，
   从每个新生成 token 的 logits 中抽取置信度。

输出 JSONL 每行字段（在原字段基础上追加，向后兼容）：
    {
        "utt_id":            "chaoshan_000459",
        "audio_path":        "...wav",
        "ref_full":          "language Chinese chaoshan<asr_text>xxx",
        "ref_text":          "xxx",
        "ref_dialect":       "chaoshan",
        "pred_full":         "language Chinese chaoshan<asr_text>yyy",
        "pred_text":         "yyy",
        "pred_dialect":      "chaoshan",
        "error":             "",

        # ---- 新增置信度字段 ----
        "dialect_conf":      0.9873,   # 方言字段联合概率（多 token 概率相乘），主指标
        "dialect_logprob":   -0.0128,  # 方言字段联合 log-prob（多 token 概率相加）
        "dialect_num_tokens": 3,       # 方言字段被拆成了多少个 subword token
        "text_avg_logprob":  -0.2144,  # 转写文本部分每个 token 的平均 log-prob
        "text_num_tokens":   17,       # 转写文本部分的 token 数
        "seq_avg_logprob":   -0.1832,  # 整段生成序列的平均 log-prob（备用）
    }

置信度字段的语义：
- `dialect_conf`：**你要的 LID 预测置信度**。当模型预测该样本为 X 方言时，
  它把这一决策的可信度打成了这个概率。范围 (0, 1]，越接近 1 越自信。
- `dialect_conf` 是**未校准**的原始概率。生成式模型经常"过度自信"，
  真正卡阈值前建议先做 temperature scaling / isotonic regression 校准，
  或者直接在 val 集上按类扫阈值（不依赖绝对值）。

用法（单卡）：
    python infer_test_with_confidence.py \
        --model  outputs_vc_v2/checkpoint-500 \
        --data   data_vc_v2/test.jsonl \
        --output outputs_vc_v2/pred_test_conf.jsonl \
        --batch-size 32

多卡：请配套使用 `run_infer_confidence_multi_gpu.sh`（与
`run_infer_multi_gpu.sh` 结构完全一致，只是换了脚本文件名）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# 解析辅助：与 infer_test.py 保持一致
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
# 数据读取（与 infer_test.py 一致）
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
# 核心：自定义带 scores 的推理循环
# =============================================================================
def _extract_confidence_from_scores(
    generated_ids: torch.Tensor,      # [T_new]，一条样本本次生成的 token id 序列（不含 prompt）
    scores: List[torch.Tensor],       # 长度 T_new，每个 [vocab]（已切到该样本这一行）
    asr_text_token_id: int,
    eos_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    从一条样本的生成序列和 per-step logits 中，抽取方言字段 / 文本字段的置信度。

    生成序列结构（示意）：
        [ dialect_tok_1, dialect_tok_2, ..., <asr_text>, text_tok_1, ..., <eos> ]

    我们把 `<asr_text>` 之前的所有 token 视为 dialect 字段（可能包含 "language"
    "Chinese" 这些前缀 token，但在贪心解码下这些前缀模型都极自信、prob ≈ 1，
    不影响判断；而 dialect 词本身若模型犹豫，联合概率会立刻掉下来）。
    """
    if generated_ids.numel() == 0 or len(scores) == 0:
        return {
            "dialect_conf":        0.0,
            "dialect_logprob":     float("-inf"),
            "dialect_num_tokens":  0,
            "text_avg_logprob":    float("-inf"),
            "text_num_tokens":     0,
            "seq_avg_logprob":     float("-inf"),
        }

    device = scores[0].device
    T_new = min(generated_ids.shape[0], len(scores))

    # 找 <asr_text> 位置
    ids_cpu = generated_ids[:T_new].tolist()
    try:
        asr_pos = ids_cpu.index(asr_text_token_id)
    except ValueError:
        asr_pos = -1  # 没找到，说明生成异常，退化处理

    # 逐 token 计算 log softmax 概率
    step_logprobs: List[float] = []
    for step in range(T_new):
        lg = scores[step]           # [vocab]
        # 数值稳定的 log_softmax
        lp = torch.log_softmax(lg.float(), dim=-1)
        chosen = int(ids_cpu[step])
        step_logprobs.append(float(lp[chosen].item()))

    # 按 <asr_text> 切成前后两段
    if asr_pos >= 0:
        dialect_logprobs = step_logprobs[:asr_pos]        # 不含 <asr_text>
        text_logprobs = step_logprobs[asr_pos + 1:]        # 不含 <asr_text>
    else:
        dialect_logprobs = step_logprobs
        text_logprobs = []

    # 剔除 text 段末尾的 eos（如果有）
    if eos_token_ids and text_logprobs:
        text_start = asr_pos + 1 if asr_pos >= 0 else 0
        # 从 ids_cpu 里同步定位 text 段的 token 序列
        text_ids = ids_cpu[text_start:text_start + len(text_logprobs)]
        while text_ids and text_ids[-1] in eos_token_ids:
            text_ids.pop()
            text_logprobs.pop()

    dialect_sum = sum(dialect_logprobs) if dialect_logprobs else float("-inf")
    dialect_conf = math.exp(dialect_sum) if dialect_sum > -700 else 0.0

    text_avg = (sum(text_logprobs) / len(text_logprobs)) if text_logprobs else float("-inf")
    seq_avg = (sum(step_logprobs) / len(step_logprobs)) if step_logprobs else float("-inf")

    return {
        "dialect_conf":        dialect_conf,
        "dialect_logprob":     dialect_sum if dialect_sum != float("-inf") else -1e9,
        "dialect_num_tokens":  len(dialect_logprobs),
        "text_avg_logprob":    text_avg if text_avg != float("-inf") else -1e9,
        "text_num_tokens":     len(text_logprobs),
        "seq_avg_logprob":     seq_avg if seq_avg != float("-inf") else -1e9,
    }


def _find_eos_ids(model, processor) -> List[int]:
    """尽量收集所有 eos 相关 token id，用于剔除末尾。"""
    eos_ids: List[int] = []
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        eid = getattr(gen_cfg, "eos_token_id", None)
        if eid is None:
            pass
        elif isinstance(eid, (list, tuple)):
            eos_ids.extend([int(x) for x in eid])
        else:
            eos_ids.append(int(eid))
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.eos_token_id is not None:
        eos_ids.append(int(tok.eos_token_id))
    return sorted(set(eos_ids))


def infer_batch_with_confidence(
    asr,
    audio_paths: List[str],
    asr_text_token_id: int,
    eos_token_ids: List[int],
    max_new_tokens: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    自定义 batch 推理：拿到 pred_full 字符串 + 置信度统计。

    与 Qwen3ASRModel._infer_asr_transformers 保持一致的调用方式，只是加了
    output_scores / return_dict_in_generate。
    """
    from qwen_asr.inference.utils import normalize_audios

    wavs = normalize_audios(audio_paths)
    contexts = [""] * len(wavs)
    languages: List[Optional[str]] = [None] * len(wavs)

    # 构造 prompt（复用模型内部的 _build_text_prompt）
    texts = [
        asr._build_text_prompt(context=c, force_language=fl)
        for c, fl in zip(contexts, languages)
    ]
    processor = asr.processor
    model = asr.model

    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        # 注意：Qwen3-ASR 的 model.generate() 内部已硬编码
        # `return_dict_in_generate=True`，这里**不能再传**，否则会报
        # "got multiple values for keyword argument 'return_dict_in_generate'"。
        # 我们只额外打开 output_scores，用于拿到 per-step logits 做置信度。
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
        )

    # `sequences` 形状 [B, prompt_len + T_new]（左 padding 情况下 prompt_len 是全局对齐后的）
    seqs = gen_out.sequences
    scores = gen_out.scores  # tuple[T_new] of [B, vocab]

    if scores is None or len(scores) == 0:
        raise RuntimeError(
            "generate() 没有返回 scores——请检查 transformers 版本是否支持 "
            "output_scores=True。当前 gen_out 字段: "
            f"{list(gen_out.keys()) if hasattr(gen_out, 'keys') else type(gen_out)}"
        )

    B = seqs.shape[0]
    new_tokens = seqs[:, prompt_len:]  # [B, T_new]

    # decode 出字符串（用同款 batch_decode）
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    outs: List[Tuple[str, Dict[str, Any]]] = []
    for b in range(B):
        # 逐样本抽 scores 的第 b 行
        per_sample_scores = [scores[t][b] for t in range(len(scores))]
        conf_info = _extract_confidence_from_scores(
            generated_ids=new_tokens[b],
            scores=per_sample_scores,
            asr_text_token_id=asr_text_token_id,
            eos_token_ids=eos_token_ids,
        )
        outs.append((decoded[b], conf_info))
    return outs


# =============================================================================
# 命令行
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch inference with per-sample confidence (Plan A)."
    )
    p.add_argument("--model", type=str, required=True,
                   help="模型或 ckpt 目录（必须含 config.json 等完整 HF 文件）")
    p.add_argument("--data", type=str, required=True,
                   help="prepare_data.py 产出的 test.jsonl")
    p.add_argument("--output", type=str, required=True,
                   help="预测 JSONL 保存路径（建议：*_conf.jsonl）")
    p.add_argument("--batch-size", type=int, default=16,
                   help="推理 batch；带 scores 后显存翻倍，建议比原来减半")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="每条音频最多生成 token 数")
    p.add_argument("--device-map", type=str, default="cuda:0",
                   help="transformers device_map，强烈建议 cuda:0")
    p.add_argument("--rank", type=int, default=0,
                   help="当前进程 rank（idx %% world_size == rank）")
    p.add_argument("--world-size", type=int, default=1,
                   help="总进程数（常为 GPU 数）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.world_size < 1:
        raise ValueError(f"--world-size 必须 >= 1，当前: {args.world_size}")
    if not (0 <= args.rank < args.world_size):
        raise ValueError(
            f"--rank 必须在 [0, world_size) 范围内，rank={args.rank}, world_size={args.world_size}"
        )

    model_dir = args.model.strip()
    if not (Path(model_dir) / "config.json").exists():
        raise FileNotFoundError(
            f"--model 需指向完整模型目录（含 config.json），当前: {model_dir}"
        )

    output_path = Path(args.output)
    if args.world_size > 1:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.with_name(f"{stem}.rank{args.rank}{suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[infer_conf] data       = {args.data}", flush=True)
    print(f"[infer_conf] model      = {model_dir}", flush=True)
    print(f"[infer_conf] output     = {output_path}", flush=True)
    print(f"[infer_conf] batch_size = {args.batch_size}", flush=True)
    print(f"[infer_conf] max_tokens = {args.max_tokens}", flush=True)
    print(f"[infer_conf] device_map = {args.device_map}", flush=True)
    print(f"[infer_conf] rank/world = {args.rank}/{args.world_size}", flush=True)

    all_samples = load_test_jsonl(args.data)
    if args.world_size > 1:
        samples = [s for i, s in enumerate(all_samples) if i % args.world_size == args.rank]
        print(f"[infer_conf] shard: total={len(all_samples)}, this rank={len(samples)}", flush=True)
    else:
        samples = all_samples
        print(f"[infer_conf] loaded {len(samples)} samples", flush=True)

    from qwen_asr import Qwen3ASRModel  # noqa: F401

    print(f"[infer_conf] loading Qwen3-ASR from {model_dir} ...", flush=True)
    asr = Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
    )

    # 定位 <asr_text> token id
    tok = asr.processor.tokenizer
    asr_text_tok_id = tok.convert_tokens_to_ids(ASR_MARKER)
    if asr_text_tok_id is None or asr_text_tok_id == tok.unk_token_id:
        raise RuntimeError(
            f"tokenizer 里找不到 special token '{ASR_MARKER}'，请检查 ckpt 是否完整"
        )
    eos_ids = _find_eos_ids(asr.model, asr.processor)
    print(f"[infer_conf] <asr_text> token id = {asr_text_tok_id}", flush=True)
    print(f"[infer_conf] eos token ids       = {eos_ids}", flush=True)

    done = 0
    total = len(samples)
    with output_path.open("w", encoding="utf-8") as fout:
        for start in range(0, total, args.batch_size):
            batch = samples[start:start + args.batch_size]
            audio_paths = [s["audio_path"] for s in batch]
            try:
                results = infer_batch_with_confidence(
                    asr=asr,
                    audio_paths=audio_paths,
                    asr_text_token_id=asr_text_tok_id,
                    eos_token_ids=eos_ids,
                    max_new_tokens=args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                print(
                    f"[error] batch {start}-{start + len(batch) - 1} failed: {err_msg}",
                    file=sys.stderr, flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                for sample in batch:
                    result = dict(sample)
                    result.update({
                        "pred_full":            "",
                        "pred_text":            "",
                        "pred_dialect":         "",
                        "error":                err_msg,
                        "dialect_conf":         0.0,
                        "dialect_logprob":      -1e9,
                        "dialect_num_tokens":   0,
                        "text_avg_logprob":     -1e9,
                        "text_num_tokens":      0,
                        "seq_avg_logprob":      -1e9,
                    })
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                done += len(batch)
                continue

            for sample, (pred_full, conf_info) in zip(batch, results):
                pred = split_asr_content(pred_full)
                result = dict(sample)
                result.update({
                    "pred_full":    (pred_full or "").strip(),
                    "pred_text":    pred["text"],
                    "pred_dialect": pred["dialect"],
                    "error":        "",
                    **conf_info,
                })
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[infer_conf] progress {done}/{total}", flush=True)
            fout.flush()

    print(f"[infer_conf] done. saved -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
