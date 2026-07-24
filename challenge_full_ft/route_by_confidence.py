#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
route_by_confidence.py
=======================

基于 LID 置信度对已有的 VC v2 联合模型预测进行"路由 + 专家改写"：

流程
----
1. 读取 baseline 置信度预测（默认 outputs_vc_v2/pred_test_conf.jsonl）。
2. 按方言查路由表：命中且 dialect_conf >= τ 的样本 → 加入对应专用模型的改写队列；
   其余样本原样保留联合模型的 pred_full / pred_text / pred_dialect。
3. 顺序加载每个专用模型（用完即释放显存），批量改写对应样本。
4. 按 baseline 原顺序输出 pred_test_routed.jsonl；每条样本追加
   route_used / route_model_ckpt / route_reason 字段（fallback 时额外 route_fallback / route_error）。

用法
----
python route_by_confidence.py \\
    --base_pred_jsonl outputs_vc_v2/pred_test_conf.jsonl \\
    --output outputs_hybrid_specialist/pred_test_routed.jsonl \\
    --tau_profile p95 \\
    --batch-size 32 --max-tokens 512

多卡本任务用不上：每个专用模型的改写队列都很小（≤51 条），单卡 batch 推理已经足够快。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# 常量：字段解析（与 infer_test.py 完全一致，避免 import 循环依赖）
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
        "full":   content,
        "prefix": prefix.strip(),
        "dialect": dialect,
        "text":   text.strip(),
    }


# =============================================================================
# 默认路由表（confidence_report.txt Section 3 的 P>=95% / P>=99% 推荐阈值）
# =============================================================================
# 用绝对路径记录 ckpt，避免脚本工作目录切换时找不到
_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_TAU_TABLES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "p95": {
        "wuyu": {
            "tau":  0.9629,
            "ckpt": str(_SCRIPT_DIR / "outputs_wuyu_stage2" / "checkpoint-50"),
            "name": "wuyu_specialist",
        },
        "kejia": {
            "tau":  0.7886,
            "ckpt": str(_SCRIPT_DIR / "outputs_specialist" / "kejia" / "checkpoint-150"),
            "name": "kejia_specialist",
        },
        "nanchang": {
            "tau":  0.9956,
            "ckpt": str(_SCRIPT_DIR / "outputs_specialist" / "nanchang" / "checkpoint-150"),
            "name": "nanchang_specialist",
        },
    },
    "p99": {
        "wuyu": {
            "tau":  0.9845,
            "ckpt": str(_SCRIPT_DIR / "outputs_wuyu_stage2" / "checkpoint-50"),
            "name": "wuyu_specialist",
        },
        "kejia": {
            "tau":  0.8626,
            "ckpt": str(_SCRIPT_DIR / "outputs_specialist" / "kejia" / "checkpoint-150"),
            "name": "kejia_specialist",
        },
        "nanchang": {
            "tau":  0.9979,
            "ckpt": str(_SCRIPT_DIR / "outputs_specialist" / "nanchang" / "checkpoint-150"),
            "name": "nanchang_specialist",
        },
    },
}


# =============================================================================
# baseline 加载 + 字段校验（对应任务 2）
# =============================================================================
REQUIRED_FIELDS = (
    "utt_id", "audio_path",
    "ref_full", "ref_text", "ref_dialect",
    "pred_full", "pred_text", "pred_dialect",
    "dialect_conf", "error",
)


def load_base_predictions(path: str) -> List[Dict[str, Any]]:
    """严格校验字段的 baseline 加载器。任一必填字段缺失即报错退出。"""
    p = Path(path)
    if not p.is_file():
        print(f"[route] ERROR: base_pred_jsonl not found: {path}", file=sys.stderr)
        sys.exit(1)

    samples: List[Dict[str, Any]] = []
    missing_field_row: Optional[int] = None
    missing_field_name: Optional[str] = None
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[route] ERROR: invalid json at {path}:{line_no}: {e}", file=sys.stderr)
                sys.exit(1)
            for fld in REQUIRED_FIELDS:
                if fld not in obj:
                    missing_field_row, missing_field_name = line_no, fld
                    break
            if missing_field_name is not None:
                break
            samples.append(obj)

    if missing_field_name is not None:
        print(
            f"[route] ERROR: base_pred_jsonl 缺少必填字段 '{missing_field_name}' "
            f"（行 {missing_field_row}, 文件 {path}）。请确认是否是 infer_test_with_confidence.py 的产物。",
            file=sys.stderr,
        )
        sys.exit(1)

    if not samples:
        print(f"[route] ERROR: base_pred_jsonl 为空: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"[route] loaded {len(samples)} base predictions from {path}", flush=True)
    return samples


# =============================================================================
# 路由表构建（对应任务 3）
# =============================================================================
def load_route_table(
    tau_profile: str,
    route_config_path: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """
    构建路由表 {pred_dialect: {tau, ckpt, name}}。
    优先级：--route_config YAML/JSON > --tau_profile 内置默认。
    """
    if tau_profile not in DEFAULT_TAU_TABLES:
        raise ValueError(
            f"--tau_profile 只允许 {list(DEFAULT_TAU_TABLES.keys())}, got {tau_profile}"
        )
    table = {k: dict(v) for k, v in DEFAULT_TAU_TABLES[tau_profile].items()}

    if route_config_path:
        cfg_path = Path(route_config_path)
        if not cfg_path.is_file():
            print(f"[route] ERROR: --route_config not found: {route_config_path}", file=sys.stderr)
            sys.exit(1)
        raw = cfg_path.read_text(encoding="utf-8")
        try:
            if cfg_path.suffix.lower() in (".yaml", ".yml"):
                import yaml  # 惰性 import
                user_cfg = yaml.safe_load(raw) or {}
            else:
                user_cfg = json.loads(raw)
        except Exception as e:
            print(f"[route] ERROR: parse --route_config failed: {e}", file=sys.stderr)
            sys.exit(1)

        if not isinstance(user_cfg, dict):
            print(
                f"[route] ERROR: --route_config 顶层必须是 dict，得到 {type(user_cfg)}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 允许 YAML 直接完全替换，也允许仅覆盖某几个字段
        for dialect, entry in user_cfg.items():
            if not isinstance(entry, dict):
                print(f"[route] ERROR: config[{dialect!r}] 必须是 dict", file=sys.stderr)
                sys.exit(1)
            if dialect in table:
                table[dialect].update(entry)
            else:
                # 新增方言必须完整
                needed = {"tau", "ckpt", "name"}
                missing = needed - set(entry.keys())
                if missing:
                    print(
                        f"[route] ERROR: config 中新增方言 {dialect!r} 必须包含字段 {needed}，缺 {missing}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                table[dialect] = dict(entry)

    # 校验 ckpt 是否可用（dry_run 时也校验，防止后续跑 8h 才发现路径写错）
    for dialect, entry in table.items():
        if "tau" not in entry or "ckpt" not in entry or "name" not in entry:
            print(f"[route] ERROR: route table entry {dialect} 缺字段: {entry}", file=sys.stderr)
            sys.exit(1)
        try:
            entry["tau"] = float(entry["tau"])
        except Exception:
            print(f"[route] ERROR: tau 必须是 float, got {entry['tau']!r}", file=sys.stderr)
            sys.exit(1)
        ckpt_dir = Path(entry["ckpt"])
        if not (ckpt_dir / "config.json").is_file():
            print(
                f"[route] ERROR: {dialect} 的 ckpt 目录不完整（缺 config.json）: {ckpt_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    return table


# =============================================================================
# 路由决策（对应任务 4）
# =============================================================================
def decide_routes(
    samples: List[Dict[str, Any]],
    route_table: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]], Dict[str, int]]:
    """
    对每条样本打路由标签，返回:
      routed_samples : 拷贝后的样本列表（已追加 route_used/route_model_ckpt/route_reason）
      queues         : {dialect: [idx, idx, ...]}  改写队列（idx 是 routed_samples 中的下标）
      stats          : 汇总统计
    """
    routed: List[Dict[str, Any]] = []
    queues: Dict[str, List[int]] = defaultdict(list)
    stats: Dict[str, int] = Counter()

    for i, obj in enumerate(samples):
        item = dict(obj)  # 浅拷贝
        stats["total"] += 1

        err = (item.get("error") or "").strip()
        if err:
            item["route_used"] = "vc_v2"
            item["route_model_ckpt"] = ""
            item["route_reason"] = f"error != '', keep vc_v2: {err[:80]}"
            stats["kept_error"] += 1
            routed.append(item)
            continue

        pd = (item.get("pred_dialect") or "").strip()
        try:
            conf = float(item.get("dialect_conf", 0.0))
        except Exception:
            conf = 0.0

        if pd not in route_table:
            item["route_used"] = "vc_v2"
            item["route_model_ckpt"] = ""
            item["route_reason"] = f"pred_dialect={pd!r} not in table"
            stats["kept_not_in_table"] += 1
            routed.append(item)
            continue

        tau = float(route_table[pd]["tau"])
        if conf < tau:
            item["route_used"] = "vc_v2"
            item["route_model_ckpt"] = ""
            item["route_reason"] = f"pred_dialect={pd}, conf={conf:.4f} < tau={tau:.4f}"
            stats[f"kept_below_tau[{pd}]"] += 1
            stats["kept_below_tau_total"] += 1
            routed.append(item)
            continue

        # 命中：先记路由信息（pred_* 稍后被专用模型覆盖）
        item["route_used"] = route_table[pd]["name"]
        item["route_model_ckpt"] = route_table[pd]["ckpt"]
        item["route_reason"] = (
            f"routed by pred_dialect={pd}, conf={conf:.4f} >= tau={tau:.4f}"
        )
        idx = len(routed)
        routed.append(item)
        queues[pd].append(idx)
        stats[f"routed[{pd}]"] += 1
        stats["routed_total"] += 1

    return routed, queues, stats


def print_route_summary(
    stats: Dict[str, int],
    queues: Dict[str, List[int]],
    route_table: Dict[str, Dict[str, Any]],
) -> None:
    print("=" * 60, flush=True)
    print(f"[route] total samples          : {stats['total']}", flush=True)
    print(f"[route] kept as vc_v2 (error)  : {stats.get('kept_error', 0)}", flush=True)
    print(f"[route] kept vc_v2 (not_in_tbl): {stats.get('kept_not_in_table', 0)}", flush=True)
    print(f"[route] kept vc_v2 (below tau) : {stats.get('kept_below_tau_total', 0)}", flush=True)
    print(f"[route] routed to specialists  : {stats.get('routed_total', 0)}", flush=True)
    for dialect in sorted(route_table.keys()):
        n = len(queues.get(dialect, []))
        tau = route_table[dialect]["tau"]
        ckpt = route_table[dialect]["ckpt"]
        print(f"    - {dialect:<10s} tau={tau:.4f}  queue={n:>4d}   ckpt={ckpt}", flush=True)
    print("=" * 60, flush=True)


# =============================================================================
# 专家模型批量推理（对应任务 5）
# =============================================================================
def run_specialist_batch(
    ckpt_path: str,
    samples_view: List[Dict[str, Any]],
    batch_size: int,
    max_tokens: int,
    device_map: str,
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    加载一个专用模型，对 samples_view 中所有样本做批量推理。
    返回 [(pred_full, err_msg_or_None), ...]，长度与输入一致。

    err_msg_or_None:
        - None 表示成功；此时 pred_full 是新预测。
        - 非 None 表示该 batch 整体或单样本失败；pred_full=None，调用方需 fallback。

    显存管理：函数返回前把模型和 CUDA 缓存释放干净。
    """
    import torch  # 延迟 import

    from qwen_asr import Qwen3ASRModel  # type: ignore
    from qwen_asr.inference.utils import normalize_audios  # type: ignore

    print(f"[specialist] loading {ckpt_path} on {device_map} ...", flush=True)
    asr = Qwen3ASRModel.from_pretrained(
        ckpt_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_tokens,
    )

    results: List[Tuple[Optional[str], Optional[str]]] = [(None, None)] * len(samples_view)

    total = len(samples_view)
    done = 0
    try:
        for start in range(0, total, batch_size):
            batch = samples_view[start:start + batch_size]
            audio_paths = [s["audio_path"] for s in batch]
            try:
                wavs = normalize_audios(audio_paths)
                contexts = [""] * len(wavs)
                languages: List[Optional[str]] = [None] * len(wavs)
                preds = asr._infer_asr(contexts, wavs, languages)
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                print(
                    f"[specialist][ERROR] batch {start}-{start + len(batch) - 1} failed: {err_msg}",
                    file=sys.stderr, flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                # 整个 batch fallback
                for j in range(len(batch)):
                    results[start + j] = (None, err_msg)
                done += len(batch)
                continue

            for j, pred_full in enumerate(preds):
                results[start + j] = ((pred_full or "").strip(), None)
                done += 1
            if done % 20 == 0 or done == total:
                print(f"[specialist] progress {done}/{total}", flush=True)
    finally:
        # 无论成功失败都释放显存
        del asr
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[specialist] released {ckpt_path}", flush=True)

    return results


# =============================================================================
# 主流程
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid specialist inference via LID-confidence routing."
    )
    p.add_argument("--base_pred_jsonl", type=str,
                   default=str(_SCRIPT_DIR / "outputs_vc_v2" / "pred_test_conf.jsonl"),
                   help="联合模型的置信度预测 JSONL（默认 outputs_vc_v2/pred_test_conf.jsonl）")
    p.add_argument("--output", type=str,
                   default=str(_SCRIPT_DIR / "outputs_hybrid_specialist" / "pred_test_routed.jsonl"),
                   help="路由后的最终预测 JSONL")
    p.add_argument("--tau_profile", type=str, default="p95",
                   choices=list(DEFAULT_TAU_TABLES.keys()),
                   help="默认阈值 profile: p95 / p99")
    p.add_argument("--route_config", type=str, default="",
                   help="可选：YAML/JSON 覆写路由表，优先级高于 --tau_profile")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--device-map", type=str, default="cuda:0",
                   help="每个专用模型固定加载到 cuda:0（不要用 auto）")
    p.add_argument("--dry_run", type=int, default=0,
                   help="1 = 只做路由决策与统计，不加载任何专用模型")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[route] base_pred_jsonl = {args.base_pred_jsonl}", flush=True)
    print(f"[route] output          = {args.output}", flush=True)
    print(f"[route] tau_profile     = {args.tau_profile}", flush=True)
    print(f"[route] route_config    = {args.route_config or '(none)'}", flush=True)
    print(f"[route] batch_size      = {args.batch_size}", flush=True)
    print(f"[route] max_tokens      = {args.max_tokens}", flush=True)
    print(f"[route] device_map      = {args.device_map}", flush=True)
    print(f"[route] dry_run         = {bool(args.dry_run)}", flush=True)

    # 1) baseline + 路由表
    base_samples = load_base_predictions(args.base_pred_jsonl)
    route_table = load_route_table(args.tau_profile, args.route_config or None)

    print("[route] route table:", flush=True)
    for dialect, entry in route_table.items():
        print(
            f"    - {dialect:<10s} tau={entry['tau']:.4f}  name={entry['name']}  ckpt={entry['ckpt']}",
            flush=True,
        )

    # 2) 路由决策
    routed, queues, stats = decide_routes(base_samples, route_table)
    print_route_summary(stats, queues, route_table)

    # 3) 专用模型改写（非 dry_run）
    fallback_count = 0
    if not args.dry_run:
        for dialect in sorted(queues.keys()):
            idxs = queues[dialect]
            if not idxs:
                continue
            ckpt = route_table[dialect]["ckpt"]
            print(f"[route] >>> specialist rewrite: {dialect} ({len(idxs)} samples)", flush=True)
            view = [routed[i] for i in idxs]
            results = run_specialist_batch(
                ckpt_path=ckpt,
                samples_view=view,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                device_map=args.device_map,
            )
            # 回填到 routed
            for local_j, (pred_full, err) in enumerate(results):
                idx = idxs[local_j]
                if err is not None or pred_full is None:
                    # fallback：保留联合模型原预测（pred_full/pred_text/pred_dialect 已经在 routed[idx]）
                    routed[idx]["route_fallback"] = True
                    routed[idx]["route_error"] = err or "empty prediction"
                    fallback_count += 1
                    continue
                parsed = split_asr_content(pred_full)
                routed[idx]["pred_full"] = pred_full
                routed[idx]["pred_text"] = parsed["text"]
                routed[idx]["pred_dialect"] = parsed["dialect"]
    else:
        print("[route] dry_run=1, skip specialist inference", flush=True)

    # 4) 落盘（任务 6）
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for item in routed:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 5) 汇总
    print("=" * 60, flush=True)
    print(f"[route] wrote {len(routed)} rows -> {out_path}", flush=True)
    if not args.dry_run:
        print(f"[route] fallback (kept vc_v2 due to error): {fallback_count}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
