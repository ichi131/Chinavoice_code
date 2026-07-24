#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""LID 超参贪心式扫描主控入口。

流程：
1. 加载 YAML 扫描配置。
2. 决定顶层扫描目录（新扫描 = exp/<prefix>_<ts>/；--resume = 已有目录）。
3. 按 sweep_order 逐维度扫描：
   - 每个候选值创建实验子目录 `<idx>_<name>__<value>/`
   - config.json → 训练 → 若 best.pt 存在则推理 + 评分 → 写 test_accuracy.json
   - 维度扫完后选出准确率最高的候选，固化到当前 baseline
   - keep_only_best_ckpt=true 时删除同维度非最优实验的 best.pt
   - 更新 sweep_progress.json
4. 全部完成后生成 sweep_summary.md 与 best_hparams.json。

强隔离约束：所有 IO 都写入扫描顶层目录内部，绝不修改现有训练/推理脚本。
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 让脚本能独立运行：把当前脚本所在目录加入 sys.path 以 import 兄弟模块
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config_loader import (  # noqa: E402
    DimensionPlan,
    SweepPlan,
    load_sweep_plan,
)
from runner import run_inference_and_score, run_training  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("sweep.main")

# 项目根目录 = 当前脚本向上 3 层：sweep/ -> lid_chinavoices/ -> examples_train/ -> <root>
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])


# =============================================================================
# 通用工具
# =============================================================================

def _format_value_for_dir(value: Any) -> str:
    """把候选值格式化为文件名安全字符串。"""
    if isinstance(value, float):
        # 保留完整精度，替换非法字符
        s = repr(value)
    else:
        s = str(value)
    s = s.replace("+", "").replace("/", "_").replace(" ", "")
    return s


def _exp_subdir_name(dim_index: int, dim_name: str, value: Any) -> str:
    return f"{dim_index:02d}_{dim_name}__{_format_value_for_dir(value)}"


def _atomic_write_json(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _hparams_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """严格比较两组超参是否完全一致（键集合 & 逐键的值等价性）。

    - 只对 baseline 白名单里的键进行比较，忽略非超参字段
    - float 用 `math.isclose`（相对/绝对容差都极小）避免因 YAML/repr 精度导致误判
    - int/str/bool 直接用 `==`
    """
    if set(a.keys()) != set(b.keys()):
        return False
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, float) or isinstance(vb, float):
            try:
                fa = float(va)
                fb = float(vb)
            except (TypeError, ValueError):
                return False
            # 相对容差 1e-12：足以判定 1e-3 == 1.0e-3 == 0.001
            import math as _math
            if not _math.isclose(fa, fb, rel_tol=1e-12, abs_tol=0.0):
                return False
        else:
            if va != vb:
                return False
    return True


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("failed to read %s: %s", path, e)
        return None


def _resolve_sweep_dir(cli_sweep_dir: Optional[str], plan: SweepPlan) -> str:
    """决定顶层扫描目录路径（绝对路径）。"""
    if cli_sweep_dir:
        return os.path.abspath(cli_sweep_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = plan.run_options.base_output_dir
    if not os.path.isabs(base):
        base = os.path.join(PROJECT_ROOT, base)
    return os.path.abspath(os.path.join(base, f"{plan.run_options.sweep_dir_prefix}_{ts}"))


# =============================================================================
# 进度追踪
# =============================================================================

class ProgressTracker:
    """管理 sweep_progress.json 的读写。"""

    def __init__(self, sweep_dir: str):
        self.sweep_dir = sweep_dir
        self.path = os.path.join(sweep_dir, "sweep_progress.json")
        self.state: Dict[str, Any] = {
            "sweep_dir": sweep_dir,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_experiments": [],
            "per_dim_best": {},        # dim_name -> {value, accuracy, exp_dir, hparams}
            "current_baseline": {},
            "last_update": "",
        }

    def load_if_exists(self) -> bool:
        old = _read_json(self.path)
        if old:
            self.state.update(old)
            return True
        return False

    def flush(self) -> None:
        self.state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _atomic_write_json(self.path, self.state)

    def already_done(self, exp_dir: str) -> Optional[Dict[str, Any]]:
        """若该实验目录下已有 test_accuracy.json（--resume 场景），返回其结果。"""
        acc = _read_json(os.path.join(exp_dir, "test_accuracy.json"))
        return acc

    def record_experiment(self, entry: Dict[str, Any]) -> None:
        # 用 exp_dir 去重（--resume 二次记录时覆盖）
        seen = {e["exp_dir"]: i for i, e in enumerate(self.state["completed_experiments"])}
        if entry["exp_dir"] in seen:
            self.state["completed_experiments"][seen[entry["exp_dir"]]] = entry
        else:
            self.state["completed_experiments"].append(entry)

    def record_dim_best(self, dim_name: str, best: Dict[str, Any]) -> None:
        self.state["per_dim_best"][dim_name] = best

    def set_baseline(self, baseline: Dict[str, Any]) -> None:
        self.state["current_baseline"] = copy.deepcopy(baseline)


# =============================================================================
# 汇总报告
# =============================================================================

def _write_sweep_summary(sweep_dir: str, plan: SweepPlan, dim_results: List[Dict[str, Any]]) -> str:
    """生成 sweep_summary.md：表格展示每个维度的候选值与准确率。"""
    lines: List[str] = []
    lines.append(f"# LID 超参扫描汇总报告")
    lines.append("")
    lines.append(f"- 扫描目录：`{sweep_dir}`")
    lines.append(f"- 配置文件：`{plan.raw_config_path}`")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 维度总数：{len(plan.dimensions)}")
    lines.append("")

    # 汇总节省实验数（reused 状态）
    reused_total = 0
    for dim_res in dim_results:
        for cand in dim_res.get("candidates", []):
            if cand.get("status") == "reused":
                reused_total += 1
    if reused_total > 0:
        lines.append(f"- 复用节省实验次数：**{reused_total}**（候选值等于当前基线且已有可复用来源）")
        lines.append("")

    lines.append("## 逐维度扫描结果")
    lines.append("")
    for dim_res in dim_results:
        dim_name = dim_res["dim_name"]
        skipped = dim_res.get("skipped", False)
        lines.append(f"### 维度 {dim_res['dim_index']:02d} · `{dim_name}`" + ("（单候选，跳过扫描）" if skipped else ""))
        lines.append("")
        lines.append("| 候选值 | overall_acc | num_samples | status | 复用来源 | exp_dir |")
        lines.append("|---|---|---|---|---|---|")
        for cand in dim_res["candidates"]:
            acc = cand.get("overall_acc")
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
            marker = " ⭐" if cand.get("is_best") else ""
            reused_src = cand.get("reused_from_rel") or ""
            reused_src_str = f"`{reused_src}`" if reused_src else ""
            lines.append(
                f"| `{cand['value']}`{marker} | {acc_str} | {cand.get('num_samples', 0)} | "
                f"{cand.get('status', 'unknown')} | {reused_src_str} | `{cand.get('exp_dir_rel', '')}` |"
            )
        best_val = dim_res.get("best_value")
        best_acc = dim_res.get("best_acc")
        if best_acc is not None:
            lines.append("")
            lines.append(f"**选中：`{dim_name} = {best_val}` (overall_acc = {best_acc:.4f})**")
        else:
            lines.append("")
            lines.append(f"**选中：`{dim_name} = {best_val}` (无有效准确率，回落到原基线)**")
        lines.append("")

    lines.append("## 最终基线")
    lines.append("")
    lines.append("```json")
    final_baseline = dim_results[-1]["baseline_after"] if dim_results else {}
    lines.append(json.dumps(final_baseline, ensure_ascii=False, indent=2))
    lines.append("```")

    summary_path = os.path.join(sweep_dir, "sweep_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary_path


# =============================================================================
# 磁盘清理
# =============================================================================

def _cleanup_non_best_ckpts(dim_results_entry: Dict[str, Any]) -> None:
    """在维度扫描完成后，删除非最优实验的 best.pt / last.pt，节省磁盘。

    保留 test_accuracy.json / train.log / pred_test.jsonl / config.json 等文本产物。
    reused 实验没有 checkpoint 需要清理，天然跳过。
    """
    for cand in dim_results_entry["candidates"]:
        if cand.get("is_best"):
            continue
        if cand.get("status") == "reused":
            continue
        exp_dir = cand.get("exp_dir")
        if not exp_dir or not os.path.isdir(exp_dir):
            continue
        for ckpt_name in ("best.pt", "last.pt"):
            p = os.path.join(exp_dir, ckpt_name)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    logger.info("cleaned checkpoint: %s", p)
                except OSError as e:
                    logger.warning("cleanup failed for %s: %s", p, e)


# =============================================================================
# 单次实验执行
# =============================================================================

def _run_one_experiment(
    sweep_dir: str,
    dim: DimensionPlan,
    value: Any,
    baseline_snapshot: Dict[str, Any],
    plan: SweepPlan,
    tracker: ProgressTracker,
    resume: bool,
    reuse_from: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行 baseline + 单个候选值 的一次实验，返回结果字典。

    reuse_from: 若提供且非 None，则跳过训练+推理，直接复用来源实验的 test_accuracy.json 结果。
        结构：{
            "exp_dir": <绝对路径>,
            "overall_acc": <float>,
            "num_samples": <int>,
            "source_dim": <str>,       # 来源维度名，仅用于记录
        }
    """
    exp_name = _exp_subdir_name(dim.index, dim.name, value)
    exp_dir = os.path.join(sweep_dir, exp_name)

    # 合成本次实际生效的超参
    hparams = copy.deepcopy(baseline_snapshot)
    hparams[dim.name] = value

    # --resume：若已有完成的 test_accuracy.json，直接读取
    if resume:
        prev = tracker.already_done(exp_dir)
        if prev is not None and prev.get("status") in ("success", "reused"):
            logger.info("[resume] skip %s (overall_acc=%s status=%s)",
                        exp_name, prev.get("overall_acc"), prev.get("status"))
            return {
                "value": value,
                "exp_dir": exp_dir,
                "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
                "hparams": hparams,
                "overall_acc": prev.get("overall_acc"),
                "num_samples": prev.get("num_samples", 0),
                "status": prev.get("status", "success"),
                "message": prev.get("message", ""),
                "resumed": True,
                "reused_from_rel": prev.get("reused_from_rel", ""),
            }

    os.makedirs(exp_dir, exist_ok=True)

    # === 复用分支：候选值 == 当前 baseline 中该维度的取值，且存在上一维度的 winner ===
    if reuse_from is not None:
        source_exp_dir = reuse_from["exp_dir"]
        source_acc = reuse_from.get("overall_acc")
        source_num = reuse_from.get("num_samples", 0)
        source_dim_name = reuse_from.get("source_dim", "")
        source_rel = os.path.relpath(source_exp_dir, sweep_dir) if source_exp_dir else ""

        logger.info("=" * 78)
        logger.info("[reuse] dim=%s value=%s -> %s | 复用来源: %s (dim=%s, acc=%s)",
                    dim.name, value, exp_name, source_rel, source_dim_name, source_acc)
        logger.info("=" * 78)

        # 写 config.json（同正常路径的记录方式），额外标注 reused
        _atomic_write_json(os.path.join(exp_dir, "config.json"), {
            "dimension": dim.name,
            "dimension_index": dim.index,
            "candidate_value": value,
            "hparams": hparams,
            "fixed_params": plan.fixed_params,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reused": True,
        })

        # 写 reused_from.json 便于事后追溯
        _atomic_write_json(os.path.join(exp_dir, "reused_from.json"), {
            "source_exp_dir": source_exp_dir,
            "source_exp_dir_rel": source_rel,
            "source_dim": source_dim_name,
            "overall_acc": source_acc,
            "num_samples": source_num,
            "hparams": hparams,
            "note": "候选值等于当前基线且上一维度已产出等效实验，故直接复用其准确率，不重新训练/推理。",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # 写 test_accuracy.json（与正常实验字段保持一致，方便下游解析统一）
        acc_payload = {
            "overall_acc": source_acc,
            "per_class_acc": {},   # 不复制以避免过大；如需可从 source_exp_dir 追溯
            "num_samples": source_num,
            "num_pred_error": 0,
            "num_unknown_key": 0,
            "elapsed_sec": 0.0,
            "status": "reused",
            "message": f"reused from {source_rel}",
            "reused_from": source_exp_dir,
            "reused_from_rel": source_rel,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _atomic_write_json(os.path.join(exp_dir, "test_accuracy.json"), acc_payload)

        return {
            "value": value,
            "exp_dir": exp_dir,
            "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
            "hparams": hparams,
            "overall_acc": source_acc,
            "num_samples": source_num,
            "status": "reused",
            "message": f"reused from {source_rel}",
            "resumed": False,
            "reused_from_rel": source_rel,
        }

    # 写 config.json（本次实验实际超参 + 环境快照）
    _atomic_write_json(os.path.join(exp_dir, "config.json"), {
        "dimension": dim.name,
        "dimension_index": dim.index,
        "candidate_value": value,
        "hparams": hparams,
        "fixed_params": plan.fixed_params,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    logger.info("=" * 78)
    logger.info("[experiment] dim=%s value=%s -> %s", dim.name, value, exp_name)
    logger.info("=" * 78)

    # 训练
    train_result = run_training(
        exp_dir=exp_dir,
        hparams=hparams,
        fixed_params=plan.fixed_params,
        project_root=PROJECT_ROOT,
    )
    if train_result["status"] != "success":
        logger.error("[experiment] training failed for %s (exit=%s)", exp_name, train_result["exit_code"])
        _atomic_write_json(os.path.join(exp_dir, "test_accuracy.json"), {
            "overall_acc": None,
            "per_class_acc": {},
            "num_samples": 0,
            "elapsed_sec": train_result["elapsed_sec"],
            "status": "failed",
            "message": f"training failed exit_code={train_result['exit_code']}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return {
            "value": value,
            "exp_dir": exp_dir,
            "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
            "hparams": hparams,
            "overall_acc": None,
            "num_samples": 0,
            "status": "failed",
            "message": f"training failed exit_code={train_result['exit_code']}",
            "resumed": False,
        }

    # 推理 + 打分
    infer_result = run_inference_and_score(
        exp_dir=exp_dir,
        ckpt_path=train_result["best_pt"],
        test_jsonl=str(plan.fixed_params["test_jsonl"]),
        fixed_params=plan.fixed_params,
        project_root=PROJECT_ROOT,
        min_acc_warning_threshold=plan.run_options.min_acc_warning_threshold,
    )
    return {
        "value": value,
        "exp_dir": exp_dir,
        "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
        "hparams": hparams,
        "overall_acc": infer_result["overall_acc"],
        "num_samples": infer_result["num_samples"],
        "status": infer_result["status"],
        "message": infer_result.get("message", ""),
        "resumed": False,
    }


# =============================================================================
# 主入口
# =============================================================================

def _select_best_candidate(candidates: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Any]:
    """从候选结果中选出准确率最高的一项；全部失败时返回 (None, 原基线值占位)。

    注意：failed 实验的 overall_acc=None，会被自动排除在选优之外。
    """
    scored = [c for c in candidates if isinstance(c.get("overall_acc"), (int, float))]
    if not scored:
        return None, None
    scored.sort(key=lambda x: (x["overall_acc"], -len(str(x["value"]))), reverse=True)
    return scored[0], scored[0]["value"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LID 超参贪心式扫描主控入口")
    parser.add_argument("--config", required=True,
                        help="YAML 扫描配置文件路径")
    parser.add_argument("--sweep_dir", default=None,
                        help="顶层扫描目录路径。--resume 时必需指向已存在的扫描目录；未指定则新建 exp/<prefix>_<ts>/")
    parser.add_argument("--resume", action="store_true",
                        help="从已存在的扫描目录恢复：跳过已产出 test_accuracy.json (status=success) 的实验")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_sweep_plan(args.config)

    sweep_dir = _resolve_sweep_dir(args.sweep_dir, plan)
    if args.resume:
        if not os.path.isdir(sweep_dir):
            logger.error("--resume 指定的目录不存在：%s", sweep_dir)
            return 2
    else:
        if os.path.isdir(sweep_dir) and any(os.scandir(sweep_dir)):
            logger.warning("扫描目录已存在且非空：%s；如需继续请加 --resume", sweep_dir)
            return 2
        os.makedirs(sweep_dir, exist_ok=True)

    logger.info("sweep_dir = %s", sweep_dir)
    logger.info("resume    = %s", args.resume)
    logger.info("project_root = %s", PROJECT_ROOT)

    # 归档配置副本（可复现）
    _atomic_write_json(os.path.join(sweep_dir, "config_snapshot.json"), {
        "config_path": plan.raw_config_path,
        "fixed_params": plan.fixed_params,
        "baseline_initial": plan.baseline,
        "sweep_order": [d.name for d in plan.dimensions],
        "sweep_space": {d.name: d.candidates for d in plan.dimensions},
        "run_options": {
            "base_output_dir": plan.run_options.base_output_dir,
            "sweep_dir_prefix": plan.run_options.sweep_dir_prefix,
            "keep_only_best_ckpt": plan.run_options.keep_only_best_ckpt,
            "min_acc_warning_threshold": plan.run_options.min_acc_warning_threshold,
        },
    })

    tracker = ProgressTracker(sweep_dir)
    if args.resume:
        tracker.load_if_exists()
    # 起始基线：优先使用 progress 中的 current_baseline（--resume 场景），否则用 plan.baseline
    baseline: Dict[str, Any] = copy.deepcopy(tracker.state.get("current_baseline") or plan.baseline)
    tracker.set_baseline(baseline)
    tracker.flush()

    # 优雅中断：接到 SIGINT/SIGTERM 时刷进度后再退出
    interrupted = {"flag": False}

    def _on_signal(signum, _frame):
        logger.warning("received signal %d, flushing progress and exiting after current subprocess ...", signum)
        interrupted["flag"] = True
        # 触发默认信号处理让当前 subprocess 收到信号（由 runner 内部转发）

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    dim_results: List[Dict[str, Any]] = []

    for dim in plan.dimensions:
        logger.info("#" * 78)
        logger.info("# 维度 %02d: %s | 候选数=%d | 当前基线快照=%s",
                    dim.index, dim.name, len(dim.candidates), baseline)
        logger.info("#" * 78)

        candidates_results: List[Dict[str, Any]] = []

        if dim.skipped:
            # 长度为 1：直接固化，不训练
            only_value = dim.candidates[0]
            logger.info("[dim %s] skipped=True，直接固化 %s = %s", dim.name, dim.name, only_value)
            baseline[dim.name] = only_value
            tracker.set_baseline(baseline)
            dim_entry = {
                "dim_index": dim.index,
                "dim_name": dim.name,
                "skipped": True,
                "candidates": [],
                "best_value": only_value,
                "best_acc": None,
                "baseline_after": copy.deepcopy(baseline),
            }
            dim_results.append(dim_entry)
            tracker.record_dim_best(dim.name, {"value": only_value, "accuracy": None, "exp_dir": None, "skipped": True})
            tracker.flush()
            continue

        for value in dim.candidates:
            if interrupted["flag"]:
                logger.warning("interrupted before running dim=%s value=%s, aborting sweep", dim.name, value)
                break
            baseline_snapshot = copy.deepcopy(baseline)

            # === 复用判定 ===
            # 触发条件：
            #   1) 候选值 == 当前 baseline 中该维度的取值（意味着这组超参 = 当前 baseline）
            #   2) 前面至少存在一个已完成维度（提供 winner 实验作为复用来源）
            #   3) 复用来源的 hparams 与本次将要跑的完整 hparams 严格一致
            reuse_from: Optional[Dict[str, Any]] = None
            hparams_this = copy.deepcopy(baseline_snapshot)
            hparams_this[dim.name] = value
            if value == baseline_snapshot.get(dim.name):
                # 从已跑过的维度里找最近一个 winner，其 hparams 快照可能等于当前 hparams_this
                # 优先取上一维度的 winner（贪心链路上的直接前驱最可能匹配），
                # 若不匹配再回退检查更早维度的 winner
                for prev_entry in reversed(dim_results):
                    prev_best = tracker.state["per_dim_best"].get(prev_entry["dim_name"])
                    if not prev_best or prev_best.get("skipped"):
                        continue
                    prev_hparams = prev_best.get("hparams") or {}
                    prev_exp_dir = prev_best.get("exp_dir")
                    prev_acc = prev_best.get("accuracy")
                    if (
                        prev_exp_dir
                        and isinstance(prev_acc, (int, float))
                        and _hparams_equal(prev_hparams, hparams_this)
                    ):
                        # 若 --resume 但源实验目录已被清理（keep_only_best_ckpt 只影响 ckpt，
                        # test_accuracy.json 仍在），复用仍然安全
                        reuse_from = {
                            "exp_dir": prev_exp_dir,
                            "overall_acc": prev_acc,
                            "num_samples": prev_best.get("num_samples", 0),
                            "source_dim": prev_entry["dim_name"],
                        }
                        logger.info(
                            "[reuse-detect] dim=%s value=%s 命中复用：来源=维度 `%s` winner(exp_dir=%s, acc=%.4f)",
                            dim.name, value, prev_entry["dim_name"], prev_exp_dir, float(prev_acc),
                        )
                        break

            result = _run_one_experiment(
                sweep_dir=sweep_dir,
                dim=dim,
                value=value,
                baseline_snapshot=baseline_snapshot,
                plan=plan,
                tracker=tracker,
                resume=args.resume,
                reuse_from=reuse_from,
            )
            candidates_results.append(result)
            tracker.record_experiment({
                "dim_index": dim.index,
                "dim_name": dim.name,
                "value": value,
                "exp_dir": result["exp_dir"],
                "overall_acc": result["overall_acc"],
                "status": result["status"],
                "resumed": result.get("resumed", False),
                "reused_from_rel": result.get("reused_from_rel", ""),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            tracker.flush()

        # 选优
        best, best_value = _select_best_candidate(candidates_results)
        if best is None:
            # 全部失败：baseline 保持不变
            best_value = baseline[dim.name]
            best_acc: Optional[float] = None
            logger.error("[dim %s] 所有候选实验均失败，baseline 保持不变：%s=%s", dim.name, dim.name, best_value)
        else:
            best_acc = best["overall_acc"]
            baseline[dim.name] = best_value
            logger.info("[dim %s] 选中 %s = %s (overall_acc=%.4f)", dim.name, dim.name, best_value, best_acc)

        # 标记 best
        for c in candidates_results:
            c["is_best"] = (c["value"] == best_value) and (best is not None)

        tracker.set_baseline(baseline)
        # per_dim_best 记录 winner 的完整 hparams 快照，供后续维度做复用判定
        winner_hparams = copy.deepcopy(best["hparams"]) if best else None
        tracker.record_dim_best(dim.name, {
            "value": best_value,
            "accuracy": best_acc,
            "exp_dir": best["exp_dir"] if best else None,
            "num_samples": best.get("num_samples", 0) if best else 0,
            "hparams": winner_hparams,
            "skipped": False,
        })
        tracker.flush()

        dim_entry = {
            "dim_index": dim.index,
            "dim_name": dim.name,
            "skipped": False,
            "candidates": candidates_results,
            "best_value": best_value,
            "best_acc": best_acc,
            "baseline_after": copy.deepcopy(baseline),
        }
        dim_results.append(dim_entry)

        if plan.run_options.keep_only_best_ckpt:
            _cleanup_non_best_ckpts(dim_entry)

        if interrupted["flag"]:
            logger.warning("interrupted after dim %s finished, stopping sweep", dim.name)
            break

    # 写最终产物
    _atomic_write_json(os.path.join(sweep_dir, "best_hparams.json"), {
        "final_baseline": baseline,
        "per_dim_best": tracker.state["per_dim_best"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    summary_path = _write_sweep_summary(sweep_dir, plan, dim_results)
    logger.info("summary written: %s", summary_path)
    logger.info("final baseline: %s", baseline)
    logger.info("done. sweep_dir=%s", sweep_dir)
    return 0 if not interrupted["flag"] else 130


if __name__ == "__main__":
    sys.exit(main())
