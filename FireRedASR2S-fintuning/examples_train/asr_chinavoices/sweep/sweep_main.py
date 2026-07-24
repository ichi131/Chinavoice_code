#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""ASR 超参贪心式扫描主控入口。

流程：
1. 加载 YAML 扫描配置 + 全部预检（config_loader）。
2. 决定顶层扫描目录（新扫描 = <base_output_dir>/<prefix>_<ts>/；
   --resume / SWEEP_DIR 指向已有目录 = 恢复模式）。
3. 在 sweep 根目录一次性生成推理输入 `test_input_converted.jsonl`（字段转换）。
4. 按 sweep_order 逐维度扫描：
   - 对每个候选值创建实验子目录 `<idx>_<name>__<value>/`
   - 组装完整 hparams → 触发复用 判定或执行"训练 → 推理 → 格式化+CER 评估"
   - 写 status.json / test_metrics.json / sweep_progress.json
   - 维度结束：选 overall_cer 最小者为 winner，更新 baseline
   - keep_only_best_ckpt=true 时删除同维度非最优实验的 model.pth.tar + last.pt
5. 全部完成后生成 sweep_summary.md 与 best_hparams.json。

强隔离约束：所有 IO 都写入扫描顶层目录内部，绝不修改现有训练/推理脚本。
选优规则：**overall_cer 最小**（与 LID 的"最大化 acc"方向相反）。
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
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
    ConfigError,
    DimensionPlan,
    SweepConfig,
    float_close,
    load_sweep_config,
)
from runner import (  # noqa: E402
    EvalFailed,
    InferFailed,
    TrainFailed,
    convert_test_jsonl_if_needed,
    run_format_and_score,
    run_inference,
    run_training,
)

# 项目根目录 = 当前脚本向上 3 层：sweep/ -> asr_chinavoices/ -> examples_train/ -> <root>
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])


# =============================================================================
# 日志配置：控制台 + sweep_console.log 双写
# =============================================================================

class _StreamToLogger:
    """给 print/logger 提供 sweep_console.log 副本（追加）。"""

    def __init__(self, stream, log_path):
        self.stream = stream
        self.log_path = log_path

    def write(self, chunk):
        self.stream.write(chunk)
        self.stream.flush()
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(chunk)
        except OSError:
            pass

    def flush(self):
        self.stream.flush()


def _setup_logging(sweep_dir: str) -> None:
    console_log = os.path.join(sweep_dir, "sweep_console.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    root = logging.getLogger()
    fh = logging.FileHandler(console_log, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    fh.setLevel(logging.INFO)
    # 避免重复添加
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == fh.baseFilename:
            return
    root.addHandler(fh)


logger = logging.getLogger("sweep.main")


# =============================================================================
# 通用工具
# =============================================================================

def _format_value_for_dir(value: Any) -> str:
    """把候选值格式化为文件名安全字符串。"""
    if isinstance(value, float):
        s = repr(value)
    else:
        s = str(value)
    return s.replace("+", "").replace("/", "_").replace(" ", "")


def _exp_subdir_name(dim_index: int, dim_name: str, value: Any) -> str:
    return f"{dim_index:02d}_{dim_name}__{_format_value_for_dir(value)}"


def _atomic_write_json(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("failed to read %s: %s", path, e)
        return None


def _hparams_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """严格比较两组超参是否完全一致（浮点用 float_close）。"""
    if set(a.keys()) != set(b.keys()):
        return False
    for k, va in a.items():
        if not float_close(va, b[k]):
            return False
    return True


def _resolve_sweep_dir(cli_sweep_dir: Optional[str], cfg: SweepConfig) -> str:
    """决定顶层扫描目录路径（绝对路径）。

    - 命令行 --sweep_dir 或环境变量 SWEEP_DIR 优先
    - 否则用 <base_output_dir>/<prefix>_<ts>
    """
    if cli_sweep_dir:
        return os.path.abspath(cli_sweep_dir)
    env_dir = os.environ.get("SWEEP_DIR", "").strip()
    if env_dir:
        return os.path.abspath(env_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = cfg.run_options.base_output_dir
    if not os.path.isabs(base):
        base = os.path.join(PROJECT_ROOT, base)
    return os.path.abspath(os.path.join(base, f"{cfg.run_options.sweep_dir_prefix}_{ts}"))


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
            "per_dim_best": {},
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

    def record_experiment(self, entry: Dict[str, Any]) -> None:
        # 用 exp_dir 去重
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
# 单次实验执行
# =============================================================================

def _run_one_experiment(
    sweep_dir: str,
    dim: DimensionPlan,
    value: Any,
    baseline_snapshot: Dict[str, Any],
    cfg: SweepConfig,
    test_input_jsonl: str,
    tracker: ProgressTracker,
    resume: bool,
    reuse_from: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行一次实验（训练 → 推理 → 评估）或走复用/续跑分支，返回结果字典。

    Returns:
        result: 至少包含 keys: value, exp_dir, exp_dir_rel, hparams,
                overall_cer(Optional[float]), num_samples, status, message,
                elapsed_s, resumed(bool), reused_from_rel(str)
    """
    exp_name = _exp_subdir_name(dim.index, dim.name, value)
    exp_dir = os.path.join(sweep_dir, exp_name)

    # 合成本次实际生效的超参（float_close 判等的对象）
    hparams = copy.deepcopy(baseline_snapshot)
    hparams[dim.name] = value

    # === 断点续跑分支：若已存在 test_metrics.json 且 status.json 为 success/reused ===
    if resume:
        prev_status = _read_json(os.path.join(exp_dir, "status.json"))
        prev_metrics = _read_json(os.path.join(exp_dir, "test_metrics.json"))
        if prev_status and prev_metrics and prev_status.get("status") in ("success", "reused"):
            logger.info(
                "[resume] skip %s (status=%s overall_cer=%s%%)",
                exp_name, prev_status.get("status"), prev_metrics.get("overall_cer"),
            )
            return {
                "value": value,
                "exp_dir": exp_dir,
                "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
                "hparams": hparams,
                "overall_cer": prev_metrics.get("overall_cer"),
                "per_dialect_cer": prev_metrics.get("per_dialect_cer", {}),
                "num_samples": prev_metrics.get("num_samples", 0),
                "status": prev_status.get("status"),
                "message": prev_status.get("message", ""),
                "elapsed_s": prev_status.get("elapsed_s", 0.0),
                "resumed": True,
                "reused_from_rel": prev_status.get("reused_from_rel", ""),
                "val_macro_cer": prev_status.get("val_macro_cer"),
                "failed_stage": None,
            }

    os.makedirs(exp_dir, exist_ok=True)

    # === 复用分支：候选值 == 当前 baseline 中该维度的取值，且存在等效历史实验 ===
    if reuse_from is not None:
        source_exp_dir = reuse_from["exp_dir"]
        source_metrics = reuse_from["test_metrics"]
        source_dim_name = reuse_from.get("source_dim", "")
        source_rel = os.path.relpath(source_exp_dir, sweep_dir) if source_exp_dir else ""

        logger.info("=" * 78)
        logger.info("[reuse] dim=%s value=%s -> %s | 来源: %s (dim=%s, overall_cer=%s%%)",
                    dim.name, value, exp_name, source_rel,
                    source_dim_name, source_metrics.get("overall_cer"))
        logger.info("=" * 78)

        # 写实验配置快照
        _atomic_write_json(os.path.join(exp_dir, "train_config.json"), {
            "dimension": dim.name,
            "dimension_index": dim.index,
            "candidate_value": value,
            "hparams": hparams,
            "reused": True,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # 写 reused_from.json
        _atomic_write_json(os.path.join(exp_dir, "reused_from.json"), {
            "source_exp_dir": source_exp_dir,
            "source_exp_dir_rel": source_rel,
            "source_dim": source_dim_name,
            "overall_cer": source_metrics.get("overall_cer"),
            "hparams": hparams,
            "note": "候选值等于当前基线且已有等效实验，直接复用其 CER 结果，不重训不重推理不重评估。",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # 复制 test_metrics.json 供后续汇总使用（同结构）
        reused_metrics = dict(source_metrics)
        reused_metrics["reused_from"] = source_rel
        _atomic_write_json(os.path.join(exp_dir, "test_metrics.json"), reused_metrics)

        # 写 status.json
        _atomic_write_json(os.path.join(exp_dir, "status.json"), {
            "status": "reused",
            "elapsed_s": 0.0,
            "failed_stage": None,
            "error": None,
            "message": f"reused from {source_rel}",
            "reused_from_rel": source_rel,
            "val_macro_cer": reuse_from.get("val_macro_cer"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        return {
            "value": value,
            "exp_dir": exp_dir,
            "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
            "hparams": hparams,
            "overall_cer": source_metrics.get("overall_cer"),
            "per_dialect_cer": source_metrics.get("per_dialect_cer", {}),
            "num_samples": source_metrics.get("num_samples", 0),
            "status": "reused",
            "message": f"reused from {source_rel}",
            "elapsed_s": 0.0,
            "resumed": False,
            "reused_from_rel": source_rel,
            "val_macro_cer": reuse_from.get("val_macro_cer"),
            "failed_stage": None,
        }

    # === 常规分支：训练 → 推理 → 格式化+CER 评估 ===
    _atomic_write_json(os.path.join(exp_dir, "train_config.json"), {
        "dimension": dim.name,
        "dimension_index": dim.index,
        "candidate_value": value,
        "hparams": hparams,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    logger.info("=" * 78)
    logger.info("[experiment] dim=%s value=%s -> %s", dim.name, value, exp_name)
    logger.info("=" * 78)

    stage_start = time.time()
    failed_stage: Optional[str] = None
    error_msg: Optional[str] = None
    val_macro_cer: Optional[float] = None

    try:
        # --- 训练 ---
        train_result = run_training(
            exp_dir=exp_dir,
            hparams=hparams,
            cfg=cfg,
            project_root=PROJECT_ROOT,
        )
        # 提取 val macro_cer 作为参考指标
        best_metrics = train_result.get("best_metrics") or {}
        val_block = best_metrics.get("validation") or {}
        raw_macro = val_block.get("macro_cer")
        if isinstance(raw_macro, (int, float)) and not math.isnan(float(raw_macro)):
            # 训练脚本内的 macro_cer 是 [0,1] 小数；统一转为百分数
            val_macro_cer = float(raw_macro) * 100.0

        # --- 推理 ---
        expected_count = _count_input_lines(test_input_jsonl)
        infer_result = run_inference(
            exp_dir=exp_dir,
            test_input_jsonl=test_input_jsonl,
            cfg=cfg,
            project_root=PROJECT_ROOT,
        )

        # --- 格式化 + CER 评估 ---
        eval_result = run_format_and_score(
            exp_dir=exp_dir,
            cfg=cfg,
            expected_sample_count=expected_count,
        )

        elapsed_s = time.time() - stage_start
        _atomic_write_json(os.path.join(exp_dir, "status.json"), {
            "status": "success",
            "elapsed_s": elapsed_s,
            "failed_stage": None,
            "error": None,
            "message": "",
            "val_macro_cer": val_macro_cer,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        return {
            "value": value,
            "exp_dir": exp_dir,
            "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
            "hparams": hparams,
            "overall_cer": eval_result["overall_cer"],
            "per_dialect_cer": eval_result.get("per_dialect_cer", {}),
            "num_samples": infer_result.get("num_samples", 0),
            "status": "success",
            "message": "",
            "elapsed_s": elapsed_s,
            "resumed": False,
            "reused_from_rel": "",
            "val_macro_cer": val_macro_cer,
            "failed_stage": None,
        }

    except TrainFailed as e:
        failed_stage = "train"
        error_msg = str(e)
    except InferFailed as e:
        failed_stage = "infer"
        error_msg = str(e)
    except EvalFailed as e:
        failed_stage = "eval"
        error_msg = str(e)
    except Exception as e:  # pragma: no cover - 兜底
        failed_stage = "unknown"
        error_msg = f"{type(e).__name__}: {e}"

    # === 失败分支：写 status.json 并返回 ===
    elapsed_s = time.time() - stage_start
    logger.error("[experiment] failed at stage=%s: %s", failed_stage, error_msg)
    _atomic_write_json(os.path.join(exp_dir, "status.json"), {
        "status": "failed",
        "elapsed_s": elapsed_s,
        "failed_stage": failed_stage,
        "error": error_msg,
        "message": error_msg,
        "val_macro_cer": val_macro_cer,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return {
        "value": value,
        "exp_dir": exp_dir,
        "exp_dir_rel": os.path.relpath(exp_dir, sweep_dir),
        "hparams": hparams,
        "overall_cer": None,
        "per_dialect_cer": {},
        "num_samples": 0,
        "status": "failed",
        "message": error_msg or "",
        "elapsed_s": elapsed_s,
        "resumed": False,
        "reused_from_rel": "",
        "val_macro_cer": val_macro_cer,
        "failed_stage": failed_stage,
    }


def _count_input_lines(path: str) -> int:
    """轻量 JSONL 行数统计。"""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# =============================================================================
# 磁盘清理
# =============================================================================

def _cleanup_non_best_ckpts(dim_results_entry: Dict[str, Any]) -> None:
    """在维度扫描完成后，删除非最优实验的 model.pth.tar / last.pt，节省磁盘。

    保留：test_metrics.json / best_metrics.json / train_config.json / pred_test.jsonl
          / pred_test_formatted.jsonl / wer_eval/ / status.json / *.log
    """
    for cand in dim_results_entry["candidates"]:
        if cand.get("is_best"):
            continue
        if cand.get("status") == "reused":
            continue
        exp_dir = cand.get("exp_dir")
        if not exp_dir or not os.path.isdir(exp_dir):
            continue
        for ckpt_name in ("model.pth.tar", "last.pt"):
            p = os.path.join(exp_dir, ckpt_name)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    logger.info("cleaned checkpoint: %s", p)
                except OSError as e:
                    logger.warning("cleanup failed for %s: %s", p, e)


# =============================================================================
# 收尾产物（对应任务 8）
# =============================================================================

def _write_sweep_summary(sweep_dir: str, cfg: SweepConfig, dim_results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# ASR 超参扫描汇总报告")
    lines.append("")
    lines.append(f"- 扫描目录：`{sweep_dir}`")
    lines.append(f"- 配置文件：`{cfg.raw_config_path}`")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 维度总数：{len(cfg.dimensions)}")
    lines.append(f"- 选优指标：**外部 test overall CER（越小越好）**")
    lines.append("")

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
        lines.append(
            f"### 维度 {dim_res['dim_index']:02d} · `{dim_name}`"
            + ("（单候选，跳过扫描）" if skipped else "")
        )
        lines.append("")
        lines.append(
            "| 候选值 | test overall CER | val macro CER | 耗时 | status | 复用来源 | exp_dir |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for cand in dim_res["candidates"]:
            cer = cand.get("overall_cer")
            cer_str = f"{cer:.2f}%" if isinstance(cer, (int, float)) else "N/A"
            val_macro = cand.get("val_macro_cer")
            val_str = f"{val_macro:.2f}%" if isinstance(val_macro, (int, float)) else "N/A"
            elapsed = cand.get("elapsed_s", 0.0)
            elapsed_str = _fmt_elapsed(elapsed)
            marker = " ⭐" if cand.get("is_best") else ""
            reused_src = cand.get("reused_from_rel") or ""
            reused_src_str = f"`{reused_src}`" if reused_src else ""
            lines.append(
                f"| `{cand['value']}`{marker} | {cer_str} | {val_str} | {elapsed_str} | "
                f"{cand.get('status', 'unknown')} | {reused_src_str} | `{cand.get('exp_dir_rel', '')}` |"
            )
        best_val = dim_res.get("best_value")
        best_cer = dim_res.get("best_cer")
        if best_cer is not None:
            lines.append("")
            lines.append(
                f"**选中：`{dim_name} = {best_val}` (test overall CER = {best_cer:.2f}%)**"
            )
        else:
            lines.append("")
            lines.append(f"**选中：`{dim_name} = {best_val}` (所有候选均失败，保持 baseline)**")
        lines.append("")

    # 贪心路径
    lines.append("## 贪心路径")
    lines.append("")
    lines.append("```")
    prev_baseline = None
    for i, dim_res in enumerate(dim_results):
        if i == 0:
            prev_baseline = cfg.baseline_hparams
            lines.append(f"initial baseline: {json.dumps({k: prev_baseline[k] for k in cfg.baseline_hparams}, ensure_ascii=False)}")
        after = dim_res.get("baseline_after") or {}
        lines.append(f"after dim `{dim_res['dim_name']}`: {dim_res['dim_name']} = {after.get(dim_res['dim_name'])}"
                     f"   (test overall CER = {dim_res.get('best_cer')})")
    lines.append("```")
    lines.append("")

    # 最终最佳
    lines.append("## 最终最佳超参组合")
    lines.append("")
    final_baseline = dim_results[-1]["baseline_after"] if dim_results else cfg.baseline_hparams
    lines.append("```json")
    lines.append(json.dumps(final_baseline, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    # 最佳实验目录（overall_cer 最小的实验）
    best_exp_dir_rel = None
    best_cer_all = float("inf")
    for dim_res in dim_results:
        for cand in dim_res["candidates"]:
            cer = cand.get("overall_cer")
            if isinstance(cer, (int, float)) and cer < best_cer_all:
                best_cer_all = cer
                best_exp_dir_rel = cand.get("exp_dir_rel")
    if best_exp_dir_rel is not None:
        lines.append(f"**最佳实验目录**（overall CER = {best_cer_all:.2f}%）：`{best_exp_dir_rel}`")
    else:
        lines.append("**未找到任何成功实验，最佳超参回退到 initial baseline**")

    summary_path = os.path.join(sweep_dir, "sweep_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary_path


def _fmt_elapsed(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# =============================================================================
# 候选选优
# =============================================================================

def _select_best_candidate(candidates: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Any]:
    """从候选结果中选出 overall_cer **最小**的一项；全部失败时返回 (None, None)。

    注意：failed 实验的 overall_cer=None，会被自动排除在选优之外。
    """
    scored = [c for c in candidates if isinstance(c.get("overall_cer"), (int, float))]
    if not scored:
        return None, None
    # 越小越好；tie-break：完整程度（reused/success 靠前）不影响，直接取 CER 最小
    scored.sort(key=lambda x: (x["overall_cer"], str(x["value"])))
    return scored[0], scored[0]["value"]


# =============================================================================
# 主入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR 超参贪心式扫描主控入口")
    parser.add_argument("--config", required=True,
                        help="YAML 扫描配置文件路径")
    parser.add_argument("--sweep_dir", default=None,
                        help="顶层扫描目录。--resume 时必需指向已存在的扫描目录；"
                             "未指定则新建 <base_output_dir>/<prefix>_<ts>/")
    parser.add_argument("--resume", action="store_true",
                        help="从已存在的扫描目录恢复：跳过 status=success/reused 的实验")
    return parser.parse_args()


def _print_console_line(sweep_dir: str, line: str) -> None:
    """既打印到 stdout 也追加到 sweep_console.log（logging 已覆盖大部分，这个是简报专用）。"""
    console_log = os.path.join(sweep_dir, "sweep_console.log")
    print(line, flush=True)
    try:
        with open(console_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def main() -> int:
    args = parse_args()

    # 支持环境变量 RESUME=1（README 用法一致）
    if os.environ.get("RESUME", "").strip() == "1":
        args.resume = True

    try:
        cfg = load_sweep_config(args.config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"[config-error] {e}", file=sys.stderr)
        return 2

    sweep_dir = _resolve_sweep_dir(args.sweep_dir, cfg)
    if args.resume:
        if not os.path.isdir(sweep_dir):
            print(f"[error] --resume 指定的目录不存在：{sweep_dir}", file=sys.stderr)
            return 2
    else:
        if os.path.isdir(sweep_dir) and any(os.scandir(sweep_dir)):
            print(
                f"[error] 扫描目录已存在且非空：{sweep_dir}；如需继续请加 --resume 或设置 RESUME=1",
                file=sys.stderr,
            )
            return 2
        os.makedirs(sweep_dir, exist_ok=True)

    _setup_logging(sweep_dir)
    logger.info("sweep_dir = %s", sweep_dir)
    logger.info("resume    = %s", args.resume)
    logger.info("project_root = %s", PROJECT_ROOT)
    logger.info("config    = %s", cfg.raw_config_path)

    # 归档配置快照
    _atomic_write_json(os.path.join(sweep_dir, "config_snapshot.json"), {
        "config_path": cfg.raw_config_path,
        "data_paths": cfg.data_paths,
        "pretrained_model_dir": cfg.pretrained_model_dir,
        "format_script": cfg.format_script,
        "eval_tool_sh": cfg.eval_tool_sh,
        "baseline_hparams": cfg.baseline_hparams,
        "sweep_order": [d.name for d in cfg.dimensions],
        "sweep_space": {d.name: d.candidates for d in cfg.dimensions},
        "infer_args": cfg.infer_args,
        "run_options": {
            "base_output_dir": cfg.run_options.base_output_dir,
            "sweep_dir_prefix": cfg.run_options.sweep_dir_prefix,
            "keep_only_best_ckpt": cfg.run_options.keep_only_best_ckpt,
            "nproc_per_node": cfg.run_options.nproc_per_node,
        },
    })

    # 生成推理输入（字段转换：test_jsonl → test_input_converted.jsonl）
    test_input_jsonl = os.path.join(sweep_dir, "test_input_converted.jsonl")
    try:
        sample_count = convert_test_jsonl_if_needed(cfg.data_paths["test_jsonl"], test_input_jsonl)
    except Exception as e:
        logger.error("test_jsonl 转换失败：%s", e)
        return 3
    logger.info("test 集样本数：%d", sample_count)

    tracker = ProgressTracker(sweep_dir)
    if args.resume:
        tracker.load_if_exists()
    baseline: Dict[str, Any] = copy.deepcopy(
        tracker.state.get("current_baseline") or cfg.baseline_hparams
    )
    tracker.set_baseline(baseline)
    tracker.flush()

    # 优雅中断
    interrupted = {"flag": False}

    def _on_signal(signum, _frame):
        logger.warning("received signal %d, will exit after current subprocess", signum)
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    dim_results: List[Dict[str, Any]] = []

    # 计算总实验编号（用于简报）
    total_experiments = sum(len(d.candidates) for d in cfg.dimensions)
    running_index = 0

    for dim in cfg.dimensions:
        logger.info("#" * 78)
        logger.info("# 维度 %02d: %s | 候选数=%d | 当前基线=%s",
                    dim.index, dim.name, len(dim.candidates), baseline)
        logger.info("#" * 78)

        # 若续跑时该维度已完成，直接读 winner
        prev_dim_best = tracker.state["per_dim_best"].get(dim.name)
        if args.resume and prev_dim_best and prev_dim_best.get("skipped") is False \
                and prev_dim_best.get("winner_locked"):
            # 检查候选状态：只要有任一候选处于 failed 状态，就重跑整维（清除锁）
            prev_snapshots = prev_dim_best.get("candidates_snapshot", []) or []
            has_failed = any(
                (c or {}).get("status") == "failed" for c in prev_snapshots
            )
            if has_failed:
                logger.warning(
                    "[resume] dim=%s winner_locked=True 但存在 failed 候选；清除锁并重跑本维度",
                    dim.name,
                )
                # 保留 hparams 但清掉 winner_locked，走常规候选循环
                # 常规循环会读每个候选的 status.json + test_metrics.json：
                #   - success/reused → 跳过
                #   - failed / 不存在 → 重跑对应阶段
            else:
                baseline[dim.name] = prev_dim_best["value"]
                tracker.set_baseline(baseline)
                logger.info("[resume] dim=%s 已完成 winner=%s；直接进入下一维度",
                            dim.name, prev_dim_best["value"])
                dim_results.append({
                    "dim_index": dim.index,
                    "dim_name": dim.name,
                    "skipped": False,
                    "candidates": prev_dim_best.get("candidates_snapshot", []),
                    "best_value": prev_dim_best["value"],
                    "best_cer": prev_dim_best.get("overall_cer"),
                    "baseline_after": copy.deepcopy(baseline),
                })
                # 累计实验编号（近似）
                running_index += len(dim.candidates)
                continue

        candidates_results: List[Dict[str, Any]] = []

        if dim.skipped:
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
                "best_cer": None,
                "baseline_after": copy.deepcopy(baseline),
            }
            dim_results.append(dim_entry)
            tracker.record_dim_best(dim.name, {
                "value": only_value,
                "overall_cer": None,
                "exp_dir": None,
                "skipped": True,
                "winner_locked": True,
                "candidates_snapshot": [],
            })
            tracker.flush()
            continue

        for value in dim.candidates:
            if interrupted["flag"]:
                logger.warning("interrupted before running dim=%s value=%s", dim.name, value)
                break

            running_index += 1
            baseline_snapshot = copy.deepcopy(baseline)

            # 复用判定：候选值 == 当前 baseline 中该维度取值 且 前序有等效 winner
            reuse_from: Optional[Dict[str, Any]] = None
            hparams_this = copy.deepcopy(baseline_snapshot)
            hparams_this[dim.name] = value
            if float_close(value, baseline_snapshot.get(dim.name)):
                for prev_entry in reversed(dim_results):
                    prev_best = tracker.state["per_dim_best"].get(prev_entry["dim_name"])
                    if not prev_best or prev_best.get("skipped"):
                        continue
                    prev_hparams = prev_best.get("hparams") or {}
                    prev_exp_dir = prev_best.get("exp_dir")
                    prev_test_metrics = prev_best.get("test_metrics") or {}
                    prev_cer = prev_test_metrics.get("overall_cer")
                    if (
                        prev_exp_dir
                        and isinstance(prev_cer, (int, float))
                        and _hparams_equal(prev_hparams, hparams_this)
                    ):
                        reuse_from = {
                            "exp_dir": prev_exp_dir,
                            "test_metrics": prev_test_metrics,
                            "source_dim": prev_entry["dim_name"],
                            "val_macro_cer": prev_best.get("val_macro_cer"),
                        }
                        logger.info(
                            "[reuse-detect] dim=%s value=%s 命中复用：来源=`%s`(overall_cer=%.2f%%)",
                            dim.name, value, prev_entry["dim_name"], float(prev_cer),
                        )
                        break

            result = _run_one_experiment(
                sweep_dir=sweep_dir,
                dim=dim,
                value=value,
                baseline_snapshot=baseline_snapshot,
                cfg=cfg,
                test_input_jsonl=test_input_jsonl,
                tracker=tracker,
                resume=args.resume,
                reuse_from=reuse_from,
            )
            candidates_results.append(result)

            # 一行简报
            cer_val = result.get("overall_cer")
            val_macro = result.get("val_macro_cer")
            cer_str = f"{cer_val:.2f}%" if isinstance(cer_val, (int, float)) else "N/A"
            val_str = f"{val_macro:.2f}%" if isinstance(val_macro, (int, float)) else "N/A"
            _print_console_line(sweep_dir, (
                f"[exp_{running_index}/{total_experiments}] dim={dim.name} value={value} "
                f"status={result['status']} test_overall_cer={cer_str} "
                f"val_macro_cer={val_str} elapsed={_fmt_elapsed(result.get('elapsed_s', 0))}"
            ))

            tracker.record_experiment({
                "dim_index": dim.index,
                "dim_name": dim.name,
                "value": value,
                "exp_dir": result["exp_dir"],
                "overall_cer": result["overall_cer"],
                "val_macro_cer": result.get("val_macro_cer"),
                "status": result["status"],
                "elapsed_s": result.get("elapsed_s", 0),
                "resumed": result.get("resumed", False),
                "reused_from_rel": result.get("reused_from_rel", ""),
                "failed_stage": result.get("failed_stage"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            tracker.flush()

        # 选优：overall_cer 最小
        best, best_value = _select_best_candidate(candidates_results)
        if best is None:
            best_value = baseline[dim.name]
            best_cer: Optional[float] = None
            logger.error("[dim %s] 所有候选实验均失败，baseline 保持不变：%s=%s",
                         dim.name, dim.name, best_value)
        else:
            best_cer = best["overall_cer"]
            baseline[dim.name] = best_value
            logger.info("[dim %s] 选中 %s = %s (overall_cer=%.2f%%)",
                        dim.name, dim.name, best_value, best_cer)

        for c in candidates_results:
            c["is_best"] = float_close(c["value"], best_value) and (best is not None)

        tracker.set_baseline(baseline)
        winner_hparams = copy.deepcopy(best["hparams"]) if best else None
        winner_test_metrics = None
        if best is not None:
            # 从实验目录读取 test_metrics.json（reused 分支也已经复制过一份）
            winner_test_metrics = _read_json(os.path.join(best["exp_dir"], "test_metrics.json"))
        tracker.record_dim_best(dim.name, {
            "value": best_value,
            "overall_cer": best_cer,
            "exp_dir": best["exp_dir"] if best else None,
            "hparams": winner_hparams,
            "test_metrics": winner_test_metrics,
            "val_macro_cer": best.get("val_macro_cer") if best else None,
            "skipped": False,
            "winner_locked": True,
            "candidates_snapshot": candidates_results,
        })
        tracker.flush()

        dim_entry = {
            "dim_index": dim.index,
            "dim_name": dim.name,
            "skipped": False,
            "candidates": candidates_results,
            "best_value": best_value,
            "best_cer": best_cer,
            "baseline_after": copy.deepcopy(baseline),
        }
        dim_results.append(dim_entry)

        if cfg.run_options.keep_only_best_ckpt:
            _cleanup_non_best_ckpts(dim_entry)

        if interrupted["flag"]:
            logger.warning("interrupted after dim %s finished", dim.name)
            break

    # ===================== 收尾产物（任务 8） =====================
    _atomic_write_json(os.path.join(sweep_dir, "best_hparams.json"), {
        "final_baseline": baseline,
        "per_dim_best": {
            k: {kk: vv for kk, vv in v.items() if kk != "candidates_snapshot"}
            for k, v in tracker.state["per_dim_best"].items()
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    summary_path = _write_sweep_summary(sweep_dir, cfg, dim_results)
    logger.info("summary written: %s", summary_path)
    logger.info("final baseline: %s", baseline)
    logger.info("done. sweep_dir=%s", sweep_dir)
    return 0 if not interrupted["flag"] else 130


if __name__ == "__main__":
    sys.exit(main())
