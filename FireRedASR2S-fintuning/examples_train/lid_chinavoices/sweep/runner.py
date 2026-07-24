# -*- coding: utf-8 -*-
"""LID 超参扫描 · 单次实验运行器。

包含两个核心函数：

1. `run_training(exp_dir, hparams, fixed_params, project_root, log_path)`
   - 通过 subprocess 调用 finetune_lid_chinavoices.py
   - 自动根据 CUDA_VISIBLE_DEVICES 选择 python3.10 / torchrun
   - stdout/stderr 实时透传 + 同时写入 train.log
   - 返回训练结果字典（含 status / best_pt 路径 / exit_code）

2. `run_inference_and_score(exp_dir, ckpt_path, test_jsonl, fixed_params, project_root, log_path)`
   - 通过 subprocess 调用 infer_lid_chinavoices.py
   - 读取预测 jsonl 与 test_jsonl（gold 来自 test_jsonl 的 accent 字段）
   - 计算 overall_acc / per_class_acc（recall）
   - 写入 test_accuracy.json 并返回结果字典

设计原则：
- 严格通过 subprocess 调用现有脚本，禁止 import 私有函数（隔离约束）。
- 所有输出路径固定在传入的 exp_dir 之内，绝不写入项目其它位置。
- 失败时不抛异常，返回 status='failed' 让主控循环继续下一组实验。
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sweep.runner")

# 训练 / 推理脚本相对项目根目录的固定路径
TRAIN_SCRIPT_REL = "examples_train/lid_chinavoices/finetune_lid_chinavoices.py"
INFER_SCRIPT_REL = "examples_train/lid_chinavoices/infer_lid_chinavoices.py"

# 训练脚本 argparse 参数名到超参键名的映射（当前保持同名）
TRAIN_HPARAM_KEYS: List[str] = [
    "lr", "encoder_lr", "batch_size", "dropout", "weight_decay",
    "label_smoothing", "seed", "grad_clip", "warmup_steps",
    "epochs", "patience", "min_delta", "freeze_encoder",
]


def _count_visible_gpus() -> int:
    """根据 CUDA_VISIBLE_DEVICES 计算可见 GPU 数量。未设置或空则视为 1。"""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 1
    return len([x for x in visible.split(",") if x.strip()])


def _build_train_launcher(nproc_per_node: int) -> List[str]:
    """构造训练启动命令前缀，与 run_finetune_lid_chinavoices.sh 保持行为一致。"""
    if nproc_per_node > 1:
        return [
            "torchrun",
            "--standalone",
            "--nnodes", "1",
            "--nproc_per_node", str(nproc_per_node),
        ]
    return ["python3.10"]


def _format_scalar_arg(value: Any) -> str:
    """把 float/int 值格式化为 CLI 字符串；float 用 repr 避免精度丢失。"""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # repr 保留完整精度，如 1e-05 -> '1e-05'
        return repr(value)
    return str(value)


class _TeeWriter:
    """把子进程的 stdout 同时透传到父进程 stdout 与日志文件。"""

    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()
        self.log_file.write(chunk)
        self.log_file.flush()


def _run_subprocess_with_tee(
    cmd: List[str],
    cwd: str,
    log_path: str,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """启动 subprocess，将合并的 stdout/stderr 同时透传到父进程与 log 文件。

    返回 subprocess 的 exit code。子进程收到 SIGINT/SIGTERM 时向其转发信号。
    """
    logger.info("run subprocess: cwd=%s cmd=%s", cwd, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )

    def _forward(signum, _frame):
        if proc.poll() is None:
            logger.warning("sweep runner: forwarding signal %d to subprocess pid=%d", signum, proc.pid)
            try:
                proc.send_signal(signum)
            except ProcessLookupError:
                pass

    prev_sigint = signal.signal(signal.SIGINT, _forward)
    prev_sigterm = signal.signal(signal.SIGTERM, _forward)

    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            tee = _TeeWriter(log_file)
            assert proc.stdout is not None
            for line in proc.stdout:
                tee.write(line)
        exit_code = proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

    return exit_code


def run_training(
    exp_dir: str,
    hparams: Dict[str, Any],
    fixed_params: Dict[str, Any],
    project_root: str,
) -> Dict[str, Any]:
    """启动一次训练实验。

    Args:
        exp_dir: 本次实验输出目录（必须由主控预先创建好，且位于扫描顶层目录之内）。
        hparams: 实际生效的超参字典（已合并 baseline + 当前维度候选）。
        fixed_params: 公共固定参数（train_jsonl / val_jsonl / pretrained_model_dir / num_workers / use_amp / log_interval 等）。
        project_root: FireRedASR2S-fintuning 项目根目录绝对路径。

    Returns:
        dict:
            status: 'success' | 'failed'
            exit_code: int
            best_pt: Optional[str]  best.pt 绝对路径（若不存在则为 None）
            elapsed_sec: float
            train_log: str          日志文件绝对路径
    """
    exp_dir = os.path.abspath(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    train_log = os.path.join(exp_dir, "train.log")

    nproc = _count_visible_gpus()
    launcher = _build_train_launcher(nproc)

    train_script = os.path.join(project_root, TRAIN_SCRIPT_REL)

    cmd: List[str] = [*launcher, train_script,
                      "--train_jsonl", str(fixed_params["train_jsonl"]),
                      "--val_jsonl", str(fixed_params["val_jsonl"]),
                      "--pretrained_model_dir", str(fixed_params["pretrained_model_dir"]),
                      "--output_dir", exp_dir]

    # 公共可选参数（num_workers / use_amp / log_interval）
    for opt_key in ("num_workers", "use_amp", "log_interval"):
        if opt_key in fixed_params:
            cmd.extend([f"--{opt_key}", _format_scalar_arg(fixed_params[opt_key])])

    # 超参：全部通过 argparse 参数传递
    for key in TRAIN_HPARAM_KEYS:
        if key not in hparams:
            continue
        cmd.extend([f"--{key}", _format_scalar_arg(hparams[key])])

    # 环境：PYTHONPATH 指向 fireredasr2s 内部包路径，与原 shell 脚本行为一致
    env = os.environ.copy()
    pypath_prefix = os.path.join(project_root, "fireredasr2s")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pypath_prefix}:{existing_pp}" if existing_pp else pypath_prefix

    logger.info("[train] exp_dir=%s nproc=%d", exp_dir, nproc)
    logger.info("[train] hparams=%s", {k: hparams[k] for k in TRAIN_HPARAM_KEYS if k in hparams})

    # 记录启动 banner 到 train.log
    with open(train_log, "a", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"[sweep-runner] training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[sweep-runner] cmd = {' '.join(cmd)}\n")
        f.write("=" * 78 + "\n")

    start = time.time()
    try:
        exit_code = _run_subprocess_with_tee(cmd, cwd=project_root, log_path=train_log, env=env)
    except FileNotFoundError as e:
        # launcher 或脚本不存在等启动阶段错误
        logger.error("[train] launch failed: %s", e)
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(f"[sweep-runner] launch failed: {e}\n")
        return {"status": "failed", "exit_code": -1, "best_pt": None,
                "elapsed_sec": time.time() - start, "train_log": train_log}
    elapsed = time.time() - start

    best_pt = os.path.join(exp_dir, "best.pt")
    if exit_code == 0 and os.path.isfile(best_pt):
        return {"status": "success", "exit_code": 0, "best_pt": best_pt,
                "elapsed_sec": elapsed, "train_log": train_log}

    # 训练进程 0 退出但没产出 best.pt（例如 patience=0 且 val_acc 从未提升）
    if exit_code == 0 and not os.path.isfile(best_pt):
        logger.warning("[train] exit_code=0 but best.pt missing at %s", best_pt)
    logger.warning("[train] failed exit_code=%d elapsed=%.1fs", exit_code, elapsed)
    return {"status": "failed", "exit_code": exit_code, "best_pt": None,
            "elapsed_sec": elapsed, "train_log": train_log}


# ---------------------------------------------------------------------------
# 推理 + 打分
# ---------------------------------------------------------------------------

def _read_jsonl_key_to_accent(path: str) -> Dict[str, str]:
    """读取 test_jsonl，返回 key -> accent 的映射（作为 gold 标签查表）。"""
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key") or os.path.splitext(os.path.basename(obj["wav_path"]))[0]
            accent = obj.get("accent", "")
            mapping[key] = accent
    return mapping


def _score_predictions(
    pred_jsonl: str,
    test_jsonl: str,
) -> Dict[str, Any]:
    """比对预测与 gold，输出 overall_acc / per_class 统计。

    - pred_jsonl 由 infer_lid_chinavoices.py 生成，每行含 key / accent（预测）
    - test_jsonl 每行含 key / accent（gold）
    - 只统计"两侧 key 都存在且 gold 非空"的样本
    - per_class 使用 recall = 正确预测 / 该类 gold 数量
    """
    gold_map = _read_jsonl_key_to_accent(test_jsonl)

    total = 0
    correct = 0
    per_class_correct: Dict[str, int] = {}
    per_class_total: Dict[str, int] = {}
    unknown_key = 0
    error_pred = 0

    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj["key"]
            pred = obj.get("accent", "")
            if pred == "error":
                error_pred += 1
                # 视为一次失败预测（但仍需 gold 存在）
            gold = gold_map.get(key)
            if gold is None:
                unknown_key += 1
                continue
            if not gold:
                # gold 为空字符串，跳过
                continue
            per_class_total[gold] = per_class_total.get(gold, 0) + 1
            total += 1
            if pred == gold:
                correct += 1
                per_class_correct[gold] = per_class_correct.get(gold, 0) + 1

    overall_acc = (correct / total) if total > 0 else None
    per_class_acc: Dict[str, float] = {}
    for cls, n in per_class_total.items():
        per_class_acc[cls] = (per_class_correct.get(cls, 0) / n) if n > 0 else 0.0

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "num_samples": total,
        "num_pred_error": error_pred,
        "num_unknown_key": unknown_key,
    }


def run_inference_and_score(
    exp_dir: str,
    ckpt_path: str,
    test_jsonl: str,
    fixed_params: Dict[str, Any],
    project_root: str,
    min_acc_warning_threshold: float = 1.0 / 16.0,
) -> Dict[str, Any]:
    """对 test_jsonl 执行推理并计算 LID 准确率。

    Args:
        exp_dir: 本次实验目录（推理输出/log/accuracy json 都写这里）。
        ckpt_path: best.pt 路径。
        test_jsonl: test 集 jsonl 路径（必须含 accent gold 字段）。
        fixed_params: 公共固定参数（pretrained_model_dir / num_workers 会被使用）。
        project_root: 项目根目录。
        min_acc_warning_threshold: 低于此值打印 warning。

    Returns:
        dict 与 test_accuracy.json 内容一致，同时含 status 字段。
    """
    exp_dir = os.path.abspath(exp_dir)
    pred_jsonl = os.path.join(exp_dir, "pred_test.jsonl")
    infer_log = os.path.join(exp_dir, "infer.log")
    accuracy_json = os.path.join(exp_dir, "test_accuracy.json")

    result: Dict[str, Any] = {
        "overall_acc": None,
        "per_class_acc": {},
        "num_samples": 0,
        "elapsed_sec": 0.0,
        "status": "failed",
        "message": "",
    }

    if not os.path.isfile(ckpt_path):
        result["message"] = f"checkpoint not found: {ckpt_path}"
        logger.error("[infer] %s", result["message"])
        _write_accuracy_json(accuracy_json, result)
        return result

    if not os.path.isfile(test_jsonl):
        result["message"] = f"test_jsonl not found: {test_jsonl}"
        logger.error("[infer] %s", result["message"])
        _write_accuracy_json(accuracy_json, result)
        return result

    infer_script = os.path.join(project_root, INFER_SCRIPT_REL)
    pretrained = str(fixed_params["pretrained_model_dir"])
    batch_size = str(fixed_params.get("infer_batch_size", 64))
    num_workers = str(fixed_params.get("num_workers", 4))

    cmd: List[str] = [
        "python3.10", infer_script,
        "--checkpoint", ckpt_path,
        "--input_jsonl", test_jsonl,
        "--output", pred_jsonl,
        "--pretrained_model_dir", pretrained,
        "--batch_size", batch_size,
        "--num_workers", num_workers,
    ]

    env = os.environ.copy()
    pypath_prefix = os.path.join(project_root, "fireredasr2s")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pypath_prefix}:{existing_pp}" if existing_pp else pypath_prefix
    # 推理默认使用单卡（与 run_decode_test_lid_chinavoices.sh 一致）
    # 若外层设置了多卡 CUDA_VISIBLE_DEVICES，此处只取第一张，避免误触 DDP 推理
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and "," in visible:
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0].strip()

    logger.info("[infer] exp_dir=%s ckpt=%s test=%s", exp_dir, ckpt_path, test_jsonl)
    with open(infer_log, "a", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"[sweep-runner] inference started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[sweep-runner] cmd = {' '.join(cmd)}\n")
        f.write("=" * 78 + "\n")

    start = time.time()
    try:
        exit_code = _run_subprocess_with_tee(cmd, cwd=project_root, log_path=infer_log, env=env)
    except FileNotFoundError as e:
        result["message"] = f"inference launch failed: {e}"
        result["elapsed_sec"] = time.time() - start
        logger.error("[infer] %s", result["message"])
        _write_accuracy_json(accuracy_json, result)
        return result

    if exit_code != 0 or not os.path.isfile(pred_jsonl):
        result["message"] = f"inference subprocess failed exit_code={exit_code}"
        result["elapsed_sec"] = time.time() - start
        logger.error("[infer] %s", result["message"])
        _write_accuracy_json(accuracy_json, result)
        return result

    try:
        stats = _score_predictions(pred_jsonl, test_jsonl)
    except Exception as e:  # 打分逻辑异常兜底
        result["message"] = f"scoring failed: {e}"
        result["elapsed_sec"] = time.time() - start
        logger.error("[infer] %s", result["message"])
        _write_accuracy_json(accuracy_json, result)
        return result

    result.update({
        "overall_acc": stats["overall_acc"],
        "per_class_acc": stats["per_class_acc"],
        "num_samples": stats["num_samples"],
        "num_pred_error": stats["num_pred_error"],
        "num_unknown_key": stats["num_unknown_key"],
        "elapsed_sec": round(time.time() - start, 2),
        "status": "success" if stats["overall_acc"] is not None else "failed",
        "message": "" if stats["overall_acc"] is not None else "no scored samples",
    })
    _write_accuracy_json(accuracy_json, result)

    if result["overall_acc"] is not None and result["overall_acc"] < min_acc_warning_threshold:
        logger.warning("[infer] overall_acc=%.4f 低于随机基线阈值 %.4f，请检查此次实验",
                       result["overall_acc"], min_acc_warning_threshold)

    return result


def _write_accuracy_json(path: str, result: Dict[str, Any]) -> None:
    payload = {
        "overall_acc": result["overall_acc"],
        "per_class_acc": result.get("per_class_acc", {}),
        "num_samples": result.get("num_samples", 0),
        "num_pred_error": result.get("num_pred_error", 0),
        "num_unknown_key": result.get("num_unknown_key", 0),
        "elapsed_sec": result.get("elapsed_sec", 0.0),
        "status": result.get("status", "failed"),
        "message": result.get("message", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


__all__ = ["run_training", "run_inference_and_score", "TRAIN_HPARAM_KEYS"]
