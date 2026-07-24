# -*- coding: utf-8 -*-
"""ASR 超参扫描 · 单次实验运行器。

包含 4 个核心函数：

1. `convert_test_jsonl_if_needed(src, dst)`
   - 把用户 test JSONL 的字段 {key, wav_path, text, accent} 转成推理脚本
     需要的 {audio, text: "language Chinese <accent><asr_text><text>", key,
     accent}
   - 若源已符合推理格式（含 `audio` 字段），直接复用
   - 输出文件仅在缺失时才写入，避免重复 IO

2. `run_training(exp_dir, hparams, cfg, project_root)`
   - torchrun --nproc_per_node=<N> 启动 finetune_asr_chinavoices.py
   - 通过检查 model.pth.tar / best_metrics.json / accents.json 判定成功
     （不依赖 return code，应对 NCCL destroy 报错场景）

3. `run_inference(exp_dir, test_input_jsonl, cfg, project_root)`
   - 单进程调用 decode_asr_chinavoices.py --gpu-ids all
   - 显式覆盖 --input-jsonl / --output-jsonl，避免落回默认 ichiwang 路径
   - 断点续跑：若 pred_test.jsonl 已存在且行数正确，直接跳过

4. `run_format_and_score(exp_dir, cfg, expected_sample_count)`
   - 依次执行 格式化 → 官方 CER 评估 → 解析 result.wer → 写 test_metrics.json
   - 严禁自己实现字符编辑距离，全部走 eval_jsonl_with_wer_tools.sh

设计原则：
- 严格通过 subprocess 调用现有脚本，禁止 import 私有函数（隔离约束）。
- 所有 IO 都在 exp_dir 之内，绝不写入项目其它位置。
- 失败时抛出自定义异常，由主控统一处理。
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sweep.runner")

# 训练 / 推理脚本相对项目根目录的固定路径
TRAIN_SCRIPT_REL = "examples_train/asr_chinavoices/finetune_asr_chinavoices.py"
INFER_SCRIPT_REL = "examples_train/asr_chinavoices/decode_asr_chinavoices.py"

# 训练脚本 argparse 参数名（与 finetune_asr_chinavoices.py 完全一致）
TRAIN_HPARAM_KEYS: List[str] = [
    "epochs", "batch_size", "grad_accum_steps", "num_workers",
    "encoder_lr", "decoder_lr", "weight_decay", "ctc_weight",
    "label_smoothing", "warmup_steps", "grad_clip",
    "max_input_frames", "max_target_length",
    "use_amp", "save_optimizer", "seed", "log_interval",
]

# 训练脚本产物齐备判据（用于绕过 NCCL destroy 引发的非零 return code）
TRAIN_SUCCESS_MARKERS: Tuple[str, ...] = (
    "model.pth.tar", "best_metrics.json", "accents.json",
)

# 推理脚本要求的 JSONL 字段
DECODE_INPUT_KEY_AUDIO = "audio"
DECODE_INPUT_KEY_TEXT = "text"

# ChinaVoices ref_full 前缀模板：language Chinese <accent><asr_text><text>
# 与 decode_asr_chinavoices.py::split_reference 的解析规则严格对齐
ASR_MARKER = "<asr_text>"


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class TrainFailed(RuntimeError):
    """训练阶段失败（产物缺失或子进程非 0 退出且产物不齐）。"""


class InferFailed(RuntimeError):
    """推理阶段失败（子进程非 0 退出、产物缺失或行数不一致）。"""


class EvalFailed(RuntimeError):
    """格式化或 CER 评估阶段失败。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _count_visible_gpus() -> int:
    """根据 CUDA_VISIBLE_DEVICES 计算可见 GPU 数量。未设置或空则视为 1。"""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 1
    return len([x for x in visible.split(",") if x.strip()])


def _format_scalar_arg(value: Any) -> str:
    """把 float/int/bool 值格式化为 CLI 字符串；float 用 repr 避免精度丢失。"""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _count_jsonl_lines(path: str) -> int:
    """统计 JSONL 有效行数（去除空行）。"""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


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

    子进程收到 SIGINT/SIGTERM 时向其转发信号，确保 Ctrl+C 能真正终止训练进程。
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
            logger.warning("forwarding signal %d to subprocess pid=%d", signum, proc.pid)
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


def _build_env(project_root: str) -> Dict[str, str]:
    env = os.environ.copy()
    pypath_prefix = os.path.join(project_root, "fireredasr2s")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pypath_prefix}:{existing_pp}" if existing_pp else pypath_prefix
    return env


# ---------------------------------------------------------------------------
# 1. Test JSONL 字段转换（推理输入格式对齐）
# ---------------------------------------------------------------------------

def convert_test_jsonl_if_needed(src: str, dst: str) -> int:
    """把用户 test JSONL 转换为推理脚本消费的格式，返回样本数。

    - 输入字段：{key, wav_path, text, accent}
    - 输出字段：{key, audio: <wav_path>, text: "language Chinese <accent><asr_text><text>", accent}

    若源文件已具备 `audio` 字段（直接是推理格式），则原地复制。
    若 dst 已存在且行数与 src 一致，则跳过重复转换（复用旧结果）。

    Args:
        src: 用户 test JSONL 路径。
        dst: 转换后的推理输入 JSONL 路径。

    Returns:
        转换后（或跳过后）的样本数。
    """
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.is_file():
        raise FileNotFoundError(f"test_jsonl 不存在：{src}")

    src_lines = _count_jsonl_lines(src)
    if dst_p.is_file():
        dst_lines = _count_jsonl_lines(dst)
        if dst_lines == src_lines and src_lines > 0:
            logger.info("[convert] 复用已存在的转换文件：%s (%d 行)", dst, dst_lines)
            return dst_lines
        else:
            logger.warning("[convert] 目标文件行数不匹配（src=%d dst=%d），重新生成：%s",
                           src_lines, dst_lines, dst)

    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_suffix(dst_p.suffix + ".tmp")

    n = 0
    with src_p.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{src}:{line_no} 非合法 JSON：{exc}") from exc

            # 若源已是推理格式（含 audio 字段），保持原样即可
            if DECODE_INPUT_KEY_AUDIO in obj:
                out = obj
            else:
                # 从 wav_path 构造 audio
                wav_path = obj.get("wav_path")
                if not isinstance(wav_path, str) or not wav_path.strip():
                    raise ValueError(f"{src}:{line_no} 缺少 wav_path 或 audio 字段")
                key = str(obj.get("key") or Path(wav_path).stem).strip()
                text = str(obj.get("text") or "").strip()
                accent = str(obj.get("accent") or "").strip()
                # 组装 ref_full 格式：language Chinese <accent><asr_text><text>
                if accent and text:
                    full_text = f"language Chinese {accent}{ASR_MARKER}{text}"
                elif text:
                    full_text = f"{ASR_MARKER}{text}"
                else:
                    full_text = ""
                out = {
                    "audio": wav_path,
                    "text": full_text,
                    "key": key,
                    "accent": accent,
                }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    os.replace(tmp, dst_p)
    logger.info("[convert] 生成推理输入文件：%s (%d 行)", dst, n)
    return n


# ---------------------------------------------------------------------------
# 2. 训练
# ---------------------------------------------------------------------------

def _train_success_by_artifacts(exp_dir: str) -> bool:
    """通过产物齐备性判定训练是否实际成功（应对 NCCL destroy 引发的非零 return code）。"""
    for marker in TRAIN_SUCCESS_MARKERS:
        if not (Path(exp_dir) / marker).is_file():
            return False
    return True


def run_training(
    exp_dir: str,
    hparams: Dict[str, Any],
    cfg,
    project_root: str,
) -> Dict[str, Any]:
    """启动一次 ASR 全量微调。

    Args:
        exp_dir: 本次实验输出目录（必须已创建好）。
        hparams: 实际生效的完整超参字典（已合并 baseline + 本次覆盖）。
        cfg: SweepConfig 实例（提供 data_paths / pretrained_model_dir / nproc_per_node）。
        project_root: 项目根目录绝对路径。

    Returns:
        dict:
            status: 'success' | 'failed'
            exit_code: int
            best_metrics: Optional[Dict]   # best_metrics.json 的内容
            elapsed_sec: float
            train_log: str

    Raises:
        TrainFailed: 缺失关键产物时抛出，主控可标记 failed_stage='train'。
    """
    exp_dir = os.path.abspath(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    train_log = os.path.join(exp_dir, "train.log")

    # 断点续跑：若训练产物已齐备（model.pth.tar / best_metrics.json / accents.json 都在），
    # 直接复用，不重训。用于应对"训练已成功但下游 eval 失败后清理 status.json 再续跑"等场景。
    # 放宽判据：若 pred_test.jsonl 已存在（意味着上次训练成功过且推理已跑完），
    # 即使 model.pth.tar 已被清理，也允许跳过训练——因为下游只需要推理产物。
    pred_jsonl_marker = Path(exp_dir) / "pred_test.jsonl"
    train_success_full = _train_success_by_artifacts(exp_dir)
    train_success_lite = (
        pred_jsonl_marker.is_file()
        and (Path(exp_dir) / "best_metrics.json").is_file()
    )
    if train_success_full or train_success_lite:
        if train_success_full:
            logger.info("[train] 复用已存在的训练产物：%s (产物齐备，跳过训练)", exp_dir)
        else:
            logger.info(
                "[train] 检测到推理产物 pred_test.jsonl 与 best_metrics.json 已存在（%s），"
                "视为训练已成功完成，跳过训练",
                exp_dir,
            )
        best_metrics_reused: Optional[Dict[str, Any]] = None
        best_metrics_path_reused = Path(exp_dir) / "best_metrics.json"
        try:
            with best_metrics_path_reused.open("r", encoding="utf-8") as f:
                best_metrics_reused = json.load(f)
        except Exception as e:
            logger.warning("[train] 复用分支无法解析 best_metrics.json: %s", e)
        return {
            "status": "success",
            "exit_code": 0,
            "best_metrics": best_metrics_reused,
            "elapsed_sec": 0.0,
            "train_log": train_log,
        }

    nproc = cfg.run_options.nproc_per_node
    visible = _count_visible_gpus()
    # 若 nproc_per_node 与可见卡数不匹配，会在 config_loader 阶段就拦截；
    # 这里再兜底一次，防止有人绕过配置直接调 runner。
    if visible > 1 and visible != nproc:
        logger.warning("[train] nproc_per_node=%d 与可见 GPU 数=%d 不一致；仍按 nproc=%d 启动",
                       nproc, visible, nproc)

    if nproc > 1:
        launcher = [
            "torchrun",
            "--standalone",
            "--nnodes", "1",
            "--nproc_per_node", str(nproc),
        ]
    else:
        launcher = ["python3.10"]

    train_script = os.path.join(project_root, TRAIN_SCRIPT_REL)

    cmd: List[str] = [
        *launcher, train_script,
        "--train_jsonl", str(cfg.data_paths["train_jsonl"]),
        "--val_jsonl", str(cfg.data_paths["val_jsonl"]),
        "--pretrained_model_dir", str(cfg.pretrained_model_dir),
        "--output_dir", exp_dir,
    ]

    # 超参：全部通过 argparse 参数传递（顺序与 TRAIN_HPARAM_KEYS 一致）
    for key in TRAIN_HPARAM_KEYS:
        if key in hparams:
            cmd.extend([f"--{key}", _format_scalar_arg(hparams[key])])

    env = _build_env(project_root)

    logger.info("[train] exp_dir=%s nproc=%d", exp_dir, nproc)
    logger.info("[train] hparams=%s", {k: hparams[k] for k in TRAIN_HPARAM_KEYS if k in hparams})

    with open(train_log, "a", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"[sweep-runner] training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[sweep-runner] cmd = {' '.join(cmd)}\n")
        f.write("=" * 78 + "\n")

    start = time.time()
    try:
        exit_code = _run_subprocess_with_tee(cmd, cwd=project_root, log_path=train_log, env=env)
    except FileNotFoundError as e:
        logger.error("[train] launch failed: %s", e)
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(f"[sweep-runner] launch failed: {e}\n")
        raise TrainFailed(f"launch failed: {e}") from e

    elapsed = time.time() - start

    # 判定实际状态：产物齐备优先，退出码作为辅助信号
    artifacts_ok = _train_success_by_artifacts(exp_dir)
    if not artifacts_ok:
        logger.error("[train] failed exit_code=%d elapsed=%.1fs 产物缺失", exit_code, elapsed)
        missing = [m for m in TRAIN_SUCCESS_MARKERS
                   if not (Path(exp_dir) / m).is_file()]
        raise TrainFailed(f"训练产物缺失: {missing}, exit_code={exit_code}")

    if exit_code != 0:
        logger.warning(
            "[train] 子进程非 0 退出 (exit=%d) 但产物齐备，视为成功（应对 NCCL destroy 报错场景）",
            exit_code,
        )

    # 读取 best_metrics.json 以便主控写入训练摘要
    best_metrics: Optional[Dict[str, Any]] = None
    best_metrics_path = Path(exp_dir) / "best_metrics.json"
    try:
        with best_metrics_path.open("r", encoding="utf-8") as f:
            best_metrics = json.load(f)
    except Exception as e:
        logger.warning("[train] 无法解析 best_metrics.json: %s", e)

    return {
        "status": "success",
        "exit_code": exit_code,
        "best_metrics": best_metrics,
        "elapsed_sec": elapsed,
        "train_log": train_log,
    }


# ---------------------------------------------------------------------------
# 3. 推理
# ---------------------------------------------------------------------------

def run_inference(
    exp_dir: str,
    test_input_jsonl: str,
    cfg,
    project_root: str,
) -> Dict[str, Any]:
    """启动 test 集推理。

    Args:
        exp_dir: 本次实验目录（推理输出 pred_test.jsonl 写入此处）。
        test_input_jsonl: 已完成字段转换的推理输入 JSONL（含 audio 字段）。
        cfg: SweepConfig 实例（提供 infer_args）。
        project_root: 项目根目录。

    Returns:
        dict:
            status: 'success' | 'reused'
            pred_jsonl: str            推理产物路径
            elapsed_sec: float
            infer_log: str
            num_samples: int

    Raises:
        InferFailed: 子进程失败、产物缺失或行数与输入不一致。
    """
    exp_dir = os.path.abspath(exp_dir)
    pred_jsonl = os.path.join(exp_dir, "pred_test.jsonl")
    infer_log = os.path.join(exp_dir, "infer.log")

    expected_count = _count_jsonl_lines(test_input_jsonl)
    if expected_count <= 0:
        raise InferFailed(f"test_input_jsonl 为空：{test_input_jsonl}")

    # 断点续跑：若 pred_test.jsonl 存在且行数一致 → 直接复用
    if os.path.isfile(pred_jsonl):
        try:
            existing = _count_jsonl_lines(pred_jsonl)
        except Exception:
            existing = -1
        if existing == expected_count:
            logger.info("[infer] 复用已存在的推理产物：%s (%d 行)", pred_jsonl, existing)
            return {
                "status": "reused",
                "pred_jsonl": pred_jsonl,
                "elapsed_sec": 0.0,
                "infer_log": infer_log,
                "num_samples": existing,
            }
        else:
            logger.warning("[infer] 已有 pred_test.jsonl 行数不匹配（实际=%d 期望=%d），重新推理",
                           existing, expected_count)

    infer_script = os.path.join(project_root, INFER_SCRIPT_REL)

    infer_args = cfg.infer_args
    cmd: List[str] = [
        "python3.10", infer_script,
        "--model-dir", exp_dir,
        "--input-jsonl", test_input_jsonl,
        "--output-jsonl", pred_jsonl,
        "--gpu-ids", str(infer_args["gpu_ids"]),
        "--batch-size", str(infer_args["batch_size"]),
        "--beam-size", str(infer_args["beam_size"]),
        "--decode-max-len", str(infer_args["decode_max_len"]),
        "--softmax-smoothing", repr(float(infer_args["softmax_smoothing"])),
        "--length-penalty", repr(float(infer_args["length_penalty"])),
        "--eos-penalty", repr(float(infer_args["eos_penalty"])),
        "--log-interval", str(infer_args["log_interval"]),
    ]
    if infer_args.get("use_half"):
        cmd.append("--use-half")

    env = _build_env(project_root)

    logger.info("[infer] exp_dir=%s input=%s expected=%d", exp_dir, test_input_jsonl, expected_count)
    with open(infer_log, "a", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"[sweep-runner] inference started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[sweep-runner] cmd = {' '.join(cmd)}\n")
        f.write("=" * 78 + "\n")

    start = time.time()
    try:
        exit_code = _run_subprocess_with_tee(cmd, cwd=project_root, log_path=infer_log, env=env)
    except FileNotFoundError as e:
        raise InferFailed(f"inference launch failed: {e}") from e

    elapsed = time.time() - start

    if exit_code != 0:
        raise InferFailed(f"inference subprocess failed exit_code={exit_code}")
    if not os.path.isfile(pred_jsonl):
        raise InferFailed(f"inference done but pred_jsonl missing: {pred_jsonl}")

    actual = _count_jsonl_lines(pred_jsonl)
    if actual != expected_count:
        raise InferFailed(
            f"inference sample count mismatch: got={actual} expected={expected_count}"
        )

    return {
        "status": "success",
        "pred_jsonl": pred_jsonl,
        "elapsed_sec": elapsed,
        "infer_log": infer_log,
        "num_samples": actual,
    }


# ---------------------------------------------------------------------------
# 4. 格式化 + 官方 CER 评估
# ---------------------------------------------------------------------------

# 匹配 tools/wer.py 输出中的 "Overall -> <cer>%" 行
# 典型格式："Overall -> 13.45 % N=xxx C=xxx S=xxx D=xxx I=xxx"
_OVERALL_CER_RE = re.compile(r"Overall\s*->\s*([\d.]+)\s*%")


def _parse_result_wer(result_wer_path: str) -> float:
    """从 result.wer 中提取 Overall CER（百分数，例如 12.34）。"""
    if not os.path.isfile(result_wer_path):
        raise EvalFailed(f"result.wer 不存在：{result_wer_path}")
    last_val: Optional[float] = None
    with open(result_wer_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _OVERALL_CER_RE.search(line)
            if m:
                try:
                    last_val = float(m.group(1))
                except ValueError:
                    continue
    if last_val is None:
        raise EvalFailed(f"result.wer 中未找到 'Overall ->' 行：{result_wer_path}")
    return last_val


def _parse_by_dialect_summary(summary_path: str) -> Dict[str, float]:
    """从 by_dialect_summary.txt 中解析 {dialect: cer_percent} 字典。

    文件格式：
        dialect            samples          wer
        ----------------------------------------------
        anhui                  350       14.91%
        cantonese              385        7.45%
        ...
    """
    result: Dict[str, float] = {}
    if not os.path.isfile(summary_path):
        return result
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            if line.lstrip().startswith("dialect") or line.lstrip().startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            dialect = parts[0]
            wer_str = parts[-1].rstrip("%")
            try:
                wer_val = float(wer_str)
            except ValueError:
                continue
            result[dialect] = wer_val
    return result


def run_format_and_score(
    exp_dir: str,
    cfg,
    expected_sample_count: int,
) -> Dict[str, Any]:
    """执行 格式化 → 官方 CER 评估 → 解析 result.wer。

    Args:
        exp_dir: 本次实验目录（必须已含 pred_test.jsonl）。
        cfg: SweepConfig 实例（提供 format_script / eval_tool_sh）。
        expected_sample_count: 推理输入样本数（用于校验 pred 行数一致）。

    Returns:
        dict:
            status: 'success'
            overall_cer: float                     # 百分数，例如 12.34
            per_dialect_cer: Dict[str, float]
            macro_cer_from_per_dialect: float      # 各方言 CER 的算术平均（仅参考）
            test_metrics_path: str

    Raises:
        EvalFailed: 格式化/评估失败或解析异常。
    """
    exp_dir = os.path.abspath(exp_dir)
    pred_jsonl = os.path.join(exp_dir, "pred_test.jsonl")
    formatted_jsonl = os.path.join(exp_dir, "pred_test_formatted.jsonl")
    wer_eval_dir = os.path.join(exp_dir, "wer_eval")
    result_wer_path = os.path.join(wer_eval_dir, "result.wer")
    summary_path = os.path.join(wer_eval_dir, "by_dialect_summary.txt")
    test_metrics_path = os.path.join(exp_dir, "test_metrics.json")
    eval_log = os.path.join(exp_dir, "eval.log")

    if not os.path.isfile(pred_jsonl):
        raise EvalFailed(f"pred_test.jsonl 不存在：{pred_jsonl}")

    # 校验行数一致
    actual = _count_jsonl_lines(pred_jsonl)
    if actual != expected_sample_count:
        raise EvalFailed(
            f"pred_test.jsonl 行数与 test 集不一致：pred={actual} test={expected_sample_count}"
        )

    # --- 步骤 A：格式化（若产物已存在则跳过） ---
    if not os.path.isfile(formatted_jsonl):
        cmd_fmt = [
            "python3.10",
            str(cfg.format_script),
            "--input", pred_jsonl,
            "--output", formatted_jsonl,
        ]
        logger.info("[eval] 格式化 pred_test.jsonl → pred_test_formatted.jsonl")
        with open(eval_log, "a", encoding="utf-8") as f:
            f.write("=" * 78 + "\n")
            f.write(f"[sweep-runner] format started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"[sweep-runner] cmd = {' '.join(cmd_fmt)}\n")
            f.write("=" * 78 + "\n")
        env = os.environ.copy()
        try:
            code = _run_subprocess_with_tee(cmd_fmt, cwd=exp_dir, log_path=eval_log, env=env)
        except FileNotFoundError as e:
            raise EvalFailed(f"format launch failed: {e}") from e
        if code != 0 or not os.path.isfile(formatted_jsonl):
            raise EvalFailed(f"format failed exit_code={code}")
    else:
        logger.info("[eval] 复用已存在的格式化产物：%s", formatted_jsonl)

    # --- 步骤 B：官方 CER 评估（若 result.wer 已存在则跳过） ---
    if not os.path.isfile(result_wer_path):
        cmd_eval = [
            "bash",
            str(cfg.eval_tool_sh),
            "--pred_jsonl", formatted_jsonl,
            "--output_dir", wer_eval_dir,
            "--apply_t2s", "1",
            "--by_dialect", "1",
        ]
        logger.info("[eval] 调用官方 CER 评估工具：%s", cfg.eval_tool_sh)
        with open(eval_log, "a", encoding="utf-8") as f:
            f.write("=" * 78 + "\n")
            f.write(f"[sweep-runner] cer eval started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"[sweep-runner] cmd = {' '.join(cmd_eval)}\n")
            f.write("=" * 78 + "\n")
        env = os.environ.copy()
        try:
            code = _run_subprocess_with_tee(cmd_eval, cwd=exp_dir, log_path=eval_log, env=env)
        except FileNotFoundError as e:
            raise EvalFailed(f"eval launch failed: {e}") from e
        if code != 0:
            raise EvalFailed(f"eval subprocess failed exit_code={code}")
        if not os.path.isfile(result_wer_path):
            raise EvalFailed(f"eval done but result.wer missing: {result_wer_path}")
    else:
        logger.info("[eval] 复用已存在的评估产物：%s", result_wer_path)

    # --- 步骤 C：解析 result.wer 与 by_dialect_summary.txt ---
    overall_cer = _parse_result_wer(result_wer_path)
    per_dialect = _parse_by_dialect_summary(summary_path)
    macro = (sum(per_dialect.values()) / len(per_dialect)) if per_dialect else float("nan")

    # --- 步骤 D：写 test_metrics.json ---
    payload = {
        "overall_cer": overall_cer,             # 百分数
        "per_dialect_cer": per_dialect,          # 百分数
        "macro_cer_from_per_dialect": (
            macro if not math.isnan(macro) else None
        ),
        "num_samples": actual,
        "result_wer_path": os.path.relpath(result_wer_path, exp_dir),
        "by_dialect_summary_path": os.path.relpath(summary_path, exp_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = test_metrics_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, test_metrics_path)

    logger.info(
        "[eval] overall_cer=%.2f%% macro_from_per_dialect=%s per_dialect_keys=%d",
        overall_cer,
        f"{macro:.2f}%" if not math.isnan(macro) else "N/A",
        len(per_dialect),
    )

    return {
        "status": "success",
        "overall_cer": overall_cer,
        "per_dialect_cer": per_dialect,
        "macro_cer_from_per_dialect": macro if not math.isnan(macro) else None,
        "test_metrics_path": test_metrics_path,
    }


__all__ = [
    "TrainFailed",
    "InferFailed",
    "EvalFailed",
    "convert_test_jsonl_if_needed",
    "run_training",
    "run_inference",
    "run_format_and_score",
    "TRAIN_HPARAM_KEYS",
]
