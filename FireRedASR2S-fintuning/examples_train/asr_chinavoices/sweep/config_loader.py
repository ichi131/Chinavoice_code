# -*- coding: utf-8 -*-
"""ASR 超参扫描配置加载与校验模块。

职责：
1. 读取 YAML 配置文件（缺失/格式错误立即抛异常，不启动任何训练）。
2. 校验 data_paths / pretrained_model_dir / format_script / eval_tool_sh /
   baseline_hparams / sweep_order / sweep_space / infer_args / run_options
   全部段落合法。
3. 展开成 `SweepConfig` 数据类，供 sweep_main.py 直接消费。

设计原则：
- 与训练/推理脚本完全解耦，只依赖 PyYAML。
- 所有可扫描超参维护在 ALLOWED_HPARAMS 白名单中，未知维度立即报错。
- baseline_hparams 必须完备覆盖 REQUIRED_BASELINE_KEYS。
- sweep_space[dim] 必须包含 baseline_hparams[dim] 的取值（否则 baseline 就
  永远失去被评估的机会，与"贪心式"语义不符）。
- 提供 `float_close` 供 sweep_main 判定"候选值 == baseline 值"以触发复用。
"""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


# 允许被扫描/覆盖的超参白名单：需与 finetune_asr_chinavoices.py 中 argparse 定义严格一致
# 说明：train_jsonl/val_jsonl/pretrained_model_dir/output_dir/resume 等属于路径类，
# 通过 data_paths / pretrained_model_dir 或运行时逻辑传入，不放入白名单。
ALLOWED_HPARAMS: Dict[str, type] = {
    "epochs": int,
    "batch_size": int,
    "grad_accum_steps": int,
    "num_workers": int,
    "encoder_lr": float,
    "decoder_lr": float,
    "weight_decay": float,
    "ctc_weight": float,
    "label_smoothing": float,
    "warmup_steps": int,
    "grad_clip": float,
    "max_input_frames": int,
    "max_target_length": int,
    "use_amp": int,           # 训练脚本按 int 接受（1/0）
    "save_optimizer": int,
    "seed": int,
    "log_interval": int,
}

# baseline_hparams 中必须显式给出的字段
REQUIRED_BASELINE_KEYS: List[str] = list(ALLOWED_HPARAMS.keys())

# data_paths 中必须显式给出的字段
REQUIRED_DATA_KEYS: List[str] = ["train_jsonl", "val_jsonl", "test_jsonl"]

# 预训练模型目录必须包含的资产
PRETRAINED_REQUIRED_FILES: List[str] = [
    "model.pth.tar", "cmvn.ark", "dict.txt", "train_bpe1000.model",
]

# infer_args 允许的字段
ALLOWED_INFER_ARGS: Dict[str, type] = {
    "gpu_ids": str,           # 'all' 或 '0,1,2'
    "batch_size": int,
    "use_half": bool,
    "beam_size": int,
    "decode_max_len": int,
    "softmax_smoothing": float,
    "length_penalty": float,
    "eos_penalty": float,
    "log_interval": int,
}


class ConfigError(RuntimeError):
    """配置校验失败时抛出。"""


@dataclasses.dataclass
class DimensionPlan:
    """单个扫描维度的展开计划。"""
    index: int                # 从 1 开始的维度序号
    name: str                 # 维度名（例如 'encoder_lr'）
    candidates: List[Any]     # 候选值列表
    skipped: bool             # 是否直接固化（长度为 1 时为 True）


@dataclasses.dataclass
class RunOptions:
    """扫描主控运行选项。"""
    base_output_dir: str
    sweep_dir_prefix: str
    keep_only_best_ckpt: bool
    nproc_per_node: int


@dataclasses.dataclass
class SweepConfig:
    """完整的扫描计划（供 sweep_main 直接消费）。"""
    data_paths: Dict[str, str]
    pretrained_model_dir: str
    format_script: str
    eval_tool_sh: str
    baseline_hparams: Dict[str, Any]
    dimensions: List[DimensionPlan]
    infer_args: Dict[str, Any]
    run_options: RunOptions
    raw_config_path: str


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def float_close(a: Any, b: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """浮点友好的相等判定：优先按 float 比较，非数值退化到 == 。

    - int 与 float 之间可以互相判等（1 == 1.0）。
    - str/bool 使用 == 。
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    return a == b


def _coerce_scalar(name: str, value: Any) -> Any:
    """按白名单类型定义把 YAML 里的标量强转为 Python 原生类型。

    - bool 不属于允许类型，会被 int(True) 意外接受，需在此显式拒绝。
    - int 字段收到 float 但值为整数时接受（例如 1000.0 -> 1000）。
    """
    expected = ALLOWED_HPARAMS[name]
    if isinstance(value, bool):
        raise ConfigError(f"超参 `{name}` 不接受布尔值：{value!r}")
    if expected is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        raise ConfigError(f"超参 `{name}` 期望 int，但收到 {type(value).__name__}: {value!r}")
    if expected is float:
        if isinstance(value, (int, float)):
            return float(value)
        raise ConfigError(f"超参 `{name}` 期望 float，但收到 {type(value).__name__}: {value!r}")
    raise ConfigError(f"超参 `{name}` 白名单类型定义异常（内部错误）")


def _require_mapping(obj: Any, section: str) -> Dict[str, Any]:
    if obj is None:
        raise ConfigError(f"配置缺少必填段：`{section}`")
    if not isinstance(obj, dict):
        raise ConfigError(f"配置段 `{section}` 必须是映射（当前类型：{type(obj).__name__}）")
    return obj


def _check_file(path: str, section: str, must_be_nonempty: bool = True) -> None:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"{section} 指向的文件不存在：{path}")
    if must_be_nonempty and p.stat().st_size == 0:
        raise ConfigError(f"{section} 指向的文件为空：{path}")


# ---------------------------------------------------------------------------
# 分段校验
# ---------------------------------------------------------------------------

def _validate_data_paths(raw: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key in REQUIRED_DATA_KEYS:
        if key not in raw:
            raise ConfigError(f"data_paths 缺少必填字段：`{key}`")
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"data_paths.{key} 必须是非空字符串路径")
        _check_file(value, f"data_paths.{key}")
        result[key] = value
    return result


def _validate_pretrained(pretrained_dir: str) -> str:
    if not isinstance(pretrained_dir, str) or not pretrained_dir.strip():
        raise ConfigError("pretrained_model_dir 必须是非空字符串路径")
    d = Path(pretrained_dir)
    if not d.is_dir():
        raise ConfigError(f"pretrained_model_dir 不存在或不是目录：{pretrained_dir}")
    for fn in PRETRAINED_REQUIRED_FILES:
        fp = d / fn
        if not fp.is_file():
            raise ConfigError(f"pretrained_model_dir 缺少必需文件：{fp}")
    return pretrained_dir


def _validate_script(path: str, section: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ConfigError(f"{section} 必须是非空字符串路径")
    _check_file(path, section)
    return path


def _validate_baseline(raw: Dict[str, Any]) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {}
    for key in REQUIRED_BASELINE_KEYS:
        if key not in raw:
            raise ConfigError(f"baseline_hparams 缺少必填字段：`{key}`")
        baseline[key] = _coerce_scalar(key, raw[key])
    # 拒绝白名单外的额外字段（避免用户拼错键名却无提示）
    for key in raw:
        if key not in ALLOWED_HPARAMS:
            raise ConfigError(
                f"baseline_hparams 出现未知字段：`{key}`；"
                f"合法字段：{sorted(ALLOWED_HPARAMS)}"
            )
    return baseline


def _validate_sweep(
    sweep_order: Sequence[Any],
    sweep_space: Dict[str, Any],
    baseline: Dict[str, Any],
) -> List[DimensionPlan]:
    if not isinstance(sweep_order, (list, tuple)) or not sweep_order:
        raise ConfigError("sweep_order 必须是非空列表")
    seen = set()
    dimensions: List[DimensionPlan] = []
    for idx, name in enumerate(sweep_order, start=1):
        if not isinstance(name, str):
            raise ConfigError(f"sweep_order 元素必须是字符串，位置 {idx} 收到：{name!r}")
        if name in seen:
            raise ConfigError(f"sweep_order 存在重复维度：`{name}`")
        seen.add(name)
        if name not in ALLOWED_HPARAMS:
            raise ConfigError(
                f"sweep_order[`{name}`] 不在允许白名单；合法值：{sorted(ALLOWED_HPARAMS)}"
            )
        if name not in sweep_space:
            raise ConfigError(f"sweep_space 缺少维度 `{name}` 的候选值列表")
        if name not in baseline:
            raise ConfigError(f"sweep_order[`{name}`] 必须先在 baseline_hparams 中给出取值")

        raw_candidates = sweep_space[name]
        if not isinstance(raw_candidates, (list, tuple)) or len(raw_candidates) == 0:
            raise ConfigError(f"sweep_space.{name} 必须是非空列表")
        candidates = [_coerce_scalar(name, v) for v in raw_candidates]

        # 保序去重（同一维度多次相同候选属于配置冗余，直接判错）
        # 用 float_close 做浮点友好去重
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if float_close(candidates[i], candidates[j]):
                    raise ConfigError(
                        f"sweep_space.{name} 存在重复（或数值等价）的候选值："
                        f"index {i}={candidates[i]!r} 与 index {j}={candidates[j]!r}"
                    )

        # 每维候选必须包含当前 baseline 值（否则贪心链断裂）
        baseline_v = baseline[name]
        if not any(float_close(c, baseline_v) for c in candidates):
            raise ConfigError(
                f"sweep_space.{name} 必须包含 baseline_hparams.{name} 的取值 {baseline_v!r}，"
                f"当前候选：{candidates}"
            )

        dimensions.append(DimensionPlan(
            index=idx,
            name=name,
            candidates=candidates,
            skipped=(len(candidates) == 1),
        ))
    return dimensions


def _validate_infer_args(raw: Any) -> Dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("infer_args 必须是映射")
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in ALLOWED_INFER_ARGS:
            raise ConfigError(
                f"infer_args 出现未知字段：`{key}`；合法值：{sorted(ALLOWED_INFER_ARGS)}"
            )
        expected = ALLOWED_INFER_ARGS[key]
        if expected is bool:
            if not isinstance(value, bool):
                raise ConfigError(f"infer_args.{key} 必须是 bool，收到：{value!r}")
            result[key] = value
        elif expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"infer_args.{key} 必须是 int，收到：{value!r}")
            result[key] = value
        elif expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"infer_args.{key} 必须是数字，收到：{value!r}")
            result[key] = float(value)
        elif expected is str:
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"infer_args.{key} 必须是非空字符串，收到：{value!r}")
            result[key] = value.strip()
        else:
            raise ConfigError(f"infer_args.{key} 类型未知")
    # 默认填充
    result.setdefault("gpu_ids", "all")
    result.setdefault("batch_size", 16)
    result.setdefault("use_half", True)
    result.setdefault("beam_size", 3)
    result.setdefault("decode_max_len", 300)
    result.setdefault("softmax_smoothing", 1.25)
    result.setdefault("length_penalty", 0.6)
    result.setdefault("eos_penalty", 1.0)
    result.setdefault("log_interval", 20)
    return result


def _validate_run_options(raw: Any) -> RunOptions:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("run_options 必须是映射")

    base_output_dir = raw.get("base_output_dir", "./exp")
    sweep_dir_prefix = raw.get("sweep_dir_prefix", "asr_sweep")
    keep_only_best = raw.get("keep_only_best_ckpt", True)
    nproc_per_node = raw.get("nproc_per_node", 8)

    if not isinstance(base_output_dir, str) or not base_output_dir.strip():
        raise ConfigError("run_options.base_output_dir 必须是非空字符串")
    if not isinstance(sweep_dir_prefix, str) or not sweep_dir_prefix.strip():
        raise ConfigError("run_options.sweep_dir_prefix 必须是非空字符串")
    if not isinstance(keep_only_best, bool):
        raise ConfigError("run_options.keep_only_best_ckpt 必须是 bool")
    if not isinstance(nproc_per_node, int) or isinstance(nproc_per_node, bool) or nproc_per_node < 1:
        raise ConfigError("run_options.nproc_per_node 必须是正整数")

    return RunOptions(
        base_output_dir=base_output_dir,
        sweep_dir_prefix=sweep_dir_prefix,
        keep_only_best_ckpt=bool(keep_only_best),
        nproc_per_node=int(nproc_per_node),
    )


def _validate_env_gpu_consistency(nproc_per_node: int) -> None:
    """校验 CUDA_VISIBLE_DEVICES 与 nproc_per_node 一致。

    - 未设置 CUDA_VISIBLE_DEVICES 时不做强校验（可能是单卡 dryrun 或自动化环境）。
    - 设置了但卡数与 nproc_per_node 不一致时报错。
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return
    count = len([x for x in visible.split(",") if x.strip()])
    if count != nproc_per_node:
        raise ConfigError(
            f"CUDA_VISIBLE_DEVICES 可见卡数={count} 与 run_options.nproc_per_node="
            f"{nproc_per_node} 不一致；请调整环境变量或配置"
        )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def load_sweep_config(config_path: str) -> SweepConfig:
    """读取并校验 YAML 配置，返回展开后的扫描计划。

    Args:
        config_path: 配置文件绝对/相对路径。

    Raises:
        FileNotFoundError: 文件不存在。
        ConfigError: 内容格式错误或字段缺失。
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"扫描配置文件不存在：{config_path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 语法错误：{e}") from e

    if not isinstance(cfg, dict):
        raise ConfigError("配置文件顶层必须是映射")

    data_paths = _validate_data_paths(_require_mapping(cfg.get("data_paths"), "data_paths"))
    pretrained_model_dir = _validate_pretrained(cfg.get("pretrained_model_dir", ""))
    format_script = _validate_script(cfg.get("format_script", ""), "format_script")
    eval_tool_sh = _validate_script(cfg.get("eval_tool_sh", ""), "eval_tool_sh")

    baseline = _validate_baseline(_require_mapping(cfg.get("baseline_hparams"), "baseline_hparams"))
    sweep_order = cfg.get("sweep_order")
    sweep_space = _require_mapping(cfg.get("sweep_space"), "sweep_space")
    dimensions = _validate_sweep(sweep_order, sweep_space, baseline)

    infer_args = _validate_infer_args(cfg.get("infer_args"))
    run_options = _validate_run_options(cfg.get("run_options"))
    _validate_env_gpu_consistency(run_options.nproc_per_node)

    return SweepConfig(
        data_paths=data_paths,
        pretrained_model_dir=pretrained_model_dir,
        format_script=format_script,
        eval_tool_sh=eval_tool_sh,
        baseline_hparams=baseline,
        dimensions=dimensions,
        infer_args=infer_args,
        run_options=run_options,
        raw_config_path=os.path.abspath(config_path),
    )


__all__ = [
    "ALLOWED_HPARAMS",
    "ALLOWED_INFER_ARGS",
    "REQUIRED_BASELINE_KEYS",
    "REQUIRED_DATA_KEYS",
    "PRETRAINED_REQUIRED_FILES",
    "ConfigError",
    "DimensionPlan",
    "RunOptions",
    "SweepConfig",
    "float_close",
    "load_sweep_config",
]
