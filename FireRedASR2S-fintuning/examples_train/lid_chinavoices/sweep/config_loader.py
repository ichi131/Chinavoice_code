# -*- coding: utf-8 -*-
"""LID 超参扫描配置加载与校验模块。

职责：
1. 读取 YAML 配置文件（缺失/格式错误立即抛异常，不启动任何训练）。
2. 校验 fixed_params、baseline、sweep_order、sweep_space、run_options 四段是否完整合法。
3. 展开成一份"扫描计划"（SweepPlan），供 sweep_main.py 直接消费。

设计原则：
- 与训练/推理脚本完全解耦，只依赖 PyYAML。
- 所有可扫描超参维护在 ALLOWED_HPARAMS 白名单中，若配置引入未知维度立即报错。
- 支持"候选值长度为 1"直接跳过扫描（记为 skipped=True）。
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

# 允许被扫描/覆盖的超参白名单：需与 finetune_lid_chinavoices.py 中 argparse 定义严格一致
# 说明：train_jsonl/val_jsonl/pretrained_model_dir/output_dir/init_ckpt 等属于路径类，
# 通过 fixed_params 或运行时逻辑传入，不放入白名单
ALLOWED_HPARAMS = {
    "lr": float,
    "encoder_lr": float,
    "batch_size": int,
    "dropout": float,
    "weight_decay": float,
    "label_smoothing": float,
    "seed": int,
    "grad_clip": float,
    "warmup_steps": int,
    "epochs": int,
    "patience": int,
    "min_delta": float,
    "freeze_encoder": int,
}

# 默认扫描顺序（当 sweep_order 缺省时使用）
DEFAULT_SWEEP_ORDER: List[str] = [
    "lr", "encoder_lr", "batch_size", "dropout",
    "weight_decay", "label_smoothing", "seed",
]

# baseline 中必须显式给出的字段（若缺失即报错，避免拼命令时漏参）
REQUIRED_BASELINE_KEYS: List[str] = [
    "lr", "encoder_lr", "batch_size", "dropout", "weight_decay",
    "label_smoothing", "seed", "grad_clip", "warmup_steps",
    "epochs", "patience", "min_delta", "freeze_encoder",
]

# fixed_params 中必须显式给出的字段
REQUIRED_FIXED_KEYS: List[str] = [
    "train_jsonl", "val_jsonl", "test_jsonl", "pretrained_model_dir",
]


class ConfigError(RuntimeError):
    """配置校验失败时抛出。"""


@dataclasses.dataclass
class DimensionPlan:
    """单个扫描维度的展开计划。"""
    index: int                # 从 1 开始的维度序号
    name: str                 # 维度名（例如 'lr'）
    candidates: List[Any]     # 候选值列表
    skipped: bool             # 是否直接固化（长度为 1 时为 True）


@dataclasses.dataclass
class RunOptions:
    """扫描主控运行选项。"""
    base_output_dir: str
    sweep_dir_prefix: str
    keep_only_best_ckpt: bool
    min_acc_warning_threshold: float


@dataclasses.dataclass
class SweepPlan:
    """完整的扫描计划。"""
    fixed_params: Dict[str, Any]
    baseline: Dict[str, Any]
    dimensions: List[DimensionPlan]
    run_options: RunOptions
    raw_config_path: str


def _coerce_scalar(name: str, value: Any) -> Any:
    """按白名单类型定义把 YAML 里的标量强转为 Python 原生类型。

    - bool 不属于允许类型，会被 int(True) 意外接受，需在此显式拒绝。
    - 对 float 类型允许接受 int，反之亦然（int 字段收到 float 会因非整数被拒绝）。
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


def _validate_fixed(fixed_params: Dict[str, Any]) -> Dict[str, Any]:
    for key in REQUIRED_FIXED_KEYS:
        if key not in fixed_params:
            raise ConfigError(f"fixed_params 缺少必填字段：`{key}`")
        if not isinstance(fixed_params[key], str) or not fixed_params[key].strip():
            raise ConfigError(f"fixed_params.{key} 必须是非空字符串路径")
    return dict(fixed_params)


def _validate_baseline(baseline_raw: Dict[str, Any]) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {}
    for key in REQUIRED_BASELINE_KEYS:
        if key not in baseline_raw:
            raise ConfigError(f"baseline 缺少必填字段：`{key}`")
        baseline[key] = _coerce_scalar(key, baseline_raw[key])
    # 允许 baseline 出现白名单内其他可选字段（虽然当前 REQUIRED == ALLOWED，但保留扩展空间）
    for key in baseline_raw:
        if key in baseline:
            continue
        if key not in ALLOWED_HPARAMS:
            raise ConfigError(f"baseline 出现未知字段：`{key}`（不在白名单）")
        baseline[key] = _coerce_scalar(key, baseline_raw[key])
    return baseline


def _validate_sweep(sweep_order: Sequence[Any], sweep_space: Dict[str, Any]) -> List[DimensionPlan]:
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
            raise ConfigError(f"sweep_order[`{name}`] 不在允许白名单中；合法值：{sorted(ALLOWED_HPARAMS)}")
        if name not in sweep_space:
            raise ConfigError(f"sweep_space 缺少维度 `{name}` 的候选值列表")
        raw_candidates = sweep_space[name]
        if not isinstance(raw_candidates, (list, tuple)) or len(raw_candidates) == 0:
            raise ConfigError(f"sweep_space.{name} 必须是非空列表")
        candidates = [_coerce_scalar(name, v) for v in raw_candidates]
        # 保序去重（同一维度多次相同候选属于配置冗余，直接判错更利于用户发现）
        if len(set(candidates)) != len(candidates):
            raise ConfigError(f"sweep_space.{name} 存在重复候选值：{candidates}")
        dimensions.append(DimensionPlan(
            index=idx,
            name=name,
            candidates=candidates,
            skipped=(len(candidates) == 1),
        ))
    return dimensions


def _validate_run_options(run_options_raw: Any) -> RunOptions:
    run_options_raw = run_options_raw or {}
    if not isinstance(run_options_raw, dict):
        raise ConfigError("run_options 必须是映射")

    base_output_dir = run_options_raw.get("base_output_dir", "./exp")
    sweep_dir_prefix = run_options_raw.get("sweep_dir_prefix", "lid_sweep")
    keep_only_best = run_options_raw.get("keep_only_best_ckpt", True)
    min_acc = run_options_raw.get("min_acc_warning_threshold", 1.0 / 16.0)

    if not isinstance(base_output_dir, str) or not base_output_dir.strip():
        raise ConfigError("run_options.base_output_dir 必须是非空字符串")
    if not isinstance(sweep_dir_prefix, str) or not sweep_dir_prefix.strip():
        raise ConfigError("run_options.sweep_dir_prefix 必须是非空字符串")
    if not isinstance(keep_only_best, bool):
        raise ConfigError("run_options.keep_only_best_ckpt 必须是 bool")
    if not isinstance(min_acc, (int, float)):
        raise ConfigError("run_options.min_acc_warning_threshold 必须是数字")

    return RunOptions(
        base_output_dir=base_output_dir,
        sweep_dir_prefix=sweep_dir_prefix,
        keep_only_best_ckpt=bool(keep_only_best),
        min_acc_warning_threshold=float(min_acc),
    )


def load_sweep_plan(config_path: str) -> SweepPlan:
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

    fixed_params = _validate_fixed(_require_mapping(cfg.get("fixed_params"), "fixed_params"))
    baseline = _validate_baseline(_require_mapping(cfg.get("baseline"), "baseline"))
    sweep_order = cfg.get("sweep_order") or DEFAULT_SWEEP_ORDER
    sweep_space = _require_mapping(cfg.get("sweep_space"), "sweep_space")
    dimensions = _validate_sweep(sweep_order, sweep_space)
    run_options = _validate_run_options(cfg.get("run_options"))

    return SweepPlan(
        fixed_params=fixed_params,
        baseline=baseline,
        dimensions=dimensions,
        run_options=run_options,
        raw_config_path=os.path.abspath(config_path),
    )


__all__ = [
    "ALLOWED_HPARAMS",
    "DEFAULT_SWEEP_ORDER",
    "ConfigError",
    "DimensionPlan",
    "RunOptions",
    "SweepPlan",
    "load_sweep_plan",
]
