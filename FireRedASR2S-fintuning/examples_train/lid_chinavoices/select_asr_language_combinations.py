#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_PRED_JSONL = (
    "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/"
    "user_ichiwang/workspace/FireRedASR2S-fintuning/exp/"
    "lid_chinavoices_data_speaker_ft_encoder/data_test_pred.jsonl"
)

DEFAULT_REF_JSONL = (
    "/mnt/wfs/mmhuizhouwfssz/project_luban_infra/x_speech/"
    "user_ichiwang/data/challenge_data_speaker/data_test.jsonl"
)

ACCENT_CN = {
    "anhui": "安徽",
    "cantonese": "粤语",
    "changsha": "长沙",
    "chaoshan": "潮汕",
    "dongbei": "东北",
    "henan": "河南",
    "kejia": "客家",
    "minnan": "闽南",
    "nanchang": "南昌",
    "nanjing": "南京",
    "shan1xi": "山西",
    "shan3xi": "陕西",
    "shandong": "山东",
    "sichuan": "四川",
    "wuyu": "吴语",
    "wuhan": "武汉",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "根据 LID Top-2/Top-3 预测，使用带成本的贪心最大覆盖算法，"
            "选择值得训练的多语种 ASR 模型组合"
        )
    )
    parser.add_argument(
        "--pred_jsonl",
        default=DEFAULT_PRED_JSONL,
        help="LID 预测 JSONL",
    )
    parser.add_argument(
        "--ref_jsonl",
        default=DEFAULT_REF_JSONL,
        help="包含真实 accent 标签的 JSONL",
    )
    parser.add_argument(
        "--objective",
        choices=["gold", "recoverable", "traffic"],
        default="gold",
        help=(
            "组合选择目标："
            "gold=最大化真实语种被组合包含的样本数；"
            "recoverable=最大化 Top1 错但 TopK 包含真实语种的样本数；"
            "traffic=最大化组合的实际调用样本数"
        ),
    )
    parser.add_argument(
        "--allowed_k",
        choices=["2", "3", "both"],
        default="both",
        help="只选择 Top-2、只选择 Top-3，或者两者一起选择",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=20.0,
        help="总训练成本预算，默认 20",
    )
    parser.add_argument(
        "--top2_cost",
        type=float,
        default=1.0,
        help="训练一个 Top-2 组合模型的相对成本，默认 1.0",
    )
    parser.add_argument(
        "--top3_cost",
        type=float,
        default=1.5,
        help="训练一个 Top-3 组合模型的相对成本，默认 1.5",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=0,
        help="最多选择多少个模型；0 表示只受 budget 限制",
    )
    parser.add_argument(
        "--min_support",
        type=int,
        default=10,
        help="组合至少出现多少次才作为候选，默认 10",
    )
    parser.add_argument(
        "--top_n_candidates",
        type=int,
        default=30,
        help="展示价值最高的前 N 个候选组合，默认 30",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="可选：保存完整分析结果为 JSON",
    )
    return parser.parse_args()


def load_references(path):
    references = {}

    with open(path, encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)
            key = obj.get("key")
            accent = obj.get("accent")

            if not key or not accent:
                raise ValueError(
                    f"{path}:{line_number}: 缺少 key 或 accent"
                )

            if key in references:
                raise ValueError(
                    f"{path}:{line_number}: key 重复: {key}"
                )

            references[key] = accent

    return references


def load_predictions(path):
    predictions = {}

    with open(path, encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)
            key = obj.get("key")
            top5 = obj.get("top5", [])

            if not key:
                raise ValueError(f"{path}:{line_number}: 缺少 key")

            if key in predictions:
                raise ValueError(
                    f"{path}:{line_number}: key 重复: {key}"
                )

            if not isinstance(top5, list):
                raise ValueError(
                    f"{path}:{line_number}: top5 必须是列表"
                )

            ranked_accents = []
            ranked_probs = []

            for item in top5:
                if not isinstance(item, dict):
                    continue

                accent = item.get("accent")
                if not accent:
                    continue

                ranked_accents.append(accent)
                ranked_probs.append(float(item.get("prob", 0.0)))

            predictions[key] = {
                "accents": ranked_accents,
                "probs": ranked_probs,
            }

    return predictions


def get_allowed_k(value):
    if value == "2":
        return [2]
    if value == "3":
        return [3]
    return [2, 3]


def combination_text(combination):
    names = []
    for accent in combination:
        accent_cn = ACCENT_CN.get(accent, "")
        names.append(
            f"{accent}({accent_cn})" if accent_cn else accent
        )
    return " + ".join(names)


def build_candidates(
    references,
    predictions,
    allowed_k,
    top2_cost,
    top3_cost,
):
    candidates = {}
    top1_correct_ids = set()
    top2_correct_ids = set()
    top3_correct_ids = set()

    for key, gold_accent in references.items():
        prediction = predictions.get(key, {})
        ranked_accents = prediction.get("accents", [])

        if ranked_accents and ranked_accents[0] == gold_accent:
            top1_correct_ids.add(key)

        if gold_accent in ranked_accents[:2]:
            top2_correct_ids.add(key)

        if gold_accent in ranked_accents[:3]:
            top3_correct_ids.add(key)

        for k in allowed_k:
            if len(ranked_accents) < k:
                continue

            # ASR 模型的语种组成不考虑 LID 排名顺序。
            combination = tuple(sorted(ranked_accents[:k]))
            candidate_id = (k, combination)

            if candidate_id not in candidates:
                candidates[candidate_id] = {
                    "id": candidate_id,
                    "k": k,
                    "combination": combination,
                    "cost": top2_cost if k == 2 else top3_cost,
                    "traffic_ids": set(),
                    "gold_ids": set(),
                    "recoverable_ids": set(),
                }

            candidate = candidates[candidate_id]
            candidate["traffic_ids"].add(key)

            if gold_accent in combination:
                candidate["gold_ids"].add(key)

                if key not in top1_correct_ids:
                    candidate["recoverable_ids"].add(key)

    summary_sets = {
        "top1_correct_ids": top1_correct_ids,
        "top2_correct_ids": top2_correct_ids,
        "top3_correct_ids": top3_correct_ids,
    }

    return candidates, summary_sets


def objective_ids(candidate, objective):
    if objective == "traffic":
        return candidate["traffic_ids"]

    if objective == "recoverable":
        return candidate["recoverable_ids"]

    return candidate["gold_ids"]


def candidate_score(candidate, objective):
    target_count = len(objective_ids(candidate, objective))
    cost = candidate["cost"]

    return target_count / cost if cost > 0 else 0.0


def greedy_select(
    candidates,
    objective,
    budget,
    max_models,
):
    selected = []
    selected_ids = set()
    covered_objective_ids = set()
    spent = 0.0

    while True:
        if max_models > 0 and len(selected) >= max_models:
            break

        best_candidate = None
        best_marginal_ids = None
        best_sort_key = None

        for candidate_id, candidate in candidates.items():
            if candidate_id in selected_ids:
                continue

            new_spent = spent + candidate["cost"]
            if new_spent > budget + 1e-9:
                continue

            target_ids = objective_ids(candidate, objective)
            marginal_ids = target_ids - covered_objective_ids

            if not marginal_ids:
                continue

            marginal_count = len(marginal_ids)
            value_per_cost = marginal_count / candidate["cost"]

            sort_key = (
                value_per_cost,
                marginal_count,
                len(candidate["gold_ids"]),
                len(candidate["traffic_ids"]),
                -candidate["cost"],
            )

            if best_sort_key is None or sort_key > best_sort_key:
                best_sort_key = sort_key
                best_candidate = candidate
                best_marginal_ids = marginal_ids

        if best_candidate is None:
            break

        spent += best_candidate["cost"]
        selected_ids.add(best_candidate["id"])
        covered_objective_ids.update(best_marginal_ids)

        selected.append(
            {
                "candidate": best_candidate,
                "marginal_objective_count": len(
                    best_marginal_ids
                ),
                "cumulative_objective_count": len(
                    covered_objective_ids
                ),
                "spent": spent,
            }
        )

    return selected


def calculate_dataset_summary(
    references,
    predictions,
    summary_sets,
):
    total = len(references)
    missing_predictions = len(set(references) - set(predictions))
    extra_predictions = len(set(predictions) - set(references))

    top1_correct = len(summary_sets["top1_correct_ids"])
    top2_correct = len(summary_sets["top2_correct_ids"])
    top3_correct = len(summary_sets["top3_correct_ids"])

    return {
        "total": total,
        "num_predictions": len(predictions),
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "top2_correct": top2_correct,
        "top2_accuracy": top2_correct / total if total else 0.0,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_correct / total if total else 0.0,
        "top2_recoverable": top2_correct - top1_correct,
        "top3_recoverable": top3_correct - top1_correct,
    }


def print_dataset_summary(summary, num_all_candidates, num_candidates):
    print()
    print("=" * 100)
    print("数据集与理论上限")
    print("=" * 100)

    print(f"参考样本数:                {summary['total']}")
    print(f"预测样本数:                {summary['num_predictions']}")
    print(f"缺少预测数:                {summary['missing_predictions']}")
    print(f"多余预测数:                {summary['extra_predictions']}")
    print()

    print(
        f"Top-1 准确率:              "
        f"{summary['top1_accuracy']:.2%} "
        f"({summary['top1_correct']}/{summary['total']})"
    )
    print(
        f"Top-2 包含真实语种比例:     "
        f"{summary['top2_accuracy']:.2%} "
        f"({summary['top2_correct']}/{summary['total']})"
    )
    print(
        f"Top-3 包含真实语种比例:     "
        f"{summary['top3_accuracy']:.2%} "
        f"({summary['top3_correct']}/{summary['total']})"
    )
    print(
        f"Top-2 相对 Top-1 可挽回样本: "
        f"{summary['top2_recoverable']}"
    )
    print(
        f"Top-3 相对 Top-1 可挽回样本: "
        f"{summary['top3_recoverable']}"
    )
    print()

    print(f"16 个语种理论 Top-2 组合数: {math.comb(16, 2)}")
    print(f"16 个语种理论 Top-3 组合数: {math.comb(16, 3)}")
    print(f"测试集中实际出现候选数:      {num_all_candidates}")
    print(f"经过 min_support 后候选数:   {num_candidates}")


def sorted_candidates(candidates, objective):
    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate_score(candidate, objective),
            -len(objective_ids(candidate, objective)),
            -len(candidate["traffic_ids"]),
            candidate["k"],
            candidate["combination"],
        ),
    )


def print_top_candidates(candidates, objective, top_n):
    print()
    print("=" * 130)
    print(f"候选组合价值排名，优化目标：{objective}")
    print("=" * 130)

    print(
        f"{'排名':>6}  "
        f"{'K':>3}  "
        f"{'语种组合':<62}"
        f"{'调用数':>10}"
        f"{'真实语种命中':>14}"
        f"{'可挽回':>10}"
        f"{'成本':>8}"
        f"{'目标/成本':>12}"
    )
    print("-" * 130)

    for rank, candidate in enumerate(
        sorted_candidates(candidates, objective)[:top_n],
        start=1,
    ):
        print(
            f"{rank:>6d}  "
            f"{candidate['k']:>3d}  "
            f"{combination_text(candidate['combination']):<62}"
            f"{len(candidate['traffic_ids']):>10d}"
            f"{len(candidate['gold_ids']):>14d}"
            f"{len(candidate['recoverable_ids']):>10d}"
            f"{candidate['cost']:>8.2f}"
            f"{candidate_score(candidate, objective):>12.2f}"
        )


def analyze_selected(
    selected,
    total_samples,
    top1_correct_ids,
):
    traffic_union = set()
    gold_union = set()
    recoverable_union = set()
    rows = []

    for index, selected_item in enumerate(selected, start=1):
        candidate = selected_item["candidate"]

        new_traffic = candidate["traffic_ids"] - traffic_union
        new_gold = candidate["gold_ids"] - gold_union
        new_recoverable = (
            candidate["recoverable_ids"] - recoverable_union
        )

        traffic_union.update(candidate["traffic_ids"])
        gold_union.update(candidate["gold_ids"])
        recoverable_union.update(candidate["recoverable_ids"])

        proxy_correct_ids = top1_correct_ids | recoverable_union

        rows.append(
            {
                "index": index,
                "candidate": candidate,
                "spent": selected_item["spent"],
                "marginal_objective_count": selected_item[
                    "marginal_objective_count"
                ],
                "new_traffic": len(new_traffic),
                "new_gold": len(new_gold),
                "new_recoverable": len(new_recoverable),
                "cumulative_traffic": len(traffic_union),
                "cumulative_gold": len(gold_union),
                "cumulative_recoverable": len(
                    recoverable_union
                ),
                "proxy_accuracy": (
                    len(proxy_correct_ids) / total_samples
                    if total_samples
                    else 0.0
                ),
            }
        )

    return rows


def print_selected(rows, objective):
    print()
    print("=" * 150)
    print(f"推荐训练的模型组合，优化目标：{objective}")
    print("=" * 150)

    if not rows:
        print("在当前 budget/min_support 条件下没有可选择的组合。")
        return

    print(
        f"{'序号':>5} "
        f"{'K':>3} "
        f"{'语种组合':<58}"
        f"{'累计成本':>10}"
        f"{'新增目标':>10}"
        f"{'新增调用':>10}"
        f"{'新增真实命中':>14}"
        f"{'新增可挽回':>12}"
        f"{'累计真实命中':>14}"
        f"{'Top1+挽回上限':>16}"
    )
    print("-" * 150)

    for row in rows:
        candidate = row["candidate"]

        print(
            f"{row['index']:>5d} "
            f"{candidate['k']:>3d} "
            f"{combination_text(candidate['combination']):<58}"
            f"{row['spent']:>10.2f}"
            f"{row['marginal_objective_count']:>10d}"
            f"{row['new_traffic']:>10d}"
            f"{row['new_gold']:>14d}"
            f"{row['new_recoverable']:>12d}"
            f"{row['cumulative_gold']:>14d}"
            f"{row['proxy_accuracy']:>15.2%}"
        )

    final = rows[-1]
    print("-" * 150)
    print(f"最终选择模型数:          {len(rows)}")
    print(f"最终训练成本:            {final['spent']:.2f}")
    print(f"覆盖调用样本数:          {final['cumulative_traffic']}")
    print(f"真实语种被组合包含数:    {final['cumulative_gold']}")
    print(f"相对 Top-1 可挽回样本数: {final['cumulative_recoverable']}")
    print(
        f"Top1 + 已选组合理论上限: "
        f"{final['proxy_accuracy']:.2%}"
    )


def candidate_to_json(candidate, objective):
    traffic_count = len(candidate["traffic_ids"])
    gold_count = len(candidate["gold_ids"])
    recoverable_count = len(candidate["recoverable_ids"])

    return {
        "k": candidate["k"],
        "combination": list(candidate["combination"]),
        "combination_cn": [
            ACCENT_CN.get(accent, "")
            for accent in candidate["combination"]
        ],
        "cost": candidate["cost"],
        "traffic_count": traffic_count,
        "gold_count": gold_count,
        "gold_rate": (
            gold_count / traffic_count if traffic_count else 0.0
        ),
        "recoverable_count": recoverable_count,
        "objective_count": len(
            objective_ids(candidate, objective)
        ),
        "objective_per_cost": candidate_score(
            candidate, objective
        ),
    }


def save_report(
    output_path,
    args,
    summary,
    all_candidates,
    candidates,
    selected_rows,
):
    report = {
        "config": {
            "pred_jsonl": args.pred_jsonl,
            "ref_jsonl": args.ref_jsonl,
            "objective": args.objective,
            "allowed_k": args.allowed_k,
            "budget": args.budget,
            "top2_cost": args.top2_cost,
            "top3_cost": args.top3_cost,
            "max_models": args.max_models,
            "min_support": args.min_support,
        },
        "summary": summary,
        "num_all_candidates": len(all_candidates),
        "num_filtered_candidates": len(candidates),
        "candidate_ranking": [
            candidate_to_json(candidate, args.objective)
            for candidate in sorted_candidates(
                candidates, args.objective
            )
        ],
        "selected_models": [],
    }

    for row in selected_rows:
        item = candidate_to_json(
            row["candidate"], args.objective
        )
        item.update(
            {
                "selection_index": row["index"],
                "cumulative_cost": row["spent"],
                "marginal_objective_count": row[
                    "marginal_objective_count"
                ],
                "new_traffic": row["new_traffic"],
                "new_gold": row["new_gold"],
                "new_recoverable": row[
                    "new_recoverable"
                ],
                "cumulative_traffic": row[
                    "cumulative_traffic"
                ],
                "cumulative_gold": row[
                    "cumulative_gold"
                ],
                "cumulative_recoverable": row[
                    "cumulative_recoverable"
                ],
                "proxy_accuracy": row["proxy_accuracy"],
            }
        )
        report["selected_models"].append(item)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)

    print()
    print(f"完整分析结果已保存到: {output_path}")


def main():
    args = parse_args()

    if args.budget <= 0:
        raise ValueError("--budget 必须大于 0")

    if args.top2_cost <= 0 or args.top3_cost <= 0:
        raise ValueError("--top2_cost 和 --top3_cost 必须大于 0")

    if args.min_support <= 0:
        raise ValueError("--min_support 必须大于 0")

    pred_path = Path(args.pred_jsonl)
    ref_path = Path(args.ref_jsonl)

    if not pred_path.is_file():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")

    if not ref_path.is_file():
        raise FileNotFoundError(f"参考文件不存在: {ref_path}")

    references = load_references(ref_path)
    predictions = load_predictions(pred_path)
    allowed_k = get_allowed_k(args.allowed_k)

    all_candidates, summary_sets = build_candidates(
        references=references,
        predictions=predictions,
        allowed_k=allowed_k,
        top2_cost=args.top2_cost,
        top3_cost=args.top3_cost,
    )

    candidates = {
        candidate_id: candidate
        for candidate_id, candidate in all_candidates.items()
        if len(candidate["traffic_ids"]) >= args.min_support
    }

    summary = calculate_dataset_summary(
        references=references,
        predictions=predictions,
        summary_sets=summary_sets,
    )

    selected = greedy_select(
        candidates=candidates,
        objective=args.objective,
        budget=args.budget,
        max_models=args.max_models,
    )

    selected_rows = analyze_selected(
        selected=selected,
        total_samples=len(references),
        top1_correct_ids=summary_sets["top1_correct_ids"],
    )

    print_dataset_summary(
        summary=summary,
        num_all_candidates=len(all_candidates),
        num_candidates=len(candidates),
    )

    print_top_candidates(
        candidates=candidates,
        objective=args.objective,
        top_n=args.top_n_candidates,
    )

    print_selected(
        rows=selected_rows,
        objective=args.objective,
    )

    if args.output_json:
        save_report(
            output_path=args.output_json,
            args=args,
            summary=summary,
            all_candidates=all_candidates,
            candidates=candidates,
            selected_rows=selected_rows,
        )


if __name__ == "__main__":
    main()
