import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALID_TYPES = {"Chemical", "Disease"}
STRATEGIES = ["T0", "T1", "T2", "T3", "T4"]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 无效: {e}")
    return records


def entity_key(item):
    if not isinstance(item, dict):
        return None

    name = item.get("entity")
    entity_type = item.get("type")

    if not isinstance(name, str) or not name.strip():
        return None
    if entity_type not in VALID_TYPES:
        return None

    return (name.strip(), entity_type)


def entity_list(items):
    result = {}
    for item in items or []:
        key = entity_key(item)
        if key is not None:
            result[key] = {
                "entity": key[0],
                "type": key[1],
            }
    return result


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def analyze_strategy(strategy, records, output_dir):
    fp_records = []
    fn_records = []
    format_errors = []
    api_errors = []

    type_stats = {
        t: {"TP": 0, "FP": 0, "FN": 0}
        for t in sorted(VALID_TYPES)
    }

    total_tp = total_fp = total_fn = 0
    valid_count = response_count = 0

    # 用于跨策略比较
    sample_entities = {}

    for record in records:
        status = record.get("status")

        if status == "api_error":
            api_errors.append(record)
            continue

        if status != "response":
            continue

        response_count += 1

        if record.get("format_valid") is True:
            valid_count += 1
        else:
            format_errors.append(record)

        gold = entity_list(record.get("gold_entities"))
        pred = entity_list(record.get("predicted_entities"))

        gold_keys = set(gold)
        pred_keys = set(pred)

        tp = gold_keys & pred_keys
        fp = pred_keys - gold_keys
        fn = gold_keys - pred_keys

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for key in tp:
            type_stats[key[1]]["TP"] += 1
        for key in fp:
            type_stats[key[1]]["FP"] += 1
        for key in fn:
            type_stats[key[1]]["FN"] += 1

        common = {
            "sample_index": record.get("sample_index"),
            "pmid": record.get("pmid"),
            "strategy": strategy,
            "text": record.get("text", ""),
        }

        if fp:
            fp_records.append({
                **common,
                "false_positives": [pred[k] for k in sorted(fp)],
            })

        if fn:
            fn_records.append({
                **common,
                "false_negatives": [gold[k] for k in sorted(fn)],
            })

        sample_id = record.get("sample_index")
        sample_entities[sample_id] = {
            "pmid": record.get("pmid"),
            "gold": gold_keys,
            "pred": pred_keys,
            "text": record.get("text", ""),
            "format_valid": record.get("format_valid"),
            "format_error": record.get("format_error"),
        }

    strategy_dir = output_dir / strategy
    write_jsonl(strategy_dir / "false_positives.jsonl", fp_records)
    write_jsonl(strategy_dir / "false_negatives.jsonl", fn_records)
    write_jsonl(strategy_dir / "format_errors.jsonl", format_errors)
    write_jsonl(strategy_dir / "api_errors.jsonl", api_errors)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    metrics = {
        "strategy": strategy,
        "samples_total": len(records),
        "samples_with_response": response_count,
        "format_valid_samples": valid_count,
        "format_valid_rate": valid_count / response_count if response_count else 0,
        "api_errors": len(api_errors),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "type_statistics": type_stats,
        "false_positive_samples": len(fp_records),
        "false_negative_samples": len(fn_records),
        "format_error_samples": len(format_errors),
    }

    write_json(strategy_dir / "summary.json", metrics)

    return metrics, sample_entities


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2 error analysis"
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/experiment2_smoke20/predictions",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment2_smoke20/error_analysis",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=STRATEGIES,
    )
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    all_samples = {}

    for strategy in args.strategies:
        input_file = input_dir / f"{strategy}.jsonl"

        if not input_file.exists():
            raise FileNotFoundError(f"找不到预测文件: {input_file}")

        records = read_jsonl(input_file)
        metrics, samples = analyze_strategy(
            strategy, records, output_dir
        )

        all_metrics[strategy] = metrics
        all_samples[strategy] = samples

        print(f"\n===== {strategy} 错误分析 =====")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 汇总五种策略的整体表现
    comparison = []
    for strategy, m in all_metrics.items():
        comparison.append({
            "strategy": strategy,
            "TP": m["TP"],
            "FP": m["FP"],
            "FN": m["FN"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "format_valid_rate": m["format_valid_rate"],
        })

    write_json(output_dir / "strategy_comparison.json", comparison)

    # 比较相同样本、相同实体在五种策略下的识别情况
    cross_records = []

    sample_indices = sorted({
        idx
        for samples in all_samples.values()
        for idx in samples
        if idx is not None
    })

    for sample_index in sample_indices:
        available = {
            s: all_samples[s][sample_index]
            for s in args.strategies
            if sample_index in all_samples[s]
        }

        if not available:
            continue

        gold_union = set()
        for item in available.values():
            gold_union.update(item["gold"])

        for key in sorted(gold_union):
            correct_strategies = [
                s for s, item in available.items()
                if key in item["pred"]
            ]

            cross_records.append({
                "sample_index": sample_index,
                "pmid": next(iter(available.values()))["pmid"],
                "entity": key[0],
                "type": key[1],
                "gold_entity": True,
                "correctly_predicted_by": correct_strategies,
                "missed_by": [
                    s for s in available
                    if key not in available[s]["pred"]
                ],
            })

    write_jsonl(
        output_dir / "cross_strategy_gold_entities.jsonl",
        cross_records,
    )

    # 汇总五种策略共同漏检的实体
    common_fn = [
        item for item in cross_records
        if len(item["missed_by"]) == len(args.strategies)
    ]

    write_jsonl(
        output_dir / "common_false_negatives.jsonl",
        common_fn,
    )

    print("\n===== 五种策略汇总 =====")
    print(
        f"{'策略':<8}{'TP':>7}{'FP':>7}{'FN':>7}"
        f"{'Precision':>12}{'Recall':>10}{'F1':>10}"
    )

    for m in comparison:
        print(
            f"{m['strategy']:<8}"
            f"{m['TP']:>7}{m['FP']:>7}{m['FN']:>7}"
            f"{m['precision']:>12.4f}"
            f"{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}"
        )

    print(f"\n分析完成，结果目录：{output_dir}")
    print(f"五种策略共同漏检实体数：{len(common_fn)}")


if __name__ == "__main__":
    main()
