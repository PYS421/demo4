import argparse
import json
import re
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_client import call_llm
from prompts.templates import build_messages
TEST_FILE = ROOT / "data/processed/bc5cdr_test_first500.jsonl"
EXAMPLE_FILE = ROOT / "data/processed/bc5cdr_train_examples.jsonl"
OUT_DIR = ROOT / "outputs/predictions"
REPORT_DIR = ROOT / "outputs/reports"

STRATEGIES = {
    "zero_shot": 0,
    "few_shot_1": 1,
    "few_shot_2": 2,
    "few_shot_3": 3,
}
VALID_TYPES = {"Chemical", "Disease"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_entities(items):
    """保留实体文本原样，按 (entity, type) 去重。"""
    result = set()
    if not isinstance(items, list):
        return result

    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("entity")
        typ = item.get("type")
        if (
            isinstance(name, str)
            and name.strip()
            and typ in VALID_TYPES
        ):
            result.add((name.strip(), typ))
    return result


def parse_response(content, original_text):
    """解析 JSON；标准解析失败时尝试修复，并严格校验输出。"""
    raw = (content or "").strip()

    # 兼容模型偶尔返回 Markdown 代码围栏的情况
    raw = re.sub(r"^```(?:json)?\\s*", "", raw, flags=re.I)
    raw = re.sub(r"\\s*```$", "", raw)

    # 先尝试标准 JSON 解析，仅在失败时调用 json_repair
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            from json_repair import repair_json
            obj = repair_json(raw, return_objects=True)
        except Exception:
            return None, False, "invalid_json"

        # 修复结果必须是字典，且包含预期字段
        if not isinstance(obj, dict):
            return obj, False, "repair_not_object"

    if not isinstance(obj, dict):
        return obj, False, "not_object"

    if set(obj.keys()) != {"entities"}:
        return obj, False, "wrong_top_level_keys"

    entities = obj.get("entities")
    if not isinstance(entities, list):
        return obj, False, "entities_not_list"

    for item in entities:
        if not isinstance(item, dict):
            return obj, False, "entity_not_object"
        if set(item.keys()) != {"entity", "type"}:
            return obj, False, "wrong_entity_keys"
        if not isinstance(item.get("entity"), str) or not item["entity"].strip():
            return obj, False, "empty_entity"
        if item.get("type") not in VALID_TYPES:
            return obj, False, "invalid_entity_type"
        if item["entity"] not in original_text:
            return obj, False, "entity_not_in_text"

    # 模型不再生成原文，由程序使用输入原文补充
    normalized_obj = {
        "text": original_text,
        "entities": entities,
    }

    return normalized_obj, True, None

def calculate_metrics(records):
    tp = fp = fn = 0
    evaluated = 0
    valid_formats = 0

    for record in records:
        if record.get("status") != "response":
            continue

        evaluated += 1
        gold = {
            (e["entity"], e["type"])
            for e in record["gold_entities"]
        }

        if record.get("format_valid"):
            valid_formats += 1
            pred = {
                (e["entity"], e["type"])
                for e in record["predicted_entities"]
            }
        else:
            # 模型有响应但格式无效：按空预测计入端到端实体指标
            pred = set()

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "samples_total": len(records),
        "samples_with_response": evaluated,
        "api_errors": sum(r.get("status") == "api_error" for r in records),
        "format_valid_samples": valid_formats,
        "format_valid_rate": valid_formats / evaluated if evaluated else 0.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=500,
        help="测试样本数量；用 --limit 1 先做单样本验证"
    )
    parser.add_argument(
        "--strategies", nargs="+",
        choices=list(STRATEGIES),
        default=list(STRATEGIES),
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="每次请求之间等待的秒数，默认 1 秒"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUT_DIR),
        help="预测结果输出目录，默认 outputs/predictions"
    )
    parser.add_argument(
        "--report-dir", type=str, default=str(REPORT_DIR),
        help="评估报告输出目录，默认 outputs/reports"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(TEST_FILE)[:args.limit]
    examples = read_jsonl(EXAMPLE_FILE)

    if not samples:
        raise RuntimeError("测试集为空")
    if len(examples) < 3:
        raise RuntimeError("训练示例不足 3 条")

    print(f"测试样本数：{len(samples)}")
    print(f"训练示例数：{len(examples)}")
    print(f"本次策略：{', '.join(args.strategies)}")
    print(f"预计最多请求：{len(samples) * len(args.strategies)} 次")
    print("评估规则：严格实体文本 + 类型匹配")
    print("断点续跑：已保存的模型响应会跳过")

    for strategy in args.strategies:
        output_file = output_dir / f"{strategy}.jsonl"
        existing = []

        if output_file.exists():
            existing = read_jsonl(output_file)

        # 按样本序号恢复已完成的模型响应
        done = {
            r["sample_index"]
            for r in existing
            if r.get("status") == "response"
        }
        records_by_index = {
            r["sample_index"]: r for r in existing
        }

        print(f"\n===== 开始 {strategy} =====")
        with output_file.open("a", encoding="utf-8") as fout:
            for idx, sample in enumerate(samples):
                if idx in done:
                    continue

                gold = normalize_entities(sample.get("entities", []))
                example_count = STRATEGIES[strategy]

                try:
                    messages = build_messages(
                        text=sample["text"],
                        examples=examples,
                        strategy=strategy,
                    )
                    result = call_llm(messages)
                    parsed, valid, error = parse_response(
                        result["content"], sample["text"]
                    )
                    pred = (
                        normalize_entities(parsed["entities"])
                        if valid else set()
                    )

                    record = {
                        "sample_index": idx,
                        "pmid": sample.get("pmid"),
                        "strategy": strategy,
                        "shots": example_count,
                        "status": "response",
                        "requested_model": result.get("requested_model"),
                        "actual_model": result.get("actual_model"),
                        "text": sample["text"],
                        "gold_entities": [
                            {"entity": n, "type": t}
                            for n, t in sorted(gold)
                        ],
                        "predicted_entities": [
                            {"entity": n, "type": t}
                            for n, t in sorted(pred)
                        ],
                        "raw_response": result.get("content", ""),
                        "format_valid": valid,
                        "format_error": error,
                        "usage": result.get("usage"),
                    }

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    records_by_index[idx] = record

                    print(
                        f"[{strategy}] {idx + 1}/{len(samples)} "
                        f"PMID={sample.get('pmid')} "
                        f"format_valid={valid}"
                    )

                except Exception as exc:
                    # 记录错误，但不把它当成有效模型预测；
                    # 下次运行会自动重试这个样本。
                    error_record = {
                        "sample_index": idx,
                        "pmid": sample.get("pmid"),
                        "strategy": strategy,
                        "status": "api_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    fout.write(
                        json.dumps(error_record, ensure_ascii=False) + "\n"
                    )
                    fout.flush()
                    records_by_index[idx] = error_record
                    print(
                        f"[{strategy}] {idx + 1}/{len(samples)} 请求失败："
                        f"{type(exc).__name__}: {exc}"
                    )

                if args.delay > 0:
                    time.sleep(args.delay)

        # 同一样本如果曾失败、后续重试成功，以最新记录为准
        final_records = [
            records_by_index[i]
            for i in sorted(records_by_index)
        ]
        metrics = calculate_metrics(final_records)
        metrics.update({
            "strategy": strategy,
            "shots": STRATEGIES[strategy],
            "model": final_records[-1].get("actual_model")
                if final_records else None,
        })

        report_path = report_dir / f"{strategy}_metrics.json"
        report_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("当前指标：")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n本次指定策略运行结束。")
    print(f"预测文件目录：{OUT_DIR}")
    print(f"指标文件目录：{REPORT_DIR}")


if __name__ == "__main__":
    main()
