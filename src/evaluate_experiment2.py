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
from prompts.experiment2.templates import build_messages, STRATEGIES

TEST_FILE = ROOT / "data/processed/bc5cdr_test_first500.jsonl"
EXAMPLE_FILE = ROOT / "data/processed/bc5cdr_train_examples.jsonl"

VALID_TYPES = {"Chemical", "Disease"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_entities(items):
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
    raw = (content or "").strip()

    # 兼容 Markdown JSON 代码围栏
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            from json_repair import repair_json
            obj = repair_json(raw, return_objects=True)
        except Exception:
            return None, False, "invalid_json"

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

        name = item.get("entity")
        typ = item.get("type")

        if not isinstance(name, str) or not name.strip():
            return obj, False, "empty_entity"

        if typ not in VALID_TYPES:
            return obj, False, "invalid_entity_type"

        if name not in original_text:
            return obj, False, "entity_not_in_text"

    return {
        "text": original_text,
        "entities": entities,
    }, True, None


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
        "api_errors": sum(
            r.get("status") == "api_error" for r in records
        ),
        "format_valid_samples": valid_formats,
        "format_valid_rate": (
            valid_formats / evaluated if evaluated else 0.0
        ),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Prompt structure comparison"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="测试样本数量，默认20",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=STRATEGIES,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/experiment2_smoke20/predictions",
    )

    parser.add_argument(
        "--report-dir",
        default="outputs/experiment2_smoke20/reports",
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
        raise RuntimeError("实验二需要至少3条训练示例")

    print("=" * 65)
    print("实验二：Prompt结构对比")
    print("=" * 65)
    print(f"测试样本数：{len(samples)}")
    print(f"可用训练示例：{len(examples)}")
    print(f"策略：{', '.join(args.strategies)}")
    print(f"Shot：2")
    print(f"预计最多请求：{len(samples) * len(args.strategies)}")
    print(f"预测目录：{output_dir}")
    print(f"报告目录：{report_dir}")

    for strategy in args.strategies:
        output_file = output_dir / f"{strategy}.jsonl"

        existing = (
            read_jsonl(output_file)
            if output_file.exists()
            else []
        )

        records_by_index = {
            r["sample_index"]: r
            for r in existing
            if "sample_index" in r
        }

        # 仅跳过已有模型响应的样本。
        # api_error 会在下一次运行时自动重试。
        done = {
            r["sample_index"]
            for r in existing
            if r.get("status") == "response"
        }

        print(f"\n===== 开始 {strategy} =====")

        with output_file.open("a", encoding="utf-8") as fout:
            for idx, sample in enumerate(samples):

                if idx in done:
                    continue

                gold = normalize_entities(
                    sample.get("entities", [])
                )

                try:
                    messages = build_messages(
                        text=sample["text"],
                        examples=examples,
                        strategy=strategy,
                    )

                    result = call_llm(messages)

                    parsed, valid, error = parse_response(
                        result.get("content", ""),
                        sample["text"],
                    )

                    pred = (
                        normalize_entities(parsed["entities"])
                        if valid else set()
                    )

                    record = {
                        "sample_index": idx,
                        "pmid": sample.get("pmid"),
                        "strategy": strategy,
                        "shots": 2,
                        "status": "response",
                        "requested_model": result.get(
                            "requested_model"
                        ),
                        "actual_model": result.get(
                            "actual_model"
                        ),
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

                except Exception as exc:
                    record = {
                        "sample_index": idx,
                        "pmid": sample.get("pmid"),
                        "strategy": strategy,
                        "shots": 2,
                        "status": "api_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                fout.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                fout.flush()

                records_by_index[idx] = record

                if record["status"] == "response":
                    print(
                        f"[{strategy}] {idx + 1}/{len(samples)} "
                        f"PMID={sample.get('pmid')} "
                        f"format_valid={record['format_valid']}"
                    )
                else:
                    print(
                        f"[{strategy}] {idx + 1}/{len(samples)} "
                        f"请求失败：{record['error']}"
                    )

                if args.delay > 0:
                    time.sleep(args.delay)

        final_records = [
            records_by_index[i]
            for i in sorted(records_by_index)
        ]

        metrics = calculate_metrics(final_records)

        models = [
            r.get("actual_model")
            for r in final_records
            if r.get("actual_model")
        ]

        metrics.update({
            "strategy": strategy,
            "shots": 2,
            "model": models[-1] if models else None,
        })

        report_file = report_dir / f"{strategy}_metrics.json"

        report_file.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n{strategy} 当前指标：")
        print(
            json.dumps(
                metrics,
                ensure_ascii=False,
                indent=2,
            )
        )

    print("\n实验二本轮运行结束。")
    print(f"预测文件：{output_dir}")
    print(f"评估报告：{report_dir}")


if __name__ == "__main__":
    main()
