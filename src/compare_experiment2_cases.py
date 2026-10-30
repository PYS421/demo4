import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

PRED_DIR = BASE_DIR / "outputs/experiment2_smoke20/predictions"
OUT_DIR = BASE_DIR / "outputs/experiment2_smoke20/error_analysis"

STRATEGIES = ["T0", "T1", "T2", "T3", "T4"]

TARGET_PMIDS = {
    "8701013",
    "439781",
    "22836123",
    "23433219",
    "23535177",
    "23666265",
    "23846525",
}


def load_jsonl(path):
    records = []
    if not path.exists():
        print(f"[警告] 文件不存在：{path}")
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] {path.name} 第{line_no}行解析失败：{e}")

    return records


def normalize_entity(item):
    return (
        str(item.get("entity", "")).strip(),
        str(item.get("type", "")).strip()
    )


def get_pmid(record):
    value = record.get("pmid", record.get("PMID", ""))
    return str(value).strip()


def partial_candidates(gold_entity, predictions):
    """仅提供可能的边界部分匹配线索，不自动判定为边界错误。"""
    gold_lower = gold_entity.lower()
    candidates = []

    for pred in predictions:
        pred_entity, pred_type = normalize_entity(pred)
        pred_lower = pred_entity.lower()

        if (
            pred_lower
            and pred_lower != gold_lower
            and (
                pred_lower in gold_lower
                or gold_lower in pred_lower
            )
        ):
            candidates.append({
                "predicted_entity": pred_entity,
                "predicted_type": pred_type,
            })

    return candidates


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    all_pmids = set()

    for strategy in STRATEGIES:
        path = PRED_DIR / f"{strategy}.jsonl"
        records = load_jsonl(path)

        strategy_data = {}
        for record in records:
            pmid = get_pmid(record)
            if pmid:
                strategy_data[pmid] = record
                all_pmids.add(pmid)

        data[strategy] = strategy_data

    results = []
    missing_pmids = []

    for pmid in sorted(TARGET_PMIDS):
        strategy_records = {
            s: data[s].get(pmid)
            for s in STRATEGIES
        }

        available = [
            record for record in strategy_records.values()
            if record is not None
        ]

        if not available:
            missing_pmids.append(pmid)
            continue

        # Gold 标注通常在五个策略的记录中重复保存，取第一条可用记录。
        gold_entities = available[0].get("gold_entities", [])
        gold_entities = [
            normalize_entity(x) for x in gold_entities
        ]

        case = {
            "pmid": pmid,
            "gold_entities": [
                {"entity": e, "type": t}
                for e, t in gold_entities
            ],
            "strategies": {},
            "gold_entity_comparison": [],
        }

        strategy_predictions = {}

        for strategy in STRATEGIES:
            record = strategy_records[strategy]

            if record is None:
                case["strategies"][strategy] = {
                    "status": "missing_record",
                    "predicted_entities": [],
                }
                strategy_predictions[strategy] = []
                continue

            predictions = record.get("predicted_entities", [])
            predictions = predictions if isinstance(predictions, list) else []
            strategy_predictions[strategy] = predictions

            gold_set = set(gold_entities)
            pred_set = {
                normalize_entity(x)
                for x in predictions
                if isinstance(x, dict)
            }

            case["strategies"][strategy] = {
                "status": "format_invalid" if record.get("format_valid") is False else "valid",
                "format_valid": record.get("format_valid"),
                "format_error": record.get("format_error"),
                "predicted_entities": predictions,
                "raw_response": record.get("raw_response", ""),
                "tp": [
                    {"entity": e, "type": t}
                    for e, t in sorted(gold_set & pred_set)
                ],
                "fn": [
                    {"entity": e, "type": t}
                    for e, t in sorted(gold_set - pred_set)
                ],
                "fp": [
                    {"entity": e, "type": t}
                    for e, t in sorted(pred_set - gold_set)
                ],
            }

        for gold_entity, gold_type in gold_entities:
            item = {
                "gold_entity": gold_entity,
                "gold_type": gold_type,
                "strategies": {},
            }

            for strategy in STRATEGIES:
                record = strategy_records[strategy]

                if record is None:
                    item["strategies"][strategy] = {
                        "status": "missing_record"
                    }
                    continue

                predictions = strategy_predictions[strategy]
                pred_pairs = {
                    normalize_entity(x)
                    for x in predictions
                    if isinstance(x, dict)
                }

                exact = (gold_entity, gold_type) in pred_pairs
                same_text_wrong_type = [
                    {"entity": e, "type": t}
                    for e, t in pred_pairs
                    if e.lower() == gold_entity.lower()
                    and t != gold_type
                ]

                item["strategies"][strategy] = {
                    "status": (
                        "format_invalid"
                        if record.get("format_valid") is False
                        else "exact_match" if exact
                        else "wrong_type" if same_text_wrong_type
                        else "not_found"
                    ),
                    "wrong_type_predictions": same_text_wrong_type,
                    "possible_boundary_matches": (
                        partial_candidates(gold_entity, predictions)
                        if not exact else []
                    ),
                }

            case["gold_entity_comparison"].append(item)

        results.append(case)

    output_path = OUT_DIR / "case_comparison.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 输出适合直接阅读的终端摘要
    print("\n========== 7篇文章逐案例对照摘要 ==========")

    for case in results:
        print(f"\nPMID: {case['pmid']}")
        print(f"Gold实体数：{len(case['gold_entities'])}")

        for strategy in STRATEGIES:
            info = case["strategies"][strategy]

            if info["status"] == "missing_record":
                print(f"  {strategy}: 缺少记录")
                continue

            tp = len(info.get("tp", []))
            fp = len(info.get("fp", []))
            fn = len(info.get("fn", []))

            print(
                f"  {strategy}: "
                f"状态={info['status']}, "
                f"TP={tp}, FP={fp}, FN={fn}"
            )

        print("  Gold实体逐条识别情况：")
        for entity_info in case["gold_entity_comparison"]:
            statuses = " | ".join(
                f"{s}:{entity_info['strategies'][s]['status']}"
                for s in STRATEGIES
            )
            print(
                f"    [{entity_info['gold_type']}] "
                f"{entity_info['gold_entity']} -> {statuses}"
            )

    print("\n========== 数据完整性 ==========")
    print(f"成功匹配文章数：{len(results)}")
    print(f"缺失PMID：{missing_pmids}")
    print(f"完整对照文件：{output_path}")


if __name__ == "__main__":
    main()
