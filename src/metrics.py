from collections import Counter


VALID_TYPES = {"Chemical", "Disease"}


def normalize_entities(entities):
    normalized = set()

    for item in entities or []:
        if not isinstance(item, dict):
            continue

        # 兼容数据集标注字段 entity 和模型输出字段 name
        name = item.get("entity", item.get("name"))
        entity_type = item.get("type")

        if not isinstance(name, str) or not name.strip():
            continue

        if entity_type not in VALID_TYPES:
            continue

        normalized.add((name.strip(), entity_type))

    return normalized


def calculate_metrics(gold_entities, pred_entities):
    gold = normalize_entities(gold_entities)
    pred = normalize_entities(pred_entities)

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_entities": len(gold),
        "pred_entities": len(pred),
    }


def evaluate_dataset(records):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    parse_errors = 0

    for record in records:
        gold = normalize_entities(record.get("gold", []))
        pred = normalize_entities(record.get("pred", []))

        total_tp += len(gold & pred)
        total_fp += len(pred - gold)
        total_fn += len(gold - pred)

        if record.get("parse_error", False):
            parse_errors += 1

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "samples": len(records),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_errors": parse_errors,
    }


if __name__ == "__main__":
    # 使用人工构造的小例子验证计算逻辑
    gold = [
        {"entity": "Famotidine", "type": "Chemical"},
        {"entity": "delirium", "type": "Disease"},
        {"entity": "ulcers", "type": "Disease"},
        {"entity": "famotidine", "type": "Chemical"},
    ]

    pred = [
        {"name": "Famotidine", "type": "Chemical"},
        {"name": "delirium", "type": "Disease"},
        {"name": "stress ulcers", "type": "Disease"},
    ]

    result = calculate_metrics(gold, pred)

    print("===== Strict Match 测试 =====")
    for key, value in result.items():
        print(f"{key}: {value}")

    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 2
    assert abs(result["f1"] - 4 / 7) < 1e-10

    print("\n评估逻辑测试通过。")
