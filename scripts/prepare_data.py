import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "CDR_Data" / "CDR.Corpus.v010516"
OUT_DIR = ROOT / "data" / "processed"

TRAIN_FILE = CORPUS_DIR / "CDR_TrainingSet.PubTator.txt"
TEST_FILE = CORPUS_DIR / "CDR_TestSet.PubTator.txt"

TARGET_TYPES = {"Chemical", "Disease"}


def read_pubtator(path):
    """读取 PubTator 文件，每篇文章作为一条样本。"""
    documents = {}
    current_id = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")

            if not line.strip():
                current_id = None
                continue

            # 标题或摘要行：PMID|t|... 或 PMID|a|...
            if "|t|" in line or "|a|" in line:
                pmid, kind, content = line.split("|", 2)
                current_id = pmid

                if pmid not in documents:
                    documents[pmid] = {
                        "pmid": pmid,
                        "title": "",
                        "abstract": "",
                        "entities": [],
                    }

                if kind == "t":
                    documents[pmid]["title"] = content
                elif kind == "a":
                    documents[pmid]["abstract"] = content

                continue

            # 实体标注行：PMID start end mention type entity_id
            parts = line.split()

            if len(parts) < 6:
                continue

            pmid = parts[0]

            # CID 是关系标注，不属于实体
            if parts[1] == "CID":
                continue

            entity_type = parts[-2]
            if entity_type not in TARGET_TYPES:
                continue

            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue

            mention = " ".join(parts[3:-2])

            if pmid not in documents:
                documents[pmid] = {
                    "pmid": pmid,
                    "title": "",
                    "abstract": "",
                    "entities": [],
                }

            documents[pmid]["entities"].append({
                "entity": mention,
                "type": entity_type,
                "start": start,
                "end": end,
            })

    result = []

    for doc in documents.values():
        text = " ".join(
            part for part in [doc["title"], doc["abstract"]] if part
        )

        if not text:
            continue

        # 输出实体文本与类型；同名同类型实体去重
        seen = set()
        entities = []

        for ent in doc["entities"]:
            key = (ent["entity"], ent["type"])
            if key not in seen:
                seen.add(key)
                entities.append({
                    "entity": ent["entity"],
                    "type": ent["type"],
                })

        result.append({
            "pmid": doc["pmid"],
            "text": text,
            "entities": entities,
        })

    return result


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = read_pubtator(TRAIN_FILE)
    test = read_pubtator(TEST_FILE)

    # 评估集固定取官方测试集按文件顺序排列的前 500 篇
    test500 = test[:500]

    # Few-shot 示例候选：独立训练集的前 3 篇
    train3 = train[:3]

    save_jsonl(OUT_DIR / "bc5cdr_train_examples.jsonl", train3)
    save_jsonl(OUT_DIR / "bc5cdr_test_first500.jsonl", test500)

    print("===== 数据处理完成 =====")
    print(f"训练集文章数：{len(train)}")
    print(f"测试集文章数：{len(test)}")
    print(f"Few-shot 候选数：{len(train3)}")
    print(f"评估文章数：{len(test500)}")
    print(f"训练示例文件：{OUT_DIR / 'bc5cdr_train_examples.jsonl'}")
    print(f"评估数据文件：{OUT_DIR / 'bc5cdr_test_first500.jsonl'}")

    if len(test500) != 500:
        print("警告：测试集不足 500 篇，请检查原始文件。")

    if test500:
        print("\n===== 第一篇测试样本预览 =====")
        print(json.dumps(test500[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
