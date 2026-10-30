import json

VALID_TYPES = {"Chemical", "Disease"}

STRATEGIES = ["T0", "T1", "T2", "T3", "T4"]

# T0/T1/T3/T4 共用相同示例
COMMON_EXAMPLE_INDICES = [0, 1]

# T2 使用另一组候选示例
T2_EXAMPLE_INDICES = [1, 2]


PROMPTS = {
    "T0": """
You are a biomedical named entity recognition system.

Extract Chemical and Disease entities from the target article.

Chemical refers to chemical compounds, drugs, and chemical substances.
Disease refers to diseases, disorders, and pathological conditions.

Return the extracted entities in JSON format.
""".strip(),

    "T1": """
You are a precise biomedical named entity recognition system.

Task:
Identify all Chemical and Disease entities explicitly mentioned in the
target article.

Entity definitions:
- Chemical: chemical compounds, drugs, and other chemical substances.
- Disease: diseases, disorders, and pathological conditions.

Strict extraction constraints:
1. Every entity must be an exact, contiguous substring of the article.
2. Do not normalize, paraphrase, expand, or invent entity names.
3. Assign exactly one correct type: Chemical or Disease.
4. Do not label physiological measurements, biological processes,
   treatments, procedures, or study groups as entities.
5. Do not output duplicate entity/type combinations.
6. Extract entities from the target article only, never copy entities
   from examples unless they also occur in the target article.
7. If no entities are found, return an empty array.
""".strip(),

    "T2": """
You are a biomedical named entity recognition system.

Learn the annotation patterns from the labeled examples, paying particular
attention to:
- How Chemical and Disease entities are distinguished.
- How entity boundaries are selected.
- How entities are represented as exact spans in the original text.

Apply the same annotation principles to the target article.

Only extract entities explicitly present in the target article.
Each entity must be an exact, contiguous substring.
Allowed types are Chemical and Disease.
Do not copy entities from examples unless they occur in the target article.
Do not output duplicate entities.
""".strip(),

    "T3": """
You are a biomedical named entity recognition system.

Identify Chemical and Disease entities in the target article.

Process the article using these steps internally:

Step 1: Read the entire target article and identify candidate entity spans.
Step 2: Determine whether each candidate is a Chemical or a Disease.
Step 3: Remove candidates that are not explicitly mentioned entities,
        are not one of the two allowed types, or are not exact text spans.
Step 4: Remove duplicate entity/type combinations.
Step 5: Construct the final JSON object.

Do not reveal these steps or provide explanations.
Only return the final JSON object.

Allowed types:
- Chemical
- Disease
""".strip(),

    "T4": """
You are a biomedical named entity recognition system.

Extract Chemical and Disease entities from the target article.

The output MUST satisfy this JSON schema:

{
  "entities": [
    {
      "entity": "exact entity text from the article",
      "type": "Chemical"
    }
  ]
}

Mandatory output rules:
1. The top-level object must contain exactly one key: "entities".
2. "entities" must be a JSON array.
3. Every array element must contain exactly two keys:
   "entity" and "type".
4. "entity" must be a non-empty string copied exactly from the article.
5. "type" must be exactly "Chemical" or "Disease".
6. Use double quotes for all JSON keys and string values.
7. Do not add trailing commas.
8. Do not use Markdown code fences.
9. Do not include explanations, comments, article text, or extra keys.
10. If no entities are found, output {"entities": []}.
11. Output one complete JSON object and then stop.
""".strip(),
}


def format_entities(entities):
    """保留合法的实体标注。"""
    result = []
    seen = set()

    for item in entities or []:
        if not isinstance(item, dict):
            continue

        name = item.get("entity")
        entity_type = item.get("type")

        if (
            isinstance(name, str)
            and name.strip()
            and entity_type in VALID_TYPES
        ):
            key = (name.strip(), entity_type)
            if key not in seen:
                seen.add(key)
                result.append({
                    "entity": name.strip(),
                    "type": entity_type,
                })

    return result


def select_examples(examples, strategy):
    """根据策略选择固定的两个示例。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"不支持的策略：{strategy}")

    indices = (
        T2_EXAMPLE_INDICES
        if strategy == "T2"
        else COMMON_EXAMPLE_INDICES
    )

    if len(examples) <= max(indices):
        raise ValueError(
            f"{strategy} 所需示例不足，当前只有 {len(examples)} 条"
        )

    selected = [examples[i] for i in indices]

    if len(selected) != 2:
        raise ValueError(f"{strategy} 必须使用2个示例")

    return selected


def build_messages(text, examples, strategy):
    """构建实验二指定策略的2-shot Prompt。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"不支持的策略：{strategy}")

    selected_examples = select_examples(examples, strategy)

    user_parts = [
        "The following are labeled biomedical NER examples."
    ]

    for index, example in enumerate(selected_examples, start=1):
        example_text = example.get("text", "")
        example_entities = format_entities(example.get("entities", []))

        user_parts.append(
            f"\nExample {index}:\n"
            f"Article:\n{example_text}\n"
            f"Gold output:\n"
            f"{json.dumps({'entities': example_entities}, ensure_ascii=False)}"
        )

    user_parts.append(
        "\nNow process the target article.\n"
        f"Target article:\n{text}\n\n"
        'Return only one JSON object in the form: {"entities": [...]}'
    )

    return [
        {
            "role": "system",
            "content": PROMPTS[strategy],
        },
        {
            "role": "user",
            "content": "\n".join(user_parts),
        },
    ]


if __name__ == "__main__":
    demo_examples = [
        {
            "text": "Aspirin may treat headache.",
            "entities": [
                {"entity": "Aspirin", "type": "Chemical"},
                {"entity": "headache", "type": "Disease"},
            ],
        },
        {
            "text": "Lidocaine caused cardiac asystole.",
            "entities": [
                {"entity": "Lidocaine", "type": "Chemical"},
                {"entity": "cardiac asystole", "type": "Disease"},
            ],
        },
        {
            "text": "Metformin was studied in diabetes.",
            "entities": [
                {"entity": "Metformin", "type": "Chemical"},
                {"entity": "diabetes", "type": "Disease"},
            ],
        },
    ]

    for strategy in STRATEGIES:
        messages = build_messages(
            text="Ibuprofen was used to treat pain.",
            examples=demo_examples,
            strategy=strategy,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        print(
            f"{strategy}: Prompt构建成功，"
            f"system长度={len(messages[0]['content'])}，"
            f"user长度={len(messages[1]['content'])}"
        )

    print("实验二 Prompt 模板测试全部通过。")
