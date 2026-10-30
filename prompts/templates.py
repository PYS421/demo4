import json

VALID_TYPES = {"Chemical", "Disease"}


def format_entities(entities):
    output = []

    for item in entities or []:
        name = item.get("entity")
        entity_type = item.get("type")

        if (
            isinstance(name, str)
            and name.strip()
            and entity_type in VALID_TYPES
        ):
            output.append({
                "entity": name.strip(),
                "type": entity_type,
            })

    return output


def build_messages(text, examples=None, strategy="zero_shot"):
    """
    strategy:
        zero_shot: 0 个示例
        few_shot_1: 1 个示例
        few_shot_2: 2 个示例
        few_shot_3: 3 个示例
    """
    examples = examples or []

    expected_counts = {
        "zero_shot": 0,
        "few_shot_1": 1,
        "few_shot_2": 2,
        "few_shot_3": 3,
    }

    if strategy not in expected_counts:
        raise ValueError(f"不支持的策略：{strategy}")

    required = expected_counts[strategy]

    if len(examples) < required:
        raise ValueError(
            f"{strategy} 需要 {required} 个示例，"
            f"实际只有 {len(examples)} 个"
        )

    system_prompt = """
You are a precise biomedical named entity recognition system.

Task:
Extract Chemical and Disease entities explicitly mentioned in the target article only.

Allowed entity types:
- Chemical: chemical compounds, drugs, and other chemical substances.
- Disease: diseases, disorders, and pathological conditions.

Do not label physiological measurements, biological processes, treatments,
procedures, or study groups as Chemical or Disease merely because they appear
in a biomedical paper.

Return exactly one valid JSON object with exactly one key:
{"entities": [{"entity": "<exact span>", "type": "Chemical"}]}

Rules:
1. Every entity must be an exact, contiguous substring of the target article.
2. Each entity object must have exactly the keys "entity" and "type".
3. The type must be exactly "Chemical" or "Disease".
4. Output each distinct entity string and type combination at most once.
5. Never repeat an entity object or append duplicate entities.
6. Examples are guidance only. Copy no entity from an example unless it is
   present in the target article and correctly typed.
7. Do not invent, paraphrase, normalize, or expand entity names.
8. If uncertain whether a phrase belongs to an allowed type, omit it.
9. If no target entities are found, return an empty entities array.
10. Return JSON only. No Markdown fences, explanations, IDs, or extra keys.
11. Do not output the article text. Only output the entities JSON object.
12. After completing the single JSON object, stop immediately.
""".strip()

    user_parts = []

    if required:
        user_parts.append(
            "Use the following labeled examples as context. "
            "Learn the expected extraction format and entity types."
        )

        for index, example in enumerate(examples[:required], start=1):
            example_text = example.get("text", "")
            example_entities = format_entities(example.get("entities", []))

            user_parts.append(
                f"\nExample {index}:\n"
                f"Article:\n{example_text}\n"
                f"Gold output:\n"
                f"{json.dumps({'entities': example_entities}, ensure_ascii=False)}"
            )

    user_parts.append(
        "\nNow extract entities from this new article.\n"
        f"Article:\n{text}\n\n"
        "Return only the required JSON object."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


if __name__ == "__main__":
    demo_examples = [
        {
            "text": "Aspirin may treat headache.",
            "entities": [
                {"entity": "Aspirin", "type": "Chemical"},
                {"entity": "headache", "type": "Disease"},
            ],
        }
    ]

    for strategy in ("zero_shot", "few_shot_1"):
        messages = build_messages(
            text="Ibuprofen was used for pain.",
            examples=demo_examples,
            strategy=strategy,
        )
        print(f"{strategy}: {len(messages)} messages; Prompt built successfully")

    print("Prompt 模板本地测试通过。")
