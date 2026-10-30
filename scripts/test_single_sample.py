import json
import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/bc5cdr_test_first500.jsonl"
OUTPUT_DIR = ROOT / "outputs/predictions"

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()

if not API_KEY or API_KEY == "在这里替换成你的真实API_Key":
    raise RuntimeError("请先在项目根目录 .env 中配置真实的 OpenRouter API Key")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"找不到测试数据：{DATA_PATH}")

# 只读取第一条样本，不遍历500条
with DATA_PATH.open("r", encoding="utf-8") as f:
    first_line = f.readline().strip()

if not first_line:
    raise RuntimeError("测试集文件为空")

sample = json.loads(first_line)

print("===== 样本字段检查 =====")
print("字段：", list(sample.keys()))

# 兼容常见的文本字段命名
text = (
    sample.get("text")
    or sample.get("input")
    or sample.get("document")
    or sample.get("content")
)

if not isinstance(text, str) or not text.strip():
    raise ValueError(
        "第一条样本中没有找到有效文本字段。"
        f"现有字段为：{list(sample.keys())}"
    )

pmid = sample.get("pmid", sample.get("id", "unknown"))

print("PMID/ID：", pmid)
print("文本长度：", len(text), "字符")
print("文本预览：", text[:300].replace("\n", " "))

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=120.0,
    max_retries=1,
)

system_prompt = """
You are a biomedical named entity recognition system.
Extract entity mentions belonging to exactly these two types:
- Chemical
- Disease

Return ONLY a valid JSON array. Each element must have exactly:
{"name": "entity mention copied from the input", "type": "Chemical"}
or
{"name": "entity mention copied from the input", "type": "Disease"}

Rules:
1. Extract entity mentions explicitly present in the input text.
2. Preserve the spelling and wording of each mention as it appears.
3. Do not output explanations, Markdown fences, or extra fields.
4. Do not output relations or entity IDs.
5. If no entities are found, return [].
6. Avoid duplicate entries with the same name and type.
"""

user_prompt = f"""
Extract Chemical and Disease entities from this biomedical article.

Article:
{text}

Return only the JSON array.
"""

print("\n===== 正在调用 API（仅一次请求） =====")
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0,
    max_tokens=4096,
)

raw_output = response.choices[0].message.content or ""
actual_model = response.model

# 尝试解析 JSON；保留原始输出，方便排查格式问题
parsed = None
parse_error = None

try:
    parsed = json.loads(raw_output)
except json.JSONDecodeError:
    # 兼容模型意外添加的 Markdown 代码围栏
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw_output,
        flags=re.IGNORECASE,
    ).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)

valid_entities = []
if isinstance(parsed, list):
    for item in parsed:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        entity_type = item.get("type")

        if (
            isinstance(name, str)
            and name.strip()
            and entity_type in {"Chemical", "Disease"}
        ):
            valid_entities.append({
                "name": name.strip(),
                "type": entity_type,
            })

# 保存预测结果和原始输出
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = OUTPUT_DIR / f"single_sample_{timestamp}.json"

result = {
    "pmid": pmid,
    "requested_model": MODEL,
    "actual_model": actual_model,
    "text_length": len(text),
    "raw_output": raw_output,
    "parsed_json": parsed,
    "valid_entities": valid_entities,
    "parse_success": isinstance(parsed, list) and parse_error is None,
    "parse_error": parse_error,
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n===== API 调用成功 =====")
print("请求模型：", MODEL)
print("实际模型：", actual_model)
print("JSON 解析成功：", result["parse_success"])
print("有效实体数量：", len(valid_entities))
print("\n===== 模型抽取结果 =====")
print(json.dumps(valid_entities, ensure_ascii=False, indent=2))

if parse_error:
    print("\nJSON 解析错误：", parse_error)
    print("原始输出：", raw_output)

print("\n结果已保存至：", output_path)
