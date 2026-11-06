# ICL-Based Chemical and Disease Named Entity Recognition

基于大语言模型上下文学习（In-Context Learning, ICL）的化学物质与疾病命名实体识别实验。

本项目使用 **Qwen2.5-7B-Instruct**，通过本地 Ollama 服务运行模型，基于 BC5CDR 数据集开展化学物质（Chemical）和疾病（Disease）命名实体识别（Named Entity Recognition, NER）实验，重点研究上下文示例数量和 Prompt 设计策略对实体识别性能的影响。

---

## 1. 项目简介

传统命名实体识别任务通常依赖大量标注数据和模型微调。本项目探索利用大语言模型的上下文学习能力，通过在 Prompt 中提供少量标注示例，引导模型直接完成实体识别任务。

项目主要研究以下两个问题：

1. **上下文示例数量的影响**：比较 Zero-shot、1-shot、2-shot 和 3-shot 设置下的实体识别效果，分析示例数量与模型性能之间的关系。
2. **Prompt 设计策略的影响**：在固定 2-shot 设置下，比较五种不同 Prompt 策略（T0–T4），分析提示词设计对实体识别性能的影响。

本项目使用本地部署的 Qwen2.5-7B-Instruct 模型，通过 Ollama 提供 OpenAI 兼容 API 接口。实验在相同的 BC5CDR 测试样本上开展，并使用 Precision、Recall、F1-score 等指标进行评价。

## 2. 技术栈

| 技术 | 用途 |
|---|---|
| Python | 实验开发与数据处理 |
| Qwen2.5-7B-Instruct | 大语言模型 |
| Ollama | 本地模型部署与推理 |
| OpenAI Python SDK | 调用 Ollama 兼容接口 |
| BC5CDR | 化学物质与疾病 NER 数据集 |
| JSON / JSONL | 数据及实验结果存储 |
| json-repair | 模型输出 JSON 格式修复 |
| python-dotenv | 环境变量配置 |

## 3. 项目结构

```text
demo4_icl_ner/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
│
├── prompts/
│   ├── __init__.py
│   ├── templates.py
│   └── experiment2/
│       ├── __init__.py
│       └── templates.py
│
├── src/
│   ├── __init__.py
│   ├── api_client.py
│   ├── metrics.py
│   ├── evaluate_icl.py
│   ├── evaluate_experiment2.py
│   ├── analyze_experiment2.py
│   └── compare_experiment2_cases.py
│
├── scripts/
│   ├── prepare_data.py
│   └── test_single_sample.py
│
├── data/
│   └── processed/
│       ├── bc5cdr_test_first500.jsonl
│       └── bc5cdr_train_examples.jsonl
│
└── outputs/
    ├── experiment1_500/
    │   ├── predictions/
    │   └── reports/
    │
    └── experiment2_500_rerun/
        ├── predictions/
        └── reports/
```

说明：

- `prompts/`：存放不同实验使用的 Prompt 模板。
- `src/api_client.py`：封装模型 API 调用。
- `src/metrics.py`：实现实体识别评价指标计算。
- `src/evaluate_icl.py`：执行不同上下文示例数量的实验。
- `src/evaluate_experiment2.py`：执行不同 Prompt 策略的实验。
- `src/analyze_experiment2.py`：分析实验二的结果。
- `src/compare_experiment2_cases.py`：对比不同 Prompt 策略下的预测案例。
- `scripts/`：存放数据准备和单样本测试脚本。
- `outputs/`：存放模型预测结果和实验指标报告。

## 4. 实验设计

### 4.1 实验一：上下文示例数量对 NER 性能的影响

在固定 Prompt 模板的条件下，改变上下文中提供的标注示例数量，比较模型的实体识别表现。

实验设置：

| 实验组 | 示例数量 |
|---|---:|
| Zero-shot | 0 |
| 1-shot | 1 |
| 2-shot | 2 |
| 3-shot | 3 |

实验配置：

- 数据集：BC5CDR 测试集前 500 条样本。
- 模型：Qwen2.5-7B-Instruct。
- 推理方式：本地 Ollama API。
- 实验规模：每种策略 500 条样本，共 2000 条预测记录。
- 评价方式：基于实体边界和实体类型的严格匹配。

实验结果：

| 策略 | Shots | Precision | Recall | F1-score | 格式有效率 |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 0.6172 | 0.3436 | 0.4414 | 85.6% |
| 1-shot | 1 | 0.6159 | 0.4112 | 0.4932 | 83.8% |
| 2-shot | 2 | 0.6188 | 0.4414 | 0.5152 | 83.0% |
| 3-shot | 3 | 0.5889 | 0.4739 | **0.5252** | 81.8% |

详细统计：

| 策略 | TP | FP | FN | 类型错误数 |
|---|---:|---:|---:|---:|
| Zero-shot | 1585 | 983 | 3028 | 13 |
| 1-shot | 1897 | 1183 | 2716 | 4 |
| 2-shot | 2036 | 1254 | 2577 | 9 |
| 3-shot | 2186 | 1526 | 2427 | 9 |

实验分析：

1. Few-shot 相比 Zero-shot 整体取得更好的实体识别效果。3-shot 的 F1-score 达到 0.5252，相比 Zero-shot 提高约 8.37 个百分点。
2. 随着示例数量增加，Recall 从 0.3436 提升到 0.4739，说明上下文示例有助于模型发现更多真实实体。
3. Precision 没有随示例数量增加而提高，3-shot 的 Precision 为 0.5889，低于其他设置，表明模型在提高实体覆盖率的同时也产生了更多误检。
4. 从 2-shot 增加到 3-shot，F1-score 仅提升约 0.99 个百分点，说明继续增加示例带来的收益有所减小。
5. Zero-shot 的格式有效率最高，为 85.6%，但其实体识别性能最低。

实验结论：

在本实验设置下，3-shot 获得最高 F1-score，说明适当增加上下文示例能够改善模型的实体识别能力，尤其是召回能力。但示例数量增加并不意味着 Precision 和输出格式有效率同步提高。

### 4.2 实验二：不同 Prompt 策略对 NER 性能的影响

固定上下文示例数量为 2-shot，比较五种不同 Prompt 策略（T0–T4），研究 Prompt 设计对实体识别性能的影响。

实验配置：

- 数据集：BC5CDR 测试集前 500 条样本。
- 模型：Qwen2.5-7B-Instruct。
- 上下文示例数量：固定为 2-shot。
- Prompt 策略：T0、T1、T2、T3、T4。
- 实验规模：每种策略 500 条样本，共 2500 条预测记录。
- 评价方式：基于实体边界和实体类型的严格匹配。

实验结果：

| 策略 | Precision | Recall | F1-score | 格式有效率 |
|---|---:|---:|---:|---:|
| T0 | 0.5873 | 0.4698 | **0.5220** | 80.6% |
| T1 | 0.6161 | 0.4450 | 0.5168 | 82.0% |
| T2 | 0.5577 | 0.4754 | 0.5133 | 78.6% |
| T3 | 0.5680 | **0.4780** | 0.5191 | 81.0% |
| T4 | **0.6155** | 0.4279 | 0.5049 | 83.2% |

详细统计：

| 策略 | TP | FP | FN | 类型错误数 |
|---|---:|---:|---:|---:|
| T0 | 2167 | 1523 | 2446 | 50 |
| T1 | 2053 | 1279 | 2560 | 11 |
| T2 | 2193 | 1739 | 2420 | 12 |
| T3 | 2205 | 1677 | 2408 | 19 |
| T4 | 1974 | 1233 | 2639 | 5 |

实验分析：

1. T0 的 F1-score 最高，为 0.5220，在 Precision 和 Recall 之间取得了相对较好的平衡。
2. T3 的 Recall 最高，为 0.4780，说明该策略能够识别更多真实实体，但其误检数量也较多。
3. T4 的 Precision 最高，为 0.6155，同时格式有效率最高，为 83.2%，但 Recall 相对较低。
4. T2 的 Recall 为 0.4754，但 Precision 仅为 0.5577，说明其在提高实体覆盖率的同时产生了较多误检。
5. T0 与 T3 的 F1-score 分别为 0.5220 和 0.5191，差距较小，二者表现较为接近。

实验结论：

不同 Prompt 策略会影响模型的实体识别行为。T0 在综合 F1-score 上表现最好，T3 在 Recall 上表现最好，T4 在 Precision 和格式有效率上表现最好。这说明 Prompt 设计需要在实体覆盖率、误检控制和输出格式约束之间进行权衡。

## 5. 两组实验综合分析

两组实验均使用相同模型和相同的 500 条测试样本，分别研究上下文示例数量和 Prompt 设计策略对 NER 性能的影响。

| 实验 | 最佳策略 | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| 实验一 | 3-shot | 0.5889 | 0.4739 | **0.5252** |
| 实验二 | T0 | 0.5873 | 0.4698 | 0.5220 |

综合观察：

1. 在当前实验设置下，Few-shot 相比 Zero-shot 能够提高实体识别性能。
2. 增加上下文示例数量主要改善 Recall，但可能带来更多误检。
3. 不同 Prompt 策略对 Precision、Recall 和 F1-score 的影响并不一致。
4. 最佳示例数量策略与最佳 Prompt 策略的 F1-score 差距较小。

需要说明的是，实验一和实验二的控制变量不同，不能仅凭两组最佳结果直接判断示例数量和 Prompt 设计哪一个影响更大。此外，本实验仅使用 500 条测试样本，且未进行多随机种子重复实验，因此目前的结果属于初步实验观察，不能据此认定差异具有统计显著性。

## 6. 评价指标

本项目采用 Precision、Recall、F1-score 和格式有效率评价模型性能。

**Precision（精确率）**

预测正确的实体数量占模型预测实体总数的比例。

\[
P=\frac{TP}{TP+FP}
\]

**Recall（召回率）**

预测正确的实体数量占真实实体总数的比例。

\[
R=\frac{TP}{TP+FN}
\]

**F1-score**

Precision 和 Recall 的调和平均数。

\[
F1=\frac{2PR}{P+R}
\]

其中：

- TP（True Positive）：正确识别的实体数量。
- FP（False Positive）：错误识别的实体数量。
- FN（False Negative）：未能识别的真实实体数量。

本项目采用严格实体匹配方式，实体边界和实体类型均正确时才计为 TP。

**格式有效率（Format Validity Rate）**

模型输出满足预期结构化 JSON 格式要求的样本比例。

\[
\text{Format Validity Rate}
=
\frac{\text{格式有效的样本数}}{\text{总样本数}}
\]

此外，实验还记录：

- API 错误数（API Errors）。
- 处理错误数（Processing Errors）。
- 类型错误数（Type Errors）。
- 存在无效实体类型的样本数（Invalid Type Samples）。

需要注意：格式有效率与实体识别准确率是不同的指标。输出格式正确不代表实体识别一定正确；部分格式存在问题的响应也可能包含可解析的有效实体。

## 7. 环境配置

### 7.1 创建 Python 环境

建议使用 Python 3.11。

```bash
conda create -n demo4_icl_ner python=3.11 -y
conda activate demo4_icl_ner
```

### 7.2 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

```text
openai
python-dotenv
json-repair
```

### 7.3 部署 Ollama

安装 Ollama 后，下载 Qwen 模型：

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

查看本地模型：

```bash
ollama list
```

检查 Ollama API 服务：

```bash
curl http://127.0.0.1:11434/api/tags
```

当前实验使用的模型为：

```text
qwen2.5:7b-instruct-q4_K_M
```

### 7.4 配置环境变量

复制环境变量示例文件：

```bash
cp .env.example .env
```

根据实际部署情况修改 `.env`：

```env
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M
OLLAMA_MAX_TOKENS=2048
```

当前 API 客户端使用本地 Ollama 的 OpenAI 兼容接口：

```text
http://127.0.0.1:11434/v1
```

请勿将包含 API 密钥的 `.env` 文件提交到 GitHub。

## 8. 运行实验

首先准备 BC5CDR 数据集，并根据项目的数据处理脚本生成实验所需的数据文件。

查看脚本支持的参数：

```bash
python scripts/prepare_data.py --help
python src/evaluate_icl.py --help
python src/evaluate_experiment2.py --help
```

### 8.1 运行实验一

实验一比较 Zero-shot、1-shot、2-shot 和 3-shot 四种设置。

```bash
python src/evaluate_icl.py --help
```

根据脚本支持的参数配置实验，正式实验使用 500 条测试样本，预测结果保存至：

```text
outputs/experiment1_500/predictions/
```

指标报告保存至：

```text
outputs/experiment1_500/reports/
```

### 8.2 运行实验二

实验二比较 T0、T1、T2、T3、T4 五种 Prompt 策略，固定为 2-shot。

```bash
python src/evaluate_experiment2.py \
  --limit 500 \
  --strategies T0 T1 T2 T3 T4 \
  --delay 0.2 \
  --output-dir outputs/experiment2_500_rerun/predictions \
  --report-dir outputs/experiment2_500_rerun/reports
```

预测结果保存至：

```text
outputs/experiment2_500_rerun/predictions/
```

指标报告保存至：

```text
outputs/experiment2_500_rerun/reports/
```

注意：如果重新运行实验，建议使用新的输出目录，避免已有 JSONL 记录触发断点续跑，影响实验结果的可追溯性。

### 8.3 实验结果分析

查看实验二分析脚本的参数：

```bash
python src/analyze_experiment2.py --help
python src/compare_experiment2_cases.py --help
```

根据脚本参数指定预测文件，开展不同策略的指标分析和案例对比。

## 9. 实验结果文件

两组正式实验的结果分别保存在以下目录：

| 目录 | 内容 |
|---|---|
| `outputs/experiment1_500/predictions/` | 实验一的 JSONL 预测记录 |
| `outputs/experiment1_500/reports/` | 实验一的指标报告 |
| `outputs/experiment2_500_rerun/predictions/` | 实验二重新运行后的 JSONL 预测记录 |
| `outputs/experiment2_500_rerun/reports/` | 实验二重新运行后的指标报告 |

每条预测记录包含样本编号、PMID、实验策略、模型响应状态、真实实体、预测实体、原始模型回答和格式校验结果等信息。

指标报告采用 JSON 格式，记录样本数量、TP、FP、FN、Precision、Recall、F1-score、格式有效率及错误统计。

预测结果和实验报告可以用于后续的误差分析、策略比较和实验复现。

模型权重、原始数据及其他大型文件不包含在当前代码仓库提交中。

---
## 11. 项目总结

本项目基于 Qwen2.5-7B-Instruct 和 Ollama，完成了 BC5CDR 化学物质与疾病命名实体识别的两组上下文学习实验。

实验一研究不同上下文示例数量对实体识别性能的影响，3-shot 获得最高 F1-score（0.5252）。

实验二研究不同 Prompt 设计策略的影响，T0 获得最高 F1-score（0.5220）。

实验结果表明，上下文示例数量和 Prompt 设计均会影响模型的实体识别表现。通过合理设计上下文示例和提示词，可以在一定程度上改善大语言模型在专业领域 NER 任务中的识别效果。
