# ICL-Based Chemical and Disease Named Entity Recognition

基于大语言模型上下文学习（In-Context Learning, ICL）的化学物质与疾病命名实体识别实验。

本项目使用 **Qwen2.5-7B-Instruct**，通过本地 Ollama 服务运行模型，基于 BC5CDR 数据集开展化学物质（Chemical）和疾病（Disease）命名实体识别（Named Entity Recognition, NER）实验，重点研究示例数量和 Prompt 设计策略对实体识别性能的影响。

---

## 1. 项目简介

传统命名实体识别任务通常依赖大量标注数据和模型微调。本项目探索利用大语言模型的上下文学习能力，通过在 Prompt 中提供少量标注示例，引导模型完成实体识别任务。

项目主要围绕以下问题展开：

1. **示例数量的影响**：比较 Zero-shot、1-shot、2-shot 和 3-shot 设置下的实体识别效果。
2. **Prompt 策略的影响**：在固定 2-shot 设置下，比较五种不同 Prompt 策略。
3. **本地模型实验**：使用本地部署的 Qwen2.5-7B-Instruct 模型开展 ICL 实验，分析不同示例数量下的性能变化。
4. **错误分析**：通过预测结果和评价指标，分析不同策略下的识别效果。

## 2. 技术栈

| 技术 | 用途 |
|---|---|
| Python | 实验开发与数据处理 |
| Qwen2.5-7B-Instruct | 大语言模型 |
| Ollama | 本地模型部署与推理 |
| OpenAI Python SDK | 调用 Ollama 兼容接口 |
| BC5CDR | 化学物质与疾病 NER 数据集 |
| JSON / JSONL | 数据及实验结果存储 |

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
└── outputs/
    ├── prompt_v3_full500/
    │   └── reports/
    ├── experiment2_500/
    │   └── reports/
    └── local_qwen25_full/
        └── reports/
```

## 4. 实验设计

### 实验一：不同示例数量的影响

在固定 Prompt 策略的条件下，改变上下文中提供的标注示例数量，比较模型的实体识别表现。

实验设置：

| 实验组 | 示例数量 |
|---|---:|
| Zero-shot | 0 |
| 1-shot | 1 |
| 2-shot | 2 |
| 3-shot | 3 |

实验规模：500 条测试样本。

模型：Qwen2.5-7B-Instruct（本地 Ollama 部署）。

实验结果：

| 策略 | Precision | Recall | F1 | 格式有效率 |
|---|---:|---:|---:|---:|
| Zero-shot | 0.6741 | 0.2506 | 0.3654 | 71.6% |
| 1-shot | 0.6900 | 0.3204 | 0.4376 | 84.4% |
| 2-shot | 0.6612 | 0.3596 | 0.4659 | 82.0% |
| 3-shot | 0.6339 | 0.3798 | 0.4750 | 79.2% |

初步观察：

- 随着示例数量增加，Recall 和 F1 整体提高。
- Precision 随示例数量增加呈下降趋势。
- 3-shot 获得最高 F1，达到 0.4750。
- 1-shot 的格式有效率最高，为 84.4%。

### 实验二：不同 Prompt 策略的影响

固定示例数量为 2-shot，比较五种 Prompt 策略（T0–T4）。

实验规模：500 条测试样本。

实验结果：

| 策略 | Precision | Recall | F1 | 格式有效率 |
|---|---:|---:|---:|---:|
| T0 | 0.6160 | 0.3683 | 0.4610 | 76.2% |
| T1 | 0.6385 | 0.3664 | 0.4656 | 81.0% |
| T2 | 0.5879 | 0.3804 | 0.4620 | 78.8% |
| T3 | 0.5968 | 0.3889 | 0.4709 | 79.8% |
| T4 | 0.6435 | 0.3588 | 0.4607 | 81.2% |

初步观察：

- T3 的 F1 最高，为 0.4709，同时具有最高 Recall。
- T4 的 Precision 最高，为 0.6435，格式有效率也最高。
- 不同策略之间的 F1 差距较小，尚不能据此认定某种策略具有统计显著优势。

### 实验三：本地 Qwen 模型的示例数量实验

使用本地 Ollama 部署的 Qwen2.5-7B-Instruct，比较不同示例数量下的实体识别表现。

实验规模：500 条测试样本。

实验结果：

| 策略 | Precision | Recall | F1 | 格式有效率 |
|---|---:|---:|---:|---:|
| Zero-shot | 0.6077 | 0.3096 | 0.4102 | 76.8% |
| 1-shot | 0.6640 | 0.3449 | 0.4540 | 78.4% |
| 2-shot | 0.6545 | 0.3581 | 0.4629 | 75.6% |
| 3-shot | 0.6110 | 0.3748 | 0.4646 | 72.6% |

初步观察：

- 3-shot 获得最高 F1，为 0.4646。
- 随着示例数量增加，Recall 整体提高。
- 格式有效率并未随示例数量增加而提高。

注意：该实验与实验一可能采用不同 Prompt 版本和配置，因此两组实验的结果不能简单用于归因某一个 Prompt 变量的影响。

## 5. 评价指标

本项目主要采用以下指标评价实体识别效果：

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

- TP：正确识别的实体数量。
- FP：错误识别的实体数量。
- FN：未能识别的真实实体数量。

**格式有效率（Format Validity Rate）**

模型输出满足预期结构化 JSON 格式要求的样本比例。

## 6. 环境配置

### 6.1 创建 Python 环境

建议使用 Python 3.10 或更新的兼容版本。

```bash
conda create -n demo4_icl_ner python=3.10 -y
conda activate demo4_icl_ner
```

### 6.2 安装依赖

```bash
pip install -r requirements.txt
```

依赖文件：

```text
openai
python-dotenv
json-repair
```

### 6.3 部署 Ollama

安装 Ollama 后，下载模型：

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

启动 Ollama 服务，并检查模型：

```bash
ollama list
```

检查 API 服务：

```bash
curl http://127.0.0.1:11434/api/tags
```

### 6.4 配置环境变量

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

## 7. 运行实验

首先准备 BC5CDR 数据集，并根据项目的数据处理脚本生成实验所需的数据文件。

运行前，可以通过以下命令查看脚本支持的参数：

```bash
python src/evaluate_icl.py --help
python src/evaluate_experiment2.py --help
python scripts/prepare_data.py --help
```

不同实验使用对应的 Prompt 模板和评估脚本：

- 示例数量实验：`src/evaluate_icl.py`
- Prompt 策略实验：`src/evaluate_experiment2.py`
- 实验结果分析：`src/analyze_experiment2.py`
- 案例对比分析：`src/compare_experiment2_cases.py`

具体数据路径和运行参数以脚本配置为准。

## 8. 实验结果文件

仓库目前保留三组完整实验的指标报告：

| 目录 | 实验内容 |
|---|---|
| `outputs/prompt_v3_full500/reports/` | 不同示例数量实验 |
| `outputs/experiment2_500/reports/` | 五种 Prompt 策略实验 |
| `outputs/local_qwen25_full/reports/` | 本地 Qwen 示例数量实验 |

预测 JSONL、原始数据及模型文件未包含在当前仓库提交中。

## 9. 注意事项

1. 本项目使用 BC5CDR 数据集，使用者需要自行获取数据，并遵守其许可及使用条款。
2. 本地推理需要提前安装 Ollama 并下载对应模型。
3. 实验结果受模型量化方式、Prompt 版本、示例选择和推理参数影响。
4. 当前报告主要展示 500 条测试样本上的实验结果，不应直接视为完整测试集上的最终性能。
5. 不同实验之间若存在 Prompt 或配置差异，应谨慎进行横向比较。
