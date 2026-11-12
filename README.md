# ICL-Based Chemical and Disease Named Entity Recognition

基于上下文学习（In-Context Learning，ICL）的化学物质与疾病命名实体识别实验。项目使用 Qwen2.5-7B-Instruct 模型，在 BC5CDR 数据集上开展零样本（Zero-shot）和少样本（Few-shot）实体识别实验，并比较不同提示策略对识别效果的影响。

## 1. 项目简介

本项目以 BC5CDR 数据集为基础，完成 Chemical（化学物质）和 Disease（疾病）两类实体的命名实体识别任务。

实验使用测试集前 500 条样本，通过 API 调用大语言模型，比较不同上下文示例数量及提示策略下的实体识别性能。

项目主要包含两个实验：

- **实验一：** 比较 Zero-shot 和不同示例数量的 Few-shot 提示策略。
- **实验二：** 比较 T0、T1、T2、T3、T4 五种提示策略，在统一示例选择条件下分析不同提示设计的效果。

实验采用严格的实体边界和实体类型匹配方式计算 Precision、Recall 和 F1-score。

## 2. 技术栈

| 技术 | 用途 |
|---|---|
| Python 3.11 | 实验开发与数据处理 |
| Qwen2.5-7B-Instruct | 命名实体识别模型 |
| Ollama | 本地大语言模型部署与调用 |
| OpenRouter API | 大语言模型 API 调用接口 |
| JSON / JSONL | 数据存储与模型输出格式 |
| scikit-learn | 辅助数据处理与指标计算 |
| Git / GitHub | 代码版本管理与项目存储 |

## 3. 项目结构

```text
demo4_icl_ner/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── prompts/
│   ├── templates.py
│   └── experiment2/
│       └── templates.py
│
├── scripts/
│   ├── prepare_data.py
│   └── test_single_sample.py
│
├── src/
│   ├── api_client.py
│   ├── evaluate_icl.py
│   ├── evaluate_experiment2.py
│   ├── metrics.py
│   ├── analyze_experiment2.py
│   └── compare_experiment2_cases.py
│
├── outputs/
│   ├── experiment1_500_entity_filter/
│   │   ├── predictions/
│   │   └── reports/
│   │
│   └── experiment2_500_controlled_examples/
│       ├── predictions/
│       └── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

说明：

- `data/`：存放数据集及预处理文件。
- `prompts/`：存放实验提示词模板。
- `scripts/`：存放数据预处理和单样本测试脚本。
- `src/`：存放 API 调用、实验评估和指标计算代码。
- `outputs/`：存放正式实验的预测结果及评估报告。

## 4. 实验设计与结果

### 4.1 实验一：不同示例数量的 ICL 实验

实验目的：研究上下文示例数量对 Chemical 和 Disease 实体识别效果的影响。

实验设置：

- 数据集：BC5CDR 测试集前 500 条样本。
- 实体类型：Chemical、Disease。
- 模型：Qwen2.5-7B-Instruct。
- 实验策略：Zero-shot、1-shot、2-shot、3-shot。
- 评价指标：Precision、Recall、F1-score、Format Validity Rate。

实验结果：

| 策略 | Precision | Recall | F1-score | Format Validity Rate |
|---|---:|---:|---:|---:|
| Zero-shot | 0.6466 | 0.3414 | 0.4469 | 99.4% |
| 1-shot | 0.6288 | 0.4091 | 0.4957 | 97.4% |
| 2-shot | 0.6292 | 0.4418 | 0.5191 | 98.2% |
| 3-shot | 0.6051 | 0.4607 | **0.5231** | 97.0% |

实验分析：

1. Few-shot 策略整体优于 Zero-shot 策略，说明提供上下文示例有助于模型识别实体。
2. 随着示例数量增加，Recall 整体提高，说明模型能够识别出更多真实实体。
3. 3-shot 获得最高 F1-score，为 0.5231，Recall 达到 0.4607。
4. 从 2-shot 到 3-shot，F1-score 仅提升 0.0040，说明增加示例数量带来的性能提升逐渐减小。
5. Zero-shot 的 Precision 最高，为 0.6466，但 Recall 较低，说明模型在没有示例的情况下预测相对保守。

### 4.2 实验二：不同提示策略对比实验

实验目的：研究不同提示词设计策略对实体识别性能的影响。

实验设置：

- 数据集：BC5CDR 测试集前 500 条样本。
- 实体类型：Chemical、Disease。
- 模型：Qwen2.5-7B-Instruct。
- 对比策略：T0、T1、T2、T3、T4。
- 示例选择：统一使用相同的两个训练示例，减少示例选择差异带来的影响。
- 评价指标：Precision、Recall、F1-score、Format Validity Rate。

实验结果：

| 策略 | Precision | Recall | F1-score | Format Validity Rate |
|---|---:|---:|---:|---:|
| T0 | 0.5979 | 0.4526 | 0.5152 | 95.2% |
| T1 | **0.6265** | 0.4284 | 0.5088 | 95.0% |
| T2 | 0.6149 | 0.4227 | 0.5010 | 97.2% |
| T3 | 0.5826 | **0.4708** | **0.5208** | **98.0%** |
| T4 | 0.6248 | 0.4173 | 0.5004 | 96.2% |

实验分析：

1. T3 获得最高 F1-score，为 0.5208，同时 Recall 最高，为 0.4708。
2. T1 获得最高 Precision，为 0.6265，说明该策略的预测结果相对更加准确。
3. T3 的 Format Validity Rate 达到 98.0%，在五种策略中最高。
4. T4 的 F1-score 最低，为 0.5004。
5. 不同提示策略在 Precision 和 Recall 之间表现出不同的权衡关系，说明提示词设计会影响模型的实体识别行为。

## 5. 综合实验分析

### 5.1 两组实验的最佳结果

| 实验 | 最优策略 | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| 实验一 | 3-shot | 0.6051 | 0.4607 | **0.5231** |
| 实验二 | T3 | 0.5826 | 0.4708 | **0.5208** |

实验一的 3-shot 策略取得最高 F1-score，为 0.5231；实验二的 T3 策略取得最高 F1-score，为 0.5208。

两组实验的设置和比较变量有所不同，因此上述结果主要用于分别分析各组实验内部的策略表现，不应直接将两组实验的差异全部归因于某一种提示策略。

### 5.2 主要发现

1. **上下文示例具有积极作用。** 实验一中，Few-shot 策略整体优于 Zero-shot 策略。
2. **增加示例数量能够提高召回率。** 从 Zero-shot 到 3-shot，Recall 从 0.3414 提升至 0.4607。
3. **提示词设计会影响识别效果。** 实验二中，不同策略在 Precision、Recall 和 F1-score 上存在差异。
4. **Precision 和 Recall 存在权衡。** T1 的 Precision 最高，而 T3 的 Recall 和 F1-score 最高。
5. **格式有效性和实体识别性能并不完全一致。** 输出格式更加有效，并不代表实体识别准确率一定更高。

## 6. 评价指标

本项目采用严格匹配方式评价实体识别效果。只有实体文本边界和实体类型均与真实标注一致，才计为正确识别。

### 6.1 Precision（精确率）

预测正确的实体数量占模型预测实体总数的比例。

$$
P=\frac{TP}{TP+FP}
$$

### 6.2 Recall（召回率）

预测正确的实体数量占真实实体总数的比例。

$$
R=\frac{TP}{TP+FN}
$$

### 6.3 F1-score

Precision 和 Recall 的调和平均数。

$$
F1=\frac{2PR}{P+R}
$$

### 6.4 Format Validity Rate（格式有效率）

模型输出满足预期结构化格式要求的样本比例。

$\text{Format Validity Rate}=\frac{\text{格式有效的样本数}}{\text{总样本数}}$

### 6.5 TP、FP、FN

| 指标 | 含义 |
|---|---|
| TP | 正确识别的实体数量 |
| FP | 模型预测存在，但与真实标注不匹配的实体数量 |
| FN | 真实存在，但模型未能正确识别的实体数量 |

说明：

- TP、FP、FN 均按照实体文本和实体类型严格匹配计算。
- `type_errors` 用于单独记录实体类型不符合要求的情况，不应与 API 请求失败混为一谈。
- 当前解析逻辑会过滤原文中不存在的预测实体，保留其他有效实体，不会因为单个未在原文中出现的实体而直接丢弃整条响应。

## 7. 环境配置

### 7.1 创建 Conda 环境

```bash
conda create -n demo4_icl_ner python=3.11 -y
conda activate demo4_icl_ner
```

### 7.2 安装依赖

```bash
pip install -r requirements.txt
```

## 8. 运行实验

### 8.1 单样本测试

```bash
python scripts/test_single_sample.py
```

### 8.2 实验一：Zero-shot 与 Few-shot

```bash
python src/evaluate_icl.py \
    --limit 500 \
    --strategies zero_shot few_shot_1 few_shot_2 few_shot_3
```

### 8.3 实验二：不同提示策略

```bash
python src/evaluate_experiment2.py \
    --limit 500 \
    --strategies T0 T1 T2 T3 T4 \
    --delay 0.2 \
    --output-dir outputs/experiment2_500_controlled_examples/predictions \
    --report-dir outputs/experiment2_500_controlled_examples/reports
```

具体参数以对应脚本中的命令行参数定义为准。

## 9. 实验结果文件

### 9.1 实验一

结果目录：

```text
outputs/experiment1_500_entity_filter/
├── predictions/
└── reports/
```

其中：

- `predictions/`：保存不同 ICL 策略下的模型预测结果。
- `reports/`：保存对应策略的评估指标和实验报告。

### 9.2 实验二

结果目录：

```text
outputs/experiment2_500_controlled_examples/
├── predictions/
└── reports/
```

其中：

- `predictions/`：保存 T0、T1、T2、T3、T4 的预测结果。
- `reports/`：保存各提示策略的评估报告。

## 10. 实验结果解释与局限性

### 10.1 输出格式处理

本项目对模型输出进行了 JSON 解析和实体类型检查。

对于模型输出中不符合实体类型要求的内容，单独统计 `type_errors`；对于原文中不存在的实体，当前解析逻辑会将其过滤，并保留同一响应中的其他有效实体。


### 10.2 实验局限性

1. 当前实验仅使用 BC5CDR 测试集前 500 条样本，实验规模有限。
2. 实验结果受到模型本身、提示词设计和上下文示例选择的影响。
3. 当前主要使用单一模型开展实验，尚未充分比较不同模型之间的性能差异。
4. 大语言模型生成结果具有一定随机性，后续可以通过多次重复实验进一步验证结果的稳定性。
5. 当前实体识别仍存在漏检和误检现象，后续可以结合错误案例分析进一步改进提示策略。

## 11. 项目总结

本项目基于 Qwen2.5-7B-Instruct 模型，完成了 BC5CDR 数据集上的 Chemical 和 Disease 两类实体识别实验。

通过两组实验，分别研究了上下文示例数量和不同提示策略对实体识别性能的影响。

实验一中，3-shot 策略取得最高 F1-score，为 0.5231；实验二中，T3 策略取得最高 F1-score，为 0.5208。

实验结果表明，合理设计上下文示例和提示词能够在一定程度上改善大语言模型的实体识别效果，但模型在召回率、实体边界识别和输出稳定性方面仍有进一步提升空间。
