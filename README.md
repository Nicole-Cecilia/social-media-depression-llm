# 基于大语言模型的社交媒体抑郁风险检测

机器学习课程设计项目：用 Qwen2.5-7B-Instruct 在两个真实社交媒体数据集上做抑郁风险检测，对比 Zero-shot / Few-shot / LoRA 微调三种方案。

- **SWDD（中文微博，三分类风险分级）**：`0=正常用户`、`1=抑郁但无自杀意念`、`2=抑郁伴自杀意念`。标签来自数据集原始标注：抑郁用户文件中 `label.symptoms.suicidal_ideation` 字段为真的归为等级 2。
- **Reddit（英文，二分类）**：`r/depression`（抑郁情绪）vs `r/SuicideWatch`（自杀意念），在统一风险框架下可对应等级 1 与等级 2（Reddit 没有正常对照组）。

两个数据集语言、样本单位（用户级 15 条微博聚合 vs 单帖）、标签粒度都不同，因此**分别训练、分别评估**，通过同一套代码接口保持可比。

## 项目结构

```
├── data/                  # 数据集（不进 git，需自行下载）
├── src/                   # 核心代码库
│   ├── config.py          # 所有路径与超参
│   ├── data_utils.py      # 双数据集加载、清洗、分层划分
│   ├── prompts.py         # 中英文 prompt 模板 + 鲁棒解析
│   ├── model_utils.py     # 4-bit 量化加载、贪心解码
│   └── evaluate.py        # Acc/P/R/F1 + 混淆矩阵
├── scripts/
│   ├── download_model.py
│   ├── run_zeroshot_fewshot.py
│   ├── run_lora.py
│   ├── verify_lora_tokenize.py
│   ├── make_dashboard.py
│   └── run_demo.py
├── docs/                  # Demo 截图
└── requirements.txt
```

## 环境

- Windows + NVIDIA GPU（RTX 4060 Ti 8GB 上跑通）
- Python 3.11 + PyTorch 2.5.1 + CUDA 12.1

```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

## 数据

| 数据集 | 语言 | 任务 | 样本量 |
|---|---|---|---|
| SWDD (Weibo) | 中文 | 三分类：正常 / 抑郁 / 高风险 | 2400 用户（每类 800）|
| Reddit | 英文 | 二分类：depression / SuicideWatch | 7000 帖子 |

划分：分层 75% train / 10% val / 15% test。

## 复现步骤

```bash
# 1. 下载模型
python -m scripts.download_model

# 2. zero-shot / few-shot
python -m scripts.run_zeroshot_fewshot --dataset both

# 3. LoRA 微调
python -m scripts.run_lora --dataset swdd
python -m scripts.run_lora --dataset reddit

# 4. 汇总看板
python -m scripts.make_dashboard

# 5. 交互 demo
python -m scripts.run_demo
```

## 交互 Demo

Gradio 界面，粘贴一段文本给出：三档风险等级 + 自动摘录的关键证据句 + 心理关怀指引。

![Demo 界面](docs/demo_screenshot.png)

启动后浏览器打开 `http://127.0.0.1:7860`，内置 4 个预设示例可一键测试。

## 实验结果

硬件：RTX 4060 Ti 8GB，模型 Qwen2.5-7B-Instruct（NF4 量化）。SWDD 三分类报告 macro-F1；Reddit 二分类报告正类 F1。

| 数据集 | 方法 | Accuracy | Precision | Recall | F1 | N |
|---|---|---|---|---|---|---|
| SWDD (中文) | Zero-shot | 0.522 | 0.589 | 0.522 | 0.467 | 360 |
| SWDD (中文) | Few-shot | 0.567 | 0.606 | 0.567 | 0.525 | 360 |
| SWDD (中文) | **LoRA r=32** | **0.619** | **0.629** | **0.619** | **0.617** | 360 |
| Reddit (英文) | Zero-shot | 0.710 | 0.701 | 0.697 | 0.699 | 1050 |
| Reddit (英文) | Few-shot | 0.699 | 0.669 | 0.748 | 0.706 | 1050 |
| Reddit (英文) | LoRA | 0.726 | 0.722 | 0.705 | **0.713** | 1050 |

### SWDD 三分类混淆矩阵（LoRA r=32）

| 真实\预测 | 正常 | 抑郁 | 高风险 | 召回 |
|---|---|---|---|---|
| 正常（120） | **92** | 24 | 4 | 0.77 |
| 抑郁（120） | 21 | **74** | 25 | 0.62 |
| 高风险（120） | 16 | 47 | **57** | 0.48 |
| 精确率 | 0.76 | 0.51 | 0.64 | — |

### 结果分析

1. LoRA 在两个数据集上都最优：SWDD macro-F1=0.617（比 few-shot 高 9 个点），Reddit F1=0.713。
2. 三分类比二分类难：把抑郁类劈成"抑郁"和"高风险"两个高度重叠的子类，F1 从 0.79 降到 0.62。
3. 类2召回 0.48 是已知局限：约一半伴自杀意念用户被误判为普通抑郁。
4. rank 消融：r=8/16/32 对应 F1=0.471/0.512/0.572，三分类任务复杂度高，更大 rank 带来收益。

## 技术要点

- **QLoRA**：NF4 量化 7B 模型（~5.2GB）+ LoRA r=32，8GB 显存可训
- **Loss masking**：只对 assistant 答案 token 算 loss（-100 mask prompt 部分）
- **答案截断修复**：用户文本先按 token 截断，给答案预留 160 token
- **8GB 适配**：batch=1 + 梯度累积 16 + gradient_checkpointing + paged_adamw_8bit
