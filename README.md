# 基于大语言模型的社交媒体抑郁风险检测

机器学习课程设计项目：用 Qwen2.5-7B-Instruct 在两个真实社交媒体数据集上做抑郁风险检测，对比 Zero-shot / Few-shot / LoRA 微调三种方案。

- **SWDD（中文微博，三分类风险分级）**：`0=正常用户`、`1=抑郁但无自杀意念`、`2=抑郁伴自杀意念`。标签来自数据集原始标注：抑郁用户文件中 `label.symptoms.suicidal_ideation` 字段为真的归为等级 2。
- **Reddit（英文，二分类）**：`r/depression`（抑郁情绪）vs `r/SuicideWatch`（自杀意念），在统一风险框架下可对应等级 1 与等级 2（Reddit 没有正常对照组）。

两个数据集语言、样本单位（用户级 15 条微博聚合 vs 单帖）、标签粒度都不同，因此**分别训练、分别评估**，通过同一套代码接口保持可比。

## 项目结构

```
├── data/                  # 数据集（不进 git，需自行下载）
│   ├── swdd/              # Sina Weibo Depression Dataset (中文)
│   └── reddit/            # Reddit depression / SuicideWatch (英文)
├── src/                   # 核心代码库
│   ├── config.py          # 所有路径与超参集中在这里
│   ├── data_utils.py      # 双数据集加载、清洗、分层 train/val/test 划分
│   ├── prompts.py         # 中英文 prompt 模板、答案词、鲁棒输出解析
│   ├── model_utils.py     # 4-bit 量化加载、贪心解码封装
│   └── evaluate.py        # Acc/P/R/F1（二分类 binary、多分类 macro）+ 混淆矩阵
├── scripts/
│   ├── download_model.py          # 从 ModelScope 下载 Qwen2.5-7B-Instruct
│   ├── run_zeroshot_fewshot.py    # zero-shot / few-shot 评估
│   ├── run_lora.py                # LoRA/QLoRA 微调 + 测试集评估
│   ├── verify_lora_tokenize.py    # 不加载模型，秒级验证训练样本构造
│   ├── make_dashboard.py          # 汇总结果 xlsx + 画对比看板
│   └── run_demo.py                # Gradio 交互界面（风险+证据句+关怀建议）
└── requirements.txt
```

## 环境

- Windows + NVIDIA GPU（本项目在 RTX 4060 Ti 8GB 上跑通）
- Python 3.11 (conda env: py311)
- PyTorch 2.5.1 + CUDA 12.1（**必须装 CUDA 版 torch**）

```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

## 数据

| 数据集 | 语言 | 任务 | 样本量 |
|---|---|---|---|
| SWDD (Weibo) | 中文 | 三分类：正常 / 抑郁无自杀意念 / 抑郁伴自杀意念 | 2400 用户（每等级 800）|
| Reddit | 英文 | 二分类：depression / SuicideWatch | 7000 帖子 |

划分：分层抽样 75% train / 10% val / 15% test（SWDD 为 1800/240/360，Reddit 为 5250/700/1050）。

## 复现步骤

```bash
# 1. 下载模型（约 15GB，从 ModelScope）
python -m scripts.download_model

# 2. zero-shot / few-shot 基线
python -m scripts.run_zeroshot_fewshot --dataset both

# 3. LoRA 微调
python -m scripts.run_lora --dataset swdd
python -m scripts.run_lora --dataset reddit

# 4. 汇总结果表 + 生成看板
python -m scripts.make_dashboard

# 5. 交互 demo
python -m scripts.run_demo
```

## 实验结果

硬件：RTX 4060 Ti 8GB，模型 Qwen2.5-7B-Instruct（NF4 量化）。SWDD 三分类报告 **macro-F1**；Reddit 二分类报告正类 F1。

| 数据集 | 方法 | Accuracy | Precision | Recall | F1 | N |
|---|---|---|---|---|---|---|
| SWDD (中文) | Zero-shot | 0.522 | 0.589 | 0.522 | 0.467 | 360 |
| SWDD (中文) | Few-shot | 0.567 | 0.606 | 0.567 | 0.525 | 360 |
| SWDD (中文) | **LoRA r=32** | **0.619** | **0.629** | **0.619** | **0.617** | 360 |
| Reddit (英文) | Zero-shot | 0.710 | 0.701 | 0.697 | 0.699 | 1050 |
| Reddit (英文) | Few-shot | 0.699 | 0.669 | 0.748 | 0.706 | 1050 |
| Reddit (英文) | LoRA | 0.726 | 0.722 | 0.705 | **0.713** | 1050 |

### SWDD 三分类混淆矩阵（LoRA r=32，行=真实，列=预测）

| 真实\预测 | 正常 | 抑郁 | 高风险 | 召回 |
|---|---|---|---|---|
| 正常（120） | **92** | 24 | 4 | 0.77 |
| 抑郁（120） | 21 | **74** | 25 | 0.62 |
| 高风险（120） | 16 | 47 | **57** | 0.48 |
| 精确率 | 0.76 | 0.51 | 0.64 | — |

### 结果分析

1. **LoRA 在两个数据集上都最优**：SWDD 三分类 macro-F1=0.617，比 few-shot（0.525）高 9 个点；Reddit 二分类 LoRA F1=0.713。
2. **三分类比二分类难**：SWDD 二分类时代 F1=0.79，三分类降到 0.62，因为把抑郁类劈成了"抑郁"和"高风险"两个高度重叠的子类。
3. **类2（高风险）召回 0.48 是已知局限**：约一半伴自杀意念的用户被误判为普通抑郁，后续改进方向是把带自杀词的微博优先排列。
4. **rank 消融**：r=8 F1=0.471、r=16 F1=0.512、r=32 F1=0.572，三分类任务复杂度高，更大的 LoRA 秩带来收益。

## LoRA 训练的两个关键实现细节

1. **答案截断修复**：用户文本先按 token 截断、给指令和答案预留 160 token，保证末尾答案不被右截断切掉。
2. **Loss masking**：只对 assistant 回答的 token 算 loss（其余填 -100），训练 loss 从 3+ 降到 0.1 以下。

## 8GB 显存适配

- 模型 NF4 量化 ~5.2GB；batch=1 + 梯度累积 16 + gradient_checkpointing + paged_adamw_8bit
- LoRA 只挂 q/k/v/o_proj；推理贪心解码 max_new_tokens=8
