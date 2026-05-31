# Fusion - 六边形开源大模型

**集百家之长，铸六边形开源大模型**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()

## 项目简介

Fusion 是一套面向**纯本地训练与推理**的开源大语言模型方案。它不只交付模型权重，更交付一整套可在消费级硬件上自行定制、微调、甚至复现预训练的全链路工具。

**核心理念：用户主权** - 所有训练数据清洗、模型微调、推理部署均在本地完成，无需依赖任何云端服务。

## 核心特性

### 滑动分块潜注意力（SBLA）
- 长序列切为定长块，块内高秩潜空间 + 块间极低秩潜向量
- 支持 32K 上下文窗口，KV 缓存仅为传统 GQA 的 1/8（SBLA 架构可扩展至 256K）
- 在 24 GB 显卡上即可对 14B 模型进行长文档微调与推理

### 动态推理强度调节器（Thinking Dial）
- 通过 `think_rank`（0-3）控制推理深度
- 0：直接作答（闲聊、翻译）
- 3：长思维链模式（数学、代码调试）
- 一个模型同时拥有快响应与深推理

### 双母语独立训练
- 中文达到 Qwen 同等深度，英文自然度对标 LLaMA 3.1
- 根除"翻译腔"，中英独立文化人格

### 教科书级知识蒸馏（T-KD）
- 使用开源教师模型改写高信誉源
- 生成风格统一、论证清晰的教学文本

## 快速开始

### 硬件要求

| 模型版本 | 全参微调 | QLoRA 微调 | 推理（量化） |
|---------|---------|------------|-------------|
| Fusion-0.5B | 8 GB | - | CPU |
| Fusion-1.5B | 16 GB | 8 GB | 4 GB |
| Fusion-8B | 24 GB | 8 GB | 8 GB |
| Fusion-14B | 48 GB | 16 GB | 16 GB |

### 安装依赖

```bash
git clone https://github.com/zhan1206/fusion-llm.git
cd fusion-llm
pip install -r requirements.txt
```

### Mini 模型快速训练

Mini 模型是一个小型字符级模型，用于快速验证训练流程，无需 GPU。

```bash
python train/train_mini.py
```

训练产物保存在 `output/mini_model/`。

### LoRA 微调（需要 GPU）

```bash
python train/lora_finetune.py --model_size 8B
```

### 全参微调（需要 GPU）

```bash
python train/full_finetune.py --model_size 8B
```

### 推理

```python
from models.fusion_model import FusionModel, FusionConfig

config = FusionConfig(vocab_size=10000, hidden_size=256, num_layers=2)
model = FusionModel(config)
model.eval()

# 基础推理
input_ids = torch.tensor([[1, 2, 3]])  # 替换为实际 token IDs
with torch.no_grad():
    outputs = model(input_ids)
```

或使用推理控制板：

```bash
python inference/dashboard.py
```

## 项目结构

```
fusion-llm/
├── models/              # 模型架构（SBLA注意力、Thinking Dial、FusionModel）
├── train/               # 训练脚本（LoRA微调、全参微调、DPO对齐、Mini模型）
├── data_pipeline/       # 数据处理（双语过滤、T-KD蒸馏）
├── inference/           # 推理部署（Dashboard、DyQuant量化、Ollama）
├── tokenizers/          # SentencePiece tokenizer 模型文件
├── configs/             # 配置文件模板（0.5B/1.5B/8B/14B/mini）
├── scripts/             # 工具脚本（tokenizer训练、数据去重）
├── tests/               # 单元测试
└── requirements.txt     # Python 依赖
```

## 核心技术

### 1. SBLA 注意力机制

```python
# models/sbla_attention.py
class SlidingBlockLatentAttention(nn.Module):
    """
    滑动分块潜注意力
    - 块内：高秩潜空间（保留细节）
    - 块间：极低秩潜向量（传递上下文）
    """
```

### 2. Thinking Dial 控制

```python
# 训练时标注 think_rank
{"text": "证明勾股定理", "think_rank": 3}

# 推理时通过 ThinkingDialProcessor 注入 thinking depth token
```

### 3. SentencePiece Tokenizer

项目使用 SentencePiece 训练专用 tokenizer，支持中英双语和 Fusion 特殊 token：

```bash
# 用自定义语料训练 tokenizer
python scripts/train_tokenizer.py --input data/tokenizer_train.txt --vocab_size 100000 --output tokenizers/

# 用内置 sample data 快速测试
python scripts/train_tokenizer.py --create_sample_data --input data/tokenizer_train.txt
python scripts/train_tokenizer.py --input data/tokenizer_train.txt --vocab_size 500 --output tokenizers/
```

## 许可证

本项目采用 **Apache License 2.0** - 详见 [LICENSE](LICENSE)

- 可商用
- 可修改
- 可私有部署
- 无附加条款

## 致谢

Fusion 项目受到以下开源项目的启发：

- [DeepSeek](https://github.com/deepseek-ai) - MLA 注意力机制
- [LLaMA](https://github.com/meta-llama/llama) - 基础 Transformer 架构
- [Qwen](https://github.com/QwenLM/Qwen) - 中文能力标杆
- [Transformers](https://github.com/huggingface/transformers) - 训练框架
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 分布式训练

## 联系方式

- 项目作者：zhan1206
- GitHub：[@zhan1206](https://github.com/zhan1206)
- 问题反馈：[Issues](https://github.com/zhan1206/fusion-llm/issues)

---

**Fusion 是一场回归用户主权的运动。你不必依赖我们提供的权重，也不必上传任何隐私数据。你手中的代码和算力，足以锻造属于你自己的最强模型。**
