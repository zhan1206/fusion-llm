# Fusion - 六边形开源大模型

**集百家之长，铸六边形开源大模型**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()

## 🎯 项目简介

Fusion 是一套面向**纯本地训练与推理**的开源大语言模型方案。它不只交付模型权重，更交付一整套可在消费级硬件上自行定制、微调、甚至复现预训练的全链路工具。

**核心理念：用户主权** - 所有训练数据清洗、模型微调、推理部署均在本地完成，无需依赖任何云端服务。

## ✨ 核心特性

### 🧠 滑动分块潜注意力（SBLA）
- 长序列切为定长块，块内高秩潜空间 + 块间极低秩潜向量
- 当前支持 32K 上下文窗口，KV 缓存仅为传统 GQA 的 1/8（SBLA 架构可扩展至 256K，需配置 RoPE scaling）
- 在 24 GB 显卡上即可对 14B 模型进行长文档微调与推理

### 🎛️ 动态推理强度调节器（Thinking Dial）
- 通过 `<|think| depth=0/1/2/3|>` 控制推理深度
- depth=0：直接作答（闲聊、翻译）
- depth=3：长思维链模式（数学、代码调试）
- 一个模型同时拥有 Mistral 的爽快与 DeepSeek 的深沉

### 🌏 双母语独立训练
- 中文达到 Qwen 同等深度，英文自然度对标 LLaMA 3.1
- 根除"翻译腔"，中英独立文化人格
- 中英比例 1:1，绝不混合 token

### 📚 教科书级知识蒸馏（T-KD）
- 使用开源教师模型改写 200+ 高信誉源
- 生成风格统一、论证清晰的教学文本（约 20T Token）
- 8B 模型知识面比肩传统 70B

## 🚀 快速开始

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

### 快速微调（LoRA）

```bash
# 8B 模型，单卡 24GB
bash train/lora_finetune.sh --model_size 8B --data_path data/example.json

# 14B 模型，QLoRA，单卡 16GB
bash train/lora_finetune.sh --model_size 14B --quantize --lora_rank 64
```

### 本地推理

```python
from inference import FusionInference

# 加载模型
model = FusionInference.from_pretrained("fusion-8b-instruct")

# 普通推理（快速）
response = model.generate("解释量子纠缠", thinking_depth=0)

# 深度推理（思维链）
response = model.generate("证明费马大定理", thinking_depth=3)
```

## 📁 项目结构

```
fusion-llm/
├── models/              # 模型架构（SBLA注意力、Thinking Dial）
├── train/               # 训练脚本（预训练、SFT、RLHF）
├── data_pipeline/       # 数据处理（Bi-Lingual TrueFilter、T-KD）
├── inference/           # 推理部署（Ollama、vLLM、API）
├── configs/             # 配置文件模板
├── scripts/             # 工具脚本（量化、转换）
├── docs/                # 详细文档
└── tests/               # 单元测试
```

## 🔧 核心技术

### 1. SBLA 注意力机制

```python
# models/sbla_attention.py
class SlidingBlockLatentAttention(nn.Module):
    """
    滑动分块潜注意力
    - 块内：高秩潜空间（保留细节）
    - 块间：极低秩潜向量（传递上下文）
    """
    def forward(self, x, block_size=512, ...):
        # 实现详见 models/sbla_attention.py
        ...
```

### 2. Thinking Dial 控制

```python
# 训练时标注 think_rank
{"text": "证明勾股定理", "think_rank": 3}

# 推理时控制
model.generate(prompt, thinking_depth=2)  # 0-3 级别
```

### 3. 双母语数据清洗

```python
# data_pipeline/bilingual_filter.py
filter = BilingualTrueFilter(
    lang="zh",
    filters=["remove_machine_translation", "remove_clickbait"]
)
clean_data = filter.process(raw_data)
```

## 📊 性能对比

| 模型 | 中文能力 | 英文能力 | 推理能力 | 长文本 | 显存占用 |
|------|---------|---------|---------|--------|---------|
| Qwen-8B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 32K | 高 |
| LLaMA-8B | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 8K | 中 |
| **Fusion-8B** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **32K** (可扩展256K) | **低** |

## 📖 文档

- [架构设计](docs/architecture.md) - SBLA、Thinking Dial 详解
- [训练指南](docs/training.md) - 从数据准备到模型部署
- [数据配方](docs/data_recipe.md) - Bi-Lingual TrueFilter、T-KD
- [推理部署](docs/inference.md) - Ollama、vLLM、API
- [硬件估算](docs/hardware.md) - 各型号显卡配置建议

## 🤝 贡献指南

我们欢迎任何形式的贡献！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📜 许可证

本项目采用 **Apache License 2.0** - 详见 [LICENSE](LICENSE)

✅ 可商用  
✅ 可修改  
✅ 可私有部署  
✅ 无附加条款  

## 🙏 致谢

Fusion 项目受到以下开源项目的启发：

- [DeepSeek](https://github.com/deepseek-ai) - MLA 注意力机制
- [LLaMA](https://github.com/meta-llama/llama) - 基础Transformer架构
- [Qwen](https://github.com/QwenLM/Qwen) - 中文能力标杆
- [Transformers](https://github.com/huggingface/transformers) - 训练框架
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 分布式训练

## 📧 联系方式

- 项目作者：zhan1206
- GitHub：[@zhan1206](https://github.com/zhan1206)
- 问题反馈：[Issues](https://github.com/zhan1206/fusion-llm/issues)

---

**Fusion 是一场回归用户主权的运动。你不必依赖我们提供的权重，也不必上传任何隐私数据。你手中的代码和算力，足以锻造属于你自己的最强模型。**
