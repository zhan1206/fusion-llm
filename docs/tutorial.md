# Fusion-LLM 使用教程

## 简介

Fusion-LLM 是一个开源大语言模型项目，采用 Apache 2.0 许可证，核心理念为用户主权、纯本地训练推理。

### 核心特性
- **SBLA 注意力**：滑动分块潜注意力（Sliding Block Latent Attention）
- **Thinking Dial**：动态推理强度控制
- **纯本地**：无需云端，完全本地训练和推理
- **用户主权**：数据完全由用户控制

---

## 安装

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (可选，用于 GPU 加速)

### 安装步骤

#### 1. 克隆仓库
```bash
git clone https://github.com/zhan1206/fusion-llm.git
cd fusion-llm
```

#### 2. 安装依赖
```bash
pip install -r requirements.txt
```

#### 3. 验证安装
```bash
python tests/test_tiny.py
```

如果看到 `[PASS] 测试通过`，说明安装成功！

---

## 快速开始

### 1. 最小训练测试（验证安装）

运行最小训练测试（1-2 步，快速验证训练功能）：

```bash
python train/test_train_mini.py
```

预期输出：
```
[TRAIN] 开始最小训练（1-2 步）...
[5] 训练 2 步...
   Step 1: Loss = 4.5879
   Step 2: Loss = 4.5768
   训练完成
[6] 验证损失下降...
   [PASS] Loss 下降: 4.5879 -> 4.5768
   训练有效！

[PASS] 训练测试通过
```

### 2. 小训练测试（10 步）

运行小训练测试（10 步，验证损失持续下降）：

```bash
python train/train_10steps.py
```

预期输出：
```
[TRAIN] 开始小训练（10 步）...
[5] 训练 10 步...
   Step  1: Loss = 6.9452
   Step  2: Loss = 6.8520
   ...
   Step 10: Loss = 6.3993
   训练完成
[6] 验证损失下降...
   [PASS] Loss 持续下降
   训练有效！

[PASS] 训练测试通过
```

### 3. 实际模型训练（100 步）

运行实际模型训练（100 步，生成模型权重）：

```bash
python train/train_real.py
```

预期输出：
```
[TRAIN] 开始实际模型训练（100 步）...
[6] 训练 100 步...
   Step  10: Loss = 3.5885 (Avg: 4.0321)
   ...
   Step 100: Loss = 1.7501 (Avg: 1.9562)
   训练完成
[7] 验证损失下降...
   [PASS] Loss 持续下降
   训练有效！

[8] 保存模型...
   模型保存路径: output/real_model
   模型保存成功

[PASS] 训练测试通过
```

训练完成后，模型权重将保存到 `output/real_model/` 目录。

---

## 模型推理

### 1. 基本推理测试

运行基本推理测试：

```bash
python tests/test_inference_basic.py
```

### 2. 使用训练好的模型进行推理

```python
import torch
from models.fusion_mini import FusionMini, FusionMiniConfig

# 加载模型
model = FusionMini.from_pretrained("output/real_model")
model.eval()

# 创建输入
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# 推理
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        return_dict=True,
    )
    
    logits = outputs["logits"]
    print(f"Logits shape: {logits.shape}")
```

---

## 高级功能

### 1. Thinking Dial（动态推理强度控制）

Thinking Dial 允许动态控制模型的推理强度。

```python
from models.thinking_dial import ThinkingDialProcessor

# 创建处理器
processor = ThinkingDialProcessor()

# 注入 think token
text = "<|think_depth_2|> 这是一个需要深入思考的问题。"
processed_text = processor.process(text)

print(processed_text)
```

### 2. SBLA 注意力

SBLA（Sliding Block Latent Attention）是 Fusion-LLM 的核心注意力机制。

```python
from models.sbla_attention import SBLAttention

# 创建 SBLA 注意力层
attention = SBLAttention(
    hidden_size=128,
    num_heads=2,
    window_size=16,
)

# 前向传播
hidden_states = torch.randn(1, 32, 128)
output = attention(hidden_states)
print(f"Output shape: {output.shape}")
```

### 3. 动态量化（DyQuant）

DyQuant（Dynamic Quantization）提供动态混合精度量化（4/8/16-bit）。

```python
from inference.dyquant import DyQuant

# 创建量化器
quantizer = DyQuant()

# 量化模型
quantized_model = quantizer.quantize(model, bits=8)

# 保存量化模型
quantizer.save(quantized_model, "output/quantized_model")
```

---

## 训练配置

### 1. 使用配置文件

创建配置文件 `configs/my_config.json`：

```json
{
    "vocab_size": 1000,
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 256,
    "max_position_embeddings": 64
}
```

使用配置文件训练：

```bash
python train/full_finetune.py --config configs/my_config.json
```

### 2. LoRA 微调

使用 LoRA 进行参数高效微调：

```bash
python train/lora_finetune.py \
    --model_name_or_path output/real_model \
    --lora_r 8 \
    --lora_alpha 16 \
    --train_epochs 3
```

---

## 评估与指标

### 1. 使用评估指标

```python
from evaluation.metrics import EvaluationMetrics, ModelEvaluator

# 创建评估器
evaluator = ModelEvaluator()

# 评估模型
metrics = evaluator.evaluate(
    model=model,
    eval_data=eval_dataset,
    metrics=["perplexity", "loss", "accuracy"],
)

print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"Loss: {metrics.loss:.4f}")
print(f"Accuracy: {metrics.accuracy:.4f}")
```

### 2. 生成模型卡片

```bash
python evaluation/model_card.py \
    --model_path output/real_model \
    --output_path output/model_card.json
```

---

## 部署

### 1. Ollama 部署

使用 Ollama 部署模型：

```bash
python inference/ollama_deploy_v2.py \
    --model_path output/real_model \
    --output_path output/ollama_model
```

### 2. 量化部署

使用动态量化减小模型大小：

```bash
python inference/dyquant.py \
    --model_path output/real_model \
    --bits 8 \
    --output_path output/quantized_model
```

---

## 常见问题

### 1. 训练时 Loss 不下降？

**可能原因**：
- 学习率太大或太小
- 数据量太少
- 模型太小

**解决方案**：
- 调整学习率（尝试 1e-4 到 5e-4）
- 增加训练数据量
- 增大模型配置（hidden_size、num_layers）

### 2. 推理时出现 NaN？

**可能原因**：
- 注意力掩码错误
- 梯度爆炸

**解决方案**：
- 检查注意力掩码格式
- 使用梯度裁剪（`torch.nn.utils.clip_grad_norm_`）

### 3. 训练速度太慢？

**可能原因**：
- SBLA 注意力计算量大
- 没有使用 GPU

**解决方案**：
- 使用更小的配置（hidden_size=32, num_layers=1）
- 使用 GPU 训练
- 启用混合精度训练（FP16/BF16）

---

## 贡献

欢迎贡献！请查看 `CONTRIBUTING.md` 了解详情。

---

## 许可证

Fusion-LLM 采用 Apache 2.0 许可证。查看 `LICENSE` 文件了解详情。
