# Fusion-LLM API 文档

## 模型 API

### FusionMini

`FusionMini` 是 Fusion-LLM 的迷你模型实现。

#### 类定义

```python
class FusionMini(nn.Module):
    """
    Fusion-LLM 迷你模型
    
    Args:
        config (FusionMiniConfig): 模型配置
    """
```

#### 方法

##### `__init__(config)`

初始化 FusionMini 模型。

**参数**：
- `config` (FusionMiniConfig): 模型配置对象

**示例**：
```python
from models.fusion_mini import FusionMini, FusionMiniConfig

config = FusionMiniConfig(
    vocab_size=1000,
    hidden_size=128,
    num_hidden_layers=2,
)
model = FusionMini(config)
```

##### `forward(input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, return_dict=True)`

模型前向传播。

**参数**：
- `input_ids` (torch.Tensor): 输入 token IDs，形状为 `(batch_size, sequence_length)`
- `attention_mask` (torch.Tensor, optional): 注意力掩码，形状为 `(batch_size, sequence_length)`
- `labels` (torch.Tensor, optional): 标签，形状为 `(batch_size, sequence_length)`
- `past_key_values` (tuple, optional): 过去的键值缓存
- `use_cache` (bool): 是否使用 KV 缓存
- `return_dict` (bool): 是否返回字典格式

**返回**：
- 如果 `return_dict=True`：返回字典，包含：
  - `loss` (torch.Tensor): 损失值（如果提供了 labels）
  - `logits` (torch.Tensor): 逻辑值，形状为 `(batch_size, sequence_length, vocab_size)`
  - `past_key_values` (tuple): 更新的键值缓存（如果 `use_cache=True`）
- 如果 `return_dict=False`：返回元组

**示例**：
```python
# 训练模式
outputs = model(
    input_ids=input_ids,
    labels=labels,
    return_dict=True,
)
loss = outputs["loss"]
logits = outputs["logits"]

# 推理模式（使用 KV 缓存）
outputs = model(
    input_ids=input_ids,
    use_cache=True,
    return_dict=True,
)
logits = outputs["logits"]
past_key_values = outputs["past_key_values"]
```

##### `generate(input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.95)`

生成文本（简易接口）。

**参数**：
- `input_ids` (torch.Tensor): 输入 token IDs
- `max_length` (int): 最大生成长度
- `temperature` (float): 温度参数
- `top_k` (int): Top-K 采样参数
- `top_p` (float): Top-P 采样参数

**返回**：
- `torch.Tensor`: 生成的 token IDs

**示例**：
```python
generated = model.generate(
    input_ids=input_ids,
    max_length=50,
    temperature=0.8,
)
```

##### `save_pretrained(save_directory)`

保存模型和配置。

**参数**：
- `save_directory` (str): 保存目录

**示例**：
```python
model.save_pretrained("output/my_model")
```

##### `from_pretrained(pretrained_model_name_or_path)`

从预训练路径加载模型。

**参数**：
- `pretrained_model_name_or_path` (str): 预训练模型路径

**返回**：
- `FusionMini`: 加载的模型

**示例**：
```python
model = FusionMini.from_pretrained("output/my_model")
```

---

### FusionMiniConfig

`FusionMiniConfig` 是 FusionMini 模型的配置类。

#### 类定义

```python
class FusionMiniConfig(PretrainedConfig):
    """
    FusionMini 模型配置
    
    Args:
        vocab_size (int): 词汇表大小
        hidden_size (int): 隐藏层大小
        num_hidden_layers (int): 隐藏层数量
        num_attention_heads (int): 注意力头数量
        intermediate_size (int): 中间层大小
        max_position_embeddings (int): 最大位置编码
        num_key_value_heads (int, optional): KV 头数量（用于 GQA）
        window_size (int, optional): SBLA 窗口大小
        ...
    """
```

#### 属性

| 属性 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `vocab_size` | int | 50257 | 词汇表大小 |
| `hidden_size` | int | 768 | 隐藏层大小 |
| `num_hidden_layers` | int | 12 | 隐藏层数量 |
| `num_attention_heads` | int | 12 | 注意力头数量 |
| `intermediate_size` | int | 3072 | 中间层大小 |
| `max_position_embeddings` | int | 1024 | 最大位置编码 |
| `num_key_value_heads` | int | None | KV 头数量（GQA） |
| `window_size` | int | 16 | SBLA 窗口大小 |

#### 方法

##### `__init__(**kwargs)`

初始化配置。

**示例**：
```python
config = FusionMiniConfig(
    vocab_size=1000,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
)
```

---

## 注意力 API

### SBLAttention

`SBLAttention` 是 SBLA（Sliding Block Latent Attention）注意力实现。

#### 类定义

```python
class SBLAttention(nn.Module):
    """
    SBLA 注意力层
    
    Args:
        hidden_size (int): 隐藏层大小
        num_heads (int): 注意力头数量
        window_size (int): 窗口大小
        num_key_value_heads (int, optional): KV 头数量（用于 GQA）
    """
```

#### 方法

##### `__init__(hidden_size, num_heads, window_size, num_key_value_heads=None)`

初始化 SBLA 注意力层。

##### `forward(hidden_states, attention_mask=None, past_key_value=None, use_cache=False, output_attentions=False)`

前向传播。

**参数**：
- `hidden_states` (torch.Tensor): 隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`
- `attention_mask` (torch.Tensor, optional): 注意力掩码
- `past_key_value` (tuple, optional): 过去的键值缓存
- `use_cache` (bool): 是否使用 KV 缓存
- `output_attentions` (bool): 是否输出注意力权重

**返回**：
- `tuple`: (output, past_key_value, attentions)

**示例**：
```python
from models.sbla_attention import SBLAttention

attention = SBLAttention(
    hidden_size=128,
    num_heads=2,
    window_size=16,
)

hidden_states = torch.randn(1, 32, 128)
output, past_key_value, _ = attention(hidden_states)
print(output.shape)  # torch.Size([1, 32, 128])
```

---

## Thinking Dial API

### ThinkingDialProcessor

`ThinkingDialProcessor` 是 Thinking Dial（动态推理强度控制）处理器。

#### 类定义

```python
class ThinkingDialProcessor:
    """
    Thinking Dial 处理器
    
    用于处理 think token，动态控制推理强度。
    """
```

#### 方法

##### `process(text)`

处理文本，注入 think token。

**参数**：
- `text` (str): 输入文本（可能包含 `<|think_depth_N|>`）

**返回**：
- `str`: 处理后的文本

**示例**：
```python
from models.thinking_dial import ThinkingDialProcessor

processor = ThinkingDialProcessor()

text = "<|think_depth_2|> 这是一个需要深入思考的问题。"
processed_text = processor.process(text)
print(processed_text)  # 处理后的文本
```

##### `get_think_depth(text)`

获取 think token 的深度。

**参数**：
- `text` (str): 输入文本

**返回**：
- `int`: think 深度（0-3）

**示例**：
```python
depth = processor.get_think_depth(text)
print(depth)  # 2
```

---

## 量化 API

### DyQuant

`DyQuant` 是动态混合精度量化器（4/8/16-bit）。

#### 类定义

```python
class DyQuant:
    """
    动态混合精度量化器
    
    支持 4-bit、8-bit、16-bit 量化。
    """
```

#### 方法

##### `quantize(model, bits=8)`

量化模型。

**参数**：
- `model` (nn.Module): 要量化的模型
- `bits` (int): 量化位数（4/8/16）

**返回**：
- `nn.Module`: 量化后的模型

**示例**：
```python
from inference.dyquant import DyQuant

quantizer = DyQuant()
quantized_model = quantizer.quantize(model, bits=8)
```

##### `save(model, path)`

保存量化模型。

**参数**：
- `model` (nn.Module): 量化模型
- `path` (str): 保存路径

**示例**：
```python
quantizer.save(quantized_model, "output/quantized_model")
```

##### `load(path)`

加载量化模型。

**参数**：
- `path` (str): 模型路径

**返回**：
- `nn.Module`: 加载的量化模型

**示例**：
```python
loaded_model = quantizer.load("output/quantized_model")
```

---

## 训练 API

### FullFinetuner

`FullFinetuner` 是全量微调器。

#### 类定义

```python
class FullFinetuner:
    """
    全量微调器
    
    用于全量微调 Fusion-LLM 模型。
    """
```

#### 方法

##### `train(model, train_dataset, eval_dataset=None, num_epochs=3, batch_size=4)`

训练模型。

**参数**：
- `model` (nn.Module): 要训练的模型
- `train_dataset` (Dataset): 训练数据集
- `eval_dataset` (Dataset, optional): 评估数据集
- `num_epochs` (int): 训练轮数
- `batch_size` (int): 批次大小

**示例**：
```python
from train.full_finetune import FullFinetuner

finetuner = FullFinetuner()

finetuner.train(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    num_epochs=3,
    batch_size=4,
)
```

---

## 评估 API

### ModelEvaluator

`ModelEvaluator` 是模型评估器。

#### 类定义

```python
class ModelEvaluator:
    """
    模型评估器
    
    用于评估模型性能（Perplexity、Loss、Accuracy 等）。
    """
```

#### 方法

##### `evaluate(model, eval_data, metrics=["perplexity", "loss", "accuracy"])`

评估模型。

**参数**：
- `model` (nn.Module): 要评估的模型
- `eval_data` (Dataset): 评估数据集
- `metrics` (List[str]): 评估指标列表

**返回**：
- `EvaluationMetrics`: 评估结果

**示例**：
```python
from evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()

metrics = evaluator.evaluate(
    model=model,
    eval_data=eval_dataset,
    metrics=["perplexity", "loss", "accuracy"],
)

print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"Loss: {metrics.loss:.4f}")
print(f"Accuracy: {metrics.accuracy:.4f}")
```

---

## 部署 API

### OllamaDeployer

`OllamaDeployer` 是 Ollama 部署器。

#### 类定义

```python
class OllamaDeployer:
    """
    Ollama 部署器
    
    用于将 Fusion-LLM 模型部署到 Ollama。
    """
```

#### 方法

##### `deploy(model_path, output_path)`

部署模型到 Ollama。

**参数**：
- `model_path` (str): 模型路径
- `output_path` (str): 输出路径

**示例**：
```python
from inference.ollama_deploy_v2 import OllamaDeployer

deployer = OllamaDeployer()

deployer.deploy(
    model_path="output/real_model",
    output_path="output/ollama_model",
)
```

---

## 数据集 API

### TextDataset

`TextDataset` 是文本数据集。

#### 类定义

```python
class TextDataset(Dataset):
    """
    文本数据集
    
    用于加载文本数据并进行编码。
    """
```

#### 方法

##### `__init__(tokenizer, file_path, block_size=128)`

初始化数据集。

**参数**：
- `tokenizer`: Tokenizer
- `file_path` (str): 文件路径
- `block_size` (int): 块大小

**示例**：
```python
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        # ...
        pass
```

---

## 完整示例

### 训练完整流程

```python
import torch
from models.fusion_mini import FusionMini, FusionMiniConfig
from train.full_finetune import FullFinetuner
from evaluation.metrics import ModelEvaluator

# 1. 创建模型
config = FusionMiniConfig(
    vocab_size=1000,
    hidden_size=128,
    num_hidden_layers=2,
)
model = FusionMini(config)

# 2. 创建训练器
finetuner = FullFinetuner()

# 3. 训练
finetuner.train(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    num_epochs=3,
    batch_size=4,
)

# 4. 评估
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(
    model=model,
    eval_data=eval_dataset,
)

print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"Loss: {metrics.loss:.4f}")
```

### 推理完整流程

```python
import torch
from models.fusion_mini import FusionMini

# 1. 加载模型
model = FusionMini.from_pretrained("output/real_model")
model.eval()

# 2. 创建输入
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# 3. 推理
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        return_dict=True,
    )
    
    logits = outputs["logits"]
    print(f"Logits shape: {logits.shape}")

# 4. 生成
generated = model.generate(
    input_ids=input_ids,
    max_length=50,
)
print(f"Generated: {generated}")
```

---

## 常见问题

### 1. 如何自定义模型配置？

使用 `FusionMiniConfig`：

```python
config = FusionMiniConfig(
    vocab_size=2000,       # 更大的词汇表
    hidden_size=256,        # 更大的隐藏层
    num_hidden_layers=4,    # 更深的模型
    num_attention_heads=4,  # 更多的注意力头
)
```

### 2. 如何使用 GQA？

在配置中设置 `num_key_value_heads`：

```python
config = FusionMiniConfig(
    num_attention_heads=8,   # 8 个查询头
    num_key_value_heads=2,    # 2 个 KV 头（GQA）
)
```

### 3. 如何启用 KV 缓存？

在推理时使用 `use_cache=True`：

```python
outputs = model(
    input_ids=input_ids,
    use_cache=True,
    return_dict=True,
)
past_key_values = outputs["past_key_values"]
```

### 4. 如何量化模型？

使用 `DyQuant`：

```python
from inference.dyquant import DyQuant

quantizer = DyQuant()
quantized_model = quantizer.quantize(model, bits=8)
```

---

## 许可证

Fusion-LLM API 文档采用 Apache 2.0 许可证。
