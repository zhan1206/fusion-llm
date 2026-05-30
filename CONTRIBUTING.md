# 贡献指南

感谢你对 Fusion 项目的兴趣！这份文档将指导你如何参与贡献。

## 🚀 快速开始

### 1. Fork 和克隆

```bash
# Fork 本仓库（在 GitHub 上点击 Fork 按钮）

# 克隆你的 Fork
git clone https://github.com/YOUR_USERNAME/fusion-llm.git
cd fusion-llm

# 添加上游仓库
git remote add upstream https://github.com/zhan1206/fusion-llm.git
```

### 2. 创建开发环境

```bash
# 创建虚拟环境（推荐）
conda create -n fusion python=3.10
conda activate fusion

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -e ".[dev]"
```

### 3. 创建分支

```bash
git checkout -b feature/your-feature-name
```

---

## 🎯 如何贡献

### 报告 Bug

打开 [Issue](https://github.com/zhan1206/fusion-llm/issues)，并包含：

- 操作系统版本
- Python 版本
- 依赖版本（`pip freeze`）
- 重现步骤
- 期望行为 vs 实际行为

### 建议新功能

打开 [Discussion](https://github.com/zhan1206/fusion-llm/discussions) 讨论你的想法。

### 提交代码

1. **遵循代码规范**
   - 使用 Python 3.8+ 语法
   - 遵循 PEP 8 规范
   - 使用类型注解（type hints）
   - 添加文档字符串（docstrings）

2. **编写测试**
   - 为所有新功能编写单元测试
   - 确保测试覆盖率 >80%
   - 运行 `pytest tests/`

3. **提交信息规范**
   
   使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

   ```
   feat: 添加 DyQuant 混合精度量化工具
   ^
   |__ 类型：feat|fix|docs|style|refactor|test|chore
   ```

   示例：
   ```
   feat: 实现 SBLA 注意力机制的 CPU 版本
   fix: 修复 Thinking Dial 在 batch 生成时的维度错误
   docs: 更新训练指南，添加 24GB 显卡配置示例
   test: 为 bilingual_filter.py 添加单元测试
   ```

4. **提交 Pull Request**
   - 填写 PR 模板
   - 关联相关的 Issue（`Fixes #123`）
   - 等待代码审查

---

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_sbla_attention.py -v

# 检查测试覆盖率
pytest tests/ --cov=models --cov-report=html
```

### 编写测试

测试文件位于 `tests/` 目录，命名格式：`test_*.py`

示例：

```python
import torch
from models.sbla_attention import SlidingBlockLatentAttention

def test_sbla_forward():
    """测试 SBLA 注意力前向传播"""
    batch_size = 2
    seq_len = 1024
    d_model = 512
    
    attn = SlidingBlockLatentAttention(
        d_model=d_model,
        n_heads=8,
        block_size=512,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
```

---

## 📂 项目结构

```
fusion-llm/
├── models/              # 模型架构
│   ├── fusion_model.py      # 完整模型定义
│   ├── sbla_attention.py   # SBLA 注意力
│   └── thinking_dial.py    # Thinking Dial
├── train/               # 训练脚本
│   ├── lora_finetune.py    # LoRA/QLoRA
│   └── full_finetune.py    # 全参数微调
├── data_pipeline/       # 数据处理
│   └── bilingual_filter.py  # 双母语清洗
├── inference/           # 推理部署
│   └── ollama_deploy.py    # Ollama 部署
├── configs/             # 配置文件
│   └── ds_zero3.json       # DeepSpeed 配置
├── tests/               # 单元测试
├── docs/                # 文档
└── scripts/             # 工具脚本
```

---

## 🎨 代码风格

### Python 规范

- 使用 **Black** 格式化：`black models/ train/`
- 使用 **isort** 排序导入：`isort models/ train/`
- 使用 **Flake8** 检查：`flake8 models/ --max-line-length=120`

### 文档规范

- 使用 Google 风格 docstrings：

```python
def my_function(param1: int, param2: str) -> bool:
    """
    函数简短描述
    
    详细描述（可选）
    
    Args:
        param1: 参数1的描述
        param2: 参数2的描述
        
    Returns:
        返回值的描述
        
    Raises:
        ValueError: 参数错误时
    """
    pass
```

---

## 🐛 Debug 技巧

### 调试 SBLA 注意力

```python
# 在 models/sbla_attention.py 添加调试信息
import pdb; pdb.set_trace()

# 或使用 print
print(f"Q shape: {q.shape}")
print(f"Attention weights: {attn_weights.shape}")
```

### 调试训练问题

```bash
# 使用 --debug 模式（如果支持）
python train/lora_finetune.py --debug ...

# 检查显存使用
nvidia-smi -l 1

# 使用 TensorBoard
tensorboard --logdir=runs
```

---

## 📊 性能优化建议

### 模型训练

- 使用 **混合精度训练**（`--fp16` 或 `--bf16`）
- 使用 **梯度累积**（`--gradient_accumulation_steps 8`）
- 使用 **DeepSpeed ZeRO**（`--deepspeed configs/ds_zero3.json`）

### 推理优化

- 使用 **量化**（`q4_k_m` 或 `q8_0`）
- 使用 **KV 缓存**（`use_cache=True`）
- 使用 **批处理**（`batch_size > 1`）

---

## 💬 社区

- **讨论区**: [GitHub Discussions](https://github.com/zhan1206/fusion-llm/discussions)
- **Discord**: （待创建）
- **微信/QQ 群**: （待创建）

---

## 📜 行为准则

请阅读并遵守我们的 [行为准则](CODE_OF_CONDUCT.md)（待创建）。

---

## ❓ 需要帮助？

- 查看 [文档](docs/)
- 搜索已有 [Issues](https://github.com/zhan1206/fusion-llm/issues)
- 在 [Discussions](https://github.com/zhan1206/fusion-llm/discussions) 提问

---

**再次感谢你的贡献！** 🙏

让我们一起打造最强的开源大模型！ 🚀
