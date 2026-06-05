"""
模型卡片生成器

为 Fusion-LLM 模型生成标准模型卡片（Markdown 格式），
包含模型信息、性能指标、使用方法等。
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import torch


@dataclass
class ModelCard:
    """模型卡片数据类"""
    model_name: str
    architecture: str
    parameters: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    
    # 可选字段
    description: str = ""
    training_data: str = ""
    performance_metrics: Dict[str, float] = None
    limitations: List[str] = None
    usage_examples: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.limitations is None:
            self.limitations = []
        if self.usage_examples is None:
            self.usage_examples = []
        if self.dependencies is None:
            self.dependencies = []


class ModelCardGenerator:
    """模型卡片生成器"""
    
    def __init__(
        self,
        model_name: str = "Fusion-LLM",
        output_dir: str = ".",
    ):
        """
        初始化生成器
        
        参数：
            model_name: 模型名称
            output_dir: 输出目录
        """
        self.model_name = model_name
        self.output_dir = output_dir
    
    def generate_from_model(
        self,
        model: torch.nn.Module,
        config: object,
        **kwargs,
    ) -> ModelCard:
        """
        从模型对象生成模型卡片
        
        参数：
            model: PyTorch 模型
            config: 模型配置
            **kwargs: 其他字段
            
        返回：
            ModelCard 对象
        """
        # 计算参数量
        param_count = sum(p.numel() for p in model.parameters())
        if param_count >= 1e9:
            param_str = f"{param_count / 1e9:.2f}B"
        elif param_count >= 1e6:
            param_str = f"{param_count / 1e6:.2f}M"
        elif param_count >= 1e3:
            param_str = f"{param_count / 1e3:.2f}K"
        else:
            param_str = f"{param_count}"
        
        # 提取配置信息
        card = ModelCard(
            model_name=self.model_name,
            architecture=model.__class__.__name__,
            parameters=param_str,
            vocab_size=getattr(config, 'vocab_size', 0),
            hidden_size=getattr(config, 'hidden_size', 0),
            num_layers=getattr(config, 'num_hidden_layers', 0),
            num_heads=getattr(config, 'num_attention_heads', 0),
            max_seq_len=getattr(config, 'max_position_embeddings', 0),
            **kwargs,
        )
        
        return card
    
    def generate_from_config(
        self,
        config: object,
        **kwargs,
    ) -> ModelCard:
        """
        从配置对象生成模型卡片（不需要加载模型）
        
        参数：
            config: 模型配置
            **kwargs: 其他字段
            
        返回：
            ModelCard 对象
        """
        card = ModelCard(
            model_name=self.model_name,
            architecture=config.model_type if hasattr(config, 'model_type') else "FusionModel",
            parameters="Unknown",  # 需要模型才能计算
            vocab_size=getattr(config, 'vocab_size', 0),
            hidden_size=getattr(config, 'hidden_size', 0),
            num_layers=getattr(config, 'num_hidden_layers', 0),
            num_heads=getattr(config, 'num_attention_heads', 0),
            max_seq_len=getattr(config, 'max_position_embeddings', 0),
            **kwargs,
        )
        
        return card
    
    def to_markdown(self, card: ModelCard) -> str:
        """
        将模型卡片转换为 Markdown 格式
        
        参数：
            card: ModelCard 对象
            
        返回：
            Markdown 字符串
        """
        lines = []
        
        # 标题
        lines.append(f"# {card.model_name}")
        lines.append("")
        
        # 描述
        if card.description:
            lines.append("## 描述")
            lines.append("")
            lines.append(card.description)
            lines.append("")
        
        # 模型信息表格
        lines.append("## 模型信息")
        lines.append("")
        lines.append("| 字段 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 架构 | {card.architecture} |")
        lines.append(f"| 参数量 | {card.parameters} |")
        lines.append(f"| 词汇表大小 | {card.vocab_size:,} |")
        lines.append(f"| 隐藏层大小 | {card.hidden_size} |")
        lines.append(f"| 层数 | {card.num_layers} |")
        lines.append(f"| 注意力头数 | {card.num_heads} |")
        lines.append(f"| 最大序列长度 | {card.max_seq_len} |")
        lines.append("")
        
        # 训练数据
        if card.training_data:
            lines.append("## 训练数据")
            lines.append("")
            lines.append(card.training_data)
            lines.append("")
        
        # 性能指标
        if card.performance_metrics:
            lines.append("## 性能指标")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for metric, value in card.performance_metrics.items():
                if isinstance(value, float):
                    lines.append(f"| {metric} | {value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value} |")
            lines.append("")
        
        # 使用方法
        if card.usage_examples:
            lines.append("## 使用方法")
            lines.append("")
            for i, example in enumerate(card.usage_examples, 1):
                lines.append(f"### 示例 {i}")
                lines.append("")
                lines.append("```python")
                lines.append(example)
                lines.append("```")
                lines.append("")
        
        # 依赖
        if card.dependencies:
            lines.append("## 依赖")
            lines.append("")
            for dep in card.dependencies:
                lines.append(f"- {dep}")
            lines.append("")
        
        # 限制
        if card.limitations:
            lines.append("## 限制")
            lines.append("")
            for limitation in card.limitations:
                lines.append(f"- {limitation}")
            lines.append("")
        
        # 底部信息
        lines.append("---")
        lines.append("")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 许可证")
        lines.append("")
        lines.append("Apache 2.0")
        lines.append("")
        
        return "\n".join(lines)
    
    def save(
        self,
        card: ModelCard,
        filename: str = "MODEL_CARD.md",
    ) -> str:
        """
        保存模型卡片到文件
        
        参数：
            card: ModelCard 对象
            filename: 文件名
            
        返回：
            保存的文件路径
        """
        import os
        markdown_content = self.to_markdown(card)
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return filepath


def generate_model_card(
    model: torch.nn.Module,
    config: object,
    output_path: str = "MODEL_CARD.md",
    **kwargs,
) -> str:
    """
    便捷函数：生成并保存模型卡片
    
    参数：
        model: PyTorch 模型
        config: 模型配置
        output_path: 输出文件路径
        **kwargs: 其他字段
        
    返回：
        保存的文件路径
    """
    import os
    
    output_dir = os.path.dirname(output_path) or "."
    filename = os.path.basename(output_path)
    
    generator = ModelCardGenerator(
        model_name=kwargs.get('model_name', 'Fusion-LLM'),
        output_dir=output_dir,
    )
    
    card = generator.generate_from_model(model, config, **kwargs)
    filepath = generator.save(card, filename)
    
    return filepath


if __name__ == "__main__":
    print("[ModelCard] 模型卡片生成器")
    print()
    print("用法：")
    print("  from evaluation.model_card import ModelCardGenerator")
    print("  generator = ModelCardGenerator('MyModel')")
    print("  card = generator.generate_from_model(model, config)")
    print("  filepath = generator.save(card)")
    print()
    print("或：")
    print("  from evaluation.model_card import generate_model_card")
    print("  filepath = generate_model_card(model, config)")
