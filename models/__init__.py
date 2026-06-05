"""
Fusion 模型架构

包含：
- fusion_mini.py: 极简可运行版本（用于验证流程）[OK] 已实现
- fusion_model.py: 完整 Transformer 模型定义（SBLA + Thinking Dial）[OK] 已实现
- sbla_attention.py: SBLA 注意力（滑动分块潜注意力）[OK] 已实现
- thinking_dial.py: 动态推理强度调节器（Thinking Dial）[OK] 已实现

使用方法：
    # 极简版本（字符级训练验证）
    from models.fusion_mini import FusionMini, FusionMiniConfig
    
    # 完整版本（Production）
    from models.fusion_model import FusionModel, FusionConfig
    from models.sbla_attention import SBLAttention
    from models.thinking_dial import ThinkingDialProcessor, ThinkingDialModel

    # 示例：创建完整模型
    config = FusionConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        block_size=128,
        latent_dim=32,
    )
    model = FusionModel(config)
    
    # 示例：SBLA 注意力
    attention = SBLAttention(
        hidden_size=512,
        num_heads=8,
        block_size=128,
        latent_dim=32,
    )
"""

# 极简可运行版本（字符级验证）
from .fusion_mini import FusionMini, FusionMiniConfig

# 完整可实例化版本
from .fusion_model import FusionModel, FusionConfig

# SBLA 注意力
from .sbla_attention import SBLAttention

# Thinking Dial
from .thinking_dial import (
    ThinkingDialProcessor,
    ThinkingDialModel,
    ThinkingConfig,
    GRPOTrainer,
    GRPOConfig,
    build_think_token,
    apply_thinking_control,
    extract_thinking_depth,
)

__all__ = [
    # 极简版本
    "FusionMini",
    "FusionMiniConfig",
    
    # 完整版本
    "FusionModel",
    "FusionConfig",
    
    # SBLA 注意力
    "SBLAttention",
    
    # Thinking Dial
    "ThinkingDialProcessor",
    "ThinkingDialModel",
    "ThinkingConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "build_think_token",
    "apply_thinking_control",
    "extract_thinking_depth",
]