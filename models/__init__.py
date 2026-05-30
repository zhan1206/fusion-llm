"""
Fusion 模型架构

包含：
- fusion_model.py: 完整 Transformer 模型定义
- sbla_attention.py: 滑动分块潜注意力（SBLA）
- thinking_dial.py: 动态推理强度调节器（Thinking Dial）

使用方法：
    from models import FusionModel, FusionConfig
    from models.sbla_attention import SlidingBlockLatentAttention
    from models.thinking_dial import ThinkingDialProcessor
"""

from .fusion_model import FusionModel, FusionConfig
from .sbla_attention import SlidingBlockLatentAttention, FusionAttentionBlock
from .thinking_dial import (
    ThinkingDialProcessor,
    ThinkingDialModel,
    ThinkingConfig,
    GRPOTrainer,
)

__all__ = [
    # 主要模型
    "FusionModel",
    "FusionConfig",
    
    # SBLA 注意力
    "SlidingBlockLatentAttention",
    "FusionAttentionBlock",
    
    # Thinking Dial
    "ThinkingDialProcessor",
    "ThinkingDialModel",
    "ThinkingConfig",
    "GRPOTrainer",
]
