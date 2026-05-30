"""
Fusion 模型架构

包含：
- fusion_mini.py: 极简可运行版本（用于验证流程）✅ 已实现
- fusion_model.py: 完整 Transformer 模型定义（待实现）
- sbla_attention.py: SBLA 注意力（滑动分块潜注意力）✅ 已实现
- thinking_dial.py: 动态推理强度调节器（Thinking Dial）（待实现）

使用方法：
    # 推荐：极简版本（已实现）
    from models import FusionMini, FusionMiniConfig
    
    # 或：直接导入
    from models.fusion_mini import FusionMini, FusionMiniConfig
    
    # SBLA 注意力
    from models.sbla_attention import SBLAttention
"""

# 极简可运行版本（已实现）
from .fusion_mini import FusionMini, FusionMiniConfig

# SBLA 注意力（已实现）
from .sbla_attention import SBLAttention

# 完整版本（暂时注释掉，因为依赖未完全实现）
# from .fusion_model import FusionModel, FusionConfig
# from .sbla_attention import SlidingBlockLatentAttention, FusionAttentionBlock
# from .thinking_dial import (
#     ThinkingDialProcessor,
#     ThinkingDialModel,
#     ThinkingConfig,
#     GRPOTrainer,
# )

__all__ = [
    # 极简版本（已实现）
    "FusionMini",
    "FusionMiniConfig",
    
    # SBLA 注意力（已实现）
    "SBLAttention",
    
    # 完整版本（待实现）
    # "FusionModel",
    # "FusionConfig",
    # "SlidingBlockLatentAttention",
    # "FusionAttentionBlock",
    # "ThinkingDialProcessor",
    # "ThinkingDialModel",
    # "ThinkingConfig",
    # "GRPOTrainer",
]
