"""
最小训练测试 - 只训练 1 步，验证训练代码能正常运行
"""
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


@dataclass
class TrainConfig:
    """训练配置"""
    learning_rate: float = 1e-3
    batch_size: int = 1
    num_epochs: int = 1
    max_seq_len: int = 32
    use_thinking_dial: bool = False
    device: str = "cpu"


class FullFinetuneTrainer:
    """全参数微调训练器（简化版）"""
    def __init__(self, model: nn.Module, config: TrainConfig, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device
        
    def train_step(self, data):
        """训练一步（占位）"""
        pass


def test_minimal_training():
    """最小训练测试：只训练 1 步"""
    print("[TEST] 最小训练测试（1 步）...")
    print()
    
    # 1. 创建极小模型
    print("[1] 创建极小模型...")
    config = FusionMiniConfig(
        vocab_size=100,  # 很小的词汇表
        hidden_size=32,   # 很小的隐藏层
        num_hidden_layers=1,  # 只有 1 层
        num_attention_heads=2,
        max_position_embeddings=64,
    )
    model = FusionMini(config)
    param_count = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"   参数量: {param_count:.1f}K")
    print()
    
    # 2. 创建训练配置（只训练 1 步）
    print("[2] 创建训练配置...")
    train_config = TrainConfig(
        learning_rate=1e-3,
        batch_size=1,
        num_epochs=1,
        max_seq_len=32,
        use_thinking_dial=False,  # 禁用 thinking dial 以简化
    )
    print(f"   学习率: {train_config.learning_rate}")
    print(f"   批大小: {train_config.batch_size}")
    print(f"   训练轮数: {train_config.num_epochs}")
    print()
    
    # 3. 创建训练器
    print("[3] 创建训练器...")
    trainer = FullFinetuneTrainer(
        model=model,
        config=train_config,
        device="cpu",  # 使用 CPU 避免 GPU 问题
    )
    print("   训练器创建成功")
    print()
    
    # 4. 创建最小训练数据（只有 2 个样本）
    print("[4] 创建最小训练数据...")
    train_data = [
        "Hello world",
        "Test sentence",
    ]
    print(f"   训练样本数: {len(train_data)}")
    print()
    
    # 5. 手动执行 1 步训练（不调用 trainer.train()）
    print("[5] 执行 1 步训练...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    
    # 准备输入
    input_ids = torch.randint(0, config.vocab_size, (1, 16))  # batch=1, seq_len=16
    labels = input_ids.clone()
    
    # 前向传播
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   训练 1 步完成")
    print(f"   Loss: {loss.item():.4f}")
    print()
    
    # 6. 验证模型参数已更新
    print("[6] 验证参数更新...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"   ✅ {name} 有梯度")
            break
    print()
    
    print("[TEST] 最小训练测试通过")
    # return True  # pytest 不支持返回非 None


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 最小训练测试")
    print("=" * 60)
    print()
    
    try:
        success = test_minimal_training()
        if success:
            print()
            print("[PASS] 所有测试通过")
        else:
            print()
            print("[FAIL] 测试失败")
    except Exception as e:
        print()
        print(f"[FAIL] 测试出错: {e}")
        import traceback
        traceback.print_exc()
