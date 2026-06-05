"""
训练优化：混合精度训练（AMP）+ 梯度累积
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

sys.path.insert(0, '.')


class AMPTrainer:
    """
    混合精度训练器（Automatic Mixed Precision）
    
    特性：
    - 自动混合精度（FP16/FP32）
    - 梯度缩放（GradScaler）
    - 支持 CPU 和 GPU
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        
        # 梯度缩放器（仅 CUDA）
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # 训练状态
        self.step_count = 0
        self.accum_count = 0
    
    def train_step(self, loss: torch.Tensor):
        """
        单步训练（含梯度累积）
        
        Args:
            loss: 损失值（已除以 grad_accum_steps）
        """
        # 缩放损失（梯度累积）
        scaled_loss = loss / self.grad_accum_steps
        
        if self.use_amp:
            # 混合精度反向传播
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.accum_count += 1
        
        # 达到累积步数时更新参数
        if self.accum_count >= self.grad_accum_steps:
            if self.use_amp:
                # 梯度裁剪（先 unscale）
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # 更新参数
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.step_count += 1
            self.accum_count = 0
    
    def state_dict(self):
        """返回训练状态"""
        return {
            'step_count': self.step_count,
            'accum_count': self.accum_count,
            'use_amp': self.use_amp,
            'grad_accum_steps': self.grad_accum_steps,
        }


class GradientAccumulator:
    """
    梯度累积器（独立于训练器的轻量级版本）
    
    用于在显存不足时模拟更大的 batch size
    """
    
    def __init__(self, accum_steps: int = 4):
        self.accum_steps = max(accum_steps, 1)
        self.current_step = 0
    
    def should_update(self) -> bool:
        """是否应该更新参数"""
        self.current_step += 1
        if self.current_step >= self.accum_steps:
            self.current_step = 0
            return True
        return False
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失以抵消累积效应"""
        return loss / self.accum_steps


def create_training_config(
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    max_steps: int = 1000,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
):
    """
    创建训练配置
    
    Args:
        batch_size: 实际 batch size
        grad_accum_steps: 梯度累积步数（有效 batch = batch_size * grad_accum_steps）
        learning_rate: 学习率
        warmup_steps: 预热步数
        max_steps: 最大训练步数
        use_amp: 是否使用混合精度
        max_grad_norm: 最大梯度范数
        weight_decay: 权重衰减
    
    Returns:
        dict: 训练配置
    """
    return {
        'batch_size': batch_size,
        'grad_accum_steps': grad_accum_steps,
        'effective_batch_size': batch_size * grad_accum_steps,
        'learning_rate': learning_rate,
        'warmup_steps': warmup_steps,
        'max_steps': max_steps,
        'use_amp': use_amp,
        'max_grad_norm': max_grad_norm,
        'weight_decay': weight_decay,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 训练优化测试（混合精度 + 梯度累积）")
    print("=" * 60)
    print()
    
    # 创建模型
    print("[1] 创建模型...")
    from models.fusion_mini import FusionMini, FusionMiniConfig
    config = FusionMiniConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
    model = FusionMini(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    print("   模型已创建")
    print()
    
    # 测试 AMPTrainer
    print("[2] 测试 AMPTrainer...")
    trainer = AMPTrainer(
        model=model,
        optimizer=optimizer,
        use_amp=False,  # CPU 上不用 AMP
        grad_accum_steps=4,
        max_grad_norm=1.0,
    )
    
    # 模拟训练
    for i in range(12):
        input_ids = torch.randint(0, 100, (2, 64))
        attention_mask = torch.ones(2, 64)
        labels = torch.randint(0, 100, (2, 64))
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        trainer.train_step(loss)
    
    state = trainer.state_dict()
    print(f"   训练步数: {state['step_count']}")
    print(f"   累积步数: {state['accum_count']}")
    print(f"   使用 AMP: {state['use_amp']}")
    print(f"   梯度累积: {state['grad_accum_steps']} 步")
    assert state['step_count'] == 3, f"Expected 3 steps, got {state['step_count']}"
    print("   AMPTrainer 测试通过")
    print()
    
    # 测试 GradientAccumulator
    print("[3] 测试 GradientAccumulator...")
    accum = GradientAccumulator(accum_steps=4)
    updates = []
    for i in range(16):
        if accum.should_update():
            updates.append(i + 1)
    
    print(f"   更新次数: {len(updates)}")
    print(f"   更新步数: {updates}")
    assert len(updates) == 4, f"Expected 4 updates, got {len(updates)}"
    print("   GradientAccumulator 测试通过")
    print()
    
    # 测试训练配置
    print("[4] 测试训练配置...")
    train_config = create_training_config(
        batch_size=4,
        grad_accum_steps=8,
        learning_rate=1e-4,
        use_amp=True,
    )
    print(f"   有效 batch size: {train_config['effective_batch_size']}")
    assert train_config['effective_batch_size'] == 32
    print("   训练配置测试通过")
    print()
    
    print("[PASS] 训练优化测试全部通过")
    sys.exit(0)
