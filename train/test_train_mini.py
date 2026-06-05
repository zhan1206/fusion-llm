"""
最小训练脚本 - 只训练 1-2 步（验证训练流程可行）
"""
import sys
import torch
import torch.optim as optim  # 正确：AdamW 在 torch.optim 中
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


def train_mini():
    """最小训练（1-2 步）"""
    print("[TRAIN] 开始最小训练（1-2 步）...")
    print()
    
    # 1. 创建极小配置
    print("[1] 创建模型配置...")
    config = FusionMiniConfig(
        vocab_size=100,        # 极小词表
        hidden_size=32,        # 极小隐层
        num_hidden_layers=1,  # 1 层
        num_attention_heads=1, # 1 个注意力头
        intermediate_size=64,
        max_position_embeddings=32,
    )
    print(f"   词汇表大小: {config.vocab_size}")
    print(f"   隐藏层大小: {config.hidden_size}")
    print(f"   层数: {config.num_hidden_layers}")
    print()
    
    # 2. 创建模型
    print("[2] 创建模型...")
    model = FusionMini(config)
    model.train()  # 训练模式
    param_count = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"   参数量: {param_count:.1f}K")
    print("   模型创建成功")
    print()
    
    # 3. 创建优化器（正确：torch.optim.AdamW）
    print("[3] 创建优化器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )
    print("   优化器创建成功")
    print()
    
    # 4. 创建假数据（极小批次）
    print("[4] 创建假数据...")
    batch_size = 2
    seq_len = 8
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"   输入形状: {input_ids.shape}")
    print(f"   标签形状: {labels.shape}")
    print("   假数据创建成功")
    print()
    
    # 5. 训练 2 步
    print("[5] 训练 2 步...")
    losses = []
    
    for step in range(2):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
        )
        
        loss = outputs["loss"]
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        print(f"   Step {step+1}: Loss = {loss.item():.4f}")
    
    print("   训练完成")
    print()
    
    # 6. 验证损失下降
    print("[6] 验证损失下降...")
    if losses[1] < losses[0]:
        print(f"   [PASS] Loss 下降: {losses[0]:.4f} -> {losses[1]:.4f}")
        print("   训练有效！")
    else:
        print(f"   [WARN] Loss 未下降: {losses[0]:.4f} -> {losses[1]:.4f}")
        print("   可能的问题：学习率太小 / 数据太少")
    print()
    
    print("[TRAIN] 最小训练完成")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 最小训练（1-2 步）")
    print("=" * 60)
    print()
    
    try:
        success = train_mini()
        if success:
            print()
            print("[PASS] 训练测试通过")
    except Exception as e:
        print()
        print(f"[FAIL] 训练测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)
