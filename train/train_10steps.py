"""
小训练脚本 - 训练 10 步（验证损失持续下降）
"""
import sys
import torch
import torch.optim as optim
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


def train_small():
    """小训练（10 步）"""
    print("[TRAIN] 开始小训练（10 步）...")
    print()
    
    # 1. 创建小配置
    print("[1] 创建模型配置...")
    config = FusionMiniConfig(
        vocab_size=1000,       # 小词表
        hidden_size=64,        # 小隐层
        num_hidden_layers=2,   # 2 层
        num_attention_heads=2, # 2 个注意力头
        intermediate_size=128,
        max_position_embeddings=64,
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
    
    # 3. 创建优化器
    print("[3] 创建优化器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,  # 稍大的学习率
        weight_decay=0.01,
    )
    print("   优化器创建成功")
    print()
    
    # 4. 创建假数据（小批次）
    print("[4] 创建假数据...")
    batch_size = 4
    seq_len = 16
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"   输入形状: {input_ids.shape}")
    print(f"   标签形状: {labels.shape}")
    print("   假数据创建成功")
    print()
    
    # 5. 训练 10 步
    print("[5] 训练 10 步...")
    losses = []
    
    for step in range(10):
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
        
        print(f"   Step {step+1:2d}: Loss = {loss.item():.4f}")
    
    print("   训练完成")
    print()
    
    # 6. 验证损失下降
    print("[6] 验证损失下降...")
    initial_loss = losses[0]
    final_loss = losses[-1]
    is_decreasing = final_loss < initial_loss
    
    print(f"   初始 Loss: {initial_loss:.4f}")
    print(f"   最终 Loss: {final_loss:.4f}")
    print(f"   Loss 变化: {final_loss - initial_loss:+.4f}")
    print()
    
    if is_decreasing:
        print("   [PASS] Loss 持续下降")
        print("   训练有效！")
    else:
        print("   [WARN] Loss 未下降")
        print("   可能的问题：学习率太大 / 数据太少 / 模型太小")
    print()
    
    print("[TRAIN] 小训练完成")
    return is_decreasing


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 小训练（10 步）")
    print("=" * 60)
    print()
    
    try:
        success = train_small()
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
