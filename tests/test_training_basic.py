"""
快速训练测试 - 验证 Fusion-LLM 基本训练功能（无 DeepSpeed 依赖）

只测试：
1. 模型能否正确计算损失
2. 反向传播能否运行
3. 优化器能否更新参数

不使用 DeepSpeed / LoRA / 完整训练脚本
"""
import sys
import torch
import torch.optim as optim  # 正确：AdamW 在 torch.optim 中
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


def test_basic_training():
    """测试基本训练功能（无 DeepSpeed）"""
    print("[TEST] 开始基本训练测试...")
    print()
    
    # 1. 创建极小配置（快速测试）
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
    
    # 4. 创建假数据
    print("[4] 创建假数据...")
    batch_size = 2
    seq_len = 8
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"   输入形状: {input_ids.shape}")
    print(f"   标签形状: {labels.shape}")
    print("   假数据创建成功")
    print()
    
    # 5. 前向传播 + 反向传播（单步）
    print("[5] 前向传播 + 反向传播（单步）...")
    
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        return_dict=True,
    )
    
    loss = outputs["loss"]
    print(f"   Loss: {loss.item():.4f}")
    print("   前向传播成功")
    print()
    
    # 反向传播
    loss.backward()
    print("   反向传播成功")
    print()
    
    # 更新参数
    optimizer.step()
    print("   参数更新成功")
    print()
    
    # 6. 验证参数已更新
    print("[6] 验证参数已更新...")
    param_before = list(model.parameters())[0].clone()
    
    # 再跑一步
    optimizer.zero_grad()
    outputs2 = model(input_ids=input_ids, labels=labels, return_dict=True)
    loss2 = outputs2["loss"]
    loss2.backward()
    optimizer.step()
    
    param_after = list(model.parameters())[0]
    
    is_different = not torch.allclose(param_before, param_after)
    print(f"   参数已更新: {is_different}")
    print()
    
    if not is_different:
        print("[WARN] 参数未更新！可能有问题")
        return False
    
    print("[TEST] 基本训练测试通过")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 基本训练测试（无 DeepSpeed）")
    print("=" * 60)
    print()
    
    try:
        success = test_basic_training()
        if success:
            print()
            print("[PASS] 测试通过")
    except Exception as e:
        print()
        print(f"[FAIL] 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)
