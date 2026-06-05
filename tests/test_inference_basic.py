"""
快速推理测试 - 验证 Fusion-LLM 基本推理功能（无 emoji 版本）
"""
import sys
import torch
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


def test_basic_inference():
    """测试基本推理功能"""
    print("[TEST] 开始基本推理测试...")
    print()
    
    # 1. 创建配置（小配置，快速测试）
    print("[1] 创建模型配置...")
    config = FusionMiniConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    print(f"   词汇表大小: {config.vocab_size}")
    print(f"   隐藏层大小: {config.hidden_size}")
    print(f"   层数: {config.num_hidden_layers}")
    print()
    
    # 2. 创建模型
    print("[2] 创建模型...")
    model = FusionMini(config)
    model.eval()  # 评估模式
    param_count = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"   参数量: {param_count:.1f}K")
    print("   模型创建成功")
    print()
    
    # 3. 测试前向传播（无标签）
    print("[3] 测试前向传播（无标签）...")
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    print(f"   输出类型: {type(outputs)}")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
    print("   前向传播成功")
    print()
    
    # 4. 测试前向传播（有标签，计算损失）
    print("[4] 测试前向传播（有标签）...")
    labels = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    print(f"   Loss: {outputs['loss'].item():.4f}")
    print("   损失计算成功")
    print()
    
    # 5. 测试生成（少量 token）
    print("[5] 测试生成（5 个 token）...")
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids[:, :4],
            max_new_tokens=5,
            temperature=1.0,
            do_sample=False,  # 贪婪解码，确定性
        )
    print(f"   生成形状: {generated.shape}")
    print("   生成成功")
    print()
    
    print("[TEST] 基本推理测试通过")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 基本推理测试")
    print("=" * 60)
    print()
    
    try:
        success = test_basic_inference()
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
