"""
极简测试 - 用最小配置测试（应该很快）
"""
import sys
import torch
import time
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


def test_tiny():
    """用极小的配置测试"""
    print("[TEST] 极简测试（极小配置）...")
    print()
    
    # 1. 创建极小配置
    print("[1] 创建配置...")
    config = FusionMiniConfig(
        vocab_size=50,        # 极小词表
        hidden_size=32,       # 极小隐层
        num_hidden_layers=1,  # 1层
        num_attention_heads=1, # 1个注意力头
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
    model.eval()
    print("   模型创建成功")
    print()
    
    # 3. 创建极小输入
    print("[3] 创建输入...")
    input_ids = torch.randint(0, config.vocab_size, (1, 8))  # batch=1, seq_len=8
    print(f"   输入形状: {input_ids.shape}")
    print()
    
    # 4. 前向传播（计时）
    print("[4] 前向传播（计时）...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    elapsed = time.time() - start_time
    print(f"   前向传播完成，耗时: {elapsed:.2f}秒")
    print()
    
    # 5. 检查输出
    print("[5] 检查输出...")
    if isinstance(outputs, dict):
        print(f"   输出类型: dict")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
    print()
    
    print("[TEST] 极简测试通过")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 极简测试（极小配置）")
    print("=" * 60)
    print()
    
    try:
        success = test_tiny()
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
