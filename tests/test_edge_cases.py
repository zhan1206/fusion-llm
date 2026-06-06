"""
边缘情况测试 - 验证模型在边缘情况下的行为
"""
import sys
import torch
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


def test_single_token():
    """测试单个 token"""
    print("[TEST] 测试单个 token...")
    
    config = FusionMiniConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
    )
    
    model = FusionMini(config)
    model.eval()
    
    # 单个 token
    input_ids = torch.tensor([[1]])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs["logits"]
        
        # 检查形状
        expected_shape = (1, 1, config.vocab_size)
        assert logits.shape == expected_shape, f"形状错误: {logits.shape} != {expected_shape}"
        
        print(f"   输入形状: {input_ids.shape}")
        print(f"   输出形状: {logits.shape}")
        print("   [PASS] 单个 token 测试通过")
        # test passes if no exception


def test_long_sequence():
    """测试长序列"""
    print("[TEST] 测试长序列...")
    
    config = FusionMiniConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        max_position_embeddings=64,
    )
    
    model = FusionMini(config)
    model.eval()
    
    # 长序列（达到最大长度）
    seq_len = config.max_position_embeddings
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs["logits"]
        
        # 检查形状
        expected_shape = (1, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"形状错误: {logits.shape} != {expected_shape}"
        
        print(f"   输入形状: {input_ids.shape}")
        print(f"   输出形状: {logits.shape}")
        print("   [PASS] 长序列测试通过")
        # test passes if no exception


def test_all_zeros():
    """测试全零输入"""
    print("[TEST] 测试全零输入...")
    
    config = FusionMiniConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
    )
    
    model = FusionMini(config)
    model.eval()
    
    # 全零输入（padding token）
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs["logits"]
        
        # 检查形状
        expected_shape = (1, 8, config.vocab_size)
        assert logits.shape == expected_shape, f"形状错误: {logits.shape} != {expected_shape}"
        
        print(f"   输入形状: {input_ids.shape}")
        print(f"   输出形状: {logits.shape}")
        print("   [PASS] 全零输入测试通过")
        # test passes if no exception


def test_nan_detection():
    """测试 NaN 检测"""
    print("[TEST] 测试 NaN 检测...")
    
    config = FusionMiniConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
    )
    
    model = FusionMini(config)
    model.train()
    
    # 创建输入
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        return_dict=True,
    )
    
    loss = outputs["loss"]
    logits = outputs["logits"]
    
    # 检查 NaN
    has_nan = torch.isnan(loss) or torch.isnan(logits).any()
    
    assert not has_nan, "检测到 NaN in loss or logits"
    print("   [PASS] 未检测到 NaN")


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 边缘情况测试")
    print("=" * 60)
    print()
    
    results = []
    
    try:
        results.append(("单个 token", test_single_token()))
    except Exception as e:
        print(f"   [FAIL] 单个 token 测试出错: {e}")
        results.append(("单个 token", False))
    
    try:
        results.append(("长序列", test_long_sequence()))
    except Exception as e:
        print(f"   [FAIL] 长序列测试出错: {e}")
        results.append(("长序列", False))
    
    try:
        results.append(("全零输入", test_all_zeros()))
    except Exception as e:
        print(f"   [FAIL] 全零输入测试出错: {e}")
        results.append(("全零输入", False))
    
    try:
        results.append(("NaN 检测", test_nan_detection()))
    except Exception as e:
        print(f"   [FAIL] NaN 检测测试出错: {e}")
        results.append(("NaN 检测", False))
    
    # 打印摘要
    print()
    print("=" * 60)
    print("测试摘要")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print()
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print("[ALL PASS] 所有测试通过")
        sys.exit(0)
    else:
        print("[SOME FAIL] 部分测试失败")
        sys.exit(1)
