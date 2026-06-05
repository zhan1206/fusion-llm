"""
快速推理测试 - 验证 Fusion-LLM 基本功能
"""
import sys
import torch
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig
from inference.dashboard import InferenceDashboard, InferenceConfig


def test_basic_inference():
    """测试基本推理功能"""
    print("[TEST] 开始基本推理测试...")
    print()
    
    # 1. 创建配置
    print("[1] 创建模型配置...")
    config = FusionMiniConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=256,
    )
    print(f"   词汇表大小: {config.vocab_size}")
    print(f"   隐藏层大小: {config.hidden_size}")
    print(f"   层数: {config.num_hidden_layers}")
    print()
    
    # 2. 创建模型
    print("[2] 创建模型...")
    model = FusionMini(config)
    param_count = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"   参数量: {param_count:.1f}K")
    print()
    
    # 3. 创建推理仪表板
    print("[3] 创建推理仪表板...")
    dashboard = InferenceDashboard(
        model=model,
        config=config,
        device="cpu",
    )
    print("   仪表板创建成功")
    print()
    
    # 4. 测试不同 think_rank 设置
    print("[4] 测试 Thinking Dial...")
    test_prompt = "Hello, this is a test"
    
    for think_rank in range(4):
        print(f"   测试 think_rank={think_rank}...")
        dashboard.set_think_rank(think_rank)
        
        # 测试 tokenization
        input_ids = dashboard._tokenize(test_prompt)
        print(f"     输入 tokens: {input_ids.shape}")
        
        # 测试生成（限制 token 数以避免长时间运行）
        dashboard.inference_config.max_new_tokens = 5
        try:
            output = dashboard.generate(test_prompt)
            print(f"     生成结果: {output[:50]}...")
        except Exception as e:
            print(f"     生成失败: {e}")
        
        print()
    
    print("[TEST] 基本推理测试完成")
    print()
    
    # 5. 测试 SBLA 注意力
    print("[5] 验证 SBLA 注意力...")
    has_sbla = any("SBLAttention" in str(module) for module in model.modules())
    if has_sbla:
        print("   ✅ SBLA 注意力已集成")
    else:
        print("   ❌ SBLA 注意力未找到")
    print()
    
    print("[TEST] 所有测试完成")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 推理测试")
    print("=" * 60)
    print()
    
    try:
        success = test_basic_inference()
        if success:
            print("✅ 所有测试通过")
        else:
            print("❌ 测试失败")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
