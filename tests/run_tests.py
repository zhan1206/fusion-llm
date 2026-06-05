"""
Fusion 项目测试脚本

运行所有单元测试，确保代码质量

使用方法：
    python -m pytest tests/ -v
    或
    python tests/run_tests.py

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import sys
import os
import unittest
import torch
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("[LOGO] Fusion 项目单元测试")
print("=" * 50)


class TestSBLAAttention(unittest.TestCase):
    """测试 SBLA 注意力机制"""
    
    def test_forward_pass(self):
        """测试前向传播"""
        from models.sbla_attention import SBLAttention
        
        batch_size = 2
        seq_len = 1024
        hidden_size = 512
        num_heads = 8
        
        attn = SBLAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            block_size=512,
            latent_dim=64,
            window_size=1024,
            mode="pure_sbla",
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        output, _ = attn(hidden_states=x, attention_mask=attention_mask)
        
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        print("[OK] SBLA 前向传播测试通过")
    
    def test_long_sequence(self):
        """测试长序列处理"""
        from models.sbla_attention import SBLAttention
        
        attn = SBLAttention(
            hidden_size=256,
            num_heads=4,
            block_size=256,
            latent_dim=32,
            window_size=512,
            mode="pure_sbla",
        )
        
        # 测试 8K 序列
        x = torch.randn(1, 8192, 256)
        attention_mask = torch.ones(1, 8192)
        output, _ = attn(hidden_states=x, attention_mask=attention_mask)
        
        self.assertEqual(output.shape, (1, 8192, 256))
        print("[OK] SBLA 长序列测试通过")


class TestThinkingDial(unittest.TestCase):
    """测试 Thinking Dial 机制"""
    
    def test_parse_depth(self):
        """测试解析推理深度"""
        from models.thinking_dial import parse_think_token
        
        # 测试解析
        depth, clean = parse_think_token(
            "<|think_depth_2|> 证明勾股定理"
        )
        
        self.assertEqual(depth, 2)
        self.assertEqual(clean, "证明勾股定理")
        print("[OK] Thinking Dial 解析测试通过")
    
    def test_inject_token(self):
        """测试注入控制 token"""
        from models.thinking_dial import apply_thinking_control
        
        result = apply_thinking_control(
            "解释量子纠缠",
            depth=1,
        )
        
        self.assertIn("<|think_depth_1|>", result)
        print("[OK] Thinking Dial 注入测试通过")


class TestBilingualFilter(unittest.TestCase):
    """测试双母语数据清洗"""
    
    def test_chinese_filter(self):
        """测试中文质量过滤"""
        from data_pipeline.bilingual_filter import BilingualTrueFilter
        
        filter = BilingualTrueFilter(lang="zh")
        
        # 小编体应该被过滤
        self.assertFalse(filter._filter_chinese_quality(
            "震惊！这个秘密竟然被曝光！！！"
        ))
        
        # 正常文本应该通过
        self.assertTrue(filter._filter_chinese_quality(
            "量子纠缠是量子力学中的一种现象，指两个或多个粒子之间存在一种特殊的关联。"
        ))
        
        print("[OK] 中文过滤器测试通过")
    
    def test_english_filter(self):
        """测试英文质量过滤"""
        from data_pipeline.bilingual_filter import BilingualTrueFilter
        
        filter = BilingualTrueFilter(lang="en")
        
        # 正常英文应该通过
        self.assertTrue(filter._filter_english_quality(
            "Quantum entanglement is a phenomenon in quantum mechanics."
        ))
        
        print("[OK] 英文过滤器测试通过")


class TestFusionModel(unittest.TestCase):
    """测试完整 Fusion 模型"""
    
    def test_model_creation(self):
        """测试模型创建"""
        from models.fusion_model import FusionModel, FusionConfig
        
        config = FusionConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        
        model = FusionModel(config)
        
        self.assertIsNotNone(model)
        print("[OK] Fusion 模型创建测试通过")
    
    def test_forward_pass(self):
        """测试前向传播"""
        from models.fusion_model import FusionModel, FusionConfig
        
        config = FusionConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        
        model = FusionModel(config)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (2, 64))
        
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                labels=input_ids,
                return_dict=True,
            )
        
        self.assertIn("loss", outputs)
        self.assertIn("logits", outputs)
        print("[OK] Fusion 模型前向传播测试通过")


class TestDataPipeline(unittest.TestCase):
    """测试数据处理管道"""
    
    def test_example_data(self):
        """测试示例数据格式"""
        data_path = project_root / "data" / "example_data.json"
        
        if not data_path.exists():
            self.skipTest("示例数据文件不存在")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # 检查数据格式
        for item in data[:3]:
            self.assertIn("prompt", item)
            self.assertIn("response", item)
            self.assertIn("think_rank", item)
            self.assertIn(item["think_rank"], [0, 1, 2, 3])
        
        print("[OK] 示例数据格式测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("开始运行测试...")
    print("=" * 50 + "\n")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestSBLAAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestThinkingDial))
    suite.addTests(loader.loadTestsFromTestCase(TestBilingualFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestFusionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"运行测试：{result.testsRun}")
    print(f"失败：{len(result.failures)}")
    print(f"错误：{len(result.errors)}")
    print(f"跳过：{len(result.skipped)}")
    
    if result.failures:
        print("\n失败详情：")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\n错误详情：")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success = result.wasSuccessful()
    
    if success:
        print("\n[DONE] 所有测试通过！")
    else:
        print("\n[FAIL] 部分测试失败，请检查代码")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
